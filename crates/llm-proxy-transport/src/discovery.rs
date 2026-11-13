//! Upstream provider model discovery.

use std::time::Duration;

use futures::TryStreamExt;
use llm_proxy_core::{
    AuthStyle, ProviderConfig, ProviderDiscoveryConfig, ProviderDiscoveryKind, ProviderRouteKind,
    StaticModelCatalogEntry,
};
use reqwest::{Client, RequestBuilder, Url};
use secrecy::ExposeSecret;
use serde_json::Value;

use llm_proxy_adapters::ProviderError;

const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

const MAX_DISCOVERY_ERROR_BODY_BYTES: usize = 2048;

/// GET-capable HTTP client for provider model discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryClient {
    http: Client,
}

impl DiscoveryClient {
    /// Build a discovery client with short, independent timeouts; errors if the
    /// HTTP client cannot be constructed.
    pub fn try_new() -> Result<Self, ProviderError> {
        let http = build_client(None)?;
        Ok(Self { http })
    }

    /// Fetch and normalize the configured provider model list; errors on missing
    /// discovery config, invalid responses, stalled or over-limit pagination,
    /// or transport failures.
    pub async fn discover(
        &self,
        provider: &ProviderConfig,
    ) -> Result<Vec<StaticModelCatalogEntry>, ProviderError> {
        let discovery = provider.discovery.as_ref().ok_or_else(|| {
            ProviderError::InvalidConfig(format!(
                "provider \"{}\" has no discovery configuration",
                provider.name
            ))
        })?;

        // Model discovery carries the provider's API key, so it is provider
        // upstream traffic like chat: an `[provider.egress]` proxy must carry
        // it too (discovery is infrequent, so the client is built per call).
        let http = match provider.egress.as_ref() {
            Some(egress) => build_client(Some(&egress.proxy))?,
            None => self.http.clone(),
        };

        let mut url = Url::parse(&discovery.endpoint).map_err(|error| {
            ProviderError::InvalidConfig(format!("invalid discovery endpoint: {error}"))
        })?;
        let mut models = Vec::new();
        let mut dropped = 0usize;
        let mut previous_token: Option<String> = None;

        for _ in 0..discovery.max_pages {
            let value = self.fetch_page(&http, provider, &url).await?;
            dropped += parse_models(discovery.kind, &value, &mut models)?;
            if models.len() > discovery.max_models {
                return Err(ProviderError::InvalidConfig(format!(
                    "provider \"{}\" discovery exceeded {} models",
                    provider.name, discovery.max_models
                )));
            }

            let Some((name, token)) = next_page(discovery.kind, &value)? else {
                models.sort_by(|a, b| a.id.cmp(&b.id));
                models.dedup_by(|a, b| a.id == b.id);
                if dropped > 0 {
                    tracing::info!(
                        provider = %provider.name,
                        kept = models.len(),
                        dropped,
                        "discovery completed with malformed or filtered model records",
                    );
                }
                return Ok(models);
            };

            if previous_token.as_deref() == Some(token.as_str()) {
                return Err(ProviderError::InvalidConfig(format!(
                    "provider \"{}\" discovery pagination token did not advance after {} model(s)",
                    provider.name,
                    models.len()
                )));
            }
            previous_token = Some(token.clone());

            set_query_parameter(&mut url, name, &token);
        }

        Err(ProviderError::InvalidConfig(format!(
            "provider \"{}\" discovery exceeded {} pages",
            provider.name, discovery.max_pages
        )))
    }

    /// GET `url` (no credentials attached) and parse the response as JSON,
    /// bounded by `max_bytes`. Operator tooling uses this — the models.dev
    /// pricing sync (plan 05 step 9).
    ///
    /// Unlike [`DiscoveryClient::discover`], the request is not stamped with
    /// the self-loop hop header: a public catalogue URL is not a provider
    /// upstream.
    pub async fn fetch_json(&self, url: &str, max_bytes: usize) -> Result<Value, ProviderError> {
        let url = Url::parse(url).map_err(|error| {
            ProviderError::InvalidConfig(format!("invalid catalogue URL: {error}"))
        })?;
        self.send_once(self.http.get(url), max_bytes).await
    }

    #[expect(
        clippy::expect_used,
        reason = "the retry loop below runs at least once and every `continue` stores the \
                  error, so `last_error` is always Some at the tail expression"
    )]
    async fn fetch_page(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        url: &Url,
    ) -> Result<Value, ProviderError> {
        let discovery = provider.discovery.as_ref().ok_or_else(|| {
            ProviderError::InvalidConfig("missing discovery configuration".to_owned())
        })?;

        let mut last_error = None;
        for attempt in 0..=Self::MAX_TRANSIENT_RETRIES {
            if attempt > 0 {
                tokio::time::sleep(Self::RETRY_BACKOFF).await;
            }
            let request = self.build_request(http, provider, discovery, url);
            match self.send_once(request, discovery.max_response_bytes).await {
                Ok(value) => return Ok(value),
                Err(error) if Self::is_transient(&error) => {
                    tracing::warn!(
                        provider = %provider.name,
                        attempt,
                        max_attempts = Self::MAX_TRANSIENT_RETRIES,
                        error = %error,
                        "transient discovery failure; retrying",
                    );
                    last_error = Some(error);
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.expect("retry loop ran at least once"))
    }

    /// Upper bound on retries after the first attempt for a transient failure;
    /// worst-case probe count is `1 + MAX_TRANSIENT_RETRIES`.
    pub const MAX_TRANSIENT_RETRIES: u32 = 2;

    const RETRY_BACKOFF: Duration = Duration::from_millis(500);

    fn is_transient(error: &ProviderError) -> bool {
        match error {
            ProviderError::Http { redirect: true, .. } => false,
            ProviderError::Http { .. } => true,
            ProviderError::Api { status, .. } => (500..600).contains(status),
            _ => false,
        }
    }

    fn build_request(
        &self,
        http: &Client,
        provider: &ProviderConfig,
        discovery: &ProviderDiscoveryConfig,
        url: &Url,
    ) -> RequestBuilder {
        let mut request = http.get(url.clone());
        // Self-loop marker (backlog 32): a discovery endpoint that points
        // back at an llm-proxy must be answered 508 instead of looping.
        request = request.header(llm_proxy_core::HOP_HEADER, llm_proxy_core::HOP_HEADER_VALUE);
        let api_key: &str = provider.api_key.expose_secret();
        request = match provider.auth_style {
            AuthStyle::Bearer => request.bearer_auth(api_key),
            AuthStyle::XApiKey => request.header("x-api-key", api_key),
            AuthStyle::XGoogleApiKey => request.header("x-goog-api-key", api_key),
            AuthStyle::Both => request.bearer_auth(api_key).header("x-api-key", api_key),
        };
        if discovery.kind == ProviderDiscoveryKind::AnthropicModels
            && !discovery
                .headers
                .keys()
                .any(|name| name.eq_ignore_ascii_case(ANTHROPIC_VERSION_HEADER))
        {
            request = request.header(ANTHROPIC_VERSION_HEADER, DEFAULT_ANTHROPIC_VERSION);
        }
        for (name, value) in &discovery.headers {
            request = request.header(name, value);
        }
        request
    }

    async fn send_once(
        &self,
        request: RequestBuilder,
        max_response_bytes: usize,
    ) -> Result<Value, ProviderError> {
        let response = request
            .send()
            .await
            .map_err(crate::client::provider_error_from_reqwest)?;
        let status = response.status();
        if !status.is_success() {
            let body_prefix =
                crate::client::read_error_body_bounded(response, MAX_DISCOVERY_ERROR_BODY_BYTES)
                    .await;
            return Err(ProviderError::api(status.as_u16(), body_prefix));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream
            .try_next()
            .await
            .map_err(crate::client::provider_error_from_reqwest)?
        {
            if bytes.len() + chunk.len() > max_response_bytes {
                return Err(ProviderError::InvalidConfig(format!(
                    "discovery response exceeded {} bytes",
                    max_response_bytes
                )));
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(ProviderError::from)
    }
}

fn parse_models(
    kind: ProviderDiscoveryKind,
    value: &Value,
    output: &mut Vec<StaticModelCatalogEntry>,
) -> Result<usize, ProviderError> {
    let records = match kind {
        ProviderDiscoveryKind::GeminiModels => value.get("models"),
        ProviderDiscoveryKind::FireworksAccountModels => {
            value.get("models").or_else(|| value.get("data"))
        }
        _ => value.get("data"),
    }
    .and_then(Value::as_array);

    let Some(records) = records else {
        return Err(ProviderError::InvalidConfig(
            "discovery response has an invalid top-level model list".to_owned(),
        ));
    };
    let mut dropped = 0usize;
    for record in records {
        let id = record
            .get("id")
            .or_else(|| record.get("name"))
            .and_then(Value::as_str);
        let Some(id) = id.filter(|id| !id.trim().is_empty()) else {
            dropped += 1;
            tracing::warn!("ignoring malformed discovery model record without an id");
            continue;
        };
        if kind == ProviderDiscoveryKind::GeminiModels {
            let supports_generate = record
                .get("supportedGenerationMethods")
                .or_else(|| record.get("supportedActions"))
                .and_then(Value::as_array)
                .is_some_and(|actions| actions.iter().any(|action| action == "generateContent"));
            if !supports_generate {
                dropped += 1;
                tracing::debug!(
                    model = id,
                    "ignoring Gemini model without generateContent support"
                );
                continue;
            }
        }
        output.push(StaticModelCatalogEntry {
            id: id.to_owned(),
            display_name: record
                .get("display_name")
                .or_else(|| record.get("displayName"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            supports: vec![
                ProviderRouteKind::ChatCompletions,
                ProviderRouteKind::Messages,
            ],
            context_length: record
                .get("context_length")
                .or_else(|| record.get("inputTokenLimit"))
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok()),
        });
    }
    Ok(dropped)
}

fn set_query_parameter(url: &mut Url, name: &str, value: &str) {
    let snapshot = url.query().unwrap_or("").to_owned();
    url.set_query(None);
    let mut serializer = url.query_pairs_mut();
    for (key, val) in form_urlencoded::parse(snapshot.as_bytes()) {
        if key != name {
            serializer.append_pair(&key, &val);
        }
    }
    serializer.append_pair(name, value);
}

fn next_page(
    kind: ProviderDiscoveryKind,
    value: &Value,
) -> Result<Option<(&'static str, String)>, ProviderError> {
    match kind {
        ProviderDiscoveryKind::GeminiModels => Ok(value
            .get("nextPageToken")
            .and_then(Value::as_str)
            .filter(|token| !token.is_empty())
            .map(|token| ("pageToken", token.to_owned()))),
        ProviderDiscoveryKind::AnthropicModels => {
            let has_more = value
                .get("has_more")
                .and_then(Value::as_bool)
                .is_some_and(|b| b);
            if !has_more {
                return Ok(None);
            }
            let token = value
                .get("last_id")
                .and_then(Value::as_str)
                .filter(|token| !token.is_empty());
            match token {
                Some(token) => Ok(Some(("after_id", token.to_owned()))),
                None => Err(ProviderError::InvalidConfig(
                    "Anthropic discovery response has `has_more` without `last_id`".to_owned(),
                )),
            }
        }
        ProviderDiscoveryKind::FireworksAccountModels => Ok(value
            .get("nextPageToken")
            .or_else(|| value.get("next_page_token"))
            .and_then(Value::as_str)
            .filter(|token| !token.is_empty())
            .map(|token| ("pageToken", token.to_owned()))),
        _ => Ok(None),
    }
}

/// Build the discovery HTTP client: direct when `proxy` is `None`, otherwise
/// routed through the configured `[provider.egress]` forward proxy, with the
/// same timeouts and same-origin redirect policy either way.
fn build_client(proxy: Option<&str>) -> Result<Client, ProviderError> {
    let mut builder = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30))
        .redirect(crate::client::same_origin_redirect_policy());
    if let Some(proxy_url) = proxy {
        crate::client::validate_egress_proxy_url(proxy_url)?;
        let proxy =
            reqwest::Proxy::all(proxy_url).map_err(crate::client::provider_error_from_reqwest)?;
        builder = builder.proxy(proxy);
    }
    builder
        .build()
        .map_err(crate::client::provider_error_from_reqwest)
}

impl Default for DiscoveryClient {
    /// Returns a default discovery client; panics if the HTTP client cannot be
    /// constructed — use [`DiscoveryClient::try_new`] for the fallible form.
    #[expect(
        clippy::expect_used,
        reason = "Default is infallible in practice; try_new is the fallible constructor"
    )]
    fn default() -> Self {
        Self::try_new().expect("default discovery HTTP client configuration is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Json;
    use axum::Router;
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::get;
    use llm_proxy_core::ProviderRoutesConfig;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn fetch_json_gets_the_catalogue_without_credentials() {
        let app = Router::new().route(
            "/api.json",
            get(|| async { Json(json!({"provider_x": {"models": {}}})) }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::trace!(%error, "test upstream stopped");
            }
        });
        let client = DiscoveryClient::try_new().expect("client");
        let value = client
            .fetch_json(&format!("http://{addr}/api.json"), 1024 * 1024)
            .await
            .expect("fetch");
        assert!(value.get("provider_x").is_some(), "got {value}");
        let error = client
            .fetch_json("not a url", 1024)
            .await
            .expect_err("invalid URL refused");
        assert!(error.to_string().contains("invalid"), "got {error}");
    }

    async fn echo_anthropic_version(headers: HeaderMap) -> Json<Value> {
        let version = headers
            .get(ANTHROPIC_VERSION_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("missing");
        Json(json!({
            "data": [{"id": version}],
            "has_more": false
        }))
    }

    async fn echo_hop(headers: HeaderMap) -> Json<Value> {
        let hop = headers
            .get(llm_proxy_core::HOP_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("missing");
        Json(json!({
            "data": [{"id": hop}],
            "has_more": false
        }))
    }

    async fn start_test_server(app: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{address}/models")
    }

    fn discovery_provider(
        endpoint: String,
        kind: ProviderDiscoveryKind,
        headers: HashMap<String, String>,
    ) -> ProviderConfig {
        ProviderConfig {
            name: "test-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::XApiKey,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: Some(ProviderDiscoveryConfig {
                kind,
                endpoint,
                headers,
                max_pages: 100,
                max_models: 20_000,
                max_response_bytes: 4 * 1024 * 1024,
            }),
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        }
    }

    #[tokio::test]
    async fn anthropic_discovery_adds_default_version_header() {
        let endpoint =
            start_test_server(Router::new().route("/models", get(echo_anthropic_version))).await;
        let provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::AnthropicModels,
            HashMap::new(),
        );

        let models = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect("Anthropic discovery should succeed");

        assert_eq!(models[0].id, DEFAULT_ANTHROPIC_VERSION);
    }

    #[tokio::test]
    async fn anthropic_discovery_preserves_configured_version_header() {
        let headers = HashMap::from([("Anthropic-Version".to_owned(), "2024-01-01".to_owned())]);
        let endpoint =
            start_test_server(Router::new().route("/models", get(echo_anthropic_version))).await;
        let provider =
            discovery_provider(endpoint, ProviderDiscoveryKind::AnthropicModels, headers);

        let models = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect("Anthropic discovery should succeed");

        assert_eq!(models[0].id, "2024-01-01");
    }

    #[tokio::test]
    async fn discovery_requests_carry_the_hop_marker() {
        let endpoint = start_test_server(Router::new().route("/models", get(echo_hop))).await;
        let provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::OpenAiCompatibleModels,
            HashMap::new(),
        );

        let models = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect("discovery should succeed");

        assert_eq!(models[0].id, "1");
    }

    #[tokio::test]
    async fn discovery_stops_at_configured_page_limit() {
        async fn paginated_models(State(calls): State<Arc<AtomicUsize>>) -> Json<Value> {
            calls.fetch_add(1, Ordering::SeqCst);
            Json(json!({
                "models": [{
                    "name": "models/gemini-pro",
                    "supportedGenerationMethods": ["generateContent"]
                }],
                "nextPageToken": "next"
            }))
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let endpoint = start_test_server(
            Router::new()
                .route("/models", get(paginated_models))
                .with_state(Arc::clone(&calls)),
        )
        .await;
        let mut provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::GeminiModels,
            HashMap::new(),
        );
        provider.discovery.as_mut().unwrap().max_pages = 1;

        let error = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect_err("page limit should stop pagination");

        assert!(error.to_string().contains("exceeded 1 pages"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn discovery_stops_at_configured_model_limit() {
        let endpoint = start_test_server(Router::new().route(
            "/models",
            get(|| async { Json(json!({"data": [{"id": "a"}, {"id": "b"}]})) }),
        ))
        .await;
        let mut provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::OpenAiCompatibleModels,
            HashMap::new(),
        );
        provider.discovery.as_mut().unwrap().max_models = 1;

        let error = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect_err("model limit should stop discovery");

        assert!(error.to_string().contains("exceeded 1 models"));
    }

    #[tokio::test]
    async fn discovery_stops_at_configured_response_size_limit() {
        let endpoint = start_test_server(Router::new().route(
            "/models",
            get(|| async { Json(json!({"data": [{"id": "model"}]})) }),
        ))
        .await;
        let mut provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::OpenAiCompatibleModels,
            HashMap::new(),
        );
        provider.discovery.as_mut().unwrap().max_response_bytes = 8;

        let error = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect_err("response size limit should stop discovery");

        assert!(error.to_string().contains("exceeded 8 bytes"));
    }

    #[tokio::test]
    async fn discovery_fails_when_pagination_token_does_not_advance() {
        async fn static_token_models(State(calls): State<Arc<AtomicUsize>>) -> Json<Value> {
            calls.fetch_add(1, Ordering::SeqCst);
            Json(json!({
                "models": [{
                    "name": "models/gemini-pro",
                    "supportedGenerationMethods": ["generateContent"]
                }],
                "nextPageToken": "stale"
            }))
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let endpoint = start_test_server(
            Router::new()
                .route("/models", get(static_token_models))
                .with_state(Arc::clone(&calls)),
        )
        .await;
        let provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::GeminiModels,
            HashMap::new(),
        );

        let error = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect_err("a stalled pagination token must fail discovery");

        assert!(
            matches!(error, ProviderError::InvalidConfig(ref message) if message.contains("did not advance")),
            "expected InvalidConfig mentioning 'did not advance', got: {error}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "pagination should stop after detecting the repeated token"
        );
    }

    #[tokio::test]
    async fn discovery_fails_when_anthropic_has_more_without_last_id() {
        let endpoint = start_test_server(Router::new().route(
            "/models",
            get(|| async { Json(json!({"data": [{"id": "claude-x"}], "has_more": true})) }),
        ))
        .await;
        let provider = discovery_provider(
            endpoint,
            ProviderDiscoveryKind::AnthropicModels,
            HashMap::new(),
        );

        let error = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect_err("`has_more` without `last_id` must fail discovery");

        assert!(
            matches!(error, ProviderError::InvalidConfig(ref message) if message.contains("has_more")),
            "expected InvalidConfig mentioning 'has_more', got: {error}"
        );
    }

    #[test]
    fn openai_records_are_sorted_and_malformed_records_are_skipped() {
        let mut models = Vec::new();
        let dropped = parse_models(
            ProviderDiscoveryKind::OpenAiCompatibleModels,
            &json!({"data": [{"id": "z"}, {"missing": "id"}, {"id": "a"}]}),
            &mut models,
        )
        .expect("valid response");
        models.sort_by(|left, right| left.id.cmp(&right.id));
        assert_eq!(
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "z"]
        );
        assert_eq!(dropped, 1);
    }

    #[test]
    fn invalid_top_level_shape_fails() {
        let error = parse_models(
            ProviderDiscoveryKind::OpenAiModels,
            &json!({"models": []}),
            &mut Vec::new(),
        )
        .expect_err("invalid shape");
        assert!(error.to_string().contains("top-level"));
    }

    #[test]
    fn gemini_filters_non_generation_models() {
        let mut models = Vec::new();
        parse_models(
            ProviderDiscoveryKind::GeminiModels,
            &json!({"models": [
                {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/generate", "supportedGenerationMethods": ["generateContent"]}
            ]}),
            &mut models,
        )
        .expect("valid response");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "models/generate");
    }

    #[test]
    fn pagination_parameter_is_replaced() {
        let mut url = Url::parse("https://example.com/models?pageToken=old&limit=10").unwrap();
        set_query_parameter(&mut url, "pageToken", "new");
        assert_eq!(
            url.as_str(),
            "https://example.com/models?limit=10&pageToken=new"
        );
    }

    fn discovery_provider_with_auth(
        endpoint: String,
        auth_style: AuthStyle,
        api_key: &str,
    ) -> ProviderConfig {
        ProviderConfig {
            name: "test-provider".to_owned(),
            api_key: secrecy::SecretString::from(api_key),
            auth_style,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: Some(ProviderDiscoveryConfig {
                kind: ProviderDiscoveryKind::OpenAiCompatibleModels,
                endpoint,
                headers: HashMap::new(),
                max_pages: 100,
                max_models: 20_000,
                max_response_bytes: 4 * 1024 * 1024,
            }),
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        }
    }

    #[derive(Clone)]
    struct EchoAuthConfig {
        header_name: String,
        strip_prefix: Option<String>,
    }

    async fn echo_auth_header(
        State(config): State<Arc<EchoAuthConfig>>,
        headers: HeaderMap,
    ) -> Json<Value> {
        let value = headers
            .get(config.header_name.as_str())
            .and_then(|v| v.to_str().ok())
            .unwrap_or("missing");
        let value = match &config.strip_prefix {
            Some(prefix) => value.strip_prefix(prefix).unwrap_or(value),
            None => value,
        };
        Json(json!({ "data": [{ "id": value }] }))
    }

    async fn run_discovery_auth_on_the_wire_and_redacted(
        auth_style: AuthStyle,
        header_name: &'static str,
        strip_prefix: Option<&'static str>,
    ) {
        const SECRET: &str = "sk-discovery-secret-xyz";

        let config = Arc::new(EchoAuthConfig {
            header_name: header_name.to_owned(),
            strip_prefix: strip_prefix.map(str::to_owned),
        });
        let endpoint = start_test_server(
            Router::new()
                .route("/models", get(echo_auth_header))
                .with_state(config),
        )
        .await;
        let provider = discovery_provider_with_auth(endpoint, auth_style, SECRET);

        let models = DiscoveryClient::default()
            .discover(&provider)
            .await
            .expect("discovery should succeed");
        assert_eq!(
            models.len(),
            1,
            "echo handler should return exactly one model"
        );
        assert_eq!(
            models[0].id, SECRET,
            "the configured api key must be sent on the wire as the auth header"
        );

        let debug = format!("{provider:?}");
        assert!(
            !debug.contains(SECRET),
            "ProviderConfig Debug must redact the api key; leaked in: {debug}"
        );
        assert!(
            debug.contains("[REDACTED]"),
            "ProviderConfig Debug should mark the api key as redacted; got: {debug}"
        );
    }

    #[tokio::test]
    async fn discovery_bearer_auth_uses_secret_on_wire_and_redacts_debug() {
        run_discovery_auth_on_the_wire_and_redacted(
            AuthStyle::Bearer,
            "authorization",
            Some("Bearer "),
        )
        .await;
    }

    #[tokio::test]
    async fn discovery_x_api_key_auth_uses_secret_on_wire_and_redacts_debug() {
        run_discovery_auth_on_the_wire_and_redacted(AuthStyle::XApiKey, "x-api-key", None).await;
    }

    #[tokio::test]
    async fn discovery_x_goog_api_key_auth_uses_secret_on_wire_and_redacts_debug() {
        run_discovery_auth_on_the_wire_and_redacted(
            AuthStyle::XGoogleApiKey,
            "x-goog-api-key",
            None,
        )
        .await;
    }

    #[tokio::test]
    async fn discovery_both_auth_uses_secret_on_wire_and_redacts_debug() {
        run_discovery_auth_on_the_wire_and_redacted(
            AuthStyle::Both,
            "authorization",
            Some("Bearer "),
        )
        .await;
    }

    async fn record_absolute_uri_target(
        State(seen): State<Arc<std::sync::Mutex<Option<String>>>>,
        uri: axum::http::Uri,
    ) -> Json<Value> {
        *seen.lock().expect("seen-target lock") = Some(uri.to_string());
        Json(json!({"data": [{"id": "m"}], "has_more": false}))
    }

    #[tokio::test]
    async fn discovery_goes_through_the_provider_egress_proxy() {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let app = Router::new()
            .fallback(record_absolute_uri_target)
            .with_state(Arc::clone(&seen));
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("addr");
        let proxy_base = format!("http://{address}");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::trace!(%error, "test egress proxy stopped");
            }
        });

        let mut provider = discovery_provider(
            "http://upstream.invalid/models".to_owned(),
            ProviderDiscoveryKind::OpenAiCompatibleModels,
            HashMap::new(),
        );
        provider.egress = Some(llm_proxy_core::ProviderEgressConfig { proxy: proxy_base });

        let client = DiscoveryClient::try_new().expect("client");
        let models = client
            .discover(&provider)
            .await
            .expect("discovery must reach upstream.invalid through the provider egress proxy");

        assert_eq!(models[0].id, "m");
        let seen_target = seen.lock().expect("seen-target lock").clone();
        assert!(
            seen_target
                .as_deref()
                .is_some_and(|target| target.contains("upstream.invalid")),
            "the proxy stub must see the absolute upstream.invalid target, got {seen_target:?}"
        );
    }
}
