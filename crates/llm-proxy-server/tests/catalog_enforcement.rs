//! HTTP-level tests for catalog enforcement (fix-brief R-5): enforcement on
//! rejects unlisted models before any upstream call, enforcement off passes
//! unknown IDs through, the model listing and enforcement agree on the same
//! allow/deny set and aliases, and the rejection follows the route's client
//! protocol shape.

#![cfg(test)]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::post;
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderAdapterConfig, ProviderCatalogConfig, ProviderCatalogMode,
    ProviderConfig, ProviderRegistry, ProviderRoutesConfig, ServerConfig, StaticModelCatalogEntry,
};
use llm_proxy_server::{AppState, BuildInfo, build_router};
use llm_proxy_transport::ProxyClient;
use tower::ServiceExt;

const OPENAI_CHAT_OK: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":12345,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#;

/// A fake upstream that counts requests and captures the last body it was
/// sent, then always answers `OPENAI_CHAT_OK` as JSON with 200.
struct CountingUpstream {
    endpoint: String,
    hits: Arc<AtomicUsize>,
    captured: Arc<tokio::sync::Mutex<Option<String>>>,
}

impl CountingUpstream {
    fn hit_count(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    /// The `model` field of the last JSON body the upstream received.
    async fn captured_model(&self) -> Option<String> {
        let guard = self.captured.lock().await;
        guard.as_ref().and_then(|raw| {
            serde_json::from_str::<serde_json::Value>(raw)
                .ok()
                .and_then(|json| json.get("model")?.as_str().map(str::to_owned))
        })
    }
}

async fn spawn_counting_upstream() -> CountingUpstream {
    let hits = Arc::new(AtomicUsize::new(0));
    let captured = Arc::new(tokio::sync::Mutex::new(None));
    let router = {
        let hits = Arc::clone(&hits);
        let captured = Arc::clone(&captured);
        Router::new().route(
            "/{*path}",
            post(move |payload: axum::body::Bytes| {
                let hits = Arc::clone(&hits);
                let captured = Arc::clone(&captured);
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    let mut guard = captured.lock().await;
                    *guard = Some(String::from_utf8_lossy(&payload).into_owned());
                    drop(guard);
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        OPENAI_CHAT_OK,
                    )
                        .into_response()
                }
            }),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("fake upstream addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, router).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    CountingUpstream {
        endpoint: format!("http://{addr}/v1/chat/completions"),
        hits,
        captured,
    }
}

fn static_model(id: &str) -> StaticModelCatalogEntry {
    StaticModelCatalogEntry {
        id: id.to_owned(),
        display_name: None,
        supports: Vec::new(),
        context_length: None,
    }
}

fn catalog_config(models: &[&str], enforce: bool) -> ProviderCatalogConfig {
    ProviderCatalogConfig {
        mode: ProviderCatalogMode::Static,
        enforce,
        cache_ttl: Duration::from_secs(60),
        allow: vec!["*".to_owned()],
        deny: Vec::new(),
        models: models.iter().map(|id| static_model(id)).collect(),
    }
}

fn provider_with_catalog(
    name: &str,
    upstream: &CountingUpstream,
    catalog: Option<ProviderCatalogConfig>,
    aliases: &[(&str, &str)],
    messages_adapter: bool,
) -> ProviderConfig {
    let mut adapters = HashMap::from([(
        "chat".to_owned(),
        ProviderAdapterConfig {
            protocol: "openai_chat_completions".to_owned(),
            endpoint: upstream.endpoint.clone(),
            headers: Arc::new(HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    )]);
    if messages_adapter {
        adapters.insert(
            "messages".to_owned(),
            ProviderAdapterConfig {
                protocol: "anthropic_messages".to_owned(),
                endpoint: upstream.endpoint.clone(),
                headers: Arc::new(HashMap::new()),
                forward_inbound_headers: None,
                lossy: false,
            },
        );
    }
    ProviderConfig {
        name: name.to_owned(),
        api_key: secrecy::SecretString::from("test-key"),
        api_keys: None,
        auth_style: AuthStyle::Bearer,
        passthrough_auth: false,
        retry: None,
        egress: None,
        adapters,
        routes: ProviderRoutesConfig {
            chat_completions: Some("chat".to_owned()),
            messages: messages_adapter.then(|| "messages".to_owned()),
            ..Default::default()
        },
        model_aliases: aliases
            .iter()
            .map(|(alias, target)| ((*alias).to_owned(), (*target).to_owned()))
            .collect(),
        discovery: None,
        catalog,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn state_with_provider(provider: ProviderConfig) -> AppState {
    let app_config = AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:3456".parse().expect("bind address"),
            request_timeout: Duration::from_secs(300),
            shutdown_timeout: Duration::from_secs(30),
            log_level: "info".to_owned(),
            log_format: Default::default(),
            hot_reload: false,
            rate_limit_rpm: 1000,
            trust_forwarded_headers: false,
            server_name: "catalog-test".to_owned(),
            allowed_origins: None,
            admin_bind: None,
            admin_on_main_listener: false,
            endpoint_families: Default::default(),
            upstream_timeouts: Default::default(),
            bare_v1: Default::default(),
            preflight: Default::default(),
            cache: Default::default(),
            idempotency: Default::default(),
            metrics: Default::default(),
            semantic_cache: Default::default(),
        },
        auth: Default::default(),
        storage: None,
        replica: None,
        mcp: None,
    };
    let registry = ProviderRegistry::from_providers(vec![provider]).expect("registry");
    AppState::new(
        app_config,
        registry,
        ProviderAdapterRegistry::builtin(),
        ProxyClient::try_new().expect("client"),
        BuildInfo {
            name: "test",
            version: "0.0.0",
            target: "test",
            git_sha: "test",
        },
    )
    .expect("app state")
}

fn chat_request(uri: &str, model: &str) -> Request<Body> {
    let body = format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#);
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("chat request")
}

fn messages_request(uri: &str, model: &str) -> Request<Body> {
    let body = format!(
        r#"{{"model":"{model}","max_tokens":16,"messages":[{{"role":"user","content":"hi"}}]}}"#
    );
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("messages request")
}

fn models_request(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("models request")
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("response body");
    serde_json::from_slice(&bytes).expect("response JSON")
}

#[tokio::test]
async fn enforce_on_rejects_unlisted_model_before_any_upstream_call() {
    let upstream = spawn_counting_upstream().await;
    let state = state_with_provider(provider_with_catalog(
        "p",
        &upstream,
        Some(catalog_config(&["gpt-4"], true)),
        &[],
        false,
    ));
    let app = build_router(state);

    let response = app
        .oneshot(chat_request("/providers/p/v1/chat/completions", "gpt-5"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = response_json(response).await;
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("gpt-5")),
        "the rejection message must name the rejected model: {json}"
    );
    assert_eq!(
        upstream.hit_count(),
        0,
        "enforcement must reject before any upstream call"
    );
}

#[tokio::test]
async fn enforce_on_allows_listed_model_and_sends_it_upstream() {
    let upstream = spawn_counting_upstream().await;
    let state = state_with_provider(provider_with_catalog(
        "p",
        &upstream,
        Some(catalog_config(&["gpt-4"], true)),
        &[],
        false,
    ));
    let app = build_router(state);

    let response = app
        .oneshot(chat_request("/providers/p/v1/chat/completions", "gpt-4"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(upstream.hit_count(), 1, "the request must reach upstream");
    assert_eq!(
        upstream.captured_model().await.as_deref(),
        Some("gpt-4"),
        "the upstream body must carry the requested model"
    );
}

#[tokio::test]
async fn enforce_off_passes_unlisted_models_through() {
    for catalog in [
        Some(catalog_config(&["gpt-4"], false)),
        // An absent [provider.catalog] section is advisory by default too.
        None,
    ] {
        let upstream = spawn_counting_upstream().await;
        let state = state_with_provider(provider_with_catalog("p", &upstream, catalog, &[], false));
        let app = build_router(state);

        let response = app
            .oneshot(chat_request("/providers/p/v1/chat/completions", "gpt-5"))
            .await
            .expect("response");
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "an unlisted model must pass through when enforcement is off"
        );
        assert_eq!(upstream.hit_count(), 1);
        assert_eq!(upstream.captured_model().await.as_deref(), Some("gpt-5"));
    }
}

#[tokio::test]
async fn listing_and_enforcement_agree_on_allow_deny() {
    let upstream = spawn_counting_upstream().await;
    let catalog = ProviderCatalogConfig {
        mode: ProviderCatalogMode::Static,
        enforce: true,
        cache_ttl: Duration::from_secs(60),
        allow: vec!["gpt-*".to_owned()],
        deny: vec!["*-preview".to_owned()],
        models: vec![
            static_model("gpt-4"),
            static_model("gpt-4-preview"),
            static_model("claude-x"),
        ],
    };
    let state = state_with_provider(provider_with_catalog(
        "p",
        &upstream,
        Some(catalog),
        &[],
        false,
    ));
    let app = build_router(state);

    let response = app
        .clone()
        .oneshot(models_request("/providers/p/v1/models"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let json = response_json(response).await;
    let listed: BTreeSet<&str> = json["data"]
        .as_array()
        .expect("data array")
        .iter()
        .filter_map(|entry| entry["id"].as_str())
        .collect();
    // `gpt-*` is a non-wildcard allow pattern, so it explicitly includes
    // `gpt-4-preview` and overrides `*-preview` (`model_allowed`: explicit
    // allow beats deny); only `claude-x` is filtered out.
    assert_eq!(listed, BTreeSet::from(["gpt-4", "gpt-4-preview"]));

    for id in ["gpt-4", "gpt-4-preview", "claude-x"] {
        let response = app
            .clone()
            .oneshot(chat_request("/providers/p/v1/chat/completions", id))
            .await
            .expect("response");
        let expected = if listed.contains(id) {
            StatusCode::OK
        } else {
            StatusCode::BAD_REQUEST
        };
        assert_eq!(
            response.status(),
            expected,
            "enforcement must accept exactly the ids the listing shows: {id}"
        );
    }
    assert_eq!(
        upstream.hit_count(),
        listed.len(),
        "exactly the listed models reach upstream"
    );
}

#[tokio::test]
async fn enforcement_checks_the_alias_target_not_the_alias() {
    let upstream = spawn_counting_upstream().await;
    let state = state_with_provider(provider_with_catalog(
        "p",
        &upstream,
        Some(catalog_config(&["gpt-4"], true)),
        &[("fast", "gpt-4")],
        false,
    ));
    let app = build_router(state);

    let response = app
        .clone()
        .oneshot(chat_request("/providers/p/v1/chat/completions", "fast"))
        .await
        .expect("response");
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "the alias must be accepted because its target is listed"
    );
    assert_eq!(upstream.hit_count(), 1);
    assert_eq!(
        upstream.captured_model().await.as_deref(),
        Some("gpt-4"),
        "the upstream must receive the alias target, not the alias"
    );

    let response = app
        .oneshot(chat_request("/providers/p/v1/chat/completions", "gpt-5"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        upstream.hit_count(),
        1,
        "the unlisted model must not add an upstream hit"
    );
}

#[tokio::test]
async fn enforcement_rejection_is_openai_shaped_on_messages_route_too() {
    let upstream = spawn_counting_upstream().await;
    let state = state_with_provider(provider_with_catalog(
        "p",
        &upstream,
        Some(catalog_config(&["gpt-4"], true)),
        &[],
        true,
    ));
    let app = build_router(state);

    let response = app
        .oneshot(messages_request("/providers/p/v1/messages", "gpt-5"))
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let json = response_json(response).await;
    assert_eq!(
        json["type"], "error",
        "the messages route must answer in the Anthropic error envelope: {json}"
    );
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("gpt-5")),
        "the Anthropic envelope must name the rejected model: {json}"
    );
    assert_eq!(
        upstream.hit_count(),
        0,
        "the messages route must reject before any upstream call"
    );
}
