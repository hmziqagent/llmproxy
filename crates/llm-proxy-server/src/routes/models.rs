use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, Response, StatusCode, header};
use axum::response::IntoResponse;
use llm_proxy_adapters::ProviderError;
use llm_proxy_protocol::core::ModelRef;
use serde::Serialize;
use tracing::warn;

use crate::state::AppState;

use super::core_pipeline;
use super::error_response::{AuthOwner, ClientProtocol, RouteError, route_error_response};
#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
    has_more: bool,
    first_id: Option<String>,
    last_id: Option<String>,
}
#[derive(Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
    #[serde(rename = "type")]
    model_type: &'static str,
    display_name: Option<String>,
    created_at: Option<String>,
    supports: Vec<&'static str>,
    context_length: Option<u32>,
}

#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct ModelsQuery {
    refresh: Option<String>,
}
pub(crate) async fn handle_models(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    req: axum::extract::Request,
) -> Response<Body> {
    match handle_models_inner(&state, &provider, req).await {
        Ok(response) => response,
        Err(error) => {
            warn!(error = %error, "models request failed");
            route_error_response(ClientProtocol::OpenAiChat, &error)
        }
    }
}
#[expect(
    clippy::too_many_lines,
    reason = "one handler with per-stage ledger emission; splitting would scatter the failure rows"
)]
async fn handle_models_inner(
    state: &AppState,
    provider_name: &str,
    req: axum::extract::Request,
) -> Result<Response<Body>, RouteError> {
    let start = std::time::Instant::now();
    let event_bus = Arc::clone(&state.event_bus);
    let request_id = "models";
    let query = Query::<ModelsQuery>::try_from_uri(req.uri())
        .map(|query| query.0)
        .map_err(|rejection| {
            warn!(error = %rejection, "malformed query string on /v1/models");
            RouteError::InvalidRequest("malformed query string".to_owned())
        })
        .inspect_err(|e| {
            core_pipeline::emit_response_failed(
                &event_bus,
                request_id,
                Some(provider_name),
                None::<&ModelRef>,
                e,
                start,
            )
        })?;
    let refresh_live = query.refresh.as_deref() == Some("live");

    core_pipeline::validate_provider_name(provider_name).inspect_err(|e| {
        core_pipeline::emit_response_failed(
            &event_bus,
            request_id,
            Some(provider_name),
            None::<&ModelRef>,
            e,
            start,
        )
    })?;
    // A pool name lists the union of its members' catalogs; a provider name
    // lists its own (plan 04 step 4: clients use the same URL shape either
    // way). One captured registry snapshot answers both lookups, so a reload
    // landing between them cannot split the view.
    let live_providers = state.providers();
    let member_names: Vec<String> = match live_providers.pool(provider_name) {
        Some(pool) => pool.members.clone(),
        None => {
            if live_providers.get(provider_name).is_none() {
                let err = RouteError::UnknownProvider(provider_name.to_owned());
                core_pipeline::emit_response_failed(
                    &event_bus,
                    request_id,
                    Some(provider_name),
                    None::<&ModelRef>,
                    &err,
                    start,
                );
                return Err(err);
            }
            vec![provider_name.to_owned()]
        }
    };

    let mut entries = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    for member_name in &member_names {
        let Some(provider) = live_providers.get(member_name) else {
            continue;
        };
        let member_entries = state
            .model_catalogs()
            .catalog(provider, refresh_live)
            .await
            .map_err(map_catalog_error)
            .inspect_err(|e| {
                core_pipeline::emit_response_failed(
                    &event_bus,
                    request_id,
                    Some(provider_name),
                    None::<&ModelRef>,
                    e,
                    start,
                )
            })?;
        for entry in member_entries {
            if seen_ids.insert(entry.id.clone()) {
                entries.push(entry);
            }
        }
    }
    let first_id = entries.first().map(|m| m.id.clone());
    let last_id = entries.last().map(|m| m.id.clone());

    let data: Vec<ModelCard> = entries
        .into_iter()
        .map(|entry| ModelCard {
            id: entry.id,
            object: "model",
            created: 0,
            owned_by: provider_name.to_owned(),
            model_type: "model",
            display_name: entry.display_name,
            created_at: None,
            supports: entry
                .supports
                .into_iter()
                .map(|k| match k {
                    llm_proxy_core::ProviderRouteKind::ChatCompletions => "chat_completions",
                    llm_proxy_core::ProviderRouteKind::Messages => "messages",
                    llm_proxy_core::ProviderRouteKind::Responses => "responses",
                    llm_proxy_core::ProviderRouteKind::GenerateContent => "generate_content",
                    llm_proxy_core::ProviderRouteKind::Embeddings => "embeddings",
                    llm_proxy_core::ProviderRouteKind::Rerank => "rerank",
                    llm_proxy_core::ProviderRouteKind::AudioTranscriptions => {
                        "audio_transcriptions"
                    }
                    llm_proxy_core::ProviderRouteKind::AudioSpeech => "audio_speech",
                    llm_proxy_core::ProviderRouteKind::ImagesGenerations => "images_generations",
                    llm_proxy_core::ProviderRouteKind::Files => "files",
                    llm_proxy_core::ProviderRouteKind::Batches => "batches",
                })
                .collect(),
            context_length: entry.context_length,
        })
        .collect();

    let response = ModelsResponse {
        object: "list",
        data,
        has_more: false,
        first_id,
        last_id,
    };

    let mut response = (axum::http::StatusCode::OK, axum::Json(response)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );

    Ok(response)
}

pub(super) fn map_catalog_error(error: ProviderError) -> RouteError {
    match error {
        ProviderError::Api { status, body, .. } => {
            let status_code = StatusCode::from_u16(status).unwrap_or_else(|_| {
                tracing::warn!(
                    status,
                    "invalid HTTP status from catalog provider; mapping to 502"
                );
                StatusCode::BAD_GATEWAY
            });
            RouteError::Upstream {
                status: status_code,
                body,
                auth_owner: AuthOwner::Operator,
                retry_after_secs: None,
            }
        }
        ProviderError::Http { timeout: true, .. } => {
            RouteError::UpstreamTimeout("upstream request timed out".to_owned())
        }
        ProviderError::Http {
            message,
            timeout: false,
            ..
        } => {
            let sanitized = llm_proxy_engine::sanitize_upstream_error_body(&message);
            RouteError::Upstream {
                status: StatusCode::BAD_GATEWAY,
                body: sanitized,
                auth_owner: AuthOwner::Operator,
                retry_after_secs: None,
            }
        }
        ProviderError::Serialize(_)
        | ProviderError::Utf8(_)
        | ProviderError::SseFraming(_)
        | ProviderError::EmptyResponse(_)
        | ProviderError::InvalidConfig(_) => RouteError::Internal(error.to_string()),
        _ => RouteError::Internal(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_proxy_core::{
        AuthStyle, ProviderCatalogConfig, ProviderConfig, ProviderRouteKind, ProviderRoutesConfig,
        StaticModelCatalogEntry,
    };
    use std::collections::HashMap;
    fn models_request(query: &str) -> axum::extract::Request {
        let uri = if query.is_empty() {
            "/v1/models".to_owned()
        } else {
            format!("/v1/models?{query}")
        };
        axum::http::Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .expect("request")
    }
    fn build_state_with_catalog() -> AppState {
        let provider = ProviderConfig {
            name: "test-provider".to_owned(),
            api_key: secrecy::SecretString::from("sk-test"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: Some(ProviderCatalogConfig {
                mode: llm_proxy_core::ProviderCatalogMode::Static,
                enforce: false,
                cache_ttl: std::time::Duration::from_secs(86400),
                allow: vec!["*".to_owned()],
                deny: vec![],
                models: vec![
                    StaticModelCatalogEntry {
                        id: "deepseek-v3.1".to_owned(),
                        display_name: Some("DeepSeek V3.1".to_owned()),
                        supports: vec![ProviderRouteKind::ChatCompletions],
                        context_length: Some(131072),
                    },
                    StaticModelCatalogEntry {
                        id: "claude-sonnet-4".to_owned(),
                        display_name: Some("Claude Sonnet 4".to_owned()),
                        supports: vec![
                            ProviderRouteKind::ChatCompletions,
                            ProviderRouteKind::Messages,
                        ],
                        context_length: Some(200000),
                    },
                ],
            }),
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };

        let registry =
            llm_proxy_core::ProviderRegistry::from_providers(vec![provider]).expect("registry");
        AppState::new(
            llm_proxy_core::AppConfig {
                server: llm_proxy_core::ServerConfig {
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: std::time::Duration::from_secs(300),
                    shutdown_timeout: std::time::Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test".to_owned(),
                    rate_limit_rpm: 100,
                    trust_forwarded_headers: false,
                    cache: Default::default(),
                    idempotency: Default::default(),
                    metrics: Default::default(),
                    log_format: Default::default(),
                    admin_bind: None,
                    admin_on_main_listener: false,
                    endpoint_families: Default::default(),
                    upstream_timeouts: Default::default(),
                    bare_v1: Default::default(),
                    preflight: Default::default(),
                    semantic_cache: Default::default(),
                },
                auth: Default::default(),
                storage: None,
                replica: None,
                mcp: None,
            },
            registry,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            llm_proxy_transport::ProxyClient::try_new().expect("client"),
            crate::state::BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }
    fn build_state_without_catalog() -> AppState {
        let provider = ProviderConfig {
            name: "no-catalog".to_owned(),
            api_key: secrecy::SecretString::from("sk-test"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };

        let registry =
            llm_proxy_core::ProviderRegistry::from_providers(vec![provider]).expect("registry");
        AppState::new(
            llm_proxy_core::AppConfig {
                server: llm_proxy_core::ServerConfig {
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: std::time::Duration::from_secs(300),
                    shutdown_timeout: std::time::Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test".to_owned(),
                    rate_limit_rpm: 100,
                    trust_forwarded_headers: false,
                    cache: Default::default(),
                    idempotency: Default::default(),
                    metrics: Default::default(),
                    log_format: Default::default(),
                    admin_bind: None,
                    admin_on_main_listener: false,
                    endpoint_families: Default::default(),
                    upstream_timeouts: Default::default(),
                    bare_v1: Default::default(),
                    preflight: Default::default(),
                    semantic_cache: Default::default(),
                },
                auth: Default::default(),
                storage: None,
                replica: None,
                mcp: None,
            },
            registry,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            llm_proxy_transport::ProxyClient::try_new().expect("client"),
            crate::state::BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    #[tokio::test]
    async fn models_with_catalog_returns_entries() {
        let state = build_state_with_catalog();
        let response = handle_models_inner(&state, "test-provider", models_request(""))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 16384)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
        assert_eq!(json["has_more"], false);
        let m1 = &json["data"][0];
        assert_eq!(m1["id"], "claude-sonnet-4");
        assert_eq!(m1["object"], "model");
        assert_eq!(m1["owned_by"], "test-provider");
        assert_eq!(m1["display_name"], "Claude Sonnet 4");
        assert_eq!(m1["context_length"], 200000);
        let m2 = &json["data"][1];
        assert_eq!(m2["id"], "deepseek-v3.1");
        assert_eq!(m2["display_name"], "DeepSeek V3.1");
        assert_eq!(m2["context_length"], 131072);
        assert_eq!(json["first_id"], "claude-sonnet-4");
        assert_eq!(json["last_id"], "deepseek-v3.1");
    }

    #[tokio::test]
    async fn models_without_catalog_returns_empty_list() {
        let state = build_state_without_catalog();
        let response = handle_models_inner(&state, "no-catalog", models_request(""))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), 16384)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 0);
        assert_eq!(json["has_more"], false);
        assert!(json["first_id"].is_null());
        assert!(json["last_id"].is_null());
    }

    #[tokio::test]
    async fn models_unknown_provider_returns_404() {
        let state = build_state_with_catalog();
        let result = handle_models_inner(&state, "nonexistent", models_request("")).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RouteError::UnknownProvider(_)));
    }

    #[test]
    fn invalid_catalog_config_maps_to_internal_error() {
        let error = ProviderError::InvalidConfig(
            "failed to parse catalog cache /Users/example/private/catalog.toml".to_owned(),
        );

        let route_error = map_catalog_error(error);

        assert!(matches!(route_error, RouteError::Internal(_)));
    }

    #[test]
    fn sanitized_upstream_api_error_remains_upstream_error() {
        let error = ProviderError::api(429, "rate limited".to_owned());

        let route_error = map_catalog_error(error);

        assert!(matches!(
            route_error,
            RouteError::Upstream {
                status: StatusCode::TOO_MANY_REQUESTS,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn model_card_has_no_credentials() {
        let state = build_state_with_catalog();
        let response = handle_models_inner(&state, "test-provider", models_request(""))
            .await
            .expect("response");

        let body = axum::body::to_bytes(response.into_body(), 16384)
            .await
            .expect("body");
        let body_str = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(
            !body_str.contains("sk-test"),
            "model endpoint output must not contain credentials, got: {body_str}"
        );
    }

    #[test]
    fn source_guard_models_uses_current_architecture() {
        let source = include_str!("models.rs");
        let prod = source
            .split_once("#[cfg(test)]")
            .map(|(p, _)| p)
            .unwrap_or(source);

        assert!(!prod.contains("ApiError"), "models.rs must use RouteError");
        assert!(
            !prod.contains("crate::error::"),
            "models.rs must not import from crate::error"
        );
    }
}
