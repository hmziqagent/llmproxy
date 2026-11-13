use std::sync::Arc;
use std::time::Instant;

use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderName, HeaderValue, Method, StatusCode, header},
    middleware::{Next, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use tower::ServiceBuilder;
use tower_http::{
    catch_panic::CatchPanicLayer, cors::CorsLayer, set_header::SetResponseHeaderLayer,
    timeout::TimeoutLayer, trace::TraceLayer,
};
use tracing::info_span;

use crate::middleware::{CallerId, RequestId, RequestIdGenerator, extract_bearer_token};
use crate::state::AppState;
use llm_proxy_storage::EventBus;
use secrecy::ExposeSecret;

mod chat;
mod core_pipeline;
mod error_response;
#[cfg(feature = "endpoint-families")]
mod families;
mod gemini_generate;
mod health;
mod messages;
mod models;
mod ollama;
mod responses;
mod token_count;

use chat::{handle_bare_chat_completions, handle_chat_completions};
#[cfg(feature = "endpoint-families")]
use families::{
    FAMILY_AUDIO_SPEECH, FAMILY_AUDIO_TRANSCRIPTIONS, FAMILY_BATCHES, FAMILY_EMBEDDINGS,
    FAMILY_FILES, FAMILY_IMAGES_GENERATIONS, FAMILY_RERANK, FamilySpec, handle_family,
    handle_family_resource,
};
use gemini_generate::handle_generate_content;
use health::{health, ready, version};
use llm_proxy_core::{family_max_body_bytes, family_request_timeout};
use messages::{handle_bare_messages, handle_messages};
use models::handle_models;
use ollama::{handle_ollama_chat, handle_ollama_generate, handle_ollama_tags};
use responses::handle_responses;
use token_count::count_tokens;

/// Endpoint families (audit conflict 16): admission, routing and accounting
/// are generic; the body limit and timeout are per-family settings.
pub(crate) const FAMILY_MESSAGES: &str = "messages";
pub(crate) const FAMILY_COUNT_TOKENS: &str = "count_tokens";
pub(crate) const FAMILY_CHAT_COMPLETIONS: &str = "chat_completions";
pub(crate) const FAMILY_MODELS: &str = "models";
pub(crate) const FAMILY_RESPONSES: &str = "responses";
pub(crate) const FAMILY_GENERATE_CONTENT: &str = "generate_content";
pub(crate) const FAMILY_OLLAMA: &str = "ollama";

/// Wraps one family's fixed wiring for its handlers.
#[cfg(feature = "endpoint-families")]
fn family(spec: FamilySpec) -> axum::Extension<FamilySpec> {
    axum::Extension(spec)
}

const NOT_FOUND_BODY_DRAIN_LIMIT: usize = 1024;

/// Build the full HTTP router: `/health`, `/ready`, `/version`, provider API
/// routes under `/providers/{provider}/v1/*`, and the 404/405 fallbacks. The
/// admin `/usage` surface merges in only under the `admin_on_main_listener` opt-in.
pub fn router(state: AppState) -> Router {
    let id_gen = Arc::clone(&state.request_id_gen);
    let trace = TraceLayer::new_for_http().make_span_with(make_request_span);
    let lightweight = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/version", get(version))
        .layer(
            ServiceBuilder::new()
                .layer(from_fn_with_state(Arc::clone(&id_gen), inject_request_id))
                .layer(from_fn_with_state(
                    state.clone(),
                    crate::host_guard::enforce_host_origin,
                ))
                .layer(trace),
        );

    let api = provider_api_router(&state);

    let router = Router::new()
        .merge(lightweight)
        .merge(api)
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed);
    let usage_on_main = match (
        crate::admin_router(&state),
        state.app_config.server.admin_on_main_listener,
    ) {
        (Some(usage), true) => Some(usage),
        _ => None,
    };
    let router = match cors_layer_for(state.allowed_origins()) {
        Some(cors) => router.layer(cors),
        None => router,
    };

    let router: Router = router
        .layer(SetResponseHeaderLayer::if_not_present(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        .layer(CatchPanicLayer::new())
        .with_state(state);
    let router = match usage_on_main {
        Some(usage) => router.merge(usage),
        None => router,
    };
    // The outermost layer (plan 05 step 1): every error the router emits —
    // including the loop guard's 508 and the host guard's 403, whose inline
    // bodies never reach `normalize_error_responses`, and the 404/405
    // fallbacks and a merged admin router — leaves carrying the one policy
    // table's `x-should-retry` row.
    router.layer(axum::middleware::from_fn(stamp_error_policy))
}

/// The provider-scoped API surface: every client dialect's route, each under
/// its endpoint family's admission chain, plus the opt-in bare `/v1/*` routes
/// (plan 04 step 11).
#[expect(
    clippy::too_many_lines,
    reason = "a route table with one block per endpoint family; the family loops below replaced a longer unrolled list"
)]
fn provider_api_router(state: &AppState) -> Router<AppState> {
    let id_gen = Arc::clone(&state.request_id_gen);
    let event_bus = Arc::clone(&state.event_bus);
    let config = &state.app_config().server;
    let trace = TraceLayer::new_for_http().make_span_with(make_request_span);

    // Shared admission middleware; the body limit and timeout layer on top
    // are per endpoint family (conflict 16), so each family composes its own.
    // Order: request id, then the self-loop refusal (508, backlog 32), then
    // the host/origin gate, then client auth. The host/origin gate is the
    // DNS-rebinding defense (backlog 11) under anonymous auth; the loop
    // refusal runs under every auth mode because a loop has no caller.
    let admission = |family: &'static str| {
        ServiceBuilder::new()
            .layer(from_fn_with_state(Arc::clone(&id_gen), inject_request_id))
            .layer(from_fn_with_state(
                state.clone(),
                crate::loop_guard::refuse_looped_requests,
            ))
            .layer(from_fn_with_state(
                state.clone(),
                crate::host_guard::enforce_host_origin,
            ))
            .layer(from_fn_with_state(state.clone(), require_client_auth))
            .layer(from_fn_with_state(
                Arc::clone(&event_bus),
                normalize_error_responses,
            ))
            .layer(DefaultBodyLimit::max(family_max_body_bytes(config, family)))
            .layer(trace.clone())
            .layer(TimeoutLayer::with_status_code(
                StatusCode::REQUEST_TIMEOUT,
                family_request_timeout(config, family),
            ))
    };

    let messages = Router::new()
        .route("/providers/{provider}/v1/messages", post(handle_messages))
        .layer(admission(FAMILY_MESSAGES));
    let count_tokens = Router::new()
        .route(
            "/providers/{provider}/v1/messages/count_tokens",
            post(count_tokens),
        )
        .layer(admission(FAMILY_COUNT_TOKENS));
    let chat = Router::new()
        .route(
            "/providers/{provider}/v1/chat/completions",
            post(handle_chat_completions),
        )
        .layer(admission(FAMILY_CHAT_COMPLETIONS));
    let models = Router::new()
        .route("/providers/{provider}/v1/models", get(handle_models))
        .layer(admission(FAMILY_MODELS));
    let responses = Router::new()
        .route("/providers/{provider}/v1/responses", post(handle_responses))
        .layer(admission(FAMILY_RESPONSES));
    let generate_content = Router::new()
        .route(
            "/providers/{provider}/v1beta/models/{model_method}",
            post(handle_generate_content),
        )
        .layer(admission(FAMILY_GENERATE_CONTENT));
    let ollama = Router::new()
        .route("/providers/{provider}/api/chat", post(handle_ollama_chat))
        .route(
            "/providers/{provider}/api/generate",
            post(handle_ollama_generate),
        )
        .layer(admission(FAMILY_OLLAMA));
    let ollama_tags = Router::new()
        .route("/providers/{provider}/api/tags", get(handle_ollama_tags))
        .layer(admission(FAMILY_MODELS));

    let mut api = messages
        .merge(count_tokens)
        .merge(chat)
        .merge(models)
        .merge(responses)
        .merge(generate_content)
        .merge(ollama)
        .merge(ollama_tags);

    // Non-chat families (plan 09 step 7), each behind its own admission
    // chain: its body limit and timeout from
    // [server.endpoint_families], the same loop/host/auth gates, and a
    // family-specific handler. Each route serves the methods of the
    // family's documented API; the handler forwards the client's method.
    #[cfg(feature = "endpoint-families")]
    {
        use axum::routing::{MethodFilter, on};
        use llm_proxy_core::ProviderRouteKind as Kind;
        let create = MethodFilter::POST;
        let list_or_create = MethodFilter::GET.or(MethodFilter::POST);
        // (path, route kind, JSON request body, family name, methods)
        for (path, kind, json, family_name, methods) in [
            (
                "/providers/{provider}/v1/embeddings",
                Kind::Embeddings,
                true,
                FAMILY_EMBEDDINGS,
                create,
            ),
            (
                "/providers/{provider}/v1/rerank",
                Kind::Rerank,
                true,
                FAMILY_RERANK,
                create,
            ),
            (
                "/providers/{provider}/v1/images/generations",
                Kind::ImagesGenerations,
                true,
                FAMILY_IMAGES_GENERATIONS,
                create,
            ),
            (
                "/providers/{provider}/v1/audio/transcriptions",
                Kind::AudioTranscriptions,
                false,
                FAMILY_AUDIO_TRANSCRIPTIONS,
                create,
            ),
            // Speech takes JSON (so the alias applies) and answers audio.
            (
                "/providers/{provider}/v1/audio/speech",
                Kind::AudioSpeech,
                true,
                FAMILY_AUDIO_SPEECH,
                create,
            ),
            // Files: list and upload; retrieve, download and delete.
            (
                "/providers/{provider}/v1/files",
                Kind::Files,
                false,
                FAMILY_FILES,
                list_or_create,
            ),
            (
                "/providers/{provider}/v1/files/{*resource}",
                Kind::Files,
                false,
                FAMILY_FILES,
                MethodFilter::GET.or(MethodFilter::DELETE),
            ),
            // Batches: list and create; retrieve and cancel.
            (
                "/providers/{provider}/v1/batches",
                Kind::Batches,
                true,
                FAMILY_BATCHES,
                list_or_create,
            ),
            (
                "/providers/{provider}/v1/batches/{*resource}",
                Kind::Batches,
                true,
                FAMILY_BATCHES,
                list_or_create,
            ),
        ] {
            let route = if path.ends_with("{*resource}") {
                on(methods, handle_family_resource)
            } else {
                on(methods, handle_family)
            };
            api = api.merge(
                Router::new()
                    .route(path, route)
                    .route_layer(family(FamilySpec { kind, json }))
                    .layer(admission(family_name)),
            );
        }
    }

    // Bare `/v1/*` is off by default (plan 04 step 11): with it off, those
    // paths fall through to the 404 fallback; enabled, the model name
    // resolves to a provider or pool through the same engine.
    if state.app_config().server.bare_v1.enabled {
        let bare_messages = Router::new()
            .route("/v1/messages", post(handle_bare_messages))
            .layer(admission(FAMILY_MESSAGES));
        let bare_chat = Router::new()
            .route("/v1/chat/completions", post(handle_bare_chat_completions))
            .layer(admission(FAMILY_CHAT_COMPLETIONS));
        api = api.merge(bare_messages).merge(bare_chat);
    }
    api
}
fn cors_layer_for(origins: Option<&[String]>) -> Option<CorsLayer> {
    let origins = origins?;
    let parsed: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|origin| match origin.parse::<HeaderValue>() {
            Ok(value) => Some(value),
            Err(error) => {
                tracing::warn!(%error, origin, "skipping invalid server.allowed_origins entry");
                None
            }
        })
        .collect();
    if parsed.is_empty() {
        return None;
    }
    let methods = [Method::GET, Method::POST, Method::OPTIONS];
    let headers = [
        header::CONTENT_TYPE,
        header::AUTHORIZATION,
        header::ACCEPT,
        header::USER_AGENT,
        HeaderName::from_static("x-api-key"),
        HeaderName::from_static("anthropic-version"),
        HeaderName::from_static("anthropic-beta"),
        HeaderName::from_static("x-goog-api-key"),
        HeaderName::from_static("x-request-id"),
    ];
    Some(
        CorsLayer::new()
            .allow_origin(parsed)
            .allow_methods(methods)
            .allow_headers(headers),
    )
}
async fn inject_request_id(
    State(id_gen): State<Arc<RequestIdGenerator>>,
    mut req: Request,
    next: Next,
) -> Response {
    let id = id_gen.next_id();
    req.extensions_mut().insert(RequestId(id));
    next.run(req).await
}
async fn require_client_auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    // The pair fetch is what makes runtime key changes visible here: the
    // daemon swaps the authenticator (together with the admission rules,
    // one lock) without a restart, so each request reads the half serving
    // right now.
    let client_auth = state.client_auth.current_client_auth();
    if !client_auth.is_enforced() {
        let mut req = req;
        req.extensions_mut().insert(CallerId::anonymous());
        return next.run(req).await;
    }
    let connect_info = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let client_ip = crate::middleware::get_client_ip(
        req.headers(),
        connect_info.as_ref(),
        state.trust_forwarded_headers(),
    );

    let authenticated = match extract_bearer_token(req.headers()) {
        Some(token) => {
            if !state.rate_limiter.would_allow(&client_ip) {
                state.metrics.record_rate_limited();
                None
            } else {
                client_auth.authenticate(token.expose_secret()).await
            }
        }
        None => None,
    };

    let mut req = req;
    match authenticated {
        Some(key_id) => {
            tracing::debug!(caller = %key_id, "client authenticated");
            req.extensions_mut().insert(CallerId::authenticated(key_id));
            next.run(req).await
        }
        None => {
            if !state.rate_limiter.is_allowed(&client_ip) {
                state.metrics.record_rate_limited();
            }
            let request_id = req
                .extensions()
                .get::<RequestId>()
                .cloned()
                .unwrap_or_else(|| RequestId("unknown".to_owned()));
            let path = req.uri().path().to_owned();
            if let Err(e) = axum::body::to_bytes(req.into_body(), NOT_FOUND_BODY_DRAIN_LIMIT).await
            {
                tracing::trace!(error = %e, "draining body of auth-rejected request failed");
            }
            let error = error_response::RouteError::Unauthorized;
            core_pipeline::emit_response_failed(
                &state.event_bus,
                &request_id.0,
                None,
                None,
                &error,
                std::time::Instant::now(),
            );
            let mut response =
                error_response::route_error_response(protocol_for_path(&path), &error);
            if let Ok(value) = request_id.0.parse() {
                response.headers_mut().insert("x-request-id", value);
            }
            response
        }
    }
}
fn make_request_span(req: &Request) -> tracing::Span {
    let id = req.extensions().get::<RequestId>();
    info_span!(
        "request",
        request_id = %id.cloned().unwrap_or_else(|| RequestId(String::new())),
        method = %req.method(),
        path = %req.uri().path(),
    )
}
/// Stamps the policy table's `x-should-retry` row onto every error response
/// that bubbles out of the inner layers without one (plan 05 step 1): the
/// guards' inline bodies, the 404/405 fallbacks, panic 500s, and a merged
/// admin router. See [`error_response::stamp_retry_policy`].
async fn stamp_error_policy(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    error_response::stamp_retry_policy(&mut response);
    response
}
async fn normalize_error_responses(
    State(event_bus): State<Arc<dyn EventBus>>,
    req: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let protocol = protocol_for_path(req.uri().path());
    let request_id = req.extensions().get::<RequestId>().cloned();
    let response = next.run(req).await;
    let status = response.status();

    if status != StatusCode::REQUEST_TIMEOUT && status != StatusCode::PAYLOAD_TOO_LARGE {
        return response;
    }
    if let Err(e) = axum::body::to_bytes(response.into_body(), NOT_FOUND_BODY_DRAIN_LIMIT).await {
        tracing::trace!(error = %e, "draining layer rejection body failed");
    }

    let error = if status == StatusCode::REQUEST_TIMEOUT {
        error_response::RouteError::RequestTimeout
    } else {
        error_response::RouteError::PayloadTooLarge
    };
    let id_str = request_id
        .as_ref()
        .map(|id| id.0.as_str())
        .unwrap_or("layer-rejection");
    core_pipeline::emit_response_failed(&event_bus, id_str, None, None, &error, start);

    let mut rewritten = error_response::route_error_response(protocol, &error);
    if let Some(id) = request_id {
        if let Ok(value) = id.0.parse() {
            rewritten.headers_mut().insert("x-request-id", value);
        }
    }
    rewritten
}
/// Path prefixes of the non-chat OpenAI-dialect endpoint families
/// (plan 09 step 7, plus the completions/moderations text families).
const OPENAI_FAMILY_PATHS: &[&str] = &[
    "/v1/completions",
    "/v1/moderations",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/audio/",
    "/v1/images/",
    "/v1/files",
    "/v1/batches",
];

fn protocol_for_path(path: &str) -> error_response::ClientProtocol {
    // The dialect of the path decides the error body shape for layer-level
    // failures (auth, limits, 404/405) that never reach a route handler.
    // Everything under a `chat` segment is OpenAI-shaped, including unknown
    // subpaths that only hit the 404/405 fallback (remodel plan). `/v1/models`
    // has no single inbound client protocol — its listing body is an
    // OpenAI+Anthropic superset and the handler already answers OpenAI-shaped
    // errors — so its layer failures are OpenAI-shaped too (remodel plan).
    if path.contains("/v1/chat/")
        || path.ends_with("/v1/chat")
        || path.ends_with("/v1/models")
        || OPENAI_FAMILY_PATHS
            .iter()
            .any(|prefix| path.contains(prefix))
    {
        error_response::ClientProtocol::OpenAiChat
    } else if path.contains("/v1/responses") {
        error_response::ClientProtocol::OpenAiResponses
    } else if path.contains("/v1beta/") {
        error_response::ClientProtocol::Gemini
    } else if path.contains("/api/") {
        error_response::ClientProtocol::Ollama
    } else {
        error_response::ClientProtocol::Anthropic
    }
}

async fn not_found(State(state): State<AppState>, req: Request) -> impl IntoResponse {
    let start = Instant::now();
    let request_id = req
        .extensions()
        .get::<RequestId>()
        .cloned()
        .unwrap_or_else(|| RequestId("unknown".to_owned()));

    let path = req.uri().path().to_owned();
    if let Err(e) = axum::body::to_bytes(req.into_body(), NOT_FOUND_BODY_DRAIN_LIMIT).await {
        tracing::trace!(error = %e, "body drain in 404 handler failed");
    }

    let error = error_response::RouteError::NotFound;
    core_pipeline::emit_response_failed(&state.event_bus, &request_id.0, None, None, &error, start);

    let protocol = protocol_for_path(&path);
    let mut response = error_response::route_error_response(protocol, &error);
    if let Ok(value) = request_id.0.parse() {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}
async fn method_not_allowed(State(state): State<AppState>, req: Request) -> impl IntoResponse {
    let start = Instant::now();
    let request_id = req
        .extensions()
        .get::<RequestId>()
        .cloned()
        .unwrap_or_else(|| RequestId("unknown".to_owned()));

    let path = req.uri().path().to_owned();
    if let Err(e) = axum::body::to_bytes(req.into_body(), NOT_FOUND_BODY_DRAIN_LIMIT).await {
        tracing::trace!(error = %e, "body drain in 405 handler failed");
    }

    let error = error_response::RouteError::MethodNotAllowed;
    core_pipeline::emit_response_failed(&state.event_bus, &request_id.0, None, None, &error, start);

    let protocol = protocol_for_path(&path);
    let mut response = error_response::route_error_response(protocol, &error);
    if let Ok(value) = request_id.0.parse() {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::{cors_layer_for, error_response, protocol_for_path};

    #[test]
    fn protocol_for_path_classifies_chat_prefixes_as_openai() {
        for path in [
            "/providers/p/v1/chat/completions",
            "/providers/p/v1/chat/typo",
            "/providers/p/v1/chat",
            "/v1/chat/completions",
            "/providers/p/v1/completions",
            "/providers/p/v1/moderations",
        ] {
            assert_eq!(
                protocol_for_path(path),
                error_response::ClientProtocol::OpenAiChat,
                "{path} must classify as OpenAI-shaped"
            );
        }
    }

    #[test]
    fn protocol_for_path_classifies_models_listing_as_openai() {
        for path in [
            "/providers/p/v1/models",
            "/providers/nope/v1/models",
            "/v1/models",
        ] {
            assert_eq!(
                protocol_for_path(path),
                error_response::ClientProtocol::OpenAiChat,
                "{path} must classify as OpenAI-shaped so layer failures match the handler"
            );
        }
    }

    #[test]
    fn protocol_for_path_keeps_non_chat_families() {
        assert_eq!(
            protocol_for_path("/providers/p/v1/messages"),
            error_response::ClientProtocol::Anthropic
        );
        assert_eq!(
            protocol_for_path("/providers/p/v1/messages/count_tokens"),
            error_response::ClientProtocol::Anthropic
        );
        assert_eq!(
            protocol_for_path("/providers/p/v1/responses"),
            error_response::ClientProtocol::OpenAiResponses
        );
        assert_eq!(
            protocol_for_path("/providers/p/v1beta/models/gemini:generateContent"),
            error_response::ClientProtocol::Gemini
        );
        assert_eq!(
            protocol_for_path("/api/tags"),
            error_response::ClientProtocol::Ollama
        );
    }

    #[test]
    fn cors_layer_absent_when_origins_unset() {
        assert!(cors_layer_for(None).is_none());
    }

    #[test]
    fn cors_layer_absent_when_origins_empty() {
        let empty: Vec<String> = Vec::new();
        assert!(cors_layer_for(Some(&empty)).is_none());
    }

    #[test]
    fn cors_layer_absent_when_every_origin_is_invalid() {
        let bad = vec!["bad\norigin".to_owned(), "\0".to_owned()];
        assert!(cors_layer_for(Some(&bad)).is_none());
    }

    #[test]
    fn cors_layer_present_when_origin_configured() {
        let origins = vec!["https://app.example.com".to_owned()];
        assert!(cors_layer_for(Some(&origins)).is_some());
    }

    #[test]
    fn cors_layer_keeps_valid_origins_while_skipping_invalid() {
        let origins = vec![
            "bad\norigin".to_owned(),
            "https://app.example.com".to_owned(),
        ];
        assert!(cors_layer_for(Some(&origins)).is_some());
    }
}
#[cfg(test)]
mod auth_gate_tests {
    use super::*;

    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;
    fn auth_router(rate_limit_rpm: u32) -> (Router, AppState) {
        use argon2::{Algorithm, Argon2, CustomizedPasswordHasher, Params, Version};

        let params = Params::new(8, 1, 1, None).expect("valid test params");
        let hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params.clone());
        let phc = hasher
            .hash_password_customized(
                b"sekrit-token",
                b"0123456789abcdef",
                Some(&argon2::ARGON2ID_IDENT),
                None,
                params,
            )
            .expect("hash token")
            .to_string();

        let mut clients = std::collections::HashMap::new();
        clients.insert(
            "laptop".to_owned(),
            llm_proxy_core::ClientKeySpec::Verifier(secrecy::SecretString::from(phc)),
        );
        let app_config = llm_proxy_core::AppConfig {
            server: llm_proxy_core::ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: std::time::Duration::from_secs(300),
                shutdown_timeout: std::time::Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "gate-test".to_owned(),
                rate_limit_rpm,
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
            auth: llm_proxy_core::AuthConfig {
                mode: llm_proxy_core::AuthMode::ClientKeys,
                clients,
                admin: None,
                legacy_bare_tokens: false,
            },
            storage: None,
            replica: None,
            mcp: None,
        };

        let state = crate::state::AppState::new(
            app_config,
            llm_proxy_core::ProviderRegistry::from_providers(vec![]).expect("empty registry"),
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            llm_proxy_transport::ProxyClient::try_new().expect("client"),
            crate::state::BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
        .expect("app state");

        let router = Router::new()
            .route(
                "/providers/x/v1/messages",
                post(|| async { StatusCode::OK }),
            )
            .layer(from_fn_with_state(
                state.clone(),
                super::require_client_auth,
            ));
        (router, state)
    }

    fn bearer_request(token: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/x/v1/messages")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn valid_token_passes_when_the_peer_has_budget() {
        let (app, _state) = auth_router(100);
        let response = app
            .oneshot(bearer_request("laptop.sekrit-token"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn exhausted_failure_budget_gates_verification_before_argon2() {
        let (app, state) = auth_router(1);
        let first = app
            .clone()
            .oneshot(bearer_request("laptop.wrong-token"))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(state.metrics.get_snapshot().rate_limited, 0);
        let second = app
            .oneshot(bearer_request("laptop.sekrit-token"))
            .await
            .unwrap();
        assert_eq!(
            second.status(),
            StatusCode::UNAUTHORIZED,
            "a peer with an exhausted failure budget must be gated before verification"
        );
        assert!(
            state.metrics.get_snapshot().rate_limited >= 1,
            "the gate records a limiter rejection"
        );
    }
}
