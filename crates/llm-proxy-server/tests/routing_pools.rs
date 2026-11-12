//! Plan 04 server-surface tests: pools served at `/providers/{pool}/v1/...`,
//! the retry-exhaustion response signals, and the bare `/v1/*` opt-in.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::post;
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{
    AppConfig, AuthStyle, BareV1Config, PoolConfig, ProviderAdapterConfig, ProviderConfig,
    ProviderRegistry, ProviderRoutesConfig, RetryConfig, ServerConfig,
};
use llm_proxy_server::{AppState, BuildInfo, build_router};
use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};
use llm_proxy_transport::ProxyClient;
use tower::ServiceExt;

const ANTHROPIC_OK_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#;

async fn spawn_json_upstream(
    status: StatusCode,
    body: &'static str,
    retry_after: Option<&'static str>,
) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, json_router(status, body, retry_after)).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    format!("http://{addr}/v1/messages")
}

fn json_router(
    status: StatusCode,
    body: &'static str,
    retry_after: Option<&'static str>,
) -> Router {
    Router::new().route(
        "/{*path}",
        post(move || async move {
            let mut response = (
                status,
                [(header::CONTENT_TYPE, "application/json")],
                body.to_owned(),
            )
                .into_response();
            if let Some(value) = retry_after
                && let Ok(header_value) = value.parse()
            {
                response
                    .headers_mut()
                    .insert(header::RETRY_AFTER, header_value);
            }
            response
        }),
    )
}

fn member(name: &str, endpoint: String) -> ProviderConfig {
    ProviderConfig {
        name: name.to_owned(),
        api_key: secrecy::SecretString::from("k"),
        api_keys: None,
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        retry: None,
        egress: None,
        adapters: HashMap::from([(
            "messages".to_owned(),
            ProviderAdapterConfig {
                protocol: "anthropic_messages".to_owned(),
                endpoint,
                headers: Arc::new(HashMap::new()),
                forward_inbound_headers: None,
                lossy: false,
            },
        )]),
        routes: ProviderRoutesConfig {
            messages: Some("messages".to_owned()),
            chat_completions: None,
            ..Default::default()
        },
        model_aliases: HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn ladder() -> RetryConfig {
    RetryConfig {
        max_attempts: 1,
        backoff: Duration::from_millis(5),
        cooldown: Duration::from_millis(50),
        total_timeout: Duration::ZERO,
        hold: Duration::ZERO,
    }
}

fn state_with(
    providers: Vec<ProviderConfig>,
    pools: Vec<PoolConfig>,
    bare: BareV1Config,
    bus: Arc<RecordingBus>,
) -> AppState {
    let app_config = AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:3456".parse().unwrap(),
            request_timeout: Duration::from_secs(300),
            shutdown_timeout: Duration::from_secs(30),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "routing-test".to_owned(),
            rate_limit_rpm: 1000,
            trust_forwarded_headers: false,
            cache: Default::default(),
            idempotency: Default::default(),
            metrics: Default::default(),
            log_format: Default::default(),
            admin_bind: None,
            admin_on_main_listener: false,
            endpoint_families: Default::default(),
            upstream_timeouts: Default::default(),
            bare_v1: bare,
            preflight: Default::default(),
            semantic_cache: Default::default(),
        },
        auth: Default::default(),
        storage: None,
        replica: None,
        mcp: None,
    };
    let registry = ProviderRegistry::from_parts(providers, pools).expect("registry with pool");
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
    .with_event_bus(bus as Arc<dyn EventBus>)
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
        .unwrap()
}

#[tokio::test]
async fn a_pool_route_survives_its_first_members_429_over_http() {
    let limited = spawn_json_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow"}"#,
        Some("1"),
    )
    .await;
    let healthy = spawn_json_upstream(StatusCode::OK, ANTHROPIC_OK_BODY, None).await;
    let bus = Arc::new(RecordingBus::new());
    let state = state_with(
        vec![member("limited", limited), member("healthy", healthy)],
        vec![PoolConfig {
            name: "team".to_owned(),
            members: vec!["limited".to_owned(), "healthy".to_owned()],
            retry: ladder(),
            affinity: Default::default(),
            health: Default::default(),
        }],
        BareV1Config::default(),
        Arc::clone(&bus),
    );
    let app = build_router(state);

    let response = app
        .oneshot(messages_request(
            "/providers/team/v1/messages",
            "claude-sonnet-4-6",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let attempt_rows = bus
        .snapshot()
        .iter()
        .filter(|event| matches!(event, ProxyEvent::AttemptCompleted(_)))
        .count();
    assert_eq!(attempt_rows, 2, "the ledger shows every attempt");
}

#[tokio::test]
async fn an_exhausted_pool_carries_the_retry_terminality_signals() {
    let first = spawn_json_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"a"}"#,
        Some("11"),
    )
    .await;
    let second = spawn_json_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"b"}"#,
        Some("13"),
    )
    .await;
    let bus = Arc::new(RecordingBus::new());
    let state = state_with(
        vec![member("first", first), member("second", second)],
        vec![PoolConfig {
            name: "limited".to_owned(),
            members: vec!["first".to_owned(), "second".to_owned()],
            retry: ladder(),
            affinity: Default::default(),
            health: Default::default(),
        }],
        BareV1Config::default(),
        bus,
    );
    let app = build_router(state);

    let response = app
        .oneshot(messages_request(
            "/providers/limited/v1/messages",
            "claude-sonnet-4-6",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("x-should-retry")
            .and_then(|v| v.to_str().ok()),
        Some("false"),
        "the proxy exhausted its own retries; the client must not retry"
    );
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("13"),
        "the computed Retry-After reflects the longest advertised reset"
    );
}

#[tokio::test]
async fn a_direct_provider_route_429_follows_the_policy_table() {
    let solo = spawn_json_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow"}"#,
        Some("3"),
    )
    .await;
    let bus = Arc::new(RecordingBus::new());
    let state = state_with(
        vec![member("solo", solo)],
        vec![],
        BareV1Config::default(),
        bus,
    );
    let app = build_router(state);

    let response = app
        .oneshot(messages_request(
            "/providers/solo/v1/messages",
            "claude-sonnet-4-6",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    // One policy table (plan 05 step 1): a 429 the proxy did not retry
    // itself is a genuine upstream throttle the client may retry after the
    // advised wait — true, unlike an exhausted ladder — and the upstream's
    // Retry-After survives the mapping.
    assert_eq!(
        response
            .headers()
            .get("x-should-retry")
            .and_then(|v| v.to_str().ok()),
        Some("true"),
        "a 429 on a route with no ladder is retryable by the client"
    );
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("3"),
        "the upstream-advised wait reaches the client"
    );
}

#[tokio::test]
async fn bare_v1_routes_return_404_when_left_off() {
    let bus = Arc::new(RecordingBus::new());
    let state = state_with(vec![], vec![], BareV1Config::default(), bus);
    let app = build_router(state);

    for uri in ["/v1/messages", "/v1/chat/completions"] {
        let response = app
            .clone()
            .oneshot(messages_request(uri, "claude-sonnet-4-6"))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "bare {uri} must 404 while server.bare_v1 is off"
        );
    }
}

#[tokio::test]
async fn bare_v1_resolves_a_model_to_a_provider_when_enabled() {
    let healthy = spawn_json_upstream(StatusCode::OK, ANTHROPIC_OK_BODY, None).await;
    let bus = Arc::new(RecordingBus::new());
    let state = state_with(
        vec![member("healthy", healthy)],
        vec![],
        BareV1Config {
            enabled: true,
            models: HashMap::from([("claude-sonnet-4-6".to_owned(), "healthy".to_owned())]),
            default_target: Some("healthy".to_owned()),
        },
        bus,
    );
    let app = build_router(state);

    // A mapped model resolves and runs through the same engine.
    let response = app
        .clone()
        .oneshot(messages_request("/v1/messages", "claude-sonnet-4-6"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // An unmapped model falls back to the default target.
    let response = app
        .clone()
        .oneshot(messages_request("/v1/messages", "claude-opus-4-6"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn bare_v1_resolves_a_model_to_a_pool_when_enabled() {
    let limited =
        spawn_json_upstream(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow"}"#, None).await;
    let healthy = spawn_json_upstream(StatusCode::OK, ANTHROPIC_OK_BODY, None).await;
    let bus = Arc::new(RecordingBus::new());
    let state = state_with(
        vec![member("limited", limited), member("healthy", healthy)],
        vec![PoolConfig {
            name: "team".to_owned(),
            members: vec!["limited".to_owned(), "healthy".to_owned()],
            retry: ladder(),
            affinity: Default::default(),
            health: Default::default(),
        }],
        BareV1Config {
            enabled: true,
            models: HashMap::from([("claude-sonnet-4-6".to_owned(), "team".to_owned())]),
            default_target: None,
        },
        Arc::clone(&bus),
    );
    let app = build_router(state);

    let response = app
        .oneshot(messages_request("/v1/messages", "claude-sonnet-4-6"))
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "the bare route runs the pool's failover like the scoped route"
    );
    let attempt_rows = bus
        .snapshot()
        .iter()
        .filter(|event| matches!(event, ProxyEvent::AttemptCompleted(_)))
        .count();
    assert_eq!(attempt_rows, 2);
}
