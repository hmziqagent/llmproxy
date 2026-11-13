//! Router-level policy enforcement (fix brief 05-2): every error the router
//! emits — guard refusals, fallbacks, layer rejections and engine errors —
//! must name a row in the engine's one status policy table and carry exactly
//! `x-should-retry: true` or `false`. A status added anywhere without a
//! policy row fails here.

#![cfg(test)]

use std::collections::HashMap;
use std::time::Duration;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{
    AppConfig, AuthConfig, AuthMode, AuthStyle, ClientKeySpec, EndpointFamilyConfig, HOP_HEADER,
    ProviderAdapterConfig, ProviderConfig, ProviderRegistry, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_server::{AppState, BuildInfo, build_router};
use llm_proxy_transport::ProxyClient;
use serde_json::{Value, json};
use tower::ServiceExt;

fn app_config(auth: AuthConfig, families: HashMap<String, EndpointFamilyConfig>) -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:3456".parse().unwrap(),
            request_timeout: Duration::from_secs(300),
            shutdown_timeout: Duration::from_secs(30),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "error-policy-router-test".to_owned(),
            rate_limit_rpm: 100,
            trust_forwarded_headers: false,
            cache: Default::default(),
            idempotency: Default::default(),
            metrics: Default::default(),
            log_format: Default::default(),
            admin_bind: None,
            admin_on_main_listener: false,
            endpoint_families: families,
            upstream_timeouts: Default::default(),
            bare_v1: Default::default(),
            preflight: Default::default(),
            semantic_cache: Default::default(),
        },
        auth,
        storage: None,
        replica: None,
        mcp: None,
    }
}

fn state(auth: AuthConfig, families: HashMap<String, EndpointFamilyConfig>) -> AppState {
    let provider = ProviderConfig {
        name: "mock".to_owned(),
        api_key: secrecy::SecretString::from("test-key"),
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters: HashMap::from([(
            "messages".to_owned(),
            ProviderAdapterConfig {
                protocol: "anthropic_messages".to_owned(),
                endpoint: "https://127.0.0.1:9/v1/messages".to_owned(),
                headers: std::sync::Arc::new(HashMap::new()),
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
    };
    let registry = ProviderRegistry::from_providers(vec![provider]).expect("registry");
    AppState::new(
        app_config(auth, families),
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

fn anonymous_state() -> AppState {
    state(AuthConfig::default(), HashMap::new())
}

fn client_keys_state() -> AppState {
    let auth = AuthConfig {
        mode: AuthMode::ClientKeys,
        clients: HashMap::from([(
            "laptop".to_owned(),
            ClientKeySpec::Verifier(secrecy::SecretString::from(
                "$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA",
            )),
        )]),
        admin: None,
        legacy_bare_tokens: false,
    };
    state(auth, HashMap::new())
}

fn tiny_body_limit_state() -> AppState {
    let families = HashMap::from([(
        "messages".to_owned(),
        EndpointFamilyConfig {
            max_body_bytes: 16,
            request_timeout: Duration::from_secs(30),
        },
    )]);
    state(AuthConfig::default(), families)
}

fn post(path: &str, headers: &[(&'static str, &'static str)], body: &str) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder.body(Body::from(body.to_owned())).unwrap()
}

fn messages_body() -> String {
    json!({
        "model": "claude-sonnet-4-6",
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 8
    })
    .to_string()
}

async fn assert_policy_row_and_header(response: axum::response::Response, expected: StatusCode) {
    let status = response.status();
    assert_eq!(status, expected, "scenario must produce its status");
    assert!(
        llm_proxy_engine::policy_for(status.as_u16()).is_some(),
        "status {status} the router emitted has no policy row"
    );
    let header = response
        .headers()
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok());
    assert!(
        header == Some("true") || header == Some("false"),
        "status {status} must carry a deliberate x-should-retry, got {header:?}"
    );
}

/// Every status the router emits is a policy row with a valid retry header:
/// the loop guard's 508, the host guard's 403, client-auth 401, the 404/405
/// fallbacks, the body limit's 413 and an engine-originated 404 (fix brief
/// 05-2; the outermost `stamp_error_policy` layer is what makes the guard
/// and fallback bodies carry the header at all).
#[tokio::test]
async fn every_error_the_router_emits_is_a_policy_row_with_a_retry_header() {
    let anonymous = build_router(anonymous_state());
    let body = messages_body();

    // The self-loop refusal: the guard's inline body never reaches the
    // normalizer, so only the outermost stamp layer can add the header.
    let hop = post("/providers/mock/v1/messages", &[(HOP_HEADER, "1")], &body);
    assert_policy_row_and_header(
        anonymous.clone().oneshot(hop).await.unwrap(),
        StatusCode::LOOP_DETECTED,
    )
    .await;

    // The host guard's foreign-Host refusal: another inline body the
    // normalizer cannot see.
    let foreign_host = post(
        "/providers/mock/v1/messages",
        &[("host", "evil.example")],
        &body,
    );
    assert_policy_row_and_header(
        anonymous.clone().oneshot(foreign_host).await.unwrap(),
        StatusCode::FORBIDDEN,
    )
    .await;

    // The 404 fallback for an unrouted path.
    let unknown_path = post("/nowhere-at-all", &[], &body);
    assert_policy_row_and_header(
        anonymous.clone().oneshot(unknown_path).await.unwrap(),
        StatusCode::NOT_FOUND,
    )
    .await;

    // The 405 fallback: the path is routed, the method is not served.
    let wrong_method = Request::builder()
        .method("GET")
        .uri("/providers/mock/v1/messages")
        .body(Body::empty())
        .unwrap();
    assert_policy_row_and_header(
        anonymous.clone().oneshot(wrong_method).await.unwrap(),
        StatusCode::METHOD_NOT_ALLOWED,
    )
    .await;

    // A body over the family limit: rejected below the normalizer, which
    // rewrites the rejection to the PayloadTooLarge route error.
    let oversized = build_router(tiny_body_limit_state());
    let too_large = post(
        "/providers/mock/v1/messages",
        &[],
        r#"{"model":"claude-sonnet-4-6","messages":"this body exceeds sixteen bytes"}"#,
    );
    assert_policy_row_and_header(
        oversized.oneshot(too_large).await.unwrap(),
        StatusCode::PAYLOAD_TOO_LARGE,
    )
    .await;

    // An engine-originated error: the provider resolves nowhere and the
    // handler renders UnknownProvider as 404 (distinct body from the
    // fallback's "not found").
    let unknown_provider = post("/providers/nosuch/v1/messages", &[], &body);
    let response = anonymous.clone().oneshot(unknown_provider).await.unwrap();
    assert_policy_row_and_header(response, StatusCode::NOT_FOUND).await;

    // Client auth with no token at all: the auth middleware's 401.
    let guarded = build_router(client_keys_state());
    let no_token = post("/providers/mock/v1/messages", &[], &body);
    assert_policy_row_and_header(
        guarded.oneshot(no_token).await.unwrap(),
        StatusCode::UNAUTHORIZED,
    )
    .await;
}

/// The engine-originated 404 must actually come from the engine, not the
/// fallback: its body names the unknown provider. This is the scenario that
/// proves handler-produced statuses join the same policy (a duplicate of the
/// fallback's status alone would prove nothing about the engine path).
#[tokio::test]
async fn the_engine_originated_error_renders_the_unknown_provider() {
    let app = build_router(anonymous_state());
    let unknown_provider = post("/providers/nosuch/v1/messages", &[], &messages_body());
    let response = app.oneshot(unknown_provider).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let message = parsed["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("unknown provider"),
        "the engine's UnknownProvider error must render itself: {parsed}"
    );
}
