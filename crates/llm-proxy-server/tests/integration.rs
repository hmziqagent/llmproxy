//! Integration tests for ops endpoints, the `/v1/messages` proxy route, and
//! TOML-mode integration scenarios.

#[cfg(test)]
mod tests {

    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::{
        body::Body,
        extract::ConnectInfo,
        http::{Request, StatusCode, header},
    };
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_core::{
        AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
        ProviderRoutesConfig, ServerConfig,
    };
    use llm_proxy_server::{AppState, BuildInfo, build_router};
    use llm_proxy_transport::ProxyClient;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    fn state() -> AppState {
        let app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: Duration::from_secs(300),
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "test-proxy".to_owned(),
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
        };
        let registry = ProviderRegistry::from_providers(vec![]).expect("empty registry");
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
    }

    fn state_with_provider() -> AppState {
        state_with_provider_at_rate(100, false)
    }

    /// The same fixture with the test's own rate-limit posture: limiter state
    /// lives in each `AppState`, so every rate-limit test builds its own.
    fn state_with_provider_at_rate(rate_limit_rpm: u32, trust_forwarded_headers: bool) -> AppState {
        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "messages".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "anthropic_messages".to_owned(),
                        endpoint: "https://127.0.0.1:0/v1/messages".to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: ProviderRoutesConfig {
                messages: Some("messages".to_owned()),
                chat_completions: Some("messages".to_owned()),
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
            AppConfig {
                server: ServerConfig {
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: Duration::from_secs(300),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test-proxy".to_owned(),
                    rate_limit_rpm,
                    trust_forwarded_headers,
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
            ProviderAdapterRegistry::builtin(),
            ProxyClient::try_new().expect("client"),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    #[tokio::test]
    async fn health_returns_ok_with_body() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["status"], "ok");
        assert!(body.get("model_counts").is_none());
    }

    #[tokio::test]
    async fn ready_returns_ready() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/ready")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["status"], "ready");
    }

    #[tokio::test]
    async fn version_returns_build_info() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/version")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["name"].is_string());
        assert_eq!(body["version"], "0.0.0");
        assert_eq!(body["target"], "test");
        assert_eq!(body["git_sha"], "test");
    }

    #[tokio::test]
    async fn messages_without_auth_header_reaches_upstream() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 16
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        assert_ne!(status, StatusCode::NOT_FOUND, "route must be registered");
        let valid_statuses = [StatusCode::BAD_GATEWAY, StatusCode::INTERNAL_SERVER_ERROR];
        assert!(
            valid_statuses.contains(&status),
            "expected downstream error for unreachable upstream without auth header, got {status}"
        );
    }

    #[tokio::test]
    async fn unknown_route_returns_not_found() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/v1/nonexistent_route")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn messages_valid_json_parses_and_validates() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let _resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert_ne!(status, StatusCode::NOT_FOUND, "route must be registered");

        let valid_statuses = [
            StatusCode::BAD_REQUEST,
            StatusCode::BAD_GATEWAY,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::UNAUTHORIZED,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::CONFLICT,
            StatusCode::OK,
        ];
        assert!(
            valid_statuses.contains(&status),
            "valid Anthropic JSON should produce a valid status, got {status}"
        );
    }

    #[tokio::test]
    async fn messages_invalid_json_returns_bad_request() {
        let app = build_router(state_with_provider());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from("this is not json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid JSON"),
            "error message should mention invalid JSON"
        );
    }

    #[tokio::test]
    async fn messages_valid_json_missing_fields_returns_validation_error() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["error"]["type"], "invalid_request_error");
        assert!(
            resp_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("missing field `messages`"),
            "should report missing messages field"
        );
    }

    #[tokio::test]
    async fn count_tokens_returns_estimate() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello world" }],
            "max_tokens": 1024
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(body["input_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn not_found_returns_json_error_envelope() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/v1/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn messages_wrong_field_type_returns_bad_request() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": "big"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn messages_unknown_fields_no_longer_fail_decode() {
        // Plan 02: the same-format request path tolerates unknown fields so
        // they can be forwarded verbatim; decode no longer answers 400. The
        // unreachable upstream proves the request travelled past decode.
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64,
            "future_field": "not supported"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_GATEWAY,
            "unknown fields must pass decode and fail at the (unreachable) upstream"
        );
    }

    #[tokio::test]
    async fn cross_format_unknown_fields_are_rejected_by_name() {
        // An Anthropic-dialect client against a Chat Completions upstream:
        // the leftover `future_field` must be rejected with a 400 naming it.
        let state = state_with_cross_format_provider();
        let app = build_router(state);
        let body = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64,
            "future_field": "not supported"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        let message = resp_body["error"]["message"].as_str().unwrap_or("");
        assert!(
            message.contains("future_field"),
            "the error must name the rejected field: {resp_body}"
        );
    }

    #[tokio::test]
    async fn messages_invalid_utf8_returns_bad_request() {
        let app = build_router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(vec![0x80, 0x81, 0x82]))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.status().is_client_error(),
            "expected 4xx for invalid UTF-8, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn messages_empty_body_returns_bad_request() {
        let app = build_router(state_with_provider());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn messages_oversized_body_returns_payload_too_large() {
        let app = build_router(state());
        let oversized_body = "X".repeat(33 * 1024 * 1024);
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(oversized_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );
        assert!(
            resp.headers().get("x-request-id").is_some(),
            "413 must carry x-request-id after normalisation"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("too large")
        );
    }

    #[tokio::test]
    async fn toml_health_returns_ok() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["service"], "test-proxy");
    }

    #[tokio::test]
    async fn toml_version_returns_app_config_server_name() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/version")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["name"], "test-proxy");
        assert_eq!(body["version"], "0.0.0");
        assert_eq!(body["target"], "test");
        assert_eq!(body["git_sha"], "test");
    }

    #[tokio::test]
    async fn toml_ready_returns_ready() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/ready")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["status"], "ready");
    }

    #[tokio::test]
    async fn toml_messages_passes_through_to_upstream() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        assert_ne!(
            status,
            StatusCode::NOT_FOUND,
            "provider is registered, route must resolve"
        );
        let valid_statuses = [StatusCode::BAD_GATEWAY, StatusCode::INTERNAL_SERVER_ERROR];
        assert!(
            valid_statuses.contains(&status),
            "expected downstream error for unreachable upstream, got {status}"
        );
    }

    #[tokio::test]
    async fn toml_count_tokens_returns_estimate() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello world" }],
            "max_tokens": 1024
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(resp_body["input_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn get_on_messages_route_returns_405() {
        let app = build_router(state_with_provider());
        let req = Request::builder()
            .method("GET")
            .uri("/providers/mock-provider/v1/messages")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body: Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), 4096).await.unwrap())
                .unwrap();
        assert_eq!(
            body["type"], "error",
            "405 must be an Anthropic-shaped envelope"
        );
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "method not allowed");
    }

    #[tokio::test]
    async fn get_on_chat_completions_route_returns_405() {
        let app = build_router(state_with_provider());
        let req = Request::builder()
            .method("GET")
            .uri("/providers/mock-provider/v1/chat/completions")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
        let body: Value =
            serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), 4096).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "method not allowed");
        assert!(body["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn post_on_health_route_returns_405() {
        let app = build_router(state());
        let req = Request::builder()
            .method("POST")
            .uri("/health")
            .body(Body::from("{}"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn invalid_provider_name_on_messages_returns_400() {
        let app = build_router(state());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/..%2F..%2Fetc/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.status() == StatusCode::BAD_REQUEST || resp.status() == StatusCode::NOT_FOUND,
            "expected 400 or 404 for path traversal provider name, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn count_tokens_invalid_json_returns_400() {
        let app = build_router(state_with_provider());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from("{ not valid json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn count_tokens_empty_body_returns_400() {
        let app = build_router(state_with_provider());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(""))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn count_tokens_unknown_provider_returns_404() {
        let app = build_router(state());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/nonexistent/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn health_never_exposes_metrics_even_with_opt_in_query() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/health?metrics=true")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            body.get("metrics").is_none(),
            "/health must not include 'metrics' even with ?metrics=true \
         (unauthenticated endpoint, LOW-13); got: {body}"
        );
        assert_eq!(body["status"], "ok");
        assert!(body["service"].is_string());
    }

    #[tokio::test]
    async fn health_omits_metrics_by_default() {
        let app = build_router(state());
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            body.get("metrics").is_none(),
            "bare /health must not include 'metrics' (unauthenticated endpoint); got: {body}"
        );
        assert_eq!(body["status"], "ok");
        assert!(body["service"].is_string());
    }

    #[tokio::test]
    async fn count_tokens_response_has_correct_shape() {
        let app = build_router(state_with_provider());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello world" }],
            "max_tokens": 1024
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .expect("content-type header");
        assert!(ct.to_str().unwrap().contains("application/json"));
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(resp_body["input_tokens"].is_number());
        assert!(resp_body["input_tokens"].as_u64().unwrap() > 0);
        assert!(
            resp_body.as_object().unwrap().len() == 1,
            "count_tokens response should only have input_tokens field"
        );
    }

    fn state_with_origins(origins: Vec<String>) -> AppState {
        let app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: Duration::from_secs(300),
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: Some(origins),
                server_name: "test-proxy".to_owned(),
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
        };
        let registry = ProviderRegistry::from_providers(vec![]).expect("empty registry");
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
    }

    #[tokio::test]
    async fn cors_preflight_echoes_configured_origin() {
        let app = build_router(state_with_origins(vec![
            "https://app.example.com".to_owned(),
        ]));
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/health")
            .header("origin", "https://app.example.com")
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "content-type")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .expect("allow-origin echoed on preflight"),
            "https://app.example.com"
        );
        let allow_methods = resp
            .headers()
            .get("access-control-allow-methods")
            .expect("allow-methods present")
            .to_str()
            .unwrap();
        assert!(allow_methods.contains("POST"));
        assert!(allow_methods.contains("GET"));
        assert!(!allow_methods.contains('*'));
    }

    #[tokio::test]
    async fn cors_absent_when_origins_unset() {
        let app = build_router(state());
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/health")
            .header("origin", "https://attacker.example")
            .header("access-control-request-method", "POST")
            .header("access-control-request-headers", "content-type")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "no allow-origin should be emitted when CORS is not configured"
        );
    }

    use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};

    fn state_with_cross_format_provider() -> AppState {
        // Anthropic-dialect client route (`/v1/messages`) resolving to a Chat
        // Completions upstream: every request crosses formats.
        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "chat".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "openai_chat_completions".to_owned(),
                        endpoint: "https://127.0.0.1:0/v1/chat/completions".to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: ProviderRoutesConfig {
                messages: Some("chat".to_owned()),
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
            AppConfig {
                server: ServerConfig {
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: Duration::from_secs(300),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test-proxy".to_owned(),
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
            llm_proxy_server::BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    fn state_with_provider_and_bus() -> (AppState, Arc<RecordingBus>) {
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_provider().with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        (state, bus)
    }

    fn count_events(bus: &RecordingBus, kind: &str) -> usize {
        bus.snapshot()
            .iter()
            .filter(|e| {
                matches!(
                    (e, kind),
                    (ProxyEvent::RequestReceived(_), "request_received")
                        | (ProxyEvent::ResponseCompleted(_), "response_completed")
                        | (ProxyEvent::ResponseFailed(_), "response_failed")
                )
            })
            .count()
    }

    #[tokio::test]
    async fn malformed_json_emits_one_response_failed_with_400() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from("not json at all"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        assert_eq!(
            count_events(&bus, "response_failed"),
            1,
            "malformed JSON must emit exactly one ResponseFailed"
        );
        let failed = bus
            .snapshot()
            .into_iter()
            .filter_map(|e| match e {
                ProxyEvent::ResponseFailed(f) => Some(f),
                _ => None,
            })
            .next()
            .expect("one ResponseFailed");
        assert_eq!(failed.http_status, 400);
    }

    #[tokio::test]
    async fn unknown_provider_emits_exactly_one_response_failed() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 16
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/does-not-exist/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        assert_eq!(
            count_events(&bus, "response_failed"),
            1,
            "unknown provider must emit exactly one ResponseFailed (no double-emit)"
        );
    }

    #[tokio::test]
    async fn identical_requests_each_run_the_pipeline_no_409() {
        // The dedup window was retired (plan 06 step 1): with every opt-in
        // off, the second identical request runs the pipeline again — the
        // upstream (unreachable here, so 502) is attempted once per request
        // and each emits its own failure row. Nothing answers 409.
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "dedup probe" }],
            "max_tokens": 16
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/providers/mock-provider/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let resp1 = app.clone().oneshot(make_req()).await.unwrap();
        let resp2 = app.oneshot(make_req()).await.unwrap();
        for (label, resp) in [("first", &resp1), ("second", &resp2)] {
            assert_ne!(
                resp.status(),
                StatusCode::CONFLICT,
                "the retired dedup window must not 409 the {label} identical request"
            );
        }
        assert_eq!(resp1.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(resp2.status(), StatusCode::BAD_GATEWAY);

        assert_eq!(
            count_events(&bus, "response_failed"),
            2,
            "each of the two identical requests emits exactly one ResponseFailed"
        );
        assert!(
            bus.snapshot().iter().all(|event| !matches!(
                event,
                ProxyEvent::ResponseFailed(f) if f.http_status == 409
            )),
            "no 409 row may be written with every opt-in off"
        );
    }

    #[tokio::test]
    async fn token_count_success_emits_request_received_and_completed() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello world" }],
            "max_tokens": 1024
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        assert_eq!(
            count_events(&bus, "request_received"),
            1,
            "token_count success must emit one RequestReceived"
        );
        assert_eq!(
            count_events(&bus, "response_completed"),
            1,
            "token_count success must emit one ResponseCompleted"
        );
        assert_eq!(
            count_events(&bus, "response_failed"),
            0,
            "token_count success must not emit any ResponseFailed"
        );
    }

    #[tokio::test]
    async fn token_count_failure_emits_one_response_failed() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from("{ broken json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        assert_eq!(
            count_events(&bus, "response_failed"),
            1,
            "token_count failure must emit exactly one ResponseFailed"
        );
    }

    #[tokio::test]
    async fn models_unknown_provider_emits_one_response_failed() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/providers/does-not-exist/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        assert_eq!(
            count_events(&bus, "response_failed"),
            1,
            "/models unknown provider must emit exactly one ResponseFailed"
        );
    }

    fn client_auth_for(key_id: &str, token: &str) -> llm_proxy_server::ClientAuth {
        use argon2::{Algorithm, Argon2, CustomizedPasswordHasher, Params, Version};
        let params = Params::new(8, 1, 1, None).expect("valid test params");
        let hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params.clone());
        let phc = hasher
            .hash_password_customized(
                token.as_bytes(),
                b"0123456789abcdef",
                Some(&argon2::ARGON2ID_IDENT),
                None,
                params,
            )
            .expect("hash token")
            .to_string();
        let mut clients = HashMap::new();
        clients.insert(
            key_id.to_owned(),
            llm_proxy_core::ClientKeySpec::Verifier(secrecy::SecretString::from(phc)),
        );
        llm_proxy_server::ClientAuth::from_config(&llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
            legacy_bare_tokens: false,
        })
        .expect("valid client auth")
    }

    async fn models_error_body(
        app: axum::Router,
        method: &'static str,
        uri: &'static str,
        bearer: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(token) = bearer {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let resp = app
            .oneshot(builder.body(Body::empty()).expect("request"))
            .await
            .expect("response");
        let status = resp.status();
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .expect("body"),
        )
        .expect("json body");
        (status, body)
    }

    #[tokio::test]
    async fn models_layer_and_handler_errors_share_the_openai_envelope() {
        let state = state_with_provider().with_client_auth(std::sync::Arc::new(client_auth_for(
            "laptop",
            "sekrit-token",
        )));
        let (auth_status, auth_body) = models_error_body(
            build_router(state.clone()),
            "GET",
            "/providers/mock-provider/v1/models",
            None,
        )
        .await;
        let (handler_status, handler_body) = models_error_body(
            build_router(state.clone()),
            "GET",
            "/providers/nope/v1/models",
            Some("laptop.sekrit-token"),
        )
        .await;
        let (method_status, method_body) = models_error_body(
            build_router(state),
            "POST",
            "/providers/mock-provider/v1/models",
            Some("laptop.sekrit-token"),
        )
        .await;

        assert_eq!(auth_status, StatusCode::UNAUTHORIZED);
        assert_eq!(handler_status, StatusCode::NOT_FOUND);
        assert!(
            method_status == StatusCode::METHOD_NOT_ALLOWED
                || method_status == StatusCode::NOT_FOUND,
            "wrong-method models request must fall through to the router's 405/404, got {method_status}"
        );

        let bodies = [
            ("layer 401", auth_status, &auth_body),
            ("handler 404", handler_status, &handler_body),
            ("method 405", method_status, &method_body),
        ];
        let mut top_level_keys: Option<Vec<String>> = None;
        for (label, status, body) in &bodies {
            let object = body.as_object().unwrap_or_else(|| {
                panic!("{label} ({status}) must answer a JSON object, got {body}")
            });
            let mut keys: Vec<String> = object.keys().cloned().collect();
            keys.sort();
            assert!(
                body["error"]["message"].is_string(),
                "{label} must carry an OpenAI error.message string, got {body}"
            );
            assert!(
                body.get("type").is_none(),
                "{label} must not carry the Anthropic top-level type key, got {body}"
            );
            match &top_level_keys {
                Some(previous) => assert_eq!(
                    &keys, previous,
                    "every /v1/models failure must share one envelope's top-level keys"
                ),
                None => top_level_keys = Some(keys),
            }
        }
    }

    #[tokio::test]
    async fn upstream_failure_inside_pipeline_emits_one_response_failed() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "pipeline failure probe" }],
            "max_tokens": 16
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let valid_statuses = [StatusCode::BAD_GATEWAY, StatusCode::INTERNAL_SERVER_ERROR];
        assert!(
            valid_statuses.contains(&status),
            "expected a pipeline-level failure (502/500), got {status}"
        );

        assert_eq!(
            count_events(&bus, "response_failed"),
            1,
            "an inside-pipeline failure must emit exactly ONE ResponseFailed (no double-emit)"
        );
    }

    #[tokio::test]
    async fn upstream_failure_inside_pipeline_chat_emits_one_response_failed() {
        let (state, bus) = state_with_provider_and_bus();
        let app = build_router(state);
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "pipeline failure probe" }],
            "max_tokens": 16
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let valid_statuses = [StatusCode::BAD_GATEWAY, StatusCode::INTERNAL_SERVER_ERROR];
        assert!(
            valid_statuses.contains(&status),
            "expected a pipeline-level failure (502/500), got {status}"
        );

        assert_eq!(
            count_events(&bus, "response_failed"),
            1,
            "an inside-pipeline failure must emit exactly ONE ResponseFailed (no double-emit)"
        );
    }

    fn count_tokens_from(addr: SocketAddr, bearer: Option<&str>) -> Request<Body> {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        });
        let mut builder = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json");
        if let Some(token) = bearer {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let mut request = builder
            .body(Body::from(body.to_string()))
            .expect("valid request");
        request.extensions_mut().insert(ConnectInfo(addr));
        request
    }

    #[tokio::test]
    async fn unauthenticated_direct_loopback_is_not_rate_limited() {
        let app = build_router(state_with_provider_at_rate(1, false));
        let addr: SocketAddr = "127.0.0.1:1234".parse().expect("valid loopback address");
        for count in 1..=5 {
            let resp = app
                .clone()
                .oneshot(count_tokens_from(addr, None))
                .await
                .expect("response");
            assert_ne!(
                resp.status(),
                StatusCode::TOO_MANY_REQUESTS,
                "direct unauthenticated loopback request {count} must bypass the rpm=1 limiter"
            );
        }
    }

    #[tokio::test]
    async fn non_loopback_clients_are_rate_limited() {
        let app = build_router(state_with_provider_at_rate(1, false));
        let addr: SocketAddr = "10.0.0.5:1234".parse().expect("valid address");
        let first = app
            .clone()
            .oneshot(count_tokens_from(addr, None))
            .await
            .expect("response");
        assert_ne!(
            first.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the first rpm=1 request must pass"
        );
        let second = app
            .oneshot(count_tokens_from(addr, None))
            .await
            .expect("response");
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "a non-loopback caller must be limited at rpm=1"
        );
        assert!(
            second.headers().contains_key(header::RETRY_AFTER),
            "a 429 must carry a retry-after header"
        );
    }

    #[tokio::test]
    async fn authenticated_loopback_caller_is_rate_limited() {
        let state = state_with_provider_at_rate(1, false)
            .with_client_auth(Arc::new(client_auth_for("laptop", "sekrit-token")));
        let app = build_router(state);
        let addr: SocketAddr = "127.0.0.1:1234".parse().expect("valid loopback address");
        let bearer = Some("laptop.sekrit-token");
        let first = app
            .clone()
            .oneshot(count_tokens_from(addr, bearer))
            .await
            .expect("response");
        assert_ne!(
            first.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the first rpm=1 request must pass"
        );
        let second = app
            .oneshot(count_tokens_from(addr, bearer))
            .await
            .expect("response");
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "an authenticated loopback caller must not inherit the direct-loopback exemption"
        );
    }

    #[tokio::test]
    async fn loopback_exemption_is_off_when_forwarded_headers_are_trusted() {
        let app = build_router(state_with_provider_at_rate(1, true));
        let addr: SocketAddr = "127.0.0.1:1234".parse().expect("valid loopback address");
        let first = app
            .clone()
            .oneshot(count_tokens_from(addr, None))
            .await
            .expect("response");
        assert_ne!(
            first.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the first rpm=1 request must pass"
        );
        let second = app
            .oneshot(count_tokens_from(addr, None))
            .await
            .expect("response");
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "trusting forwarded headers must turn the direct-loopback exemption off"
        );
    }
}
