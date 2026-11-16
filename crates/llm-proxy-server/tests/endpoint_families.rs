//! Endpoint families (audit conflict 16, plan 03 step 8): the body limit and
//! request timeout are per-family settings, not one flat server-wide number.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use axum::routing::post;
    use futures::FutureExt;
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_core::{
        AppConfig, AuthStyle, EndpointFamilyConfig, ProviderAdapterConfig, ProviderConfig,
        ProviderRegistry, ServerConfig,
    };
    use llm_proxy_server::{AppState, BuildInfo, build_router};
    use llm_proxy_transport::ProxyClient;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    fn config_with_chat_limit(chat_max_body: usize) -> AppConfig {
        let mut families = HashMap::new();
        families.insert(
            "chat_completions".to_owned(),
            EndpointFamilyConfig {
                max_body_bytes: chat_max_body,
                request_timeout: Duration::from_secs(60),
            },
        );
        AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".parse().expect("bind"),
                request_timeout: Duration::from_secs(60),
                shutdown_timeout: Duration::from_secs(5),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "families-test".to_owned(),
                rate_limit_rpm: 10_000,
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
            auth: Default::default(),
            storage: None,
            replica: None,
            mcp: None,
        }
    }

    fn state(config: AppConfig) -> AppState {
        let providers = ProviderRegistry::from_providers(vec![]).expect("empty registry is valid");
        AppState::new(
            config,
            providers,
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

    #[tokio::test]
    async fn body_limit_is_per_family_not_global() {
        // The chat family is capped at 1 KiB; the messages family keeps the
        // 32 MiB default. The same oversized body must be rejected on one
        // family and accepted (past the limit layer) on the other.
        let app = build_router(state(config_with_chat_limit(1024)));

        let big_body = vec![b'x'; 8 * 1024];

        let chat = Request::builder()
            .method("POST")
            .uri("/providers/mock/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(big_body.clone()))
            .expect("request");
        let chat_response = app.clone().oneshot(chat).await.expect("chat served");
        assert_eq!(
            chat_response.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "the chat family's 1 KiB limit must reject an 8 KiB body"
        );

        let messages = Request::builder()
            .method("POST")
            .uri("/providers/mock/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(big_body))
            .expect("request");
        let messages_response = app.oneshot(messages).await.expect("messages served");
        assert_ne!(
            messages_response.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "the messages family keeps the 32 MiB default and must not 413"
        );
    }

    #[tokio::test]
    async fn families_default_to_the_server_wide_settings() {
        // No family overrides: both families share the 32 MiB default.
        let app = build_router(state(config_with_chat_limit(usize::MAX)));

        let uri = Request::builder()
            .method("POST")
            .uri("/providers/mock/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(vec![b'x'; 1024]))
            .expect("request");
        let response = app.oneshot(uri).await.expect("served");
        assert_ne!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn family_settings_parse_from_toml() {
        let raw = r#"
            [server]
            bind = "127.0.0.1:8080"
            request_timeout = "300s"
            shutdown_timeout = "30s"
            log_level = "info"
            hot_reload = false
            server_name = "toml"

            [server.endpoint_families.chat_completions]
            max_body_bytes = 2048
            request_timeout = "45s"

            [server.endpoint_families.messages]
            max_body_bytes = 1048576
            request_timeout = "10m"
        "#;
        let config: AppConfig = toml::from_str(raw).expect("parse");
        let chat = &config.server.endpoint_families["chat_completions"];
        assert_eq!(chat.max_body_bytes, 2048);
        assert_eq!(chat.request_timeout, Duration::from_secs(45));
        let messages = &config.server.endpoint_families["messages"];
        assert_eq!(messages.max_body_bytes, 1_048_576);
        assert_eq!(messages.request_timeout, Duration::from_secs(600));

        assert_eq!(
            llm_proxy_core::family_max_body_bytes(&config.server, "chat_completions"),
            2048
        );
        assert_eq!(
            llm_proxy_core::family_max_body_bytes(&config.server, "count_tokens"),
            llm_proxy_core::DEFAULT_MAX_BODY_BYTES
        );
        assert_eq!(
            llm_proxy_core::family_request_timeout(&config.server, "models"),
            Duration::from_secs(300),
            "an unconfigured family inherits the server timeout"
        );
    }

    async fn wait_for_ready(addr: std::net::SocketAddr) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("mock server at {addr} did not become ready within 5 s");
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn spawn_mock_serve(listener: tokio::net::TcpListener, app: Router) {
        tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(axum::serve(listener, app).into_future())
                .catch_unwind()
                .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("mock axum::serve error: {e}"),
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic.downcast_ref::<&'static str>().copied())
                        .unwrap_or("<non-string panic>");
                    eprintln!("mock server task panicked: {msg}");
                    std::process::abort();
                }
            }
        });
    }

    fn openai_chat_success_response() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 12345,
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from OpenAI mock!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .expect("serialize")
    }

    /// A loopback upstream that accepts the request and never answers.
    async fn spawn_hanging_mock() -> String {
        let app = Router::new().route(
            "/{*path}",
            post(|| async move {
                tokio::time::sleep(Duration::from_secs(30)).await;
                StatusCode::OK
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let addr = listener.local_addr().expect("addr");
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{addr}/providers/mock-provider/v1/chat/completions")
    }

    /// A loopback upstream that answers 200 with a valid chat completion
    /// only after `delay`.
    async fn spawn_delayed_success_mock(delay: Duration) -> String {
        let body = openai_chat_success_response();
        let app = Router::new().route(
            "/{*path}",
            post(move || async move {
                tokio::time::sleep(delay).await;
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    body.clone(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let addr = listener.local_addr().expect("addr");
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{addr}/providers/mock-provider/v1/chat/completions")
    }

    /// A state with one OpenAI chat provider pointed at `mock_endpoint`,
    /// a server-wide `request_timeout`, and a chat_completions family
    /// override of `family_timeout` (built the way the chat_completions
    /// suite builds its timeout state).
    fn state_with_family_timeout(
        mock_endpoint: &str,
        server_timeout: Duration,
        family_timeout: Duration,
    ) -> AppState {
        let mut families = HashMap::new();
        families.insert(
            "chat_completions".to_owned(),
            EndpointFamilyConfig {
                max_body_bytes: llm_proxy_core::DEFAULT_MAX_BODY_BYTES,
                request_timeout: family_timeout,
            },
        );
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
                        endpoint: mock_endpoint.to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: llm_proxy_core::ProviderRoutesConfig {
                chat_completions: Some("chat".to_owned()),
                messages: None,
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
        let app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:0".parse().expect("bind"),
                request_timeout: server_timeout,
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "families-timeout-test".to_owned(),
                rate_limit_rpm: 10_000,
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
            auth: Default::default(),
            storage: None,
            replica: None,
            mcp: None,
        };
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

    fn chat_completion_request() -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                })
                .to_string(),
            ))
            .expect("request")
    }

    #[tokio::test]
    async fn a_family_timeout_shorter_than_the_server_default_fires() {
        let mock_url = spawn_hanging_mock().await;
        let state = state_with_family_timeout(
            &mock_url,
            Duration::from_secs(60),
            Duration::from_millis(100),
        );
        let app = build_router(state);

        let started = Instant::now();
        let resp = app
            .oneshot(chat_completion_request())
            .await
            .expect("served");
        let elapsed = started.elapsed();

        assert_eq!(
            resp.status(),
            StatusCode::REQUEST_TIMEOUT,
            "the chat family's 100 ms timeout must fire, not the server-wide 60 s"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let body: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(
            body["error"]["type"], "timeout_error",
            "the 408 must be the normalised timeout error"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the family timeout fires at ~100 ms; the request took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn a_family_timeout_longer_than_the_server_default_does_not_fire() {
        let mock_url = spawn_delayed_success_mock(Duration::from_millis(500)).await;
        let state = state_with_family_timeout(
            &mock_url,
            Duration::from_millis(100),
            Duration::from_secs(60),
        );
        let app = build_router(state);

        let resp = app
            .oneshot(chat_completion_request())
            .await
            .expect("served");

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "the chat family's 60 s timeout governs; the server-wide 100 ms value must not fire"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096)
            .await
            .expect("body");
        let body: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(
            body["choices"][0]["message"]["content"],
            json!("Hello from OpenAI mock!"),
            "the upstream answer after 500 ms must come back complete"
        );
    }
}
