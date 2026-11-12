//! Endpoint families (audit conflict 16, plan 03 step 8): the body limit and
//! request timeout are per-family settings, not one flat server-wide number.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_core::{AppConfig, EndpointFamilyConfig, ProviderRegistry, ServerConfig};
    use llm_proxy_server::{AppState, BuildInfo, build_router};
    use llm_proxy_transport::ProxyClient;
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
}
