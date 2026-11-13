//! Router-level tests for the admin `/usage` surface on both mounts: the
//! dedicated admin listener router and the opt-in main-listener mount.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;
    use axum::http::StatusCode;
    use llm_proxy_core::{AppConfig, ServerConfig, StorageConfig};
    use llm_proxy_server::{AppState, BuildInfo, admin_router, build_router};
    use llm_proxy_storage::{EventBus, SqliteEventBus};
    use tower::ServiceExt;

    const ADMIN_TOKEN: &str = "integration-admin-token";

    async fn seeded_ledger_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let path = dir.path().join("usage.db");
        let bus = SqliteEventBus::open(
            &path,
            llm_proxy_storage::RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: Duration::from_secs(3600),
            },
        )
        .expect("open ledger");
        bus.emit(&llm_proxy_storage::ProxyEvent::RequestReceived(
            llm_proxy_storage::RequestReceived {
                request_id: "srv-1".to_owned(),
                timestamp: time::OffsetDateTime::now_utc(),
                caller: "laptop".to_owned(),
                provider: "prov-a".to_owned(),
                route_kind: "messages".to_owned(),
                client_protocol: "anthropic".to_owned(),
                model: llm_proxy_protocol::core::ModelRef {
                    requested: "claude-3".to_owned(),
                    upstream: None,
                },
                streaming: false,
                body_hash: "digest".to_owned(),
                session_id: None,
            },
        ));
        let closer = std::sync::Arc::new(bus);
        tokio::task::spawn_blocking(move || closer.close())
            .await
            .expect("close joined");
        path
    }

    fn make_config(
        db_path: std::path::PathBuf,
        admin: Option<llm_proxy_core::AdminAuthConfig>,
        on_main_listener: bool,
    ) -> AppConfig {
        AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: Duration::from_secs(300),
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "admin-usage-test".to_owned(),
                rate_limit_rpm: 100,
                trust_forwarded_headers: false,
                cache: Default::default(),
                idempotency: Default::default(),
                metrics: Default::default(),
                log_format: Default::default(),
                admin_bind: None,
                admin_on_main_listener: on_main_listener,
                endpoint_families: Default::default(),
                upstream_timeouts: Default::default(),
                bare_v1: Default::default(),
                preflight: Default::default(),
                semantic_cache: Default::default(),
            },
            auth: llm_proxy_core::AuthConfig {
                mode: llm_proxy_core::AuthMode::None,
                clients: std::collections::HashMap::new(),
                admin,
                legacy_bare_tokens: false,
            },
            storage: Some(StorageConfig {
                path: db_path,
                max_age: None,
                max_rows: None,
                prune_interval: Duration::from_secs(3600),
            }),
            replica: None,
            mcp: None,
        }
    }

    fn make_state(config: AppConfig) -> AppState {
        AppState::new(
            config,
            llm_proxy_core::ProviderRegistry::from_providers(vec![]).expect("empty registry"),
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            llm_proxy_transport::ProxyClient::try_new().expect("client"),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
        .expect("app state")
    }

    async fn get(app: axum::Router, uri: &str, token: Option<&str>) -> (StatusCode, String) {
        let mut builder = Request::builder().uri(uri);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let response = app
            .oneshot(builder.body(Body::empty()).expect("request builds"))
            .await
            .expect("oneshot");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn main_listener_has_no_usage_surface_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = seeded_ledger_path(&dir).await;
        let state = make_state(make_config(
            path,
            Some(llm_proxy_core::AdminAuthConfig {
                token: secrecy::SecretString::from(ADMIN_TOKEN),
                key_store: None,
            }),
            false,
        ));
        let (status, _) = get(build_router(state), "/usage/summary", Some(ADMIN_TOKEN)).await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "without the opt-in, the main listener must not serve /usage even to the admin token"
        );
    }

    #[cfg(feature = "admin")]
    #[tokio::test]
    async fn main_listener_optin_serves_usage_behind_the_admin_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = seeded_ledger_path(&dir).await;
        let state = make_state(make_config(
            path,
            Some(llm_proxy_core::AdminAuthConfig {
                token: secrecy::SecretString::from(ADMIN_TOKEN),
                key_store: None,
            }),
            true,
        ));

        let (status, body) = get(build_router(state.clone()), "/usage/summary", None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(
            !body.contains(ADMIN_TOKEN),
            "401 must not echo the configured token"
        );

        let (status, body) = get(
            build_router(state.clone()),
            "/usage/summary",
            Some("wrong-token"),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(
            !body.contains("wrong-token") && !body.contains(ADMIN_TOKEN),
            "401 must not echo any token: {body}"
        );

        let (status, body) = get(build_router(state), "/usage/summary", Some(ADMIN_TOKEN)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.contains("prov-a"),
            "the seeded provider group is served: {body}"
        );
    }

    #[cfg(feature = "admin")]
    #[tokio::test]
    async fn dedicated_admin_router_serves_usage_behind_the_admin_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = seeded_ledger_path(&dir).await;
        let state = make_state(make_config(
            path,
            Some(llm_proxy_core::AdminAuthConfig {
                token: secrecy::SecretString::from(ADMIN_TOKEN),
                key_store: None,
            }),
            false,
        ));

        let admin = admin_router(&state).expect("admin router with [auth.admin]");
        let (status, _) = get(admin.clone(), "/usage/keys", None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);

        let (status, body) = get(admin, "/usage/keys", Some(ADMIN_TOKEN)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.contains("laptop"),
            "the seeded caller is listed: {body}"
        );
    }

    #[tokio::test]
    async fn admin_router_is_none_without_admin_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = seeded_ledger_path(&dir).await;
        let state = make_state(make_config(path, None, false));
        assert!(
            admin_router(&state).is_none(),
            "no [auth.admin] means no admin surface anywhere"
        );
    }
}
