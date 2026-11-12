//! Secret-abuse family: secret material must never be observable. Every test
//! drives the real surfaces (router, admin router, client-auth, ledger,
//! `/usage`) and pairs each negative scan with a positive control.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::time::Duration;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use axum::routing::post;
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_core::{
        AdminAuthConfig, AppConfig, AuthMode, AuthStyle, ProviderAdapterConfig, ProviderConfig,
        ProviderRegistry, ProviderRoutesConfig, ServerConfig, StorageConfig,
    };
    use llm_proxy_server::{AppState, BuildInfo, ClientAuth, admin_router, build_router};
    use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus, SqliteEventBus};
    use llm_proxy_transport::ProxyClient;
    use tower::ServiceExt;

    const LAPTOP_TOKEN: &str = "prx6-laptop-client-token-4f1a9c73";
    const PHONE_TOKEN: &str = "prx6-phone-client-token-b27e03d1";
    const ADMIN_TOKEN: &str = "prx6-admin-token-9d5c48ea";
    const WRONG_TOKEN: &str = "prx6-attacker-guess-00000011";
    const STRAY_TOKEN: &str = "prx6-stray-bearer-under-noauth-77c1";
    const PROVIDER_KEY: &str = "prx6-provider-upstream-key-6b8f2e45";
    const BODY_MARKER: &str = "PRX6-BODY-MARKER-never-persist-me";
    const SECRETS: &[&str] = &[
        LAPTOP_TOKEN,
        PHONE_TOKEN,
        ADMIN_TOKEN,
        WRONG_TOKEN,
        STRAY_TOKEN,
        PROVIDER_KEY,
        BODY_MARKER,
    ];
    const KEY_ID_LAPTOP: &str = "laptop";
    const KEY_ID_PHONE: &str = "phone";

    const ANTHROPIC_SUCCESS: &str = concat!(
        r#"{"id":"msg_prx6","type":"message","role":"assistant","content":"#,
        r#"[{"type":"text","text":"prx6-mock-reply"}],"model":"claude-sonnet-4-6","#,
        r#""stop_reason":"end_turn","stop_sequence":null,"#,
        r#""usage":{"input_tokens":10,"output_tokens":5}}"#
    );

    fn cheap_phc(token: &str, salt: &[u8]) -> String {
        use argon2::{Algorithm, Argon2, CustomizedPasswordHasher, Params, Version};
        let params = Params::new(8, 1, 1, None).expect("valid cheap test params");
        Argon2::new(Algorithm::Argon2id, Version::V0x13, params.clone())
            .hash_password_customized(
                token.as_bytes(),
                salt,
                Some(&argon2::ARGON2ID_IDENT),
                None,
                params,
            )
            .expect("hash token")
            .to_string()
    }

    fn client_auth(pairs: &[(&str, &str)]) -> (ClientAuth, Vec<String>) {
        let mut clients = HashMap::new();
        let mut phcs = Vec::new();
        for (index, (key_id, token)) in pairs.iter().enumerate() {
            let salt = format!("{index:016x}");
            let phc = cheap_phc(token, salt.as_bytes());
            clients.insert(
                (*key_id).to_owned(),
                llm_proxy_core::ClientKeySpec::Verifier(secrecy::SecretString::from(phc.clone())),
            );
            phcs.push(phc);
        }
        let auth = ClientAuth::from_config(&llm_proxy_core::AuthConfig {
            mode: AuthMode::ClientKeys,
            clients,
            admin: None,
            legacy_bare_tokens: false,
        })
        .expect("valid client auth");
        (auth, phcs)
    }

    fn admin_config() -> AdminAuthConfig {
        AdminAuthConfig {
            token: secrecy::SecretString::from(ADMIN_TOKEN),
            key_store: None,
        }
    }

    fn base_state(
        mock_endpoint: &str,
        storage_path: Option<PathBuf>,
        admin: Option<AdminAuthConfig>,
        on_main: bool,
    ) -> AppState {
        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from(PROVIDER_KEY),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut adapters = HashMap::new();
                adapters.insert(
                    "main".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "anthropic_messages".to_owned(),
                        endpoint: mock_endpoint.to_owned(),
                        headers: Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                adapters
            },
            routes: ProviderRoutesConfig {
                messages: Some("main".to_owned()),
                chat_completions: None,
                ..Default::default()
            },
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: HashMap::new(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };

        AppState::new(
            AppConfig {
                server: ServerConfig {
                    bind: "127.0.0.1:3456".parse().expect("bind parses"),
                    request_timeout: Duration::from_secs(300),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "secret-abuse-test".to_owned(),
                    rate_limit_rpm: 100,
                    trust_forwarded_headers: false,
                    cache: Default::default(),
                    idempotency: Default::default(),
                    metrics: Default::default(),
                    log_format: Default::default(),
                    admin_bind: None,
                    admin_on_main_listener: on_main,
                    endpoint_families: Default::default(),
                    upstream_timeouts: Default::default(),
                    bare_v1: Default::default(),
                    preflight: Default::default(),
                    semantic_cache: Default::default(),
                },
                auth: llm_proxy_core::AuthConfig {
                    mode: AuthMode::None,
                    clients: HashMap::new(),
                    admin,
                    legacy_bare_tokens: false,
                },
                storage: storage_path.map(|path| StorageConfig {
                    path,
                    max_age: None,
                    max_rows: None,
                    prune_interval: Duration::from_secs(3600),
                }),
                replica: None,
                mcp: None,
            },
            ProviderRegistry::from_providers(vec![provider]).expect("registry builds"),
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

    fn marked_body(marker: &str) -> String {
        serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": format!("hello {marker}") }],
            "max_tokens": 64,
            "stream": false
        })
        .to_string()
    }

    fn post_messages(token: Option<&str>, body: &str) -> Request<Body> {
        request_with_token("POST", "/providers/mock-provider/v1/messages", token, body)
    }

    fn get_usage(uri: &str, token: Option<&str>) -> Request<Body> {
        request_with_token("GET", uri, token, "")
    }

    fn request_with_token(
        method: &str,
        uri: &str,
        token: Option<&str>,
        body: &str,
    ) -> Request<Body> {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        builder
            .body(Body::from(body.to_owned()))
            .expect("request builds")
    }

    async fn send(app: Router, request: Request<Body>) -> (StatusCode, String, Option<String>) {
        let response = app.oneshot(request).await.expect("oneshot completes");
        let status = response.status();
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        (
            status,
            String::from_utf8_lossy(&bytes).into_owned(),
            request_id,
        )
    }

    fn assert_free_of_secrets(context: &str, text: &str) {
        for secret in SECRETS {
            assert!(
                !text.contains(secret),
                "{context} leaked secret material ({secret}): {text}"
            );
        }
    }

    fn hex_sha256(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn events_for_request(bus: &RecordingBus, request_id: &str) -> Vec<String> {
        bus.snapshot()
            .iter()
            .filter_map(|event| {
                let json = serde_json::to_string(event).expect("event serializes");
                json.contains(request_id).then_some(json)
            })
            .collect()
    }

    fn open_ledger(path: &Path) -> SqliteEventBus {
        SqliteEventBus::open(
            path,
            llm_proxy_storage::RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: Duration::from_secs(3600),
            },
        )
        .expect("ledger opens")
    }

    async fn close_ledger(bus: Arc<SqliteEventBus>) {
        tokio::task::spawn_blocking(move || bus.close())
            .await
            .expect("close joins");
    }

    fn ledger_bytes(db: &Path) -> Vec<u8> {
        let mut all = Vec::new();
        for suffix in ["", "-wal", "-shm"] {
            let sidecar = PathBuf::from(format!("{}{suffix}", db.display()));
            if let Ok(bytes) = std::fs::read(&sidecar) {
                all.extend_from_slice(&bytes);
            }
        }
        all
    }

    async fn wait_ready(addr: std::net::SocketAddr) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "mock upstream at {addr} never became ready"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn spawn_anthropic_mock() -> String {
        let app = Router::new().route(
            "/{*path}",
            post(|| async {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    ANTHROPIC_SUCCESS,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock binds");
        let addr = listener.local_addr().expect("mock address");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("mock upstream serve error: {error}");
            }
        });
        wait_ready(addr).await;
        format!("http://{addr}/providers/mock-provider/v1/messages")
    }

    async fn spawn_echo_mock() -> String {
        let app = Router::new().route(
            "/{*path}",
            post(|body: axum::body::Bytes| async move {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock binds");
        let addr = listener.local_addr().expect("mock address");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("mock upstream serve error: {error}");
            }
        });
        wait_ready(addr).await;
        format!("http://{addr}/providers/mock-provider/v1/messages")
    }

    struct TraceCapture {
        buffer: Arc<Mutex<Vec<u8>>>,
    }

    static CAPTURE_BUFFER: std::sync::OnceLock<Arc<Mutex<Vec<u8>>>> = std::sync::OnceLock::new();
    static CAPTURE_INSTALLED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

    impl TraceCapture {
        fn install() -> Self {
            use tracing_subscriber::layer::SubscriberExt;
            let buffer = CAPTURE_BUFFER.get_or_init(|| Arc::new(Mutex::new(Vec::<u8>::new())));
            CAPTURE_INSTALLED.get_or_init(|| {
                let layer = tracing_subscriber::fmt::layer()
                    .with_writer(BufferWriter(Arc::clone(buffer)))
                    .with_ansi(false);
                tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer))
                    .expect("the global capture subscriber installs exactly once");
            });
            Self {
                buffer: Arc::clone(buffer),
            }
        }

        fn text(&self) -> String {
            let bytes = self.buffer.lock().expect("buffer lock").clone();
            String::from_utf8_lossy(&bytes).into_owned()
        }
    }

    #[derive(Clone)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BufferWriter {
        type Writer = BufferWriterGuard<'a>;

        fn make_writer(&'a self) -> Self::Writer {
            BufferWriterGuard(self.0.lock().expect("buffer lock"))
        }
    }

    struct BufferWriterGuard<'a>(MutexGuard<'a, Vec<u8>>);

    impl std::io::Write for BufferWriterGuard<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn client_and_admin_tokens_never_reach_events_tracing_or_responses() {
        let mock_url = spawn_anthropic_mock().await;
        let (auth, phcs) =
            client_auth(&[(KEY_ID_LAPTOP, LAPTOP_TOKEN), (KEY_ID_PHONE, PHONE_TOKEN)]);
        let bus = Arc::new(RecordingBus::new());
        let state = base_state(&mock_url, None, None, false)
            .with_client_auth(Arc::new(auth))
            .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let app = build_router(state.clone());

        let laptop_bearer = format!("{KEY_ID_LAPTOP}.{LAPTOP_TOKEN}");
        let phone_bearer = format!("{KEY_ID_PHONE}.{PHONE_TOKEN}");
        let traffic: [(Option<&str>, StatusCode); 5] = [
            (Some(laptop_bearer.as_str()), StatusCode::OK),
            (Some(phone_bearer.as_str()), StatusCode::OK),
            (Some(WRONG_TOKEN), StatusCode::UNAUTHORIZED),
            (None, StatusCode::UNAUTHORIZED),
            (Some(ADMIN_TOKEN), StatusCode::UNAUTHORIZED),
        ];

        let capture = TraceCapture::install();
        for (token, expected) in traffic {
            let (status, body, _) =
                send(app.clone(), post_messages(token, &marked_body(BODY_MARKER))).await;
            assert_eq!(
                status, expected,
                "unexpected status for token {token:?}: {body}"
            );
            assert_free_of_secrets(&format!("response body for {token:?}"), &body);
            for phc in &phcs {
                assert!(
                    !body.contains(phc.as_str()),
                    "response body leaked the argon2 verifier: {body}"
                );
            }
        }

        let events = bus.snapshot();
        assert!(
            events.len() >= 6,
            "traffic must produce events (got {})",
            events.len()
        );
        for event in &events {
            let json = serde_json::to_string(event).expect("event serializes");
            assert_free_of_secrets("recorded event", &json);
            for phc in &phcs {
                assert!(
                    !json.contains(phc.as_str()),
                    "event leaked the argon2 verifier: {json}"
                );
            }
        }
        let callers: Vec<&str> = events
            .iter()
            .filter_map(|event| match event {
                ProxyEvent::RequestReceived(received) => Some(received.caller.as_str()),
                _ => None,
            })
            .collect();
        assert!(callers.contains(&KEY_ID_LAPTOP), "callers: {callers:?}");
        assert!(callers.contains(&KEY_ID_PHONE), "callers: {callers:?}");

        let captured = capture.text();
        assert!(
            captured.contains("client authenticated"),
            "control: debug-level request traffic must land in the capture, got: {captured}"
        );
        assert_free_of_secrets("tracing output", &captured);
        for phc in &phcs {
            assert!(
                !captured.contains(phc.as_str()),
                "tracing output leaked the argon2 verifier: {captured}"
            );
        }

        let state_debug = format!("{state:?}");
        assert!(
            !state_debug.contains(LAPTOP_TOKEN) && !state_debug.contains(PHONE_TOKEN),
            "AppState Debug leaked a client token: {state_debug}"
        );
        for phc in &phcs {
            assert!(
                !state_debug.contains(phc.as_str()),
                "AppState Debug leaked a verifier: {state_debug}"
            );
        }
    }

    #[tokio::test]
    async fn secrets_absent_from_raw_ledger_file_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("abuse.db");
        let mock_url = spawn_anthropic_mock().await;
        let (auth, phcs) =
            client_auth(&[(KEY_ID_LAPTOP, LAPTOP_TOKEN), (KEY_ID_PHONE, PHONE_TOKEN)]);
        let bus = Arc::new(open_ledger(&db));
        let state = base_state(&mock_url, Some(db.clone()), Some(admin_config()), true)
            .with_client_auth(Arc::new(auth))
            .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let app = build_router(state);

        let body = marked_body(BODY_MARKER);
        let digest = hex_sha256(body.as_bytes());
        let laptop_bearer = format!("{KEY_ID_LAPTOP}.{LAPTOP_TOKEN}");
        let (status, response, _) = send(
            app.clone(),
            post_messages(Some(laptop_bearer.as_str()), &body),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "valid client request: {response}");
        let (status, _, _) = send(app.clone(), post_messages(Some(WRONG_TOKEN), &body)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _, _) =
            send(app.clone(), get_usage("/usage/requests", Some(ADMIN_TOKEN))).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, _) = send(
            app.clone(),
            get_usage("/usage/requests", Some(LAPTOP_TOKEN)),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "client keys are not admin"
        );

        close_ledger(Arc::clone(&bus)).await;
        let raw = ledger_bytes(&db);
        assert!(!raw.is_empty(), "the ledger file must exist on disk");

        let text = String::from_utf8_lossy(&raw);
        for expected in [KEY_ID_LAPTOP, "mock-provider", digest.as_str()] {
            assert!(
                text.contains(expected),
                "control: persisted value {expected} must be findable in the ledger bytes"
            );
        }
        assert_free_of_secrets("raw ledger file bytes", &text);
        for phc in &phcs {
            assert!(
                !text.contains(phc.as_str()),
                "ledger bytes leaked an argon2 verifier"
            );
        }
    }

    #[tokio::test]
    async fn usage_serves_the_body_digest_but_never_the_body_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("abuse.db");
        let mock_url = spawn_echo_mock().await;
        let (auth, _phcs) = client_auth(&[(KEY_ID_LAPTOP, LAPTOP_TOKEN)]);
        let bus = Arc::new(open_ledger(&db));
        let state = base_state(&mock_url, Some(db.clone()), Some(admin_config()), true)
            .with_client_auth(Arc::new(auth))
            .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let main_app = build_router(state.clone());
        let dedicated = admin_router(&state).expect("admin router for the dedicated mount");

        let body = marked_body(BODY_MARKER);
        let digest = hex_sha256(body.as_bytes());
        let laptop_bearer = format!("{KEY_ID_LAPTOP}.{LAPTOP_TOKEN}");
        let (status, response, _) = send(
            main_app.clone(),
            post_messages(Some(laptop_bearer.as_str()), &body),
        )
        .await;
        assert_ne!(
            status,
            StatusCode::OK,
            "a reflected request-shaped body must not decode as a valid response"
        );
        assert_free_of_secrets("client-facing error response", &response);

        close_ledger(Arc::clone(&bus)).await;
        for (mount, app) in [("dedicated", dedicated), ("main-opt-in", main_app)] {
            for uri in [
                "/usage/requests?limit=1000",
                "/usage/summary?group_by=caller",
                "/usage/keys",
            ] {
                let (status, body, _) = send(app.clone(), get_usage(uri, Some(ADMIN_TOKEN))).await;
                assert_eq!(status, StatusCode::OK, "{mount} {uri}: {body}");
                assert_free_of_secrets(&format!("{mount} {uri} response"), &body);
                assert!(
                    !body.contains(BODY_MARKER),
                    "{mount} {uri} must never carry request-body content: {body}"
                );
            }

            let (_, body, _) = send(
                app.clone(),
                get_usage("/usage/requests?limit=1000", Some(ADMIN_TOKEN)),
            )
            .await;
            let rows = serde_json::from_str::<serde_json::Value>(&body)
                .expect("rows JSON")
                .get("rows")
                .cloned()
                .unwrap_or_default();
            let digested: Vec<&str> = rows
                .as_array()
                .map(|rows| {
                    rows.iter()
                        .filter_map(|row| row["body_hash"].as_str())
                        .collect()
                })
                .unwrap_or_default();
            assert_eq!(
                digested,
                vec![digest.as_str()],
                "{mount}: the request row must carry the sha256 of the actual body, not the body"
            );
            assert!(
                body.contains(KEY_ID_LAPTOP),
                "{mount}: caller attribution must be visible to the admin"
            );
        }
    }

    #[tokio::test]
    async fn admin_surface_refuses_non_admin_credentials_on_both_mounts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("abuse.db");
        let bus = Arc::new(open_ledger(&db));
        bus.emit(&ProxyEvent::RequestReceived(
            llm_proxy_storage::RequestReceived {
                request_id: "prx6-seed".to_owned(),
                timestamp: time::OffsetDateTime::now_utc(),
                caller: "prx6-seed-caller".to_owned(),
                provider: "mock-provider".to_owned(),
                route_kind: "messages".to_owned(),
                client_protocol: "anthropic".to_owned(),
                model: llm_proxy_protocol::core::ModelRef {
                    requested: "claude-3".to_owned(),
                    upstream: None,
                },
                streaming: false,
                body_hash: "seed-digest".to_owned(),
                session_id: None,
            },
        ));
        close_ledger(Arc::clone(&bus)).await;

        let (auth, _phcs) = client_auth(&[(KEY_ID_LAPTOP, LAPTOP_TOKEN)]);
        let state = base_state("http://127.0.0.1:9", Some(db), Some(admin_config()), true)
            .with_client_auth(Arc::new(auth));
        let mounts = [
            ("dedicated", admin_router(&state).expect("admin router")),
            ("main-opt-in", build_router(state)),
        ];

        for (mount, app) in mounts {
            let (missing_status, missing_body, _) =
                send(app.clone(), get_usage("/usage/summary", None)).await;
            assert_eq!(
                missing_status,
                StatusCode::UNAUTHORIZED,
                "{mount}: missing credentials must be 401 (not 403/5xx)"
            );

            for token in [WRONG_TOKEN, LAPTOP_TOKEN] {
                let (status, body, _) =
                    send(app.clone(), get_usage("/usage/summary", Some(token))).await;
                assert_eq!(
                    status,
                    StatusCode::UNAUTHORIZED,
                    "{mount}: {token:?} must not authenticate to the admin surface"
                );
                assert_free_of_secrets(&format!("{mount} rejection for {token:?}"), &body);
                assert_eq!(
                    body, missing_body,
                    "{mount}: every rejection carries the identical uniform body"
                );
            }

            let (status, _, _) =
                send(app.clone(), get_usage("/usage/summary", Some(ADMIN_TOKEN))).await;
            assert_eq!(status, StatusCode::OK, "{mount}: the admin token serves");
        }

        let (proxy_auth, _phcs) = client_auth(&[(KEY_ID_LAPTOP, LAPTOP_TOKEN)]);
        let proxy_state = base_state("http://127.0.0.1:9", None, None, false)
            .with_client_auth(Arc::new(proxy_auth));
        let (status, body, _) = send(
            build_router(proxy_state),
            post_messages(Some(ADMIN_TOKEN), &marked_body(BODY_MARKER)),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "the admin token must not authenticate proxy routes"
        );
        assert_free_of_secrets("proxy-surface rejection of the admin token", &body);
    }

    #[tokio::test]
    async fn one_callers_key_id_is_never_observable_by_another_caller() {
        const ALPHA_ID: &str = "prx6-iso-alpha";
        const BETA_ID: &str = "prx6-iso-beta";
        const ALPHA_TOKEN: &str = "prx6-iso-alpha-token-3d5e";
        const BETA_TOKEN: &str = "prx6-iso-beta-token-8f1b";

        let mock_url = spawn_anthropic_mock().await;
        let (auth, _phcs) = client_auth(&[(ALPHA_ID, ALPHA_TOKEN), (BETA_ID, BETA_TOKEN)]);
        let bus = Arc::new(RecordingBus::new());
        let state = base_state(&mock_url, None, None, false)
            .with_client_auth(Arc::new(auth))
            .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let app = build_router(state);

        let alpha_bearer = format!("{ALPHA_ID}.{ALPHA_TOKEN}");
        let beta_bearer = format!("{BETA_ID}.{BETA_TOKEN}");
        let (status, _, _) = send(
            app.clone(),
            post_messages(Some(alpha_bearer.as_str()), &marked_body("alpha traffic")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let capture = TraceCapture::install();
        let (status, body, request_id) = send(
            app.clone(),
            post_messages(Some(beta_bearer.as_str()), &marked_body("beta traffic")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            !body.contains(ALPHA_ID),
            "beta's response names alpha's key id: {body}"
        );

        let request_id = request_id.expect("x-request-id header present");
        let captured = capture.text();
        let correlated: Vec<&str> = captured
            .lines()
            .filter(|line| line.contains(&request_id) || line.contains(BETA_ID))
            .collect();
        assert!(
            !correlated.is_empty(),
            "control: beta's traffic must produce log lines"
        );
        for line in &correlated {
            assert!(
                !line.contains(ALPHA_ID),
                "a log line for beta's request names alpha's key id: {line}"
            );
        }

        let beta_event_json = events_for_request(&bus, &request_id);
        assert!(beta_event_json.len() >= 2, "beta's request emitted events");
        for json in &beta_event_json {
            assert!(
                !json.contains(ALPHA_ID),
                "an event for beta's request names alpha's key id: {json}"
            );
        }
        let beta_callers: Vec<String> = bus
            .snapshot()
            .into_iter()
            .filter_map(|event| match event {
                ProxyEvent::RequestReceived(received) if received.request_id == request_id => {
                    Some(received.caller)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            beta_callers,
            vec![BETA_ID.to_owned()],
            "self-attribution intact"
        );
    }

    #[tokio::test]
    async fn anonymous_attribution_survives_the_loopback_exemption() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("abuse.db");
        let mock_url = spawn_anthropic_mock().await;
        let bus = Arc::new(open_ledger(&db));
        let state = base_state(&mock_url, Some(db.clone()), Some(admin_config()), false)
            .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let app = build_router(state.clone());

        let body = marked_body(BODY_MARKER);
        let (status, response, _) =
            send(app.clone(), post_messages(Some(STRAY_TOKEN), &body)).await;
        assert_eq!(status, StatusCode::OK);
        assert_free_of_secrets("response under the loopback exemption", &response);

        close_ledger(Arc::clone(&bus)).await;
        let admin = admin_router(&state).expect("admin router");
        let (status, keys, _) =
            send(admin.clone(), get_usage("/usage/keys", Some(ADMIN_TOKEN))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            keys.contains("anonymous"),
            "attribution survives as the anonymous caller: {keys}"
        );
        assert_free_of_secrets("/usage/keys under the loopback exemption", &keys);
        assert!(
            !keys.contains(STRAY_TOKEN),
            "the stray bearer must not become the caller: {keys}"
        );

        let (status, rows, _) = send(
            admin,
            get_usage("/usage/requests?limit=1000", Some(ADMIN_TOKEN)),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            rows.contains("anonymous") && rows.contains(&hex_sha256(body.as_bytes())),
            "the anonymous row carries the digest: {rows}"
        );
        assert_free_of_secrets("/usage/requests under the loopback exemption", &rows);

        let bytes = ledger_bytes(&db);
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("anonymous"),
            "control: the anonymous caller is persisted"
        );
        assert_free_of_secrets("ledger bytes under the loopback exemption", &text);
    }
}
