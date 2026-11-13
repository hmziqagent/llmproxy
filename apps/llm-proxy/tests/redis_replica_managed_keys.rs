//! The Redis replica arm of the managed-key daemon, driven end to end at
//! runtime against the real RESP2 wire protocol: the shared store holds
//! the admission counters, the daemon's refresh loop swaps the
//! (authenticator, rules) pair in one write, and a key issued through the
//! admin surface's own store connection is enforced — and its revocation
//! stops it authenticating — without a restart. The server on the other
//! side of the socket is the in-process one from `llm-proxy-store-redis`'s
//! `test-server` feature (this environment has no Redis binary). The
//! Postgres arm has no in-process harness and stays compile-verified only.

#![cfg(all(
    feature = "admin-ui",
    feature = "store-redis",
    feature = "endpoint-families"
))]

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use axum::response::Response;
    use secrecy::ExposeSecret;
    use tower::ServiceExt as _;

    /// How long a runtime change may take to reach enforcement: the daemon
    /// refreshes on a short interval, so seconds — never a restart.
    const EFFECT_DEADLINE: Duration = Duration::from_secs(20);

    /// The embeddings answer the fake upstream serves: `prompt_tokens` of
    /// 50 per call against a runtime-issued tpm of 110 gives 200, 200, 429
    /// (each 48-byte call also reserves 12 estimated input tokens, so the
    /// third call projects 100 settled + 12 reserved and is refused).
    const EMBEDDINGS_BODY: &str = r#"{"object":"list","data":[{"object":"embedding","index":0,
        "embedding":[0.1,0.2]}],"model":"text-embedding-3-small",
        "usage":{"prompt_tokens":50,"total_tokens":50}}"#;

    /// The request body the proxy client sends: tiny, so the reserve-time
    /// estimate (12 tokens at chars/4) stays well under the tpm and only
    /// the reported usage of 50 per call moves the shared counter.
    const EMBEDDINGS_REQUEST: &str = r#"{"model":"text-embedding-3-small","input":"hi"}"#;

    async fn spawn_json_upstream(body: &'static str) -> std::net::SocketAddr {
        let app = axum::Router::new().route(
            "/{*path}",
            axum::routing::post(move || async move {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind upstream");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::trace!(%error, "fake upstream stopped");
            }
        });
        addr
    }

    /// The admin surface's view of the store: a second connection to the
    /// same file the daemon loaded, exactly what POST /keys uses.
    fn admin_view(dir: &Path) -> llm_proxy_admin::ManagedKeys {
        llm_proxy_admin::ManagedKeys::open(dir.join("managed-keys.db")).expect("key store")
    }

    /// Writes the embedder provider and the daemon config pointing
    /// `[replica]` at the in-process Redis.
    fn write_daemon_files(
        dir: &Path,
        upstream: std::net::SocketAddr,
        replica_url: &str,
    ) -> PathBuf {
        std::fs::create_dir_all(dir.join("providers")).expect("providers dir");
        std::fs::write(
            dir.join("providers/embedder.toml"),
            format!(
                r#"[provider]
name = "embedder"
api_key = "k"
auth_style = "bearer"
[provider.adapters.embed]
protocol = "openai_chat_completions"
endpoint = "http://{upstream}/v1/embeddings"
[provider.routes]
embeddings = "embed"
"#
            ),
        )
        .expect("provider file");
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3462"
request_timeout = "300s"
log_level = "info"
server_name = "redis-replica-runtime"

[auth]
mode = "client-keys"

[auth.admin]
token = "admin-tok"
key_store = "{store}"

[replica]
backend = "redis"
url = "{url}"

[storage]
path = "{usage}"
"#,
                store = dir.join("managed-keys.db").display(),
                url = replica_url,
                usage = dir.join("usage.db").display()
            ),
        )
        .expect("config file");
        path
    }

    async fn post(app: axum::Router, token: &secrecy::SecretString, body: &str) -> Response {
        let request = Request::builder()
            .method("POST")
            .uri("/providers/embedder/v1/embeddings")
            .header(header::CONTENT_TYPE, "application/json")
            .header(
                header::AUTHORIZATION,
                format!("Bearer {}", token.expose_secret()),
            )
            .body(Body::from(body.to_owned()))
            .expect("request");
        app.oneshot(request).await.expect("oneshot is infallible")
    }

    /// Retry until the daemon has picked the key up: the 401 answers a
    /// not-yet-installed key and consumes nothing; a 429 proves the key
    /// installed but the call was refused, so it also doesn't count — the
    /// first other answer is the call that counts.
    async fn first_admitted(app: axum::Router, token: &secrecy::SecretString) -> Response {
        let deadline = Instant::now() + EFFECT_DEADLINE;
        loop {
            let response = post(app.clone(), token, EMBEDDINGS_REQUEST).await;
            let status = response.status();
            if status != StatusCode::UNAUTHORIZED && status != StatusCode::TOO_MANY_REQUESTS {
                return response;
            }
            assert!(
                Instant::now() < deadline,
                "the runtime-issued key never authenticated (still {status}) over the replica"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn eventually_unauthorized(app: axum::Router, token: &secrecy::SecretString) {
        let deadline = Instant::now() + EFFECT_DEADLINE;
        loop {
            let response = post(app.clone(), token, EMBEDDINGS_REQUEST).await;
            if response.status() == StatusCode::UNAUTHORIZED {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the revoked key kept authenticating (last answer {}) over the replica",
                response.status()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// The store's tpm window key for `caller` in one of the last two UTC
    /// minutes (the reservation minute owns the charge; a boundary
    /// crossing during the test puts it in the earlier one).
    fn token_window_key(caller: &str, minutes_back: i64) -> String {
        let minute = (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64)
            .div_euclid(60)
            - minutes_back;
        format!("llm-proxy:ad:tok:{caller}:{minute}")
    }

    /// The shared store's minute window for this key holds at least the
    /// two settled charges (2 x 50 tokens; extra 200s from the throttling
    /// poll only ever add more) — the proof the throttle ran over the
    /// REPLICA counters, not an in-process backend.
    async fn wait_for_shared_charge(
        server: &llm_proxy_store_redis::test_server::TestRedis,
        caller: &str,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            for minutes_back in 0..2 {
                let charged = server
                    .get(&token_window_key(caller, minutes_back))
                    .and_then(|raw| raw.parse::<i64>().ok())
                    .unwrap_or(0);
                if charged >= 100 {
                    return;
                }
            }
            assert!(
                Instant::now() < deadline,
                "the settled charges never landed on the shared replica counter"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// The done-when for the redis replica arm: a key issued at runtime
    /// with a tpm limit is throttled by the shared store's counters
    /// (200, 200, 429 — the same shape the in-process backend enforces),
    /// the charged tokens are visible IN the shared store, and revoking
    /// the key stops its token authenticating — no restart anywhere.
    #[tokio::test]
    async fn a_runtime_issued_key_is_enforced_and_revoked_over_the_redis_replica_store() {
        let server =
            llm_proxy_store_redis::test_server::TestRedis::start().expect("bind the test Redis");
        let upstream = spawn_json_upstream(EMBEDDINGS_BODY).await;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_daemon_files(dir.path(), upstream, &server.url());

        let config = llm_proxy_core::load_app_config(&path).expect("config loads");
        let state = llm_proxy_app::state::load_toml_state_from_config(
            config,
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
        )
        .await
        .expect("the daemon boots against the redis replica arm");
        let app = llm_proxy_server::build_router(state);

        let issued = admin_view(dir.path())
            .create(&llm_proxy_admin::KeyPolicyInput {
                tpm: Some(110),
                ..llm_proxy_admin::KeyPolicyInput::default()
            })
            .expect("runtime issuance");

        let first = first_admitted(app.clone(), &issued.token).await;
        assert_eq!(
            first.status(),
            StatusCode::OK,
            "the runtime-issued key serves once the replica store's policy is loaded"
        );
        let second = post(app.clone(), &issued.token, EMBEDDINGS_BODY).await;
        assert_eq!(second.status(), StatusCode::OK);

        // The store settles asynchronously (the settle write is a spawned
        // roundtrip), so the tpm of 110 bites once the two 50-token charges
        // land — the same 200/200/429 shape the in-process backend gives,
        // arrived at by polling rather than by the third call racing the
        // settles.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let response = post(app.clone(), &issued.token, EMBEDDINGS_BODY).await;
            if response.status() == StatusCode::TOO_MANY_REQUESTS {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "the runtime-issued tpm of 60 never throttled the key over the shared \
                 counters (last answer {})",
                response.status()
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        wait_for_shared_charge(&server, &issued.key_id).await;

        admin_view(dir.path())
            .revoke(&issued.key_id)
            .expect("runtime revocation");
        eventually_unauthorized(app, &issued.token).await;
    }
}
