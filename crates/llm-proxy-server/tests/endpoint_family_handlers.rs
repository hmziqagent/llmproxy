//! Non-chat endpoint families (plan 09 step 7, audit conflict 16) end to
//! end: each family handler resolves its route through the provider
//! registry, admits through the same step the chat pipeline uses,
//! forwards with the provider's credential and the surgical edit (model
//! aliasing), and writes usage rows. Audio and file bodies pass through
//! verbatim; unknown JSON fields survive. Requests keep the client's
//! method and query, answers keep the upstream's status and content
//! type, and upstream failures follow the one client-status policy.
//!
//! The fake upstream answers the way the real implementations do (see
//! `live_families.rs`, which runs the same shapes against them).

#![cfg(test)]
#![cfg(feature = "endpoint-families")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Method, Request, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{
    AppConfig, AuthMode, AuthStyle, ClientKeySpec, KeyWindowLimits, ProviderAdapterConfig,
    ProviderConfig, ProviderRegistry, ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_server::{AppState, BuildInfo, build_router};
use llm_proxy_storage::{ProxyEvent, RecordingBus};
use llm_proxy_transport::ProxyClient;
use tokio::sync::Mutex;
use tower::ServiceExt;

/// One request as the upstream saw it.
#[derive(Debug, Clone)]
struct Seen {
    method: Method,
    path: String,
    query: Option<String>,
    auth: String,
    content_types: Vec<String>,
    body: Bytes,
}

type Capture = Arc<Mutex<Vec<Seen>>>;

/// What the fake answers for a path: the real servers' shapes for the
/// families, and scripted failures for the paths that name one.
fn answer(path: &str) -> Response {
    let json = |status: StatusCode, body: &'static str| {
        (status, [(header::CONTENT_TYPE, "application/json")], body).into_response()
    };
    if path.ends_with("audio/speech") {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "audio/wav")],
            &b"RIFF\x24\x00\x00\x00WAVEfmt "[..],
        )
            .into_response();
    }
    if path.ends_with("audio/transcriptions") {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            "the quick brown fox",
        )
            .into_response();
    }
    if path.ends_with("/content") {
        return (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"in.jsonl\"",
                ),
            ],
            "{\"custom_id\":\"a\"}\n",
        )
            .into_response();
    }
    if path.contains("missing") {
        return json(
            StatusCode::NOT_FOUND,
            r#"{"error":{"message":"No such File object: file-missing","type":"invalid_request_error"}}"#,
        );
    }
    if path.contains("rejected") {
        return json(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"upstream-private detail: purpose must be batch"}}"#,
        );
    }
    if path.contains("denied") {
        return json(
            StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"Incorrect API key provided: sk-ups*****-key"}}"#,
        );
    }
    if path.contains("throttled") {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [
                (header::CONTENT_TYPE, "application/json"),
                (header::RETRY_AFTER, "7"),
            ],
            r#"{"error":{"message":"slow down"}}"#,
        )
            .into_response();
    }
    if path.contains("notthere") {
        return json(StatusCode::NOT_FOUND, r#"{"detail":"Not Found"}"#);
    }
    json(
        StatusCode::OK,
        r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":5,"total_tokens":5}}"#,
    )
}

async fn spawn_upstream() -> (String, Capture) {
    let captured: Capture = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::any(
            move |method: Method,
                  path: axum::extract::Path<String>,
                  uri: Uri,
                  headers: HeaderMap,
                  body: Bytes| {
                let sink = Arc::clone(&sink);
                async move {
                    let header_text = |name: header::HeaderName| {
                        headers
                            .get(name)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or_default()
                            .to_owned()
                    };
                    sink.lock().await.push(Seen {
                        method,
                        path: path.to_string(),
                        query: uri.query().map(str::to_owned),
                        auth: header_text(header::AUTHORIZATION),
                        content_types: headers
                            .get_all(header::CONTENT_TYPE)
                            .iter()
                            .filter_map(|value| value.to_str().ok().map(str::to_owned))
                            .collect(),
                        body: body.clone(),
                    });
                    answer(&path)
                }
            },
        ),
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
    (format!("http://{addr}/v1"), captured)
}

/// Each family at its own endpoint under `base`, as a real provider
/// lays them out.
fn provider_config(base: &str) -> ProviderConfig {
    let mut adapters: HashMap<String, ProviderAdapterConfig> = HashMap::new();
    for (family, path) in [
        (ProviderRouteKind::Embeddings, "embeddings"),
        (ProviderRouteKind::Rerank, "rerank"),
        (
            ProviderRouteKind::AudioTranscriptions,
            "audio/transcriptions",
        ),
        (ProviderRouteKind::AudioSpeech, "audio/speech"),
        (ProviderRouteKind::ImagesGenerations, "images/generations"),
        (ProviderRouteKind::Files, "files"),
        (ProviderRouteKind::Batches, "batches"),
    ] {
        adapters.insert(
            format!("{family:?}"),
            ProviderAdapterConfig {
                protocol: "openai_chat_completions".to_owned(),
                endpoint: format!("{base}/{path}"),
                headers: Arc::new(HashMap::new()),
                forward_inbound_headers: None,
                lossy: false,
            },
        );
    }
    let mut model_aliases = HashMap::new();
    model_aliases.insert(
        "embed-small".to_owned(),
        "text-embedding-3-small".to_owned(),
    );
    model_aliases.insert("voice".to_owned(), "tts-1-hd".to_owned());
    ProviderConfig {
        name: "fake".to_owned(),
        api_key: secrecy::SecretString::from("sk-upstream-key"),
        auth_style: AuthStyle::Bearer,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes: ProviderRoutesConfig {
            messages: None,
            chat_completions: None,
            responses: None,
            generate_content: None,
            embeddings: Some("Embeddings".to_owned()),
            rerank: Some("Rerank".to_owned()),
            audio_transcriptions: Some("AudioTranscriptions".to_owned()),
            audio_speech: Some("AudioSpeech".to_owned()),
            images_generations: Some("ImagesGenerations".to_owned()),
            files: Some("Files".to_owned()),
            batches: Some("Batches".to_owned()),
        },
        model_aliases,
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn state(config: AppConfig, registry: ProviderRegistry, bus: Arc<RecordingBus>) -> AppState {
    AppState::new(
        config,
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
    .with_event_bus(bus)
}

fn base_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind"),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "families-test".to_owned(),
            rate_limit_rpm: 10_000,
            trust_forwarded_headers: false,
            log_format: Default::default(),
            admin_bind: None,
            admin_on_main_listener: false,
            endpoint_families: Default::default(),
            upstream_timeouts: Default::default(),
            bare_v1: Default::default(),
            preflight: Default::default(),
            semantic_cache: Default::default(),
            cache: Default::default(),
            idempotency: Default::default(),
            metrics: Default::default(),
        },
        auth: Default::default(),
        storage: None,
        replica: None,
        mcp: None,
    }
}

fn app_over(config: ProviderConfig) -> (Router, Arc<RecordingBus>) {
    let registry = ProviderRegistry::from_providers(vec![config]).expect("registry builds");
    let bus = Arc::new(RecordingBus::new());
    let app = build_router(state(base_config(), registry, Arc::clone(&bus)));
    (app, bus)
}

/// A PHC verifier for `secret`, as `[auth.clients]` stores them (the
/// same one-argon2-run shape `ManagedKeys::create` produces).
fn phc_of(secret: &str) -> secrecy::SecretString {
    use argon2::PasswordHasher as _;
    secrecy::SecretString::from(
        argon2::Argon2::default()
            .hash_password(secret.as_bytes())
            .expect("phc")
            .to_string(),
    )
}

async fn fixture() -> (Router, Capture, Arc<RecordingBus>) {
    let (endpoint, captured) = spawn_upstream().await;
    let (app, bus) = app_over(provider_config(&endpoint));
    (app, captured, bus)
}

async fn call(
    app: &Router,
    method: &str,
    uri: &str,
    content_type: Option<&str>,
    body: &'static [u8],
) -> Response {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(content_type) = content_type {
        request = request.header(header::CONTENT_TYPE, content_type);
    }
    app.clone()
        .oneshot(request.body(Body::from(body)).expect("request"))
        .await
        .expect("served")
}

async fn body_of(response: Response) -> Bytes {
    axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body")
}

fn header_of(response: &Response, name: header::HeaderName) -> &str {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
}

#[tokio::test]
async fn embeddings_routes_through_with_the_model_alias_and_credential() {
    let (app, captured, bus) = fixture().await;
    let body = r#"{"model":"embed-small","input":["hello"],"vendor_extra_field":true}"#;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/providers/fake/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .expect("request"),
        )
        .await
        .expect("served");
    assert_eq!(response.status(), StatusCode::OK);

    let requests = captured.lock().await;
    let seen = &requests[0];
    assert_eq!(
        seen.path, "v1/embeddings",
        "routed to the configured family endpoint"
    );
    assert_eq!(
        seen.auth, "Bearer sk-upstream-key",
        "the provider credential"
    );
    assert_eq!(seen.content_types, vec!["application/json".to_owned()]);
    let parsed: serde_json::Value = serde_json::from_slice(&seen.body).expect("json");
    assert_eq!(
        parsed["model"].as_str(),
        Some("text-embedding-3-small"),
        "the requested model is aliased for egress"
    );
    assert_eq!(
        parsed["vendor_extra_field"].as_bool(),
        Some(true),
        "unknown fields survive (the same-dialect rule)"
    );
    drop(requests);

    // Usage rows: family-scoped route kind, the requested model kept.
    let events = bus.snapshot();
    let request_row = events
        .iter()
        .find_map(|event| match event {
            ProxyEvent::RequestReceived(row) => Some(row),
            _ => None,
        })
        .expect("request row");
    assert_eq!(request_row.route_kind, "embeddings");
    assert_eq!(request_row.model.requested, "embed-small");
    let completed = events
        .iter()
        .find_map(|event| match event {
            ProxyEvent::ResponseCompleted(row) => Some(row),
            _ => None,
        })
        .expect("completed row");
    // The upstream answered usage {prompt_tokens: 5, total_tokens: 5};
    // the row must carry it as provider-reported, not synthetic zero —
    // settlement and the ledger count family traffic through it.
    assert_eq!(completed.usage.input_tokens, 5);
    assert_eq!(
        completed.usage.provenance,
        llm_proxy_protocol::core::UsageProvenance::ProviderReported,
        "the family's reported tokens reach the ledger row"
    );
}

/// The admission half of the family usage fix: a tpm-limited key must be
/// throttled by the usage its family calls report. The fake upstream
/// answers `usage {prompt_tokens: 5, total_tokens: 5}`; with tpm = 8 the
/// third call inside the minute is refused before any upstream call.
#[tokio::test]
async fn a_tpm_limited_key_is_throttled_by_reported_family_usage() {
    let (endpoint, _captured) = spawn_upstream().await;
    let mut config = base_config();
    config.auth.mode = AuthMode::ClientKeys;
    config.auth.clients.insert(
        "klimited".to_owned(),
        ClientKeySpec::Policy {
            verifier: phc_of("fam-secret"),
            allow_models: vec![],
            allow_providers: vec![],
            limits: Some(KeyWindowLimits {
                rpm: None,
                tpm: Some(8),
                rpd: None,
            }),
            budget: None,
        },
    );
    let registry =
        ProviderRegistry::from_providers(vec![provider_config(&endpoint)]).expect("registry");
    let bus = Arc::new(RecordingBus::new());
    // The token window resets on the calendar minute; a wall clock could
    // roll the minute over between the first and third call and admit the
    // third spuriously. A fixed clock keeps the whole sequence inside one
    // window. The admission step is still the production
    // PolicyAdmission::from_config over this config's key policies.
    let admission = llm_proxy_engine::PolicyAdmission::from_config(&config.auth, &registry)
        .with_clock(|| {
            time::OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("fixed instant")
        });
    let app =
        build_router(state(config, registry, Arc::clone(&bus)).with_admission(Arc::new(admission)));

    for (call, expected) in [
        (1, StatusCode::OK),
        (2, StatusCode::OK),
        (3, StatusCode::TOO_MANY_REQUESTS),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/providers/fake/v1/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::AUTHORIZATION, "Bearer klimited.fam-secret")
                    .body(Body::from(
                        br#"{"model":"embed-small","input":"hi"}"#.to_vec(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("served");
        assert_eq!(
            response.status(),
            expected,
            "call {call}: 5 tokens per call against a tpm of 8 throttles the third"
        );
    }
    // The throttled call never reached the upstream but is still
    // accounted as an attributed refusal pair.
    let events = bus.snapshot();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ProxyEvent::ResponseFailed(_)))
            .count(),
        1,
        "the throttled call wrote a failure row"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ProxyEvent::RequestReceived(_)))
            .count(),
        3,
        "every call, refused or not, wrote its request row"
    );
}

#[tokio::test]
async fn rerank_and_images_and_batches_share_the_spine() {
    let (app, captured, _bus) = fixture().await;
    for uri in [
        "/providers/fake/v1/rerank",
        "/providers/fake/v1/images/generations",
        "/providers/fake/v1/batches",
    ] {
        let response = call(
            &app,
            "POST",
            uri,
            Some("application/json"),
            br#"{"model":"embed-small"}"#,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
    assert_eq!(captured.lock().await.len(), 3, "each family routed once");
}

#[tokio::test]
async fn audio_bodies_pass_through_verbatim_with_their_content_type() {
    let (app, captured, _bus) = fixture().await;
    // A synthetic multipart body; the proxy must not rewrite it.
    let multipart = b"--boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF-bytes\r\n--boundary--\r\n";
    let response = call(
        &app,
        "POST",
        "/providers/fake/v1/audio/transcriptions",
        Some("multipart/form-data; boundary=boundary"),
        multipart,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        header_of(&response, header::CONTENT_TYPE).starts_with("text/plain"),
        "a plain-text transcript keeps its content type"
    );
    assert_eq!(body_of(response).await.as_ref(), b"the quick brown fox");
    let requests = captured.lock().await;
    let seen = &requests[0];
    assert_eq!(
        seen.content_types,
        vec!["multipart/form-data; boundary=boundary".to_owned()],
        "the client's boundary survives, sent once"
    );
    assert_eq!(
        seen.body.as_ref(),
        multipart,
        "audio bytes are forwarded verbatim"
    );
}

#[tokio::test]
async fn speech_is_a_json_request_aliased_and_answered_with_the_upstreams_audio() {
    let (app, captured, _bus) = fixture().await;
    let response = call(
        &app,
        "POST",
        "/providers/fake/v1/audio/speech",
        Some("application/json"),
        br#"{"model":"voice","input":"hello","voice":"alloy","response_format":"wav"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        header_of(&response, header::CONTENT_TYPE),
        "audio/wav",
        "the audio is not relabelled as JSON"
    );
    assert_eq!(
        body_of(response).await.as_ref(),
        b"RIFF\x24\x00\x00\x00WAVEfmt "
    );
    let requests = captured.lock().await;
    let sent: serde_json::Value = serde_json::from_slice(&requests[0].body).expect("json");
    assert_eq!(
        sent["model"].as_str(),
        Some("tts-1-hd"),
        "speech requests are JSON, so the alias applies"
    );
}

#[tokio::test]
async fn file_and_batch_resources_keep_the_clients_method_and_query() {
    let (app, captured, _bus) = fixture().await;
    let exchanges = [
        ("GET", "/providers/fake/v1/files?purpose=batch&limit=2"),
        ("GET", "/providers/fake/v1/files/file-abc"),
        ("GET", "/providers/fake/v1/files/file-abc/content"),
        ("DELETE", "/providers/fake/v1/files/file-abc"),
        ("GET", "/providers/fake/v1/batches?limit=1&after=batch_0"),
        ("GET", "/providers/fake/v1/batches/batch_1"),
        ("POST", "/providers/fake/v1/batches/batch_1/cancel"),
    ];
    for (method, uri) in exchanges {
        let response = call(&app, method, uri, None, b"").await;
        assert_eq!(response.status(), StatusCode::OK, "{method} {uri}");
    }
    let seen: Vec<(String, String, Option<String>, Vec<String>, usize)> = captured
        .lock()
        .await
        .iter()
        .map(|s| {
            (
                s.method.to_string(),
                s.path.clone(),
                s.query.clone(),
                s.content_types.clone(),
                s.body.len(),
            )
        })
        .collect();
    let bodiless = |method: &str, path: &str, query: Option<&str>| {
        (
            method.to_owned(),
            path.to_owned(),
            query.map(str::to_owned),
            Vec::<String>::new(),
            0,
        )
    };
    assert_eq!(
        seen,
        vec![
            bodiless("GET", "v1/files", Some("purpose=batch&limit=2")),
            bodiless("GET", "v1/files/file-abc", None),
            bodiless("GET", "v1/files/file-abc/content", None),
            bodiless("DELETE", "v1/files/file-abc", None),
            bodiless("GET", "v1/batches", Some("limit=1&after=batch_0")),
            bodiless("GET", "v1/batches/batch_1", None),
            bodiless("POST", "v1/batches/batch_1/cancel", None),
        ]
    );
}

#[tokio::test]
async fn a_file_download_keeps_the_upstreams_content_headers() {
    let (app, _captured, _bus) = fixture().await;
    let response = call(
        &app,
        "GET",
        "/providers/fake/v1/files/file-abc/content",
        None,
        b"",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        header_of(&response, header::CONTENT_TYPE),
        "application/octet-stream"
    );
    assert_eq!(
        header_of(&response, header::CONTENT_DISPOSITION),
        "attachment; filename=\"in.jsonl\""
    );
    assert_eq!(body_of(response).await.as_ref(), b"{\"custom_id\":\"a\"}\n");
}

#[tokio::test]
async fn the_endpoint_query_survives_and_the_resource_goes_before_it() {
    // Azure-style endpoints carry their API version in the query.
    let (endpoint, captured) = spawn_upstream().await;
    let mut config = provider_config(&endpoint);
    for adapter in config.adapters.values_mut() {
        adapter.endpoint = format!("{}?api-version=2024-10-21", adapter.endpoint);
    }
    let (app, _bus) = app_over(config);

    let response = call(
        &app,
        "POST",
        "/providers/fake/v1/batches/batch_1/cancel",
        None,
        b"",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let response = call(&app, "GET", "/providers/fake/v1/files?limit=5", None, b"").await;
    assert_eq!(response.status(), StatusCode::OK);

    let seen: Vec<(String, Option<String>)> = captured
        .lock()
        .await
        .iter()
        .map(|s| (s.path.clone(), s.query.clone()))
        .collect();
    assert_eq!(
        seen,
        vec![
            (
                "v1/batches/batch_1/cancel".to_owned(),
                Some("api-version=2024-10-21".to_owned())
            ),
            (
                "v1/files".to_owned(),
                Some("api-version=2024-10-21&limit=5".to_owned())
            ),
        ]
    );
}

#[tokio::test]
async fn resource_paths_that_could_leave_the_family_are_refused() {
    let (app, captured, bus) = fixture().await;
    for (method, uri) in [
        (
            "POST",
            "/providers/fake/v1/batches/..%2F..%2Fchat%2Fcompletions",
        ),
        ("POST", "/providers/fake/v1/batches/b/../../embeddings"),
        ("GET", "/providers/fake/v1/files/%2e%2e/%2e%2e/models"),
        ("GET", "/providers/fake/v1/files/file-abc%3Fpurpose=x"),
    ] {
        let response = call(&app, method, uri, None, b"").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{method} {uri}");
        let body: serde_json::Value =
            serde_json::from_slice(&body_of(response).await).expect("json error");
        assert_eq!(
            body["error"]["type"].as_str(),
            Some("invalid_request_error"),
            "an OpenAI-shaped error: {body}"
        );
    }
    assert!(
        captured.lock().await.is_empty(),
        "nothing outside the family endpoint reached the upstream"
    );
    // Each refusal writes an attributed pair: a RequestReceived naming
    // the caller, provider and route kind, then the failure row.
    let events = bus.snapshot();
    let requests = events
        .iter()
        .filter(|event| matches!(event, ProxyEvent::RequestReceived(_)))
        .count();
    let failures = events
        .iter()
        .filter(|event| matches!(event, ProxyEvent::ResponseFailed(_)))
        .count();
    assert_eq!(requests, 4, "every refused call leaves a request row");
    assert_eq!(failures, 4, "every refused call leaves a failure row");
    assert_eq!(events.len(), 8, "nothing else: {events:?}");
    assert!(
        events.iter().all(|event| match event {
            ProxyEvent::RequestReceived(row) => {
                row.provider == "fake" && (row.route_kind == "files" || row.route_kind == "batches")
            }
            _ => true,
        }),
        "the request rows carry the caller's provider and route kind"
    );
}

#[tokio::test]
async fn upstream_failures_follow_the_one_client_status_policy() {
    let (endpoint, _captured) = spawn_upstream().await;
    let mut config = provider_config(&endpoint);
    // Point three families at scripted failures.
    for (name, path) in [
        ("Rerank", "rejected"),
        ("ImagesGenerations", "denied"),
        ("Embeddings", "throttled"),
    ] {
        if let Some(adapter) = config.adapters.get_mut(name) {
            adapter.endpoint = format!("{endpoint}/{path}");
        }
    }
    let (app, _bus) = app_over(config);
    let post = |uri: &'static str| call(&app, "POST", uri, Some("application/json"), b"{}");

    // A client-fixable 400 passes through, without the upstream's text.
    let rejected = post("/providers/fake/v1/rerank").await;
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        header_of(&rejected, "x-should-retry".parse().expect("name")),
        "false"
    );
    let text = String::from_utf8_lossy(&body_of(rejected).await).into_owned();
    assert!(
        !text.contains("upstream-private"),
        "no upstream body: {text}"
    );

    // The operator's credential failing is the proxy's problem: 502,
    // and the upstream's key hint never reaches the client.
    let denied = post("/providers/fake/v1/images/generations").await;
    assert_eq!(denied.status(), StatusCode::BAD_GATEWAY);
    let text = String::from_utf8_lossy(&body_of(denied).await).into_owned();
    assert!(!text.contains("sk-ups"), "no key hint: {text}");

    // A rate limit passes through with the upstream's reset advice.
    let throttled = post("/providers/fake/v1/embeddings").await;
    assert_eq!(throttled.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(header_of(&throttled, header::RETRY_AFTER), "7");
}

#[tokio::test]
async fn an_unknown_upstream_resource_is_a_404_not_an_upstream_fault() {
    let (endpoint, _captured) = spawn_upstream().await;
    let mut config = provider_config(&endpoint);
    if let Some(adapter) = config.adapters.get_mut("Embeddings") {
        adapter.endpoint = format!("{endpoint}/notthere");
    }
    let (app, _bus) = app_over(config);

    // The client named a resource the provider does not have.
    for (method, uri) in [
        ("GET", "/providers/fake/v1/files/file-missing"),
        ("POST", "/providers/fake/v1/batches/batch_missing/cancel"),
    ] {
        let response = call(&app, method, uri, None, b"").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {uri}");
        assert_eq!(
            header_of(&response, "x-should-retry".parse().expect("name")),
            "false"
        );
    }

    // A collection endpoint that 404s is a misconfigured upstream, which
    // is still the proxy's fault.
    let misrouted = call(
        &app,
        "POST",
        "/providers/fake/v1/embeddings",
        Some("application/json"),
        b"{}",
    )
    .await;
    assert_eq!(misrouted.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn a_provider_with_egress_sends_its_family_traffic_through_its_proxy() {
    // [provider.egress]: the engine routes this provider's chat traffic
    // through its forward proxy, so its family calls must go there too,
    // never directly. The upstream host does not resolve; only the proxy
    // can answer.
    let proxy = Router::new().fallback(|uri: Uri| async move {
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "proxied": uri.to_string() }).to_string(),
        )
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let proxy_addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, proxy).await {
            tracing::trace!(%error, "fake egress proxy stopped");
        }
    });

    let config = provider_config("http://egress-upstream.invalid/v1");
    let registry = ProviderRegistry::from_providers(vec![config]).expect("registry builds");
    let egress = ProxyClient::try_with_egress_proxy(&format!("http://{proxy_addr}"))
        .expect("egress client builds");
    let bus = Arc::new(RecordingBus::new());
    let app = build_router(
        state(base_config(), registry, Arc::clone(&bus))
            .with_egress_clients(HashMap::from([("fake".to_owned(), Arc::new(egress))])),
    );

    let response = call(
        &app,
        "POST",
        "/providers/fake/v1/embeddings",
        Some("application/json"),
        br#"{"model":"embed-small","input":"hi"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(&body_of(response).await).expect("json");
    assert_eq!(
        body["proxied"].as_str(),
        Some("http://egress-upstream.invalid/v1/embeddings"),
        "the family call went through the provider's egress proxy"
    );
}

#[tokio::test]
async fn a_passthrough_provider_gets_the_clients_own_key_on_family_calls() {
    let (endpoint, captured) = spawn_upstream().await;
    let mut config = provider_config(&endpoint);
    config.passthrough_auth = true;
    config.api_key = secrecy::SecretString::from("sk-operator-key-never-sent");
    let (app, _bus) = app_over(config);

    let with_key = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/providers/fake/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer sk-client-own-key")
                .body(Body::from(r#"{"model":"embed-small","input":"hi"}"#))
                .expect("request"),
        )
        .await
        .expect("served");
    assert_eq!(with_key.status(), StatusCode::OK);
    assert_eq!(
        captured.lock().await[0].auth,
        "Bearer sk-client-own-key",
        "the client's key, as on the chat routes"
    );

    // Without a key there is nothing to forward: refused before egress.
    let without_key = call(
        &app,
        "POST",
        "/providers/fake/v1/embeddings",
        Some("application/json"),
        br#"{"model":"embed-small","input":"hi"}"#,
    )
    .await;
    assert_eq!(without_key.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(captured.lock().await.len(), 1, "no second upstream call");
}

#[tokio::test]
async fn layer_refusals_on_a_family_path_use_the_openai_error_shape() {
    // The limit layer and the method fallback answer before the handler;
    // their bodies follow the path's dialect, which for every family is
    // OpenAI's.
    let (endpoint, captured) = spawn_upstream().await;
    let registry =
        ProviderRegistry::from_providers(vec![provider_config(&endpoint)]).expect("registry");
    let mut config = base_config();
    config.server.endpoint_families.insert(
        "embeddings".to_owned(),
        llm_proxy_core::EndpointFamilyConfig {
            max_body_bytes: 64,
            request_timeout: Duration::from_secs(30),
        },
    );
    let app = build_router(state(config, registry, Arc::new(RecordingBus::new())));

    let too_large = call(
        &app,
        "POST",
        "/providers/fake/v1/embeddings",
        Some("application/json"),
        &[b'x'; 1024],
    )
    .await;
    let wrong_method = call(&app, "DELETE", "/providers/fake/v1/files", None, b"").await;
    for (response, status) in [
        (too_large, StatusCode::PAYLOAD_TOO_LARGE),
        (wrong_method, StatusCode::METHOD_NOT_ALLOWED),
    ] {
        assert_eq!(response.status(), status);
        let body: serde_json::Value =
            serde_json::from_slice(&body_of(response).await).expect("json error");
        assert!(
            body.get("type").is_none() && body["error"]["message"].is_string(),
            "an OpenAI error body, not another dialect's: {body}"
        );
    }
    assert!(captured.lock().await.is_empty());
}

#[tokio::test]
async fn an_unconfigured_family_is_refused_with_its_route_named() {
    // A registry with no family routes: the family kind is unconfigured,
    // refused as the chat routes refuse an unconfigured route kind.
    let (endpoint, captured) = spawn_upstream().await;
    let mut config = provider_config(&endpoint);
    config.routes.embeddings = None;
    let (app, bus) = app_over(config);

    let response = call(
        &app,
        "POST",
        "/providers/fake/v1/embeddings",
        Some("application/json"),
        br#"{"model":"embed-small"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value =
        serde_json::from_slice(&body_of(response).await).expect("json error");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("\"embeddings\""),
        "the refusal names the missing route: {message}"
    );
    assert!(captured.lock().await.is_empty(), "no upstream call");
    // The refusal is accounted and attributable: the request row comes
    // first (caller, provider, route kind), then the failure row.
    let events = bus.snapshot();
    match (events.first(), events.get(1)) {
        (Some(ProxyEvent::RequestReceived(request)), Some(ProxyEvent::ResponseFailed(_))) => {
            assert_eq!(request.route_kind, "embeddings");
            assert_eq!(request.provider, "fake");
            assert_eq!(request.model.requested, "embed-small");
        }
        other => panic!(
            "an unconfigured-family refusal writes an attributed request \
             row then a failure row, got {other:?}"
        ),
    }
}
