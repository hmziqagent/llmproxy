//! Plan 06 integration tests: per-caller single-flight, the opt-in
//! `Idempotency-Key` store, and the opt-in exact-match response cache,
//! against a loopback fake upstream.

#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::IntoResponse;
use futures::StreamExt;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_core::{
    AppConfig, AuthStyle, IdempotencyConfig, ModelId, ModelPricing, PoolConfig,
    ProviderAdapterConfig, ProviderConfig, ProviderRegistry, ProviderRouteKind,
    ProviderRoutesConfig, RetryConfig, ServerConfig,
};
use llm_proxy_engine::{
    Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket, ClientDialect, Engine,
    EngineBuilder, EngineError, EngineRequest, EngineStream, InboundRequest, MemoryCache,
};
use llm_proxy_protocol::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreRole, Cost, ModelRef, SamplingOptions,
};
use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};
use rust_decimal::Decimal;
use sha2::{Digest as _, Sha256};

/// A slow fake upstream: counts calls and answers a fixed non-streaming
/// response after `delay`, so identical requests overlap in flight.
struct CountingUpstream {
    calls: Arc<AtomicUsize>,
    delay: Duration,
    status: u16,
}

impl CountingUpstream {
    fn new(delay: Duration) -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            delay,
            status: 200,
        }
    }

    fn failing() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            delay: Duration::ZERO,
            status: 500,
        }
    }

    fn router(&self) -> Router {
        let calls = Arc::clone(&self.calls);
        let delay = self.delay;
        let status = self.status;
        Router::new().route(
            "/{*path}",
            axum::routing::post(move || async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                if status == 200 {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello from upstream"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
                    )
                        .into_response()
                } else {
                    (StatusCode::INTERNAL_SERVER_ERROR, "upstream broke")
                        .into_response()
                }
            }),
        )
    }
}

async fn spawn_upstream(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("fake upstream addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    format!("http://{addr}/v1/messages")
}

fn test_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind addr"),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "engine-test".to_owned(),
            rate_limit_rpm: 1000,
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

fn provider(endpoint: String) -> ProviderConfig {
    provider_named("fake", endpoint)
}

fn provider_named(name: &str, endpoint: String) -> ProviderConfig {
    let mut adapters = std::collections::HashMap::new();
    adapters.insert(
        "messages".to_owned(),
        ProviderAdapterConfig {
            protocol: "anthropic_messages".to_owned(),
            endpoint,
            headers: Arc::new(std::collections::HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    );
    let routes = ProviderRoutesConfig {
        messages: Some("messages".to_owned()),
        chat_completions: None,
        ..Default::default()
    };
    ProviderConfig {
        name: name.to_owned(),
        api_key: secrecy::SecretString::from("upstream-key"),
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes,
        model_aliases: std::collections::HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn core_request(stream: bool, prompt: &str) -> CoreRequest {
    CoreRequest {
        model: ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        messages: vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: prompt.to_owned(),
                cache: None,
            }],
        }],
        system: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        sampling: SamplingOptions::default(),
        stream,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

fn body_hash_of(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn engine_request(
    stream: bool,
    prompt: &str,
    caller: &str,
    idempotency_key: Option<&str>,
) -> EngineRequest {
    engine_request_for("fake", stream, prompt, caller, idempotency_key)
}

fn engine_request_for(
    provider_name: &str,
    stream: bool,
    prompt: &str,
    caller: &str,
    idempotency_key: Option<&str>,
) -> EngineRequest {
    let mut headers = std::collections::HashMap::new();
    if let Some(key) = idempotency_key {
        headers.insert("idempotency-key".to_owned(), key.to_owned());
    }
    EngineRequest::new(
        provider_name,
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core_request(stream, prompt),
        InboundRequest {
            auth: None,
            headers,
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_caller(caller)
    .with_body_hash(body_hash_of(prompt))
}

fn engine_with(
    providers: Vec<ProviderConfig>,
    config: AppConfig,
    bus: Arc<RecordingBus>,
    admission: Option<Arc<dyn Admission>>,
    cache: Option<MemoryCache>,
) -> Engine {
    let registry = ProviderRegistry::from_providers(providers).expect("provider registry");
    let mut builder = EngineBuilder::new(config)
        .providers(Arc::new(registry))
        .event_bus(bus as Arc<dyn EventBus>);
    if let Some(admission) = admission {
        builder = builder.admission(admission);
    }
    if let Some(cache) = cache {
        builder = builder.cache(Arc::new(cache));
    }
    builder.build().expect("engine builds")
}

fn build_engine(endpoint: String, bus: Arc<RecordingBus>, cache: Option<MemoryCache>) -> Engine {
    engine_with(vec![provider(endpoint)], test_config(), bus, None, cache)
}

fn response_text(response: &llm_proxy_engine::EngineResponse) -> String {
    response
        .core
        .content
        .iter()
        .find_map(|content| match content {
            CoreContent::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn completed_rows(bus: &RecordingBus) -> Vec<(bool, String)> {
    bus.snapshot()
        .into_iter()
        .filter_map(|event| match event {
            ProxyEvent::ResponseCompleted(row) => Some((row.cache_hit, row.request_id)),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn with_every_opt_in_off_nothing_is_stored_and_every_request_runs() {
    // Done-when 1: no cache, no idempotency — the default engine. Two
    // identical requests both reach the upstream (nothing was retained that
    // could answer or reject the second), and no recorded ledger row carries
    // prompt or response bytes.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let first = engine
        .execute(engine_request(
            false,
            "secret prompt alpha",
            "caller-a",
            None,
        ))
        .await
        .expect("first request");
    let second = engine
        .execute(engine_request(
            false,
            "secret prompt alpha",
            "caller-a",
            None,
        ))
        .await
        .expect("second identical request is answered, not rejected");

    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2);
    assert_eq!(response_text(&first), "hello from upstream");
    assert_eq!(response_text(&second), "hello from upstream");

    for event in bus.snapshot() {
        let serialized = serde_json::to_string(&event).expect("events serialize");
        assert!(
            !serialized.contains("secret prompt alpha"),
            "a ledger row carries the prompt: {serialized}"
        );
        assert!(
            !serialized.contains("hello from upstream"),
            "a ledger row carries the response: {serialized}"
        );
    }
    assert!(
        completed_rows(&bus).iter().all(|(hit, _)| !hit),
        "no row may claim a hit with every opt-in off"
    );
}

#[tokio::test]
async fn identical_concurrent_requests_share_one_upstream_call() {
    let upstream = CountingUpstream::new(Duration::from_millis(150));
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let mut requests = Vec::new();
    for i in 0..4 {
        let engine = engine.clone();
        requests.push(async move {
            engine
                .execute(
                    engine_request(false, "shared prompt", "caller-a", None)
                        .with_request_id(format!("req-{i}")),
                )
                .await
        });
    }
    let results = futures::future::join_all(requests).await;
    assert_eq!(results.len(), 4);
    for result in results {
        assert_eq!(
            response_text(&result.expect("joiner shares the leader's response")),
            "hello from upstream"
        );
    }

    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        1,
        "four identical concurrent requests from one caller must share one upstream call"
    );
    let rows = completed_rows(&bus);
    let hits = rows.iter().filter(|(hit, _)| *hit).count();
    let misses = rows.len() - hits;
    assert_eq!(misses, 1, "exactly one request made the upstream call");
    assert_eq!(hits, 3, "the three joiners are recorded as served copies");
}

#[tokio::test]
async fn single_flight_is_scoped_to_the_caller() {
    let upstream = CountingUpstream::new(Duration::from_millis(150));
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, bus, None);

    let a = engine.execute(engine_request(false, "same body", "caller-a", None));
    let b = engine.execute(engine_request(false, "same body", "caller-b", None));
    let (a, b) = futures::future::join(a, b).await;
    a.expect("caller a runs");
    b.expect("caller b runs");
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "one caller's in-flight request must not be shared with another caller"
    );
}

#[tokio::test]
async fn a_single_flight_leader_failure_fails_its_joiners() {
    let upstream = CountingUpstream::failing();
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let a = engine.execute(engine_request(false, "shared prompt", "caller-a", None));
    let b = engine.execute(engine_request(false, "shared prompt", "caller-a", None));
    let (a, b) = futures::future::join(a, b).await;
    assert!(a.is_err(), "the leader fails");
    assert!(matches!(b, Err(EngineError::Upstream { status: 500, .. })));
    assert!(
        completed_rows(&bus).is_empty(),
        "no completion rows for failed requests"
    );
}

#[tokio::test]
async fn identical_concurrent_streams_share_one_upstream_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&counted);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(120)).await;
                let chunks = vec![
                    bytes::Bytes::from_static(
                        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"shared\"}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    ),
                ];
                let body_stream =
                    futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
                (
                    StatusCode::OK,
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    )],
                    Body::from_stream(body_stream),
                )
                    .into_response()
            }
        }),
    );
    let endpoint = spawn_upstream(app).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let leader = engine.execute_stream(engine_request(true, "shared stream", "caller-a", None));
    let joiner = engine.execute_stream(
        engine_request(true, "shared stream", "caller-a", None).with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;
    let leader = leader.expect("leader streams");
    let joiner = joiner.expect("joiner joins the shared stream");

    let collect = |stream: llm_proxy_engine::EngineStream| async move {
        let mut all = stream.head;
        let mut tail = stream.tail;
        while let Some(event) = tail.next().await {
            all.push(event);
        }
        all
    };
    let leader_events = collect(leader).await;
    let joiner_events = collect(joiner).await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        leader_events, joiner_events,
        "the joiner sees exactly the leader's events, in order"
    );
    assert!(
        joiner_events
            .iter()
            .any(|event| matches!(event, CoreEvent::MessageStop { .. })),
        "the joiner's replay reaches the terminal: {joiner_events:?}"
    );

    // Both requests reconcile: one charged row (the leader) and one served
    // copy (the joiner).
    let rows = completed_rows(&bus);
    assert_eq!(rows.len(), 2, "leader and joiner each record a row");
    assert_eq!(rows.iter().filter(|(hit, _)| *hit).count(), 1);
}

#[tokio::test]
async fn a_cache_hit_answers_where_the_dedup_window_used_to_409() {
    // Done-when 2: identical request, answered from the cache with the same
    // body the first call produced — where the retired dedup window answered
    // 409.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let cache = MemoryCache::new(Duration::from_secs(60), 16);
    let engine = build_engine(endpoint, Arc::clone(&bus), Some(cache));

    let first = engine
        .execute(engine_request(false, "cached prompt", "caller-a", None))
        .await
        .expect("first");
    let second = engine
        .execute(engine_request(false, "cached prompt", "caller-a", None))
        .await
        .expect("a cache hit succeeds where dedup used to answer 409");

    assert_eq!(upstream.calls.load(Ordering::SeqCst), 1);
    assert_eq!(response_text(&first), response_text(&second));
    assert_eq!(second.core.usage.input_tokens, 10);

    let rows = completed_rows(&bus);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows.iter().filter(|(hit, _)| *hit).count(), 1);
}

#[tokio::test]
async fn one_callers_cached_response_is_never_returned_to_another() {
    // Done-when 3: the cache key is caller-scoped.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let cache = MemoryCache::new(Duration::from_secs(60), 16);
    let engine = build_engine(endpoint, Arc::clone(&bus), Some(cache));

    engine
        .execute(engine_request(false, "scoped prompt", "caller-a", None))
        .await
        .expect("caller a stores");
    engine
        .execute(engine_request(false, "scoped prompt", "caller-b", None))
        .await
        .expect("caller b must run its own request");
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "caller b may not be served caller a's stored response"
    );

    // And caller a's own second request is a hit.
    engine
        .execute(engine_request(false, "scoped prompt", "caller-a", None))
        .await
        .expect("caller a hits");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        completed_rows(&bus).iter().filter(|(hit, _)| *hit).count(),
        1
    );
}

#[tokio::test]
async fn expired_cache_entries_are_misses() {
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let cache = MemoryCache::new(Duration::from_millis(50), 16);
    let engine = build_engine(endpoint, Arc::clone(&bus), Some(cache));

    engine
        .execute(engine_request(false, "short ttl", "caller-a", None))
        .await
        .expect("first");
    tokio::time::sleep(Duration::from_millis(80)).await;
    engine
        .execute(engine_request(false, "short ttl", "caller-a", None))
        .await
        .expect("after expiry");
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "an entry past its TTL must not be served"
    );
}

#[tokio::test]
async fn the_cache_evicts_oldest_entries_at_its_bound() {
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let cache = MemoryCache::new(Duration::from_secs(60), 1);
    let engine = build_engine(endpoint, bus, Some(cache));

    engine
        .execute(engine_request(false, "first body", "caller-a", None))
        .await
        .expect("first stored");
    engine
        .execute(engine_request(false, "second body", "caller-a", None))
        .await
        .expect("second evicts the first at capacity 1");
    engine
        .execute(engine_request(false, "first body", "caller-a", None))
        .await
        .expect("the evicted body runs again");
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        3,
        "the first entry must have been evicted by the second"
    );
}

fn idempotency_config() -> AppConfig {
    let mut config = test_config();
    config.server.idempotency = IdempotencyConfig {
        enabled: true,
        ttl: Duration::from_secs(3600),
        max_entries: 1024,
    };
    config
}

fn build_idempotent_engine(endpoint: String, bus: Arc<RecordingBus>) -> Engine {
    engine_with(
        vec![provider(endpoint)],
        idempotency_config(),
        bus,
        None,
        None,
    )
}

#[tokio::test]
async fn idempotency_key_replays_after_completion_and_409s_while_in_flight() {
    let upstream = CountingUpstream::new(Duration::from_millis(150));
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, Arc::clone(&bus));

    // While the first is in flight, a matching retry is 409 — only then.
    let leader_engine = engine.clone();
    let first = tokio::spawn(async move {
        leader_engine
            .execute(engine_request(
                false,
                "idem body",
                "caller-a",
                Some("key-1"),
            ))
            .await
    });
    tokio::time::sleep(Duration::from_millis(30)).await;
    let conflict = engine
        .execute(engine_request(
            false,
            "idem body",
            "caller-a",
            Some("key-1"),
        ))
        .await
        .expect_err("an in-flight key conflicts");
    assert!(
        matches!(conflict, EngineError::IdempotencyInFlight),
        "got {conflict:?}"
    );
    assert_eq!(conflict.status_code(), 409);
    let first = first
        .await
        .expect("join of the spawned first request")
        .expect("first completes");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 1);

    // A retry after completion replays the stored response — no second call.
    let replay = engine
        .execute(engine_request(
            false,
            "idem body",
            "caller-a",
            Some("key-1"),
        ))
        .await
        .expect("matching retry replays");
    assert_eq!(response_text(&replay), response_text(&first));
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 1);

    let rows = completed_rows(&bus);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows.iter().filter(|(hit, _)| *hit).count(), 1);
}

#[tokio::test]
async fn idempotency_keys_are_scoped_to_the_caller() {
    let upstream = CountingUpstream::new(Duration::from_millis(150));
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, bus);

    let a = engine.execute(engine_request(
        false,
        "caller scoped",
        "caller-a",
        Some("key-1"),
    ));
    tokio::time::sleep(Duration::from_millis(30)).await;
    let b = engine.execute(engine_request(
        false,
        "caller scoped",
        "caller-b",
        Some("key-1"),
    ));
    let (a, b) = futures::future::join(a, b).await;
    a.expect("caller a runs under its own key claim");
    b.expect("another caller's key claim must not 409 or replay for this one");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_key_reused_with_a_different_body_is_a_client_error() {
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, bus);

    engine
        .execute(engine_request(
            false,
            "original body",
            "caller-a",
            Some("key-1"),
        ))
        .await
        .expect("first");
    let error = engine
        .execute(engine_request(
            false,
            "different body",
            "caller-a",
            Some("key-1"),
        ))
        .await
        .expect_err("a key cannot be reused with a different body");
    assert!(matches!(error, EngineError::InvalidRequest(_)));
    assert_eq!(error.status_code(), 400);
}

#[tokio::test]
async fn a_failed_first_request_releases_its_key() {
    let upstream = CountingUpstream::failing();
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, bus);

    let first = engine
        .execute(engine_request(false, "retry me", "caller-a", Some("key-1")))
        .await;
    first.expect_err("the first attempt fails");

    // Failures are not stored: the retry executes again.
    let retry = engine
        .execute(engine_request(false, "retry me", "caller-a", Some("key-1")))
        .await;
    retry.expect_err("the retry fails too — but by executing, not replaying");
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "a retry after a failure must execute, not replay or 409"
    );
}

#[tokio::test]
async fn without_the_opt_in_the_idempotency_header_is_ignored() {
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    engine
        .execute(engine_request(
            false,
            "ignored header",
            "caller-a",
            Some("key-1"),
        ))
        .await
        .expect("first");
    engine
        .execute(engine_request(
            false,
            "ignored header",
            "caller-a",
            Some("key-1"),
        ))
        .await
        .expect("second runs again — the feature is off");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 2);
    assert!(
        completed_rows(&bus).iter().all(|(hit, _)| !hit),
        "no hit rows with the opt-in off"
    );
}

#[tokio::test]
async fn a_streaming_outcome_replays_as_a_stream_for_a_matching_retry() {
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(|| async {
            tokio::time::sleep(Duration::from_millis(40)).await;
            let chunks = vec![
                bytes::Bytes::from_static(
                    b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_i2\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                ),
                bytes::Bytes::from_static(
                    b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"replayed\"}}\n\n",
                ),
                bytes::Bytes::from_static(
                    b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                ),
            ];
            let body_stream = futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
            (
                StatusCode::OK,
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("text/event-stream"),
                )],
                Body::from_stream(body_stream),
            )
                .into_response()
        }),
    );
    let endpoint = spawn_upstream(app).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, Arc::clone(&bus));

    let first = engine
        .execute_stream(engine_request(
            true,
            "stream idem",
            "caller-a",
            Some("key-s"),
        ))
        .await
        .expect("first streams");
    let mut events = first.head;
    let mut tail = first.tail;
    while let Some(event) = tail.next().await {
        events.push(event);
    }
    tokio::time::sleep(Duration::from_millis(30)).await;

    let replay = engine
        .execute_stream(engine_request(
            true,
            "stream idem",
            "caller-a",
            Some("key-s"),
        ))
        .await
        .expect("matching streaming retry replays the stored events");
    let mut replayed = replay.head;
    let mut tail = replay.tail;
    while let Some(event) = tail.next().await {
        replayed.push(event);
    }
    assert_eq!(
        events, replayed,
        "the replay yields exactly the stored events"
    );

    let rows = completed_rows(&bus);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows.iter().filter(|(hit, _)| *hit).count(), 1);
}

/// A streaming upstream that commits the caller with one valid event and
/// then fails in-band: an upstream `error` event, then the stream ends
/// without a terminal. The leader's client sees a well-formed 200 whose
/// stream ends in errors, exactly like a real mid-stream provider failure.
fn mid_stream_failure_upstream(calls: Arc<AtomicUsize>, delay: Duration) -> Router {
    Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                let chunks = vec![
                    bytes::Bytes::from_static(
                        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_f\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"upstream fell over mid-stream\"}}\n\n",
                    ),
                ];
                let body_stream =
                    futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
                (
                    StatusCode::OK,
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    )],
                    Body::from_stream(body_stream),
                )
                    .into_response()
            }
        }),
    )
}

/// A streaming upstream that waits `delay` and then refuses with a 400, so
/// the leader fails before its first byte while a joiner is already waiting.
fn delayed_400_upstream(calls: Arc<AtomicUsize>, delay: Duration) -> Router {
    Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                (StatusCode::BAD_REQUEST, "model is not available").into_response()
            }
        }),
    )
}

/// An admission step that records the usage and billed cost each settlement
/// was charged with, so a test can assert what reached the budget counters.
/// When `refuse_rekey_for` names a member, moving the reservation to that
/// member fails, which drives the streaming ladder through `stop_with`.
#[derive(Debug, Default)]
struct UsageRecordingAdmission {
    settled_usage: std::sync::Mutex<Vec<Option<u64>>>,
    settled_costs: std::sync::Mutex<Vec<Option<Cost>>>,
    refuse_rekey_for: Option<String>,
}

#[async_trait::async_trait]
impl Admission for UsageRecordingAdmission {
    async fn reserve(&self, _claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, outcome: &AdmissionOutcome<'_>) {
        self.settled_usage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(outcome.usage.map(total_usage_tokens));
        self.settled_costs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(outcome.cost.cloned());
    }

    async fn rekey(
        &self,
        _ticket: &AdmissionTicket,
        member: &str,
        _model: &str,
    ) -> Result<(), EngineError> {
        match self.refuse_rekey_for.as_deref() {
            Some(refused) if refused == member => Err(EngineError::Internal(format!(
                "test admission refuses to rekey to {member}"
            ))),
            _ => Ok(()),
        }
    }
}

fn total_usage_tokens(usage: &llm_proxy_protocol::core::Usage) -> u64 {
    [
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens.unwrap_or(0),
        usage.cache_read_input_tokens.unwrap_or(0),
        usage.reasoning_tokens.unwrap_or(0),
    ]
    .into_iter()
    .map(|count| u64::try_from(count).unwrap_or(0))
    .sum()
}

#[tokio::test]
async fn a_pre_commit_failure_charges_the_usage_its_prefix_already_reported() {
    // The buffered prefix reported provider usage before the stream failed
    // before committing: settlement must charge those tokens — the provider
    // billed them — exactly like the committed-head capture does on a
    // post-commit failure.
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&counted);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                // One chunk: a usage delta, then a leading in-band error —
                // the whole prefix decodes before the failure is visible.
                let chunks = vec![bytes::Bytes::from_static(
                    b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"input_tokens\":7,\"output_tokens\":5}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"dying before the first byte\"}}\n\n",
                )];
                let body_stream =
                    futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
                (
                    StatusCode::OK,
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    )],
                    Body::from_stream(body_stream),
                )
                    .into_response()
            }
        }),
    );
    let endpoint = spawn_upstream(app).await;
    let bus = Arc::new(RecordingBus::new());
    let admission = Arc::new(UsageRecordingAdmission::default());
    let engine = engine_with(
        vec![provider(endpoint)],
        test_config(),
        bus,
        Some(Arc::clone(&admission) as Arc<dyn Admission>),
        None,
    );

    let failed = engine
        .execute_stream(engine_request(true, "billed prefix", "caller-a", None))
        .await;
    assert!(failed.is_err(), "the stream never commits: {failed:?}");

    let settled = async {
        loop {
            if !admission
                .settled_usage
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), settled)
        .await
        .expect("the failed stream settles its reservation");
    let recorded = admission
        .settled_usage
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    assert_eq!(
        recorded,
        vec![Some(12)],
        "settlement must charge the prefix's reported usage (7 in + 5 out), got {recorded:?}"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// One SSE chunk: a `message_delta` that reports 7 input + 5 output tokens,
/// then a leading in-band error — the whole prefix decodes before the
/// failure is visible, so the pre-commit funnel stashes its usage.
const PREFIX_USAGE_AND_ERROR_CHUNK: &[u8] = b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"input_tokens\":7,\"output_tokens\":5}}\n\nevent: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"dying before the first byte\"}}\n\n";

/// How the upstream fails the call after the one that reported prefix
/// usage: a terminal 400, a transient 500, or a 200 whose bytes the SSE
/// framer rejects.
#[derive(Clone, Copy)]
enum NextCallFailure {
    TerminalBadRequest,
    TransientServerError,
    FramingGarbage,
}

/// The response for one upstream call: call 0 streams the prefix that
/// reports provider usage and then dies with a leading in-band error; every
/// later call fails the way `failure` says.
async fn prefix_failure_response(
    call: usize,
    failure: NextCallFailure,
) -> axum::response::Response {
    if call == 0 {
        let body = futures::stream::iter(vec![Ok::<_, std::convert::Infallible>(
            bytes::Bytes::from_static(PREFIX_USAGE_AND_ERROR_CHUNK),
        )]);
        return (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            )],
            Body::from_stream(body),
        )
            .into_response();
    }
    match failure {
        NextCallFailure::TerminalBadRequest => {
            (StatusCode::BAD_REQUEST, "model is not available").into_response()
        }
        NextCallFailure::TransientServerError => {
            (StatusCode::INTERNAL_SERVER_ERROR, "upstream broke").into_response()
        }
        NextCallFailure::FramingGarbage => {
            let body = futures::stream::iter(vec![Ok::<_, std::convert::Infallible>(
                bytes::Bytes::copy_from_slice(&[0xFF, 0xFE, b'\n']),
            )]);
            (
                StatusCode::OK,
                [(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("text/event-stream"),
                )],
                Body::from_stream(body),
            )
                .into_response()
        }
    }
}

/// A streaming upstream whose first call streams a prefix that reports
/// provider usage and then dies with a leading in-band error; every later
/// call fails the way `failure` says. Returns the router and its call
/// counter.
fn prefix_usage_then_failure_upstream(failure: NextCallFailure) -> (Router, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&counted);
            async move {
                let call = calls.fetch_add(1, Ordering::SeqCst);
                prefix_failure_response(call, failure).await
            }
        }),
    );
    (app, calls)
}

/// The provider pricing the prefix's cost is billed at: $0.000002 per input
/// token and $0.000005 per output token.
fn prefix_pricing() -> ModelPricing {
    ModelPricing {
        input: Decimal::new(2, 6),
        output: Decimal::new(5, 6),
        ..Default::default()
    }
}

/// A provider config with [`prefix_pricing()`] on the test model and a
/// two-attempt ladder, so the call that fails after the prefix is a real
/// second attempt.
fn priced_provider(name: &str, endpoint: String, pricing: ModelPricing) -> ProviderConfig {
    let mut provider = provider_named(name, endpoint);
    provider
        .pricing
        .insert(ModelId::new("claude-sonnet-4-6"), pricing);
    provider.retry = Some(RetryConfig {
        max_attempts: 2,
        backoff: Duration::from_millis(10),
        cooldown: Duration::from_millis(50),
        total_timeout: Duration::ZERO,
        hold: Duration::ZERO,
    });
    provider
}

/// The prefix's 7 input + 5 output tokens priced at [`prefix_pricing()`]:
/// 7 × $0.000002 + 5 × $0.000005 = $0.000039.
fn prefix_billed_cost() -> Cost {
    Cost {
        input: Decimal::new(14, 6),
        output: Decimal::new(25, 6),
        cache_creation: Decimal::ZERO,
        cache_read: Decimal::ZERO,
        reasoning: Decimal::ZERO,
        total: Decimal::new(39, 6),
    }
}

/// Waits until the failed stream has settled, then snapshots everything
/// settled so far: one (usage-tokens, cost) pair per settle call.
async fn settled_billed(admission: &UsageRecordingAdmission) -> Vec<(Option<u64>, Option<Cost>)> {
    let wait = async {
        loop {
            let empty = admission
                .settled_usage
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty();
            if !empty {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), wait)
        .await
        .expect("the failed stream settles its reservation");
    let usage = admission
        .settled_usage
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let costs = admission
        .settled_costs
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    usage.into_iter().zip(costs).collect()
}

#[tokio::test]
async fn a_terminal_pre_commit_failure_charges_the_prefix_billed_cost() {
    // The prefix reported usage, the ladder retried, and the retry died with
    // a terminal 400: the terminal settle must charge both the stashed
    // tokens and the cost the provider bills for them — not just the tokens.
    let (app, calls) = prefix_usage_then_failure_upstream(NextCallFailure::TerminalBadRequest);
    let endpoint = spawn_upstream(app).await;
    let admission = Arc::new(UsageRecordingAdmission::default());
    let engine = engine_with(
        vec![priced_provider("fake", endpoint, prefix_pricing())],
        test_config(),
        Arc::new(RecordingBus::new()),
        Some(Arc::clone(&admission) as Arc<dyn Admission>),
        None,
    );

    let failed = engine
        .execute_stream(engine_request(true, "billed terminal", "caller-a", None))
        .await;
    assert!(failed.is_err(), "the stream never commits: {failed:?}");

    let settled = settled_billed(&admission).await;
    assert_eq!(
        settled,
        vec![(Some(12), Some(prefix_billed_cost()))],
        "the terminal settle must charge the prefix's usage (7 in + 5 out) and its \
         billed cost, got {settled:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one prefix attempt, then the terminal 400"
    );
}

#[tokio::test]
async fn an_exhausted_ladder_charges_the_last_reported_prefix_billed_cost() {
    // The prefix reported usage on the first attempt; the retry fails with a
    // transient 500 and no usage of its own, exhausting the ladder: the
    // exhausted settle must keep the stashed report (a report-less attempt
    // never erases one) and charge its billed cost.
    let (app, calls) = prefix_usage_then_failure_upstream(NextCallFailure::TransientServerError);
    let endpoint = spawn_upstream(app).await;
    let admission = Arc::new(UsageRecordingAdmission::default());
    let engine = engine_with(
        vec![priced_provider("fake", endpoint, prefix_pricing())],
        test_config(),
        Arc::new(RecordingBus::new()),
        Some(Arc::clone(&admission) as Arc<dyn Admission>),
        None,
    );

    let failed = engine
        .execute_stream(engine_request(true, "billed exhaustion", "caller-a", None))
        .await;
    assert!(failed.is_err(), "the stream never commits: {failed:?}");

    let settled = settled_billed(&admission).await;
    assert_eq!(
        settled,
        vec![(Some(12), Some(prefix_billed_cost()))],
        "the exhausted settle must charge the stashed prefix usage and its billed \
         cost, got {settled:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one prefix attempt, then the transient 500"
    );
}

#[tokio::test]
async fn a_decode_garbage_after_a_usage_frame_charges_the_prefix_billed_cost() {
    // The prefix reported usage on the first attempt; the retry answers 200
    // but its bytes cannot be framed, so the attempt dies with a decode
    // error and no usage of its own: the exhausted settle must still charge
    // the earlier prefix's usage and billed cost.
    let (app, calls) = prefix_usage_then_failure_upstream(NextCallFailure::FramingGarbage);
    let endpoint = spawn_upstream(app).await;
    let admission = Arc::new(UsageRecordingAdmission::default());
    let engine = engine_with(
        vec![priced_provider("fake", endpoint, prefix_pricing())],
        test_config(),
        Arc::new(RecordingBus::new()),
        Some(Arc::clone(&admission) as Arc<dyn Admission>),
        None,
    );

    let failed = engine
        .execute_stream(engine_request(true, "billed garbage", "caller-a", None))
        .await;
    assert!(failed.is_err(), "the stream never commits: {failed:?}");

    let settled = settled_billed(&admission).await;
    assert_eq!(
        settled,
        vec![(Some(12), Some(prefix_billed_cost()))],
        "the decode-failure settle must charge the earlier prefix's usage and its \
         billed cost, got {settled:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one prefix attempt, then the unframable 200"
    );
}

#[tokio::test]
async fn stop_with_charges_the_cost_of_the_member_that_reported_the_prefix() {
    // A pool whose first member reports prefix usage and then exhausts its
    // attempts; the second member is refused at rekey, so the reservation
    // settles in `stop_with` after the ladder ends. The charged cost must be
    // priced at the member that produced the usage — not whichever member
    // the ladder was last on.
    let (first_app, first_calls) =
        prefix_usage_then_failure_upstream(NextCallFailure::TransientServerError);
    let (second_app, second_calls) =
        prefix_usage_then_failure_upstream(NextCallFailure::TerminalBadRequest);
    let first_endpoint = spawn_upstream(first_app).await;
    let second_endpoint = spawn_upstream(second_app).await;
    let admission = Arc::new(UsageRecordingAdmission {
        refuse_rekey_for: Some("second".to_owned()),
        ..Default::default()
    });
    let registry = ProviderRegistry::from_parts(
        vec![
            priced_provider("first", first_endpoint, prefix_pricing()),
            priced_provider(
                "second",
                second_endpoint,
                ModelPricing {
                    input: Decimal::new(2, 4),
                    output: Decimal::new(5, 4),
                    ..Default::default()
                },
            ),
        ],
        vec![PoolConfig {
            name: "both".to_owned(),
            members: vec!["first".to_owned(), "second".to_owned()],
            retry: RetryConfig {
                max_attempts: 2,
                backoff: Duration::from_millis(10),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            affinity: Default::default(),
            health: Default::default(),
        }],
    )
    .expect("provider registry with pool");
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("engine builds");

    let failed = engine
        .execute_stream(engine_request_for(
            "both",
            true,
            "billed stop",
            "caller-a",
            None,
        ))
        .await;
    assert!(failed.is_err(), "the stream never commits: {failed:?}");

    let settled = settled_billed(&admission).await;
    assert_eq!(
        settled,
        vec![(Some(12), Some(prefix_billed_cost()))],
        "stop_with must charge the reporting member's usage and pricing, not the \
         refused member's, got {settled:?}"
    );
    assert_eq!(
        first_calls.load(Ordering::SeqCst),
        2,
        "both attempts ran on the reporting member"
    );
    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        0,
        "the refused member is never called"
    );
}

async fn collect_stream(stream: EngineStream) -> Vec<CoreEvent> {
    let mut all = stream.head;
    let mut tail = stream.tail;
    while let Some(event) = tail.next().await {
        all.push(event);
    }
    all
}

fn failed_rows(bus: &RecordingBus) -> Vec<String> {
    bus.snapshot()
        .into_iter()
        .filter_map(|event| match event {
            ProxyEvent::ResponseFailed(row) => Some(row.request_id),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_failed_streaming_first_attempt_releases_its_key() {
    // Done-when of the streaming idempotency claim: only a stream that runs
    // cleanly to its terminal is stored. A stream that dies mid-response
    // releases the key, so the keyed retry executes a fresh request instead
    // of replaying the failed one for the TTL.
    let calls = Arc::new(AtomicUsize::new(0));
    let endpoint = spawn_upstream(mid_stream_failure_upstream(
        Arc::clone(&calls),
        Duration::from_millis(40),
    ))
    .await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, Arc::clone(&bus));

    let first = engine
        .execute_stream(engine_request(
            true,
            "flaky prompt",
            "caller-a",
            Some("key-f"),
        ))
        .await
        .expect("the first attempt still hands back a committed stream");
    let events = collect_stream(first).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, CoreEvent::Error { .. })),
        "the first attempt fails in-band: {events:?}"
    );
    tokio::time::sleep(Duration::from_millis(30)).await;

    let retry = engine
        .execute_stream(engine_request(
            true,
            "flaky prompt",
            "caller-a",
            Some("key-f"),
        ))
        .await;
    match retry {
        Ok(_) => {}
        Err(EngineError::IdempotencyInFlight) => panic!("a failed stream must release its key"),
        Err(other) => panic!("the retry must execute as a fresh stream, got {other:?}"),
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the keyed retry must execute upstream again, not replay the failed stream"
    );
}

#[tokio::test]
async fn a_stream_joiner_completes_its_idempotency_claim_from_the_shared_outcome() {
    // A joiner holds its own claimed Idempotency-Key while it shares the
    // leader's stream (a different key: the leader's own is in flight and
    // would 409). When the leader ends cleanly, the joiner's key must store
    // that shared outcome, so a matching retry replays it instead of
    // re-executing — the same rule the non-streaming follower already
    // follows.
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&counted);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(120)).await;
                let chunks = vec![
                    bytes::Bytes::from_static(
                        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_j\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"shared\"}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    ),
                ];
                let body_stream =
                    futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
                (
                    StatusCode::OK,
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    )],
                    Body::from_stream(body_stream),
                )
                    .into_response()
            }
        }),
    );
    let endpoint = spawn_upstream(app).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_idempotent_engine(endpoint, bus);

    let leader = engine.execute_stream(engine_request(
        true,
        "joiner idem",
        "caller-a",
        Some("leader-key"),
    ));
    tokio::time::sleep(Duration::from_millis(30)).await;
    let joiner = engine.execute_stream(
        engine_request(true, "joiner idem", "caller-a", Some("joiner-key"))
            .with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;
    let leader_events = collect_stream(leader.expect("leader streams")).await;
    let joiner_events = collect_stream(joiner.expect("joiner joins the shared stream")).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        joiner_events
            .iter()
            .any(|event| matches!(event, CoreEvent::MessageStop { .. })),
        "the joiner's replay reaches the terminal: {joiner_events:?}"
    );
    assert_eq!(leader_events, joiner_events);

    let replay = engine
        .execute_stream(
            engine_request(true, "joiner idem", "caller-a", Some("joiner-key"))
                .with_request_id("req-retry"),
        )
        .await
        .expect("a retry under the joiner's key must not conflict");
    let replay_events = collect_stream(replay).await;
    assert_eq!(
        joiner_events, replay_events,
        "the retry replays the shared outcome the joiner stored"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the retry replayed from the joiner's claim; no second upstream call"
    );
}

/// An admission step that waits (so a joiner arrives while the leader holds
/// the flight) and then refuses, for the follower's error propagation.
#[derive(Debug)]
struct SlowRefusingAdmission;

#[async_trait]
impl Admission for SlowRefusingAdmission {
    async fn reserve(&self, _claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        tokio::time::sleep(Duration::from_millis(120)).await;
        Err(EngineError::RateLimited {
            message: "rpm window spent".to_owned(),
            retry_after: Some(Duration::from_secs(5)),
        })
    }

    fn settle(&self, _ticket: &AdmissionTicket, _outcome: &AdmissionOutcome<'_>) {}
}

#[tokio::test]
async fn an_admission_refusal_reaches_single_flight_joiners() {
    // The leader's admission error (429 here) is what its joiners get —
    // not a 500 from the leader-without-outcome backstop.
    let calls = Arc::new(AtomicUsize::new(0));
    let endpoint = spawn_upstream(CountingUpstream::new(Duration::ZERO).router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = engine_with(
        vec![provider(endpoint)],
        test_config(),
        Arc::clone(&bus),
        Some(Arc::new(SlowRefusingAdmission)),
        None,
    );

    let leader = engine.execute(engine_request(false, "shared prompt", "caller-a", None));
    tokio::time::sleep(Duration::from_millis(30)).await;
    let joiner = engine.execute(
        engine_request(false, "shared prompt", "caller-a", None).with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;

    let leader_err = leader.expect_err("the leader is refused");
    let joiner_err = joiner.expect_err("the joiner is refused the same way");
    assert!(
        matches!(joiner_err, EngineError::RateLimited { .. }),
        "the joiner must see the leader's admission refusal, got {joiner_err:?}"
    );
    assert_eq!(joiner_err.status_code(), 429);
    assert_eq!(joiner_err.kind_name(), leader_err.kind_name());
    assert_eq!(calls.load(Ordering::SeqCst), 0, "nothing reached upstream");
}

#[tokio::test]
async fn a_single_flight_follower_reexecutes_when_its_leader_vanishes() {
    // The leader's handler future dropped mid-flight (its client went
    // away): the joiner runs the request itself instead of failing with an
    // internal error.
    let upstream = CountingUpstream::new(Duration::from_millis(150));
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let leader_engine = engine.clone();
    let leader = tokio::spawn(async move {
        leader_engine
            .execute(engine_request(false, "shared prompt", "caller-a", None))
            .await
    });
    // The joiner starts while the leader holds the flight; it cannot finish
    // before the leader does, so it is parked on the shared outcome here.
    let joiner = engine.execute(
        engine_request(false, "shared prompt", "caller-a", None).with_request_id("req-joiner"),
    );
    tokio::time::sleep(Duration::from_millis(40)).await;
    leader.abort();

    let response = joiner
        .await
        .expect("the joiner answers by running the request itself");
    assert_eq!(response_text(&response), "hello from upstream");
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        2,
        "the vanished leader's upstream call plus the joiner's own re-execution"
    );
}

#[tokio::test]
async fn a_streaming_joiner_gets_the_leaders_pre_stream_error_not_a_broken_200() {
    // The leader fails before its first byte (a terminal upstream 400): the
    // joiner fails with the same error — not a 200 whose stream then ends in
    // a generic in-band failure.
    let calls = Arc::new(AtomicUsize::new(0));
    let endpoint = spawn_upstream(delayed_400_upstream(
        Arc::clone(&calls),
        Duration::from_millis(120),
    ))
    .await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let leader = engine.execute_stream(engine_request(true, "shared stream", "caller-a", None));
    let joiner = engine.execute_stream(
        engine_request(true, "shared stream", "caller-a", None).with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;

    let leader_err = leader.expect_err("the leader's stream fails pre-commit");
    let joiner_err = joiner.expect_err("the joiner must fail the same way, not stream a 200");
    assert_eq!(
        joiner_err.kind_name(),
        leader_err.kind_name(),
        "the joiner sees the leader's error ({leader_err:?}), got {joiner_err:?}"
    );
    assert_eq!(joiner_err.status_code(), 400);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        failed_rows(&bus).contains(&"req-joiner".to_owned()),
        "the joiner's ledger row records the failure: {:?}",
        failed_rows(&bus)
    );
    assert!(
        !completed_rows(&bus).iter().any(|(hit, _)| *hit),
        "no joiner row may claim a served copy for a failed shared stream"
    );
}

#[tokio::test]
async fn a_streaming_joiner_of_a_failed_shared_stream_records_a_failed_row() {
    // The leader committed, then its stream died in-band: the joiner's
    // replay ends in the same error and its ledger row is a failure — not a
    // completed served copy with synthetic zero usage.
    let calls = Arc::new(AtomicUsize::new(0));
    let endpoint = spawn_upstream(mid_stream_failure_upstream(
        Arc::clone(&calls),
        Duration::from_millis(120),
    ))
    .await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let leader = engine.execute_stream(engine_request(true, "shared stream", "caller-a", None));
    let joiner = engine.execute_stream(
        engine_request(true, "shared stream", "caller-a", None).with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;
    let leader = leader.expect("the leader committed before failing");
    let joiner = joiner.expect("the joiner still gets the committed stream");

    let leader_events = collect_stream(leader).await;
    let joiner_events = collect_stream(joiner).await;
    assert!(
        joiner_events
            .iter()
            .any(|event| matches!(event, CoreEvent::Error { .. })),
        "the joiner's replay ends in the leader's error: {joiner_events:?}"
    );
    assert_eq!(joiner_events, leader_events);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let failed = failed_rows(&bus);
    assert!(
        failed.contains(&"req-joiner".to_owned()),
        "the joiner records a failure row, got completed {:?} and failed {failed:?}",
        completed_rows(&bus)
    );
    assert!(
        !completed_rows(&bus)
            .iter()
            .any(|(hit, id)| *hit && id == "req-joiner"),
        "the failed shared stream must not record the joiner as a completed hit"
    );
}

#[tokio::test]
async fn served_copies_record_no_cost() {
    // Plan 05 step 6: cache hits charge nothing. The row keeps the stored
    // usage for the record, but the cost lives only on the request that
    // made the upstream call.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let mut priced = provider(endpoint);
    priced.pricing.insert(
        ModelId::new("claude-sonnet-4-6"),
        ModelPricing {
            input: Decimal::from_str_exact("0.000003").expect("price"),
            output: Decimal::from_str_exact("0.000015").expect("price"),
            ..ModelPricing::default()
        },
    );
    let cache = MemoryCache::new(Duration::from_secs(60), 16);
    let engine = engine_with(
        vec![priced],
        test_config(),
        Arc::clone(&bus),
        None,
        Some(cache),
    );

    engine
        .execute(engine_request(false, "priced prompt", "caller-a", None))
        .await
        .expect("first runs and is priced");
    engine
        .execute(engine_request(false, "priced prompt", "caller-a", None))
        .await
        .expect("second is served from the cache");

    let rows: Vec<(bool, Option<_>)> = bus
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            ProxyEvent::ResponseCompleted(row) => Some((row.cache_hit, row.cost)),
            _ => None,
        })
        .collect();
    assert_eq!(rows.len(), 2);
    let charged = rows.iter().find(|(hit, _)| !hit);
    let served = rows.iter().find(|(hit, _)| *hit);
    assert!(
        charged.is_some_and(|(_, cost)| cost.is_some()),
        "the request that made the upstream call carries its cost: {rows:?}"
    );
    assert!(
        served.is_some_and(|(_, cost)| cost.is_none()),
        "the served copy charges nothing: {rows:?}"
    );
}

#[tokio::test]
async fn a_key_reused_across_providers_is_a_client_error() {
    // Body bytes alone do not say which provider served them: the same
    // caller, key and body against a second provider is a mismatch, not a
    // replay of the first provider's response under the second's name.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = engine_with(
        vec![
            provider_named("fake", endpoint.clone()),
            provider_named("other", endpoint),
        ],
        idempotency_config(),
        Arc::clone(&bus),
        None,
        None,
    );

    engine
        .execute(engine_request_for(
            "fake",
            false,
            "same body",
            "caller-a",
            Some("key-x"),
        ))
        .await
        .expect("first completes under the fake provider");
    let error = engine
        .execute(engine_request_for(
            "other",
            false,
            "same body",
            "caller-a",
            Some("key-x"),
        ))
        .await
        .expect_err("the same key under another provider is a mismatch");
    assert!(matches!(error, EngineError::InvalidRequest(_)));
    assert_eq!(error.status_code(), 400);
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        1,
        "the mismatched retry must not execute"
    );
}

/// A pre-call guard that waits (so a joiner attaches to the leader's share)
/// and then rejects the request, for the streaming refusal propagation.
#[derive(Debug)]
struct SlowRejectingGuard;

#[async_trait]
impl llm_proxy_engine::Guard for SlowRejectingGuard {
    async fn inspect_request(
        &self,
        _request: &CoreRequest,
    ) -> Result<llm_proxy_engine::GuardDecision, EngineError> {
        tokio::time::sleep(Duration::from_millis(120)).await;
        Ok(llm_proxy_engine::GuardDecision::Reject(
            "prompt blocked by policy".to_owned(),
        ))
    }
}

#[tokio::test]
async fn a_streaming_leaders_admission_refusal_reaches_its_joiners() {
    // Streaming twin of an_admission_refusal_reaches_single_flight_joiners:
    // the leader's admission error (429 here) is what its joiners get before
    // any 200 is sent — not a 500 from the lead drop backstop.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = engine_with(
        vec![provider(endpoint)],
        test_config(),
        Arc::clone(&bus),
        Some(Arc::new(SlowRefusingAdmission)),
        None,
    );

    let leader = engine.execute_stream(engine_request(true, "shared stream", "caller-a", None));
    let joiner = engine.execute_stream(
        engine_request(true, "shared stream", "caller-a", None).with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;

    let leader_err = leader.expect_err("the leader is refused");
    let joiner_err = joiner.expect_err("the joiner is refused the same way");
    assert!(
        matches!(joiner_err, EngineError::RateLimited { .. }),
        "the joiner must see the leader's admission refusal, got {joiner_err:?}"
    );
    assert_eq!(joiner_err.status_code(), 429);
    assert_eq!(joiner_err.kind_name(), leader_err.kind_name());
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        0,
        "nothing reached upstream"
    );
    assert!(
        failed_rows(&bus).contains(&"req-joiner".to_owned()),
        "the joiner's ledger row records the refusal: {:?}",
        failed_rows(&bus)
    );
}

#[tokio::test]
async fn a_streaming_leaders_guard_rejection_reaches_its_joiners() {
    // Same rule for a pre-call guard rejection: the joiner gets the leader's
    // 400 with its message, not an internal error.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let registry = ProviderRegistry::from_providers(vec![provider(endpoint)]).expect("registry");
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::clone(&bus) as Arc<dyn EventBus>)
        .guard(Arc::new(SlowRejectingGuard))
        .build()
        .expect("engine builds");

    let leader = engine.execute_stream(engine_request(true, "shared stream", "caller-a", None));
    let joiner = engine.execute_stream(
        engine_request(true, "shared stream", "caller-a", None).with_request_id("req-joiner"),
    );
    let (leader, joiner) = futures::future::join(leader, joiner).await;

    let leader_err = leader.expect_err("the leader is rejected by the guard");
    let joiner_err = joiner.expect_err("the joiner is rejected the same way");
    assert!(
        matches!(&joiner_err, EngineError::InvalidRequest(message) if message.contains("prompt blocked")),
        "the joiner must see the leader's guard rejection, got {joiner_err:?}"
    );
    assert_eq!(joiner_err.status_code(), 400);
    assert_eq!(joiner_err.kind_name(), leader_err.kind_name());
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_streaming_follower_reruns_when_its_leader_vanishes_before_the_first_byte() {
    // Streaming twin of a_single_flight_follower_reexecutes_when_its_leader_
    // vanishes: the leader's caller disconnects while the joiner — whose own
    // client is connected — waits on the shared first byte. The joiner re-runs
    // the request itself instead of failing with an internal error.
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&counted);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(150)).await;
                let chunks = vec![
                    bytes::Bytes::from_static(
                        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_v\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    ),
                ];
                let body_stream =
                    futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
                (
                    StatusCode::OK,
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    )],
                    Body::from_stream(body_stream),
                )
                    .into_response()
            }
        }),
    );
    let endpoint = spawn_upstream(app).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(endpoint, Arc::clone(&bus), None);

    let leader_engine = engine.clone();
    let leader = tokio::spawn(async move {
        leader_engine
            .execute_stream(engine_request(true, "shared stream", "caller-a", None))
            .await
    });
    // The joiner starts while the leader holds the share; it cannot get a
    // first byte before the leader does, so it is parked on the shared
    // commit wait when the leader goes away.
    let joiner = engine.execute_stream(
        engine_request(true, "shared stream", "caller-a", None).with_request_id("req-joiner"),
    );
    tokio::time::sleep(Duration::from_millis(40)).await;
    leader.abort();

    let stream = joiner
        .await
        .expect("the joiner answers by running the request itself");
    let events = collect_stream(stream).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, CoreEvent::MessageStop { .. })),
        "the re-run stream completes: {events:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "the vanished leader's upstream call plus the joiner's own re-run"
    );
    assert!(
        !failed_rows(&bus).contains(&"req-joiner".to_owned()),
        "the vanished leader is not the joiner's failure; the joiner's own run records its row"
    );
    assert!(
        completed_rows(&bus)
            .iter()
            .any(|(_, id)| id == "req-joiner"),
        "the joiner's re-run records a completed row: {:?}",
        completed_rows(&bus)
    );
}

/// An admission step whose slot-2 verdict flips at runtime: the shape of a
/// key whose allowlist was tightened after an answer was already stored for
/// it. `reserve` always succeeds, so a refusal can only come from
/// `authorize`.
#[derive(Debug)]
struct ToggleAdmission {
    forbid: Arc<AtomicBool>,
}

#[async_trait]
impl Admission for ToggleAdmission {
    async fn authorize(&self, claim: &AdmissionClaim<'_>) -> Result<(), EngineError> {
        if self.forbid.load(Ordering::SeqCst) {
            return Err(EngineError::Forbidden(format!(
                "this key may not use model \"{}\"",
                claim.requested_model
            )));
        }
        Ok(())
    }

    async fn reserve(&self, _claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, _outcome: &AdmissionOutcome<'_>) {}
}

/// A [`ToggleAdmission`] and the flag that restricts its key.
fn toggle_admission() -> (Arc<AtomicBool>, Arc<dyn Admission>) {
    let forbid = Arc::new(AtomicBool::new(false));
    let admission = Arc::new(ToggleAdmission {
        forbid: Arc::clone(&forbid),
    });
    (forbid, admission)
}

/// A streaming upstream that counts calls and runs cleanly to its terminal,
/// so the stream's outcome is stored under its idempotency key.
fn completing_stream_upstream(calls: Arc<AtomicUsize>) -> Router {
    Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&calls);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let chunks = vec![
                    bytes::Bytes::from_static(
                        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_t\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"stored\"}}\n\n",
                    ),
                    bytes::Bytes::from_static(
                        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    ),
                ];
                let body_stream =
                    futures::stream::iter(chunks).map(Ok::<_, std::convert::Infallible>);
                (
                    StatusCode::OK,
                    [(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("text/event-stream"),
                    )],
                    Body::from_stream(body_stream),
                )
                    .into_response()
            }
        }),
    )
}

#[tokio::test]
async fn a_tightened_allowlist_beats_a_cached_answer() {
    // The allowlist verdict is slot 2, ahead of the cache lookup (slot 5):
    // the cache outlives the rules it was filled under, so a key restricted
    // after its answer was stored is refused, not served the stored answer.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let (forbid, admission) = toggle_admission();
    let engine = engine_with(
        vec![provider(endpoint)],
        test_config(),
        Arc::clone(&bus),
        Some(admission),
        Some(MemoryCache::new(Duration::from_secs(60), 16)),
    );

    engine
        .execute(
            engine_request(false, "restricted prompt", "caller-a", None).with_request_id("req-1"),
        )
        .await
        .expect("the first request runs while the key is still allowed");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 1);

    forbid.store(true, Ordering::SeqCst);

    let refused = engine
        .execute(
            engine_request(false, "restricted prompt", "caller-a", None).with_request_id("req-2"),
        )
        .await
        .expect_err("a restricted key must not be served the answer stored before the restriction");
    assert!(
        matches!(refused, EngineError::Forbidden(_)),
        "got {refused:?}"
    );
    assert_eq!(refused.status_code(), 403);
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        1,
        "the refused request never reaches the upstream"
    );
    assert!(
        failed_rows(&bus).contains(&"req-2".to_owned()),
        "the refusal is recorded on the request's ledger row: {:?}",
        failed_rows(&bus)
    );
    assert!(
        !completed_rows(&bus).iter().any(|(hit, _)| *hit),
        "no row may record a served copy for the restricted key: {:?}",
        completed_rows(&bus)
    );
}

#[tokio::test]
async fn a_tightened_allowlist_beats_an_idempotent_replay() {
    // Slot 2 is also ahead of the idempotency claim (slot 4): a matching
    // retry of a completed request replays the stored outcome — unless the
    // key has been restricted since.
    let upstream = CountingUpstream::new(Duration::ZERO);
    let endpoint = spawn_upstream(upstream.router()).await;
    let bus = Arc::new(RecordingBus::new());
    let (forbid, admission) = toggle_admission();
    let engine = engine_with(
        vec![provider(endpoint)],
        idempotency_config(),
        Arc::clone(&bus),
        Some(admission),
        None,
    );

    engine
        .execute(
            engine_request(false, "idem restricted", "caller-a", Some("key-1"))
                .with_request_id("req-1"),
        )
        .await
        .expect("the first request runs while the key is still allowed");
    assert_eq!(upstream.calls.load(Ordering::SeqCst), 1);

    forbid.store(true, Ordering::SeqCst);

    let refused = engine
        .execute(
            engine_request(false, "idem restricted", "caller-a", Some("key-1"))
                .with_request_id("req-2"),
        )
        .await
        .expect_err(
            "a restricted key must not be replayed the outcome stored before the restriction",
        );
    assert!(
        matches!(refused, EngineError::Forbidden(_)),
        "got {refused:?}"
    );
    assert_eq!(refused.status_code(), 403);
    assert_eq!(
        upstream.calls.load(Ordering::SeqCst),
        1,
        "the refused retry never reaches the upstream"
    );
    assert!(
        failed_rows(&bus).contains(&"req-2".to_owned()),
        "the refusal is recorded on the retry's ledger row: {:?}",
        failed_rows(&bus)
    );
    assert!(
        !completed_rows(&bus).iter().any(|(hit, _)| *hit),
        "no row may record a replay for the restricted key: {:?}",
        completed_rows(&bus)
    );
}

#[tokio::test]
async fn a_tightened_allowlist_beats_a_cached_answer_on_the_streaming_path() {
    // Streaming twin. The stored answer a stream can be served is its
    // idempotent replay (the response cache holds only complete
    // non-streaming responses); slot 2 is ahead of it on this path too.
    let calls = Arc::new(AtomicUsize::new(0));
    let endpoint = spawn_upstream(completing_stream_upstream(Arc::clone(&calls))).await;
    let bus = Arc::new(RecordingBus::new());
    let (forbid, admission) = toggle_admission();
    let engine = engine_with(
        vec![provider(endpoint)],
        idempotency_config(),
        Arc::clone(&bus),
        Some(admission),
        None,
    );

    let first = engine
        .execute_stream(
            engine_request(true, "stream restricted", "caller-a", Some("key-s"))
                .with_request_id("req-1"),
        )
        .await
        .expect("the first stream runs while the key is still allowed");
    let events = collect_stream(first).await;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, CoreEvent::MessageStop { .. })),
        "the first stream runs to its terminal, so its outcome is stored: {events:?}"
    );
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    forbid.store(true, Ordering::SeqCst);

    let refused = engine
        .execute_stream(
            engine_request(true, "stream restricted", "caller-a", Some("key-s"))
                .with_request_id("req-2"),
        )
        .await
        .expect_err(
            "a restricted key must not be replayed the stream stored before the restriction",
        );
    assert!(
        matches!(refused, EngineError::Forbidden(_)),
        "got {refused:?}"
    );
    assert_eq!(refused.status_code(), 403);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the refused retry never reaches the upstream"
    );
    assert!(
        failed_rows(&bus).contains(&"req-2".to_owned()),
        "the refusal is recorded on the retry's ledger row: {:?}",
        failed_rows(&bus)
    );
    assert!(
        !completed_rows(&bus).iter().any(|(hit, _)| *hit),
        "no row may record a replay for the restricted key: {:?}",
        completed_rows(&bus)
    );
}
