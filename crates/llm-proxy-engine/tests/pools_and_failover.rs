//! Plan 04 behaviour tests against loopback fake upstreams: pools, the
//! retry ladder, pre-output failover, hold budgets, credential pools,
//! session affinity and the exhaustion signals — through the same engine
//! the server and the SDK run.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::post;
use futures::StreamExt;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_core::{
    AffinityConfig, AffinityConflictPolicy, AppConfig, AuthStyle, PoolConfig,
    ProviderAdapterConfig, ProviderConfig, ProviderRegistry, ProviderRouteKind,
    ProviderRoutesConfig, RetryConfig, ServerConfig, UpstreamTimeoutsConfig,
};
use llm_proxy_engine::{
    ClientDialect, CooldownStore, Cooldowns, Engine, EngineBuilder, EngineError, EngineRequest,
    Guard, GuardDecision,
};
use llm_proxy_protocol::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreResponse, CoreRole, ModelRef,
    SamplingOptions,
};
use llm_proxy_storage::{
    AttemptCompleted, AttemptOutcome, EventBus, ProxyEvent, RecordingBus, RequestReceived,
};
use secrecy::SecretString;

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

const ANTHROPIC_OK_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello from upstream"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#;

fn upstream_json(status: StatusCode, body: &'static str) -> Router {
    Router::new().route(
        "/{*path}",
        post(move || async move { (status, [(header::CONTENT_TYPE, "application/json")], body) }),
    )
}

/// An upstream that always answers 429, advertising the given `Retry-After`.
fn throttled_upstream_with_retry_after(retry_after: &'static str) -> Router {
    Router::new().route(
        "/{*path}",
        post(move || async move {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [
                    (header::CONTENT_TYPE, "application/json"),
                    (header::RETRY_AFTER, retry_after),
                ],
                r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
            )
        }),
    )
}

/// A 429 whose only reset signal is an Anthropic RFC 3339 reset timestamp —
/// no `Retry-After` — the shape a real Anthropic throttle answers with.
fn throttled_upstream_with_reset_header(reset: String) -> Router {
    Router::new().route(
        "/{*path}",
        post(move || async move {
            let mut response = (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
            )
                .into_response();
            response.headers_mut().insert(
                axum::http::HeaderName::from_static("anthropic-ratelimit-requests-reset"),
                axum::http::HeaderValue::from_str(&reset).expect("valid reset header value"),
            );
            response
        }),
    )
}

fn upstream_ok_stream() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#),
                Event::default()
                    .event("message_delta")
                    .data(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#),
                Event::default()
                    .event("message_stop")
                    .data(r#"{"type":"message_stop"}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Sse::new(stream)
                    .keep_alive(KeepAlive::default())
                    .into_response(),
            )
        }),
    )
}

/// A 200 whose first SSE frame is an error event (backlog 2): the stream
/// rejects in-band before any output.
fn upstream_stream_leading_error() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let events = vec![Event::default().event("error").data(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#,
            )];
            let stream =
                futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Sse::new(stream).into_response(),
            )
        }),
    )
}

/// A stream that starts, then ends without a terminal event.
fn upstream_stream_truncated() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_t","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Sse::new(stream).into_response(),
            )
        }),
    )
}

/// A stream that opens, then stays silent forever: headers arrive, no
/// event ever does.
fn upstream_silent_stream() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let never = futures::stream::pending::<Result<Event, std::convert::Infallible>>();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Sse::new(never).into_response(),
            )
        }),
    )
}

/// A stream that closes without ever emitting an event.
fn upstream_empty_stream() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let stream = futures::stream::empty::<Result<Event, std::convert::Infallible>>();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Sse::new(stream).into_response(),
            )
        }),
    )
}

/// An upstream that only answers when the right key arrives; 401 otherwise.
fn upstream_key_check(
    expected_key: &'static str,
    seen: Arc<std::sync::Mutex<Vec<String>>>,
) -> Router {
    upstream_key_check_tagged(expected_key, "upstream", seen)
}

/// [`upstream_key_check`] whose success body names the member (`from-{tag}`).
fn upstream_key_check_tagged(
    expected_key: &'static str,
    tag: &'static str,
    seen: Arc<std::sync::Mutex<Vec<String>>>,
) -> Router {
    Router::new().route(
        "/{*path}",
        post(move |headers: axum::http::HeaderMap| async move {
            let got = headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            seen.lock().expect("seen lock").push(got.clone());
            if got == expected_key {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    tagged_ok_body(tag),
                )
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"type":"error","error":{"type":"authentication_error","message":"bad key"}}"#
                        .to_owned(),
                )
            }
        }),
    )
}

/// A successful Anthropic body whose text names the member that served it.
fn tagged_ok_body(tag: &str) -> String {
    format!(
        r#"{{"id":"msg_{tag}","type":"message","role":"assistant","content":[{{"type":"text","text":"from-{tag}"}}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{{"input_tokens":10,"output_tokens":4}}}}"#
    )
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
    }
}

/// Test aid: a JSON tuple response that also carries `Retry-After`.
trait IntoResponseWithRetryAfter {
    fn into_response_with_retry_after(self, retry_after: &'static str) -> axum::response::Response;
}

impl IntoResponseWithRetryAfter
    for (
        StatusCode,
        [(axum::http::HeaderName, &'static str); 1],
        &'static str,
    )
{
    fn into_response_with_retry_after(self, retry_after: &'static str) -> axum::response::Response {
        let (status, headers, body) = self;
        let mut response = (status, headers, body).into_response();
        response.headers_mut().insert(
            header::RETRY_AFTER,
            axum::http::HeaderValue::from_static(retry_after),
        );
        response
    }
}

fn member(name: &str, endpoint: String, api_keys: Option<Vec<&str>>) -> ProviderConfig {
    let adapters = std::collections::HashMap::from([(
        "messages".to_owned(),
        ProviderAdapterConfig {
            protocol: "anthropic_messages".to_owned(),
            endpoint,
            headers: Arc::new(std::collections::HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    )]);
    ProviderConfig {
        name: name.to_owned(),
        api_key: SecretString::from("base-key"),
        api_keys: api_keys.map(|keys| {
            keys.into_iter()
                .map(|k| SecretString::from(k.to_owned()))
                .collect()
        }),
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        retry: None,
        egress: None,
        adapters,
        routes: ProviderRoutesConfig {
            messages: Some("messages".to_owned()),
            chat_completions: None,
            ..Default::default()
        },
        model_aliases: std::collections::HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn pool(
    name: &str,
    members: Vec<&str>,
    retry: RetryConfig,
    affinity: AffinityConfig,
) -> PoolConfig {
    PoolConfig {
        name: name.to_owned(),
        members: members.into_iter().map(str::to_owned).collect(),
        retry,
        affinity,
        health: Default::default(),
    }
}

fn ladder_retry() -> RetryConfig {
    RetryConfig {
        max_attempts: 2,
        backoff: Duration::from_millis(10),
        cooldown: Duration::from_millis(50),
        total_timeout: Duration::ZERO,
        hold: Duration::ZERO,
    }
}

fn build_engine_with(
    config: AppConfig,
    providers: Vec<ProviderConfig>,
    pools: Vec<PoolConfig>,
) -> (Engine, Arc<RecordingBus>) {
    let registry = ProviderRegistry::from_parts(providers, pools).expect("registry with pools");
    let bus = Arc::new(RecordingBus::new());
    let engine = EngineBuilder::new(config)
        .providers(Arc::new(registry))
        .event_bus(Arc::<RecordingBus>::clone(&bus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");
    (engine, bus)
}

fn build_engine(
    providers: Vec<ProviderConfig>,
    pools: Vec<PoolConfig>,
) -> (Engine, Arc<RecordingBus>) {
    build_engine_with(test_config(), providers, pools)
}

fn core_request(stream: bool) -> CoreRequest {
    CoreRequest {
        model: ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        messages: vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hello".to_owned(),
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

fn engine_request(route: &str, stream: bool) -> EngineRequest {
    EngineRequest::new(
        route,
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core_request(stream),
        llm_proxy_engine::InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id("req-plan04")
    .with_body_hash("hash")
}

fn attempts(bus: &RecordingBus) -> Vec<AttemptCompleted> {
    bus.snapshot()
        .into_iter()
        .filter_map(|event| match event {
            ProxyEvent::AttemptCompleted(attempt) => Some(attempt),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn pool_survives_a_429_from_its_first_member_and_the_ledger_shows_every_attempt() {
    let primary = spawn_upstream(throttled_upstream_with_retry_after("0")).await;
    let secondary = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let (engine, bus) = build_engine(
        vec![
            member("primary", primary, None),
            member("secondary", secondary, None),
        ],
        vec![pool(
            "both",
            vec!["primary", "secondary"],
            ladder_retry(),
            Default::default(),
        )],
    );

    let response = engine
        .execute(engine_request("both", false))
        .await
        .expect("the pool fails over past the 429");
    assert_eq!(response.core.usage.input_tokens, 10);

    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        3,
        "two attempts against the first member (max_attempts=2), then the failover"
    );
    assert_eq!(rows[0].provider, "primary");
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[0].http_status, Some(429));
    assert_eq!(rows[0].route, "both");
    assert_eq!(rows[0].attempt, 0);
    assert_eq!(rows[1].provider, "primary");
    assert_eq!(rows[1].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[1].attempt, 1);
    assert_eq!(rows[2].provider, "secondary");
    assert_eq!(rows[2].outcome, AttemptOutcome::Succeeded);
    assert_eq!(rows[2].attempt, 2);
    assert!(
        bus.snapshot()
            .iter()
            .any(|event| matches!(event, ProxyEvent::ResponseCompleted(_))),
        "the successful attempt completes the request row"
    );
}

#[tokio::test]
async fn pool_survives_a_5xx_from_its_first_member_and_the_ledger_shows_every_attempt() {
    let broken = spawn_upstream(upstream_json(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":"boom"}"#,
    ))
    .await;
    let good = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let (engine, bus) = build_engine(
        vec![member("broken", broken, None), member("good", good, None)],
        vec![pool(
            "pair",
            vec!["broken", "good"],
            ladder_retry(),
            Default::default(),
        )],
    );

    let response = engine
        .execute(engine_request("pair", false))
        .await
        .expect("the pool fails over past the 500");
    assert_eq!(response.core.usage.input_tokens, 10);

    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        3,
        "two attempts against the broken member (max_attempts=2), then the failover"
    );
    assert_eq!(rows[0].provider, "broken");
    assert_eq!(rows[0].attempt, 0);
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[0].http_status, Some(500));
    assert_eq!(rows[1].provider, "broken");
    assert_eq!(rows[1].attempt, 1);
    assert_eq!(rows[1].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[1].http_status, Some(500));
    assert_eq!(rows[2].provider, "good");
    assert_eq!(rows[2].attempt, 2);
    assert_eq!(rows[2].outcome, AttemptOutcome::Succeeded);
    assert_eq!(rows[2].http_status, None);
    assert!(
        rows.iter().all(|row| row.route == "pair"),
        "every attempt row names the pool route"
    );
}

#[tokio::test]
async fn pool_survives_a_connect_error_from_its_first_member_and_the_ledger_shows_every_attempt() {
    // Bind a listener, take its port, drop it: a reliably closed port whose
    // connect is refused immediately (never a hard-coded port).
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a throwaway listener");
    let closed_port = listener.local_addr().expect("listener addr").port();
    drop(listener);
    let dead = format!("http://127.0.0.1:{closed_port}/v1/messages");
    let good = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let (engine, bus) = build_engine(
        vec![member("dead", dead, None), member("good", good, None)],
        vec![pool(
            "pair",
            vec!["dead", "good"],
            ladder_retry(),
            Default::default(),
        )],
    );

    let response = engine
        .execute(engine_request("pair", false))
        .await
        .expect("the pool fails over past the connect error");
    assert_eq!(response.core.usage.input_tokens, 10);

    // A refused connect is a transport failure, not an `Api` error and not a
    // timeout: `map_provider_error` surfaces it as `Upstream` with the
    // generic 502 on every failed attempt row.
    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        3,
        "two connect errors against the dead member (max_attempts=2), then the failover"
    );
    assert_eq!(rows[0].provider, "dead");
    assert_eq!(rows[0].attempt, 0);
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[0].http_status, Some(502));
    assert_eq!(rows[0].error_kind.as_deref(), Some("Upstream"));
    assert_eq!(rows[1].provider, "dead");
    assert_eq!(rows[1].attempt, 1);
    assert_eq!(rows[1].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[1].http_status, Some(502));
    assert_eq!(rows[1].error_kind.as_deref(), Some("Upstream"));
    assert_eq!(rows[2].provider, "good");
    assert_eq!(rows[2].attempt, 2);
    assert_eq!(rows[2].outcome, AttemptOutcome::Succeeded);
    assert_eq!(rows[2].http_status, None);
    assert!(
        rows.iter().all(|row| row.route == "pair"),
        "every attempt row names the pool route"
    );
}

#[tokio::test]
async fn a_direct_provider_route_never_fails_over_or_wraps_the_error() {
    let provider_upstream = spawn_upstream(throttled_upstream_with_retry_after("0")).await;
    let (engine, bus) = build_engine(vec![member("solo", provider_upstream, None)], vec![]);

    let error = engine
        .execute(engine_request("solo", false))
        .await
        .expect_err("the 429 surfaces to the client");
    assert!(
        matches!(error, EngineError::Upstream { status: 429, .. }),
        "a direct provider route returns the raw 429, got {error:?}"
    );
    assert_eq!(error.status_code(), 429);
    assert_eq!(attempts(&bus).len(), 1, "exactly one attempt, no ladder");
}

#[tokio::test]
async fn an_exhausted_pool_reports_retries_exhausted_with_a_computed_retry_after() {
    let first = spawn_upstream(throttled_upstream_with_retry_after("17")).await;
    let second = spawn_upstream(throttled_upstream_with_retry_after("23")).await;
    let (engine, _bus) = build_engine(
        vec![member("first", first, None), member("second", second, None)],
        vec![pool(
            "limited",
            vec!["first", "second"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    let error = engine
        .execute(engine_request("limited", false))
        .await
        .expect_err("both members are rate limited");
    let EngineError::RetryExhausted {
        attempts,
        retry_after,
        ..
    } = &error
    else {
        panic!("expected RetryExhausted, got {error:?}");
    };
    assert_eq!(*attempts, 2);
    assert_eq!(
        *retry_after,
        Some(Duration::from_secs(23)),
        "the longest advertised reset wins"
    );
    assert_eq!(error.status_code(), 429);
    assert!(!error.should_retry(), "the client is told not to retry");
}

#[tokio::test]
async fn a_stream_that_fails_before_commit_fails_over() {
    let bad = spawn_upstream(upstream_stream_leading_error()).await;
    let good = spawn_upstream(upstream_ok_stream()).await;
    let (engine, bus) = build_engine(
        vec![member("bad", bad, None), member("good", good, None)],
        vec![pool(
            "streamed",
            vec!["bad", "good"],
            ladder_retry(),
            Default::default(),
        )],
    );

    let stream = engine
        .execute_stream(engine_request("streamed", true))
        .await
        .expect("failover happens before any byte is committed");
    assert!(
        !stream.head.is_empty(),
        "the head comes from the second member"
    );
    let mut all = stream.head.clone();
    let mut tail = stream.tail;
    while let Some(event) = tail.next().await {
        all.push(event);
    }
    assert!(
        all.iter()
            .any(|event| matches!(event, CoreEvent::MessageStop { .. })),
        "the good member's terminal arrives"
    );

    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        3,
        "two failed attempts on the bad member, then the commit"
    );
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[1].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[2].outcome, AttemptOutcome::Streaming);
    assert_eq!(rows[2].provider, "good");
}

#[tokio::test]
async fn a_stream_that_fails_after_commit_ends_with_an_in_band_error_and_a_failed_row() {
    let bad = spawn_upstream(upstream_stream_truncated()).await;
    let good = spawn_upstream(upstream_ok_stream()).await;
    let (engine, bus) = build_engine(
        vec![member("bad", bad, None), member("good", good, None)],
        vec![pool(
            "streamed",
            vec!["bad", "good"],
            ladder_retry(),
            Default::default(),
        )],
    );

    let stream = engine
        .execute_stream(engine_request("streamed", true))
        .await
        .expect("the first member's prefix commits");
    assert!(!stream.head.is_empty());
    let mut all = stream.head.clone();
    let mut tail = stream.tail;
    while let Some(event) = tail.next().await {
        all.push(event);
    }
    assert!(
        all.iter()
            .any(|event| matches!(event, CoreEvent::Error { .. })),
        "the truncation is an in-band error, got {all:?}"
    );

    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        bus.snapshot()
            .iter()
            .any(|event| matches!(event, ProxyEvent::ResponseFailed(_))),
        "a committed stream that dies records a failed ledger row"
    );
}

#[tokio::test]
async fn the_hold_budget_commits_a_silent_stream_and_covers_the_gap() {
    // The upstream opens and stays silent forever: with a hold budget, the
    // engine commits (empty head) instead of holding the HTTP response
    // hostage — from there heartbeats cover the silence.
    let silent = spawn_upstream(upstream_silent_stream()).await;
    let (engine, bus) = build_engine(
        vec![member("silent", silent, None)],
        vec![pool(
            "held",
            vec!["silent"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::from_millis(300),
            },
            Default::default(),
        )],
    );

    let started = std::time::Instant::now();
    let stream = engine
        .execute_stream(engine_request("held", true))
        .await
        .expect("the hold budget commits the stream");
    assert!(
        stream.head.is_empty(),
        "silence commits with an empty head; heartbeats cover it from here"
    );
    assert!(
        started.elapsed() >= Duration::from_millis(250),
        "the engine waited out the hold budget before committing"
    );
    let rows = attempts(&bus);
    assert_eq!(
        rows.last().map(|row| row.outcome),
        Some(AttemptOutcome::Streaming),
        "the committed attempt is recorded even though no event arrived"
    );
    // Dropping the tail cancels the stream (the client went away).
    drop(stream.tail);
}

#[tokio::test]
async fn without_a_hold_budget_a_silent_stream_keeps_http_error_semantics() {
    let silent = spawn_upstream(upstream_empty_stream()).await;
    let (engine, _bus) = build_engine(
        vec![member("silent", silent, None)],
        vec![pool(
            "unheld",
            vec!["silent"],
            ladder_retry(),
            Default::default(),
        )],
    );

    let error = engine
        .execute_stream(engine_request("unheld", true))
        .await
        .expect_err("nothing was committed, so the failure keeps its status");
    assert_eq!(error.status_code(), 502);
}

#[tokio::test]
async fn the_ladder_deadline_spans_all_attempts() {
    // The member fails fast; the backoff between attempts outlives the
    // ladder's total deadline, so the second attempt must not run.
    let failing = spawn_upstream(upstream_json(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":"down"}"#,
    ))
    .await;
    let (engine, _bus) = build_engine(
        vec![member("failing", failing, None)],
        vec![pool(
            "racing",
            vec!["failing"],
            RetryConfig {
                max_attempts: 5,
                backoff: Duration::from_millis(500),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::from_millis(100),
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    let started = std::time::Instant::now();
    let error = engine
        .execute(engine_request("racing", false))
        .await
        .expect_err("the deadline stops the ladder");
    assert!(
        matches!(error, EngineError::RetryExhausted { .. }),
        "the deadline exhaustion is reported as an exhausted ladder, got {error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(400),
        "the ladder stopped at its deadline instead of sleeping the full backoff"
    );
}

/// A always-500 upstream plus a long backoff: whichever deadline the ladder
/// honours decides whether it retries past 400ms.
async fn spawn_racing_deadline_upstream() -> String {
    spawn_upstream(upstream_json(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":"down"}"#,
    ))
    .await
}

fn racing_deadline_pool() -> PoolConfig {
    pool(
        "racing",
        vec!["failing"],
        RetryConfig {
            max_attempts: 5,
            backoff: Duration::from_millis(500),
            cooldown: Duration::from_millis(50),
            total_timeout: Duration::ZERO,
            hold: Duration::ZERO,
        },
        Default::default(),
    )
}

#[tokio::test]
async fn the_server_total_tightens_a_looser_route_deadline() {
    // The route allows ten seconds; the server-wide upstream total of 100ms
    // must cut the ladder short anyway (fix brief 04-3). `first_byte` is
    // shrunk below the total so the config stays self-consistent even though
    // constructing it directly skips validation.
    let failing = spawn_racing_deadline_upstream().await;
    let mut config = test_config();
    config.server.upstream_timeouts = UpstreamTimeoutsConfig {
        first_byte: Duration::from_millis(50),
        total: Some(Duration::from_millis(100)),
        ..Default::default()
    };
    let mut route = ladder_retry();
    route.max_attempts = 5;
    route.backoff = Duration::from_millis(500);
    route.total_timeout = Duration::from_secs(10);
    let (engine, _bus) = build_engine_with(
        config,
        vec![member("failing", failing, None)],
        vec![PoolConfig {
            retry: route,
            ..racing_deadline_pool()
        }],
    );

    let started = std::time::Instant::now();
    let error = engine
        .execute(engine_request("racing", false))
        .await
        .expect_err("the server total stops the ladder");
    assert!(
        matches!(error, EngineError::RetryExhausted { .. }),
        "the deadline exhaustion is reported as an exhausted ladder, got {error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(400),
        "the ladder honoured the server total instead of running to the 10s route deadline"
    );
}

#[tokio::test]
async fn a_looser_server_total_does_not_loosen_the_route_deadline() {
    // The inverse: a ten-second server total must not push the ladder past
    // the route's own 100ms deadline.
    let failing = spawn_racing_deadline_upstream().await;
    let mut config = test_config();
    config.server.upstream_timeouts = UpstreamTimeoutsConfig {
        total: Some(Duration::from_secs(10)),
        ..Default::default()
    };
    let mut route = ladder_retry();
    route.max_attempts = 5;
    route.backoff = Duration::from_millis(500);
    route.total_timeout = Duration::from_millis(100);
    let (engine, _bus) = build_engine_with(
        config,
        vec![member("failing", failing, None)],
        vec![PoolConfig {
            retry: route,
            ..racing_deadline_pool()
        }],
    );

    let started = std::time::Instant::now();
    let error = engine
        .execute(engine_request("racing", false))
        .await
        .expect_err("the route deadline still stops the ladder");
    assert!(
        matches!(error, EngineError::RetryExhausted { .. }),
        "the deadline exhaustion is reported as an exhausted ladder, got {error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_millis(400),
        "the ladder honoured the route deadline instead of running to the 10s server total"
    );
}

/// An upstream that accepts the request and never answers at all: neither
/// headers nor an error, so only the ladder's total deadline (not the
/// transport's 60s first-byte timeout) can end the attempt.
fn upstream_never_answers() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let never = std::future::pending::<axum::response::Response>();
            never.await
        }),
    )
}

#[tokio::test]
async fn the_ladder_deadline_bounds_an_attempt_in_flight() {
    // The member hangs forever with the transport's first-byte timeout at
    // its 60s default: the ladder's 200ms total must cut the attempt short,
    // record its row, and end without a further attempt.
    let silent = spawn_upstream(upstream_never_answers()).await;
    let (engine, bus) = build_engine(
        vec![member("silent", silent, None)],
        vec![pool(
            "racing",
            vec!["silent"],
            RetryConfig {
                max_attempts: 3,
                backoff: Duration::from_millis(10),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::from_millis(200),
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    let started = std::time::Instant::now();
    let error = engine
        .execute(engine_request("racing", false))
        .await
        .expect_err("the ladder deadline ends the hanging attempt");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the ladder deadline must bound the attempt instead of the 60s first-byte timeout"
    );
    assert!(
        matches!(error, EngineError::RetryExhausted { .. }),
        "the deadline exhaustion is reported as an exhausted ladder, got {error:?}"
    );
    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        1,
        "the deadline check prevents a further attempt after the timeout"
    );
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(
        rows[0].error_kind.as_deref(),
        Some("UpstreamTimeout"),
        "the hanging attempt is recorded as a timeout, got {:?}",
        rows[0].error_kind
    );
}

#[tokio::test]
async fn the_ladder_deadline_bounds_a_stream_before_commit() {
    // The upstream opens the stream and then never sends an event; with no
    // hold budget the pre-commit window waits on it, and the ladder's 200ms
    // total — not the 60s idle timeout — must end it before any commit.
    let silent = spawn_upstream(upstream_silent_stream()).await;
    let (engine, bus) = build_engine(
        vec![member("silent", silent, None)],
        vec![pool(
            "racing",
            vec!["silent"],
            RetryConfig {
                max_attempts: 3,
                backoff: Duration::from_millis(10),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::from_millis(200),
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    let started = std::time::Instant::now();
    let error = engine
        .execute_stream(engine_request("racing", true))
        .await
        .expect_err("the ladder deadline ends the silent stream before commit");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "the ladder deadline must bound the pre-commit wait instead of the 60s idle timeout"
    );
    assert!(
        matches!(error, EngineError::RetryExhausted { .. }),
        "the deadline exhaustion is reported as an exhausted ladder, got {error:?}"
    );
    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        1,
        "the deadline check prevents a further attempt after the timeout"
    );
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(
        rows[0].error_kind.as_deref(),
        Some("UpstreamTimeout"),
        "the silent attempt is recorded as a timeout, got {:?}",
        rows[0].error_kind
    );
}

#[tokio::test]
async fn a_provider_retry_ladder_retries_the_same_member() {
    // The upstream fails the first request with 500 and succeeds after.
    let counter = Arc::new(std::sync::Mutex::new(0u32));
    let seen = Arc::clone(&counter);
    let app = Router::new().route(
        "/{*path}",
        post(move || {
            let seen = Arc::clone(&seen);
            async move {
                let mut n = seen.lock().expect("counter");
                *n += 1;
                if *n == 1 {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        [(header::CONTENT_TYPE, "application/json")],
                        r#"{"error":"first try fails"}"#.to_owned(),
                    )
                } else {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        ANTHROPIC_OK_BODY.to_owned(),
                    )
                }
            }
        }),
    );
    let endpoint = spawn_upstream(app).await;
    let mut provider = member("flaky", endpoint, None);
    provider.retry = Some(RetryConfig {
        max_attempts: 2,
        backoff: Duration::from_millis(10),
        cooldown: Duration::from_millis(50),
        total_timeout: Duration::ZERO,
        hold: Duration::ZERO,
    });
    let registry =
        llm_proxy_core::ProviderRegistry::from_parts(vec![provider], vec![]).expect("registry");
    let bus = Arc::new(RecordingBus::new());
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::<RecordingBus>::clone(&bus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");

    let response = engine
        .execute(engine_request("flaky", false))
        .await
        .expect("the same member is retried");
    assert_eq!(response.core.usage.input_tokens, 10);
    let rows = attempts(&bus);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].provider, "flaky");
    assert_eq!(rows[1].provider, "flaky");
}

#[tokio::test]
async fn credential_pools_rotate_away_from_a_rejected_key() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let upstream = spawn_upstream(upstream_key_check("good-key", Arc::clone(&seen))).await;
    let (engine, bus) = build_engine(
        vec![member("keys", upstream, Some(vec!["good-key"]))],
        vec![pool(
            "keyed",
            vec!["keys"],
            RetryConfig {
                max_attempts: 3,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    let response = engine
        .execute(engine_request("keyed", false))
        .await
        .expect("rotation finds the working key");

    let tried = seen.lock().expect("seen lock").clone();
    assert!(
        tried.len() >= 2,
        "the first (base) key must be rejected before rotation, tried {tried:?}"
    );
    assert_eq!(*tried.last().expect("at least one try"), "good-key");

    let snapshot = bus.snapshot();
    let completed = snapshot
        .iter()
        .find_map(|event| match event {
            ProxyEvent::ResponseCompleted(completed) => Some(completed),
            _ => None,
        })
        .expect("completed row");
    assert_eq!(
        completed.credential_id.as_deref(),
        Some("key-1"),
        "the ledger records which key served the request"
    );
    let _ = response;
}

#[tokio::test]
async fn a_rejected_key_stays_out_of_rotation_for_later_requests_while_it_cools() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let upstream = spawn_upstream(upstream_key_check("good-key", Arc::clone(&seen))).await;
    let (engine, bus) = build_engine(
        vec![member("keys", upstream, Some(vec!["good-key"]))],
        vec![pool(
            "keyed",
            vec!["keys"],
            RetryConfig {
                max_attempts: 3,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_secs(30),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    // First request: the base key is rejected (and cools down for the
    // configured 30s); rotation moves on to the working key.
    engine
        .execute(engine_request("keyed", false))
        .await
        .expect("rotation finds the working key");
    assert_eq!(
        seen.lock().expect("seen lock").clone(),
        vec!["base-key".to_owned(), "good-key".to_owned()]
    );

    // Second request: round-robin would come back to the base key, but it
    // is still cooling from the rejection — the pick must skip it without
    // spending an upstream call on it.
    engine
        .execute(engine_request("keyed", false))
        .await
        .expect("served by the working key alone");
    assert_eq!(
        seen.lock().expect("seen lock").clone(),
        vec![
            "base-key".to_owned(),
            "good-key".to_owned(),
            "good-key".to_owned()
        ],
        "a cooling key is not retried on a later request"
    );
    let rows = attempts(&bus);
    assert_eq!(
        rows.len(),
        3,
        "two attempts for the first request, one for the second"
    );
    assert_eq!(rows[2].outcome, AttemptOutcome::Succeeded);
    assert_eq!(rows[2].credential_id.as_deref(), Some("key-1"));
}

/// A [`CredentialSource`] call against the default source, outside the
/// engine: the pick must still hand back a key when every key is excluded.
#[tokio::test]
async fn the_default_pick_falls_back_to_rotation_when_every_key_is_cooling() {
    use llm_proxy_engine::{ConfigCredentialSource, CredentialQuery, CredentialSource};

    let source = ConfigCredentialSource::new();
    let keys = vec![
        SecretString::from("k0".to_owned()),
        SecretString::from("k1".to_owned()),
    ];
    let cooling = vec!["key-0".to_owned(), "key-1".to_owned()];
    let pick = source
        .pick(&CredentialQuery {
            provider: "p",
            keys: &keys,
            passthrough: false,
            inbound: None,
            avoid: Some("key-1"),
            cooling: &cooling,
        })
        .await
        .expect("cooldowns reorder, they never refuse service");
    assert_eq!(
        pick.credential_id.as_deref(),
        Some("key-0"),
        "the key that just failed is still the last resort"
    );
}

#[tokio::test]
async fn session_ids_land_on_usage_rows_from_header_and_metadata() {
    let good = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let (engine, bus) = build_engine(vec![member("solo", good, None)], vec![]);

    let mut request = engine_request("solo", false);
    request.inbound.headers.insert(
        "x-claude-code-session-id".to_owned(),
        "sess-header".to_owned(),
    );
    engine.execute(request).await.expect("runs");

    let mut request = engine_request("solo", false);
    request.core.metadata.user_id = Some("user-42".to_owned());
    engine.execute(request).await.expect("runs");

    let snapshot = bus.snapshot();
    let sessions: Vec<Option<String>> = snapshot
        .iter()
        .filter_map(|event| match event {
            ProxyEvent::RequestReceived(RequestReceived { session_id, .. }) => {
                Some(session_id.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        sessions,
        vec![Some("sess-header".to_owned()), Some("user-42".to_owned())],
        "the session header wins; metadata.user_id is the fallback"
    );
}

/// An upstream whose body names its member, failing its first N requests.
fn upstream_tagged(tag: &'static str, fail_first: u32) -> (Router, Arc<std::sync::Mutex<u32>>) {
    upstream_tagged_failing_with(tag, fail_first, StatusCode::INTERNAL_SERVER_ERROR)
}

/// [`upstream_tagged`] with the status its first `fail_first` calls answer.
fn upstream_tagged_failing_with(
    tag: &'static str,
    fail_first: u32,
    status: StatusCode,
) -> (Router, Arc<std::sync::Mutex<u32>>) {
    let counter = Arc::new(std::sync::Mutex::new(0u32));
    let seen = Arc::clone(&counter);
    let router = Router::new().route(
        "/{*path}",
        post(move || {
            let seen = Arc::clone(&seen);
            async move {
                let calls = {
                    let mut n = seen.lock().expect("counter");
                    *n += 1;
                    *n
                };
                if calls <= fail_first {
                    (
                        status,
                        [(header::CONTENT_TYPE, "application/json")],
                        r#"{"error":"warming up"}"#.to_owned(),
                    )
                } else {
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        tagged_ok_body(tag),
                    )
                }
            }
        }),
    );
    (router, counter)
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

#[tokio::test]
async fn sticky_affinity_pins_a_session_to_the_member_that_served_it() {
    let (primary_router, _primary_calls) = upstream_tagged("primary", 1);
    let (secondary_router, _secondary_calls) = upstream_tagged("secondary", 0);
    let primary = spawn_upstream(primary_router).await;
    let secondary = spawn_upstream(secondary_router).await;
    let (engine, _bus) = build_engine(
        vec![
            member("primary", primary, None),
            member("secondary", secondary, None),
        ],
        vec![pool(
            "sticky",
            vec!["primary", "secondary"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(80),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            AffinityConfig {
                sticky: true,
                on_conflict: AffinityConflictPolicy::Strip,
            },
        )],
    );

    let with_session = || {
        let mut request = engine_request("sticky", false);
        request
            .inbound
            .headers
            .insert("x-claude-code-session-id".to_owned(), "sess-1".to_owned());
        request
    };

    // First request: primary is warming up (500) and cools down; secondary
    // serves and gets the pin.
    let first = engine.execute(with_session()).await.expect("served");
    assert_eq!(response_text(&first), "from-secondary");

    // Cooldown expiry: primary is healthy again and first in config order.
    tokio::time::sleep(Duration::from_millis(120)).await;

    // A session-less request walks the configured order: primary first.
    let anonymous = engine
        .execute(engine_request("sticky", false))
        .await
        .expect("served");
    assert_eq!(
        response_text(&anonymous),
        "from-primary",
        "without a session the configured order stands"
    );

    // The pinned session goes back to its member.
    let second = engine.execute(with_session()).await.expect("served");
    assert_eq!(
        response_text(&second),
        "from-secondary",
        "a sticky session returns to the member that served it"
    );
}

/// An upstream that serves its first `healthy_calls` calls, then always
/// fails: lets a test pin a session to a member before killing it.
fn upstream_dies_after(tag: &'static str, healthy_calls: u32) -> Router {
    let counter = Arc::new(std::sync::Mutex::new(0u32));
    Router::new().route(
        "/{*path}",
        post(move || {
            let counter = Arc::clone(&counter);
            async move {
                let calls = {
                    let mut n = counter.lock().expect("counter");
                    *n += 1;
                    *n
                };
            if calls <= healthy_calls {
                let body = format!(
                    r#"{{"id":"msg_{tag}","type":"message","role":"assistant","content":[{{"type":"text","text":"from-{tag}"}}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{{"input_tokens":10,"output_tokens":4}}}}"#
                );
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"error":"died"}"#.to_owned(),
                )
            }
            }
        }),
    )
}

#[tokio::test]
async fn refuse_policy_fails_the_switch_instead_of_serving_elsewhere() {
    let alpha = spawn_upstream(upstream_dies_after("alpha", 1)).await;
    let beta = spawn_upstream(upstream_dies_after("beta", u32::MAX)).await;
    let (engine, _bus) = build_engine(
        vec![member("alpha", alpha, None), member("beta", beta, None)],
        vec![pool(
            "strict",
            vec!["alpha", "beta"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(500),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            AffinityConfig {
                sticky: true,
                on_conflict: AffinityConflictPolicy::Refuse,
            },
        )],
    );

    let with_session = || {
        let mut request = engine_request("strict", false);
        request
            .inbound
            .headers
            .insert("x-claude-code-session-id".to_owned(), "sess-r".to_owned());
        request
    };

    // The session pins to alpha, which serves exactly one request.
    let first = engine.execute(with_session()).await.expect("served");
    assert_eq!(response_text(&first), "from-alpha");

    // alpha now fails; the refuse policy must stop the pinned session from
    // being switched to beta.
    let error = engine
        .execute(with_session())
        .await
        .expect_err("the switch is refused");
    assert!(
        matches!(error, EngineError::AffinityRefused { .. }),
        "expected AffinityRefused, got {error:?}"
    );
    assert_eq!(error.status_code(), 503);
}

#[tokio::test]
async fn refuse_policy_holds_when_the_pinned_member_is_already_cooling() {
    // alpha serves exactly one request, then 500s; a long cooldown keeps it
    // out of the candidate order for the rest of the test.
    let alpha = spawn_upstream(upstream_dies_after("alpha", 1)).await;
    let beta = spawn_upstream(upstream_dies_after("beta", u32::MAX)).await;
    let (engine, bus) = build_engine(
        vec![member("alpha", alpha, None), member("beta", beta, None)],
        vec![pool(
            "strict",
            vec!["alpha", "beta"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_secs(30),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            AffinityConfig {
                sticky: true,
                on_conflict: AffinityConflictPolicy::Refuse,
            },
        )],
    );

    let with_session = || {
        let mut request = engine_request("strict", false);
        request.inbound.headers.insert(
            "x-claude-code-session-id".to_owned(),
            "sess-cool".to_owned(),
        );
        request
    };

    // The session pins to alpha.
    let first = engine.execute(with_session()).await.expect("served");
    assert_eq!(response_text(&first), "from-alpha");

    // A session-less request burns alpha's health: alpha 500s, enters its
    // cooldown, and the anonymous request fails over to beta.
    let anonymous = engine
        .execute(engine_request("strict", false))
        .await
        .expect("served by beta");
    assert_eq!(response_text(&anonymous), "from-beta");
    let rows_before = attempts(&bus).len();

    // alpha is cooling when the pinned session's next request orders its
    // candidates, so beta would be the ladder's *first* attempt. The refuse
    // policy must stop that switch too, without ever reaching beta.
    let error = engine
        .execute(with_session())
        .await
        .expect_err("the switch is refused before any attempt");
    assert!(
        matches!(error, EngineError::AffinityRefused { .. }),
        "expected AffinityRefused, got {error:?}"
    );
    assert_eq!(
        attempts(&bus).len(),
        rows_before,
        "no attempt may be sent to a member other than the pinned one"
    );
}

#[tokio::test]
async fn a_client_fault_does_not_demote_the_pool_member() {
    // alpha answers its first call with a 400, then serves; beta always
    // serves. A long cooldown would keep alpha demoted if a 400 counted.
    let (alpha_router, _alpha_calls) =
        upstream_tagged_failing_with("alpha", 1, StatusCode::BAD_REQUEST);
    let (beta_router, _beta_calls) = upstream_tagged("beta", 0);
    let alpha = spawn_upstream(alpha_router).await;
    let beta = spawn_upstream(beta_router).await;
    let (engine, bus) = build_engine(
        vec![member("alpha", alpha, None), member("beta", beta, None)],
        vec![pool(
            "both",
            vec!["alpha", "beta"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_secs(30),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    // One client's malformed request: terminal for that request, and no
    // failover.
    let error = engine
        .execute(engine_request("both", false))
        .await
        .expect_err("a 400 is terminal");
    assert!(
        matches!(error, EngineError::Upstream { status: 400, .. }),
        "got {error:?}"
    );

    // Everyone else's next request still reaches alpha first: a fault
    // deterministic for one request is no verdict on the member.
    let response = engine
        .execute(engine_request("both", false))
        .await
        .expect("served");
    assert_eq!(
        response_text(&response),
        "from-alpha",
        "a 400 must not cool the member for every other client"
    );
    let rows = attempts(&bus);
    assert_eq!(rows.len(), 2, "the 400, then the served request");
    assert_eq!(rows[0].provider, "alpha");
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[1].provider, "alpha");
    assert_eq!(rows[1].outcome, AttemptOutcome::Succeeded);
}

#[tokio::test]
async fn a_rejected_key_cools_the_key_not_the_member() {
    // "keyed" has two keys and only accepts the second; "spare" always
    // serves. A rejected key must not push the whole member behind spare.
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let keyed = spawn_upstream(upstream_key_check_tagged(
        "good-key",
        "keyed",
        Arc::clone(&seen),
    ))
    .await;
    let (spare_router, _spare_calls) = upstream_tagged("spare", 0);
    let spare = spawn_upstream(spare_router).await;
    let (engine, bus) = build_engine(
        vec![
            member("keyed", keyed, Some(vec!["good-key"])),
            member("spare", spare, None),
        ],
        vec![pool(
            "pooled",
            vec!["keyed", "spare"],
            RetryConfig {
                max_attempts: 2,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_secs(30),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    );

    // key-0 is rejected; the member rotates to its working key and serves.
    let first = engine
        .execute(engine_request("pooled", false))
        .await
        .expect("served after rotation");
    assert_eq!(response_text(&first), "from-keyed");

    // Only key-0 cooled: the member stays first in the order and serves
    // the next request with its working key in one attempt.
    let second = engine
        .execute(engine_request("pooled", false))
        .await
        .expect("served");
    assert_eq!(
        response_text(&second),
        "from-keyed",
        "a rejected key must not demote the whole member"
    );
    assert_eq!(
        seen.lock().expect("seen lock").clone(),
        vec![
            "base-key".to_owned(),
            "good-key".to_owned(),
            "good-key".to_owned()
        ]
    );
    let rows = attempts(&bus);
    assert_eq!(rows.len(), 3, "rejection, rotation, then one direct hit");
    assert_eq!(rows[2].provider, "keyed");
    assert_eq!(rows[2].outcome, AttemptOutcome::Succeeded);
    assert_eq!(rows[2].credential_id.as_deref(), Some("key-1"));
}

fn strict_pool() -> PoolConfig {
    pool(
        "strict",
        vec!["alpha", "beta"],
        RetryConfig {
            max_attempts: 1,
            backoff: Duration::from_millis(5),
            cooldown: Duration::from_millis(50),
            total_timeout: Duration::ZERO,
            hold: Duration::ZERO,
        },
        AffinityConfig {
            sticky: true,
            on_conflict: AffinityConflictPolicy::Refuse,
        },
    )
}

fn with_session(route: &str, session: &str) -> EngineRequest {
    let mut request = engine_request(route, false);
    request
        .inbound
        .headers
        .insert("x-claude-code-session-id".to_owned(), session.to_owned());
    request
}

#[tokio::test]
async fn a_direct_route_does_not_pin_the_session() {
    let solo = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let (alpha_router, _) = upstream_tagged("alpha", 0);
    let (beta_router, _) = upstream_tagged("beta", 0);
    let alpha = spawn_upstream(alpha_router).await;
    let beta = spawn_upstream(beta_router).await;
    let (engine, _bus) = build_engine(
        vec![
            member("solo", solo, None),
            member("alpha", alpha, None),
            member("beta", beta, None),
        ],
        vec![strict_pool()],
    );

    // Served by a direct provider route: pinning is a sticky pool's
    // opt-in, so this leaves no pin behind.
    engine
        .execute(with_session("solo", "sess-direct"))
        .await
        .expect("served directly");

    // The same session on a Refuse pool whose members do not include
    // "solo" is served by the pool's first member, not refused.
    let response = engine
        .execute(with_session("strict", "sess-direct"))
        .await
        .expect("a direct route leaves no pin to refuse against");
    assert_eq!(response_text(&response), "from-alpha");
}

#[tokio::test]
async fn a_non_sticky_pool_does_not_pin_the_session() {
    let (gamma_router, _) = upstream_tagged("gamma", 0);
    let (alpha_router, _) = upstream_tagged("alpha", 0);
    let (beta_router, _) = upstream_tagged("beta", 0);
    let gamma = spawn_upstream(gamma_router).await;
    let alpha = spawn_upstream(alpha_router).await;
    let beta = spawn_upstream(beta_router).await;
    let (engine, _bus) = build_engine(
        vec![
            member("gamma", gamma, None),
            member("alpha", alpha, None),
            member("beta", beta, None),
        ],
        vec![
            pool(
                "loose",
                vec!["gamma"],
                ladder_retry(),
                AffinityConfig {
                    sticky: false,
                    on_conflict: AffinityConflictPolicy::Strip,
                },
            ),
            strict_pool(),
        ],
    );

    // Served by a pool that did not opt into pinning.
    let first = engine
        .execute(with_session("loose", "sess-loose"))
        .await
        .expect("served by the loose pool");
    assert_eq!(response_text(&first), "from-gamma");

    // No pin was recorded, so the Refuse pool serves the session normally.
    let response = engine
        .execute(with_session("strict", "sess-loose"))
        .await
        .expect("a non-sticky pool leaves no pin to refuse against");
    assert_eq!(response_text(&response), "from-alpha");
}

/// A streaming upstream that only answers when the right key arrives; 401
/// otherwise.
fn upstream_key_check_stream(
    expected_key: &'static str,
    seen: Arc<std::sync::Mutex<Vec<String>>>,
) -> Router {
    Router::new().route(
        "/{*path}",
        post(move |headers: axum::http::HeaderMap| async move {
            let got = headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            seen.lock().expect("seen lock").push(got.clone());
            if got == expected_key {
                let events = vec![
                    Event::default()
                        .event("message_start")
                        .data(r#"{"type":"message_start","message":{"id":"msg_k","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}"#),
                    Event::default()
                        .event("message_stop")
                        .data(r#"{"type":"message_stop"}"#),
                ];
                let stream = futures::stream::iter(
                    events.into_iter().map(Ok::<_, std::convert::Infallible>),
                );
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    Sse::new(stream).into_response(),
                )
                    .into_response()
            } else {
                (
                    StatusCode::UNAUTHORIZED,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"type":"error","error":{"type":"authentication_error","message":"bad key"}}"#,
                )
                    .into_response()
            }
        }),
    )
}

/// Two members, each with a two-key pool. The first member fails every
/// call transiently, so the ladder moves on with its `key-0` as the failed
/// credential; the second member only accepts its own `key-0` (the shared
/// base key). A hint that leaked across the member boundary would skip
/// that key and the request would fail.
fn two_members_with_key_pools(
    first_endpoint: String,
    second_endpoint: String,
) -> (Engine, Arc<RecordingBus>) {
    build_engine(
        vec![
            member("first", first_endpoint, Some(vec!["first-extra"])),
            member("second", second_endpoint, Some(vec!["second-extra"])),
        ],
        vec![pool(
            "keyed-pair",
            vec!["first", "second"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
    )
}

#[tokio::test]
async fn a_failed_key_on_one_member_does_not_skip_the_next_members_key() {
    let first = spawn_upstream(upstream_json(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":"down"}"#,
    ))
    .await;
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let second = spawn_upstream(upstream_key_check("base-key", Arc::clone(&seen))).await;
    let (engine, bus) = two_members_with_key_pools(first, second);

    engine
        .execute(engine_request("keyed-pair", false))
        .await
        .expect("the second member serves with its own first key");

    assert_eq!(
        seen.lock().expect("seen lock").clone(),
        vec!["base-key".to_owned()],
        "the second member's first pick is its own key-0, not key-1"
    );
    let rows = attempts(&bus);
    assert_eq!(rows.len(), 2, "one failed attempt, then the serving one");
    assert_eq!(rows[0].provider, "first");
    assert_eq!(rows[0].credential_id.as_deref(), Some("key-0"));
    assert_eq!(rows[1].provider, "second");
    assert_eq!(rows[1].outcome, AttemptOutcome::Succeeded);
    assert_eq!(rows[1].credential_id.as_deref(), Some("key-0"));
}

#[tokio::test]
async fn a_failed_key_on_one_member_does_not_skip_the_next_members_key_when_streaming() {
    let first = spawn_upstream(upstream_json(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":"down"}"#,
    ))
    .await;
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let second = spawn_upstream(upstream_key_check_stream("base-key", Arc::clone(&seen))).await;
    let (engine, bus) = two_members_with_key_pools(first, second);

    let stream = engine
        .execute_stream(engine_request("keyed-pair", true))
        .await
        .expect("the second member's stream commits with its own first key");
    let mut tail = stream.tail;
    while tail.next().await.is_some() {}

    assert_eq!(
        seen.lock().expect("seen lock").clone(),
        vec!["base-key".to_owned()],
        "the second member's first pick is its own key-0, not key-1"
    );
    let rows = attempts(&bus);
    assert_eq!(rows.len(), 2, "one failed attempt, then the commit");
    assert_eq!(rows[0].provider, "first");
    assert_eq!(rows[0].credential_id.as_deref(), Some("key-0"));
    assert_eq!(rows[1].provider, "second");
    assert_eq!(rows[1].outcome, AttemptOutcome::Streaming);
    assert_eq!(rows[1].credential_id.as_deref(), Some("key-0"));
}

#[tokio::test]
async fn the_health_probe_clears_a_members_cooldown_when_it_recovers() {
    // "probed" 429s once with a long advertised reset, then serves with a
    // distinguishable body (input_tokens=1); "spare" always serves with
    // input_tokens=2. The first request fails over to the spare and cools
    // "probed" for the advertised 300s; the probe (every 50ms) must clear
    // that cooldown so the next request reaches "probed" again.
    let probed_endpoint = spawn_upstream(probed_upstream()).await;
    let spare_endpoint = spawn_upstream(upstream_json(StatusCode::OK, SPARE_OK_BODY)).await;
    let mut probed_member = member("probed", probed_endpoint, None);
    // The probe needs a model the member serves.
    probed_member.model_aliases.insert(
        "claude-sonnet-4-6".to_owned(),
        "claude-sonnet-4-6".to_owned(),
    );
    let mut spare_member = member("spare", spare_endpoint, None);
    spare_member.model_aliases.insert(
        "claude-sonnet-4-6".to_owned(),
        "claude-sonnet-4-6".to_owned(),
    );
    let (engine, _bus) = build_engine(
        vec![probed_member, spare_member],
        vec![PoolConfig {
            name: "probing".to_owned(),
            members: vec!["probed".to_owned(), "spare".to_owned()],
            retry: RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            affinity: Default::default(),
            health: llm_proxy_core::provider_config::PoolHealthConfig {
                probe_interval: Some(Duration::from_millis(50)),
            },
        }],
    );

    // First request: probed 429s (300s advertised reset) and the pool fails
    // over to the spare.
    let first = engine
        .execute(engine_request("probing", false))
        .await
        .expect("the pool fails over to the spare");
    assert_eq!(first.core.usage.input_tokens, 2, "the spare served it");

    // Without the probe, the 300s cooldown keeps probed out of rotation.
    // The probe recovers it within its interval.
    let mut recovered = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        if let Ok(response) = engine.execute(engine_request("probing", false)).await
            && response.core.usage.input_tokens == 1
        {
            recovered = true;
            break;
        }
    }
    assert!(
        recovered,
        "the probe must restore the cooled member before the advertised reset"
    );
}

/// Serves one 429 with `Retry-After: 300`, then a distinguishable 200.
fn probed_upstream() -> Router {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CALLS: AtomicU32 = AtomicU32::new(0);
    Router::new().route(
        "/{*path}",
        post(|| async {
            let n = CALLS.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"error":"slow"}"#,
                )
                    .into_response_with_retry_after("300");
            }
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                PROBED_OK_BODY,
            )
                .into_response()
        }),
    )
}

const PROBED_OK_BODY: &str = r#"{"id":"msg_p","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;
const SPARE_OK_BODY: &str = r#"{"id":"msg_s","type":"message","role":"assistant","content":[{"type":"text","text":"spare"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":1}}"#;

/// Builds the engine over `providers`/`pools` with the test's own cooldown
/// store, so the passive-health view stays observable from the test.
fn build_engine_with_cooldowns(
    providers: Vec<ProviderConfig>,
    pools: Vec<PoolConfig>,
    cooldowns: &Arc<CooldownStore>,
) -> Engine {
    let registry = ProviderRegistry::from_parts(providers, pools).expect("registry with pools");
    EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .cooldowns(Arc::clone(cooldowns) as Arc<dyn Cooldowns>)
        .build()
        .expect("engine builds")
}

/// A one-attempt pool whose first member throttles: fix brief 04-5 pins the
/// passive-health cooldown to the reset the upstream advertised, not the
/// ladder's configured cooldown.
async fn throttle_failover_setup(throttle: Router) -> (Engine, Arc<CooldownStore>) {
    let primary = spawn_upstream(throttle).await;
    let secondary = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let cooldowns = Arc::new(CooldownStore::new());
    let engine = build_engine_with_cooldowns(
        vec![
            member("primary", primary, None),
            member("secondary", secondary, None),
        ],
        vec![pool(
            "both",
            vec!["primary", "secondary"],
            RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::from_millis(50),
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            Default::default(),
        )],
        &cooldowns,
    );
    (engine, cooldowns)
}

/// The cooldown must last the advertised reset (2s), not the configured
/// 50ms ladder cooldown, and lift once the reset passes.
#[tokio::test]
async fn a_429_cools_the_member_until_the_advertised_reset() {
    let (engine, cooldowns) =
        throttle_failover_setup(throttled_upstream_with_retry_after("2")).await;

    let response = engine
        .execute(engine_request("both", false))
        .await
        .expect("the pool fails over to the secondary");
    assert_eq!(response.core.usage.input_tokens, 10, "the secondary served");

    let now = time::OffsetDateTime::now_utc();
    assert!(
        !cooldowns.provider_available("primary", now).await,
        "the 429 puts the member into cooldown"
    );
    let remaining = cooldowns
        .soonest_reset(&["primary"], now)
        .await
        .expect("the cooled member reports its reset");
    assert!(
        (Duration::from_millis(1500)..=Duration::from_millis(2500)).contains(&remaining),
        "the cooldown lasts the advertised 2s, got {remaining:?}"
    );

    tokio::time::sleep(Duration::from_millis(2200)).await;
    assert!(
        cooldowns
            .provider_available("primary", time::OffsetDateTime::now_utc())
            .await,
        "the member is back once the advertised reset passes"
    );
}

/// An Anthropic reset timestamp with no `Retry-After` must set the cooldown
/// too: before reset-header parsing this fell back to the configured 50ms.
#[tokio::test]
async fn an_anthropic_rfc3339_reset_header_sets_the_cooldown() {
    let reset = (time::OffsetDateTime::now_utc() + time::Duration::seconds(2))
        .format(&time::format_description::well_known::Rfc3339)
        .expect("format the reset timestamp");
    let (engine, cooldowns) =
        throttle_failover_setup(throttled_upstream_with_reset_header(reset)).await;

    let response = engine
        .execute(engine_request("both", false))
        .await
        .expect("the pool fails over to the secondary");
    assert_eq!(response.core.usage.input_tokens, 10, "the secondary served");

    let now = time::OffsetDateTime::now_utc();
    assert!(
        !cooldowns.provider_available("primary", now).await,
        "the 429 puts the member into cooldown"
    );
    let remaining = cooldowns
        .soonest_reset(&["primary"], now)
        .await
        .expect("the cooled member reports its reset");
    assert!(
        (Duration::from_millis(1500)..=Duration::from_millis(2500)).contains(&remaining),
        "the RFC 3339 reset (2s ahead) sets the cooldown, got {remaining:?}"
    );
}

// ---- Plan 04-1: the failover strip (provider-bound state) ----

type CapturedBodies = Arc<std::sync::Mutex<Vec<String>>>;

fn captured_bodies() -> CapturedBodies {
    Arc::new(std::sync::Mutex::new(Vec::new()))
}

fn bodies_of(captured: &CapturedBodies) -> Vec<String> {
    captured.lock().expect("capture lock").clone()
}

/// An upstream that records every request body it receives, then answers
/// with the fixed status/body (modelled on [`upstream_key_check`], which
/// records headers the same way).
fn upstream_body_capture(
    status: StatusCode,
    body: &'static str,
    captured: CapturedBodies,
) -> Router {
    Router::new().route(
        "/{*path}",
        post(move |raw: axum::body::Bytes| async move {
            captured
                .lock()
                .expect("capture lock")
                .push(String::from_utf8_lossy(&raw).into_owned());
            (status, [(header::CONTENT_TYPE, "application/json")], body)
        }),
    )
}

/// An `EngineRequest` whose raw client body is the same-format Anthropic
/// mirror of the core, so same-format passthrough is possible.
fn engine_request_with_wire(
    route: &str,
    core: CoreRequest,
    wire_body: serde_json::Value,
) -> EngineRequest {
    EngineRequest::new(
        route,
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core,
        llm_proxy_engine::InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(ClientWireFormat::AnthropicMessages, wire_body),
        },
    )
    .with_request_id("req-plan04-1")
    .with_body_hash("hash")
}

fn text_block(text: &str) -> CoreContent {
    CoreContent::Text {
        text: text.to_owned(),
        cache: None,
    }
}

fn plain_core(messages: Vec<CoreMessage>) -> CoreRequest {
    CoreRequest {
        model: ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        system: Vec::new(),
        messages,
        tools: Vec::new(),
        tool_choice: None,
        sampling: SamplingOptions::default(),
        stream: false,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

async fn failing_then_serving_pair() -> (Engine, CapturedBodies, CapturedBodies) {
    let failing = captured_bodies();
    let serving = captured_bodies();
    let a = spawn_upstream(upstream_body_capture(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"type":"error","error":{"type":"api_error","message":"boom"}}"#,
        Arc::clone(&failing),
    ))
    .await;
    let b = spawn_upstream(upstream_body_capture(
        StatusCode::OK,
        ANTHROPIC_OK_BODY,
        Arc::clone(&serving),
    ))
    .await;
    let (engine, _bus) = build_engine(
        vec![member("a", a, None), member("b", b, None)],
        vec![pool(
            "pair",
            vec!["a", "b"],
            ladder_retry(),
            Default::default(),
        )],
    );
    (engine, failing, serving)
}

#[tokio::test]
async fn failover_drops_earlier_turn_thinking_and_tool_signatures() {
    let (engine, served_to_a, served_to_b) = failing_then_serving_pair().await;
    // Two assistant turns. The first carries signed thinking plus a signed
    // tool_use, answered by a user tool_result; the second is plain text and
    // the request ends with a user text message, so no tool loop is open.
    let core = plain_core(vec![
        CoreMessage {
            role: CoreRole::User,
            content: vec![text_block("find the weather")],
        },
        CoreMessage {
            role: CoreRole::Assistant,
            content: vec![
                CoreContent::Thinking {
                    text: "pondering".to_owned(),
                    signature: Some("sig-1".to_owned()),
                },
                CoreContent::ToolUse {
                    id: "toolu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({"city": "sf"}),
                    signature: Some("tsig-1".to_owned()),
                },
            ],
        },
        CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::ToolResult {
                tool_use_id: "toolu_1".to_owned(),
                content: vec![text_block("sunny")],
                is_error: false,
                cache: None,
            }],
        },
        CoreMessage {
            role: CoreRole::Assistant,
            content: vec![text_block("It is sunny.")],
        },
        CoreMessage {
            role: CoreRole::User,
            content: vec![text_block("thanks")],
        },
    ]);
    let wire_body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "find the weather"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "pondering", "signature": "sig-1"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "sf"}, "signature": "tsig-1"}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "sunny"}]}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is sunny."}]},
            {"role": "user", "content": [{"type": "text", "text": "thanks"}]}
        ],
    });

    engine
        .execute(engine_request_with_wire("pair", core, wire_body))
        .await
        .expect("the pool fails over to member b");

    let to_a = bodies_of(&served_to_a);
    assert_eq!(to_a.len(), 2, "member a serves both ladder attempts");
    for body in &to_a {
        assert!(
            body.contains(r#""signature":"sig-1""#) && body.contains(r#""type":"thinking""#),
            "member a issued the signed thinking, so its raw body carries it: {body}"
        );
    }
    let to_b = bodies_of(&served_to_b);
    assert_eq!(to_b.len(), 1, "member b serves the failover once");
    assert!(
        !to_b[0].contains("signature"),
        "member b must not receive any provider signature: {}",
        to_b[0]
    );
    assert!(
        !to_b[0].contains("thinking"),
        "member b must not receive thinking blocks: {}",
        to_b[0]
    );
}

#[tokio::test]
async fn failover_refuses_signed_thinking_in_an_open_tool_loop() {
    let (engine, _served_to_a, served_to_b) = failing_then_serving_pair().await;

    // The last assistant turn carries signed thinking plus a tool_use, and
    // the final user message answers it: the tool loop is open, so the
    // thinking block must go back unmodified to its own provider.
    let core = plain_core(vec![
        CoreMessage {
            role: CoreRole::User,
            content: vec![text_block("find the weather")],
        },
        CoreMessage {
            role: CoreRole::Assistant,
            content: vec![
                CoreContent::Thinking {
                    text: "pondering".to_owned(),
                    signature: Some("sig-1".to_owned()),
                },
                CoreContent::ToolUse {
                    id: "toolu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({"city": "sf"}),
                    signature: Some("tsig-1".to_owned()),
                },
            ],
        },
        CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::ToolResult {
                tool_use_id: "toolu_1".to_owned(),
                content: vec![text_block("sunny")],
                is_error: false,
                cache: None,
            }],
        },
    ]);
    let wire_body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "find the weather"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "pondering", "signature": "sig-1"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "sf"}, "signature": "tsig-1"}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "sunny"}]}
            ]}
        ],
    });

    let error = engine
        .execute(engine_request_with_wire("pair", core, wire_body))
        .await
        .expect_err("signed thinking in an open tool loop cannot fail over");
    assert!(
        error.to_string().contains("signed thinking"),
        "the error must name the cause: {error}"
    );
    assert!(
        bodies_of(&served_to_b).is_empty(),
        "member b must receive zero calls"
    );
}

#[tokio::test]
async fn failover_refuses_to_move_a_file_reference_between_members() {
    let (engine, _served_to_a, served_to_b) = failing_then_serving_pair().await;

    let core = plain_core(vec![CoreMessage {
        role: CoreRole::User,
        content: vec![CoreContent::Image {
            source: serde_json::json!({"type": "file", "file_id": "file_123"}),
        }],
    }]);
    let wire_body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "file", "file_id": "file_123"}}
            ]}
        ],
    });

    let error = engine
        .execute(engine_request_with_wire("pair", core, wire_body))
        .await
        .expect_err("a provider file reference cannot move between members");
    assert!(
        error.to_string().contains("file reference"),
        "the error must name the cause: {error}"
    );
    assert!(
        bodies_of(&served_to_b).is_empty(),
        "member b must receive zero calls"
    );
}

#[tokio::test]
async fn a_request_with_nothing_to_strip_keeps_same_format_passthrough() {
    let (engine, _served_to_a, served_to_b) = failing_then_serving_pair().await;

    let core = plain_core(vec![
        CoreMessage {
            role: CoreRole::User,
            content: vec![text_block("hello")],
        },
        CoreMessage {
            role: CoreRole::Assistant,
            content: vec![text_block("hi there")],
        },
        CoreMessage {
            role: CoreRole::User,
            content: vec![text_block("thanks")],
        },
    ]);
    let wire_body = serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
            {"role": "user", "content": [{"type": "text", "text": "thanks"}]}
        ],
        // Not modelled by the core: only same-format passthrough carries it.
        "seed": 7,
    });
    let expected_on_the_wire = serde_json::to_string(&wire_body).expect("serialize client body");

    engine
        .execute(engine_request_with_wire("pair", core, wire_body))
        .await
        .expect("the pool fails over with the raw body");

    let to_b = bodies_of(&served_to_b);
    assert_eq!(to_b.len(), 1, "member b serves the failover once");
    assert_eq!(
        to_b[0], expected_on_the_wire,
        "with nothing to strip the client's raw body must reach member b byte-for-byte"
    );
}

/// A guard whose `inspect_response` refuses every completed response.
#[derive(Debug)]
struct RejectingResponseGuard;

#[async_trait]
impl Guard for RejectingResponseGuard {
    async fn inspect_response(
        &self,
        _response: &CoreResponse,
    ) -> Result<GuardDecision, EngineError> {
        Ok(GuardDecision::Reject(
            "refused by the response guard".to_owned(),
        ))
    }
}

/// A guard whose `inspect_event` refuses every streamed event.
#[derive(Debug)]
struct RejectingStreamGuard;

#[async_trait]
impl Guard for RejectingStreamGuard {
    async fn inspect_event(&self, _event: &CoreEvent) -> Result<GuardDecision, EngineError> {
        Ok(GuardDecision::Reject(
            "refused by the stream guard".to_owned(),
        ))
    }
}

/// An ok-everywhere `[primary, secondary]` pool wired to the test's own
/// cooldown store and ledger bus, with `guard` installed: the shape the
/// guard-rejection tests need to observe both the ledger and the cooldowns.
fn build_guarded_engine(
    guard: Arc<dyn Guard>,
    cooldowns: &Arc<CooldownStore>,
    bus: &Arc<RecordingBus>,
    endpoints: &[String; 2],
) -> Engine {
    let registry = ProviderRegistry::from_parts(
        vec![
            member("primary", endpoints[0].clone(), None),
            member("secondary", endpoints[1].clone(), None),
        ],
        vec![pool(
            "both",
            vec!["primary", "secondary"],
            ladder_retry(),
            Default::default(),
        )],
    )
    .expect("registry with pools");
    EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .cooldowns(Arc::clone(cooldowns) as Arc<dyn Cooldowns>)
        .guard(guard)
        .event_bus(Arc::clone(bus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds")
}

/// Fix brief 04-6: the provider answered and billed the attempt the proxy's
/// guard then refused, so the ledger carries its failed row — and because a
/// guard refusal is proxy policy, not an upstream fault, the member does not
/// cool and the ladder does not fail over (exactly one attempt happened).
#[tokio::test]
async fn a_guard_rejected_response_still_writes_a_failed_attempt_row() {
    let primary = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let secondary = spawn_upstream(upstream_json(StatusCode::OK, ANTHROPIC_OK_BODY)).await;
    let cooldowns = Arc::new(CooldownStore::new());
    let bus = Arc::new(RecordingBus::new());
    let engine = build_guarded_engine(
        Arc::new(RejectingResponseGuard),
        &cooldowns,
        &bus,
        &[primary, secondary],
    );

    let error = engine
        .execute(engine_request("both", false))
        .await
        .expect_err("the guard refuses the completed response");
    assert!(
        matches!(error, EngineError::InvalidRequest(_)),
        "the guard rejection surfaces as InvalidRequest, got {error:?}"
    );

    let rows = attempts(&bus);
    assert_eq!(rows.len(), 1, "the billed attempt writes exactly one row");
    assert_eq!(rows[0].provider, "primary");
    assert_eq!(rows[0].attempt, 0);
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(
        rows[0].http_status,
        Some(400),
        "the row records the guard error's status"
    );
    assert_eq!(rows[0].error_kind.as_deref(), Some("InvalidRequest"));

    let now = time::OffsetDateTime::now_utc();
    assert!(
        cooldowns.provider_available("primary", now).await,
        "a guard refusal is proxy policy: the member that answered is not cooled"
    );
    assert!(
        cooldowns.provider_available("secondary", now).await,
        "the member the ladder never reached is not cooled"
    );
}

/// The streaming twin: a guard rejection of the buffered prefix also ends
/// the attempt after the provider billed it, so its failed row exists even
/// though the attempt never committed as `Streaming`.
#[tokio::test]
async fn a_guard_rejected_stream_still_writes_a_failed_attempt_row() {
    let primary = spawn_upstream(upstream_ok_stream()).await;
    let secondary = spawn_upstream(upstream_ok_stream()).await;
    let cooldowns = Arc::new(CooldownStore::new());
    let bus = Arc::new(RecordingBus::new());
    let engine = build_guarded_engine(
        Arc::new(RejectingStreamGuard),
        &cooldowns,
        &bus,
        &[primary, secondary],
    );

    let error = engine
        .execute_stream(engine_request("both", true))
        .await
        .expect_err("the guard refuses the streamed prefix");
    assert!(
        matches!(error, EngineError::InvalidRequest(_)),
        "the guard rejection surfaces as InvalidRequest, got {error:?}"
    );

    let rows = attempts(&bus);
    assert_eq!(rows.len(), 1, "the billed attempt writes exactly one row");
    assert_eq!(rows[0].provider, "primary");
    assert_eq!(rows[0].outcome, AttemptOutcome::Failed);
    assert_eq!(rows[0].http_status, Some(400));

    let now = time::OffsetDateTime::now_utc();
    assert!(
        cooldowns.provider_available("primary", now).await,
        "a guard refusal is proxy policy: the member that answered is not cooled"
    );
}
