//! Engine integration tests against a loopback fake upstream: the full
//! pipeline runs without the proxy binding a port (plan 03's first
//! done-when).

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::post;
use futures::StreamExt;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
    ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_engine::{ClientDialect, Engine, EngineBuilder, EngineRequest, InboundRequest};
use llm_proxy_protocol::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreRole, ModelRef, SamplingOptions,
};
use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};

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

fn upstream_router_anthropic_nonstream() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello from upstream"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
            )
        }),
    )
}

fn upstream_router_anthropic_stream_truncated() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_t","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
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

fn provider(endpoint: String) -> ProviderConfig {
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
        name: "fake".to_owned(),
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

fn engine_request(stream: bool) -> EngineRequest {
    engine_request_with_core(core_request(stream))
}

fn engine_request_with_core(core: CoreRequest) -> EngineRequest {
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core,
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id("req-engine-test")
    .with_body_hash("hash")
}

fn build_engine(endpoint: String, bus: Arc<RecordingBus>) -> Engine {
    build_engine_for(provider(endpoint), bus)
}

fn build_engine_for(provider: ProviderConfig, bus: Arc<RecordingBus>) -> Engine {
    let registry = ProviderRegistry::from_providers(vec![provider]).expect("provider registry");
    EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(bus as Arc<dyn EventBus>)
        .build()
        .expect("engine builds")
}

/// A fake upstream that records every request body it receives and always
/// answers with the same successful Anthropic body.
fn upstream_capturing_body(seen: Arc<std::sync::Mutex<Vec<serde_json::Value>>>) -> Router {
    Router::new().route(
        "/{*path}",
        post(move |body: axum::body::Bytes| async move {
            let value = serde_json::from_slice::<serde_json::Value>(&body)
                .unwrap_or(serde_json::Value::Null);
            seen.lock().expect("captured bodies lock").push(value);
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello from upstream"}],"model":"real-model","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
            )
        }),
    )
}

#[tokio::test]
async fn non_streaming_pipeline_reaches_the_fake_upstream_and_settles() {
    let upstream = spawn_upstream(upstream_router_anthropic_nonstream()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(upstream, Arc::clone(&bus));

    let response = engine
        .execute(engine_request(false))
        .await
        .expect("execute ok");
    let text = response
        .core
        .content
        .iter()
        .find_map(|content| match content {
            CoreContent::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default();
    assert_eq!(text, "hello from upstream");
    assert_eq!(response.core.usage.input_tokens, 10);
    assert_eq!(response.core.usage.output_tokens, 4);

    // Settlement: RequestReceived + attempt + ResponseCompleted rows on
    // the bus, in order.
    let snapshot = bus.snapshot();
    let mut kinds = snapshot.iter().map(|event| match event {
        ProxyEvent::RequestReceived(_) => "received",
        ProxyEvent::ResponseCompleted(_) => "completed",
        ProxyEvent::ResponseFailed(_) => "failed",
        ProxyEvent::AttemptCompleted(_) => "attempt",
    });
    assert_eq!(kinds.next(), Some("received"));
    assert_eq!(kinds.next(), Some("attempt"));
    assert_eq!(kinds.next(), Some("completed"));
    assert!(kinds.next().is_none(), "no failure rows expected");
}

#[tokio::test]
async fn streaming_pipeline_yields_core_events_and_marks_truncation() {
    let upstream = spawn_upstream(upstream_router_anthropic_stream_truncated()).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(upstream, Arc::clone(&bus));

    let stream = engine
        .execute_stream(engine_request(true))
        .await
        .expect("stream starts");
    assert!(
        !stream.head.is_empty(),
        "the head crosses the first-byte window"
    );

    let mut all = stream.head.clone();
    let mut tail = stream.tail;
    while let Some(event) = tail.next().await {
        all.push(event);
    }

    // The prefix, the truncation error, and the synthetic terminal all
    // reach the caller: nothing is dropped after commit.
    assert!(
        all.iter()
            .any(|event| matches!(event, CoreEvent::MessageStart { .. }))
    );
    assert!(
        all.iter()
            .any(|event| matches!(event, CoreEvent::Error { .. })),
        "the truncation error must be yielded in-band, got: {all:?}"
    );
    assert!(
        all.iter()
            .any(|event| matches!(event, CoreEvent::MessageStop { .. })),
        "the synthetic terminal must follow the truncation error, got: {all:?}"
    );

    // Settlement marks the stream failed.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let snapshot = bus.snapshot();
    assert!(
        snapshot
            .iter()
            .any(|event| matches!(event, ProxyEvent::ResponseFailed(_))),
        "a truncated stream must record a failure row"
    );
}

/// The pause the fake upstream takes before each SSE chunk. An engine that
/// buffered the stream before yielding could not deliver anything until
/// every chunk had arrived; incremental delivery hands the first event over
/// after one gap and each later event within one gap of its predecessor.
const UPSTREAM_CHUNK_GAP: Duration = Duration::from_millis(300);
/// The most `execute_stream` may take: one chunk gap to commit the first
/// event plus headroom — comfortably below the four gaps a whole-stream
/// buffer would need.
const START_DEADLINE: Duration = Duration::from_millis(750);
/// The most one tail event may lag the previous: comfortably above one
/// chunk gap, comfortably below the three a tail dumped only after the
/// upstream ended would impose on its first event.
const PER_EVENT_DEADLINE: Duration = Duration::from_millis(600);

#[tokio::test]
async fn streaming_pipeline_delivers_events_incrementally_not_buffered() {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            let chunks = vec![
                bytes::Bytes::from(
                    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_i\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
                ),
                bytes::Bytes::from(
                    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"a\"}}\n\n",
                ),
                bytes::Bytes::from(
                    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
                ),
                bytes::Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
            ];
            let body_stream = futures::stream::iter(chunks).then(|chunk| async move {
                tokio::time::sleep(UPSTREAM_CHUNK_GAP).await;
                Ok::<bytes::Bytes, std::convert::Infallible>(chunk)
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(body_stream).into_response(),
            )
        }),
    );
    let upstream = spawn_upstream(app).await;
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(upstream, Arc::clone(&bus));

    // The first chunk must commit the stream: a whole-stream buffer would
    // hold this future until the last chunk had been sent.
    let stream = tokio::time::timeout(START_DEADLINE, engine.execute_stream(engine_request(true)))
        .await
        .expect("execute_stream resolves after the first chunk, not the whole stream")
        .expect("stream starts");
    assert!(
        stream
            .head
            .iter()
            .any(|event| matches!(event, CoreEvent::MessageStart { .. })),
        "the first chunk commits the stream, got head: {:?}",
        stream.head
    );

    // Each later event must arrive within one chunk gap of the previous
    // one; the deadline cannot be met by a tail that is withheld until the
    // upstream has finished sending.
    let mut all = stream.head;
    let mut tail = stream.tail;
    loop {
        let next = tokio::time::timeout(PER_EVENT_DEADLINE, tail.next())
            .await
            .expect("the next event arrives before the per-event deadline: the engine buffered");
        match next {
            Some(event) => all.push(event),
            None => break,
        }
    }

    let kinds: Vec<&str> = all
        .iter()
        .map(|event| match event {
            CoreEvent::MessageStart { .. } => "message_start",
            CoreEvent::TextDelta { .. } => "text_delta",
            CoreEvent::UsageDelta { .. } => "usage_delta",
            CoreEvent::MessageStop { .. } => "message_stop",
            other => {
                panic!("unexpected event in a clean four-chunk stream: {other:?}");
            }
        })
        .collect();
    assert_eq!(
        kinds,
        ["message_start", "text_delta", "usage_delta", "message_stop"],
        "every upstream event reaches the caller exactly once, in order"
    );

    tokio::time::sleep(Duration::from_millis(50)).await;
    let snapshot = bus.snapshot();
    assert!(
        snapshot.iter().any(
            |event| matches!(event, ProxyEvent::ResponseCompleted(c) if c.usage.input_tokens == 3)
        ),
        "the completed row carries provider-reported usage"
    );
}

#[tokio::test]
async fn unknown_provider_fails_without_egress() {
    let bus = Arc::new(RecordingBus::new());
    let engine = build_engine(
        "http://127.0.0.1:9/v1/messages".to_owned(),
        Arc::clone(&bus),
    );
    let request = EngineRequest::new(
        "nope",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core_request(false),
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    );
    let error = engine.execute(request).await.expect_err("unknown provider");
    assert!(matches!(
        error,
        llm_proxy_engine::EngineError::UnknownProvider(_)
    ));
}

/// The alias maps the client's `fast` to `real-model`: the rewritten model
/// reaches the upstream body, and the ledger row records it as the request's
/// upstream model. These are the observable counterparts of the engine's
/// `upstream_model` tracing field (fix brief R-10); the log field itself has
/// no assertion here — capturing tracing output is not worth a subscriber in
/// this suite, so the field is checked by the `upstream_model = %` grep in
/// the item's done-when instead.
#[tokio::test]
async fn aliased_model_reaches_upstream_as_upstream_model() {
    let captured = Arc::new(std::sync::Mutex::new(Vec::<serde_json::Value>::new()));
    let upstream = spawn_upstream(upstream_capturing_body(Arc::clone(&captured))).await;
    let bus = Arc::new(RecordingBus::new());
    let mut provider_config = provider(upstream);
    provider_config
        .model_aliases
        .insert("fast".to_owned(), "real-model".to_owned());
    let engine = build_engine_for(provider_config, Arc::clone(&bus));

    let mut core = core_request(false);
    core.model = ModelRef {
        requested: "fast".to_owned(),
        upstream: None,
    };
    engine
        .execute(engine_request_with_core(core))
        .await
        .expect("execute ok");

    let bodies = captured.lock().expect("captured bodies lock");
    assert_eq!(bodies.len(), 1, "exactly one upstream request");
    assert_eq!(
        bodies[0].get("model").and_then(|model| model.as_str()),
        Some("real-model"),
        "the alias must be rewritten before the body reaches the upstream"
    );

    let ledger_upstream_model = bus.snapshot().into_iter().find_map(|event| match event {
        ProxyEvent::ResponseCompleted(completed) => Some(completed.model.upstream),
        _ => None,
    });
    assert_eq!(
        ledger_upstream_model,
        Some(Some("real-model".to_owned())),
        "the ledger row must name the upstream model"
    );
}
