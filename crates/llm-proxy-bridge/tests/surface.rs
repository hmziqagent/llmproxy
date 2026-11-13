//! Behaviour tests for the shared binding surface (plan 10, step 1):
//! configuration from code, translation, request execution, streaming and
//! cancellation — all observable through the JSON boundary the Python and
//! Node bindings wrap. The upstream is a loopback fake; no real provider is
//! contacted.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::post;
use futures::StreamExt;
use llm_proxy_bridge::{Bridge, EventStream, ExecuteOptions};
use serde_json::{Value, json};
use tokio::sync::Mutex as AsyncMutex;

type Capture = Arc<AsyncMutex<Option<(HeaderMap, Value)>>>;

struct FakeUpstream {
    base_url: String,
    capture: Capture,
}

/// Serves both chat families with fixed bodies and captures the last request.
async fn spawn_fake_upstream() -> FakeUpstream {
    let capture: Capture = Arc::new(AsyncMutex::new(None));
    let capture_for_handler = Arc::clone(&capture);
    // `anthropic_side` selects the response family per route: /v1/messages
    // answers Anthropic, /v1/chat/completions answers chat completions, like
    // a real upstream.
    let handler = Arc::new(
        move |anthropic_side: bool, headers: HeaderMap, body: axum::body::Bytes| {
            let capture = Arc::clone(&capture_for_handler);
            async move {
                let json: Value = serde_json::from_slice(&body)
                    .unwrap_or_else(|_| Value::Object(Default::default()));
                *capture.lock().await = Some((headers, json));
                let (content_type, bytes) = if anthropic_side {
                    (
                    "application/json",
                    r#"{"id":"msg_bridge_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello from the fake"}],"model":"claude-sonnet-4-5","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":3},"upstream_custom_field":"kept"}"#.as_bytes().to_owned(),
                )
                } else {
                    (
                    "application/json",
                    r#"{"id":"chatcmpl-bridge-1","object":"chat.completion","created":1770000000,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hello from the fake"},"finish_reason":"stop"}],"usage":{"prompt_tokens":21,"completion_tokens":4,"total_tokens":25},"upstream_custom_field":"kept"}"#.as_bytes().to_owned(),
                )
                };
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, content_type)],
                    bytes,
                )
                    .into_response()
            }
        },
    );
    let app = Router::new()
        .route(
            "/v1/messages",
            post({
                let handler = Arc::clone(&handler);
                move |headers, body| handler(true, headers, body)
            }),
        )
        .route(
            "/v1/chat/completions",
            post({
                let handler = Arc::clone(&handler);
                move |headers, body| handler(false, headers, body)
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    FakeUpstream {
        base_url: format!("http://{addr}"),
        capture,
    }
}

fn bridge_config(upstream: &FakeUpstream, record_usage: bool) -> Value {
    json!({
        "record_usage": record_usage,
        "providers": [
            {
                "name": "acme",
                "api_key": "operator-key",
                "auth_style": "bearer",
                "adapters": {
                    "messages": {
                        "protocol": "anthropic_messages",
                        "endpoint": format!("{}/v1/messages", upstream.base_url)
                    }
                },
                "routes": { "messages": "messages", "chat_completions": "messages" },
                "model_aliases": { "claude-sonnet-4-5": "claude-sonnet-4-5-20250929" }
            },
            {
                "name": "openai-co",
                "api_key": "operator-key",
                "auth_style": "bearer",
                "adapters": {
                    "chat": {
                        "protocol": "openai_chat_completions",
                        "endpoint": format!("{}/v1/chat/completions", upstream.base_url)
                    }
                },
                "routes": { "chat_completions": "chat" }
            }
        ]
    })
}

fn anthropic_body() -> Value {
    json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": "be terse",
        "messages": [{ "role": "user", "content": "hi" }],
        "stop_sequences": ["\n\nHuman:"],
        "custom_future_field": { "unknown": true }
    })
}

fn openai_body() -> Value {
    json!({
        "model": "gpt-4o",
        "messages": [
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": "hi" }
        ]
    })
}

async fn captured(upstream: &FakeUpstream) -> (HeaderMap, Value) {
    let guard = upstream.capture.lock().await;
    guard.clone().expect("upstream received a request")
}

#[tokio::test]
async fn execute_runs_the_pipeline_and_returns_the_caller_dialect() {
    let upstream = spawn_fake_upstream().await;
    let bridge = Bridge::from_json(&bridge_config(&upstream, false)).expect("bridge builds");

    let response = bridge
        .execute(
            "acme",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("execute succeeds");

    // Same-format passthrough: the upstream body returns verbatim, unknown
    // upstream fields and all.
    assert_eq!(response["id"], "msg_bridge_1");
    assert_eq!(response["upstream_custom_field"], "kept");
    assert_eq!(response["usage"]["input_tokens"], 11);

    // The model alias reached the upstream; the stop sequence survived; the
    // unknown client field survived same-format passthrough.
    let (_, upstream_body) = captured(&upstream).await;
    assert_eq!(upstream_body["model"], "claude-sonnet-4-5-20250929");
    assert_eq!(upstream_body["stop_sequences"][0], "\n\nHuman:");
    assert_eq!(upstream_body["custom_future_field"]["unknown"], true);
}

#[tokio::test]
async fn execute_cross_format_encodes_the_caller_dialect_back() {
    let upstream = spawn_fake_upstream().await;
    let bridge = Bridge::from_json(&bridge_config(&upstream, false)).expect("bridge builds");

    // An OpenAI-chat caller reaching the Anthropic upstream: the request is
    // re-encoded, the response comes back as a chat completion.
    let response = bridge
        .execute(
            "acme",
            "openai_chat_completions",
            &openai_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("cross-format execute succeeds");

    assert_eq!(response["object"], "chat.completion");
    assert_eq!(
        response["choices"][0]["message"]["content"],
        "hello from the fake"
    );
    // Same-format passthrough cannot apply; the client-visible response is
    // the core re-encoded into the caller's dialect.
    assert!(response.get("upstream_custom_field").is_none());

    let (_, upstream_body) = captured(&upstream).await;
    assert_eq!(
        upstream_body["messages"][0]["content"][0]["text"], "hi",
        "the user message re-encodes into Anthropic blocks"
    );
}

#[tokio::test]
async fn forwarded_headers_reach_the_upstream_and_credentials_do_not() {
    let upstream = spawn_fake_upstream().await;
    let bridge = Bridge::from_json(&bridge_config(&upstream, false)).expect("bridge builds");

    let options = ExecuteOptions {
        model: None,
        headers: [
            ("anthropic-beta", "context-1m-2025-08-07"),
            ("x-claude-code-session-id", "sess-1"),
            ("authorization", "Bearer client-secret-token"),
            ("origin", "https://evil.example"),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect(),
    };
    bridge
        .execute("acme", "anthropic_messages", &anthropic_body(), &options)
        .await
        .expect("execute succeeds");

    let (headers, _) = captured(&upstream).await;
    assert_eq!(
        headers.get("anthropic-beta").and_then(|v| v.to_str().ok()),
        Some("context-1m-2025-08-07"),
        "the beta header is allowlisted through"
    );
    assert_eq!(
        headers
            .get("x-claude-code-session-id")
            .and_then(|v| v.to_str().ok()),
        Some("sess-1")
    );
    // The operator's key goes upstream, never the client's token; Origin is
    // never forwarded.
    assert_eq!(
        headers.get("authorization").and_then(|v| v.to_str().ok()),
        Some("Bearer operator-key")
    );
    assert!(headers.get("origin").is_none());
}

/// Serves one fixed JSON error status on both chat families, with the given
/// `retry-after` header, so a bare single-attempt route surfaces the raw
/// upstream failure with its policy fields.
async fn spawn_status_upstream(status: u16, retry_after: Option<&str>) -> String {
    let body = r#"{"error":{"message":"upstream refused the request","type":"api_error"}}"#;
    let retry_after = retry_after.map(str::to_owned);
    let app = Router::new().route(
        "/{*path}",
        post(move || async move {
            let code = StatusCode::from_u16(status).expect("a valid status for the fake");
            let mut response = (
                code,
                [(header::CONTENT_TYPE, "application/json")],
                body.to_owned(),
            )
                .into_response();
            if let Some(advised) = retry_after
                && let Ok(value) = axum::http::HeaderValue::from_str(&advised)
            {
                response.headers_mut().insert(header::RETRY_AFTER, value);
            }
            response
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "status upstream stopped");
        }
    });
    format!("http://{addr}")
}

/// The retry-policy fields must survive the bridge boundary on the errors
/// the engine maps from upstream statuses (fix brief 10-8): a 429 keeps its
/// `Retry-After` as `retry_after` and its policy row's `should_retry: true`,
/// and a bare 502 keeps its row's `true` with no advised wait. Both
/// providers here are single-attempt routes, so no ladder wraps the error
/// into `RetryExhausted` (`false`), which is what a dropped or mis-copied
/// field would degrade to.
#[tokio::test]
async fn retryable_upstream_failures_carry_should_retry_and_retry_after() {
    let rate_limited = spawn_status_upstream(429, Some("7")).await;
    let gateway_fault = spawn_status_upstream(502, None).await;
    let config = json!({
        "providers": [
            {
                "name": "acme",
                "api_key": "operator-key",
                "auth_style": "bearer",
                "adapters": {
                    "messages": {
                        "protocol": "anthropic_messages",
                        "endpoint": format!("{rate_limited}/v1/messages")
                    }
                },
                "routes": { "messages": "messages" }
            },
            {
                "name": "openai-co",
                "api_key": "operator-key",
                "auth_style": "bearer",
                "adapters": {
                    "chat": {
                        "protocol": "openai_chat_completions",
                        "endpoint": format!("{gateway_fault}/v1/chat/completions")
                    }
                },
                "routes": { "chat_completions": "chat" }
            }
        ]
    });
    let bridge = Bridge::from_json(&config).expect("bridge builds");

    let limited = bridge
        .execute(
            "acme",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect_err("a 429 upstream fails the request");
    assert_eq!(limited.status, 429);
    assert_eq!(limited.kind, "Upstream");
    assert!(limited.should_retry, "the 429 policy row says retry");
    assert_eq!(
        limited.retry_after,
        Some(Duration::from_secs(7)),
        "the upstream Retry-After crosses the bridge"
    );

    let gateway = bridge
        .execute(
            "openai-co",
            "openai_chat_completions",
            &openai_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect_err("a 502 upstream fails the request");
    assert_eq!(gateway.status, 502);
    assert!(gateway.should_retry, "the 502 policy row says retry");
    assert_eq!(gateway.retry_after, None, "no header, no advised wait");
}

#[tokio::test]
async fn failures_carry_kind_status_and_retry_policy() {
    let upstream = spawn_fake_upstream().await;
    let bridge = Bridge::from_json(&bridge_config(&upstream, false)).expect("bridge builds");

    let unknown = bridge
        .execute(
            "nope",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect_err("unknown provider fails");
    assert_eq!(unknown.kind, "UnknownProvider");
    assert_eq!(unknown.status, 404);
    assert!(!unknown.should_retry);

    let bad_dialect = bridge
        .execute(
            "acme",
            "mqtt",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect_err("unknown dialect fails");
    assert_eq!(bad_dialect.status, 400);
    assert!(bad_dialect.message.contains("mqtt"));

    let bad_body = bridge
        .execute(
            "acme",
            "anthropic_messages",
            &json!({ "no": "model here" }),
            &ExecuteOptions::default(),
        )
        .await
        .expect_err("body without a model fails");
    assert_eq!(bad_body.kind, "InvalidRequest");
    assert_eq!(bad_body.status, 400);
}

#[test]
fn invalid_configuration_is_rejected_with_names() {
    let bad_protocol = json!({
        "providers": [{
            "name": "broken",
            "api_key": "k",
            "auth_style": "bearer",
            "adapters": { "a": { "protocol": "carrier-pigeon", "endpoint": "http://x" } },
            "routes": { "messages": "a" }
        }]
    });
    let error = Bridge::from_json(&bad_protocol).expect_err("unknown protocol rejected");
    assert_eq!(error.kind, "InvalidConfig");
    assert!(error.message.contains("carrier-pigeon"));

    let bad_shape = json!({ "providers": "not-a-list" });
    let error = Bridge::from_json(&bad_shape).expect_err("wrong config shape rejected");
    assert_eq!(error.kind, "InvalidConfig");
}

#[test]
fn request_timeout_secs_zero_is_a_config_error() {
    let mut config = offline_bridge_config();
    config["request_timeout_secs"] = json!(0);
    let error = Bridge::from_json(&config).expect_err("a zero request timeout is a config error");
    assert_eq!(error.kind, "InvalidConfig");
    assert!(
        error
            .message
            .contains("request_timeout_secs must be at least 1")
    );
}

/// Serves nothing for 30 s, then a JSON body: a non-streaming execute parks
/// on it, the shape a whole-request deadline has to cut short.
async fn spawn_stalling_json_upstream() -> String {
    let app = Router::new().route(
        "/v1/messages",
        post(|| async {
            tokio::time::sleep(Duration::from_secs(30)).await;
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"id":"msg_stall","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#,
            )
                .into_response()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "stalling json upstream stopped");
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn request_timeout_secs_bounds_a_stalled_execute() {
    let base_url = spawn_stalling_json_upstream().await;
    let config = json!({
        "request_timeout_secs": 1,
        "providers": [{
            "name": "stallco",
            "api_key": "operator-key",
            "auth_style": "bearer",
            "adapters": {
                "messages": {
                    "protocol": "anthropic_messages",
                    "endpoint": format!("{base_url}/v1/messages")
                }
            },
            "routes": { "messages": "messages" }
        }]
    });
    let bridge = Bridge::from_json(&config).expect("bridge builds");

    let started = std::time::Instant::now();
    let attempt = tokio::time::timeout(
        Duration::from_secs(5),
        bridge.execute(
            "stallco",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        ),
    )
    .await
    .expect("the whole-request deadline must end a stalled execute within 5s");
    let error = attempt.expect_err("a stalled upstream must fail the execute");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the deadline fired while the request was still parked"
    );

    // A bare single-provider route has no retry ladder, so the deadline
    // surfaces raw: kind UpstreamTimeout, status 504, and the 504 policy row
    // says a client retry may fit the window.
    assert_eq!(error.kind, "UpstreamTimeout");
    assert_eq!(error.status, 504);
    assert!(error.should_retry);
    assert_eq!(error.retry_after, None);
}

#[tokio::test]
async fn an_unset_request_timeout_leaves_the_engine_defaults() {
    let upstream = spawn_fake_upstream().await;
    let bridge = Bridge::from_json(&bridge_config(&upstream, false)).expect("bridge builds");

    let response = bridge
        .execute(
            "acme",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("a fast request still succeeds with no whole-request deadline");
    assert_eq!(response["id"], "msg_bridge_1");
}

/// Translation does no I/O: the provider endpoints point at an unroutable
/// address and nothing fails.
fn offline_bridge_config() -> Value {
    json!({
        "providers": [{
            "name": "offline",
            "api_key": "k",
            "auth_style": "bearer",
            "adapters": {
                "a": { "protocol": "anthropic_messages", "endpoint": "http://127.0.0.1:9/v1/messages" }
            },
            "routes": { "messages": "a" }
        }]
    })
}

fn offline_bridge() -> Bridge {
    Bridge::from_json(&offline_bridge_config()).expect("bridge builds")
}

#[test]
fn translate_request_moves_between_dialects_without_io() {
    let bridge = offline_bridge();

    // Cross-format: a body with only carriable fields.
    let carriable = json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "system": "be terse",
        "messages": [{ "role": "user", "content": "hi" }],
        "stop_sequences": ["\n\nHuman:"]
    });
    let translated = bridge
        .translate_request(
            &carriable,
            "anthropic_messages",
            "openai_chat_completions",
            None,
        )
        .expect("translation succeeds");
    assert_eq!(translated["model"], "claude-sonnet-4-5");
    assert_eq!(
        translated["messages"][0]["role"], "system",
        "the system prompt becomes the first chat message"
    );
    assert_eq!(
        translated["messages"][1]["content"], "hi",
        "content blocks flatten to a string"
    );
    assert_eq!(translated["max_completion_tokens"], 64);
    assert_eq!(translated["stop"], "\n\nHuman:");

    // A field with no chat equivalent is rejected by name, never dropped.
    let error = bridge
        .translate_request(
            &anthropic_body(),
            "anthropic_messages",
            "openai_chat_completions",
            None,
        )
        .expect_err("uncarriable fields reject cross-format translation");
    assert_eq!(error.status, 400);
    assert!(
        error.message.contains("custom_future_field"),
        "the rejection names the field: {error}"
    );

    // Same-format translation is the passthrough path: unknown fields
    // survive untouched.
    let same = bridge
        .translate_request(
            &anthropic_body(),
            "anthropic_messages",
            "anthropic_messages",
            None,
        )
        .expect("same-format translation succeeds");
    assert_eq!(same["custom_future_field"]["unknown"], true);
    assert_eq!(same["stop_sequences"][0], "\n\nHuman:");

    let error = bridge
        .translate_request(&carriable, "anthropic_messages", "smtp", None)
        .expect_err("unknown target protocol rejected");
    assert_eq!(error.status, 400);
}

#[test]
fn translate_request_rejects_uncarriable_fields_by_name() {
    let bridge = offline_bridge();

    let body = json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": [{ "type": "web_search_20250305", "name": "web_search" }]
    });
    let error = bridge
        .translate_request(&body, "anthropic_messages", "openai_chat_completions", None)
        .expect_err("server tools are rejected cross-format");
    assert_eq!(error.status, 400);
    assert!(
        error.message.contains("web_search"),
        "the rejection names the field: {error}"
    );
}

#[test]
fn translate_response_decodes_the_provider_and_encodes_the_client() {
    let bridge = offline_bridge();

    let anthropic_response = json!({
        "id": "msg_t9",
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "text", "text": "translated hello" }],
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 7, "output_tokens": 2 }
    });
    let chat = bridge
        .translate_response(
            &anthropic_response,
            "anthropic_messages",
            "openai_chat_completions",
            None,
        )
        .expect("response translation succeeds");
    assert_eq!(chat["object"], "chat.completion");
    assert_eq!(chat["choices"][0]["message"]["content"], "translated hello");
    assert_eq!(chat["usage"]["prompt_tokens"], 7);

    let error = bridge
        .translate_response(&anthropic_response, "ftp", "openai_chat_completions", None)
        .expect_err("unknown source protocol rejected");
    assert_eq!(error.status, 400);
}

/// Serves an SSE body with a gap between events, so a consumer can cancel
/// mid-stream.
async fn spawn_slow_sse_upstream() -> String {
    let app = Router::new().route(
        "/v1/messages",
        post(|| async {
            let events = [
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s9\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n".to_owned(),
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n".to_owned(),
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"one\"}}\n\n".to_owned(),
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"two\"}}\n\n".to_owned(),
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n".to_owned(),
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_owned(),
            ];
            let stream = futures::stream::iter(events).then(|chunk| async move {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<axum::body::Bytes, std::convert::Infallible>(axum::body::Bytes::from(chunk))
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "slow upstream stopped");
        }
    });
    format!("http://{addr}")
}

/// Serves one SSE event, then holds the connection open for an hour before
/// the second: a consumer past the first event is parked on a stalled
/// upstream, the shape a watchdog `cancel()` has to break.
async fn spawn_stalling_sse_upstream() -> String {
    let app = Router::new().route(
        "/v1/messages",
        post(|| async {
            let first = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s10\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n".to_owned();
            let second =
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_owned();
            let stream = futures::stream::once(async move {
                Ok::<axum::body::Bytes, std::convert::Infallible>(axum::body::Bytes::from(
                    first,
                ))
            })
            .chain(futures::stream::once(async move {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                Ok(axum::body::Bytes::from(second))
            }));
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "stalling upstream stopped");
        }
    });
    format!("http://{addr}")
}

fn streaming_bridge(base_url: &str) -> Bridge {
    let config = json!({
        "record_usage": true,
        "providers": [{
            "name": "streamco",
            "api_key": "operator-key",
            "auth_style": "x-api-key",
            "adapters": {
                "messages": {
                    "protocol": "anthropic_messages",
                    "endpoint": format!("{base_url}/v1/messages")
                }
            },
            "routes": { "messages": "messages" }
        }]
    });
    Bridge::from_json(&config).expect("bridge builds")
}

fn stream_body() -> Value {
    json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "stream": true,
        "messages": [{ "role": "user", "content": "hi" }]
    })
}

#[tokio::test]
async fn streaming_events_flow_and_cancellation_settles_client_cancelled() {
    let bridge = streaming_bridge(&spawn_slow_sse_upstream().await);

    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");

    let first = stream.next().await.expect("first event arrives");
    assert!(
        first.get("MessageStart").is_some(),
        "events are tagged core events, got: {first}"
    );

    // Cancel mid-stream: delivery stops...
    stream.cancel();
    assert!(stream.is_cancelled());
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none(), "cancel is terminal");

    // ...and the engine settles the request as a client cancellation.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        let cancelled = bridge.usage_events().iter().any(|event| {
            event.get("kind").and_then(Value::as_str) == Some("response_failed")
                && event.get("error_kind").and_then(Value::as_str) == Some("ClientCancelled")
        });
        if cancelled {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the ClientCancelled row must land after cancel, got: {:?}",
            bridge.usage_events()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn a_cancelled_bridge_stays_usable() {
    let json_upstream = spawn_fake_upstream().await;
    let sse_base = spawn_slow_sse_upstream().await;
    let config = json!({
        "record_usage": true,
        "providers": [
            {
                "name": "streamco",
                "api_key": "operator-key",
                "auth_style": "x-api-key",
                "adapters": {
                    "messages": {
                        "protocol": "anthropic_messages",
                        "endpoint": format!("{sse_base}/v1/messages")
                    }
                },
                "routes": { "messages": "messages" }
            },
            {
                "name": "acme",
                "api_key": "operator-key",
                "auth_style": "bearer",
                "adapters": {
                    "messages": {
                        "protocol": "anthropic_messages",
                        "endpoint": format!("{}/v1/messages", json_upstream.base_url)
                    }
                },
                "routes": { "messages": "messages" }
            }
        ]
    });
    let bridge = Bridge::from_json(&config).expect("bridge builds");

    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");
    let _ = stream.next().await.expect("first event arrives");
    stream.cancel();

    let response = bridge
        .execute(
            "acme",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("bridge usable after cancellation");
    assert!(response.get("id").is_some(), "got: {response}");
}

#[tokio::test]
async fn usage_events_record_the_request_lifecycle_only_when_opted_in() {
    let upstream = spawn_fake_upstream().await;

    let recording = Bridge::from_json(&bridge_config(&upstream, true)).expect("bridge builds");
    recording
        .execute(
            "acme",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("execute succeeds");
    let events = recording.usage_events();
    let kinds: Vec<&str> = events
        .iter()
        .filter_map(|event| event.get("kind").and_then(Value::as_str))
        .collect();
    assert!(kinds.contains(&"request_received"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"response_completed"), "kinds: {kinds:?}");
    let completed = events
        .iter()
        .find(|event| event.get("kind").and_then(Value::as_str) == Some("response_completed"))
        .expect("completed row present");
    assert_eq!(completed["usage"]["input_tokens"], 11);

    let silent = Bridge::from_json(&bridge_config(&upstream, false)).expect("bridge builds");
    silent
        .execute(
            "acme",
            "anthropic_messages",
            &anthropic_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("execute succeeds");
    assert!(
        silent.usage_events().is_empty(),
        "no usage rows without the opt-in"
    );
}

/// A multi-thread flavor is required: the test thread blocks on the reader
/// below, exactly like a Python interpreter thread, so the runtime the
/// engine's stream task runs on must have workers of its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn next_blocking_yields_the_same_events_off_runtime() {
    let bridge = streaming_bridge(&spawn_slow_sse_upstream().await);

    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");

    // Drain on a plain thread, the way a Python interpreter thread would.
    let (tx, rx) = std::sync::mpsc::channel::<Value>();
    std::thread::spawn(move || {
        while let Some(event) = stream.next_blocking() {
            if tx.send(event).is_err() {
                break;
            }
        }
    });
    let mut collected = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while collected.len() < 6 {
        let event = rx
            .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            .expect("events keep flowing to the blocking reader");
        collected.push(event);
    }
    assert!(
        collected[0].get("MessageStart").is_some(),
        "first event tags a message start: {:?}",
        collected[0]
    );
    assert!(
        collected
            .iter()
            .any(|event| event.get("MessageStop").is_some()),
        "the terminal event reaches the blocking reader"
    );
}

/// Polls the recorded usage rows until the request this stream served is
/// settled as a client cancellation.
async fn assert_settles_client_cancelled(bridge: &Bridge) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        let cancelled = bridge.usage_events().iter().any(|event| {
            event.get("kind").and_then(Value::as_str) == Some("response_failed")
                && event.get("error_kind").and_then(Value::as_str) == Some("ClientCancelled")
        });
        if cancelled {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the ClientCancelled row must land after cancel, got: {:?}",
            bridge.usage_events()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The watchdog pattern the bindings expose: a consumer parked in `next()`
/// on a stalled upstream must be unblocked by a concurrent `cancel()` —
/// which needs no exclusive access — with the request settling as
/// `ClientCancelled`, not at the request timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_unblocks_a_pending_next_and_settles_client_cancelled() {
    let bridge = streaming_bridge(&spawn_stalling_sse_upstream().await);
    let stream = Arc::new(
        bridge
            .execute_stream(
                "streamco",
                "anthropic_messages",
                &stream_body(),
                &ExecuteOptions::default(),
            )
            .await
            .expect("stream starts"),
    );

    let first = Arc::clone(&stream)
        .next()
        .await
        .expect("first event arrives");
    assert!(first.get("MessageStart").is_some(), "got: {first}");

    let waiter = Arc::clone(&stream);
    let pending = tokio::spawn(async move { waiter.next().await });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !pending.is_finished(),
        "next() is parked on the stalled upstream"
    );

    stream.cancel();
    let resolved = tokio::time::timeout(Duration::from_secs(5), pending)
        .await
        .expect("cancel unblocks the pending next")
        .expect("the pending next joins");
    assert!(
        resolved.is_none(),
        "a cancelled stream delivers no more events"
    );
    assert!(stream.is_cancelled());

    assert_settles_client_cancelled(&bridge).await;
}

/// Drains a stream with `next_blocking` on the calling thread, forwarding
/// each event to `tx`; `None` marks the observed end.
fn forward_blocking(stream: &EventStream, tx: &std::sync::mpsc::Sender<Option<Value>>) {
    loop {
        let event = stream.next_blocking();
        let ended = event.is_none();
        if tx.send(event).is_err() || ended {
            return;
        }
    }
}

/// The Python shape of the same pattern: a plain thread blocked in
/// `next_blocking()` on a stalled upstream is released by `cancel()` from
/// another thread.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_unblocks_a_blocked_next_blocking_reader_thread() {
    let bridge = streaming_bridge(&spawn_stalling_sse_upstream().await);
    let stream = Arc::new(
        bridge
            .execute_stream(
                "streamco",
                "anthropic_messages",
                &stream_body(),
                &ExecuteOptions::default(),
            )
            .await
            .expect("stream starts"),
    );

    // The reader thread blocks in next_blocking like a Python interpreter
    // thread, forwarding events (None marks the observed end).
    let (tx, rx) = std::sync::mpsc::channel::<Option<Value>>();
    let reader = {
        let stream = Arc::clone(&stream);
        std::thread::spawn(move || forward_blocking(&stream, &tx))
    };

    let first = rx
        .recv_timeout(Duration::from_secs(3))
        .expect("first event reaches the reader")
        .expect("first event present");
    assert!(first.get("MessageStart").is_some(), "got: {first}");
    assert!(
        rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the reader is parked on the stalled upstream"
    );

    stream.cancel();
    let resolved = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("cancel unblocks the blocked reader");
    assert!(
        resolved.is_none(),
        "a cancelled stream delivers no more events"
    );
    reader
        .join()
        .expect("the reader thread exits cleanly after cancel");

    assert_settles_client_cancelled(&bridge).await;
}

/// Polls the recorded usage rows until the request settled as a completion,
/// then asserts no client-cancellation failure row was recorded at all: a
/// stream the consumer read to its end (or past its terminal event) must
/// stay `Completed`, never become `ClientCancelled` by a drop.
async fn assert_completed_without_client_cancel(bridge: &Bridge) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let events = bridge.usage_events();
        let completed = events
            .iter()
            .any(|event| event.get("kind").and_then(Value::as_str) == Some("response_completed"));
        if completed {
            assert!(
                !events.iter().any(|event| {
                    event.get("kind").and_then(Value::as_str) == Some("response_failed")
                        && event.get("error_kind").and_then(Value::as_str)
                            == Some("ClientCancelled")
                }),
                "a consumed stream must not settle as ClientCancelled: {events:?}"
            );
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the completion row must land after the stream ends, got: {:?}",
            bridge.usage_events()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Dropping the stream without an explicit `cancel()` must still cancel the
/// upstream request: the engine settles `ClientCancelled` promptly, well
/// under the 60 s idle default, instead of letting a stalled upstream keep
/// the request running.
#[tokio::test]
async fn dropping_a_stream_settles_client_cancelled_without_waiting_for_the_upstream() {
    let bridge = streaming_bridge(&spawn_stalling_sse_upstream().await);

    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");
    let first = stream.next().await.expect("first event arrives");
    assert!(first.get("MessageStart").is_some(), "got: {first}");

    drop(stream);
    let started = tokio::time::Instant::now();
    assert_settles_client_cancelled(&bridge).await;
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "the drop must settle promptly (the idle default is 60 s), took {elapsed:?}"
    );
}

/// The same settlement must hold when the consumer never reads a single
/// event: dropping the stream right after `execute_stream` cancels too.
#[tokio::test]
async fn a_stream_dropped_before_the_first_read_is_cancelled() {
    let bridge = streaming_bridge(&spawn_stalling_sse_upstream().await);

    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");
    drop(stream);

    assert_settles_client_cancelled(&bridge).await;
}

/// A stream the consumer read to the end must stay `Completed` when dropped:
/// both the fully drained form (polled to `None`) and the drop right after
/// the terminal event (before the pump observed the end itself).
#[tokio::test]
async fn a_fully_consumed_stream_that_is_dropped_stays_completed() {
    // Drained to `None`, then dropped.
    let bridge = streaming_bridge(&spawn_slow_sse_upstream().await);
    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");
    while stream.next().await.is_some() {}
    drop(stream);
    assert_completed_without_client_cancel(&bridge).await;

    // Dropped right after the terminal event, without polling to `None`.
    let bridge = streaming_bridge(&spawn_slow_sse_upstream().await);
    let stream = bridge
        .execute_stream(
            "streamco",
            "anthropic_messages",
            &stream_body(),
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");
    loop {
        let Some(event) = stream.next().await else {
            panic!("the stream ended before the terminal event");
        };
        if event.get("MessageStop").is_some() {
            break;
        }
    }
    drop(stream);
    assert_completed_without_client_cancel(&bridge).await;
}
