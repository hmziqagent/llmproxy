//! The shared binding contract (plan 10, step 4): the plan-02 contract
//! fixtures replay through the bridge — the same JSON surface the Python
//! and Node bindings expose — and the results are pinned to checked-in
//! snapshots under `bindings/shared/snapshots/`. The Python and Node test
//! suites replay the *same* cases from `bindings/shared/cases.json` against
//! the *same* snapshots, so a divergence in any language fails its own
//! suite.
//!
//! Regenerate snapshots with `UPDATE_SNAPSHOTS=1 cargo test -p
//! llm-proxy-bridge --test contract`.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use futures::StreamExt;
use llm_proxy_bridge::{Bridge, BridgeError, EventStream, ExecuteOptions};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex as AsyncMutex;

type Capture = Arc<AsyncMutex<Option<(HeaderMap, Value)>>>;

const REPO_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const SHARED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bindings/shared");

struct CannedUpstream {
    base_url: String,
    capture: Capture,
}

/// Serves one canned response (JSON body or SSE event list) for every POST,
/// with `gap` between SSE chunks when set, and captures the last request.
/// The SSE chunk stream: each chunk arrives after the optional gap, so a
/// consumer can cancel between events.
fn sse_body(events: Vec<String>, gap: Option<Duration>) -> Body {
    let stream = futures::stream::iter(events).then(move |chunk| async move {
        if let Some(gap) = gap {
            tokio::time::sleep(gap).await;
        }
        Ok::<axum::body::Bytes, std::convert::Infallible>(axum::body::Bytes::from(chunk))
    });
    Body::from_stream(stream)
}

/// The canned JSON response: optional canned `status` (default 200) and
/// `headers`, with the body serialized canonically (compact, sorted keys —
/// the same bytes `serde_json` produces, which the Python and Node harnesses
/// must match because an error message can embed the raw body text).
fn canned_json_response(response: &Value) -> Response {
    let status = response
        .get("status")
        .and_then(Value::as_u64)
        .and_then(|code| u16::try_from(code).ok())
        .and_then(|code| StatusCode::from_u16(code).ok())
        .unwrap_or(StatusCode::OK);
    let payload =
        serde_json::to_vec(response.get("body").unwrap_or(&Value::Null)).unwrap_or_default();
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(extra) = response.get("headers").and_then(Value::as_object) {
        for (name, value) in extra {
            if let Some(value) = value.as_str() {
                builder = builder.header(name.as_str(), value);
            }
        }
    }
    builder
        .body(Body::from(payload))
        .unwrap_or_else(|error| panic!("canned json response builds: {error}"))
}

async fn spawn_canned(response: &Value, gap: Option<Duration>) -> CannedUpstream {
    let capture: Capture = Arc::new(AsyncMutex::new(None));
    let capture_for_handler = Arc::clone(&capture);
    let response = response.clone();
    let handler = Arc::new(move |headers: HeaderMap, body: axum::body::Bytes| {
        let capture = Arc::clone(&capture_for_handler);
        let response = response.clone();
        async move {
            let json: Value =
                serde_json::from_slice(&body).unwrap_or_else(|_| Value::Object(Default::default()));
            *capture.lock().await = Some((headers, json));
            match response.get("kind").and_then(Value::as_str) {
                Some("json") => {
                    let stall = response
                        .get("stall_ms")
                        .and_then(Value::as_u64)
                        .map(Duration::from_millis);
                    if let Some(stall) = stall {
                        tokio::time::sleep(stall).await;
                    }
                    canned_json_response(&response)
                }
                Some("sse") => {
                    let events: Vec<String> = response
                        .get("events")
                        .and_then(Value::as_array)
                        .map(|list| {
                            list.iter()
                                .filter_map(Value::as_str)
                                .map(str::to_owned)
                                .collect()
                        })
                        .unwrap_or_default();
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        sse_body(events, gap),
                    )
                        .into_response()
                }
                other => panic!("canned response must be kind json or sse, got {other:?}"),
            }
        }
    });
    let app = Router::new().route(
        "/{*path}",
        post(move |headers, body| {
            let handler = Arc::clone(&handler);
            async move { handler(headers, body).await }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "canned upstream stopped");
        }
    });
    CannedUpstream {
        base_url: format!("http://{addr}"),
        capture,
    }
}

async fn captured(upstream: &CannedUpstream) -> Option<(HeaderMap, Value)> {
    upstream.capture.lock().await.clone()
}

fn load_shared() -> Value {
    let text = std::fs::read_to_string(format!("{SHARED}/cases.json"))
        .expect("bindings/shared/cases.json is readable");
    serde_json::from_str(&text).expect("cases.json parses")
}

fn fixture_body(fixture_root: &str, relative: &str) -> Value {
    let text = std::fs::read_to_string(format!("{REPO_ROOT}/{fixture_root}/{relative}"))
        .unwrap_or_else(|error| panic!("read fixture {relative}: {error}"));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse fixture {relative}: {error}"))
}

/// Substitutes `{upstream}` in a provider template with the base URL.
fn provider_for(shared: &Value, name: &str, base_url: &str) -> Value {
    let template = shared
        .get("providers")
        .and_then(|providers| providers.get(name))
        .unwrap_or_else(|| panic!("provider template {name} exists"));
    serde_json::to_string(template)
        .map(|text| text.replace("{upstream}", base_url))
        .and_then(|text| serde_json::from_str(&text))
        .expect("provider template round-trips")
}

fn header_subset(shared: &Value, headers: &HeaderMap) -> Value {
    let mut subset = Map::new();
    for name in shared
        .get("header_subset")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if let Some(value) = headers.get(name).and_then(|v| v.to_str().ok()) {
            subset.insert(name.to_owned(), Value::String(value.to_owned()));
        }
    }
    Value::Object(subset)
}

fn error_value(error: &BridgeError) -> Value {
    json!({
        "kind": error.kind,
        "status": error.status,
        "message": error.message,
        "retry_after_secs": error.retry_after.map(|d| d.as_secs_f64()),
        "should_retry": error.should_retry,
    })
}

async fn drain(stream: &EventStream) -> Vec<Value> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

async fn run_case(shared: &Value, case: &Value) -> Value {
    let response_name = case
        .get("upstream_response")
        .and_then(Value::as_str)
        .expect("case names its upstream response");
    let canned = shared
        .get("upstream_responses")
        .and_then(|responses| responses.get(response_name))
        .unwrap_or_else(|| panic!("canned response {response_name} exists"));
    let provider_name = case
        .get("provider")
        .and_then(Value::as_str)
        .expect("provider");
    let dialect = case
        .get("dialect")
        .and_then(Value::as_str)
        .expect("dialect");
    let fixture = case
        .get("fixture")
        .and_then(Value::as_str)
        .expect("fixture");
    let stream = case.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let fixture_root = shared
        .get("fixture_root")
        .and_then(Value::as_str)
        .expect("fixture_root");

    let upstream = spawn_canned(canned, None).await;
    let config = json!({
        "record_usage": false,
        "providers": [provider_for(shared, provider_name, &upstream.base_url)],
    });
    let bridge = Bridge::from_json(&config).expect("bridge builds for the case");
    let body = fixture_body(fixture_root, fixture);
    let options = case
        .get("options")
        .map(|options| ExecuteOptions::from_json(options).expect("case options parse"))
        .unwrap_or_default();

    let outcome: Result<Vec<Value>, BridgeError> = if stream {
        match bridge
            .execute_stream(provider_name, dialect, &body, &options)
            .await
        {
            Ok(stream) => Ok(drain(&stream).await),
            Err(error) => Err(error),
        }
    } else {
        bridge
            .execute(provider_name, dialect, &body, &options)
            .await
            .map(|response| vec![response])
    };

    let (response, stream_events, error) = match outcome {
        Ok(mut events) => {
            let response = if stream { None } else { Some(events.pop()) };
            (response.flatten(), Some(events).filter(|_| stream), None)
        }
        Err(error) => (None, None, Some(error_value(&error))),
    };

    let (upstream_request, upstream_headers) = match captured(&upstream).await {
        Some((headers, body)) => (Some(body), Some(header_subset(shared, &headers))),
        None => (None, None),
    };

    json!({
        "upstream_request": upstream_request,
        "upstream_headers": upstream_headers,
        "response": response,
        "stream_events": stream_events,
        "error": error,
    })
}

fn run_translate(shared: &Value, case: &Value) -> Value {
    let fixture_root = shared
        .get("fixture_root")
        .and_then(Value::as_str)
        .expect("fixture_root");
    let body = case
        .get("fixture")
        .and_then(Value::as_str)
        .map(|fixture| fixture_body(fixture_root, fixture))
        .or_else(|| case.get("body").cloned())
        .expect("translate case carries a fixture or inline body");
    let from = case.get("from").and_then(Value::as_str).expect("from");
    let to = case.get("to").and_then(Value::as_str).expect("to");

    // Translation does no I/O; an offline bridge proves it.
    let config = json!({
        "providers": [{
            "name": "offline",
            "api_key": "k",
            "auth_style": "bearer",
            "adapters": {
                "a": { "protocol": "anthropic_messages", "endpoint": "http://127.0.0.1:9/v1/messages" }
            },
            "routes": { "messages": "a" }
        }]
    });
    let bridge = Bridge::from_json(&config).expect("bridge builds");
    let result = match case.get("mode").and_then(Value::as_str) {
        Some("response") => bridge.translate_response(&body, from, to, None),
        Some("request") => bridge.translate_request(&body, from, to, None),
        other => panic!("translate case must set mode to request or response, got {other:?}"),
    };
    match result {
        Ok(value) => json!({ "result": value, "error": null }),
        Err(error) => json!({ "result": null, "error": error_value(&error) }),
    }
}

/// The shared normalization every harness applies before comparing: JSON
/// numbers under a `created` key are wall-clock stamps the chat encoder
/// writes, so they become a marker (see `bindings/shared/cases.json`).
fn normalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| {
                    let normalized = if key == "created" && value.is_number() {
                        Value::String("<created-timestamp>".to_owned())
                    } else {
                        normalize(value)
                    };
                    (key.clone(), normalized)
                })
                .collect(),
        ),
        Value::Array(list) => Value::Array(list.iter().map(normalize).collect()),
        other => other.clone(),
    }
}

fn compare_or_write_snapshot(name: &str, snapshot: &Value) {
    let snapshot = normalize(snapshot);
    let path = format!("{SHARED}/snapshots/{name}.json");
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        let text = format!(
            "{}\n",
            serde_json::to_string_pretty(&snapshot).expect("serialize")
        );
        std::fs::write(&path, text)
            .unwrap_or_else(|error| panic!("write snapshot {name}: {error}"));
        return;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("read snapshot {name} (run with UPDATE_SNAPSHOTS=1): {error}")
    });
    let expected: Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parse snapshot {name}: {error}"));
    assert_eq!(
        &snapshot, &expected,
        "binding contract diverged for {name}; if the change is intended, regenerate with \
         UPDATE_SNAPSHOTS=1 and review the diff"
    );
}

#[tokio::test]
async fn plan02_contract_fixtures_match_the_snapshots_through_the_bridge() {
    let shared = load_shared();
    for case in shared
        .get("cases")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = case.get("name").and_then(Value::as_str).expect("case name");
        let snapshot = run_case(&shared, case).await;
        compare_or_write_snapshot(name, &snapshot);
    }
}

/// Upstream error responses replay through every language binding with the
/// same retry-policy fields (`kind`, `status`, `message`, `retry_after_secs`,
/// `should_retry`), pinned to the shared `error-*.json` snapshots (fix brief
/// 10-8). Both canned providers are bare single-attempt routes, so the raw
/// upstream error surfaces: the 429 keeps its `Retry-After` as
/// `retry_after_secs` and its policy row says retry, and the 502 says retry
/// with no advised wait.
#[tokio::test]
async fn error_cases_match_the_shared_snapshots() {
    let shared = load_shared();
    for case in shared
        .get("errors")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let snapshot_name = case
            .get("snapshot")
            .and_then(Value::as_str)
            .expect("error case names its snapshot");
        let outcome = run_case(&shared, case).await;
        let error = outcome
            .get("error")
            .filter(|error| !error.is_null())
            .expect("an upstream error case must fail with a bridge error");
        compare_or_write_snapshot(snapshot_name, error);
    }
}

#[test]
fn translation_cases_match_the_snapshots() {
    let shared = load_shared();
    for case in shared
        .get("translates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = case.get("name").and_then(Value::as_str).expect("case name");
        let snapshot = run_translate(&shared, case);
        compare_or_write_snapshot(name, &snapshot);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_settles_client_cancelled_and_the_bridge_stays_usable() {
    let shared = load_shared();
    let cancellation = shared
        .get("cancellation")
        .expect("cancellation scenario present");
    let responses = shared.get("upstream_responses").expect("responses");
    let stream_canned = responses
        .get(
            cancellation
                .get("upstream_response")
                .and_then(Value::as_str)
                .expect("response"),
        )
        .expect("canned stream exists");
    let gap = Duration::from_millis(
        cancellation
            .get("gap_ms")
            .and_then(Value::as_u64)
            .unwrap_or(250),
    );
    let settle_timeout = Duration::from_secs(
        cancellation
            .get("settle_timeout_secs")
            .and_then(Value::as_u64)
            .unwrap_or(5),
    );
    // The follow-up provider answers with the Anthropic JSON body; reuse any
    // JSON canned response through the same provider template substitution.
    let json_canned = responses
        .get("anthropic_json")
        .expect("anthropic_json exists");

    let sse_upstream = spawn_canned(stream_canned, Some(gap)).await;
    let json_upstream = spawn_canned(json_canned, None).await;
    let stream_provider = cancellation
        .get("stream_provider")
        .and_then(Value::as_str)
        .expect("stream provider");
    let followup_provider = cancellation
        .get("followup_provider")
        .and_then(Value::as_str)
        .expect("followup provider");
    let config = json!({
        "record_usage": true,
        "providers": [
            provider_for(&shared, stream_provider, &sse_upstream.base_url),
            provider_for(&shared, followup_provider, &json_upstream.base_url),
        ],
    });
    let bridge = Bridge::from_json(&config).expect("bridge builds");
    let request = cancellation.get("request").expect("request body").clone();
    let dialect = cancellation
        .get("dialect")
        .and_then(Value::as_str)
        .expect("dialect");

    let stream = bridge
        .execute_stream(
            stream_provider,
            dialect,
            &request,
            &ExecuteOptions::default(),
        )
        .await
        .expect("stream starts");
    let first = stream.next().await.expect("first event arrives");
    assert!(first.get("MessageStart").is_some(), "got: {first}");

    stream.cancel();
    assert!(stream.next().await.is_none(), "cancel stops delivery");

    let deadline = tokio::time::Instant::now() + settle_timeout;
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
            "the ClientCancelled row must land, got: {:?}",
            bridge.usage_events()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let followup_request = json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "again" }],
    });
    let response = bridge
        .execute(
            followup_provider,
            dialect,
            &followup_request,
            &ExecuteOptions::default(),
        )
        .await
        .expect("bridge usable after cancellation");
    assert!(response.get("id").is_some(), "got: {response}");
}
