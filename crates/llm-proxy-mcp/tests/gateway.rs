//! The MCP gateway end to end (plan 09 step 6): its own token auth,
//! tools namespaced from a fake upstream MCP server, tool calls
//! forwarded upstream with the revision's headers, and — the point of
//! the plan — every `tools/call` reserving and settling through the same
//! admission step the chat pipeline uses, with usage rows on the same
//! event bus.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use llm_proxy_engine::{Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket, EngineError};
use llm_proxy_mcp::McpGateway;
use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};
use llm_proxy_transport::ProxyClient;
use secrecy::SecretString;
use tokio::sync::Mutex as AsyncMutex;

/// An admission step that records every reserve/settle pair and can be
/// told to refuse.
#[derive(Debug, Default)]
struct RecordingAdmission {
    reserves: AsyncMutex<Vec<String>>,
    settles: AsyncMutex<Vec<bool>>,
    refuse: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl Admission for RecordingAdmission {
    async fn reserve(&self, claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        if self.refuse.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(EngineError::RateLimited {
                message: "gateway token is over its limit".to_owned(),
                retry_after: Some(Duration::from_secs(1)),
            });
        }
        self.reserves
            .lock()
            .await
            .push(format!("{}/{}", claim.caller, claim.model));
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, outcome: &AdmissionOutcome<'_>) {
        if let Ok(mut settles) = self.settles.try_lock() {
            settles.push(outcome.succeeded);
        }
    }
}

/// A fake upstream MCP server: one tool, JSON-RPC over HTTP, echoing the
/// headers it saw (so the revision's required headers are observable).
async fn spawn_fake_upstream() -> (String, Arc<AsyncMutex<Vec<(String, String)>>>) {
    let seen: Arc<AsyncMutex<Vec<(String, String)>>> = Arc::new(AsyncMutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let app = Router::new().route(
        "/mcp",
        axum::routing::post(move |headers: axum::http::HeaderMap, body: String| {
            let sink = Arc::clone(&sink);
            async move {
                for name in ["mcp-protocol-version", "mcp-method", "mcp-name"] {
                    if let Some(value) = headers.get(name) {
                        sink.lock().await.push((
                            name.to_owned(),
                            value.to_str().unwrap_or_default().to_owned(),
                        ));
                    }
                }
                let message: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let method = message["method"].as_str().unwrap_or_default();
                let reply = match method {
                    "tools/list" => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "result": { "tools": [
                            { "name": "echo", "description": "echoes its argument",
                              "inputSchema": { "type": "object" } }
                        ] }
                    }),
                    "tools/call" => {
                        let name = message["params"]["name"].as_str().unwrap_or_default();
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "result": { "content": [
                                { "type": "text", "text": format!("upstream tool {name} answered") }
                            ], "isError": false }
                        })
                    }
                    _ => serde_json::json!({
                        "jsonrpc": "2.0", "id": message["id"],
                        "error": { "code": -32601, "message": "unknown method" }
                    }),
                };
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    serde_json::to_string(&reply).unwrap_or_default(),
                )
                    .into_response()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    (format!("http://{addr}/mcp"), seen)
}

fn gateway_config(upstream_url: String, bind_port: u16) -> llm_proxy_core::McpGatewayConfig {
    llm_proxy_core::McpGatewayConfig {
        bind: format!("127.0.0.1:{bind_port}").parse().expect("bind"),
        tokens: vec![SecretString::from("gateway-token")],
        upstreams: vec![llm_proxy_core::McpUpstreamConfig {
            name: "docs".to_owned(),
            url: upstream_url,
            token: Some(SecretString::from("upstream-token")),
        }],
        session_ttl: Duration::from_secs(600),
    }
}

struct Fixture {
    app: Router,
    admission: Arc<RecordingAdmission>,
    bus: Arc<RecordingBus>,
    upstream_headers: Arc<AsyncMutex<Vec<(String, String)>>>,
}

async fn fixture(bind_port: u16) -> Fixture {
    let (upstream_url, seen) = spawn_fake_upstream().await;
    let admission = Arc::new(RecordingAdmission::default());
    let bus = Arc::new(RecordingBus::new());
    let gateway = McpGateway::new(
        gateway_config(upstream_url, bind_port),
        Arc::clone(&admission) as Arc<dyn Admission>,
        Arc::clone(&bus) as Arc<dyn EventBus>,
        Arc::new(ProxyClient::try_new().expect("client")),
    );
    Fixture {
        app: gateway.router(),
        admission,
        bus,
        upstream_headers: seen,
    }
}

async fn post(app: &Router, token: Option<&str>, body: &str) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header(header::CONTENT_TYPE, "application/json")
        .header("mcp-protocol-version", "2026-07-28");
    if let Some(token) = token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    app.clone()
        .oneshot(builder.body(Body::from(body.to_owned())).expect("request"))
        .await
        .expect("response")
}

async fn json_of(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json")
}

use tower::ServiceExt;

#[tokio::test]
async fn its_own_tokens_guard_the_endpoint() {
    let fixture = fixture(3971).await;
    for bad in [None, Some("wrong-token"), Some("client-key-not-mcp")] {
        let response = post(
            &fixture.app,
            bad,
            r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "token {bad:?}");
    }
    let ok = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#,
    )
    .await;
    assert_eq!(ok.status(), StatusCode::OK);
}

#[tokio::test]
async fn tools_list_namespaces_upstream_tools() {
    let fixture = fixture(3972).await;
    let response = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_of(response).await;
    let tools = body["result"]["tools"].as_array().expect("tools");
    assert!(
        tools
            .iter()
            .any(|tool| tool["name"].as_str() == Some("docs__echo")),
        "upstream tools are namespaced, got {tools:?}"
    );
    // No tool call: no admission, no usage row.
    assert!(fixture.admission.reserves.lock().await.is_empty());
    assert!(fixture.bus.snapshot().is_empty());
}

#[tokio::test]
async fn a_tool_call_goes_upstream_through_shared_admission_and_accounting() {
    let fixture = fixture(3973).await;
    let response = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":7,"method":"tools/call",
            "params":{"name":"docs__echo","arguments":{"text":"hi"}}}"#,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_of(response).await;
    let text = body["result"]["content"][0]["text"]
        .as_str()
        .expect("text content");
    assert!(text.contains("upstream tool echo answered"), "got {text}");

    // The upstream saw the revision's required headers.
    let headers = fixture.upstream_headers.lock().await;
    assert!(
        headers
            .iter()
            .any(|(name, value)| name == "mcp-method" && value == "tools/call"),
        "the upstream call carries Mcp-Method, saw {headers:?}"
    );
    assert!(
        headers
            .iter()
            .any(|(name, value)| name == "mcp-name" && value == "echo"),
        "the upstream call carries Mcp-Name, saw {headers:?}"
    );
    drop(headers);

    // Shared admission: one reserve for the caller's session token id,
    // charged against the mcp-namespaced route and the upstream tool.
    let reserves = fixture.admission.reserves.lock().await;
    assert_eq!(reserves.len(), 1, "exactly one reservation");
    assert!(
        reserves[0].contains("mcp-token-0"),
        "the gateway's own session model identifies the caller, got {}",
        reserves[0]
    );
    assert!(reserves[0].ends_with("/echo"), "got {}", reserves[0]);
    drop(reserves);
    let settles = fixture.admission.settles.lock().await;
    assert_eq!(*settles, vec![true], "the successful call settled");

    // Shared accounting: request + completion rows with route_kind mcp.
    let events = fixture.bus.snapshot();
    let kinds: Vec<&str> = events
        .iter()
        .map(|event| match event {
            ProxyEvent::RequestReceived(row) => {
                assert_eq!(row.route_kind, "mcp");
                assert_eq!(row.provider, "mcp:docs");
                "request"
            }
            ProxyEvent::ResponseCompleted(row) => {
                assert_eq!(row.model.requested, "echo");
                "completed"
            }
            other => panic!("unexpected event {other:?}"),
        })
        .collect();
    assert_eq!(kinds, vec!["request", "completed"]);
}

#[tokio::test]
async fn an_admission_refusal_refuses_the_tool_call_and_settles_a_refund() {
    let fixture = fixture(3974).await;
    fixture
        .admission
        .refuse
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let response = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":8,"method":"tools/call",
            "params":{"name":"docs__echo","arguments":{}}}"#,
    )
    .await;
    let body = json_of(response).await;
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("limit"),
        "the admission refusal reaches the client, got {message}"
    );
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32603),
        "a refusal to run a known method is an internal error, not an unknown method (-32601)"
    );
    // No upstream call was made: the headers log stays empty of tool calls.
    let headers = fixture.upstream_headers.lock().await;
    assert!(
        !headers.iter().any(|(name, _)| name == "mcp-name"),
        "a refused call never reaches the upstream"
    );
    // The refusal still wrote its usage row.
    assert!(
        fixture.bus.snapshot().iter().any(
            |event| matches!(event, ProxyEvent::RequestReceived(row) if row.route_kind == "mcp")
        ),
        "the refused call is accounted"
    );
}

#[tokio::test]
async fn protocol_framing_rules_hold() {
    let fixture = fixture(3975).await;
    // GET is 405.
    let get = fixture
        .app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/mcp")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(get.status(), StatusCode::METHOD_NOT_ALLOWED);

    // Unknown method: 404 with -32601.
    let unknown = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":2,"method":"resources/list"}"#,
    )
    .await;
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    let body = json_of(unknown).await;
    assert_eq!(body["error"]["code"].as_i64(), Some(-32601));

    // A notification: 202, no body.
    let notification = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
    )
    .await;
    assert_eq!(notification.status(), StatusCode::ACCEPTED);

    // initialize is answered for older-era clients, echoing the revision
    // they negotiate (a client that names 2025-11-25 and receives
    // 2026-07-28 rejects the server — found against the official
    // Inspector).
    let initialize = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":3,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"inspector-cli","version":"2.14.0"}}}"#,
    )
    .await;
    assert_eq!(initialize.status(), StatusCode::OK);
    let body = json_of(initialize).await;
    assert_eq!(
        body["result"]["protocolVersion"].as_str(),
        Some("2025-11-25"),
        "the negotiated revision is echoed, got {body}"
    );

    // Without a named revision the endpoint answers its own.
    let initialize = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":4,"method":"initialize"}"#,
    )
    .await;
    let body = json_of(initialize).await;
    assert_eq!(
        body["result"]["protocolVersion"].as_str(),
        Some("2026-07-28")
    );
}

#[tokio::test]
async fn an_unknown_tool_is_rejected_without_an_upstream_call() {
    let fixture = fixture(3976).await;
    let response = post(
        &fixture.app,
        Some("gateway-token"),
        r#"{"jsonrpc":"2.0","id":9,"method":"tools/call",
            "params":{"name":"nope__missing","arguments":{}}}"#,
    )
    .await;
    let body = json_of(response).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("unknown tool")),
        "got {body}"
    );
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32602),
        "an unknown tool is the client's invalid params, not a server error: {body}"
    );
    assert!(fixture.admission.reserves.lock().await.is_empty());
}

/// A fake upstream whose `tools/call` parks until released, so a test can
/// drop a call mid-flight — exactly where a client disconnect lands.
async fn spawn_hanging_upstream(release: Arc<tokio::sync::watch::Receiver<bool>>) -> String {
    let app = Router::new().route(
        "/mcp",
        axum::routing::post(move |body: String| {
            let mut release = release.as_ref().clone();
            async move {
                let message: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                if message["method"].as_str() != Some("tools/call") {
                    return (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        serde_json::to_string(&serde_json::json!({
                            "jsonrpc": "2.0", "id": message["id"],
                            "error": { "code": -32601, "message": "unknown method" }
                        }))
                        .unwrap_or_default(),
                    )
                        .into_response();
                }
                while !*release.borrow() {
                    if release.changed().await.is_err() {
                        break;
                    }
                }
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    serde_json::to_string(&serde_json::json!({
                        "jsonrpc": "2.0", "id": message["id"],
                        "result": { "content": [], "isError": false }
                    }))
                    .unwrap_or_default(),
                )
                    .into_response()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind hanging upstream");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "hanging upstream stopped");
        }
    });
    format!("http://{addr}/mcp")
}

#[tokio::test]
async fn a_tool_call_dropped_mid_upstream_still_settles_its_reservation() {
    // A client disconnect mid tools/call drops the handler future between
    // reserve and settle; the reservation must be released as failed, or
    // its in-flight slot is held forever (the same rule the chat pipeline
    // enforces with UnsettledLeaseGuard).
    let release = Arc::new(tokio::sync::watch::Sender::new(false));
    let upstream_url = spawn_hanging_upstream(Arc::new(release.subscribe())).await;
    let admission = Arc::new(RecordingAdmission::default());
    let gateway = McpGateway::new(
        gateway_config(upstream_url, 3979),
        Arc::clone(&admission) as Arc<dyn Admission>,
        Arc::new(RecordingBus::new()) as Arc<dyn EventBus>,
        Arc::new(ProxyClient::try_new().expect("client")),
    );
    let app = gateway.router();

    let call_app = app.clone();
    let call = tokio::spawn(async move {
        post(
            &call_app,
            Some("gateway-token"),
            r#"{"jsonrpc":"2.0","id":11,"method":"tools/call",
            "params":{"name":"docs__echo","arguments":{"text":"hi"}}}"#,
        )
        .await
    });
    let reserved = async {
        loop {
            if !admission.reserves.lock().await.is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), reserved)
        .await
        .expect("the call reserved before parking upstream");

    call.abort();
    let settled = async {
        loop {
            if !admission.settles.lock().await.is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), settled)
        .await
        .expect("the dropped call must settle its reservation (drop backstop)");
    assert_eq!(
        admission.settles.lock().await.as_slice(),
        &[false],
        "the backstop settles the dropped call as failed"
    );
}
