//! Plan 09 step 6 against a **real** MCP server implementation: the
//! official reference "everything" server
//! (`@modelcontextprotocol/server-everything`, Streamable HTTP
//! transport), run locally. The gateway aggregates its tools, and a
//! `tools/call` reaches the real server's `echo` tool and returns its
//! answer — through the gateway's own auth, admission and accounting.
//!
//! This test needs the server running and is `#[ignore]`d. Run it with:
//!
//! ```text
//! npx -y @modelcontextprotocol/server-everything streamableHttp &
//! LLM_PROXY_TEST_MCP_UPSTREAM=http://127.0.0.1:3001/mcp \
//!     cargo test -p llm-proxy-mcp --test live_upstream -- --ignored
//! ```
//!
//! The default test run uses in-repo fake MCP servers; this file is the
//! same properties checked against a real third-party implementation,
//! so protocol divergences surface here.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use llm_proxy_engine::{Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket};
use llm_proxy_mcp::McpGateway;
use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};
use llm_proxy_transport::ProxyClient;
use secrecy::SecretString;

/// An admission step that records reserves, so the live call's claim is
/// observable.
#[derive(Debug, Default)]
struct RecordingAdmission {
    reserves: tokio::sync::Mutex<Vec<String>>,
}

#[async_trait]
impl Admission for RecordingAdmission {
    async fn reserve(
        &self,
        claim: &AdmissionClaim<'_>,
    ) -> Result<AdmissionTicket, llm_proxy_engine::EngineError> {
        self.reserves
            .lock()
            .await
            .push(format!("{}/{}", claim.caller, claim.model));
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, _outcome: &AdmissionOutcome<'_>) {}
}

fn upstream_url() -> Option<String> {
    std::env::var("LLM_PROXY_TEST_MCP_UPSTREAM").ok()
}

async fn gateway_over(upstream: &str) -> (axum::Router, Arc<RecordingBus>) {
    let config = llm_proxy_core::McpGatewayConfig {
        bind: "127.0.0.1:0".parse().expect("bind"),
        tokens: vec![SecretString::from("live-gateway-token")],
        upstreams: vec![llm_proxy_core::McpUpstreamConfig {
            name: "everything".to_owned(),
            url: upstream.to_owned(),
            token: None,
        }],
        session_ttl: Duration::from_secs(600),
    };
    let admission = Arc::new(RecordingAdmission::default());
    let bus = Arc::new(RecordingBus::new());
    let gateway = McpGateway::new(
        config,
        admission,
        Arc::clone(&bus) as Arc<dyn EventBus>,
        Arc::new(ProxyClient::try_new().expect("client")),
    );
    (gateway.router(), bus)
}

async fn post(app: &axum::Router, method: &str, body: &str) -> axum::response::Response {
    use tower::ServiceExt;
    app.clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("authorization", "Bearer live-gateway-token")
                .header("content-type", "application/json")
                .header("mcp-protocol-version", "2026-07-28")
                .header("mcp-method", method)
                .body(axum::body::Body::from(body.to_owned()))
                .expect("request"),
        )
        .await
        .expect("response")
}

async fn json_of(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json body")
}

#[tokio::test]
#[ignore = "needs a real MCP server; set LLM_PROXY_TEST_MCP_UPSTREAM"]
async fn tools_list_aggregates_the_real_reference_server() {
    let url = upstream_url().expect("LLM_PROXY_TEST_MCP_UPSTREAM must be set");
    let (app, _bus) = gateway_over(&url).await;
    let response = post(
        &app,
        "tools/list",
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#,
    )
    .await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = json_of(response).await;
    let tools = body["result"]["tools"].as_array().expect("tools array");
    assert!(
        !tools.is_empty(),
        "the real server exposes tools; got {tools:?}"
    );
    assert!(
        tools.iter().all(|tool| tool["name"]
            .as_str()
            .is_some_and(|name| name.starts_with("everything__"))),
        "every tool is namespaced under the configured upstream, got {tools:?}"
    );
    // The reference server documents an echo tool.
    assert!(
        tools
            .iter()
            .any(|tool| tool["name"].as_str() == Some("everything__echo")),
        "the real echo tool is present, got {:?}",
        tools.iter().map(|t| t["name"].as_str()).collect::<Vec<_>>()
    );
}

#[tokio::test]
#[ignore = "needs a real MCP server; set LLM_PROXY_TEST_MCP_UPSTREAM"]
async fn a_tool_call_runs_the_real_servers_echo_tool_end_to_end() {
    let url = upstream_url().expect("LLM_PROXY_TEST_MCP_UPSTREAM must be set");
    let (app, bus) = gateway_over(&url).await;
    let marker = format!("live-{}", std::process::id());
    let request = format!(
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"everything__echo","arguments":{{"message":"{marker}"}}}}}}"#
    );
    let response = post(&app, "tools/call", &request).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let body = json_of(response).await;
    assert!(
        body.get("error").is_none(),
        "the real server answered, got {body}"
    );
    let text = body["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(
        text.contains(&marker),
        "the echo tool returned our own message through the gateway, got {text:?}"
    );

    // Shared accounting: request + completion rows with route kind mcp.
    let events = bus.snapshot();
    let kinds: Vec<&str> = events
        .iter()
        .map(|event| match event {
            ProxyEvent::RequestReceived(row) => {
                assert_eq!(row.route_kind, "mcp");
                "request"
            }
            ProxyEvent::ResponseCompleted(_) => "completed",
            other => panic!("unexpected event {other:?}"),
        })
        .collect();
    assert_eq!(kinds, vec!["request", "completed"]);
}
