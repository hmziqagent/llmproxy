//! Plan 09 step 6 against a **real MCP client implementation**: the
//! official MCP Inspector CLI (`@modelcontextprotocol/inspector`), run
//! against the gateway's own `POST /mcp` endpoint on a real TCP socket.
//! The client lists the gateway's tools and calls one, exercising the
//! gateway's framing, auth and header validation from the outside.
//!
//! This test shells out to `npx` (it downloads the inspector on first
//! run) and needs a real upstream for the tool call, so it is
//! `#[ignore]`d. Run it with:
//!
//! ```text
//! npx -y @modelcontextprotocol/server-everything streamableHttp &
//! LLM_PROXY_TEST_MCP_UPSTREAM=http://127.0.0.1:3001/mcp \
//! LLM_PROXY_TEST_MCP_CLIENT=1 \
//!     cargo test -p llm-proxy-mcp --test live_client -- --ignored --test-threads 1
//! ```

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use llm_proxy_mcp::McpGateway;
use llm_proxy_storage::{EventBus, NoopBus};
use llm_proxy_transport::ProxyClient;
use secrecy::SecretString;

fn want_live() -> bool {
    std::env::var("LLM_PROXY_TEST_MCP_CLIENT").is_ok()
        && std::env::var("LLM_PROXY_TEST_MCP_UPSTREAM").is_ok()
}

async fn spawn_gateway() -> String {
    let config = llm_proxy_core::McpGatewayConfig {
        bind: "127.0.0.1:0".parse().expect("bind"),
        tokens: vec![SecretString::from("live-gateway-token")],
        upstreams: vec![llm_proxy_core::McpUpstreamConfig {
            name: "everything".to_owned(),
            url: std::env::var("LLM_PROXY_TEST_MCP_UPSTREAM").expect("upstream url"),
            token: None,
        }],
        session_ttl: Duration::from_secs(600),
    };
    let gateway = McpGateway::new(
        config,
        Arc::new(llm_proxy_engine::AllowAllAdmission),
        Arc::new(NoopBus) as Arc<dyn EventBus>,
        Arc::new(ProxyClient::try_new().expect("client")),
    );
    let app = gateway.router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind gateway");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "live gateway stopped");
        }
    });
    format!("http://{addr}/mcp")
}

/// Runs the inspector off the async runtime: the gateway's serve task
/// must keep being polled while the (slow, blocking) subprocess runs —
/// on a current-thread test runtime a blocking call on the test thread
/// would starve it and the inspector would time out.
async fn inspector(endpoint: &str, call: Option<(&str, &str)>) -> String {
    let endpoint = endpoint.to_owned();
    let call = call.map(|(name, arg)| (name.to_owned(), arg.to_owned()));
    tokio::task::spawn_blocking(move || {
        inspector_blocking(
            &endpoint,
            call.as_ref().map(|(n, a)| (n.as_str(), a.as_str())),
        )
    })
    .await
    .expect("spawn_blocking joins")
}

fn inspector_blocking(endpoint: &str, tool: Option<(&str, &str)>) -> String {
    let mut command = std::process::Command::new("npx");
    command.args([
        "-y",
        "@modelcontextprotocol/inspector",
        "--cli",
        "--transport",
        "http",
        "--server-url",
        endpoint,
        "--header",
        "Authorization: Bearer live-gateway-token",
        "--connect-timeout",
        "120000",
        "--method",
        if tool.is_some() {
            "tools/call"
        } else {
            "tools/list"
        },
    ]);
    if let Some((tool_name, tool_arg)) = tool {
        command.args(["--tool-name", tool_name, "--tool-arg", tool_arg]);
    }
    let output = command.output().expect("npx runs");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "the inspector failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
    stdout
}

#[tokio::test]
#[ignore = "shells out to npx; set LLM_PROXY_TEST_MCP_CLIENT and LLM_PROXY_TEST_MCP_UPSTREAM"]
async fn the_official_mcp_inspector_lists_and_calls_the_gateways_tools() {
    if !want_live() {
        // --ignored runs this even without the env; nothing to do then.
        return;
    }
    let endpoint = spawn_gateway().await;

    // tools/list through the real client: the reference server's echo
    // tool, namespaced, is visible to it.
    let listed = inspector(&endpoint, None).await;
    assert!(
        listed.contains("everything__echo"),
        "the inspector saw the namespaced tool, got: {listed}"
    );

    // tools/call through the real client, answered by the real upstream.
    let marker = format!("live-client-{}", std::process::id());
    let called = inspector(
        &endpoint,
        Some(("everything__echo", format!("message={marker}").as_str())),
    )
    .await;
    assert!(
        called.contains(&marker),
        "the reference server's echo tool answered with our own message, through the gateway and the real client, got: {called}"
    );
}
