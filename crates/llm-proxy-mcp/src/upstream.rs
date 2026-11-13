//! The client-facing endpoint and the upstream MCP client.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use llm_proxy_adapters::request::{AuthHeaders, ProxyRequest};
use llm_proxy_core::{AuthStyle, McpUpstreamConfig};
use secrecy::SecretString;
use serde_json::{Value, json};

use crate::McpGateway;
use crate::accounting::{self, ToolCall};

/// The protocol revision the gateway speaks and requires of its clients.
pub(crate) const MCP_PROTOCOL_VERSION: &str = "2026-07-28";

/// JSON-RPC error codes.
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_REQUEST: i64 = -32600;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

/// A tools dispatch failure: whose fault it is picks the JSON-RPC error
/// code, so a client mistake is never answered as a server error.
enum ToolFailure {
    /// The client's request was invalid (a missing or unknown tool name):
    /// `-32602` Invalid params.
    InvalidParams(String),
    /// The server failed to run a known method (an admission refusal, an
    /// upstream outage): `-32603` Internal error.
    Server(String),
}

impl ToolFailure {
    fn code(&self) -> i64 {
        match self {
            Self::InvalidParams(_) => INVALID_PARAMS,
            Self::Server(_) => INTERNAL_ERROR,
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::InvalidParams(message) | Self::Server(message) => message,
        }
    }
}

/// One configured upstream MCP server.
#[derive(Debug)]
pub struct UpstreamMcp {
    /// The name tools are namespaced under.
    pub name: String,
    /// The upstream's MCP endpoint URL.
    pub url: String,
    /// The upstream's bearer token, when configured.
    pub token: Option<SecretString>,
}

impl UpstreamMcp {
    /// From `[mcp.upstreams]` config.
    #[must_use]
    pub fn from_config(config: &McpUpstreamConfig) -> Self {
        Self {
            name: config.name.clone(),
            url: config.url.clone(),
            token: config.token.clone(),
        }
    }
}

/// `GET /mcp`: this revision has no GET stream; the spec's answer is 405.
pub(crate) async fn method_not_allowed() -> Response {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        "the MCP endpoint accepts POST only (revision 2026-07-28)",
    )
        .into_response()
}

/// The gateway's own bearer auth: `[mcp]` tokens, never the proxy's
/// client keys.
fn authenticate(gateway: &McpGateway, headers: &HeaderMap) -> Result<String, Box<Response>> {
    match gateway.token_id(bearer_token(headers)) {
        Some(token_id) => Ok(token_id),
        None => Err(Box::new(
            (
                StatusCode::UNAUTHORIZED,
                "missing or invalid gateway token ([mcp] tokens only)",
            )
                .into_response(),
        )),
    }
}

/// The same Authorization-header parse as
/// `crates/llm-proxy-admin/src/auth.rs::bearer_token` (case-insensitive
/// scheme, trimmed non-empty token); duplicated rather than depending on
/// the admin crate.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?;
    let header_value = value.to_str().ok()?;
    let mut parts = header_value.splitn(2, ' ');
    match (parts.next(), parts.next()) {
        (Some(scheme), Some(token)) if scheme.eq_ignore_ascii_case("bearer") => {
            let token = token.trim();
            (!token.is_empty()).then_some(token)
        }
        _ => None,
    }
}

/// `POST /mcp`: one JSON-RPC message per POST.
pub(crate) async fn gateway_endpoint(
    State(gateway): State<Arc<McpGateway>>,
    headers: HeaderMap,
    body: Option<Json<Value>>,
) -> Response {
    let token_id = match authenticate(&gateway, &headers) {
        Ok(token_id) => token_id,
        Err(response) => return *response,
    };
    let Some(Json(message)) = body else {
        return error_no_id(
            StatusCode::BAD_REQUEST,
            INVALID_REQUEST,
            "the MCP endpoint needs one JSON-RPC message per POST",
        );
    };
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        return error_no_id(
            StatusCode::BAD_REQUEST,
            INVALID_REQUEST,
            "not a JSON-RPC message: no method",
        );
    };

    // Notifications: 202, no body (this revision defines no client
    // notifications; unknown ones are still acknowledged).
    let Some(id) = message.get("id").cloned() else {
        if method.starts_with("notifications/") {
            return StatusCode::ACCEPTED.into_response();
        }
        return error_no_id(
            StatusCode::BAD_REQUEST,
            INVALID_REQUEST,
            "only JSON-RPC requests and notifications are accepted",
        );
    };

    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        // Version negotiation for initialization-era clients (found by
        // running the official MCP Inspector against this endpoint): a
        // client that initializes names the revision it speaks, and a
        // server answering a different one is rejected as unsupported.
        // Echo the requested revision when the client named one; the
        // per-request framing this gateway speaks stays 2026-07-28.
        "initialize" => {
            let negotiated = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(MCP_PROTOCOL_VERSION);
            Ok(json!({
                "protocolVersion": negotiated,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "llm-proxy-mcp", "version": env!("CARGO_PKG_VERSION") },
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => list_tools(&gateway).await,
        "tools/call" => call_tool(&gateway, &token_id, &params).await,
        other => {
            return error_for_id(
                StatusCode::NOT_FOUND,
                &id,
                METHOD_NOT_FOUND,
                &format!("method not found: {other}"),
            );
        }
    };
    match result {
        Ok(result) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })),
        )
            .into_response(),
        // A tools/list or tools/call failure is not an unknown method
        // (-32601, kept for `other` above): client mistakes answer
        // -32602, server-side failures -32603.
        Err(failure) => error_for_id(StatusCode::OK, &id, failure.code(), failure.message()),
    }
}

/// Aggregates `tools/list` across every configured upstream, namespacing
/// tool names as `<server>__<tool>` so two upstreams exposing the same
/// name never collide.
async fn list_tools(gateway: &McpGateway) -> Result<Value, ToolFailure> {
    let mut tools: Vec<Value> = Vec::new();
    for upstream in gateway.upstreams() {
        let listed = upstream_request(gateway, upstream, "tools/list", &json!({}), None)
            .await
            .map_err(ToolFailure::Server)?;
        let upstream_tools = listed
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for mut tool in upstream_tools {
            if let Some(name) = tool.get("name").and_then(Value::as_str) {
                tool["name"] = json!(format!("{}__{}", upstream.name, name));
                tools.push(tool);
            }
        }
    }
    Ok(json!({ "tools": tools }))
}

/// Dispatches one namespaced tool call to its upstream, wrapped in the
/// shared admission and accounting.
async fn call_tool(
    gateway: &McpGateway,
    token_id: &str,
    params: &Value,
) -> Result<Value, ToolFailure> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolFailure::InvalidParams("tools/call needs a tool name".to_owned()))?;
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    let Some((upstream, tool)) = gateway.resolve_tool(name) else {
        return Err(ToolFailure::InvalidParams(format!("unknown tool: {name}")));
    };
    let call = ToolCall {
        caller: token_id,
        server: &upstream.name,
        tool: &tool,
        params: &arguments,
    };

    // The gateway's own session model: this token now has a live session.
    let _session = gateway.session_for(token_id);

    // Shared admission: the same reserve-then-settle step the chat
    // pipeline runs (plan 09 step 6). A refusal is still accounted — the
    // chat pipeline writes failure rows for refused requests, and so does
    // the gateway.
    let request_id = uuid::Uuid::new_v4().to_string();
    let start = std::time::Instant::now();
    accounting::emit_request(gateway.bus.as_ref(), &request_id, &call);
    let ticket = match accounting::reserve(gateway.admission.as_ref(), &call).await {
        Ok(ticket) => ticket,
        Err(error) => {
            let message = error.to_string();
            accounting::emit_failed(
                gateway.bus.as_ref(),
                &request_id,
                &call,
                &message,
                error.status_code().max(400),
                start.elapsed().as_millis().min(u64::MAX as u128) as u64,
            );
            return Err(ToolFailure::Server(message));
        }
    };
    // The reservation lives inside this handler future: a drop while the
    // upstream call runs (client disconnect mid tools/call) must release
    // the lease. Both arms below settle explicitly, so the guard disarms
    // right before them — the same backstop the chat pipeline's family
    // routes use.
    let mut settle_guard = ticket.as_ref().map(|ticket| {
        llm_proxy_engine::UnsettledLeaseGuard::new(Arc::clone(&gateway.admission), ticket)
    });

    let upstream_params = json!({ "name": tool, "arguments": arguments });
    let outcome = upstream_request(
        gateway,
        upstream,
        "tools/call",
        &upstream_params,
        Some(&tool),
    )
    .await;
    if let Some(guard) = settle_guard.as_mut() {
        guard.disarm();
    }
    match outcome {
        Ok(result) => {
            accounting::settle(gateway.admission.as_ref(), &ticket, true);
            accounting::emit_completed(
                gateway.bus.as_ref(),
                &request_id,
                &call,
                start.elapsed().as_millis().min(u64::MAX as u128) as u64,
            );
            Ok(result)
        }
        Err(message) => {
            accounting::settle(gateway.admission.as_ref(), &ticket, false);
            accounting::emit_failed(
                gateway.bus.as_ref(),
                &request_id,
                &call,
                &message,
                StatusCode::BAD_GATEWAY.as_u16(),
                start.elapsed().as_millis().min(u64::MAX as u128) as u64,
            );
            Err(ToolFailure::Server(message))
        }
    }
}

/// Sends one JSON-RPC request to an upstream and returns its `result`.
async fn upstream_request(
    gateway: &McpGateway,
    upstream: &UpstreamMcp,
    method: &str,
    params: &Value,
    name_header: Option<&str>,
) -> Result<Value, String> {
    // The revision requires these headers on every POST (upstream
    // servers validate them against the body).
    let mut extra_headers = std::collections::HashMap::new();
    extra_headers.insert(
        "mcp-protocol-version".to_owned(),
        MCP_PROTOCOL_VERSION.to_owned(),
    );
    extra_headers.insert("mcp-method".to_owned(), method.to_owned());
    if let Some(name) = name_header {
        extra_headers.insert("mcp-name".to_owned(), name.to_owned());
    }
    extra_headers.insert("content-type".to_owned(), "application/json".to_owned());
    extra_headers.insert("accept".to_owned(), "application/json".to_owned());

    // The revision mirrors the header metadata into the request body:
    // every request carries its protocol version, client info and client
    // capabilities in `_meta.io.modelcontextprotocol/*`, and servers
    // validate the envelope against the headers. Found by running the
    // gateway against the official reference server, which rejects a
    // header-only request with -32602 "missing _meta".
    let mut params = params.clone();
    if let Some(object) = params.as_object_mut() {
        object.insert(
            "_meta".to_owned(),
            json!({
                "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientInfo": {
                    "name": "llm-proxy-mcp",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "io.modelcontextprotocol/clientCapabilities": {},
            }),
        );
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "id": uuid::Uuid::new_v4().to_string(),
        "method": method,
        "params": params,
    });
    let body = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    let request = ProxyRequest {
        url: upstream.url.clone(),
        auth: AuthHeaders {
            style: AuthStyle::Bearer,
            api_key: upstream
                .token
                .clone()
                .unwrap_or_else(|| SecretString::from("")),
        },
        body,
        stream: false,
        extra_headers,
        forwarded_headers: std::collections::HashMap::new(),
    };
    let bytes = gateway
        .transport
        .send(request)
        .await
        .map_err(|error| format!("upstream {method} failed: {error}"))?;
    let message: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("upstream reply is not JSON: {error}"))?;
    if let Some(error) = message.get("error") {
        return Err(format!(
            "upstream refused {method}: {}",
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        ));
    }
    Ok(message.get("result").cloned().unwrap_or(Value::Null))
}

fn error_no_id(status: StatusCode, code: i64, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        Json(json!({
            "jsonrpc": "2.0",
            "error": { "code": code, "message": message },
        })),
    )
        .into_response()
}

fn error_for_id(status: StatusCode, id: &Value, code: i64, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message },
        })),
    )
        .into_response()
}
