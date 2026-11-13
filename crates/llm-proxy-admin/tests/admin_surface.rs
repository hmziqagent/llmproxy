//! The admin surface end to end (plan 09 steps 2 and 3): the REST routes
//! behind the admin token, key lifecycle through HTTP, the sanitized
//! config view, and the MCP tools endpoint speaking protocol revision
//! 2026-07-28 — same auth, same handlers.

#![cfg(test)]

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use llm_proxy_admin::{AdminSurface, KeyPolicyInput, ManagedKeys};
use tower::ServiceExt;

const ADMIN_TOKEN: &str = "admin-secret";

fn surface() -> Router {
    let keys = Arc::new(ManagedKeys::in_memory().expect("key store"));
    let usage = llm_proxy_storage::UsageReader::new(":memory:");
    AdminSurface::new(ADMIN_TOKEN)
        .keys(keys)
        .usage(usage)
        .router()
}

fn get(path: &str) -> Request<Body> {
    request(axum::http::Method::GET, path, None)
}

fn post(path: &str, body: Option<String>) -> Request<Body> {
    request(axum::http::Method::POST, path, body)
}

fn request(method: axum::http::Method, path: &str, body: Option<String>) -> Request<Body> {
    let builder = Request::builder()
        .method(method)
        .uri(path)
        .header("authorization", format!("Bearer {ADMIN_TOKEN}"));
    if let Some(body) = body {
        builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .expect("request builds")
    } else {
        builder.body(Body::empty()).expect("request builds")
    }
}

async fn json_body(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json body")
}

#[tokio::test]
async fn every_route_requires_the_admin_token() {
    let app = surface();
    for path in ["/keys", "/config", "/usage/summary", "/"] {
        let unauthenticated = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(axum::http::Method::GET)
                    .uri(path)
                    .body(Body::empty())
                    .expect("request builds"),
            )
            .await
            .expect("response");
        assert_eq!(
            unauthenticated.status(),
            StatusCode::UNAUTHORIZED,
            "{path} must sit behind the admin token"
        );
    }
    // A wrong token is refused too.
    let wrong = app
        .oneshot(
            Request::builder()
                .method(axum::http::Method::GET)
                .uri("/keys")
                .header("authorization", "Bearer wrong")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("response");
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn responses_are_never_cacheable() {
    let app = surface();
    let response = app.oneshot(get("/keys")).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("no-store"),
        "authenticated admin data must never be cached (backlog 37)"
    );
}

#[tokio::test]
async fn a_key_lives_a_full_lifecycle_through_http() {
    let app = surface();
    // Create.
    let created = app
        .clone()
        .oneshot(post(
            "/keys",
            Some(r#"{"rpm": 60, "usd_per_day": "5.00"}"#.to_owned()),
        ))
        .await
        .expect("response");
    assert_eq!(created.status(), StatusCode::CREATED);
    let body = json_body(created).await;
    let key_id = body["key_id"].as_str().expect("key id").to_owned();
    let token = body["token"].as_str().expect("one-time token").to_owned();
    assert!(token.starts_with(&key_id), "tokens are key-id-prefixed");

    // List shows the record, never the verifier or secret.
    let listed = json_body(app.clone().oneshot(get("/keys")).await.expect("response")).await;
    let records = listed["keys"].as_array().expect("array");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["id"].as_str(), Some(key_id.as_str()));
    let listed_text = listed.to_string();
    assert!(!listed_text.contains(&token), "secrets never come back");

    // Revoke, then the record shows it.
    let revoked = app
        .clone()
        .oneshot(post(&format!("/keys/{key_id}/revoke"), None))
        .await
        .expect("response");
    assert_eq!(revoked.status(), StatusCode::OK);
    let listed = json_body(app.clone().oneshot(get("/keys")).await.expect("response")).await;
    assert!(
        listed["keys"][0]["revoked_at"].is_string(),
        "the record carries the revocation"
    );
}

/// An issued token must authenticate through the daemon's own
/// `ClientAuth` built from `client_specs()` — the exact path a restart
/// takes. `create()` used to hash the whole `<key-id>.<secret>` token
/// while `ClientAuth` verifies only the part after the first `.`, so
/// every issued key was a 401.
#[tokio::test]
async fn an_issued_token_authenticates_through_client_auth() {
    use llm_proxy_core::{AuthConfig, AuthMode};
    use secrecy::ExposeSecret;

    let keys = Arc::new(ManagedKeys::in_memory().expect("key store"));
    let issued = keys
        .create(&KeyPolicyInput {
            rpm: Some(60),
            ..KeyPolicyInput::default()
        })
        .expect("issued");
    let mut auth = AuthConfig {
        mode: AuthMode::ClientKeys,
        ..AuthConfig::default()
    };
    for (id, spec) in keys.client_specs().expect("specs") {
        auth.clients.insert(id, spec);
    }
    let client_auth = llm_proxy_server::ClientAuth::from_config(&auth).expect("client auth");

    assert_eq!(
        client_auth
            .authenticate(issued.token.expose_secret())
            .await
            .as_deref(),
        Some(issued.key_id.as_str()),
        "the once-shown token authenticates as its key id"
    );
    assert!(
        client_auth
            .authenticate(&format!("{}.not-the-secret", issued.key_id))
            .await
            .is_none(),
        "a wrong secret is refused"
    );
    assert!(
        client_auth
            .authenticate("unknown-key.anything")
            .await
            .is_none(),
        "an unknown key id is refused"
    );
}

/// A `usd_per_day` typo used to be stored unvalidated and silently became
/// a 0 USD/day budget after the next restart. Creation must answer 400
/// and store nothing.
#[tokio::test]
async fn a_malformed_usd_per_day_answers_400_and_stores_nothing() {
    let app = surface();
    let refused = app
        .clone()
        .oneshot(post("/keys", Some(r#"{"usd_per_day": "five"}"#.to_owned())))
        .await
        .expect("response");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    let body = json_body(refused).await;
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("usd_per_day"),
        "the refusal names the field: {message}"
    );
    let listed = json_body(app.oneshot(get("/keys")).await.expect("response")).await;
    assert!(
        listed["keys"]
            .as_array()
            .is_some_and(std::vec::Vec::is_empty),
        "no key was stored for the refused policy"
    );
}

#[tokio::test]
async fn the_web_ui_serves_and_names_the_surface() {
    let app = surface();
    let response = app.oneshot(get("/")).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/html"))
    );
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    let html = String::from_utf8_lossy(&bytes);
    assert!(html.contains("llm-proxy admin"));
    assert!(html.contains("/keys"), "the UI calls the JSON API");
}

/// The MCP tools endpoint (protocol revision 2026-07-28): one POST per
/// message, single-JSON responses, notifications acknowledged with 202,
/// header validation with `-32020`, unknown methods 404 with `-32601`.
#[tokio::test]
async fn mcp_tools_list_and_call_behind_the_same_token() {
    let app = surface();

    let listed = app
        .clone()
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#.to_owned(),
            Some("tools/list"),
            None,
        ))
        .await
        .expect("response");
    assert_eq!(listed.status(), StatusCode::OK);
    let body = json_body(listed).await;
    let tools: Vec<String> = body["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|tool| tool["name"].as_str().unwrap_or_default().to_owned())
        .collect();
    for expected in [
        "usage_summary",
        "list_keys",
        "create_key",
        "revoke_key",
        "get_config",
    ] {
        assert!(
            tools.iter().any(|name| name == expected),
            "tools/list must expose {expected}, got {tools:?}"
        );
    }

    // A tool call through the same handlers as the REST routes.
    let called = app
        .clone()
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_keys","arguments":{}}}"#.to_owned(),
            Some("tools/call"),
            Some("list_keys"),
        ))
        .await
        .expect("response");
    assert_eq!(called.status(), StatusCode::OK);
    let body = json_body(called).await;
    let text = body["result"]["content"][0]["text"]
        .as_str()
        .expect("text content block");
    assert!(text.contains("\"keys\""), "the tool returns the key list");
}

#[tokio::test]
async fn mcp_protocol_rules_are_enforced() {
    let app = surface();

    // Missing protocol-version header.
    let missing = app
        .clone()
        .oneshot(
            Request::builder()
                .method(axum::http::Method::POST)
                .uri("/mcp")
                .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                ))
                .expect("request builds"),
        )
        .await
        .expect("response");
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);
    let body = json_body(missing).await;
    assert_eq!(body["error"]["code"].as_i64(), Some(-32020));

    // Unsupported version.
    let old_version = app
        .clone()
        .oneshot(
            Request::builder()
                .method(axum::http::Method::POST)
                .uri("/mcp")
                .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
                .header("mcp-protocol-version", "2025-03-26")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                ))
                .expect("request builds"),
        )
        .await
        .expect("response");
    assert_eq!(old_version.status(), StatusCode::BAD_REQUEST);
    let body = json_body(old_version).await;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("2026-07-28")),
        "the refusal names the supported revision"
    );

    // Method-header mismatch.
    let mismatched = app
        .clone()
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":3,"method":"ping"}"#.to_owned(),
            Some("tools/list"),
            None,
        ))
        .await
        .expect("response");
    assert_eq!(mismatched.status(), StatusCode::BAD_REQUEST);
    let body = json_body(mismatched).await;
    assert_eq!(body["error"]["code"].as_i64(), Some(-32020));

    // Unknown method: 404 with -32601.
    let unknown = app
        .clone()
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":4,"method":"resources/list"}"#.to_owned(),
            Some("resources/list"),
            None,
        ))
        .await
        .expect("response");
    assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    let body = json_body(unknown).await;
    assert_eq!(body["error"]["code"].as_i64(), Some(-32601));

    // A notification: 202, no body.
    let notification = app
        .clone()
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#.to_owned(),
            Some("notifications/initialized"),
            None,
        ))
        .await
        .expect("response");
    assert_eq!(notification.status(), StatusCode::ACCEPTED);

    // initialize is answered for older-era clients.
    let initialize = app
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":5,"method":"initialize"}"#.to_owned(),
            Some("initialize"),
            None,
        ))
        .await
        .expect("response");
    assert_eq!(initialize.status(), StatusCode::OK);
    let body = json_body(initialize).await;
    assert_eq!(
        body["result"]["protocolVersion"].as_str(),
        Some("2026-07-28")
    );
}

#[tokio::test]
async fn mcp_create_key_tool_issues_a_working_token() {
    let app = surface();
    let called = app
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"create_key","arguments":{"rpm":10}}}"#.to_owned(),
            Some("tools/call"),
            Some("create_key"),
        ))
        .await
        .expect("response");
    assert_eq!(called.status(), StatusCode::OK);
    let body = json_body(called).await;
    let text = body["result"]["content"][0]["text"].as_str().expect("text");
    let payload: serde_json::Value = serde_json::from_str(text).expect("tool payload is JSON");
    let token = payload["token"].as_str().expect("one-time token");
    assert!(token.contains('.'), "the tool returns the prefixed token");
}

/// Tool failures carry the JSON-RPC code of their blame, never "method
/// not found": a client mistake (a malformed budget) answers -32602
/// invalid params, a server-side failure (no key store attached to the
/// surface) answers -32603 internal error.
#[tokio::test]
async fn mcp_tool_failures_map_client_and_server_blame() {
    let app = surface();
    let invalid = app
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"create_key","arguments":{"usd_per_day":"five"}}}"#.to_owned(),
            Some("tools/call"),
            Some("create_key"),
        ))
        .await
        .expect("response");
    assert_eq!(invalid.status(), StatusCode::OK);
    let body = json_body(invalid).await;
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32602),
        "a client mistake is invalid params, got {body}"
    );
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("usd_per_day"),
        "the refusal names the field: {message}"
    );

    // A surface with no key store attached cannot run the tool at all:
    // the server's failure, not the client's.
    let bare = AdminSurface::new(ADMIN_TOKEN).router();
    let server_failure = bare
        .oneshot(mcp_post(
            r#"{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"create_key","arguments":{}}}"#.to_owned(),
            Some("tools/call"),
            Some("create_key"),
        ))
        .await
        .expect("response");
    assert_eq!(server_failure.status(), StatusCode::OK);
    let body = json_body(server_failure).await;
    assert_eq!(
        body["error"]["code"].as_i64(),
        Some(-32603),
        "a store failure is an internal error, got {body}"
    );
}

fn mcp_post(body: String, header_method: Option<&str>, header_name: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder()
        .method(axum::http::Method::POST)
        .uri("/mcp")
        .header("authorization", format!("Bearer {ADMIN_TOKEN}"))
        .header(header::CONTENT_TYPE, "application/json")
        .header("mcp-protocol-version", "2026-07-28");
    if let Some(value) = header_method {
        builder = builder.header("mcp-method", value);
    }
    if let Some(value) = header_name {
        builder = builder.header("mcp-name", value);
    }
    builder.body(Body::from(body)).expect("request builds")
}

/// The token-gated `/metrics` route (plan 07 step 7): the Prometheus
/// exposition of the live runtime counters, behind the same admin token
/// as every other route on the surface, answering a scraper's
/// `text/plain` content type and never cacheable.
#[tokio::test]
async fn metrics_route_serves_prometheus_text_behind_the_admin_token() {
    let metrics = Arc::new(llm_proxy_core::Metrics::new());
    metrics.record_request(false);
    metrics.record_failure();
    let app = AdminSurface::new(ADMIN_TOKEN)
        .metrics(Arc::clone(&metrics))
        .router();

    // No or wrong token: the same 401 as every other admin route.
    for token in [None, Some("wrong-token")] {
        let mut builder = Request::builder().uri("/metrics");
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::empty()).expect("request builds"))
            .await
            .expect("response");
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "/metrics must sit behind the admin token"
        );
    }

    let response = app.oneshot(get("/metrics")).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; version=0.0.4; charset=utf-8"),
        "the exposition answers a scraper's content type"
    );
    assert_eq!(
        response
            .headers()
            .get(header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
        Some("no-store"),
        "authenticated metrics are never cacheable (backlog 37)"
    );
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("body");
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("llm_proxy_requests_total 1"),
        "the scraped exposition carries the live request counter: {text}"
    );
    assert!(
        text.contains("llm_proxy_requests_failed_total 1"),
        "the scraped exposition carries the live failure counter: {text}"
    );
}

/// Without the metrics registry attached the surface keeps its exact
/// shape: `/metrics` is not a route at all, like `/keys` without a key
/// store.
#[tokio::test]
async fn without_metrics_attached_the_route_does_not_exist() {
    let app = surface();
    let response = app.oneshot(get("/metrics")).await.expect("response");
    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "/metrics must not exist unless the metrics registry is attached"
    );
}
