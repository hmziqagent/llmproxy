//! The admin REST routes and their JSON handlers (shared verbatim with
//! the MCP tools): key management, the sanitized config view, and the
//! read-only usage views over the ledger.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::from_fn_with_state;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::trace::TraceLayer;

use crate::AdminSurface;
use crate::auth::NO_STORE;
use crate::keys::{IssuedKey, KeyPolicyInput};
use crate::mcp;
use crate::ui;

#[derive(Clone)]
pub(crate) struct AdminState {
    pub(crate) surface: Arc<AdminSurface>,
}

pub(crate) fn router(surface: AdminSurface) -> Router {
    let auth = Arc::clone(&surface.auth);
    let state = AdminState {
        surface: Arc::new(surface),
    };
    let mut routes = Router::new()
        .route("/", get(ui::index))
        .route("/keys", get(list_keys).post(create_key))
        .route("/keys/{key_id}/revoke", post(revoke_key))
        .route("/config", get(get_config))
        .route("/usage/summary", get(usage_summary))
        .route("/usage/requests", get(usage_requests))
        .route("/usage/keys", get(usage_keys))
        .route("/mcp", post(mcp::endpoint));
    // Like `UsageApi`, the route does not exist unless the metrics
    // registry was attached, so the surface shape is unchanged for
    // embedders that never call `AdminSurface::metrics`.
    if state.surface.metrics.is_some() {
        routes = routes.route("/metrics", get(prometheus_metrics));
    }
    routes
        .layer(from_fn_with_state(auth, crate::auth::require_admin))
        .layer(TraceLayer::new_for_http())
        .layer(SetResponseHeaderLayer::overriding(
            header::X_CONTENT_TYPE_OPTIONS,
            HeaderValue::from_static("nosniff"),
        ))
        // Authenticated admin data must never be cached by an
        // intermediary (backlog 37).
        .layer(SetResponseHeaderLayer::overriding(
            header::CACHE_CONTROL,
            NO_STORE,
        ))
        .layer(SetResponseHeaderLayer::overriding(
            header::PRAGMA,
            HeaderValue::from_static("no-cache"),
        ))
        .layer(CatchPanicLayer::new())
        .with_state(state)
}

/// Prometheus exposition of the runtime counters (plan 07 step 7),
/// behind the admin token like every other route on this surface. The
/// route exists only when `AdminSurface::metrics` was called, so the
/// missing-registry arm is unreachable through the router.
async fn prometheus_metrics(State(state): State<AdminState>) -> Response {
    let Some(metrics) = state.surface.metrics_ref() else {
        return admin_error("no metrics were attached to this surface".to_owned());
    };
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
        )],
        metrics.get_snapshot().to_prometheus_text(),
    )
        .into_response()
}

async fn list_keys(State(state): State<AdminState>) -> Response {
    match list_keys_json(&state.surface) {
        Ok(keys) => (StatusCode::OK, axum::Json(json!({ "keys": keys }))).into_response(),
        Err(error) => admin_error(error),
    }
}

pub(crate) fn list_keys_json(surface: &AdminSurface) -> Result<Value, String> {
    let keys = surface
        .keys_ref()
        .ok_or_else(|| "no key store was attached to this surface".to_owned())?
        .list()
        .map_err(|error| format!("the key store failed: {error}"))?;
    serde_json::to_value(keys).map_err(|error| format!("serializing key records: {error}"))
}

async fn create_key(
    State(state): State<AdminState>,
    axum::Json(policy): axum::Json<KeyPolicyInput>,
) -> Response {
    match create_key_json(&state.surface, &policy) {
        Ok(issued) => {
            use secrecy::ExposeSecret;
            (
                StatusCode::CREATED,
                axum::Json(json!({
                    "key_id": issued.key_id,
                    "token": issued.token.expose_secret(),
                    "created_at": issued.created_at,
                    "note": "the token is shown once; store it now",
                })),
            )
                .into_response()
        }
        Err(error) => action_error(error),
    }
}

/// An admin action failure with the HTTP status the client earned: the
/// operator's own input mistake is a 4xx, a broken surface or store is a
/// `500`. The MCP tools map 4xx to JSON-RPC `-32602` and the rest to
/// `-32603`.
pub(crate) struct AdminActionError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl AdminActionError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    /// Renders the JSON error body the REST routes answer with.
    pub(crate) fn into_response(self) -> Response {
        let kind = if self.status.is_client_error() {
            "invalid_request_error"
        } else {
            "internal_error"
        };
        (
            self.status,
            axum::Json(json!({
                "error": { "type": kind, "message": self.message },
            })),
        )
            .into_response()
    }
}

fn action_error(error: AdminActionError) -> Response {
    error.into_response()
}

pub(crate) fn create_key_json(
    surface: &AdminSurface,
    policy: &KeyPolicyInput,
) -> Result<IssuedKey, AdminActionError> {
    surface
        .keys_ref()
        .ok_or_else(|| AdminActionError::internal("no key store was attached to this surface"))?
        .create(policy)
        .map_err(|error| {
            let status = match &error {
                crate::keys::ManagedKeyError::InvalidPolicy(_) => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            AdminActionError {
                status,
                message: format!("issuing the key failed: {error}"),
            }
        })
}

async fn revoke_key(State(state): State<AdminState>, Path(key_id): Path<String>) -> Response {
    match revoke_key_json(&state.surface, &key_id) {
        Ok(()) => (StatusCode::OK, axum::Json(json!({ "revoked": key_id }))).into_response(),
        Err(error) => action_error(error),
    }
}

pub(crate) fn revoke_key_json(
    surface: &AdminSurface,
    key_id: &str,
) -> Result<(), AdminActionError> {
    let store = surface
        .keys_ref()
        .ok_or_else(|| AdminActionError::internal("no key store was attached to this surface"))?;
    store.revoke(key_id).map_err(|error| {
        let status = match &error {
            crate::keys::ManagedKeyError::UnknownKey(_) => StatusCode::NOT_FOUND,
            crate::keys::ManagedKeyError::AlreadyRevoked(_) => StatusCode::CONFLICT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        AdminActionError {
            status,
            message: format!("revoking the key failed: {error}"),
        }
    })
}

async fn get_config(State(state): State<AdminState>) -> Response {
    match state.surface.config_ref() {
        Some(config) => (
            StatusCode::OK,
            axum::Json(crate::config_view::sanitize_config(config)),
        )
            .into_response(),
        None => admin_error("no config was attached to this surface".to_owned()),
    }
}

/// The usage summary over a humantime window, grouped by provider, model
/// or caller — the shape the REST route and the MCP tool share.
pub(crate) async fn usage_summary_json(
    surface: &AdminSurface,
    window: &str,
    group_by: &str,
) -> Result<Value, AdminActionError> {
    use llm_proxy_storage::{GroupBy, now_unix_millis};
    let reader = surface.usage_ref().ok_or_else(|| {
        AdminActionError::internal("no usage views were attached to this surface")
    })?;
    let parsed = humantime::parse_duration(window).map_err(|parse_error| {
        AdminActionError::invalid(format!(
            "window must be a duration like \"24h\", got {window:?} ({parse_error})"
        ))
    })?;
    if parsed.is_zero() {
        return Err(AdminActionError::invalid(
            "window must be a positive duration",
        ));
    }
    let group = GroupBy::parse(group_by).ok_or_else(|| {
        AdminActionError::invalid(format!(
            "group_by must be one of provider, model, caller; got {group_by:?}"
        ))
    })?;
    let until_ms = now_unix_millis();
    let since_ms = until_ms.saturating_sub(i64::try_from(parsed.as_millis()).unwrap_or(i64::MAX));
    let groups = reader.summary(since_ms, group).await.map_err(|error| {
        AdminActionError::internal(format!("the usage ledger query failed: {error}"))
    })?;
    serde_json::to_value(json!({
        "since_ms": since_ms,
        "until_ms": until_ms,
        "group_by": group.as_str(),
        "groups": groups,
    }))
    .map_err(|error| AdminActionError::internal(format!("serializing the summary: {error}")))
}

#[derive(serde::Deserialize)]
struct SummaryParams {
    window: Option<String>,
    group_by: Option<String>,
}

async fn usage_summary(
    State(state): State<AdminState>,
    Query(params): Query<SummaryParams>,
) -> Response {
    match usage_summary_json(
        &state.surface,
        params.window.as_deref().unwrap_or("24h"),
        params.group_by.as_deref().unwrap_or("provider"),
    )
    .await
    {
        Ok(summary) => (StatusCode::OK, axum::Json(summary)).into_response(),
        Err(error) => action_error(error),
    }
}

async fn usage_requests(
    State(state): State<AdminState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Response {
    use llm_proxy_storage::{MAX_PAGE_LIMIT, RequestFilter};
    let reader = match state.surface.usage_ref() {
        Some(reader) => reader,
        None => return admin_error("no usage views were attached to this surface".to_owned()),
    };
    let limit = params
        .get("limit")
        .and_then(|raw| raw.parse::<i64>().ok())
        .unwrap_or(100);
    if !(1..=MAX_PAGE_LIMIT).contains(&limit) {
        return admin_error(format!(
            "limit must be between 1 and {MAX_PAGE_LIMIT}, got {limit}"
        ));
    }
    let page = reader
        .requests(RequestFilter {
            since_ms: None,
            until_ms: None,
            provider: params.get("provider").cloned(),
            limit,
            cursor: None,
        })
        .await;
    match page {
        Ok(page) => (
            StatusCode::OK,
            axum::Json(json!({ "rows": page.rows, "next_cursor": page.next_cursor })),
        )
            .into_response(),
        Err(error) => admin_error(format!("the usage ledger query failed: {error}")),
    }
}

async fn usage_keys(State(state): State<AdminState>) -> Response {
    let reader = match state.surface.usage_ref() {
        Some(reader) => reader,
        None => return admin_error("no usage views were attached to this surface".to_owned()),
    };
    match reader.callers().await {
        Ok(callers) => (StatusCode::OK, axum::Json(json!({ "callers": callers }))).into_response(),
        Err(error) => admin_error(format!("the usage ledger query failed: {error}")),
    }
}

fn admin_error(message: impl Into<String>) -> Response {
    let message = message.into();
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        axum::Json(json!({
            "error": { "type": "internal_error", "message": message },
        })),
    )
        .into_response()
}
