use axum::body::Body;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub(super) enum ClientProtocol {
    Anthropic,
    OpenAiChat,
    /// OpenAI Responses (`/v1/responses`, the Codex dialect; plan 08).
    OpenAiResponses,
    /// Gemini native (`/v1beta/models/{model}:generateContent`; plan 08).
    Gemini,
    /// Ollama native (`/api/chat`, `/api/generate`; plan 08).
    Ollama,
}

impl ClientProtocol {
    /// The wire framing a streaming response for this dialect uses: SSE with
    /// named events (Anthropic, Responses), SSE data-only lines (OpenAI Chat,
    /// Gemini `alt=sse`), NDJSON lines (Ollama) or a JSON array of chunks
    /// (Gemini `streamGenerateContent` without `alt=sse`).
    pub(super) fn stream_wire(self) -> StreamWire {
        match self {
            Self::Anthropic | Self::OpenAiResponses => StreamWire::SseNamed,
            Self::OpenAiChat => StreamWire::SseData,
            // The Gemini route chooses between SSE and the JSON array from
            // the `alt` query parameter; both flow through the same encoder.
            Self::Gemini => StreamWire::SseData,
            Self::Ollama => StreamWire::Ndjson,
        }
    }
}

/// How encoded stream chunks are framed onto the client connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamWire {
    /// SSE with an `event:` line per chunk (Anthropic Messages, Responses).
    SseNamed,
    /// SSE data-only lines (OpenAI Chat, Gemini `alt=sse`).
    SseData,
    /// NDJSON: one JSON object per line (Ollama).
    Ndjson,
    /// A JSON array of chunks (Gemini without `alt=sse`).
    JsonArray,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum AuthOwner {
    #[default]
    Operator,
    Client,
}
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub(super) enum RouteError {
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("model not allowed: {0}")]
    ModelNotAllowed(String),
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("unsupported route: {0}")]
    UnsupportedRoute(String),
    #[error("invalid provider name: {0}")]
    InvalidProviderName(String),
    #[error("upstream error: {status}")]
    Upstream {
        status: StatusCode,
        body: String,
        auth_owner: AuthOwner,
        /// The upstream-advised wait, in seconds, when the error response
        /// carried one (rendered as `Retry-After`).
        retry_after_secs: Option<u64>,
    },
    #[error("upstream timeout: {0}")]
    UpstreamTimeout(String),
    /// The engine's retry/failover ladder ran out (plan 04 step 12): the
    /// proxy already retried, so the response also carries
    /// `x-should-retry: false` and a computed `Retry-After`.
    #[error("retries exhausted")]
    RetryExhausted {
        /// The status of the final underlying error.
        status: StatusCode,
        /// The computed wait, in seconds (always present: the ladder derives
        /// one from the cooldown when the upstream advertised nothing).
        retry_after_secs: u64,
    },
    /// The route's affinity policy refused a failover (plan 04 step 10).
    #[error("session affinity refused the failover")]
    AffinityRefused,
    #[error("provider decode error: {0}")]
    ProviderDecode(String),
    #[error("internal error: {0}")]
    Internal(String),
    /// A rate window (RPM, TPM, RPD) refused the request: 429, with the
    /// window reset as `Retry-After` when known (plan 05 step 5).
    #[error("rate limit exceeded: {message}")]
    RateLimited {
        /// Which window refused (proxy-authored, safe for the client).
        message: String,
        /// The window reset in seconds, when the admission step knows it.
        retry_after_secs: Option<u64>,
    },
    /// The caller's spend budget refused the request: 402 (plan 05 step 4).
    #[error("spend budget exhausted: {message}")]
    BudgetExhausted {
        /// Which budget refused (proxy-authored, safe for the client).
        message: String,
        /// The budget window reset in seconds, when known.
        retry_after_secs: Option<u64>,
    },
    /// A concurrency cap is saturated (plan 05 step 7): 503 with a short
    /// `Retry-After`, never 429.
    #[error("at capacity: {message}")]
    Overloaded {
        /// Which cap saturated (proxy-authored, safe for the client).
        message: String,
        /// A short wait in seconds after which a slot may be free.
        retry_after_secs: Option<u64>,
    },
    /// The caller's key may not reach this provider or model (plan 05 step
    /// 3): 403.
    #[error("{0}")]
    Forbidden(String),
    /// Another request from this caller holds the same `Idempotency-Key`
    /// and is still in flight (plan 06 step 3): 409 while it runs.
    #[error("a request with this Idempotency-Key is already in flight")]
    IdempotencyInFlight,
    #[error("not found")]
    NotFound,
    #[error("request timeout")]
    RequestTimeout,
    #[error("request body too large")]
    PayloadTooLarge,
    #[error("method not allowed")]
    MethodNotAllowed,
    #[error("missing client auth token for passthrough provider")]
    Unauthorized,
}
#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct AnthropicErrorBody {
    r#type: &'static str,
    error: AnthropicErrorDetail,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct AnthropicErrorDetail {
    r#type: String,
    message: String,
}
#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct OpenAiErrorBody {
    error: OpenAiErrorDetail,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct OpenAiErrorDetail {
    message: String,
    r#type: String,
    code: serde_json::Value,
}
pub(super) fn route_error_response(protocol: ClientProtocol, error: &RouteError) -> Response<Body> {
    let mut response = match protocol {
        ClientProtocol::Anthropic => anthropic_error_response(error),
        ClientProtocol::OpenAiChat | ClientProtocol::OpenAiResponses => {
            openai_error_response(error)
        }
        ClientProtocol::Gemini => gemini_error_response(error),
        ClientProtocol::Ollama => ollama_error_response(error),
    };
    // One policy table (plan 05 step 1, audit conflict 14): every error
    // response carries the deliberate x-should-retry choice for its status,
    // not just the ladder-exhaustion case. An exhausted ladder overrides its
    // row: the proxy already did the retrying (plan 04 step 12).
    let (status, _type, _message) = extract_error_fields(error);
    let should_retry = match error {
        RouteError::RetryExhausted { .. } => false,
        // A status outside the table (none ship today — the coverage test
        // pins this) defaults to "do not retry".
        _ => llm_proxy_engine::policy_for(status.as_u16()).is_some_and(|row| row.should_retry),
    };
    response.headers_mut().insert(
        X_SHOULD_RETRY,
        HeaderValue::from_static(if should_retry { "true" } else { "false" }),
    );
    if let Some(secs) = error.retry_after_secs() {
        if let Ok(value) = HeaderValue::from_str(&secs.to_string()) {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
    }
    response
}
pub(crate) const INTERNAL_ERROR_CLIENT_MESSAGE: &str = "internal server error";
pub(crate) const PROVIDER_DECODE_CLIENT_MESSAGE: &str = "provider response decode error";
pub(crate) const UPSTREAM_PASSTHROUGH_CLIENT_MESSAGE: &str = "upstream rejected the request";
pub(crate) const UPSTREAM_COLLAPSE_CLIENT_MESSAGE: &str = "upstream service error";

/// The `x-should-retry` header name, shared by [`route_error_response`] and
/// [`stamp_retry_policy`].
const X_SHOULD_RETRY: &str = "x-should-retry";

/// Stamps `x-should-retry` on an error response that was not built through
/// [`route_error_response`] — the loop guard's 508, the host guard's 403, a
/// `CatchPanicLayer` 500 — so the one policy table governs every error the
/// router emits (plan 05 step 1). An existing header is never overwritten
/// (`RetryExhausted` relies on its own `false`), and non-error responses are
/// left alone.
pub(crate) fn stamp_retry_policy(response: &mut Response) {
    let status = response.status();
    if status.as_u16() < 400 {
        return;
    }
    if response.headers().contains_key(X_SHOULD_RETRY) {
        return;
    }
    let should_retry = match llm_proxy_engine::policy_for(status.as_u16()) {
        Some(row) => row.should_retry,
        None => {
            tracing::error!(
                status = status.as_u16(),
                "response status has no policy row"
            );
            false
        }
    };
    response.headers_mut().insert(
        X_SHOULD_RETRY,
        HeaderValue::from_static(if should_retry { "true" } else { "false" }),
    );
}
fn anthropic_error_response(error: &RouteError) -> Response<Body> {
    let (status, error_type, message) = extract_error_fields(error);

    let body = AnthropicErrorBody {
        r#type: "error",
        error: AnthropicErrorDetail {
            r#type: error_type.to_owned(),
            message,
        },
    };
    let mut response = (status, axum::Json(body)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}
fn openai_error_response(error: &RouteError) -> Response<Body> {
    let (status, error_type, message) = extract_error_fields(error);
    let error_type = match status {
        StatusCode::INTERNAL_SERVER_ERROR => "server_error",
        StatusCode::NOT_FOUND => "invalid_request_error",
        _ => error_type,
    };

    let body = OpenAiErrorBody {
        error: OpenAiErrorDetail {
            message,
            r#type: error_type.to_owned(),
            code: serde_json::Value::Null,
        },
    };
    let mut response = (status, axum::Json(body)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Gemini error bodies: `{"error": {"code", "message", "status"}}`.
fn gemini_error_response(error: &RouteError) -> Response<Body> {
    let (status, _type, message) = extract_error_fields(error);
    let body = llm_proxy_protocol::client::gemini::error_payload(status.as_u16(), &message);
    let mut response = (status, axum::Json(body)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Ollama error bodies: `{"error": "<message>"}`.
fn ollama_error_response(error: &RouteError) -> Response<Body> {
    let (status, _type, message) = extract_error_fields(error);
    let body = llm_proxy_protocol::client::ollama::error_payload(&message);
    let mut response = (status, axum::Json(body)).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}
impl RouteError {
    /// The advised wait in seconds this error carries, when any: the value
    /// rendered as `Retry-After` (from an upstream header, a rate-window
    /// reset, or a ladder computation). `None` for errors with no advice.
    fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::Upstream {
                retry_after_secs, ..
            }
            | Self::RateLimited {
                retry_after_secs, ..
            }
            | Self::BudgetExhausted {
                retry_after_secs, ..
            }
            | Self::Overloaded {
                retry_after_secs, ..
            } => *retry_after_secs,
            Self::RetryExhausted {
                retry_after_secs, ..
            } => Some(*retry_after_secs),
            _ => None,
        }
    }
}

impl From<llm_proxy_engine::EngineError> for RouteError {
    fn from(error: llm_proxy_engine::EngineError) -> Self {
        use llm_proxy_engine::EngineError as E;
        let secs = |wait: Option<std::time::Duration>| wait.map(|w| w.as_secs().max(1));
        match error {
            E::InvalidProviderName(m) => Self::InvalidProviderName(m),
            E::UnknownProvider(m) => Self::UnknownProvider(m),
            E::UnsupportedRoute(m) => Self::UnsupportedRoute(m),
            E::ModelNotAllowed(m) => Self::ModelNotAllowed(m),
            E::Unauthorized => Self::Unauthorized,
            E::InvalidRequest(m) => Self::InvalidRequest(m),
            E::ProviderDecode(m) => Self::ProviderDecode(m),
            E::Upstream {
                status,
                body,
                auth_owner,
                retry_after,
            } => {
                let owner = match auth_owner {
                    llm_proxy_engine::AuthOwner::Client => AuthOwner::Client,
                    llm_proxy_engine::AuthOwner::Operator => AuthOwner::Operator,
                };
                let code = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
                Self::Upstream {
                    status: code,
                    body,
                    auth_owner: owner,
                    retry_after_secs: secs(retry_after),
                }
            }
            E::UpstreamTimeout(m) => Self::UpstreamTimeout(m),
            E::RetryExhausted {
                last, retry_after, ..
            } => Self::RetryExhausted {
                status: StatusCode::from_u16(last.status_code()).unwrap_or(StatusCode::BAD_GATEWAY),
                retry_after_secs: retry_after.map(|wait| wait.as_secs().max(1)).unwrap_or(1),
            },
            E::AffinityRefused { .. } => Self::AffinityRefused,
            E::RateLimited {
                message,
                retry_after,
            } => Self::RateLimited {
                message,
                retry_after_secs: secs(retry_after),
            },
            E::BudgetExhausted {
                message,
                retry_after,
            } => Self::BudgetExhausted {
                message,
                retry_after_secs: secs(retry_after),
            },
            E::Overloaded {
                message,
                retry_after,
            } => Self::Overloaded {
                message,
                retry_after_secs: secs(retry_after),
            },
            E::Forbidden(m) => Self::Forbidden(m),
            E::IdempotencyInFlight => Self::IdempotencyInFlight,
            E::Internal(m) => Self::Internal(m),
            other => Self::Internal(other.to_string()),
        }
    }
}

pub(crate) fn route_error_variant_name(error: &RouteError) -> &'static str {
    match error {
        RouteError::InvalidRequest(_) => "InvalidRequest",
        RouteError::ModelNotAllowed(_) => "ModelNotAllowed",
        RouteError::UnknownProvider(_) => "UnknownProvider",
        RouteError::UnsupportedRoute(_) => "UnsupportedRoute",
        RouteError::InvalidProviderName(_) => "InvalidProviderName",
        RouteError::Upstream { .. } => "Upstream",
        RouteError::UpstreamTimeout(_) => "UpstreamTimeout",
        RouteError::RetryExhausted { .. } => "RetryExhausted",
        RouteError::AffinityRefused => "AffinityRefused",
        RouteError::ProviderDecode(_) => "ProviderDecode",
        RouteError::Internal(_) => "Internal",
        RouteError::RateLimited { .. } => "RateLimited",
        RouteError::BudgetExhausted { .. } => "BudgetExhausted",
        RouteError::Overloaded { .. } => "Overloaded",
        RouteError::Forbidden(_) => "Forbidden",
        RouteError::IdempotencyInFlight => "IdempotencyInFlight",
        RouteError::NotFound => "NotFound",
        RouteError::RequestTimeout => "RequestTimeout",
        RouteError::PayloadTooLarge => "PayloadTooLarge",
        RouteError::MethodNotAllowed => "MethodNotAllowed",
        RouteError::Unauthorized => "Unauthorized",
    }
}
#[expect(
    clippy::too_many_lines,
    reason = "one row per RouteError variant: the status/type/message table reads as a unit"
)]
pub(crate) fn extract_error_fields(error: &RouteError) -> (StatusCode, &'static str, String) {
    match error {
        RouteError::InvalidRequest(msg) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            msg.clone(),
        ),
        RouteError::ModelNotAllowed(model) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("model not allowed: {model}"),
        ),
        RouteError::UnknownProvider(provider) => (
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("unknown provider: {provider}"),
        ),
        RouteError::UnsupportedRoute(msg) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            msg.clone(),
        ),
        RouteError::InvalidProviderName(msg) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("invalid provider name: {msg}"),
        ),
        RouteError::Upstream {
            status,
            body: _body,
            auth_owner,
            retry_after_secs: _,
        } => {
            let mapped = map_upstream_status(*status, *auth_owner);
            let message = if mapped == *status {
                UPSTREAM_PASSTHROUGH_CLIENT_MESSAGE
            } else {
                UPSTREAM_COLLAPSE_CLIENT_MESSAGE
            };
            (mapped, "api_error", truncate_error_body(message))
        }
        RouteError::UpstreamTimeout(_msg) => (
            StatusCode::GATEWAY_TIMEOUT,
            "api_error",
            "upstream request timed out".to_owned(),
        ),
        RouteError::RetryExhausted { status, .. } => (
            *status,
            "api_error",
            "upstream unavailable after retries; the proxy has already retried this request"
                .to_owned(),
        ),
        RouteError::AffinityRefused => (
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            "session is pinned to an upstream that is unavailable".to_owned(),
        ),
        RouteError::ProviderDecode(_msg) => (
            StatusCode::BAD_GATEWAY,
            "api_error",
            PROVIDER_DECODE_CLIENT_MESSAGE.to_owned(),
        ),
        RouteError::Internal(_msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            INTERNAL_ERROR_CLIENT_MESSAGE.to_owned(),
        ),
        RouteError::RateLimited { message, .. } => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            truncate_error_body(message),
        ),
        RouteError::BudgetExhausted { message, .. } => (
            StatusCode::PAYMENT_REQUIRED,
            "billing_error",
            truncate_error_body(message),
        ),
        RouteError::Overloaded { message, .. } => (
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            truncate_error_body(message),
        ),
        RouteError::Forbidden(message) => (
            StatusCode::FORBIDDEN,
            "permission_error",
            truncate_error_body(message),
        ),
        RouteError::IdempotencyInFlight => (
            StatusCode::CONFLICT,
            "invalid_request_error",
            "a request with this Idempotency-Key is already in flight; retry once \
             it completes to replay its response"
                .to_owned(),
        ),
        RouteError::NotFound => (
            StatusCode::NOT_FOUND,
            "not_found_error",
            "not found".to_owned(),
        ),
        RouteError::RequestTimeout => (
            StatusCode::REQUEST_TIMEOUT,
            "timeout_error",
            "request timed out".to_owned(),
        ),
        RouteError::PayloadTooLarge => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            "request body too large".to_owned(),
        ),
        RouteError::MethodNotAllowed => (
            StatusCode::METHOD_NOT_ALLOWED,
            "invalid_request_error",
            "method not allowed".to_owned(),
        ),
        RouteError::Unauthorized => (
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "missing or invalid client auth token".to_owned(),
        ),
    }
}
fn map_upstream_status(upstream: StatusCode, auth_owner: AuthOwner) -> StatusCode {
    // One policy, defined in the engine (audit conflict 14): the server only
    // translates it to `StatusCode` and keeps the diagnostics.
    let code = upstream.as_u16();
    let owner = match auth_owner {
        AuthOwner::Client => llm_proxy_engine::AuthOwner::Client,
        AuthOwner::Operator => llm_proxy_engine::AuthOwner::Operator,
    };
    let mapped = llm_proxy_engine::map_upstream_status(code, owner);
    if matches!(code, 401 | 403 | 404) && mapped != code {
        tracing::warn!(
            upstream_status = code,
            auth_owner = ?auth_owner,
            "upstream returned a status that may indicate a configuration \
             issue (auth, forbidden, or not found); mapping to 502 for client"
        );
    } else if code < 400 && mapped != code {
        tracing::debug!(
            upstream_status = code,
            "unexpected 1xx/2xx/3xx upstream status in error path; mapping to 502"
        );
    }
    StatusCode::from_u16(mapped).unwrap_or(StatusCode::BAD_GATEWAY)
}
const MAX_ERROR_MESSAGE_LEN: usize = 512;
const TRUNCATED_SUFFIX: &str = "...[truncated]";
pub(crate) fn truncate_with_suffix(s: &str, max_len: usize, suffix: &str) -> String {
    llm_proxy_protocol::util::truncate_with_suffix(s, max_len, suffix)
}
fn truncate_error_body(body: &str) -> String {
    truncate_with_suffix(body, MAX_ERROR_MESSAGE_LEN, TRUNCATED_SUFFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_invalid_request_returns_400() {
        let err = RouteError::InvalidRequest("bad input".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn anthropic_model_not_allowed_returns_400() {
        let err = RouteError::ModelNotAllowed("gpt-99".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn anthropic_upstream_500_returns_502() {
        let err = RouteError::Upstream {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: "upstream error".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn anthropic_upstream_429_returns_429() {
        let err = RouteError::Upstream {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: "rate limited".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn anthropic_provider_decode_returns_502() {
        let err = RouteError::ProviderDecode("bad frame with sensitive data".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn anthropic_internal_returns_500() {
        let err = RouteError::Internal("config missing".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn anthropic_rate_limited_returns_429() {
        let err = RouteError::RateLimited {
            message: "rate limit exceeded".to_owned(),
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn internal_error_message_is_sanitized_in_response_body() {
        let err = RouteError::Internal("secret config detail: /etc/proxy.toml".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let message = json["error"]["message"].as_str().unwrap();
        assert_eq!(message, INTERNAL_ERROR_CLIENT_MESSAGE);
        assert!(
            !message.contains("secret"),
            "internal error message must not contain internal details"
        );
    }

    #[tokio::test]
    async fn provider_decode_error_message_is_sanitized_in_response_body() {
        let err = RouteError::ProviderDecode(
            "decode response: upstream returned malformed JSON with api_key=sk-ant-abc123".into(),
        );
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let message = json["error"]["message"].as_str().unwrap();
        assert_eq!(message, PROVIDER_DECODE_CLIENT_MESSAGE);
        assert!(
            !message.contains("upstream"),
            "provider decode error must not contain upstream details"
        );
    }

    #[test]
    fn openai_invalid_request_returns_400() {
        let err = RouteError::InvalidRequest("bad input".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn openai_model_not_allowed_returns_400() {
        let err = RouteError::ModelNotAllowed("gpt-99".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn openai_upstream_500_returns_502() {
        let err = RouteError::Upstream {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: "upstream error".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn openai_upstream_429_returns_429() {
        let err = RouteError::Upstream {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: "rate limited".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn openai_provider_decode_returns_502() {
        let err = RouteError::ProviderDecode("bad frame".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn openai_internal_returns_500() {
        let err = RouteError::Internal("config missing".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn openai_rate_limited_returns_429() {
        let err = RouteError::RateLimited {
            message: "rate limit exceeded".to_owned(),
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn openai_invalid_request_body_has_correct_structure() {
        let err = RouteError::InvalidRequest("bad input".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert!(json["error"].is_object(), "must have error object");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "bad input");
        assert!(json["error"]["code"].is_null(), "code must be null");
        assert!(
            json.get("type").is_none() || json["type"].is_null(),
            "must not have Anthropic 'type' field"
        );
    }

    #[tokio::test]
    async fn openai_model_not_allowed_body_has_correct_structure() {
        let err = RouteError::ModelNotAllowed("gpt-99".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("gpt-99")
        );
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn openai_upstream_500_body_has_correct_structure() {
        let err = RouteError::Upstream {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: "model 'gpt-foo' is overloaded: request echoed here".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["error"]["type"], "api_error");
        assert_eq!(json["error"]["message"], UPSTREAM_COLLAPSE_CLIENT_MESSAGE);
        assert!(json["error"]["code"].is_null());
        let message = json["error"]["message"].as_str().unwrap();
        assert!(!message.contains("gpt-foo"));
        assert!(!message.contains("overloaded"));
        assert!(!message.contains("echoed"));
    }

    #[tokio::test]
    async fn upstream_passthrough_400_uses_generic_message() {
        let err = RouteError::Upstream {
            status: StatusCode::BAD_REQUEST,
            body: r#"{"error":{"message":"model 'gpt-foo' does not exist","type":"invalid_request_error"}}"#.into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let message = json["error"]["message"].as_str().unwrap();
        assert_eq!(message, UPSTREAM_PASSTHROUGH_CLIENT_MESSAGE);
        assert!(!message.contains("gpt-foo"));
        assert!(!message.contains("invalid_request_error"));
    }

    #[tokio::test]
    async fn upstream_passthrough_401_client_uses_generic_message_and_passes_through() {
        let err = RouteError::Upstream {
            status: StatusCode::UNAUTHORIZED,
            body: "invalid api key sk-leaked".into(),
            auth_owner: AuthOwner::Client,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let message = json["error"]["message"].as_str().unwrap();
        assert_eq!(message, UPSTREAM_PASSTHROUGH_CLIENT_MESSAGE);
        assert!(!message.contains("sk-leaked"));
    }

    #[tokio::test]
    async fn upstream_401_operator_collapses_to_502() {
        let err = RouteError::Upstream {
            status: StatusCode::UNAUTHORIZED,
            body: "invalid api key".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn openai_internal_message_is_sanitized() {
        let err = RouteError::Internal("secret config detail: /etc/proxy.toml".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let message = json["error"]["message"].as_str().unwrap();
        assert_eq!(message, INTERNAL_ERROR_CLIENT_MESSAGE);
        assert!(
            !message.contains("secret"),
            "internal error message must not contain internal details"
        );
    }

    #[tokio::test]
    async fn openai_provider_decode_message_is_sanitized() {
        let err = RouteError::ProviderDecode(
            "decode response: upstream returned malformed JSON with api_key=sk-ant-abc123".into(),
        );
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let message = json["error"]["message"].as_str().unwrap();
        assert_eq!(message, PROVIDER_DECODE_CLIENT_MESSAGE);
    }

    #[tokio::test]
    async fn openai_rate_limited_body_has_correct_structure() {
        let err = RouteError::RateLimited {
            message: "rate limit exceeded".to_owned(),
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["error"]["type"], "rate_limit_error");
        assert_eq!(json["error"]["message"], "rate limit exceeded");
        assert!(json["error"]["code"].is_null());
    }

    #[test]
    fn openai_error_has_json_content_type() {
        let err = RouteError::InvalidRequest("test".into());
        let response = route_error_response(ClientProtocol::OpenAiChat, &err);
        let ct = response
            .headers()
            .get(header::CONTENT_TYPE)
            .expect("content-type header");
        assert_eq!(ct, "application/json");
    }

    #[test]
    fn request_timeout_returns_408() {
        let response = route_error_response(ClientProtocol::Anthropic, &RouteError::RequestTimeout);
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn payload_too_large_returns_413() {
        let response =
            route_error_response(ClientProtocol::Anthropic, &RouteError::PayloadTooLarge);
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn request_timeout_anthropic_body_is_json_envelope() {
        let response = route_error_response(ClientProtocol::Anthropic, &RouteError::RequestTimeout);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "timeout_error");
        assert_eq!(json["error"]["message"], "request timed out");
    }

    #[tokio::test]
    async fn payload_too_large_openai_body_is_json_envelope() {
        let response =
            route_error_response(ClientProtocol::OpenAiChat, &RouteError::PayloadTooLarge);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(json["error"]["code"].is_null());
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("too large")
        );
    }

    #[test]
    fn method_not_allowed_returns_405() {
        let response =
            route_error_response(ClientProtocol::Anthropic, &RouteError::MethodNotAllowed);
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn method_not_allowed_anthropic_body_is_json_envelope() {
        let response =
            route_error_response(ClientProtocol::Anthropic, &RouteError::MethodNotAllowed);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "method not allowed");
    }

    #[tokio::test]
    async fn method_not_allowed_openai_body_is_json_envelope() {
        let response =
            route_error_response(ClientProtocol::OpenAiChat, &RouteError::MethodNotAllowed);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "method not allowed");
        assert!(json["error"]["code"].is_null());
        assert!(
            json.get("type").is_none() || json["type"].is_null(),
            "OpenAI 405 must not carry the Anthropic 'type' field"
        );
    }

    #[test]
    fn route_error_variant_name_returns_variant_identifier() {
        assert_eq!(
            route_error_variant_name(&RouteError::UnknownProvider("p".into())),
            "UnknownProvider"
        );
        assert_eq!(route_error_variant_name(&RouteError::NotFound), "NotFound");
        assert_eq!(
            route_error_variant_name(&RouteError::Forbidden("no".into())),
            "Forbidden"
        );
        assert_eq!(
            route_error_variant_name(&RouteError::RequestTimeout),
            "RequestTimeout"
        );
        assert_eq!(
            route_error_variant_name(&RouteError::PayloadTooLarge),
            "PayloadTooLarge"
        );
        assert_eq!(
            route_error_variant_name(&RouteError::MethodNotAllowed),
            "MethodNotAllowed"
        );
        assert_eq!(
            route_error_variant_name(&RouteError::Upstream {
                status: StatusCode::BAD_GATEWAY,
                body: String::new(),
                auth_owner: AuthOwner::Operator,
                retry_after_secs: None,
            }),
            "Upstream"
        );
        let (status, client_type, _msg) =
            extract_error_fields(&RouteError::UnknownProvider("p".into()));
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_ne!(client_type, "UnknownProvider");
    }

    #[test]
    fn upstream_400_passes_through() {
        assert_eq!(
            map_upstream_status(StatusCode::BAD_REQUEST, AuthOwner::Operator),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn upstream_401_operator_maps_to_502() {
        assert_eq!(
            map_upstream_status(StatusCode::UNAUTHORIZED, AuthOwner::Operator),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn upstream_401_client_passes_through() {
        assert_eq!(
            map_upstream_status(StatusCode::UNAUTHORIZED, AuthOwner::Client),
            StatusCode::UNAUTHORIZED
        );
    }

    #[test]
    fn upstream_403_operator_maps_to_502() {
        assert_eq!(
            map_upstream_status(StatusCode::FORBIDDEN, AuthOwner::Operator),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn upstream_403_client_passes_through() {
        assert_eq!(
            map_upstream_status(StatusCode::FORBIDDEN, AuthOwner::Client),
            StatusCode::FORBIDDEN
        );
    }

    #[test]
    fn upstream_413_passes_through() {
        assert_eq!(
            map_upstream_status(StatusCode::PAYLOAD_TOO_LARGE, AuthOwner::Operator),
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[test]
    fn upstream_429_maps_to_429() {
        assert_eq!(
            map_upstream_status(StatusCode::TOO_MANY_REQUESTS, AuthOwner::Operator),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn upstream_404_maps_to_502() {
        assert_eq!(
            map_upstream_status(StatusCode::NOT_FOUND, AuthOwner::Client),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn upstream_500_maps_to_502() {
        assert_eq!(
            map_upstream_status(StatusCode::INTERNAL_SERVER_ERROR, AuthOwner::Operator),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn truncate_short_body_unchanged() {
        let body = "short error";
        assert_eq!(truncate_error_body(body), body);
    }

    #[test]
    fn truncate_long_body() {
        let body = "x".repeat(600);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("...[truncated]"));
        assert!(
            truncated.len() <= MAX_ERROR_MESSAGE_LEN,
            "truncated body must not exceed MAX_ERROR_MESSAGE_LEN, got {}",
            truncated.len()
        );
    }

    #[test]
    fn truncate_exact_length_is_unchanged() {
        let body = "a".repeat(MAX_ERROR_MESSAGE_LEN);
        assert_eq!(truncate_error_body(&body), body);
        let body = "a".repeat(MAX_ERROR_MESSAGE_LEN + 1);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("...[truncated]"));
        assert!(truncated.len() <= MAX_ERROR_MESSAGE_LEN);
    }

    #[test]
    fn truncate_multibyte_at_boundary_does_not_panic() {
        let body = "あ".repeat(200);
        let truncated = truncate_error_body(&body);
        assert!(truncated.len() <= MAX_ERROR_MESSAGE_LEN);
        assert!(truncated.ends_with("...[truncated]"));
    }

    #[test]
    fn anthropic_error_has_json_content_type() {
        let err = RouteError::InvalidRequest("test".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let ct = response
            .headers()
            .get(header::CONTENT_TYPE)
            .expect("content-type header");
        assert_eq!(ct, "application/json");
    }

    #[tokio::test]
    async fn anthropic_invalid_request_body_has_correct_structure() {
        let err = RouteError::InvalidRequest("bad input".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "bad input");
    }

    #[tokio::test]
    async fn anthropic_model_not_allowed_body_has_correct_structure() {
        let err = RouteError::ModelNotAllowed("gpt-99".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("gpt-99")
        );
    }

    #[tokio::test]
    async fn anthropic_upstream_500_body_has_correct_structure() {
        let err = RouteError::Upstream {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: "model 'gpt-foo' is overloaded: request echoed here".into(),
            auth_owner: AuthOwner::Operator,
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "api_error");
        assert_eq!(json["error"]["message"], UPSTREAM_COLLAPSE_CLIENT_MESSAGE);
        let message = json["error"]["message"].as_str().unwrap();
        assert!(!message.contains("gpt-foo"));
        assert!(!message.contains("overloaded"));
        assert!(!message.contains("echoed"));
    }

    #[tokio::test]
    async fn anthropic_internal_body_uses_generic_message() {
        let err = RouteError::Internal("config missing".into());
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "api_error");
        assert_eq!(json["error"]["message"], INTERNAL_ERROR_CLIENT_MESSAGE);
    }

    #[tokio::test]
    async fn anthropic_rate_limited_body_has_correct_structure() {
        let err = RouteError::RateLimited {
            message: "rate limit exceeded".to_owned(),
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::Anthropic, &err);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "rate_limit_error");
        assert_eq!(json["error"]["message"], "rate limit exceeded");
    }

    #[tokio::test]
    async fn idempotency_in_flight_renders_409_in_both_dialects() {
        for protocol in [ClientProtocol::Anthropic, ClientProtocol::OpenAiChat] {
            let response = route_error_response(protocol, &RouteError::IdempotencyInFlight);
            assert_eq!(response.status(), StatusCode::CONFLICT);
            // The policy row's deliberate choice: a stock SDK that auto-retries
            // 409 clears this by itself once the first request finishes.
            assert_eq!(
                response
                    .headers()
                    .get("x-should-retry")
                    .map(|v| v.to_str().unwrap()),
                Some("true")
            );
            // The client-visible message reads as one sentence — no run of
            // spaces where a string continuation was folded.
            let body = axum::body::to_bytes(response.into_parts().1, 4096)
                .await
                .expect("body");
            let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
            let message = json["error"]["message"].as_str().unwrap();
            assert!(
                message.contains("retry once it completes to replay its response"),
                "the 409 message must read cleanly: {message}"
            );
            assert!(
                !message.contains("  "),
                "the 409 message must not carry a run of spaces: {message}"
            );
        }
    }

    #[test]
    fn client_protocol_equality() {
        assert_eq!(ClientProtocol::Anthropic, ClientProtocol::Anthropic);
        assert_ne!(ClientProtocol::Anthropic, ClientProtocol::OpenAiChat);
    }

    /// One representative per `RouteError` variant: the status table the
    /// server renders is exactly the engine's policy table (plan 05 step 1).
    fn every_route_error() -> Vec<RouteError> {
        vec![
            RouteError::InvalidRequest("x".into()),
            RouteError::ModelNotAllowed("x".into()),
            RouteError::UnknownProvider("x".into()),
            RouteError::UnsupportedRoute("x".into()),
            RouteError::InvalidProviderName("x".into()),
            RouteError::Upstream {
                status: StatusCode::TOO_MANY_REQUESTS,
                body: String::new(),
                auth_owner: AuthOwner::Operator,
                retry_after_secs: Some(7),
            },
            RouteError::Upstream {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: String::new(),
                auth_owner: AuthOwner::Operator,
                retry_after_secs: None,
            },
            RouteError::UpstreamTimeout("x".into()),
            RouteError::RetryExhausted {
                status: StatusCode::BAD_GATEWAY,
                retry_after_secs: 4,
            },
            RouteError::AffinityRefused,
            RouteError::ProviderDecode("x".into()),
            RouteError::Internal("x".into()),
            RouteError::RateLimited {
                message: "rpm".into(),
                retry_after_secs: Some(30),
            },
            RouteError::BudgetExhausted {
                message: "daily".into(),
                retry_after_secs: None,
            },
            RouteError::Overloaded {
                message: "cap".into(),
                retry_after_secs: Some(1),
            },
            RouteError::Forbidden("no".into()),
            RouteError::IdempotencyInFlight,
            RouteError::NotFound,
            RouteError::RequestTimeout,
            RouteError::PayloadTooLarge,
            RouteError::MethodNotAllowed,
            RouteError::Unauthorized,
        ]
    }

    #[test]
    fn every_server_status_is_in_the_engine_policy_table() {
        for error in every_route_error() {
            let (status, _type, _message) = extract_error_fields(&error);
            assert!(
                llm_proxy_engine::policy_for(status.as_u16()).is_some(),
                "{error:?} renders {status}, which has no policy row"
            );
        }
    }

    #[test]
    fn every_error_response_carries_the_tables_retry_choice() {
        for error in every_route_error() {
            for protocol in [ClientProtocol::Anthropic, ClientProtocol::OpenAiChat] {
                let response = route_error_response(protocol, &error);
                let status = response.status().as_u16();
                let expected = if matches!(error, RouteError::RetryExhausted { .. }) {
                    "false"
                } else if llm_proxy_engine::policy_for(status)
                    .expect("row exists")
                    .should_retry
                {
                    "true"
                } else {
                    "false"
                };
                assert_eq!(
                    response
                        .headers()
                        .get("x-should-retry")
                        .and_then(|v| v.to_str().ok()),
                    Some(expected),
                    "{error:?} ({status}) must carry the table's x-should-retry"
                );
            }
        }
    }

    #[test]
    fn advised_waits_render_as_retry_after_only_when_carried() {
        let with_wait = RouteError::Overloaded {
            message: "cap".into(),
            retry_after_secs: Some(1),
        };
        let response = route_error_response(ClientProtocol::OpenAiChat, &with_wait);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );

        let without = RouteError::BudgetExhausted {
            message: "daily".into(),
            retry_after_secs: None,
        };
        let response = route_error_response(ClientProtocol::Anthropic, &without);
        assert_eq!(response.status(), StatusCode::PAYMENT_REQUIRED);
        assert!(response.headers().get(header::RETRY_AFTER).is_none());
        assert_eq!(
            response
                .headers()
                .get("x-should-retry")
                .and_then(|v| v.to_str().ok()),
            Some("false"),
            "a spend budget is not something an SDK retry ladder can wait out"
        );
    }

    #[test]
    fn engine_admission_refusals_keep_their_status_through_the_server() {
        use llm_proxy_engine::EngineError;
        let cases: Vec<(EngineError, StatusCode)> = vec![
            (
                EngineError::RateLimited {
                    message: "rpm".into(),
                    retry_after: Some(std::time::Duration::from_secs(12)),
                },
                StatusCode::TOO_MANY_REQUESTS,
            ),
            (
                EngineError::BudgetExhausted {
                    message: "usd".into(),
                    retry_after: None,
                },
                StatusCode::PAYMENT_REQUIRED,
            ),
            (
                EngineError::Overloaded {
                    message: "cap".into(),
                    retry_after: Some(std::time::Duration::from_secs(1)),
                },
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (EngineError::Forbidden("no".into()), StatusCode::FORBIDDEN),
        ];
        for (engine_error, expected) in cases {
            let engine_status = engine_error.status_code();
            let route: RouteError = engine_error.into();
            let (status, _type, _message) = extract_error_fields(&route);
            assert_eq!(status, expected, "{route:?}");
            assert_eq!(
                status.as_u16(),
                engine_status,
                "engine and server must agree on the status for {route:?}"
            );
        }
    }
}
