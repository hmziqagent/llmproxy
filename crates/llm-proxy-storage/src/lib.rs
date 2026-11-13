//! Event types and the [`EventBus`] trait for the per-request usage ledger.
//! Backends: [`NoopBus`] (default), [`RecordingBus`] (behind the `test-utils`
//! feature), and [`SqliteEventBus`] (persistent SQLite ledger, behind the
//! `sqlite` feature which is on by default).

#![deny(missing_docs)]

use llm_proxy_protocol::core::{Cost, ModelRef, StopReason, Usage};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
mod usage_query;

#[cfg(feature = "sqlite")]
pub use sqlite::{RetentionPolicy, SqliteEventBus, StorageError, run_retention_pass};
#[cfg(feature = "sqlite")]
pub use usage_query::{
    CallerDaySpend, CallerSummary, CostTotals, GroupBy, MAX_PAGE_LIMIT, RequestFilter, RequestRow,
    RequestsPage, SummaryGroup, TokenCounts, UsageQueryError, UsageReader, now_unix_millis,
    spend_since,
};

/// One request/response lifecycle event, ready to persist or emit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProxyEvent {
    /// A request was received and routed.
    RequestReceived(RequestReceived),
    /// A response completed successfully.
    ResponseCompleted(ResponseCompleted),
    /// A response failed before or during upstream dispatch.
    ResponseFailed(ResponseFailed),
    /// One upstream attempt of a request (plan 04 step 1, conflict 8). A
    /// request whose route retries or fails over produces one row per
    /// attempt, linked by `request_id`; failed attempts can still cost
    /// money, and cost lands on the request's completed/failed row.
    AttemptCompleted(AttemptCompleted),
}

/// Event payload for a received request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestReceived {
    /// Proxy-generated unique request identifier.
    pub request_id: String,
    /// When the request was received (RFC 3339).
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    /// Authenticated client key id, or `"anonymous"` under the loopback
    /// no-auth exemption. Client secrets never appear on events.
    pub caller: String,
    /// Provider name from the URL path.
    pub provider: String,
    /// Route kind (`"chat_completions"` or `"messages"`).
    pub route_kind: String,
    /// Client protocol (`"openai_chat"` or `"anthropic"`).
    pub client_protocol: String,
    /// Requested and upstream model.
    pub model: ModelRef,
    /// Whether the request requested streaming.
    pub streaming: bool,
    /// SHA-256 of the raw client request body for correlation and cache keys
    /// (not the body itself, which may contain secrets).
    pub body_hash: String,
    /// Session id attributed to the request (plan 04 step 10, backlog 17):
    /// the inbound session header (for example
    /// `x-claude-code-session-id`) or the request metadata's `user_id`,
    /// when either is present.
    pub session_id: Option<String>,
}

/// Event payload for a completed response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCompleted {
    /// Proxy-generated unique request identifier.
    pub request_id: String,
    /// When the response completed (RFC 3339).
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    /// Provider name.
    pub provider: String,
    /// Upstream message id, when the provider supplied one.
    pub upstream_message_id: Option<String>,
    /// Requested and upstream model.
    pub model: ModelRef,
    /// Token usage reported by the provider.
    pub usage: Usage,
    /// Computed cost, if pricing was configured for the model.
    pub cost: Option<Cost>,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// Request latency in milliseconds.
    pub latency_ms: u64,
    /// Which credential from the provider's pool served the request (its
    /// id, never the secret), when the provider has more than one key
    /// (plan 04 step 5).
    pub credential_id: Option<String>,
    /// Whether the response was served from the response cache (plan 05
    /// step 6): a hit never reserved and is never charged, and is recorded
    /// as a hit with the cached usage kept for the record.
    #[serde(default)]
    pub cache_hit: bool,
}

/// Event payload for a failed response; `model` and `provider` are optional
/// because early failures can occur before either is confirmed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFailed {
    /// Proxy-generated unique request identifier.
    pub request_id: String,
    /// When the failure occurred (RFC 3339).
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    /// Provider name, if known at failure time.
    pub provider: Option<String>,
    /// Requested and upstream model, if decoded before failure.
    pub model: Option<ModelRef>,
    /// Error kind string from the `RouteError` variant name.
    pub error_kind: String,
    /// Sanitized error message; API keys and URLs are always redacted, and
    /// the provider's own response schema never reaches the client — statuses
    /// passed through hold a generic proxy message, 502s a truncated body.
    pub message: String,
    /// HTTP status code returned to the client.
    pub http_status: u16,
    /// Request latency in milliseconds.
    pub latency_ms: u64,
}

/// Event payload for one upstream attempt (plan 04 step 1, conflict 8):
/// what was tried, against which member with which credential, and how it
/// ended. Emitted for every attempt including the one that succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptCompleted {
    /// Proxy-generated unique request identifier this attempt belongs to.
    pub request_id: String,
    /// When the attempt ended (RFC 3339).
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: OffsetDateTime,
    /// Zero-based sequence number of this attempt within the request.
    pub attempt: u32,
    /// The route the client addressed: a provider name or a pool name.
    pub route: String,
    /// The member provider this attempt was sent to.
    pub provider: String,
    /// The upstream model this attempt encoded for.
    pub model: String,
    /// Which credential from the member's pool was used (its id, never the
    /// secret).
    pub credential_id: Option<String>,
    /// `"succeeded"`, `"failed"` (the attempt errored and the ladder moved
    /// on), or `"streaming"` (the attempt's stream was committed to the
    /// client; its outcome lands on the request's terminal row).
    pub outcome: AttemptOutcome,
    /// Error kind string when the attempt failed.
    pub error_kind: Option<String>,
    /// Upstream HTTP status when the attempt failed with one.
    pub http_status: Option<u16>,
    /// Sanitized error message when the attempt failed.
    pub message: Option<String>,
    /// Attempt latency in milliseconds.
    pub latency_ms: u64,
}

/// How one upstream attempt ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptOutcome {
    /// The attempt produced the response the client gets.
    Succeeded,
    /// The attempt failed; the ladder retried or failed over (or gave up).
    Failed,
    /// The attempt's stream was committed to the client before completion;
    /// the terminal outcome is recorded on the request's row.
    Streaming,
}

/// In-process event sink.
pub trait EventBus: Send + Sync + std::fmt::Debug {
    /// Emit one event. Must not block the calling task; backends do IO on a
    /// dedicated writer task.
    fn emit(&self, event: &ProxyEvent);

    /// Signal that no further events will be emitted and flush anything
    /// queued; called once at process shutdown. Implementations must stay
    /// bounded and may block briefly; wrap in `spawn_blocking` from async code.
    fn close(&self) {}
}

/// Default no-op sink.
#[derive(Debug, Clone, Default)]
pub struct NoopBus;

impl EventBus for NoopBus {
    fn emit(&self, _event: &ProxyEvent) {}
}

/// Test-only sink that records events for assertions; available behind the
/// `test-utils` cargo feature.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Default)]
pub struct RecordingBus {
    /// Recorded events.
    pub events: std::sync::Mutex<Vec<ProxyEvent>>,
}

#[cfg(any(test, feature = "test-utils"))]
impl RecordingBus {
    /// Create an empty recording bus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the recorded events (cloned under the lock); recovers from
    /// mutex poison rather than dropping events recorded before the panic.
    pub fn snapshot(&self) -> Vec<ProxyEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl EventBus for RecordingBus {
    fn emit(&self, event: &ProxyEvent) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_proxy_protocol::core::UsageProvenance;

    fn sample_request_received() -> RequestReceived {
        RequestReceived {
            request_id: "req-1".to_owned(),
            timestamp: OffsetDateTime::now_utc(),
            caller: "anonymous".to_owned(),
            provider: "p".to_owned(),
            route_kind: "messages".to_owned(),
            client_protocol: "anthropic".to_owned(),
            model: ModelRef {
                requested: "claude".to_owned(),
                upstream: None,
            },
            streaming: false,
            body_hash: "abc123".to_owned(),
            session_id: None,
        }
    }

    #[test]
    fn noop_bus_emit_does_not_panic_or_record() {
        let bus = NoopBus;
        let event = ProxyEvent::RequestReceived(sample_request_received());
        bus.emit(&event);
    }

    #[test]
    fn recording_bus_captures_events_in_order() {
        let bus = RecordingBus::new();
        let req = ProxyEvent::RequestReceived(sample_request_received());
        let done = ProxyEvent::ResponseCompleted(ResponseCompleted {
            request_id: "req-1".to_owned(),
            timestamp: OffsetDateTime::now_utc(),
            provider: "p".to_owned(),
            upstream_message_id: Some("msg_42".to_owned()),
            model: ModelRef {
                requested: "claude".to_owned(),
                upstream: None,
            },
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            },
            cost: None,
            stop_reason: StopReason::EndTurn,
            latency_ms: 123,
            credential_id: None,
            cache_hit: false,
        });
        bus.emit(&req);
        bus.emit(&done);

        let events = bus.snapshot();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ProxyEvent::RequestReceived(_)));
        assert!(matches!(events[1], ProxyEvent::ResponseCompleted(_)));
    }

    #[test]
    fn proxy_event_serializes_with_kind_tag_and_rfc3339_timestamp() {
        use time::OffsetDateTime;
        use time::format_description::well_known::Rfc3339;

        let timestamp = OffsetDateTime::from_unix_timestamp(1_700_000_000)
            .expect("valid unix timestamp")
            .replace_nanosecond(123_456_789)
            .expect("valid nanoseconds");
        let event = ProxyEvent::RequestReceived(RequestReceived {
            request_id: "req-1".to_owned(),
            timestamp,
            caller: "anonymous".to_owned(),
            provider: "p".to_owned(),
            route_kind: "messages".to_owned(),
            client_protocol: "anthropic".to_owned(),
            model: ModelRef {
                requested: "claude".to_owned(),
                upstream: None,
            },
            streaming: false,
            body_hash: "abc123".to_owned(),
            session_id: None,
        });
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains(r#""kind":"request_received""#), "got: {json}");

        let extracted = extract_json_string_field(&json, "timestamp")
            .expect("timestamp field present and a JSON string");
        assert!(
            extracted.ends_with('Z') || is_numeric_offset_suffix(&extracted),
            "timestamp must end in Z or a numeric offset, got: {extracted}"
        );
        let parsed =
            OffsetDateTime::parse(&extracted, &Rfc3339).expect("timestamp parses as RFC 3339");
        assert_eq!(
            parsed, timestamp,
            "timestamp must round-trip through RFC 3339; got: {extracted}"
        );
        let frac = extracted
            .split('.')
            .nth(1)
            .and_then(|rest| rest.split(['+', '-', 'Z']).next())
            .unwrap_or("");
        assert!(
            !frac.is_empty(),
            "timestamp must carry subsecond precision, got: {extracted}"
        );
        assert_eq!(
            frac.len(),
            9,
            "nanosecond precision expected (9 fractional digits), got: {extracted}"
        );
    }

    fn is_numeric_offset_suffix(s: &str) -> bool {
        if s.len() < 6 {
            return false;
        }
        let tail = &s[s.len() - 6..];
        let sign = tail.as_bytes()[0];
        if sign != b'+' && sign != b'-' {
            return false;
        }
        tail[1..].chars().all(|c| c.is_ascii_digit() || c == ':')
    }

    fn extract_json_string_field(json: &str, field: &str) -> Option<String> {
        let needle = format!("\"{field}\":\"");
        let start = json.find(&needle)? + needle.len();
        let bytes = json.as_bytes();
        let mut end = start;
        let mut i = start;
        let mut escaped = false;
        while i < bytes.len() {
            let c = bytes[i];
            if escaped {
                escaped = false;
                i += 1;
                continue;
            }
            if c == b'\\' {
                escaped = true;
                i += 1;
                continue;
            }
            if c == b'"' {
                end = i;
                break;
            }
            i += 1;
        }
        Some(json[start..end].to_owned())
    }

    #[test]
    fn proxy_event_has_no_secret_field_by_construction() {
        const FORBIDDEN_KEY_FRAGMENTS: &[&str] = &[
            "api_key",
            "apikey",
            "secret",
            "token",
            "bearer",
            "authorization",
            "password",
            "passwd",
        ];

        const SENTINEL: &str = "ZZ-sentinel-content-ZZ";

        let req = ProxyEvent::RequestReceived(RequestReceived {
            request_id: SENTINEL.to_owned(),
            timestamp: OffsetDateTime::now_utc(),
            caller: SENTINEL.to_owned(),
            provider: SENTINEL.to_owned(),
            route_kind: SENTINEL.to_owned(),
            client_protocol: SENTINEL.to_owned(),
            model: ModelRef {
                requested: SENTINEL.to_owned(),
                upstream: Some(SENTINEL.to_owned()),
            },
            streaming: false,
            body_hash: SENTINEL.to_owned(),
            session_id: None,
        });
        let req_json = serde_json::to_string(&req).expect("serialize RequestReceived");
        assert_no_forbidden_secret_key(&req_json, FORBIDDEN_KEY_FRAGMENTS);

        let done = ProxyEvent::ResponseCompleted(ResponseCompleted {
            request_id: SENTINEL.to_owned(),
            timestamp: OffsetDateTime::now_utc(),
            provider: SENTINEL.to_owned(),
            upstream_message_id: Some(SENTINEL.to_owned()),
            model: ModelRef {
                requested: SENTINEL.to_owned(),
                upstream: Some(SENTINEL.to_owned()),
            },
            usage: Usage::default(),
            cost: None,
            stop_reason: StopReason::EndTurn,
            latency_ms: 1,
            credential_id: None,
            cache_hit: false,
        });
        let done_json = serde_json::to_string(&done).expect("serialize ResponseCompleted");
        assert_no_forbidden_secret_key(&done_json, FORBIDDEN_KEY_FRAGMENTS);

        let failed = ProxyEvent::ResponseFailed(ResponseFailed {
            request_id: SENTINEL.to_owned(),
            timestamp: OffsetDateTime::now_utc(),
            provider: Some(SENTINEL.to_owned()),
            model: Some(ModelRef {
                requested: SENTINEL.to_owned(),
                upstream: Some(SENTINEL.to_owned()),
            }),
            error_kind: SENTINEL.to_owned(),
            message: SENTINEL.to_owned(),
            http_status: 429,
            latency_ms: 2,
        });
        let failed_json = serde_json::to_string(&failed).expect("serialize ResponseFailed");
        assert_no_forbidden_secret_key(&failed_json, FORBIDDEN_KEY_FRAGMENTS);
    }

    fn assert_no_forbidden_secret_key(json: &str, fragments: &[&str]) {
        for frag in fragments {
            let key_shape_quoted = format!("\"{frag}\":");
            assert!(
                !json.contains(&key_shape_quoted),
                "forbidden secret-bearing key `{frag}` found in ProxyEvent JSON: {json}"
            );
        }
    }

    #[test]
    fn response_failed_optional_fields_round_trip() {
        let event = ProxyEvent::ResponseFailed(ResponseFailed {
            request_id: "req-2".to_owned(),
            timestamp: OffsetDateTime::now_utc(),
            provider: None,
            model: None,
            error_kind: "UnknownProvider".to_owned(),
            message: "no such provider".to_owned(),
            http_status: 404,
            latency_ms: 1,
        });
        let json = serde_json::to_string(&event).expect("serialize");
        let back: ProxyEvent = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(back, ProxyEvent::ResponseFailed(_)));
        let _ = UsageProvenance::Unknown;
    }
}
