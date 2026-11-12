//! The one client-status policy (audit conflict 14, plan 05 step 1): every
//! HTTP status the proxy can produce, in one table, with the deliberate
//! `x-should-retry` choice for each.
//!
//! Stock SDKs auto-retry a fixed set of statuses (the OpenAI and Anthropic
//! SDKs both retry 408, 409, 429 and 5xx by default, and both honour an
//! `x-should-retry` response header over their built-in list), so a status
//! emitted without a policy silently multiplies load. Every status the
//! engine can name ([`EngineError::status_code`], including the upstream
//! passthrough set of [`crate::map_upstream_status`]) and every status the
//! server can name must be a row here; tests on both sides enforce that.
//!
//! The rules, in one place:
//!
//! - Client-fixable failures (400, 401, 403, 404, 405, 413, 402) say
//!   `x-should-retry: false`: the same request will fail the same way until
//!   the client or operator changes something.
//! - The proxy's own timeout (408) and the in-flight idempotency conflict
//!   (409) say `true`: both clear by themselves within a bounded, short
//!   window.
//! - Rate windows (429) say `true` and carry `Retry-After` with the window
//!   reset when the proxy knows it.
//! - Spend budgets (402) say `false` even though the budget eventually
//!   resets: an SDK retry ladder cannot wait out a day-long window, and
//!   hammering a capped key hides the signal from its owner.
//! - Upstream faults (502, 503, 504) say `true` — unless the proxy's own
//!   ladder already retried them, in which case
//!   [`EngineError::RetryExhausted`] overrides the row and says `false`
//!   (plan 04 step 12).
//! - The self-loop refusal (508) says `false`: the request is looping, and a
//!   retry loops again until the operator fixes the upstream config.
//! - The proxy's own bugs (500) say `false`: a retry re-runs the same bug
//!   and multiplies load without evidence it helps.

use crate::error::EngineError;

/// One policy row: the status, the `x-should-retry` value rendered with it,
/// and the one-line reason the value was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusCodePolicy {
    /// The HTTP status this row governs.
    pub code: u16,
    /// Whether the proxy wants the client to retry the request when it sees
    /// this status (the `x-should-retry` header value).
    pub should_retry: bool,
    /// Why this retry choice was made.
    pub note: &'static str,
}

/// Every status the proxy produces, and its retry policy. Rows are ordered
/// by code.
pub const STATUS_POLICY: &[StatusCodePolicy] = &[
    StatusCodePolicy {
        code: 400,
        should_retry: false,
        note: "client-fixable request error; the same body fails again",
    },
    StatusCodePolicy {
        code: 401,
        should_retry: false,
        note: "a credential must change (client token or operator key)",
    },
    StatusCodePolicy {
        code: 402,
        should_retry: false,
        note: "per-key spend budget exhausted; retrying cannot refill it",
    },
    StatusCodePolicy {
        code: 403,
        should_retry: false,
        note: "this key is not allowed to reach the provider or model",
    },
    StatusCodePolicy {
        code: 404,
        should_retry: false,
        note: "unknown provider, route or model target",
    },
    StatusCodePolicy {
        code: 405,
        should_retry: false,
        note: "the method is not served on this route",
    },
    StatusCodePolicy {
        code: 408,
        should_retry: true,
        note: "the proxy's request timeout; a retry can fit the window",
    },
    StatusCodePolicy {
        code: 409,
        should_retry: true,
        note: "another request holds this Idempotency-Key; it clears when that \
               request finishes",
    },
    StatusCodePolicy {
        code: 413,
        should_retry: false,
        note: "the request body is over the endpoint family limit",
    },
    StatusCodePolicy {
        code: 429,
        should_retry: true,
        note: "a rate window is full; Retry-After carries the reset",
    },
    StatusCodePolicy {
        code: 500,
        should_retry: false,
        note: "proxy bug; a retry re-runs the same bug and multiplies load",
    },
    StatusCodePolicy {
        code: 502,
        should_retry: true,
        note: "upstream fault; a retry may reach healthy capacity",
    },
    StatusCodePolicy {
        code: 503,
        should_retry: true,
        note: "capacity (concurrency cap or affinity); Retry-After is short",
    },
    StatusCodePolicy {
        code: 504,
        should_retry: true,
        note: "upstream timeout; a retry may fit the window",
    },
    StatusCodePolicy {
        code: 508,
        should_retry: false,
        note: "the upstream config loops back to this proxy; a retry loops again \
               until the operator fixes the config",
    },
];

/// The policy row for `code`, or `None` when the proxy never produces it.
#[must_use]
pub fn policy_for(code: u16) -> Option<&'static StatusCodePolicy> {
    STATUS_POLICY.iter().find(|row| row.code == code)
}

/// The `x-should-retry` value this error's response carries. The table row
/// for its status, except that an exhausted retry ladder always says
/// `false`: the proxy already did the retrying (plan 04 step 12).
#[must_use]
pub fn should_retry(error: &EngineError) -> bool {
    if matches!(error, EngineError::RetryExhausted { .. }) {
        return false;
    }
    policy_for(error.status_code()).is_some_and(|row| row.should_retry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AuthOwner;
    use std::time::Duration;

    fn representative_errors() -> Vec<EngineError> {
        let upstream_passthrough = |status: u16, owner: AuthOwner| EngineError::Upstream {
            status,
            body: String::new(),
            auth_owner: owner,
            retry_after: None,
        };
        vec![
            EngineError::InvalidProviderName("x".to_owned()),
            EngineError::UnknownProvider("x".to_owned()),
            EngineError::UnsupportedRoute("x".to_owned()),
            EngineError::ModelNotAllowed("x".to_owned()),
            EngineError::Unauthorized,
            EngineError::InvalidRequest("x".to_owned()),
            EngineError::IdempotencyInFlight,
            EngineError::ProviderDecode("x".to_owned()),
            EngineError::UpstreamTimeout("x".to_owned()),
            EngineError::AffinityRefused {
                pinned: "p".to_owned(),
                session: "s".to_owned(),
            },
            EngineError::FailoverRefused {
                reason: "x".to_owned(),
            },
            EngineError::Internal("x".to_owned()),
            upstream_passthrough(400, AuthOwner::Operator),
            upstream_passthrough(401, AuthOwner::Client),
            upstream_passthrough(403, AuthOwner::Client),
            upstream_passthrough(413, AuthOwner::Operator),
            upstream_passthrough(429, AuthOwner::Operator),
            upstream_passthrough(500, AuthOwner::Operator),
        ]
    }

    #[test]
    fn every_engine_error_status_is_a_policy_row() {
        for error in representative_errors() {
            let code = error.status_code();
            assert!(
                policy_for(code).is_some(),
                "status {code} from {error:?} has no policy row"
            );
        }
    }

    #[test]
    fn upstream_status_mapping_never_leaves_the_table() {
        // Every status an upstream can name, mapped through the one policy,
        // must land on a row — not just the representative errors above.
        for status in 400u16..600 {
            for owner in [AuthOwner::Client, AuthOwner::Operator] {
                let mapped = crate::map_upstream_status(status, owner);
                assert!(
                    policy_for(mapped).is_some(),
                    "upstream {status} ({owner:?}) maps to {mapped}, which has no policy row"
                );
            }
        }
    }

    #[test]
    fn table_rows_are_unique_and_ordered() {
        let codes: Vec<u16> = STATUS_POLICY.iter().map(|row| row.code).collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            codes, sorted,
            "policy rows must be ordered by code with no duplicates"
        );
    }

    #[test]
    fn should_retry_follows_the_table_except_exhausted_ladders() {
        for error in representative_errors() {
            let expected = policy_for(error.status_code())
                .expect("status is in the table")
                .should_retry;
            assert_eq!(
                error.should_retry(),
                expected,
                "engine should_retry disagrees with the table for {error:?}"
            );
        }
        // The ladder already retried: false regardless of the last status.
        for last in representative_errors() {
            let exhausted = EngineError::RetryExhausted {
                attempts: 2,
                last: Box::new(last),
                retry_after: Some(Duration::from_secs(1)),
            };
            assert!(
                !exhausted.should_retry(),
                "an exhausted ladder must say x-should-retry: false, got {exhausted:?}"
            );
        }
    }

    #[test]
    fn codes_stock_sdks_auto_retry_all_have_deliberate_rows() {
        // The SDK default retry sets: 408, 409, 429 and 5xx. Each must be a
        // row (deliberate true or false), never an accidental gap.
        for code in [408, 409, 429, 500, 502, 503, 504] {
            assert!(policy_for(code).is_some(), "SDK-retryable {code} missing");
        }
    }
}
