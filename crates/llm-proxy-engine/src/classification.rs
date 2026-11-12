//! Error classification for the retry ladder (plan 04 step 2, backlog 1):
//! the one place that sorts transient from terminal upstream errors and
//! honours `Retry-After`. Retries wrap the send itself — 429s, 5xx
//! responses and connect errors all return before any stream exists — and
//! the same classification covers failures inside the streaming pre-commit
//! window, where nothing has been sent to the client yet.

use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;

use crate::error::EngineError;

/// What the ladder should do with an attempt failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorClass {
    /// The failure may clear on another attempt or member: rate limits,
    /// upstream 5xx, timeouts and connect errors.
    Transient {
        /// A server-advised wait, when the upstream sent one (`Retry-After`
        /// or a rate-limit reset header).
        retry_after: Option<Duration>,
    },
    /// The failure is deterministic for this request: a 4xx the client (or
    /// the operator's config) must fix. Retrying or failing over would
    /// produce the same answer or leak a different upstream's behaviour.
    Terminal,
}

/// Upstream statuses that justify another attempt. Everything else in the
/// 4xx range is the request's or the credential's fault; 5xx and 408/425/429
/// are the upstream's.
fn status_is_transient(status: u16) -> bool {
    matches!(status, 408 | 425 | 429 | 500..=599)
}

/// Classifies one attempt failure.
///
/// - [`EngineError::Upstream`] with a transient status is `Transient`,
///   carrying the upstream-advised `Retry-After` when it parsed.
/// - [`EngineError::UpstreamTimeout`] is `Transient`: a per-attempt
///   time-to-first-byte expiry may clear on another member.
/// - Decode errors are `Transient` for the ladder (an upstream that answers
///   200 with garbage is unhealthy), but surface unchanged to the client
///   when the ladder is exhausted or absent.
/// - Everything the client or operator controls — bad request, unknown
///   provider, auth, model policy, admission — is `Terminal`.
#[must_use]
pub fn classify(error: &EngineError) -> ErrorClass {
    match error {
        EngineError::Upstream {
            status,
            retry_after,
            ..
        } => {
            if status_is_transient(*status) {
                ErrorClass::Transient {
                    retry_after: *retry_after,
                }
            } else {
                ErrorClass::Terminal
            }
        }
        EngineError::UpstreamTimeout(_) => ErrorClass::Transient { retry_after: None },
        EngineError::ProviderDecode(_) => ErrorClass::Transient { retry_after: None },
        EngineError::Internal(_)
        | EngineError::InvalidProviderName(_)
        | EngineError::UnknownProvider(_)
        | EngineError::UnsupportedRoute(_)
        | EngineError::ModelNotAllowed(_)
        | EngineError::Unauthorized
        | EngineError::RateLimited { .. }
        | EngineError::BudgetExhausted { .. }
        | EngineError::Overloaded { .. }
        | EngineError::IdempotencyInFlight
        | EngineError::Forbidden(_)
        | EngineError::InvalidRequest(_) => ErrorClass::Terminal,
        EngineError::RetryExhausted { .. }
        | EngineError::AffinityRefused { .. }
        | EngineError::FailoverRefused { .. } => ErrorClass::Terminal,
    }
}

/// Ceiling on any upstream-advertised wait, so a hostile or broken upstream
/// cannot park the ladder for days. One hour also covers real token-window
/// resets.
pub const MAX_ADVERTISED_RETRY_AFTER_SECS: u64 = 3600;

/// Parses one `Retry-After` value: delay-seconds (an integer, per RFC 9110)
/// or an HTTP-date (the absolute reset moment, converted to a remaining
/// delay). Returns `None` for anything else rather than guessing.
#[must_use]
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(
            seconds.min(MAX_ADVERTISED_RETRY_AFTER_SECS),
        ));
    }
    // HTTP dates ("Sun, 06 Nov 1994 08:49:37 GMT") are RFC 2822 dates with a
    // fixed zone; time's Rfc2822 parses them.
    let parsed =
        time::OffsetDateTime::parse(value, &time::format_description::well_known::Rfc2822).ok()?;
    let now = time::OffsetDateTime::now_utc();
    if parsed <= now {
        return Some(Duration::ZERO);
    }
    let delta = parsed - now;
    let delta = delta.unsigned_abs();
    let secs = delta.as_secs().min(MAX_ADVERTISED_RETRY_AFTER_SECS);
    let wait = Duration::new(secs, delta.subsec_nanos());
    clamp_advised(wait)
}

fn clamp_advised(wait: Duration) -> Option<Duration> {
    Some(wait.min(Duration::from_secs(MAX_ADVERTISED_RETRY_AFTER_SECS)))
}

/// Extracts the best reset signal from an error response's headers
/// (lowercased names), covering the three rate-limit dialects the audit
/// found in the wild:
///
/// - plain `retry-after` (seconds or HTTP-date);
/// - Anthropic-style duration suffixes (`x-ratelimit-*-reset`,
///   `anthropic-ratelimit-*-reset`: `21s`, `1m20s`);
/// - OpenAI/OpenRouter-style resets (`x-ratelimit-reset-requests`,
///   `x-ratelimit-reset`, `x-ratelimit-reset-tokens`: `1ms`, `6m0s`,
///   `0.403s`).
///
/// The longest advertised reset wins: the tokens window usually renews later
/// than the requests window, and honouring the shorter one would send the
/// next attempt into the same wall.
#[must_use]
pub fn rate_limit_reset(headers: &[(String, String)]) -> Option<Duration> {
    let mut best: Option<Duration> = None;
    for (name, value) in headers {
        let reset = if name.eq_ignore_ascii_case("retry-after") {
            parse_retry_after(value)
        } else {
            parse_duration_suffix(value)
        };
        if let Some(reset) = reset
            && reset > best.unwrap_or(Duration::ZERO)
        {
            best = Some(reset);
        }
    }
    best
}

#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time-constant regex literal; a failure is a programming \
              error caught the moment this static is first used, not a runtime input"
)]
static DURATION_PAIR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)").expect("valid duration regex")
});

/// Parses one Go-style `time.Duration` unit (`21s`, `1m`, `6m0s` handled by
/// callers as separate pairs, `2ms`, `0.403s`, `1h`) — the shape both
/// Anthropic and OpenAI emit in their rate-limit reset headers. Zero parses
/// to `Some(ZERO)` (a window that already passed); a bare number without a
/// unit or any other shape yields `None`.
#[must_use]
pub fn parse_duration_suffix(value: &str) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    // Compound values ("1m20s"): consume number/unit pairs left to right.
    let mut total_ns: f64 = 0.0;
    let mut rest = value;
    let mut progressed = false;
    while !rest.is_empty() {
        let captures = DURATION_PAIR.captures(rest)?;
        let number_raw = captures.get(1)?.as_str();
        let unit = captures.get(2)?.as_str();
        let whole_end = captures.get(0)?.end();
        let number: f64 = number_raw.parse().ok()?;
        let mult = match unit {
            "ns" => 1.0,
            "us" | "µs" => 1e3,
            "ms" => 1e6,
            "s" => 1e9,
            "m" => 6e10,
            "h" => 3.6e12,
            _ => return None,
        };
        total_ns += number * mult;
        progressed = true;
        rest = rest.get(whole_end..)?;
    }
    if !progressed {
        return None;
    }
    let secs = total_ns / 1e9;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(secs)
        .ok()
        .and_then(clamp_advised)
}

/// Computes the wait to apply before the next attempt: the upstream-advised
/// wait when present, else the configured backoff, clamped to the ladder's
/// remaining time.
#[must_use]
pub fn next_wait(
    advised: Option<Duration>,
    backoff: Duration,
    remaining: Option<Duration>,
) -> Duration {
    let base = advised.filter(|wait| !wait.is_zero()).unwrap_or(backoff);
    match remaining {
        Some(remaining) => base.min(remaining),
        None => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AuthOwner;

    fn upstream(status: u16) -> EngineError {
        EngineError::Upstream {
            status,
            body: String::new(),
            auth_owner: AuthOwner::Operator,
            retry_after: None,
        }
    }

    #[test]
    fn rate_limits_timeouts_and_5xx_are_transient() {
        let cases: &[EngineError] = &[
            upstream(429),
            upstream(500),
            upstream(503),
            upstream(529),
            EngineError::UpstreamTimeout("first byte".to_owned()),
            EngineError::ProviderDecode("garbage frame".to_owned()),
        ];
        for error in cases {
            assert!(
                matches!(classify(error), ErrorClass::Transient { .. }),
                "{error:?} must be transient"
            );
        }
    }

    #[test]
    fn transient_classification_carries_the_advertised_wait() {
        let error = EngineError::Upstream {
            status: 429,
            body: String::new(),
            auth_owner: AuthOwner::Operator,
            retry_after: Some(Duration::from_secs(17)),
        };
        assert_eq!(
            classify(&error),
            ErrorClass::Transient {
                retry_after: Some(Duration::from_secs(17))
            }
        );
    }

    #[test]
    fn client_faults_and_auth_are_terminal() {
        let cases: &[EngineError] = &[
            upstream(400),
            upstream(401),
            upstream(404),
            upstream(413),
            upstream(422),
            EngineError::InvalidRequest("nope".to_owned()),
            EngineError::ModelNotAllowed("m".to_owned()),
        ];
        for error in cases {
            assert!(
                matches!(classify(error), ErrorClass::Terminal),
                "{error:?} must be terminal"
            );
        }
    }

    #[test]
    fn retry_after_seconds_and_dates_parse() {
        assert_eq!(parse_retry_after("30"), Some(Duration::from_secs(30)));
        assert_eq!(parse_retry_after(" 7 "), Some(Duration::from_secs(7)));
        // An HTTP date in the past means "now".
        assert_eq!(
            parse_retry_after("Mon, 01 Jan 1990 00:00:00 GMT"),
            Some(Duration::ZERO)
        );
        // A date an hour ahead parses to roughly an hour.
        let future = time::OffsetDateTime::now_utc() + time::Duration::hours(1);
        let formatted = future
            .format(&time::format_description::well_known::Rfc2822)
            .unwrap_or_default();
        let parsed = parse_retry_after(&formatted).unwrap_or(Duration::ZERO);
        assert!(
            parsed >= Duration::from_secs(59 * 60) && parsed <= Duration::from_secs(61 * 60),
            "one hour ahead parsed to {parsed:?}"
        );
        assert_eq!(parse_retry_after("soon"), None);
        // Absurd advertised waits are clamped.
        assert_eq!(
            parse_retry_after("999999"),
            Some(Duration::from_secs(MAX_ADVERTISED_RETRY_AFTER_SECS))
        );
    }

    #[test]
    fn duration_suffixes_parse_across_dialects() {
        assert_eq!(parse_duration_suffix("21s"), Some(Duration::from_secs(21)));
        assert_eq!(
            parse_duration_suffix("1m20s"),
            Some(Duration::from_secs(80))
        );
        assert_eq!(
            parse_duration_suffix("6m0s"),
            Some(Duration::from_secs(360))
        );
        assert_eq!(parse_duration_suffix("2ms"), Some(Duration::from_millis(2)));
        assert_eq!(parse_duration_suffix("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(
            parse_duration_suffix("1ms2µs"),
            Some(Duration::from_micros(1002)),
            "a compound value continues past the multi-byte µs unit"
        );
        let frac = parse_duration_suffix("0.403s").unwrap_or(Duration::ZERO);
        assert!(
            (frac.as_millis() as i64 - 403).abs() <= 1,
            "0.403s parsed to {frac:?}"
        );
        assert_eq!(parse_duration_suffix(""), None);
        assert_eq!(parse_duration_suffix("soon"), None);
        assert_eq!(
            parse_duration_suffix("10"),
            None,
            "bare numbers are ambiguous"
        );
    }

    #[test]
    fn rate_limit_reset_takes_the_longest_advertised_window() {
        let headers = vec![
            ("retry-after".to_owned(), "2".to_owned()),
            ("x-ratelimit-reset-requests".to_owned(), "6m0s".to_owned()),
            ("x-ratelimit-requests-reset".to_owned(), "1s".to_owned()),
            ("content-type".to_owned(), "application/json".to_owned()),
        ];
        assert_eq!(rate_limit_reset(&headers), Some(Duration::from_secs(360)));
    }

    #[test]
    fn next_wait_prefers_advice_and_honours_the_deadline() {
        assert_eq!(
            next_wait(
                Some(Duration::from_secs(9)),
                Duration::from_millis(100),
                None
            ),
            Duration::from_secs(9)
        );
        assert_eq!(
            next_wait(None, Duration::from_millis(100), None),
            Duration::from_millis(100)
        );
        assert_eq!(
            next_wait(
                Some(Duration::from_secs(9)),
                Duration::from_millis(100),
                Some(Duration::from_secs(2))
            ),
            Duration::from_secs(2),
            "the ladder never waits past its own deadline"
        );
    }
}
