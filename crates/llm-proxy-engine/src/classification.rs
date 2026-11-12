//! Error classification for the retry ladder (plan 04 step 2, backlog 1):
//! the one place that sorts transient from terminal upstream errors and
//! honours `Retry-After`. Retries wrap the send itself — 429s, 5xx
//! responses and connect errors all return before any stream exists — and
//! the same classification covers failures inside the streaming pre-commit
//! window, where nothing has been sent to the client yet.

use std::time::Duration;

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
/// (lowercased names). A header only counts when its name is `retry-after`,
/// or contains both `ratelimit` and `reset` — the names the rate-limit
/// dialects in the wild use:
///
/// - plain `retry-after` (seconds or HTTP-date);
/// - Anthropic resets as RFC 3339 timestamps (`anthropic-ratelimit-*-reset`:
///   `2026-01-01T00:01:30Z`) or Go-style durations (`x-ratelimit-*-reset`:
///   `21s`, `1m20s`);
/// - OpenAI/OpenRouter-style resets (`x-ratelimit-reset-requests`,
///   `x-ratelimit-reset`, `x-ratelimit-reset-tokens`: `1ms`, `6m0s`,
///   `0.403s`).
///
/// Any other header is ignored, so an unrelated value like `x-trace-id: 5s`
/// cannot extend a cooldown. The longest advertised reset wins: the tokens
/// window usually renews later than the requests window, and honouring the
/// shorter one would send the next attempt into the same wall.
#[must_use]
pub fn rate_limit_reset(headers: &[(String, String)]) -> Option<Duration> {
    rate_limit_reset_at(headers, time::OffsetDateTime::now_utc())
}

/// [`rate_limit_reset`] with the clock injected: a pure function for callers
/// that already hold the response time.
#[must_use]
pub fn rate_limit_reset_at(
    headers: &[(String, String)],
    now: time::OffsetDateTime,
) -> Option<Duration> {
    let mut best: Option<Duration> = None;
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        let reset = if name == "retry-after" {
            parse_retry_after(value)
        } else if name.contains("ratelimit") && name.contains("reset") {
            parse_duration_suffix(value).or_else(|| rfc3339_reset_after(value, now))
        } else {
            None
        };
        let wins_over_current = match best {
            Some(current) => reset.is_some_and(|reset| reset > current),
            None => reset.is_some(),
        };
        if wins_over_current {
            best = reset;
        }
    }
    best
}

/// Converts one RFC 3339 reset timestamp into the remaining wait from `now`.
/// A timestamp that already passed is `Some(ZERO)` (the window is open);
/// anything unparseable is `None` rather than a guess.
fn rfc3339_reset_after(value: &str, now: time::OffsetDateTime) -> Option<Duration> {
    let reset =
        time::OffsetDateTime::parse(value.trim(), &time::format_description::well_known::Rfc3339)
            .ok()?;
    if reset <= now {
        return Some(Duration::ZERO);
    }
    clamp_advised((reset - now).unsigned_abs())
}

/// One Go `time.Duration` unit in nanoseconds. Longest first, so a two-byte
/// unit (`ms`) wins over its one-byte prefix (`m`) and `5ms` is not read as
/// five minutes plus an orphan `s`.
const DURATION_UNITS: [(&str, f64); 7] = [
    ("ns", 1.0),
    ("us", 1e3),
    ("µs", 1e3),
    ("ms", 1e6),
    ("s", 1e9),
    ("m", 6e10),
    ("h", 3.6e12),
];

/// Splits `s` at the first non-digit character into `(digit run, remainder)`.
fn split_digits(s: &str) -> (&str, &str) {
    match s.find(|c: char| !c.is_ascii_digit()) {
        Some(end) => (s.get(..end).unwrap_or(""), s.get(end..).unwrap_or("")),
        None => (s, ""),
    }
}

/// Consumes one `(number)(unit)` pair from the start of `rest`: digits with
/// an optional fractional part (`0.403`), then one of [`DURATION_UNITS`] —
/// the multi-byte `µs` included, matched on `str` prefixes.
fn parse_duration_pair(rest: &str) -> Option<(f64, f64, &str)> {
    let (whole, rest) = split_digits(rest);
    if whole.is_empty() {
        return None;
    }
    let mut number: f64 = whole.parse().ok()?;
    let rest = match rest.strip_prefix('.') {
        Some(after_dot) => {
            let (fraction, rest) = split_digits(after_dot);
            if fraction.is_empty() {
                return None;
            }
            let places = i32::try_from(fraction.len()).ok()?;
            number += fraction.parse::<f64>().ok()? / 10f64.powi(places);
            rest
        }
        None => rest,
    };
    for (unit, multiplier) in DURATION_UNITS {
        if let Some(remainder) = rest.strip_prefix(unit) {
            return Some((number, multiplier, remainder));
        }
    }
    None
}

/// Parses one Go-style `time.Duration` value (`21s`, `1m20s`, `6m0s`, `2ms`,
/// `0.403s`, `1h`) — the shape both Anthropic and OpenAI emit in their
/// rate-limit reset headers. Compound values are consumed left to right as
/// number/unit pairs. Zero parses to `Some(ZERO)` (a window that already
/// passed); a bare number without a unit or any other shape yields `None`.
#[must_use]
pub fn parse_duration_suffix(value: &str) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let mut total_ns: f64 = 0.0;
    let mut rest = value;
    while !rest.is_empty() {
        let (number, multiplier, remainder) = parse_duration_pair(rest)?;
        total_ns += number * multiplier;
        rest = remainder;
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

    fn test_now() -> time::OffsetDateTime {
        time::OffsetDateTime::from_unix_timestamp(1_760_000_000).unwrap()
    }

    fn rfc3339_reset_in(now: time::OffsetDateTime, seconds: i64) -> String {
        (now + time::Duration::seconds(seconds))
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_default()
    }

    #[test]
    fn an_rfc3339_reset_header_becomes_the_remaining_duration() {
        let now = test_now();
        let headers = vec![(
            "anthropic-ratelimit-requests-reset".to_owned(),
            rfc3339_reset_in(now, 90),
        )];
        let reset = rate_limit_reset_at(&headers, now).unwrap_or_default();
        assert!(
            reset >= Duration::from_secs(89) && reset <= Duration::from_secs(91),
            "a reset 90s ahead parsed to {reset:?}"
        );
    }

    #[test]
    fn a_past_rfc3339_reset_is_zero() {
        let now = test_now();
        let headers = vec![(
            "anthropic-ratelimit-tokens-reset".to_owned(),
            rfc3339_reset_in(now, -60),
        )];
        assert_eq!(rate_limit_reset_at(&headers, now), Some(Duration::ZERO));
    }

    #[test]
    fn unrelated_headers_never_extend_a_cooldown() {
        let headers = vec![
            ("x-trace-id".to_owned(), "5s".to_owned()),
            ("x-ratelimit-remaining".to_owned(), "5s".to_owned()),
            ("content-type".to_owned(), "application/json".to_owned()),
        ];
        assert_eq!(rate_limit_reset_at(&headers, test_now()), None);
    }

    #[test]
    fn the_longest_ratelimit_reset_still_wins() {
        let now = test_now();
        let headers = vec![
            (
                "anthropic-ratelimit-requests-reset".to_owned(),
                rfc3339_reset_in(now, 120),
            ),
            ("x-ratelimit-reset-tokens".to_owned(), "1m20s".to_owned()),
        ];
        assert_eq!(
            rate_limit_reset_at(&headers, now),
            Some(Duration::from_secs(120))
        );
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
