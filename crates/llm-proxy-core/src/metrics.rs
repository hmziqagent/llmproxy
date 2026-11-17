//! Runtime metrics for the LLM proxy.
//!
//! All counters are lock-free ([`std::sync::atomic::AtomicI64`]). The only
//! mutex-guarded state is the latency ring-buffer (last 1 000 samples) and
//! the per-provider-model request counter map.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

const LATENCY_CAP: usize = 1000;

const MODEL_COUNTS_CAP: usize = 1024;

const MODEL_COUNTS_OVERFLOW: &str = "~overflow:";

const KEY_SEPARATOR: &str = ":";

/// Thread-safe metrics collector.
///
/// Internally uses [`AtomicI64`] counters for hot-path increments and
/// [`Mutex`] only for the latency ring-buffer and provider-model map.
#[derive(Debug)]
pub struct Metrics {
    requests_received: AtomicI64,
    requests_streamed: AtomicI64,
    requests_success: AtomicI64,
    requests_failed: AtomicI64,
    upstream_calls: AtomicI64,
    rate_limited: AtomicI64,
    cache_hits: AtomicI64,
    client_cancelled: AtomicI64,
    /// Ring-buffer holding the last [`LATENCY_CAP`] latency samples.
    latencies: Mutex<VecDeque<Duration>>,
    /// Per-provider-model request counts. Key format: `"{provider}:{model}"`.
    model_counts: Mutex<HashMap<String, i64>>,
}

impl Metrics {
    /// Create a new, zeroed [`Metrics`] instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests_received: AtomicI64::new(0),
            requests_streamed: AtomicI64::new(0),
            requests_success: AtomicI64::new(0),
            requests_failed: AtomicI64::new(0),
            upstream_calls: AtomicI64::new(0),
            rate_limited: AtomicI64::new(0),
            cache_hits: AtomicI64::new(0),
            client_cancelled: AtomicI64::new(0),
            latencies: Mutex::new(VecDeque::with_capacity(LATENCY_CAP)),
            model_counts: Mutex::new(HashMap::new()),
        }
    }

    /// Record an incoming request.
    ///
    /// If `streaming` is `true`, both `requests_received` and
    /// `requests_streamed` are incremented.
    pub fn record_request(&self, streaming: bool) {
        self.requests_received.fetch_add(1, Ordering::Relaxed);
        if streaming {
            self.requests_streamed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a successful upstream response.
    ///
    /// Increments `requests_success` and `upstream_calls`, stores the
    /// `latency` sample, and bumps the per-`provider`/`model` counter.
    pub fn record_success(&self, provider: &str, model: &str, latency: Duration) {
        self.requests_success.fetch_add(1, Ordering::Relaxed);
        self.upstream_calls.fetch_add(1, Ordering::Relaxed);

        {
            let mut latencies = self.latencies.lock().unwrap_or_else(|e| e.into_inner());
            if latencies.len() >= LATENCY_CAP {
                latencies.pop_front();
            }
            latencies.push_back(latency);
        }

        thread_local! {
            static KEY_BUF: std::cell::RefCell<String> =
                std::cell::RefCell::new(String::with_capacity(64));
        }
        let mut map = self.model_counts.lock().unwrap_or_else(|e| e.into_inner());
        KEY_BUF.with(|buf| {
            let mut buf = buf.borrow_mut();
            buf.clear();
            buf.push_str(provider);
            buf.push_str(KEY_SEPARATOR);
            buf.push_str(model);
            if let Some(count) = map.get_mut(buf.as_str()) {
                *count += 1;
            } else if map.len() < MODEL_COUNTS_CAP {
                *map.entry(buf.clone()).or_insert(0) += 1;
            } else {
                *map.entry(MODEL_COUNTS_OVERFLOW.to_owned()).or_insert(0) += 1;
            }
        });
    }

    /// Record a failed request.
    ///
    /// Increments `requests_failed` and `upstream_calls`. Reserve this for
    /// genuine upstream/decode failures: record client-initiated disconnects
    /// via [`Metrics::record_client_cancel`] instead, so that "stop" presses
    /// and timeout teardowns do not corrupt the error-rate SLO.
    pub fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
        self.upstream_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a client-initiated cancellation.
    ///
    /// Distinct from [`Metrics::record_failure`]: this does not touch
    /// `requests_failed` or `upstream_calls`.
    pub fn record_client_cancel(&self) {
        self.client_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rate-limited request.
    pub fn record_rate_limited(&self) {
        self.rate_limited.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a request served without a fresh upstream attempt: a response
    /// cache hit, a single-flight join, or an idempotent replay (plan 06).
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Take an approximate point-in-time snapshot of all counters.
    ///
    /// Individual fields are each loaded atomically, but the snapshot is not
    /// transactional: other threads may mutate state between the first and
    /// last load.
    pub fn get_snapshot(&self) -> Snapshot {
        let requests_received = self.requests_received.load(Ordering::Relaxed);
        let requests_streamed = self.requests_streamed.load(Ordering::Relaxed);
        let requests_success = self.requests_success.load(Ordering::Relaxed);
        let requests_failed = self.requests_failed.load(Ordering::Relaxed);
        let upstream_calls = self.upstream_calls.load(Ordering::Relaxed);
        let rate_limited = self.rate_limited.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let client_cancelled = self.client_cancelled.load(Ordering::Relaxed);

        let latencies = self
            .latencies
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .collect();

        let model_counts = self
            .model_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        Snapshot {
            requests_received,
            requests_streamed,
            requests_success,
            requests_failed,
            upstream_calls,
            rate_limited,
            cache_hits,
            client_cancelled,
            latencies,
            model_counts,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// An approximate point-in-time copy of all metric counters.
///
/// Each field is loaded atomically, but the snapshot as a whole is not a
/// transactional view.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    /// Total requests received (streaming + non-streaming).
    pub requests_received: i64,
    /// Subset of received requests that used streaming.
    pub requests_streamed: i64,
    /// Requests that completed successfully.
    pub requests_success: i64,
    /// Requests that failed.
    pub requests_failed: i64,
    /// Total calls made to upstream providers.
    pub upstream_calls: i64,
    /// Requests rejected by the rate limiter.
    pub rate_limited: i64,
    /// Requests served without a fresh upstream attempt: a response cache
    /// hit, a single-flight join, or an idempotent replay.
    pub cache_hits: i64,
    /// Requests whose client disconnected (or were torn down by the timeout
    /// layer) mid-stream; tracked separately from `requests_failed`.
    pub client_cancelled: i64,
    /// Collected latency samples (up to 1 000 entries).
    pub latencies: Vec<Duration>,
    /// Per-provider-model request counts. Key format: `"{provider}:{model}"`.
    pub model_counts: HashMap<String, i64>,
}

impl Snapshot {
    /// Compute the latency at an arbitrary percentile (0..=100).
    ///
    /// Uses the "exclusive" interpolation method common in monitoring systems:
    /// `rank = pct/100 * (n + 1)`, then clamped to `[0, n-1]`.
    ///
    /// Returns [`Duration::ZERO`] for an empty sample set.
    pub fn calculate_percentile(&self, pct: f64) -> Duration {
        percentile(&self.latencies, pct)
    }

    /// Compute the p95 latency from the collected samples.
    ///
    /// Returns [`Duration::ZERO`] when no samples have been recorded.
    pub fn calculate_p95(&self) -> Duration {
        percentile(&self.latencies, 95.0)
    }

    /// Compute the p99 latency from the collected samples.
    ///
    /// Returns [`Duration::ZERO`] when no samples have been recorded.
    pub fn calculate_p99(&self) -> Duration {
        percentile(&self.latencies, 99.0)
    }

    /// Render the snapshot in the Prometheus text exposition format
    /// (format version 0.0.4): every counter as a `_total` counter, the
    /// latency samples as a summary with 0.5/0.95/0.99 quantiles, and the
    /// per-provider-model request counts as a labeled counter family.
    /// Attribute values are escaped per the exposition spec.
    #[must_use]
    pub fn to_prometheus_text(&self) -> String {
        let mut out = String::with_capacity(2048);
        let counter = |name: &str, help: &str, value: i64, out: &mut String| {
            out.push_str("# HELP ");
            out.push_str(name);
            out.push(' ');
            out.push_str(help);
            out.push_str("\n# TYPE ");
            out.push_str(name);
            out.push_str(" counter\n");
            out.push_str(name);
            out.push(' ');
            // i64 formats exactly; no float precision loss for counters.
            out.push_str(&value.to_string());
            out.push('\n');
        };
        counter(
            "llm_proxy_requests_total",
            "Requests received (streaming and non-streaming).",
            self.requests_received,
            &mut out,
        );
        counter(
            "llm_proxy_requests_streamed_total",
            "Requests that asked for a stream.",
            self.requests_streamed,
            &mut out,
        );
        counter(
            "llm_proxy_requests_success_total",
            "Requests that completed successfully.",
            self.requests_success,
            &mut out,
        );
        counter(
            "llm_proxy_requests_failed_total",
            "Requests that failed (genuine upstream/decode failures).",
            self.requests_failed,
            &mut out,
        );
        counter(
            "llm_proxy_upstream_calls_total",
            "Calls made to upstream providers.",
            self.upstream_calls,
            &mut out,
        );
        counter(
            "llm_proxy_rate_limited_total",
            "Requests rejected by the rate limiter.",
            self.rate_limited,
            &mut out,
        );
        counter(
            "llm_proxy_cache_hits_total",
            "Requests served without a fresh upstream attempt (cache hit, single-flight join, idempotent replay).",
            self.cache_hits,
            &mut out,
        );
        counter(
            "llm_proxy_client_cancelled_total",
            "Requests whose client disconnected or was torn down mid-stream.",
            self.client_cancelled,
            &mut out,
        );

        out.push_str("# HELP llm_proxy_request_latency_seconds Request latency.\n");
        out.push_str("# TYPE llm_proxy_request_latency_seconds summary\n");
        let quantiles = [
            ("0.5", self.calculate_percentile(50.0)),
            ("0.95", self.calculate_p95()),
            ("0.99", self.calculate_p99()),
        ];
        for (label, value) in quantiles {
            out.push_str(&format!(
                "llm_proxy_request_latency_seconds{{quantile=\"{label}\"}} {}\n",
                seconds(value)
            ));
        }
        let sum: Duration = self.latencies.iter().sum();
        out.push_str(&format!(
            "llm_proxy_request_latency_seconds_sum {}\n",
            seconds(sum)
        ));
        out.push_str(&format!(
            "llm_proxy_request_latency_seconds_count {}\n",
            i64::try_from(self.latencies.len()).unwrap_or(i64::MAX),
        ));

        out.push_str(
            "# HELP llm_proxy_model_requests_total Requests served per provider and model.\n",
        );
        out.push_str("# TYPE llm_proxy_model_requests_total counter\n");
        for (key, value) in &self.model_counts {
            // Key format is "{provider}:{model}" (KEY_SEPARATOR); split at the
            // first separator so a model id containing ':' stays whole.
            let (provider, model) = key.split_once(KEY_SEPARATOR).unwrap_or((key.as_str(), ""));
            out.push_str(&format!(
                "llm_proxy_model_requests_total{{provider=\"{}\",model=\"{}\"}} {}\n",
                escape_label(provider),
                escape_label(model),
                (*value)
            ));
        }
        out
    }
}

/// A duration as fractional seconds for the exposition format.
fn seconds(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    let secs = (nanos / 1_000_000_000).to_string();
    let frac = (nanos % 1_000_000_000).to_string();
    if frac == "0" {
        format!("{secs}.0")
    } else {
        format!("{secs}.{}", {
            let mut padded = frac;
            while padded.len() < 9 {
                padded.insert(0, '0');
            }
            padded.trim_end_matches('0').to_owned()
        })
    }
}

/// Escapes a label value per the exposition spec: backslash, double quote
/// and line feeds.
fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn percentile(samples: &[Duration], pct: f64) -> Duration {
    let pct = pct.clamp(0.0, 100.0);

    if samples.is_empty() {
        return Duration::ZERO;
    }

    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort();

    let n = sorted.len();
    let rank = (pct / 100.0) * (n as f64 + 1.0);
    #[expect(
        clippy::cast_sign_loss,
        reason = "`pct` is clamped to 0.0..=100.0 and `n >= 1`, so `rank = pct/100 * (n+1)` is always >= 0.0 and the cast cannot lose a sign"
    )]
    let idx = (rank.floor() as usize).saturating_sub(1).min(n - 1);
    sorted
        .get(idx)
        .or_else(|| sorted.last())
        .copied()
        .unwrap_or(Duration::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metrics_all_zero() {
        let m = Metrics::new();
        let snap = m.get_snapshot();
        assert_eq!(snap.requests_received, 0);
        assert_eq!(snap.requests_streamed, 0);
        assert_eq!(snap.requests_success, 0);
        assert_eq!(snap.requests_failed, 0);
        assert_eq!(snap.upstream_calls, 0);
        assert_eq!(snap.rate_limited, 0);
        assert_eq!(snap.cache_hits, 0);
        assert_eq!(snap.client_cancelled, 0);
        assert!(snap.latencies.is_empty());
        assert!(snap.model_counts.is_empty());
    }

    #[test]
    fn record_request_non_streaming() {
        let m = Metrics::new();
        m.record_request(false);
        let snap = m.get_snapshot();
        assert_eq!(snap.requests_received, 1);
        assert_eq!(snap.requests_streamed, 0);
    }

    #[test]
    fn record_request_streaming() {
        let m = Metrics::new();
        m.record_request(true);
        let snap = m.get_snapshot();
        assert_eq!(snap.requests_received, 1);
        assert_eq!(snap.requests_streamed, 1);
    }

    #[test]
    fn record_success_tracks_latency_and_provider_model() {
        let m = Metrics::new();
        m.record_success("openai", "gpt-4o", Duration::from_millis(120));
        m.record_success("openai", "gpt-4o", Duration::from_millis(80));
        m.record_success("anthropic", "claude-3", Duration::from_millis(200));

        let snap = m.get_snapshot();
        assert_eq!(snap.requests_success, 3);
        assert_eq!(snap.upstream_calls, 3);
        assert_eq!(snap.latencies.len(), 3);
        assert_eq!(snap.model_counts.get("openai:gpt-4o"), Some(&2));
        assert_eq!(snap.model_counts.get("anthropic:claude-3"), Some(&1));
    }

    #[test]
    fn record_success_same_model_different_providers_are_distinct() {
        let m = Metrics::new();
        m.record_success("provider-a", "gpt-4o", Duration::from_millis(100));
        m.record_success("provider-b", "gpt-4o", Duration::from_millis(200));

        let snap = m.get_snapshot();
        assert_eq!(snap.model_counts.get("provider-a:gpt-4o"), Some(&1));
        assert_eq!(snap.model_counts.get("provider-b:gpt-4o"), Some(&1));
        assert_eq!(snap.model_counts.len(), 2);
    }

    #[test]
    fn record_failure_increments_counters() {
        let m = Metrics::new();
        m.record_failure();
        m.record_failure();
        let snap = m.get_snapshot();
        assert_eq!(snap.requests_failed, 2);
        assert_eq!(snap.upstream_calls, 2);
    }

    #[test]
    fn record_client_cancel_is_distinct_from_failure() {
        let m = Metrics::new();
        m.record_client_cancel();
        m.record_client_cancel();
        m.record_failure();
        let snap = m.get_snapshot();
        assert_eq!(snap.client_cancelled, 2);
        assert_eq!(snap.requests_failed, 1);
        assert_eq!(
            snap.upstream_calls, 1,
            "client cancellation must NOT inflate upstream_calls / error-rate SLO"
        );
    }

    #[test]
    fn record_rate_limited_and_cache_hit() {
        let m = Metrics::new();
        m.record_rate_limited();
        m.record_rate_limited();
        m.record_rate_limited();
        m.record_cache_hit();

        let snap = m.get_snapshot();
        assert_eq!(snap.rate_limited, 3);
        assert_eq!(snap.cache_hits, 1);
    }

    #[test]
    fn latency_ring_buffer_capped_at_1000() {
        let m = Metrics::new();
        for i in 0..1050 {
            m.record_success("p", "model", Duration::from_millis(i));
        }

        let snap = m.get_snapshot();
        assert_eq!(snap.latencies.len(), LATENCY_CAP);
        assert_eq!(snap.latencies[0], Duration::from_millis(50));
        assert_eq!(snap.latencies[LATENCY_CAP - 1], Duration::from_millis(1049));
    }

    #[test]
    fn p95_p99_empty() {
        let snap = Snapshot::default();
        assert_eq!(snap.calculate_p95(), Duration::ZERO);
        assert_eq!(snap.calculate_p99(), Duration::ZERO);
    }

    #[test]
    fn p95_p99_on_known_samples() {
        let latencies: Vec<Duration> = (0..100).map(Duration::from_millis).collect();

        let snap = Snapshot {
            latencies,
            ..Snapshot::default()
        };

        assert_eq!(snap.calculate_p95(), Duration::from_millis(94));
        assert_eq!(snap.calculate_p99(), Duration::from_millis(98));
    }

    #[test]
    fn snapshot_is_clone_and_debug() {
        let m = Metrics::new();
        m.record_success("p", "test", Duration::from_millis(10));
        let snap = m.get_snapshot();

        let cloned = snap.clone();
        assert_eq!(cloned.requests_success, 1);

        let _debug_str = format!("{snap:?}");
    }

    #[test]
    fn concurrent_access_does_not_panic() {
        use std::sync::Arc;
        use std::thread;

        fn record_one(m: &Metrics, i: u64) {
            m.record_request(i % 2 == 0);
            if i % 3 == 0 {
                m.record_success("p", "model-a", Duration::from_micros(i));
            } else {
                m.record_failure();
            }
            if i % 10 == 0 {
                m.record_rate_limited();
            }
            if i % 7 == 0 {
                m.record_cache_hit();
            }
        }

        let m = Arc::new(Metrics::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let m = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    record_one(&m, i);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let snap = m.get_snapshot();
        assert_eq!(snap.requests_received, 2000);
        assert_eq!(snap.requests_success + snap.requests_failed, 2000);
    }

    #[test]
    fn prometheus_exposition_carries_every_counter_with_types() {
        let m = Metrics::new();
        m.record_request(true);
        m.record_request(false);
        m.record_success("openai", "gpt-4o", Duration::from_millis(120));
        m.record_failure();
        m.record_rate_limited();
        m.record_cache_hit();
        m.record_client_cancel();
        let text = m.get_snapshot().to_prometheus_text();

        for (name, expected) in [
            ("llm_proxy_requests_total", "2"),
            ("llm_proxy_requests_streamed_total", "1"),
            ("llm_proxy_requests_success_total", "1"),
            ("llm_proxy_requests_failed_total", "1"),
            ("llm_proxy_upstream_calls_total", "2"),
            ("llm_proxy_rate_limited_total", "1"),
            ("llm_proxy_cache_hits_total", "1"),
            ("llm_proxy_client_cancelled_total", "1"),
        ] {
            let line = format!("{name} {expected}");
            assert!(
                text.contains(&line),
                "exposition must carry {name} {expected}:\n{text}"
            );
            assert!(
                text.contains(&format!("# TYPE {name} counter")),
                "exposition must type {name} as counter:\n{text}"
            );
            assert!(
                text.contains(&format!("# HELP {name} ")),
                "exposition must document {name}:\n{text}"
            );
        }
    }

    #[test]
    fn prometheus_exposition_renders_latency_quantiles_and_model_labels() {
        let m = Metrics::new();
        for i in 0..100 {
            m.record_success("prov-a", "model:x", Duration::from_millis(i));
        }
        m.record_success("prov-b", "gpt-4o", Duration::from_millis(5));
        let text = m.get_snapshot().to_prometheus_text();

        assert!(
            text.contains("# TYPE llm_proxy_request_latency_seconds summary"),
            "latency must be a summary:\n{text}"
        );
        for quantile in ["0.5", "0.95", "0.99"] {
            assert!(
                text.contains(&format!("quantile=\"{quantile}\"")),
                "quantile {quantile} missing:\n{text}"
            );
        }
        assert!(
            text.contains("llm_proxy_request_latency_seconds_count 101"),
            "the sample count must reflect every recorded latency:\n{text}"
        );
        assert!(
            text.contains(
                "llm_proxy_model_requests_total{provider=\"prov-a\",model=\"model:x\"} 100"
            ),
            "provider/model labels must carry the counts:\n{text}"
        );
        assert!(
            text.contains("llm_proxy_model_requests_total{provider=\"prov-b\",model=\"gpt-4o\"} 1"),
            "provider/model labels must carry the counts:\n{text}"
        );
    }

    #[test]
    fn prometheus_exposition_escapes_label_values() {
        let snapshot = Snapshot {
            model_counts: HashMap::from([("weird\\".to_owned() + "prov:\"model\nline", 1)]),
            ..Snapshot::default()
        };
        let text = snapshot.to_prometheus_text();
        assert!(
            text.contains("provider=\"weird\\\\prov\""),
            "backslash must be escaped: {text}"
        );
        assert!(
            !text.contains("\nline"),
            "a raw newline inside a label value would corrupt the exposition: {text:?}"
        );
    }

    #[test]
    fn default_trait() {
        let m = Metrics::default();
        let snap = m.get_snapshot();
        assert_eq!(snap.requests_received, 0);
    }

    #[test]
    fn snapshot_default_is_zeroed() {
        let snap = Snapshot::default();
        assert_eq!(snap.requests_received, 0);
        assert_eq!(snap.latencies.len(), 0);
        assert!(snap.model_counts.is_empty());
    }

    #[test]
    fn snapshot_partial_eq_works() {
        let a = Snapshot::default();
        let b = Snapshot::default();
        assert_eq!(a, b);
    }

    #[test]
    fn p95_single_sample() {
        let snap = Snapshot {
            latencies: vec![Duration::from_millis(42)],
            ..Snapshot::default()
        };
        assert_eq!(snap.calculate_p95(), Duration::from_millis(42));
        assert_eq!(snap.calculate_p99(), Duration::from_millis(42));
    }

    #[test]
    fn p95_two_samples() {
        let snap = Snapshot {
            latencies: vec![Duration::from_millis(10), Duration::from_millis(100)],
            ..Snapshot::default()
        };
        assert_eq!(snap.calculate_p95(), Duration::from_millis(100));
        assert_eq!(snap.calculate_p99(), Duration::from_millis(100));
    }

    #[test]
    fn calculate_percentile_public_api() {
        let snap = Snapshot {
            latencies: (0..100).map(Duration::from_millis).collect(),
            ..Snapshot::default()
        };
        assert_eq!(snap.calculate_percentile(50.0), Duration::from_millis(49));
    }

    #[test]
    fn calculate_percentile_clamps_out_of_range_input() {
        let snap = Snapshot {
            latencies: (0..100).map(Duration::from_millis).collect(),
            ..Snapshot::default()
        };
        assert_eq!(snap.calculate_percentile(150.0), Duration::from_millis(99));
        assert_eq!(snap.calculate_percentile(100.0), Duration::from_millis(99));
        assert_eq!(snap.calculate_percentile(-42.0), Duration::from_millis(0));
        assert_eq!(snap.calculate_percentile(0.0), Duration::from_millis(0));
    }

    #[test]
    fn get_snapshot_concurrent_with_writes() {
        use std::sync::Arc;
        use std::thread;

        let m = Arc::new(Metrics::new());
        let mut handles = vec![];

        for _ in 0..2 {
            let m = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for i in 0..200 {
                    m.record_success("p", "model", Duration::from_micros(i));
                }
            }));
        }

        let m_reader = Arc::clone(&m);
        let reader = thread::spawn(move || {
            for _ in 0..100 {
                let snap = m_reader.get_snapshot();
                assert!(snap.requests_success >= 0);
                assert!(snap.latencies.len() <= LATENCY_CAP);
            }
        });

        for h in handles {
            h.join().unwrap();
        }
        reader.join().unwrap();
    }

    #[test]
    fn key_separator_does_not_collide_with_provider_names() {
        let m = Metrics::new();
        m.record_success("provider-a", "model-x", Duration::from_millis(10));
        m.record_success("provider-a", "model-y", Duration::from_millis(20));
        let snap = m.get_snapshot();
        assert_eq!(snap.model_counts.len(), 2);
        assert!(snap.model_counts.contains_key("provider-a:model-x"));
        assert!(snap.model_counts.contains_key("provider-a:model-y"));
    }

    #[test]
    fn model_counts_capped_with_overflow_bucket() {
        let m = Metrics::new();

        for i in 0..MODEL_COUNTS_CAP {
            m.record_success("p", &format!("model-{i}"), Duration::from_micros(i as u64));
        }
        let snap = m.get_snapshot();
        assert_eq!(
            snap.model_counts.len(),
            MODEL_COUNTS_CAP,
            "map should be exactly at capacity before overflow"
        );
        assert!(
            !snap.model_counts.contains_key(MODEL_COUNTS_OVERFLOW),
            "overflow bucket must not exist before capacity is reached"
        );

        m.record_success("p", "overflow-1", Duration::from_micros(0));
        m.record_success("p", "overflow-2", Duration::from_micros(0));
        let snap = m.get_snapshot();
        assert_eq!(
            snap.model_counts.len(),
            MODEL_COUNTS_CAP + 1,
            "only the overflow bucket may be added once capacity is reached"
        );
        assert_eq!(
            snap.model_counts.get(MODEL_COUNTS_OVERFLOW),
            Some(&2),
            "spilled counts should accumulate in the overflow bucket"
        );

        m.record_success("p", "overflow-3", Duration::from_micros(0));
        let snap = m.get_snapshot();
        assert_eq!(
            snap.model_counts.len(),
            MODEL_COUNTS_CAP + 1,
            "subsequent overflow must not grow the map"
        );
        assert_eq!(snap.model_counts.get(MODEL_COUNTS_OVERFLOW), Some(&3));

        m.record_success("p", "model-0", Duration::from_micros(0));
        let snap = m.get_snapshot();
        assert_eq!(
            snap.model_counts.get("p:model-0"),
            Some(&2),
            "existing key must still be incremented after overflow began"
        );
        assert_eq!(
            snap.model_counts.get(MODEL_COUNTS_OVERFLOW),
            Some(&3),
            "overflow bucket must not change for an existing key"
        );
    }
}
