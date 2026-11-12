//! OTLP metric export of the existing lock-free counters (plan 07 step 7,
//! backlog 21): an opt-in push to an OpenTelemetry Collector over OTLP/HTTP
//! JSON. The counters themselves are unchanged — every instrument below is
//! an observable that reads a fresh [`Metrics::get_snapshot`] at collection
//! time, so nothing on the request path gains a lock.
//!
//! The OTLP endpoint must be `http://`: this exporter's reqwest 0.13 client
//! carries no TLS stack (see [`start_otlp_metrics`]).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use llm_proxy_core::Metrics;
use opentelemetry::metrics::MeterProvider;
use opentelemetry_otlp::WithExportConfig;

/// Starts the export loop and returns a guard that shuts it down (flushing
/// one last export) when dropped — the serve path holds it until the server
/// stops.
///
/// The OTLP endpoint must be an `http://` URL: the exporter's blocking
/// reqwest 0.13 client is built without a TLS stack (enabling rustls would
/// pull `aws-lc-sys`, whose C build the musl release targets cannot do), so
/// an `https://` endpoint would silently fail at connect time. TLS lives in
/// the single transport-crate reqwest 0.12 rustls stack only (BUG-23).
///
/// # Errors
///
/// `anyhow` when the endpoint is `https://` (rejected up front, above) or
/// when the OTLP exporter cannot be built (the endpoint rejected by the HTTP
/// client builder).
pub fn start_otlp_metrics(
    metrics: &Arc<Metrics>,
    endpoint: &str,
    interval: Duration,
) -> Result<OtlpMetricsGuard> {
    if endpoint
        .trim()
        .split_once("://")
        .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("https"))
    {
        return Err(anyhow::anyhow!(
            "the OTLP metric exporter runs without a TLS stack and cannot connect \
             to an https:// endpoint: server.metrics.otlp.endpoint must be an \
             http:// URL, got {endpoint:?}"
        ));
    }
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_endpoint(endpoint)
        .with_protocol(opentelemetry_otlp::Protocol::HttpJson)
        .build()
        .map_err(|error| {
            anyhow::anyhow!("building the OTLP metric exporter for {endpoint}: {error}")
        })
        .context("the exporter needs an http(s) OTLP metrics endpoint")?;
    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
        .with_interval(interval)
        .build();
    let provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_reader(reader)
        .build();

    let meter = provider.meter("llm-proxy");
    let counters = [
        (
            "llm_proxy_requests_total",
            "Requests received (streaming and non-streaming).",
            Counter::RequestsReceived,
        ),
        (
            "llm_proxy_requests_streamed_total",
            "Requests that asked for a stream.",
            Counter::RequestsStreamed,
        ),
        (
            "llm_proxy_requests_success_total",
            "Requests that completed successfully.",
            Counter::RequestsSuccess,
        ),
        (
            "llm_proxy_requests_failed_total",
            "Requests that failed (genuine upstream/decode failures).",
            Counter::RequestsFailed,
        ),
        (
            "llm_proxy_upstream_calls_total",
            "Calls made to upstream providers.",
            Counter::UpstreamCalls,
        ),
        (
            "llm_proxy_rate_limited_total",
            "Requests rejected by the rate limiter.",
            Counter::RateLimited,
        ),
        (
            "llm_proxy_cache_hits_total",
            "Requests served without a fresh upstream attempt.",
            Counter::CacheHits,
        ),
        (
            "llm_proxy_client_cancelled_total",
            "Requests whose client disconnected or was torn down mid-stream.",
            Counter::ClientCancelled,
        ),
    ];
    for (name, description, counter) in counters {
        let metrics = Arc::clone(metrics);
        meter
            .u64_observable_counter(name)
            .with_description(description)
            .with_unit("1")
            .with_callback(move |observer| {
                let snapshot = metrics.get_snapshot();
                let value = match counter {
                    Counter::RequestsReceived => snapshot.requests_received,
                    Counter::RequestsStreamed => snapshot.requests_streamed,
                    Counter::RequestsSuccess => snapshot.requests_success,
                    Counter::RequestsFailed => snapshot.requests_failed,
                    Counter::UpstreamCalls => snapshot.upstream_calls,
                    Counter::RateLimited => snapshot.rate_limited,
                    Counter::CacheHits => snapshot.cache_hits,
                    Counter::ClientCancelled => snapshot.client_cancelled,
                };
                observer.observe(u64::try_from(value).unwrap_or(u64::MAX), &[]);
            })
            .build();
        // The provider retains the instrument; observables need no handle.
    }

    tracing::info!(
        endpoint,
        interval = ?interval,
        "pushing runtime counters to an OTLP collector"
    );
    Ok(OtlpMetricsGuard {
        provider: Some(provider),
    })
}

#[derive(Clone, Copy)]
enum Counter {
    RequestsReceived,
    RequestsStreamed,
    RequestsSuccess,
    RequestsFailed,
    UpstreamCalls,
    RateLimited,
    CacheHits,
    ClientCancelled,
}

/// Keeps the OTLP export loop alive; dropping it flushes and stops the
/// provider.
pub struct OtlpMetricsGuard {
    provider: Option<opentelemetry_sdk::metrics::SdkMeterProvider>,
}

impl std::fmt::Debug for OtlpMetricsGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OtlpMetricsGuard")
            .field("running", &self.provider.is_some())
            .finish()
    }
}

impl Drop for OtlpMetricsGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            if let Err(error) = provider.shutdown() {
                tracing::warn!(%error, "OTLP metric provider shutdown reported errors");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, routing::post};
    use std::time::Instant;

    /// The exporter's blocking reqwest 0.13 client is built without a TLS
    /// stack (enabling rustls would pull aws-lc-sys, which the musl release
    /// targets cannot build), so an `https://` endpoint must be rejected at
    /// startup with a message naming the http:// requirement instead of
    /// failing later at first handshake.
    #[test]
    fn an_https_endpoint_is_rejected_with_a_clear_http_only_message() {
        let metrics = Arc::new(Metrics::new());
        let error = start_otlp_metrics(
            &metrics,
            "https://127.0.0.1:1/v1/metrics",
            Duration::from_millis(100),
        )
        .expect_err("an https:// endpoint cannot work without a TLS stack");
        let message = error.to_string();
        assert!(
            message.contains("http://"),
            "the rejection must name the required http:// scheme: {message}"
        );
        assert!(
            message.to_ascii_lowercase().contains("https"),
            "the rejection must name the offending https scheme: {message}"
        );
    }

    /// The observable end-to-end property: a configured exporter pushes the
    /// existing counters, as OTLP JSON, to the configured endpoint.
    #[tokio::test]
    async fn counters_arrive_at_the_collector_as_otlp_json() {
        let captured = Arc::new(tokio::sync::Mutex::new(Vec::<Vec<u8>>::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let sink = Arc::clone(&captured);
        let app = Router::new().route(
            "/v1/metrics",
            post(move |body: axum::body::Bytes| {
                let sink = Arc::clone(&sink);
                async move {
                    sink.lock().await.push(body.to_vec());
                    "ok"
                }
            }),
        );
        tokio::spawn(async move {
            let serve = axum::serve(listener, app);
            if let Err(error) = serve.await {
                tracing::warn!(%error, "collector stopped");
            }
        });

        let metrics = Arc::new(Metrics::new());
        metrics.record_request(false);
        metrics.record_request(true);
        metrics.record_failure();
        let _guard = start_otlp_metrics(
            &metrics,
            &format!("http://{addr}/v1/metrics"),
            Duration::from_millis(100),
        )
        .expect("exporter starts against the local collector");

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let bodies = captured.lock().await;
            if bodies
                .iter()
                .any(|body| body.windows(8).any(|w| w == b"resource"))
                && bodies
                    .iter()
                    .any(|body| body.windows(9).any(|w| w == b"llm_proxy"))
            {
                let body = String::from_utf8_lossy(bodies.last().unwrap()).into_owned();
                assert!(
                    body.contains("llm_proxy_requests_total"),
                    "the pushed payload names our counter: {}",
                    &body[..body.len().min(600)]
                );
                return;
            }
            drop(bodies);
            assert!(
                Instant::now() < deadline,
                "no OTLP payload with our counters arrived within 10s"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}
