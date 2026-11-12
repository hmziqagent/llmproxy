//! The Node binding for llm-proxy (plan 10, step 3). A thin shim over
//! `llm_proxy_bridge`: structured results and errors cross as plain objects
//! and the JS layer (`index.js`) turns them into thrown `ProxyError`s and an
//! async-iterable stream, so no logic lives here.
//!
//! Streaming works through `next()` promises; `cancel()` stops delivery and
//! settles the request as `ClientCancelled`. `break` and `throw` inside a
//! `for await` loop cancel through the iterator's `return()`, and a stream
//! abandoned without a cancel is cancelled when the JS object is garbage
//! collected — for prompt cancellation, call `await stream.cancel()`.

use std::collections::HashMap;
use std::sync::Arc;

use llm_proxy_bridge::{Bridge, BridgeError, EventStream as BridgeStream, ExecuteOptions};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value;

/// The structured failure payload the JS layer unpacks into a `ProxyError`.
#[napi(object)]
pub struct ErrorInfo {
    /// The ledger's `error_kind` string.
    pub kind: String,
    /// The HTTP status the server would map this to.
    pub status: u16,
    /// The sanitized client-facing message.
    pub message: String,
    /// The advised wait in seconds, when the failure carries one.
    pub retry_after_seconds: Option<f64>,
    /// Whether the proxy wants the client to retry itself.
    pub should_retry: bool,
}

/// Converts a bridge error into the structured JS payload.
fn error_info(error: &BridgeError) -> ErrorInfo {
    ErrorInfo {
        kind: error.kind.clone(),
        status: error.status,
        message: error.message.clone(),
        retry_after_seconds: error.retry_after.map(|d| d.as_secs_f64()),
        should_retry: error.should_retry,
    }
}

/// The JSON an error crosses the napi boundary as: the [`ErrorInfo`] fields
/// under the camelCase key names the JS layer unpacks (`ProxyError` reads
/// `retryAfterSeconds` / `shouldRetry`), so a thrown error and a structured
/// result carry the same policy fields. The single source of those key
/// names.
fn error_json(error: &BridgeError) -> String {
    let info = error_info(error);
    serde_json::json!({
        "kind": info.kind,
        "status": info.status,
        "message": info.message,
        "retryAfterSeconds": info.retry_after_seconds,
        "shouldRetry": info.should_retry,
    })
    .to_string()
}

/// Converts the per-call options object into bridge options.
fn options_from(obj: Option<OptionsObj>) -> ExecuteOptions {
    let Some(obj) = obj else {
        return ExecuteOptions::default();
    };
    let mut options = ExecuteOptions::default();
    if let Some(model) = obj.model {
        options.model = Some(model);
    }
    if let Some(headers) = obj.headers {
        options.headers = headers
            .into_iter()
            .map(|(name, value)| (name.to_ascii_lowercase(), value))
            .collect();
    }
    options
}

/// Per-call options: the URL-path model (Gemini) and inbound headers.
#[napi(object)]
pub struct OptionsObj {
    /// The model named in the URL path (Gemini's `generateContent`).
    pub model: Option<String>,
    /// Inbound headers, subject to the provider's forward allowlist.
    pub headers: Option<HashMap<String, String>>,
}

/// The result of one request: exactly one of `response` / `error`.
#[napi(object)]
pub struct ExecuteResult {
    /// The response in the caller's dialect, on success.
    pub response: Option<Value>,
    /// The structured failure, on failure.
    pub error: Option<ErrorInfo>,
}

/// The result of one translation: exactly one of `result` / `error`.
#[napi(object)]
pub struct TranslateResult {
    /// The translated JSON, on success.
    pub result: Option<Value>,
    /// The structured failure, on failure.
    pub error: Option<ErrorInfo>,
}

/// The engine wrapper: built from a configuration object, then executes or
/// streams requests.
#[napi]
pub struct Proxy {
    /// The shared bridge this proxy drives.
    bridge: Arc<Bridge>,
}

#[napi]
impl Proxy {
    #[napi(constructor)]
    pub fn new(config: Value) -> Result<Self> {
        let bridge = Bridge::from_json(&config).map_err(|error| {
            Error::new(
                Status::GenericFailure,
                format!("{}: {}", error.kind, error.message),
            )
        })?;
        Ok(Self {
            bridge: Arc::new(bridge),
        })
    }

    /// Run one non-streaming request end to end.
    #[napi]
    pub async fn execute(
        &self,
        provider: String,
        dialect: String,
        body: Value,
        options: Option<OptionsObj>,
    ) -> Result<ExecuteResult> {
        let options = options_from(options);
        match self
            .bridge
            .execute(&provider, &dialect, &body, &options)
            .await
        {
            Ok(response) => Ok(ExecuteResult {
                response: Some(response),
                error: None,
            }),
            Err(error) => Ok(ExecuteResult {
                response: None,
                error: Some(error_info(&error)),
            }),
        }
    }

    /// Run one streaming request; the returned stream yields event objects.
    /// `cancel()` stops the upstream request and settles it as
    /// `ClientCancelled`; `break` and `throw` inside a `for await` loop do
    /// the same through the iterator's `return()`, and an abandoned stream
    /// is cancelled when the JS object is garbage collected.
    ///
    /// A failure before the first event commits throws a napi error whose
    /// message is the JSON-encoded [`ErrorInfo`] under the JS-facing
    /// (camelCase) key names — napi 3.4 cannot carry a class inside a plain
    /// object, and the JS layer parses it back into a `ProxyError`.
    #[napi]
    pub async fn execute_stream(
        &self,
        provider: String,
        dialect: String,
        body: Value,
        options: Option<OptionsObj>,
    ) -> Result<Stream> {
        let options = options_from(options);
        let stream = self
            .bridge
            .execute_stream(&provider, &dialect, &body, &options)
            .await
            .map_err(|error| Error::new(Status::GenericFailure, error_json(&error)))?;
        Ok(Stream { inner: stream })
    }

    /// Translate a request between wire dialects, no I/O.
    #[napi]
    pub fn translate_request(
        &self,
        body: Value,
        source: String,
        target: String,
        model: Option<String>,
    ) -> TranslateResult {
        match self
            .bridge
            .translate_request(&body, &source, &target, model.as_deref())
        {
            Ok(result) => TranslateResult {
                result: Some(result),
                error: None,
            },
            Err(error) => TranslateResult {
                result: None,
                error: Some(error_info(&error)),
            },
        }
    }

    /// Translate a provider response into a client dialect, no I/O.
    #[napi]
    pub fn translate_response(
        &self,
        body: Value,
        source: String,
        target: String,
        model: Option<String>,
    ) -> TranslateResult {
        match self
            .bridge
            .translate_response(&body, &source, &target, model.as_deref())
        {
            Ok(result) => TranslateResult {
                result: Some(result),
                error: None,
            },
            Err(error) => TranslateResult {
                result: None,
                error: Some(error_info(&error)),
            },
        }
    }

    /// The recorded usage rows, when the proxy was built with
    /// `record_usage: true`.
    #[napi]
    pub fn usage_events(&self) -> Vec<Value> {
        self.bridge.usage_events()
    }
}

/// A streaming response. `next()` resolves with the next event object, or
/// `null` once the stream has ended or been cancelled; `cancel()` stops
/// delivery and settles the request as `ClientCancelled` — including while a
/// `next()` call is still pending, so a watchdog timer can cancel a stream
/// mid-iteration.
#[napi]
pub struct Stream {
    /// The bridge stream this handle drains; its methods take `&self`, so
    /// cancel never waits on a pending next.
    inner: BridgeStream,
}

#[napi]
impl Stream {
    /// The next event, or null when the stream has ended or been cancelled.
    #[napi]
    pub async fn next(&self) -> Result<Option<Value>> {
        Ok(self.inner.next().await)
    }

    /// Cancel the stream: stop delivering events and settle the request as
    /// `ClientCancelled`.
    #[napi]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Whether this stream has been cancelled.
    #[napi(getter)]
    pub fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}
