//! The Python extension module for llm-proxy (plan 10, step 2). A thin shim
//! over `llm_proxy_bridge`: JSON in via dict/list conversion, JSON out the
//! same way, no logic of its own. The user-facing API lives in the pure
//! Python package (`python/llm_proxy/__init__.py`), which unpacks the error
//! tuple into a `ProxyError` with attributes.

use std::sync::Arc;

use llm_proxy_bridge::{Bridge, BridgeError, EventStream, ExecuteOptions};
use pyo3::Py;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use serde_json::Value;

pyo3::create_exception!(
    llm_proxy,
    ProxyError,
    pyo3::exceptions::PyException,
    "Raised when a request through the llm-proxy engine fails."
);

/// The engine wrapper: built from a configuration dict, then executes or
/// streams requests.
#[pyclass]
struct Proxy {
    /// The shared bridge this proxy drives.
    bridge: Arc<Bridge>,
}

/// The error payload raised as `ProxyError.args`:
/// `(kind, status, message, retry_after_seconds, should_retry)`.
fn proxy_error(error: &BridgeError) -> PyErr {
    ProxyError::new_err((
        error.kind.clone(),
        error.status,
        error.message.clone(),
        error.retry_after.map(|d| d.as_secs_f64()),
        error.should_retry,
    ))
}

/// Converts a Python object (None, bool, int, float, str, list, dict with
/// string keys) into a JSON value.
fn value_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return obj.extract::<bool>().map(Value::Bool);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::Number(u.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return serde_json::Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err("float is not a JSON number"));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(value_from_py(&item)?);
        }
        return Ok(Value::Array(items));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key = key
                .cast::<PyString>()
                .map_err(|_| PyValueError::new_err("dict keys must be strings"))?
                .to_cow()?;
            map.insert(key.into_owned(), value_from_py(&value)?);
        }
        return Ok(Value::Object(map));
    }
    Err(PyValueError::new_err(format!(
        "cannot convert {} to JSON",
        obj.get_type()
    )))
}

/// Converts a JSON value into the matching Python object.
fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    let object = match value {
        Value::Null => py.None(),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                PyInt::new(py, i).to_owned().into_any().unbind()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                PyFloat::new(py, f).to_owned().into_any().unbind()
            }
        }
        Value::String(s) => PyString::new(py, s).to_owned().into_any().unbind(),
        Value::Array(list) => {
            let py_list = PyList::empty(py);
            for item in list {
                py_list.append(value_to_py(py, item)?)?;
            }
            py_list.into_any().unbind()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, value_to_py(py, item)?)?;
            }
            dict.into_any().unbind()
        }
    };
    Ok(object)
}

#[pymethods]
impl Proxy {
    /// Build the proxy from a configuration dict (the bridge config shape).
    #[new]
    fn new(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = value_from_py(config)?;
        let bridge = Bridge::from_json(&value).map_err(|error| proxy_error(&error))?;
        Ok(Self {
            bridge: Arc::new(bridge),
        })
    }

    /// Run one non-streaming request end to end and return the response in
    /// the caller's dialect.
    #[pyo3(signature = (provider, dialect, body, options=None))]
    fn execute(
        &self,
        py: Python<'_>,
        provider: &str,
        dialect: &str,
        body: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let body = value_from_py(body)?;
        let options = parse_options(options)?;
        let bridge = Arc::clone(&self.bridge);
        let provider = provider.to_owned();
        let dialect = dialect.to_owned();
        let result = py.detach(move || {
            bridge
                .runtime()
                .block_on(bridge.execute(&provider, &dialect, &body, &options))
        });
        match result {
            Ok(response) => value_to_py(py, &response),
            Err(error) => Err(proxy_error(&error)),
        }
    }

    /// Run one streaming request; returns an iterator of event dicts.
    /// Dropping the iterator before it ends, or calling `cancel()`, cancels
    /// the upstream request; a stream read to its end stays completed.
    #[pyo3(signature = (provider, dialect, body, options=None))]
    fn execute_stream(
        &self,
        py: Python<'_>,
        provider: &str,
        dialect: &str,
        body: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Stream> {
        let body = value_from_py(body)?;
        let options = parse_options(options)?;
        let bridge = Arc::clone(&self.bridge);
        let provider = provider.to_owned();
        let dialect = dialect.to_owned();
        let stream = py.detach(move || {
            bridge
                .runtime()
                .block_on(bridge.execute_stream(&provider, &dialect, &body, &options))
        });
        match stream {
            Ok(stream) => Ok(Stream { inner: stream }),
            Err(error) => Err(proxy_error(&error)),
        }
    }

    /// Translate a request between wire dialects with no I/O.
    #[pyo3(signature = (body, source, target, model=None))]
    fn translate_request(
        &self,
        py: Python<'_>,
        body: &Bound<'_, PyAny>,
        source: &str,
        target: &str,
        model: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let body = value_from_py(body)?;
        match self.bridge.translate_request(&body, source, target, model) {
            Ok(translated) => value_to_py(py, &translated),
            Err(error) => Err(proxy_error(&error)),
        }
    }

    /// Translate a provider response into a client dialect with no I/O.
    #[pyo3(signature = (body, source, target, model=None))]
    fn translate_response(
        &self,
        py: Python<'_>,
        body: &Bound<'_, PyAny>,
        source: &str,
        target: &str,
        model: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let body = value_from_py(body)?;
        match self.bridge.translate_response(&body, source, target, model) {
            Ok(translated) => value_to_py(py, &translated),
            Err(error) => Err(proxy_error(&error)),
        }
    }

    /// The recorded usage rows (JSON dicts), when the proxy was built with
    /// `record_usage: true`.
    fn usage_events(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = PyList::empty(py);
        for event in self.bridge.usage_events() {
            list.append(value_to_py(py, &event)?)?;
        }
        Ok(list.into_any().unbind())
    }
}

/// Parses the optional per-call options argument.
fn parse_options(options: Option<&Bound<'_, PyAny>>) -> PyResult<ExecuteOptions> {
    let Some(options) = options else {
        return Ok(ExecuteOptions::default());
    };
    if options.is_none() {
        return Ok(ExecuteOptions::default());
    }
    let value = value_from_py(options)?;
    ExecuteOptions::from_json(&value).map_err(|error| proxy_error(&error))
}

/// A streaming response: iterate for event dicts; `cancel()` stops
/// delivery and settles the request as `ClientCancelled`. Dropping this
/// native object before it ends cancels too; the pure-Python `Stream`
/// wrapper (`python/llm_proxy/__init__.py`) adds the user-facing paths —
/// `with`, `close()` and a finalizer — and documents the contract.
///
/// Every method goes through the bridge's shared-state synchronization, so
/// `cancel()` and the `cancelled` getter work from any thread while another
/// thread is blocked in `__next__` — no exclusive borrow is held across the
/// wait.
#[pyclass]
struct Stream {
    /// The bridge stream this iterator drains.
    inner: EventStream,
}

#[pymethods]
impl Stream {
    /// The iterator protocol: the stream is its own iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Blocks until the next event is available; releases the GIL while
    /// waiting, so other threads can call `cancel()` meanwhile.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let event = py.detach(|| self.inner.next_blocking());
        match event {
            Some(event) => value_to_py(py, &event).map(Some),
            None => Ok(None),
        }
    }

    /// Cancel the stream: stop delivering events and settle the request as
    /// `ClientCancelled`. Safe to call from any thread, including while
    /// another thread is blocked in `__next__`.
    fn cancel(&self) {
        self.inner.cancel();
    }

    /// Whether this stream has been cancelled.
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

/// The extension module: the native shim the pure-Python package wraps.
#[pymodule]
fn _llm_proxy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<Proxy>()?;
    m.add_class::<Stream>()?;
    m.add("ProxyError", py.get_type::<crate::ProxyError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
