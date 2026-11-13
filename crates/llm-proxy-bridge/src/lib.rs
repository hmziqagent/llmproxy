//! llm-proxy-bridge: the language-neutral surface the Python and Node
//! bindings wrap (plan 10, step 1). Everything crosses this boundary as plain
//! JSON — configuration in, requests and streaming events out — so the two
//! bindings stay thin shims with no logic of their own and the three
//! implementations (Rust, Python, Node) share one behaviour.
//!
//! The surface covers the four areas the plan names:
//!
//! - **Translation** ([`Bridge::translate_request`],
//!   [`Bridge::translate_response`]): pure codec work with no I/O, using the
//!   same client decoders, provider encoders and cross-format field policy as
//!   the server routes.
//! - **Running a request** ([`Bridge::execute`]): the full engine pipeline —
//!   admission, routing, credentials, the upstream attempt, settlement —
//!   against the configured providers, with the caller's dialect decoded in
//!   and encoded back out exactly as the HTTP routes do it.
//! - **Streaming events** ([`Bridge::execute_stream`]): the engine's core
//!   events as a JSON iterator; dropping it before it ends, or cancelling
//!   it, cancels the upstream request and settles it as `ClientCancelled`,
//!   while a stream read to its end stays completed.
//! - **Configuration from code** ([`Bridge::from_json`]): providers, model
//!   aliases and routing as a JSON document, mirroring the TOML daemon
//!   configuration.
//!
//! The bridge owns one multi-threaded tokio runtime so language runtimes
//! without an async executor can still drive it synchronously.

#![deny(missing_docs)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_adapters::{
    ProtocolName, ProviderAdapterRegistry, ProviderAdapterTarget, ProviderError, ProxyRequest,
    protocols,
};
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderConfig, ProviderRegistry, ProviderRouteKind, ServerConfig,
};
use llm_proxy_engine::{
    ClientDialect, Engine, EngineBuilder, EngineError, EngineRequest, EngineResponse, EngineStream,
    InboundRequest, StaticConfigSource,
};
use llm_proxy_protocol::anthropic::MessageRequest;
use llm_proxy_protocol::client::{
    ProtocolError, anthropic, gemini, ollama, openai_chat, responses,
};
use llm_proxy_protocol::core::{CoreEvent, CoreRequest, CoreResponse};
use llm_proxy_protocol::openai::ChatCompletionRequest;
use llm_proxy_storage::{EventBus, ProxyEvent};
use secrecy::SecretString;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::watch;

/// The channel feeding a stream's consumer; a handful of events buffered so
/// the upstream pipeline keeps moving while a slow consumer processes one.
const STREAM_CHANNEL_CAPACITY: usize = 16;

/// Everything a binding-level failure carries, mirroring
/// [`EngineError`] in a shape that crosses FFI unchanged.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{kind} ({status}): {message}")]
pub struct BridgeError {
    /// The `error_kind` string the usage ledger records (for example
    /// `UnknownProvider`, `InvalidRequest`, `Upstream`).
    pub kind: String,
    /// The HTTP status the server would map this failure to.
    pub status: u16,
    /// The sanitized client-facing message.
    pub message: String,
    /// The upstream- or ladder-advised wait, when this failure carries one.
    pub retry_after: Option<Duration>,
    /// Whether the proxy wants the client to retry this request itself.
    pub should_retry: bool,
}

impl BridgeError {
    /// A request the caller can fix (bad dialect, bad body, bad config
    /// shape): kind `InvalidRequest`, status 400.
    fn invalid(message: String) -> Self {
        Self {
            kind: "InvalidRequest".to_owned(),
            status: 400,
            message,
            retry_after: None,
            should_retry: false,
        }
    }

    /// A configuration the bridge cannot assemble: kind `InvalidConfig`,
    /// status 500.
    fn config(message: String) -> Self {
        Self {
            kind: "InvalidConfig".to_owned(),
            status: 500,
            message,
            retry_after: None,
            should_retry: false,
        }
    }

    /// The process could not start the async runtime: kind
    /// `RuntimeUnavailable`, status 500.
    fn runtime(message: String) -> Self {
        Self {
            kind: "RuntimeUnavailable".to_owned(),
            status: 500,
            message,
            retry_after: None,
            should_retry: false,
        }
    }
}

impl From<EngineError> for BridgeError {
    fn from(error: EngineError) -> Self {
        Self {
            kind: error.kind_name().to_owned(),
            status: error.status_code(),
            message: error.client_message(),
            retry_after: error.retry_after(),
            should_retry: error.should_retry(),
        }
    }
}

impl From<ProtocolError> for BridgeError {
    fn from(error: ProtocolError) -> Self {
        Self::invalid(error.to_string())
    }
}

impl From<ProviderError> for BridgeError {
    fn from(error: ProviderError) -> Self {
        match &error {
            // A cross-format field-policy rejection names the fields and is
            // the client's to fix, like every other decode failure here.
            ProviderError::InvalidRequest(_) | ProviderError::InvalidConfig(_) => {
                Self::invalid(error.to_string())
            }
            _ => Self {
                kind: "ProviderDecode".to_owned(),
                status: 502,
                message: llm_proxy_engine::sanitize_upstream_error_body(&error.to_string()),
                retry_after: None,
                should_retry: false,
            },
        }
    }
}

/// The binding-level configuration document: the providers to route to plus
/// bridge-only knobs. Deserialized from the JSON a Python dict or JS object
/// converts into.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeConfig {
    /// The providers, in the same shape as `[[provider]]` TOML tables.
    pub providers: Vec<ProviderConfig>,
    /// The whole-request timeout, overriding the 300 s default.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
    /// Record every usage row the engine emits and expose it through
    /// [`Bridge::usage_events`] (off by default; nothing is persisted).
    #[serde(default)]
    pub record_usage: bool,
}

/// Per-call options: the model for dialects that carry it in the URL
/// (Gemini), and inbound headers for the per-provider forward allowlist.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecuteOptions {
    /// The model named in the URL path (Gemini's `generateContent`); ignored
    /// by dialects that carry the model in the body.
    #[serde(default)]
    pub model: Option<String>,
    /// Inbound headers, names lowercased, subject to the provider's forward
    /// allowlist minus the never-forwarded set (client credentials, `Origin`,
    /// browser and transport headers).
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

impl ExecuteOptions {
    /// Options from a JSON object (`{}` when the caller passed nothing).
    ///
    /// # Errors
    ///
    /// [`BridgeError`] when the value is not an object with the two known
    /// keys.
    pub fn from_json(value: &Value) -> Result<Self, BridgeError> {
        serde_json::from_value(value.clone())
            .map_err(|error| BridgeError::invalid(format!("invalid options object: {error}")))
    }
}

/// A usage sink the bridge can expose across FFI: every [`ProxyEvent`] the
/// engine emits, in order, in memory. Nothing is persisted.
#[derive(Debug, Default)]
pub struct RecordingSink {
    events: std::sync::Mutex<Vec<ProxyEvent>>,
}

impl EventBus for RecordingSink {
    fn emit(&self, event: &ProxyEvent) {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(event.clone());
    }
}

impl RecordingSink {
    /// The recorded events, oldest first.
    pub fn snapshot(&self) -> Vec<ProxyEvent> {
        self.events
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

/// Which wire dialect a binding caller speaks, mapped to the wire format,
/// the usage-row dialect and the route kind the engine needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DialectSpec {
    format: ClientWireFormat,
    dialect: ClientDialect,
    route_kind: ProviderRouteKind,
}

fn dialect_spec(name: &str) -> Result<DialectSpec, BridgeError> {
    let spec = |format, dialect, route_kind| DialectSpec {
        format,
        dialect,
        route_kind,
    };
    match name {
        "anthropic_messages" => Ok(spec(
            ClientWireFormat::AnthropicMessages,
            ClientDialect::Anthropic,
            ProviderRouteKind::Messages,
        )),
        "openai_chat_completions" => Ok(spec(
            ClientWireFormat::OpenAiChatCompletions,
            ClientDialect::OpenAiChat,
            ProviderRouteKind::ChatCompletions,
        )),
        "openai_responses" => Ok(spec(
            ClientWireFormat::OpenAiResponses,
            ClientDialect::OpenAiResponses,
            ProviderRouteKind::Responses,
        )),
        "gemini_generate_content" => Ok(spec(
            ClientWireFormat::GeminiGenerateContent,
            ClientDialect::Gemini,
            ProviderRouteKind::GenerateContent,
        )),
        // Both Ollama-native routes resolve through the chat_completions
        // route kind (see `ProviderRoutesConfig`).
        "ollama_chat" => Ok(spec(
            ClientWireFormat::OllamaChat,
            ClientDialect::Ollama,
            ProviderRouteKind::ChatCompletions,
        )),
        "ollama_generate" => Ok(spec(
            ClientWireFormat::OllamaGenerate,
            ClientDialect::Ollama,
            ProviderRouteKind::ChatCompletions,
        )),
        other => Err(BridgeError::invalid(format!(
            "unknown dialect {other:?}; expected one of anthropic_messages, \
             openai_chat_completions, openai_responses, gemini_generate_content, \
             ollama_chat, ollama_generate"
        ))),
    }
}

/// A client request decoded into the pieces the engine needs.
struct DecodedClient {
    core: CoreRequest,
    wire: ClientWireRequest,
    spec: DialectSpec,
}

fn decode_client(
    spec: DialectSpec,
    body: &Value,
    options: &ExecuteOptions,
    streaming: bool,
) -> Result<DecodedClient, BridgeError> {
    let core = match spec.format {
        ClientWireFormat::AnthropicMessages => {
            let request: MessageRequest = serde_json::from_value(body.clone())
                .map_err(|error| BridgeError::invalid(format!("invalid request body: {error}")))?;
            request.validate()?;
            anthropic::decode_request(request)?
        }
        ClientWireFormat::OpenAiChatCompletions => {
            let request: ChatCompletionRequest = serde_json::from_value(body.clone())
                .map_err(|error| BridgeError::invalid(format!("invalid request body: {error}")))?;
            openai_chat::decode_request(request)?
        }
        ClientWireFormat::OpenAiResponses => responses::decode_request(body)?,
        ClientWireFormat::GeminiGenerateContent => {
            let model = options.model.as_deref().unwrap_or_default();
            gemini::decode_request(body, model, streaming)?
        }
        ClientWireFormat::OllamaChat => ollama::decode_chat_request(body)?,
        ClientWireFormat::OllamaGenerate => ollama::decode_generate_request(body)?,
        // `ClientWireFormat` is non-exhaustive; a future dialect has no
        // decode path here until this bridge learns it.
        other => {
            return Err(BridgeError::invalid(format!(
                "unsupported dialect {:?}",
                other.name()
            )));
        }
    };
    let wire = ClientWireRequest::new(spec.format, body.clone());
    Ok(DecodedClient { core, wire, spec })
}

/// Encodes a completed response back into the caller's dialect, the same
/// codecs the server routes use.
fn encode_client_response(spec: DialectSpec, response: CoreResponse) -> Result<Value, BridgeError> {
    match spec.format {
        ClientWireFormat::AnthropicMessages => {
            serde_json::to_value(anthropic::encode_response(response)?)
                .map_err(|error| BridgeError::invalid(format!("encode error: {error}")))
        }
        ClientWireFormat::OpenAiChatCompletions => {
            serde_json::to_value(openai_chat::encode_response(response)?)
                .map_err(|error| BridgeError::invalid(format!("encode error: {error}")))
        }
        ClientWireFormat::OpenAiResponses => {
            responses::encode_response(&response).map_err(Into::into)
        }
        ClientWireFormat::GeminiGenerateContent => {
            gemini::encode_response(&response).map_err(Into::into)
        }
        ClientWireFormat::OllamaChat => ollama::encode_chat_response(&response).map_err(Into::into),
        ClientWireFormat::OllamaGenerate => {
            ollama::encode_generate_response(&response).map_err(Into::into)
        }
        other => Err(BridgeError::invalid(format!(
            "unsupported dialect {:?}",
            other.name()
        ))),
    }
}

fn body_hash(body: &Value) -> String {
    let bytes = serde_json::to_vec(body).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Lowercased inbound headers for the forward allowlist.
fn normalized_headers(options: &ExecuteOptions) -> HashMap<String, String> {
    options
        .headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect()
}

fn engine_request(
    provider: &str,
    decoded: DecodedClient,
    options: &ExecuteOptions,
    body: &Value,
) -> EngineRequest {
    let inbound = InboundRequest {
        auth: None,
        headers: normalized_headers(options),
        wire: decoded.wire,
    };
    EngineRequest::new(
        provider,
        decoded.spec.route_kind,
        decoded.spec.dialect,
        decoded.core,
        inbound,
    )
    .with_body_hash(body_hash(body))
}

/// The JSON facade over the engine. Clone is cheap; every clone shares the
/// same engine, and all bridges in the process share one async runtime (a
/// per-bridge runtime would panic when dropped from an async context, and
/// language embedders create bridges freely).
#[derive(Clone)]
pub struct Bridge {
    engine: Engine,
    recording: Option<Arc<RecordingSink>>,
    runtime: &'static Runtime,
}

impl std::fmt::Debug for Bridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bridge")
            .field("engine", &self.engine)
            .field(
                "recording",
                &self.recording.as_ref().map(|sink| sink.snapshot().len()),
            )
            .field("runtime", &"tokio multi-thread")
            .finish()
    }
}

impl Bridge {
    /// Assemble a bridge from a JSON configuration document
    /// ([`BridgeConfig`]'s shape).
    ///
    /// # Errors
    ///
    /// [`BridgeError`] when the configuration cannot be deserialized, the
    /// provider set fails validation, or the runtime cannot start.
    pub fn from_json(config: &Value) -> Result<Self, BridgeError> {
        let config: BridgeConfig = serde_json::from_value(config.clone())
            .map_err(|error| BridgeError::config(format!("invalid bridge config: {error}")))?;
        Self::from_config(config)
    }

    /// Assemble a bridge from a [`BridgeConfig`].
    ///
    /// # Errors
    ///
    /// [`BridgeError`] when the provider set fails validation, the transport
    /// client cannot be built, or the runtime cannot start.
    pub fn from_config(config: BridgeConfig) -> Result<Self, BridgeError> {
        let BridgeConfig {
            providers,
            request_timeout_secs,
            record_usage,
        } = config;
        let registry = ProviderRegistry::from_providers(providers)
            .map_err(|error| BridgeError::config(error.to_string()))?;
        // Fail with a configuration error (not an internal one) when a
        // provider names a protocol no registered adapter serves.
        let builtin_protocols = ProviderAdapterRegistry::builtin().protocol_names();
        registry
            .validate_protocols(builtin_protocols.iter().map(String::as_str))
            .map_err(|error| BridgeError::config(error.to_string()))?;
        let mut server = embedded_server_config();
        if let Some(secs) = request_timeout_secs {
            server.request_timeout = Duration::from_secs(secs);
        }
        let app_config = AppConfig {
            server,
            auth: Default::default(),
            storage: None,
            replica: None,
            mcp: None,
        };
        let recording = record_usage.then(|| Arc::new(RecordingSink::default()));
        let client = llm_proxy_transport::ProxyClient::try_with_timeouts(
            &app_config.server.upstream_timeouts,
        )
        .map_err(|error| BridgeError::config(error.to_string()))?;
        let mut builder = EngineBuilder::from_parts(
            Arc::new(StaticConfigSource::new(Arc::new(app_config))),
            Arc::new(registry),
            Arc::new(ProviderAdapterRegistry::builtin()),
            Arc::new(client),
        );
        if let Some(sink) = &recording {
            let bus: Arc<dyn EventBus> = Arc::clone(sink) as Arc<dyn EventBus>;
            builder = builder.event_bus(bus);
        }
        let runtime = shared_runtime()?;
        let engine = builder.build()?;
        Ok(Self {
            engine,
            recording,
            runtime,
        })
    }

    /// The engine this bridge runs (for Rust embedders that want the typed
    /// surface underneath).
    #[must_use]
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Run one non-streaming request end to end: decode the caller's
    /// dialect, run the pipeline, encode the response back into it. A
    /// same-format passthrough returns the upstream body verbatim, exactly
    /// as the HTTP route does.
    ///
    /// # Errors
    ///
    /// [`BridgeError`] for every failure the pipeline can produce; usage
    /// rows are still written for failures.
    pub async fn execute(
        &self,
        provider: &str,
        dialect: &str,
        body: &Value,
        options: &ExecuteOptions,
    ) -> Result<Value, BridgeError> {
        let spec = dialect_spec(dialect)?;
        let decoded = decode_client(spec, body, options, false)?;
        let request = engine_request(provider, decoded, options, body);
        let response: EngineResponse = self.engine.execute(request).await?;
        client_visible_response(spec, response)
    }

    /// Run one streaming request. Returns once the pre-commit window has
    /// crossed (so pre-commit failures surface as [`BridgeError`]); the
    /// [`EventStream`] then carries the engine's core events as JSON.
    ///
    /// # Errors
    ///
    /// [`BridgeError`] for any failure before the first event is committed.
    pub async fn execute_stream(
        &self,
        provider: &str,
        dialect: &str,
        body: &Value,
        options: &ExecuteOptions,
    ) -> Result<EventStream, BridgeError> {
        let spec = dialect_spec(dialect)?;
        let decoded = decode_client(spec, body, options, true)?;
        let request = engine_request(provider, decoded, options, body);
        let stream: EngineStream = self.engine.execute_stream(request).await?;
        let (tx, rx) = mpsc::channel(STREAM_CHANNEL_CAPACITY);
        let cancel = Arc::new(StreamCancel::default());
        self.runtime
            .spawn(pump_events(stream, tx, Arc::clone(&cancel)));
        Ok(EventStream {
            consumer: tokio::sync::Mutex::new(ConsumerEnd {
                rx,
                wake: cancel.wake.subscribe(),
            }),
            cancel,
            ended: AtomicBool::new(false),
        })
    }

    /// Translate a request between wire dialects with no I/O: the client
    /// dialect in, a provider protocol out, through the same decoders,
    /// encoders and cross-format field policy (carried / mapped / rejected
    /// by name) the pipeline applies. `model` supplies the URL-path model
    /// when `from` is `gemini_generate_content`.
    ///
    /// # Errors
    ///
    /// [`BridgeError`] when either side names an unknown dialect/protocol,
    /// the body does not decode, or the cross-format policy rejects a field.
    pub fn translate_request(
        &self,
        body: &Value,
        from: &str,
        to: &str,
        model: Option<&str>,
    ) -> Result<Value, BridgeError> {
        let spec = dialect_spec(from)?;
        let options = ExecuteOptions {
            model: model.map(str::to_owned),
            headers: HashMap::new(),
        };
        let decoded = decode_client(spec, body, &options, false)?;
        let target = translate_target(to, &decoded.core.model.requested)?;
        let adapter = self
            .engine
            .adapters()
            .get(to)
            .ok_or_else(|| unknown_protocol(to))?;
        let proxy_request: ProxyRequest =
            adapter.encode_request(&decoded.core, &target, &decoded.wire)?;
        serde_json::from_slice(&proxy_request.body)
            .map_err(|error| BridgeError::invalid(format!("encoded request is not JSON: {error}")))
    }

    /// Translate a provider response into a client dialect with no I/O:
    /// decode with the provider's adapter, encode with the client codec.
    /// `model` overrides the requested model recorded on the result;
    /// otherwise the response body's `model` field is used when present.
    ///
    /// # Errors
    ///
    /// [`BridgeError`] when either side names an unknown protocol/dialect or
    /// the body does not decode.
    pub fn translate_response(
        &self,
        body: &Value,
        from: &str,
        to: &str,
        model: Option<&str>,
    ) -> Result<Value, BridgeError> {
        let adapter = self
            .engine
            .adapters()
            .get(from)
            .ok_or_else(|| unknown_protocol(from))?;
        let inferred = model
            .map(str::to_owned)
            .or_else(|| body.get("model").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_default();
        let target = translate_target(from, &inferred)?;
        let bytes = serde_json::to_vec(body)
            .map_err(|error| BridgeError::invalid(format!("response is not JSON: {error}")))?;
        let core: CoreResponse = adapter.decode_response(&bytes, &target)?;
        let spec = dialect_spec(to)?;
        encode_client_response(spec, core)
    }

    /// The usage rows recorded so far, serialized as JSON, when the bridge
    /// was built with `record_usage`. Nothing is recorded otherwise.
    #[must_use]
    pub fn usage_events(&self) -> Vec<Value> {
        let Some(sink) = &self.recording else {
            return Vec::new();
        };
        sink.snapshot()
            .into_iter()
            .filter_map(|event| match serde_json::to_value(&event) {
                Ok(value) => Some(value),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "usage event could not be serialized for the binding surface"
                    );
                    None
                }
            })
            .collect()
    }

    /// The process-wide async runtime, for language runtimes that need to
    /// drive the async surface from synchronous code.
    #[must_use]
    pub fn runtime(&self) -> &'static Runtime {
        self.runtime
    }
}

/// The one async runtime every bridge in the process shares, created on
/// first use and never dropped (dropping a tokio runtime from an async
/// context is a panic, and language embedders cannot control where the last
/// bridge goes away).
fn shared_runtime() -> Result<&'static Runtime, BridgeError> {
    static RUNTIME: std::sync::OnceLock<Result<Runtime, String>> = std::sync::OnceLock::new();
    match RUNTIME.get_or_init(|| Runtime::new().map_err(|error| error.to_string())) {
        Ok(runtime) => Ok(runtime),
        Err(message) => Err(BridgeError::runtime(format!(
            "could not start the async runtime: {message}"
        ))),
    }
}

fn client_visible_response(
    spec: DialectSpec,
    response: EngineResponse,
) -> Result<Value, BridgeError> {
    if let Some(raw) = response.raw_body {
        return serde_json::from_slice(&raw).map_err(|error| {
            BridgeError::invalid(format!("passthrough body is not JSON: {error}"))
        });
    }
    encode_client_response(spec, response.core)
}

fn unknown_protocol(name: &str) -> BridgeError {
    BridgeError::invalid(format!(
        "unknown protocol {name:?}; expected one of {}",
        protocols::BUILTIN.join(", ")
    ))
}

/// A synthetic adapter target for pure translation: no credentials, a
/// placeholder endpoint, no aliasing (requested and upstream model are the
/// same string).
fn translate_target(
    protocol_name: &str,
    model: &str,
) -> Result<ProviderAdapterTarget, BridgeError> {
    let protocol = ProtocolName::new(protocol_name);
    if !matches!(
        protocol.as_str(),
        protocols::OPENAI_CHAT_COMPLETIONS
            | protocols::ANTHROPIC_MESSAGES
            | protocols::OPENAI_RESPONSES
            | protocols::GEMINI_GENERATE_CONTENT
    ) {
        return Err(unknown_protocol(protocol_name));
    }
    let endpoint = match protocol.as_str() {
        protocols::ANTHROPIC_MESSAGES => "http://translate.invalid/v1/messages".to_owned(),
        protocols::OPENAI_CHAT_COMPLETIONS => {
            "http://translate.invalid/v1/chat/completions".to_owned()
        }
        protocols::OPENAI_RESPONSES => "http://translate.invalid/v1/responses".to_owned(),
        _ => "http://translate.invalid/v1beta/models/{model}:generateContent".to_owned(),
    };
    Ok(ProviderAdapterTarget {
        provider_name: "translate".to_owned(),
        adapter_name: "translate".to_owned(),
        protocol,
        endpoint,
        auth_style: AuthStyle::Bearer,
        api_key: SecretString::from("translation-never-sends"),
        requested_model: model.to_owned(),
        upstream_model: model.to_owned(),
        headers: Arc::new(HashMap::new()),
        lossy: false,
        forward_inbound_headers: Arc::new(Vec::new()),
        prompt_cache: None,
    })
}

/// The server-independent parts of an `AppConfig` the engine needs; the
/// bridge never binds a port, so the bind address is a placeholder.
fn embedded_server_config() -> ServerConfig {
    ServerConfig {
        bind: std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
        request_timeout: Duration::from_secs(300),
        shutdown_timeout: Duration::from_secs(5),
        log_level: "warn".to_owned(),
        hot_reload: false,
        allowed_origins: None,
        server_name: "llm-proxy-bridge".to_owned(),
        rate_limit_rpm: 0,
        trust_forwarded_headers: false,
        cache: Default::default(),
        idempotency: Default::default(),
        metrics: Default::default(),
        log_format: Default::default(),
        admin_bind: None,
        admin_on_main_listener: false,
        endpoint_families: Default::default(),
        upstream_timeouts: Default::default(),
        bare_v1: Default::default(),
        preflight: Default::default(),
        semantic_cache: Default::default(),
    }
}

/// Forwards the engine's events into the consumer channel until the stream
/// ends, the consumer goes away (its receiver drops) or the consumer
/// cancels. Returning drops the engine stream here, which is what records
/// the `ClientCancelled` settlement on the cancel and drop paths.
async fn pump_events(stream: EngineStream, tx: mpsc::Sender<Value>, cancel: Arc<StreamCancel>) {
    if cancel.is_cancelled() {
        return;
    }
    let mut wake = cancel.wake.subscribe();
    let EngineStream { head, tail } = stream;
    let mut tail = tail;
    for event in head {
        if !send_event(&cancel, &mut wake, &tx, &event).await {
            return;
        }
    }
    loop {
        if cancel.is_cancelled() {
            return;
        }
        tokio::select! {
            biased;
            changed = wake.changed() => {
                // The sender cannot drop while this task holds its Arc.
                if changed.is_err() || cancel.is_cancelled() {
                    return;
                }
            }
            event = tail.next() => match event {
                Some(event) => {
                    if !send_event(&cancel, &mut wake, &tx, &event).await {
                        return;
                    }
                }
                None => return,
            },
        }
    }
}

async fn send_event(
    cancel: &StreamCancel,
    wake: &mut watch::Receiver<bool>,
    tx: &mpsc::Sender<Value>,
    event: &CoreEvent,
) -> bool {
    if cancel.is_cancelled() {
        return false;
    }
    let value = match serde_json::to_value(event) {
        Ok(value) => value,
        Err(error) => {
            tracing::warn!(%error, "core event could not be serialized for the binding stream");
            return true;
        }
    };
    tokio::select! {
        biased;
        changed = wake.changed() => changed.is_ok() && !cancel.is_cancelled(),
        sent = tx.send(value) => sent.is_ok(),
    }
}

/// Cancellation state shared between an [`EventStream`] and the pump task
/// feeding it. The flag is the decision; the `watch` channel is the wakeup,
/// and it is level-triggered (a cancel that lands between a flag check and
/// the `changed()` registration still fires) so a cancel can never be lost
/// to a race.
#[derive(Debug, Default)]
struct StreamCancel {
    cancelled: AtomicBool,
    wake: watch::Sender<bool>,
}

impl StreamCancel {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// The consumer side of a stream's channel, plus its private wakeup feed.
/// Guarded by one mutex so sequential `next` callers cannot interleave
/// reads; `cancel` never takes this lock.
struct ConsumerEnd {
    rx: mpsc::Receiver<Value>,
    wake: watch::Receiver<bool>,
}

/// A streaming response as an iterator of JSON events. Dropping it before it
/// ends, or calling [`EventStream::cancel`], cancels the upstream request;
/// the engine then settles it as `ClientCancelled`. A stream the consumer
/// has read to its end — the terminal event or `None` — does not cancel on
/// drop; the engine settles that request as completed on its own.
///
/// Every method takes `&self`, so a cancel from any thread (a watchdog, a
/// Python thread) works while another thread is blocked in
/// [`EventStream::next`] or [`EventStream::next_blocking`].
pub struct EventStream {
    consumer: tokio::sync::Mutex<ConsumerEnd>,
    cancel: Arc<StreamCancel>,
    /// The stream ended of its own accord — the pump closed the channel after
    /// a normal end, or the consumer saw the terminal `MessageStop` event —
    /// so dropping it must not cancel a request that finished cleanly. Only
    /// the consumer knows this, which is why it lives here and not on
    /// [`StreamCancel`].
    ended: AtomicBool,
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream")
            .field("cancelled", &self.cancel.is_cancelled())
            .finish()
    }
}

impl EventStream {
    /// The next event, or `None` once the stream has ended or been
    /// cancelled. A concurrent [`EventStream::cancel`] unblocks a pending
    /// call; sequential callers are serialized.
    pub async fn next(&self) -> Option<Value> {
        let mut consumer = self.consumer.lock().await;
        let ConsumerEnd { rx, wake } = &mut *consumer;
        loop {
            if self.cancel.is_cancelled() {
                return None;
            }
            tokio::select! {
                biased;
                changed = wake.changed() => {
                    // The sender cannot drop while this stream holds it.
                    if changed.is_err() {
                        return None;
                    }
                }
                event = rx.recv() => {
                    return match event {
                        // An event that raced with a cancel is not delivered.
                        Some(event) if !self.cancel.is_cancelled() => {
                            self.mark_ended_if_terminal(&event);
                            Some(event)
                        }
                        // The pump closed the channel after a normal end: a
                        // later drop must not cancel the finished request.
                        None if !self.cancel.is_cancelled() => {
                            self.ended.store(true, Ordering::Release);
                            None
                        }
                        _ => None,
                    };
                }
            }
        }
    }

    /// The blocking form of [`EventStream::next`], for language runtimes
    /// without an event loop. Must be called from a thread that is not a
    /// tokio worker. A concurrent [`EventStream::cancel`] unblocks a pending
    /// call: cancelling stops the pump task, which closes the channel.
    ///
    /// # Panics
    ///
    /// Panics if called from within a tokio runtime worker thread.
    pub fn next_blocking(&self) -> Option<Value> {
        if self.cancel.is_cancelled() {
            return None;
        }
        let mut consumer = self.consumer.blocking_lock();
        if self.cancel.is_cancelled() {
            return None;
        }
        match consumer.rx.blocking_recv() {
            // An event that raced with a cancel is not delivered.
            Some(event) if !self.cancel.is_cancelled() => {
                self.mark_ended_if_terminal(&event);
                Some(event)
            }
            // The pump closed the channel after a normal end: a later drop
            // must not cancel the finished request.
            None if !self.cancel.is_cancelled() => {
                self.ended.store(true, Ordering::Release);
                None
            }
            _ => None,
        }
    }

    /// Cancel the stream: stop delivering events and settle the request as
    /// `ClientCancelled`. Idempotent, and safe to call while another thread
    /// is blocked waiting for an event.
    pub fn cancel(&self) {
        self.cancel.cancelled.store(true, Ordering::Release);
        // Infallible (a `watch` send needs no receiver) and level-triggered,
        // so both the pump and any pending consumer observe the cancel even
        // if they have not registered a waiter yet.
        self.cancel.wake.send_replace(true);
    }

    /// Whether this stream has been cancelled (not merely ended).
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Marks the stream ended when `event` is the engine's terminal event.
    ///
    /// Core events serialize as single-key tagged objects (`{"MessageStop":
    /// …}`), so the key's presence identifies the terminal variant. The pump
    /// may not yet have observed the engine stream's own end when the
    /// consumer sees it, and a drop at that point must not cancel a request
    /// that is about to settle as completed.
    fn mark_ended_if_terminal(&self, event: &Value) {
        if event.get("MessageStop").is_some() {
            self.ended.store(true, Ordering::Release);
        }
    }
}

impl Drop for EventStream {
    fn drop(&mut self) {
        // A stream that did not end of its own accord still has a live
        // upstream request behind the pump task; cancel it, or the request
        // runs until the engine's idle timeout instead of settling as
        // `ClientCancelled`. `cancel` only touches an `AtomicBool` and a
        // `watch` sender, so it is safe from any dropping thread, including
        // a tokio worker or a language runtime's finalizer.
        if !self.ended.load(Ordering::Acquire) {
            self.cancel();
        }
    }
}
