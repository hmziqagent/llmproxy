//! Normalised core protocol types between client and provider adapters;
//! adapters translate to/from these, never to each other's wire formats.
//! Opaque JSON fields redact their `Debug` output — do not store secrets.

use std::fmt;

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

struct OpaqueJsonRef<'a>(&'a serde_json::Value);

impl fmt::Debug for OpaqueJsonRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        redact_value(self.0, f)
    }
}

fn redact_value(val: &serde_json::Value, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match val {
        serde_json::Value::Null => f.write_str("Null"),
        serde_json::Value::Bool(_) => f.write_str("Bool(_)"),
        serde_json::Value::Number(_) => f.write_str("Number(_)"),
        serde_json::Value::String(s) => write!(f, "String({} bytes)", s.len()),
        serde_json::Value::Array(arr) => write!(f, "Array({} items)", arr.len()),
        serde_json::Value::Object(map) => write!(f, "Object({} keys)", map.len()),
    }
}

fn redact_map(
    map: &serde_json::Map<String, serde_json::Value>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    write!(f, "Object({} keys)", map.len())
}

/// A model identifier tracking both the client-facing name and the resolved
/// upstream name; adapters must not replace `requested`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    /// The model name as requested by the client.
    pub requested: String,
    /// The resolved upstream model name, if different from `requested`.
    pub upstream: Option<String>,
}

impl ModelRef {
    /// Returns the effective model identifier: `upstream` if set, else `requested`.
    pub fn provider_model(&self) -> &str {
        self.upstream.as_deref().unwrap_or(&self.requested)
    }
}

/// A normalized chat request: the canonical internal representation that
/// client adapters decode into and provider adapters encode from.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreRequest {
    /// Model identifier (client-facing + optional upstream override).
    pub model: ModelRef,
    /// System / developer instructions (canonical location); client adapters
    /// normalise system-role messages into this field where possible.
    pub system: Vec<CoreContent>,
    /// Conversation turns after system content has been separated.
    pub messages: Vec<CoreMessage>,
    /// Tool definitions available to the model.
    pub tools: Vec<CoreTool>,
    /// Controls which (if any) tool the model must call.
    pub tool_choice: Option<CoreToolChoice>,
    /// Sampling parameters (temperature, top-p, etc.).
    pub sampling: SamplingOptions,
    /// Whether the client requested streaming.
    pub stream: bool,
    /// Caller metadata (user id, arbitrary key-value pairs).
    pub metadata: RequestMetadata,
    /// Opaque hints consumed only by provider adapters.
    pub provider_hints: ProviderHints,
}

impl fmt::Debug for CoreRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreRequest")
            .field("model", &self.model)
            .field("system", &self.system)
            .field("messages", &self.messages)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("sampling", &self.sampling)
            .field("stream", &self.stream)
            .field("metadata", &self.metadata)
            .field("provider_hints", &self.provider_hints)
            .finish()
    }
}

/// A single conversation turn.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreMessage {
    /// Speaker role.
    pub role: CoreRole,
    /// Content blocks in this turn.
    pub content: Vec<CoreContent>,
}

impl fmt::Debug for CoreMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreMessage")
            .field("role", &self.role)
            .field("content", &self.content)
            .finish()
    }
}

/// Speaker role within a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CoreRole {
    /// System-level instructions (only for protocols that cannot separate
    /// system content cleanly; prefer `CoreRequest.system`).
    System,
    /// End-user message.
    User,
    /// Model response.
    Assistant,
    /// Tool result.
    Tool,
}

/// A single content block inside a message or system prompt: text, media,
/// tool interactions, and extended thinking. Unknown variant fields are
/// rejected on decode; passthrough belongs in `provider_hints`/extra maps.
#[non_exhaustive]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CoreContent {
    /// Plain text.
    Text {
        /// The text content.
        text: String,
        /// Optional cache-control directive.
        cache: Option<CacheControl>,
    },
    /// Image content (source is provider-specific JSON).
    Image {
        /// Opaque image source data.
        source: serde_json::Value,
    },
    /// Document / file attachment.
    Document {
        /// Opaque document source data.
        source: serde_json::Value,
    },
    /// Audio content.
    Audio {
        /// Opaque audio source data.
        source: serde_json::Value,
    },
    /// Video content.
    Video {
        /// Opaque video source data.
        source: serde_json::Value,
    },
    /// A tool invocation emitted by the model.
    ToolUse {
        /// Unique tool-call identifier.
        id: String,
        /// Name of the tool being called.
        name: String,
        /// JSON arguments for the tool call.
        input: serde_json::Value,
        /// Opaque tool-call signature a provider requires on replay (Gemini
        /// `thoughtSignature`); redacted to `[REDACTED]` in `Debug` output.
        /// Anthropic-style thinking signatures live on [`CoreContent::Thinking`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// The result of a tool invocation, fed back to the model.
    ToolResult {
        /// The tool-call id this result corresponds to.
        tool_use_id: String,
        /// Content of the result (can nest text, images, etc.).
        content: Vec<CoreContent>,
        /// Whether this result represents an error.
        is_error: bool,
        /// Optional cache-control directive carried on the `tool_result` block.
        cache: Option<CacheControl>,
    },
    /// Extended thinking content.
    Thinking {
        /// The thinking text.
        text: String,
        /// Optional cryptographic signature (e.g. Anthropic-style); redacted to
        /// `[REDACTED]` in `Debug` output.
        signature: Option<String>,
    },
    /// Redacted thinking block whose plaintext is not available.
    RedactedThinking {
        /// Opaque redacted data (`serde_json::Value` since providers encode it
        /// differently); the Anthropic adapter converts non-strings to strings.
        data: serde_json::Value,
    },
    /// A model refusal to answer.
    Refusal {
        /// Human-readable refusal text.
        text: String,
    },
    /// A client block the core does not model (an Anthropic `server_tool_use`
    /// or `web_search_tool_result` replay, an MCP block, ...), carried as its
    /// raw wire JSON. Only same-format passthrough can forward it; cross-format
    /// it is rejected by name.
    Opaque {
        /// The raw wire block; `Debug` shows only its `type`.
        block: serde_json::Value,
    },
}

impl fmt::Debug for CoreContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreContent::Text { text, cache } => f
                .debug_struct("Text")
                .field("text", &text)
                .field("cache", &cache)
                .finish(),
            CoreContent::Image { source } => f
                .debug_struct("Image")
                .field("source", &OpaqueJsonRef(source))
                .finish(),
            CoreContent::Document { source } => f
                .debug_struct("Document")
                .field("source", &OpaqueJsonRef(source))
                .finish(),
            CoreContent::Audio { source } => f
                .debug_struct("Audio")
                .field("source", &OpaqueJsonRef(source))
                .finish(),
            CoreContent::Video { source } => f
                .debug_struct("Video")
                .field("source", &OpaqueJsonRef(source))
                .finish(),
            CoreContent::ToolUse {
                id,
                name,
                input,
                signature,
            } => f
                .debug_struct("ToolUse")
                .field("id", &id)
                .field("name", &name)
                .field("input", &OpaqueJsonRef(input))
                .field("signature", &signature.as_ref().map(|_| "[REDACTED]"))
                .finish(),
            CoreContent::ToolResult {
                tool_use_id,
                content,
                is_error,
                cache,
            } => f
                .debug_struct("ToolResult")
                .field("tool_use_id", &tool_use_id)
                .field("content", &content)
                .field("is_error", &is_error)
                .field("cache", &cache)
                .finish(),
            CoreContent::Thinking { text, signature } => f
                .debug_struct("Thinking")
                .field("text", &text)
                .field("signature", &signature.as_ref().map(|_| "[REDACTED]"))
                .finish(),
            CoreContent::RedactedThinking { data } => f
                .debug_struct("RedactedThinking")
                .field("data", &OpaqueJsonRef(data))
                .finish(),
            CoreContent::Refusal { text } => {
                f.debug_struct("Refusal").field("text", &text).finish()
            }
            CoreContent::Opaque { block } => f
                .debug_struct("Opaque")
                .field(
                    "type",
                    &block
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .map(|t| crate::util::truncate_str_safe(t, 64)),
                )
                .field("block", &OpaqueJsonRef(block))
                .finish(),
        }
    }
}

/// Cache-control directive attached to a content block.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheControl {
    /// Discriminator -- typically `"ephemeral"`.
    pub r#type: CacheControlType,
    /// Cache lifetime such as `"5m"` or `"1h"`; the provider default applies when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl fmt::Debug for CacheControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CacheControl")
            .field("type", &self.r#type)
            .field("ttl", &self.ttl)
            .finish()
    }
}

/// Known cache control type variants; unknown provider values are preserved
/// in `Other(String)` rather than rejected.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheControlType {
    /// Anthropic-style ephemeral cache control.
    Ephemeral,
    /// An unknown cache control type, preserved as a raw string.
    Other(String),
}

impl From<String> for CacheControlType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "ephemeral" => Self::Ephemeral,
            _ => Self::Other(value),
        }
    }
}

impl CacheControlType {
    /// Returns the string representation without consuming `self`.
    pub fn as_str(&self) -> &str {
        match self {
            CacheControlType::Ephemeral => "ephemeral",
            CacheControlType::Other(s) => s,
        }
    }
}

impl From<CacheControlType> for String {
    fn from(val: CacheControlType) -> Self {
        match val {
            CacheControlType::Ephemeral => "ephemeral".to_owned(),
            CacheControlType::Other(s) => s,
        }
    }
}

impl Serialize for CacheControlType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for CacheControlType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(s.into())
    }
}

/// A tool definition available to the model.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreTool {
    /// Name the model uses to reference this tool.
    pub name: String,
    /// Human-readable description.
    pub description: Option<String>,
    /// JSON Schema describing the tool's parameters; must be a JSON object,
    /// enforced on decode via [`deserialize_input_schema_object`]. Struct-literal
    /// builders supply an object (empty when upstream omits it).
    #[serde(deserialize_with = "deserialize_input_schema_object")]
    pub input_schema: serde_json::Value,
}

impl fmt::Debug for CoreTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("input_schema", &OpaqueJsonRef(&self.input_schema))
            .finish()
    }
}

/// Controls which (if any) tool the model must call.
#[non_exhaustive]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum CoreToolChoice {
    /// The model decides whether to call a tool.
    Auto,
    /// The model must call **some** tool.
    Any,
    /// The model must not call any tool.
    None,
    /// The model must call the named tool.
    Tool {
        /// Name of the tool to call.
        name: String,
    },
    /// A client tool-choice shape with no canonical variant yet; adapter tests
    /// must round-trip it intentionally or move it into metadata/hints.
    Raw(serde_json::Value),
}

impl fmt::Debug for CoreToolChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreToolChoice::Auto => f.write_str("Auto"),
            CoreToolChoice::Any => f.write_str("Any"),
            CoreToolChoice::None => f.write_str("None"),
            CoreToolChoice::Tool { name } => f.debug_struct("Tool").field("name", &name).finish(),
            CoreToolChoice::Raw(v) => f.debug_tuple("Raw").field(&OpaqueJsonRef(v)).finish(),
        }
    }
}

/// Sampling parameters. The decode boundary enforces `temperature` finite
/// `>= 0` and `top_p` finite in `[0, 1]`; `max_tokens` is not range-checked
/// (a `0` sentinel is legitimate for some providers; the upstream decides).
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingOptions {
    /// Sampling temperature; must be finite and `>= 0`, enforced on both the
    /// decode and encode paths (serde_json would turn non-finite into `null`).
    #[serde(
        default,
        serialize_with = "serialize_finite_f64",
        deserialize_with = "deserialize_temperature"
    )]
    pub temperature: Option<f64>,
    /// Nucleus sampling parameter; must be finite and in `[0, 1]`, enforced
    /// symmetrically on the decode and encode paths.
    #[serde(
        default,
        serialize_with = "serialize_finite_f64",
        deserialize_with = "deserialize_top_p"
    )]
    pub top_p: Option<f64>,
    /// Maximum number of tokens to generate. Not range-checked at the core
    /// decode boundary; providers treat `<= 0` differently, so forward it.
    pub max_tokens: Option<i32>,
    /// Stop sequences; normalised at deserialization into `Vec<String>` — a
    /// bare OpenAI string becomes a single-element vec.
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
    /// Top-K sampling (Anthropic `top_k`, Gemini `topK`); OpenAI Chat has no
    /// equivalent, so the cross-format policy rejects it there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Reasoning effort level (e.g. `"low"`, `"medium"`, `"high"`).
    pub reasoning_effort: Option<String>,
    /// Extended thinking configuration (provider-specific JSON, e.g. Anthropic
    /// `{"type": "enabled", "budget_tokens": N}`); redacted in `Debug` output.
    pub thinking: Option<serde_json::Value>,
}

fn serialize_finite_f64<S>(val: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match val {
        Some(v) if !v.is_finite() => Err(serde::ser::Error::custom(format!(
            "non-finite float value ({v}); temperature and top_p must be finite numbers"
        ))),
        Some(v) => serializer.serialize_some(v),
        None => serializer.serialize_none(),
    }
}

fn deserialize_temperature<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<f64> = Option::deserialize(deserializer)?;
    match val {
        Some(v) if !v.is_finite() => Err(serde::de::Error::custom(format!(
            "non-finite temperature value ({v}); temperature must be a finite number"
        ))),
        Some(v) if v < 0.0 => Err(serde::de::Error::custom(format!(
            "temperature must be >= 0, got {v}"
        ))),
        other => Ok(other),
    }
}

fn deserialize_top_p<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<f64> = Option::deserialize(deserializer)?;
    match val {
        Some(v) if !v.is_finite() => Err(serde::de::Error::custom(format!(
            "non-finite top_p value ({v}); top_p must be a finite number"
        ))),
        Some(v) if !(0.0..=1.0).contains(&v) => Err(serde::de::Error::custom(format!(
            "top_p must be in [0, 1], got {v}"
        ))),
        other => Ok(other),
    }
}

fn deserialize_input_schema_object<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(deserializer)?;
    if !v.is_object() {
        return Err(serde::de::Error::custom(
            "input_schema must be a JSON object",
        ));
    }
    Ok(v)
}

fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    let val = Option::<serde_json::Value>::deserialize(deserializer)?;
    match val {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(vec![s])),
        Some(serde_json::Value::Array(arr)) => {
            let mut result = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    serde_json::Value::String(s) => result.push(s),
                    other => {
                        return Err(de::Error::custom(format!(
                            "stop array must contain only strings, found {}",
                            json_type_name(&other)
                        )));
                    }
                }
            }
            if result.is_empty() {
                Ok(None)
            } else {
                Ok(Some(result))
            }
        }
        Some(other) => Err(de::Error::custom(format!(
            "stop must be a string, array of strings, or null, found {}",
            json_type_name(&other)
        ))),
    }
}

pub(crate) fn json_type_name(val: &serde_json::Value) -> &'static str {
    match val {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

impl fmt::Debug for SamplingOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SamplingOptions")
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("max_tokens", &self.max_tokens)
            .field("stop", &self.stop)
            .field("top_k", &self.top_k)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("thinking", &self.thinking.as_ref().map(OpaqueJsonRef))
            .finish()
    }
}

/// Caller metadata attached to a request.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestMetadata {
    /// Optional user identifier.
    pub user_id: Option<String>,
    /// Arbitrary key-value pairs for opaque data.
    pub raw: serde_json::Map<String, serde_json::Value>,
}

impl fmt::Debug for RequestMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestMetadata")
            .field("user_id", &self.user_id)
            .field("raw", &redact_map_helper(&self.raw))
            .finish()
    }
}

struct RedactedMap<'a>(&'a serde_json::Map<String, serde_json::Value>);

fn redact_map_helper(map: &serde_json::Map<String, serde_json::Value>) -> RedactedMap<'_> {
    RedactedMap(map)
}

impl fmt::Debug for RedactedMap<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        redact_map(self.0, f)
    }
}

/// Opaque hints consumed only by provider adapters (e.g. OpenAI
/// `stream_options`). No adapter may special-case another protocol's hints;
/// unknown keys must be silently ignored.
#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderHints {
    /// Arbitrary key-value pairs.
    pub raw: serde_json::Map<String, serde_json::Value>,
}

impl fmt::Debug for ProviderHints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderHints")
            .field("raw", &redact_map_helper(&self.raw))
            .finish()
    }
}

/// A normalized non-streaming chat response.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreResponse {
    /// Response identifier (echoed from the provider when available).
    pub id: Option<String>,
    /// Model identifier for this response.
    pub model: ModelRef,
    /// Content blocks in the response.
    pub content: Vec<CoreContent>,
    /// Why the model stopped generating.
    pub stop_reason: StopReason,
    /// The stop sequence that caused generation to stop, if any.
    pub stop_sequence: Option<String>,
    /// Token usage statistics.
    pub usage: Usage,
    /// Provider-specific metadata; a `Map` (not bare `Value`) so it serializes
    /// as `{}` when empty rather than `null`.
    #[serde(default)]
    pub provider_meta: serde_json::Map<String, serde_json::Value>,
    /// Computed USD cost, `None` (and omitted from serialization) when no
    /// pricing is configured. Not on the Anthropic/OpenAI client wire;
    /// available to the event log and API crate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
}

impl fmt::Debug for CoreResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreResponse")
            .field("id", &self.id)
            .field("model", &self.model)
            .field("content", &self.content)
            .field("stop_reason", &self.stop_reason)
            .field("stop_sequence", &self.stop_sequence)
            .field("usage", &self.usage)
            .field("provider_meta", &redact_map_helper(&self.provider_meta))
            .field("cost", &self.cost)
            .finish()
    }
}

/// Computed cost for one request, in USD, derived from [`Usage`] and a
/// per-token price table; `Decimal` throughout to avoid floating-point money.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    /// Input-token cost.
    pub input: Decimal,
    /// Output-token cost.
    pub output: Decimal,
    /// Cache-creation-token cost.
    pub cache_creation: Decimal,
    /// Cache-read-token cost.
    pub cache_read: Decimal,
    /// Reasoning-token cost.
    pub reasoning: Decimal,
    /// Total cost = input + output + cache_creation + cache_read + reasoning.
    pub total: Decimal,
}

/// Why the model stopped generating.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// The model finished its turn naturally.
    EndTurn,
    /// The maximum token count was reached.
    MaxTokens,
    /// The model stopped to invoke a tool.
    ToolUse,
    /// A stop sequence was matched.
    StopSequence,
    /// The model refused to answer.
    Refusal,
    /// An error occurred.
    Error,
    /// The stop reason was not recognised by the adapter's finish-reason map.
    /// The original provider value is carried so it survives into diagnostics;
    /// the sentinel `"absent"` means the provider omitted the field entirely.
    Unknown(String),
}

/// Token usage statistics; must carry [`UsageProvenance`] — never report
/// fabricated or zero-filled usage as `ProviderReported`. Counts are `i32`,
/// matching provider wire formats.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub input_tokens: i32,
    /// Number of tokens in the completion.
    pub output_tokens: i32,
    /// Reasoning token count (OpenAI/Responses-style).
    pub reasoning_tokens: Option<i32>,
    /// Tokens written to the cache during this request.
    pub cache_creation_input_tokens: Option<i32>,
    /// Tokens read from the cache during this request.
    pub cache_read_input_tokens: Option<i32>,
    /// Where the usage numbers came from.
    pub provenance: UsageProvenance,
}

impl Usage {
    /// Creates a synthetic-zero usage; provenance is
    /// [`UsageProvenance::SyntheticZero`], distinguishing it from provider zeros.
    pub fn synthetic_zero() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            provenance: UsageProvenance::SyntheticZero,
            ..Default::default()
        }
    }

    /// Creates a provider-reported usage; callers must ensure the numbers
    /// actually came from the provider.
    pub fn provider_reported(input_tokens: i32, output_tokens: i32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            provenance: UsageProvenance::ProviderReported,
            ..Default::default()
        }
    }
}

/// Provenance of token usage numbers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UsageProvenance {
    /// The provider reported these numbers.
    ProviderReported,
    /// The proxy synthesised zero values because the provider did not report.
    SyntheticZero,
    /// Provenance is unknown.
    #[default]
    Unknown,
}

/// A single normalised streaming event. V1 streams only text, thinking, and
/// tool-use incrementally; image/document/audio/video have no per-delta
/// event and travel via the non-stream [`CoreResponse`] path or are rejected.
#[non_exhaustive]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CoreEvent {
    /// The response has started.
    MessageStart {
        /// Response identifier.
        id: Option<String>,
        /// Model being used.
        model: ModelRef,
    },
    /// A new content block has started.
    ContentStart {
        /// Index of the content block.
        index: usize,
        /// Kind of content block.
        kind: ContentKind,
    },
    /// Incremental text content.
    TextDelta {
        /// Index of the content block.
        index: usize,
        /// The text delta.
        text: String,
    },
    /// Incremental thinking content.
    ThinkingDelta {
        /// Index of the content block.
        index: usize,
        /// The thinking text delta.
        text: String,
    },
    /// A tool call has started.
    ToolCallStart {
        /// Index of the content block.
        index: usize,
        /// Unique tool-call identifier.
        id: String,
        /// Name of the tool being called.
        name: String,
        /// Provider signature accompanying the call (e.g. a Gemini thought
        /// signature), which clients must replay with the call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Incremental tool-call arguments.
    ToolCallDelta {
        /// Index of the content block.
        index: usize,
        /// Delta of the JSON arguments string.
        args_delta: String,
    },
    /// A tool call has finished.
    ToolCallStop {
        /// Index of the content block.
        index: usize,
    },
    /// Usage information update.
    UsageDelta {
        /// Token usage so far.
        usage: Usage,
    },
    /// The response has finished.
    MessageStop {
        /// Why the model stopped generating.
        stop_reason: StopReason,
        /// The stop sequence that caused generation to stop, if any.
        stop_sequence: Option<String>,
    },
    /// An error occurred during streaming.
    Error {
        /// The stream error details.
        error: CoreStreamError,
    },
    /// Keep-alive / heartbeat.
    Ping,
}

impl fmt::Debug for CoreEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreEvent::MessageStart { id, model } => f
                .debug_struct("MessageStart")
                .field("id", &id)
                .field("model", &model)
                .finish(),
            CoreEvent::ContentStart { index, kind } => f
                .debug_struct("ContentStart")
                .field("index", &index)
                .field("kind", &kind)
                .finish(),
            CoreEvent::TextDelta { index, text } => f
                .debug_struct("TextDelta")
                .field("index", &index)
                .field("text", &text)
                .finish(),
            CoreEvent::ThinkingDelta { index, text } => f
                .debug_struct("ThinkingDelta")
                .field("index", &index)
                .field("text", &text)
                .finish(),
            CoreEvent::ToolCallStart {
                index,
                id,
                name,
                signature,
            } => f
                .debug_struct("ToolCallStart")
                .field("index", &index)
                .field("id", &id)
                .field("name", &name)
                .field("signature", &signature.as_ref().map(|_| "[REDACTED]"))
                .finish(),
            CoreEvent::ToolCallDelta { index, args_delta } => f
                .debug_struct("ToolCallDelta")
                .field("index", &index)
                .field("args_delta", &args_delta)
                .finish(),
            CoreEvent::ToolCallStop { index } => f
                .debug_struct("ToolCallStop")
                .field("index", &index)
                .finish(),
            CoreEvent::UsageDelta { usage } => {
                f.debug_struct("UsageDelta").field("usage", &usage).finish()
            }
            CoreEvent::MessageStop {
                stop_reason,
                stop_sequence,
            } => f
                .debug_struct("MessageStop")
                .field("stop_reason", &stop_reason)
                .field("stop_sequence", &stop_sequence)
                .finish(),
            CoreEvent::Error { error } => f.debug_struct("Error").field("error", &error).finish(),
            CoreEvent::Ping => f.write_str("Ping"),
        }
    }
}

/// Discriminator for content block types.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentKind {
    /// Plain text.
    Text,
    /// Extended thinking.
    Thinking,
    /// Tool invocation.
    ToolUse,
    /// Tool result. Has no streaming delta event; used only for `ContentStart`
    /// signalling — actual content travels through the non-stream
    /// [`CoreResponse`] content path.
    ToolResult,
    /// Image.
    Image,
    /// Document / file.
    Document,
    /// Audio.
    Audio,
    /// Video.
    Video,
    /// Model refusal.
    Refusal,
}

/// An error that occurred during streaming. The `message` is `pub(crate)`
/// and must be sanitized (strip API keys/tokens) before
/// [`CoreStreamError::new`]; `Debug`/`Display` redact it — never serialize it to clients.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoreStreamError {
    /// Category of the error.
    pub kind: CoreStreamErrorKind,
    #[serde(skip_serializing, default)]
    pub(crate) message: String,
}

impl CoreStreamError {
    /// Constructs a new `CoreStreamError`; callers **must** sanitize `message`
    /// (strip API keys, bearer tokens, and other secrets) before passing it.
    pub fn new(kind: CoreStreamErrorKind, message: String) -> Self {
        Self { kind, message }
    }

    /// Returns the error message. Because the field is
    /// `#[serde(skip_serializing)]`, instances rebuilt via `Deserialize`
    /// return `""`; only [`new`](CoreStreamError::new) instances carry it.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Debug for CoreStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreStreamError")
            .field("kind", &self.kind)
            .field("message", &format_args!("{} chars", self.message.len()))
            .finish()
    }
}

impl fmt::Display for CoreStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for CoreStreamError {}

/// Category of a stream error; use [`CoreStreamErrorKind::http_status`]
/// for the canonical HTTP mapping rather than adapter-specific codes.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreStreamErrorKind {
    /// The request was malformed or invalid.
    InvalidRequest,
    /// Authentication failed.
    Authentication,
    /// The caller lacks permission.
    Permission,
    /// A rate limit was exceeded.
    RateLimit,
    /// The upstream provider returned an error.
    Upstream,
    /// An internal proxy error.
    Internal,
}

impl CoreStreamErrorKind {
    /// Returns the canonical HTTP status code for this error kind.
    pub fn http_status(&self) -> u16 {
        match self {
            Self::InvalidRequest => 400,
            Self::Authentication => 401,
            Self::Permission => 403,
            Self::RateLimit => 429,
            Self::Upstream => 502,
            Self::Internal => 500,
        }
    }
}

impl fmt::Display for CoreStreamErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => f.write_str("invalid_request"),
            Self::Authentication => f.write_str("authentication"),
            Self::Permission => f.write_str("permission"),
            Self::RateLimit => f.write_str("rate_limit"),
            Self::Upstream => f.write_str("upstream"),
            Self::Internal => f.write_str("internal"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ref_provider_model_uses_requested_without_override() {
        let mr = ModelRef {
            requested: "gpt-4o".into(),
            upstream: None,
        };
        assert_eq!(mr.provider_model(), "gpt-4o");
    }

    #[test]
    fn model_ref_provider_model_uses_upstream_override() {
        let mr = ModelRef {
            requested: "my-alias".into(),
            upstream: Some("gpt-4o-2024-08-06".into()),
        };
        assert_eq!(mr.provider_model(), "gpt-4o-2024-08-06");
    }

    #[test]
    fn sampling_options_default_has_no_overrides() {
        let opts = SamplingOptions::default();
        assert!(opts.temperature.is_none());
        assert!(opts.top_p.is_none());
        assert!(opts.max_tokens.is_none());
        assert!(opts.stop.is_none());
        assert!(opts.reasoning_effort.is_none());
        assert!(opts.thinking.is_none());
    }

    #[test]
    fn usage_default_is_zero() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert!(usage.reasoning_tokens.is_none());
        assert!(usage.cache_creation_input_tokens.is_none());
        assert!(usage.cache_read_input_tokens.is_none());
    }

    #[test]
    fn core_content_tool_result_can_nest_text() {
        let result = CoreContent::ToolResult {
            tool_use_id: "call_123".into(),
            content: vec![CoreContent::Text {
                text: "it worked".into(),
                cache: None,
            }],
            is_error: false,
            cache: None,
        };
        match result {
            CoreContent::ToolResult {
                content,
                is_error,
                tool_use_id,
                ..
            } => {
                assert!(!is_error);
                assert_eq!(tool_use_id, "call_123");
                assert_eq!(content.len(), 1);
                if let CoreContent::Text { text, .. } = &content[0] {
                    assert_eq!(text, "it worked");
                } else {
                    panic!("expected Text variant");
                }
            }
            _ => panic!("expected ToolResult variant"),
        }
    }

    #[test]
    fn core_response_debug_includes_cost_when_present() {
        use std::str::FromStr;

        let resp = CoreResponse {
            id: Some("resp_cost".into()),
            model: ModelRef {
                requested: "claude-sonnet-4-20250514".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: Some(Cost {
                total: Decimal::from_str("0.003").unwrap(),
                ..Cost::default()
            }),
        };
        let debug = format!("{resp:?}");
        assert!(
            debug.contains("cost:"),
            "CoreResponse Debug must include the `cost` field, got: {debug}"
        );
        assert!(
            debug.contains("0.003"),
            "CoreResponse Debug must surface the total cost value, got: {debug}"
        );
    }

    #[test]
    fn core_response_debug_sensible_when_cost_absent() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "gpt-4o".into(),
                upstream: None,
            },
            content: Vec::new(),
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let debug = format!("{resp:?}");
        assert!(
            debug.contains("cost: None"),
            "CoreResponse Debug must render `cost: None` when cost is absent, got: {debug}"
        );
    }

    #[test]
    fn core_response_can_preserve_stop_sequence() {
        let resp = CoreResponse {
            id: Some("resp_abc".into()),
            model: ModelRef {
                requested: "claude-sonnet-4-20250514".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
            stop_reason: StopReason::StopSequence,
            stop_sequence: Some("\n".into()),
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        assert_eq!(resp.stop_reason, StopReason::StopSequence);
        assert_eq!(resp.stop_sequence.as_deref(), Some("\n"));
    }

    #[test]
    fn message_stop_event_can_preserve_stop_sequence() {
        let event = CoreEvent::MessageStop {
            stop_reason: StopReason::StopSequence,
            stop_sequence: Some("END".into()),
        };
        match event {
            CoreEvent::MessageStop {
                stop_reason,
                stop_sequence,
            } => {
                assert_eq!(stop_reason, StopReason::StopSequence);
                assert_eq!(stop_sequence.as_deref(), Some("END"));
            }
            _ => panic!("expected MessageStop variant"),
        }
    }

    #[test]
    fn core_request_preserves_client_intent() {
        let req = CoreRequest {
            model: ModelRef {
                requested: "my-model".into(),
                upstream: None,
            },
            system: vec![CoreContent::Text {
                text: "you are helpful".into(),
                cache: None,
            }],
            messages: vec![CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: "hello".into(),
                    cache: None,
                }],
            }],
            tools: vec![CoreTool {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            }],
            tool_choice: Some(CoreToolChoice::Auto),
            sampling: SamplingOptions {
                temperature: Some(0.7),
                ..Default::default()
            },
            stream: false,
            metadata: RequestMetadata::default(),
            provider_hints: ProviderHints::default(),
        };
        assert_eq!(req.model.requested, "my-model");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.tools.len(), 1);
        assert!(!req.stream);
    }

    #[test]
    fn core_request_preserves_metadata_and_provider_hints() {
        let mut raw_meta = serde_json::Map::new();
        raw_meta.insert(
            "trace_id".into(),
            serde_json::Value::String("abc-123".into()),
        );
        let mut raw_hints = serde_json::Map::new();
        raw_hints.insert(
            "prefer_region".into(),
            serde_json::Value::String("us-east".into()),
        );

        let req = CoreRequest {
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            system: vec![],
            messages: vec![],
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: RequestMetadata {
                user_id: Some("user-42".into()),
                raw: raw_meta,
            },
            provider_hints: ProviderHints { raw: raw_hints },
        };
        assert_eq!(req.metadata.user_id.as_deref(), Some("user-42"));
        assert_eq!(
            req.metadata.raw.get("trace_id").unwrap().as_str(),
            Some("abc-123")
        );
        assert_eq!(
            req.provider_hints
                .raw
                .get("prefer_region")
                .unwrap()
                .as_str(),
            Some("us-east")
        );
    }

    #[test]
    fn core_request_system_field_is_canonical() {
        let system = vec![
            CoreContent::Text {
                text: "system prompt".into(),
                cache: None,
            },
            CoreContent::Text {
                text: "extra instruction".into(),
                cache: Some(CacheControl {
                    r#type: CacheControlType::Ephemeral,
                    ttl: None,
                }),
            },
        ];
        let req = CoreRequest {
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            system,
            messages: vec![CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: "hi".into(),
                    cache: None,
                }],
            }],
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: RequestMetadata::default(),
            provider_hints: ProviderHints::default(),
        };
        assert_eq!(req.system.len(), 2);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, CoreRole::User);
    }

    #[test]
    fn core_request_round_trips_through_json() {
        let req = CoreRequest {
            model: ModelRef {
                requested: "model-a".into(),
                upstream: Some("model-b".into()),
            },
            system: vec![CoreContent::Text {
                text: "sys".into(),
                cache: None,
            }],
            messages: vec![CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: "hi".into(),
                    cache: None,
                }],
            }],
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions {
                temperature: Some(0.5),
                max_tokens: Some(100),
                ..Default::default()
            },
            stream: true,
            metadata: RequestMetadata::default(),
            provider_hints: ProviderHints::default(),
        };
        let json = serde_json::to_string(&req).expect("serialize CoreRequest");
        let back: CoreRequest = serde_json::from_str(&json).expect("deserialize CoreRequest");
        assert_eq!(req, back);
    }

    #[test]
    fn core_response_round_trips_through_json() {
        let resp = CoreResponse {
            id: Some("resp_1".into()),
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Text {
                    text: "hello".into(),
                    cache: None,
                },
                CoreContent::ToolUse {
                    id: "call_1".into(),
                    name: "fn".into(),
                    input: serde_json::json!({"x": 1}),
                    signature: None,
                },
            ],
            stop_reason: StopReason::ToolUse,
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
                reasoning_tokens: Some(5),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: Some(3),
                provenance: UsageProvenance::ProviderReported,
            },
            provider_meta: serde_json::from_str(r#"{"warnings":[]}"#).expect("parse provider_meta"),
            cost: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize CoreResponse");
        let back: CoreResponse = serde_json::from_str(&json).expect("deserialize CoreResponse");
        assert_eq!(resp, back);
    }

    #[test]
    fn core_event_round_trips_through_json() {
        let events: Vec<CoreEvent> = vec![
            CoreEvent::MessageStart {
                id: Some("msg_1".into()),
                model: ModelRef {
                    requested: "m".into(),
                    upstream: None,
                },
            },
            CoreEvent::ContentStart {
                index: 0,
                kind: ContentKind::Text,
            },
            CoreEvent::ContentStart {
                index: 1,
                kind: ContentKind::Thinking,
            },
            CoreEvent::TextDelta {
                index: 0,
                text: "hello ".into(),
            },
            CoreEvent::TextDelta {
                index: 0,
                text: "world".into(),
            },
            CoreEvent::ThinkingDelta {
                index: 1,
                text: "let me think".into(),
            },
            CoreEvent::ToolCallStart {
                index: 2,
                id: "call_1".into(),
                name: "get_weather".into(),
                signature: Some("sig_rt".into()),
            },
            CoreEvent::ToolCallDelta {
                index: 2,
                args_delta: "{\"ci".into(),
            },
            CoreEvent::ToolCallDelta {
                index: 2,
                args_delta: "ty\": \"SF\"}".into(),
            },
            CoreEvent::ToolCallStop { index: 2 },
            CoreEvent::UsageDelta {
                usage: Usage {
                    input_tokens: 50,
                    output_tokens: 100,
                    reasoning_tokens: None,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                    provenance: UsageProvenance::ProviderReported,
                },
            },
            CoreEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
                stop_sequence: None,
            },
            CoreEvent::Ping,
            CoreEvent::Error {
                error: CoreStreamError::new(
                    CoreStreamErrorKind::RateLimit,
                    "too many requests".into(),
                ),
            },
        ];

        let json = serde_json::to_string(&events).expect("serialize CoreEvent vec");
        let back: Vec<CoreEvent> = serde_json::from_str(&json).expect("deserialize CoreEvent vec");
        for (i, (original, round_tripped)) in events.iter().zip(back.iter()).enumerate() {
            match (original, round_tripped) {
                (CoreEvent::Error { error: orig }, CoreEvent::Error { error: rt }) => {
                    assert_eq!(orig.kind, rt.kind, "kind mismatch at event {i}");
                    assert!(
                        rt.message.is_empty(),
                        "message should be empty after round-trip at event {i}"
                    );
                }
                _ => assert_eq!(original, round_tripped, "mismatch at event {i}"),
            }
        }
    }

    #[test]
    fn core_tool_choice_raw_round_trips_when_intentional() {
        let raw_val = serde_json::json!({"type": "some_custom", "mode": "strict"});
        let json = serde_json::to_string(&raw_val).unwrap();
        let choice = CoreToolChoice::Raw(raw_val);
        let serialized = serde_json::to_string(&choice).expect("serialize CoreToolChoice");
        let back: CoreToolChoice =
            serde_json::from_str(&serialized).expect("deserialize CoreToolChoice");
        match back {
            CoreToolChoice::Raw(v) => {
                let original: serde_json::Value = serde_json::from_str(&json).unwrap();
                assert_eq!(v, original);
            }
            _ => panic!("expected Raw variant"),
        }
    }

    #[test]
    fn usage_default_provenance_is_unknown() {
        let usage = Usage::default();
        assert_eq!(usage.provenance, UsageProvenance::Unknown);
    }

    #[test]
    fn usage_preserves_reasoning_tokens() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 200,
            reasoning_tokens: Some(42),
            cache_creation_input_tokens: Some(10),
            cache_read_input_tokens: Some(20),
            provenance: UsageProvenance::ProviderReported,
        };
        let json = serde_json::to_string(&usage).expect("serialize Usage");
        let back: Usage = serde_json::from_str(&json).expect("deserialize Usage");
        assert_eq!(back.reasoning_tokens, Some(42));
        assert_eq!(back.cache_creation_input_tokens, Some(10));
        assert_eq!(back.cache_read_input_tokens, Some(20));
        assert_eq!(back.provenance, UsageProvenance::ProviderReported);
    }

    #[test]
    fn cache_control_type_known_and_unknown_variants() {
        let json = serde_json::to_string(&CacheControlType::Ephemeral).unwrap();
        assert_eq!(json, "\"ephemeral\"");
        let back: CacheControlType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, CacheControlType::Ephemeral);
        assert_eq!(String::from(back), "ephemeral");

        let other = CacheControlType::Other("future_type".to_owned());
        let json = serde_json::to_string(&other).unwrap();
        let back: CacheControlType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, CacheControlType::Other("future_type".to_owned()));
        assert_eq!(String::from(other), "future_type");
    }

    #[test]
    fn cache_control_type_as_str() {
        assert_eq!(CacheControlType::Ephemeral.as_str(), "ephemeral");
        assert_eq!(
            CacheControlType::Other("custom".to_owned()).as_str(),
            "custom"
        );
    }

    #[test]
    fn model_ref_equality_semantics() {
        let a = ModelRef {
            requested: "gpt-4o".into(),
            upstream: Some("gpt-4o-2024-08-06".into()),
        };
        let b = ModelRef {
            requested: "gpt-4o".into(),
            upstream: Some("gpt-4o-2024-08-06".into()),
        };
        assert_eq!(a, b);

        let c = ModelRef {
            requested: "gpt-4o".into(),
            upstream: None,
        };
        assert_ne!(a, c);

        let d = ModelRef {
            requested: "claude".into(),
            upstream: Some("gpt-4o-2024-08-06".into()),
        };
        assert_ne!(a, d);
    }

    #[test]
    fn core_content_handles_empty_strings() {
        let variants = vec![
            CoreContent::Text {
                text: String::new(),
                cache: None,
            },
            CoreContent::ToolUse {
                id: String::new(),
                name: String::new(),
                input: serde_json::Value::Null,
                signature: None,
            },
            CoreContent::ToolResult {
                tool_use_id: String::new(),
                content: vec![],
                is_error: false,
                cache: None,
            },
            CoreContent::Thinking {
                text: String::new(),
                signature: Some(String::new()),
            },
            CoreContent::Refusal {
                text: String::new(),
            },
        ];
        for content in &variants {
            let json = serde_json::to_string(content).expect("serialize");
            let back: CoreContent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*content, back);
        }
    }

    #[test]
    fn core_request_rejects_malformed_json() {
        let cases = vec!["not json at all", "{", r#"{"model": 123}"#];
        for bad in cases {
            assert!(
                serde_json::from_str::<CoreRequest>(bad).is_err(),
                "expected error for: {bad}"
            );
        }
    }

    #[test]
    fn core_response_rejects_malformed_json() {
        let cases = vec!["not json at all", "{", r#"{"id": 123}"#];
        for bad in cases {
            assert!(
                serde_json::from_str::<CoreResponse>(bad).is_err(),
                "expected error for: {bad}"
            );
        }
    }

    #[test]
    fn core_event_rejects_malformed_json() {
        let cases = vec!["not json at all", "[", r#"{"TextDelta": "wrong"}"#];
        for bad in cases {
            assert!(
                serde_json::from_str::<CoreEvent>(bad).is_err(),
                "expected error for: {bad}"
            );
        }
    }

    #[test]
    fn sampling_options_handles_boundary_values() {
        let opts = SamplingOptions {
            temperature: Some(0.0),
            top_p: Some(0.0),
            max_tokens: Some(0),
            stop: None,
            top_k: None,
            reasoning_effort: None,
            thinking: None,
        };
        let json = serde_json::to_string(&opts).expect("serialize boundary opts");
        let back: SamplingOptions = serde_json::from_str(&json).expect("deserialize boundary opts");
        assert_eq!(back.temperature, Some(0.0));
        assert_eq!(back.max_tokens, Some(0));

        let opts_max = SamplingOptions {
            max_tokens: Some(i32::MAX),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts_max).expect("serialize max opts");
        let back: SamplingOptions = serde_json::from_str(&json).expect("deserialize max opts");
        assert_eq!(back.max_tokens, Some(i32::MAX));
    }

    #[test]
    fn usage_handles_i32_max() {
        let usage = Usage {
            input_tokens: i32::MAX,
            output_tokens: i32::MAX,
            reasoning_tokens: Some(i32::MAX),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            provenance: UsageProvenance::ProviderReported,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, usage);
    }

    #[test]
    fn core_response_handles_missing_optional_fields() {
        let json = r#"{
            "model": {"requested": "m"},
            "content": [],
            "stop_reason": "EndTurn",
            "usage": {"input_tokens": 0, "output_tokens": 0, "provenance": "Unknown"}
        }"#;
        let resp: CoreResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.id, None);
        assert!(resp.model.upstream.is_none());
        assert_eq!(resp.stop_sequence, None);
        assert_eq!(resp.usage.input_tokens, 0);
        assert_eq!(resp.usage.provenance, UsageProvenance::Unknown);
        assert!(resp.provider_meta.is_empty());
    }

    #[test]
    fn core_request_rejects_wrong_field_types() {
        let cases = vec![
            r#"{"model": {"requested": 123}, "messages": [], "system": [], "tools": [], "sampling": {}, "stream": false, "metadata": {}, "provider_hints": {}}"#,
            r#"{"model": {"requested": "m"}, "messages": [], "system": [], "tools": [], "sampling": {}, "stream": "yes", "metadata": {}, "provider_hints": {}}"#,
        ];
        for bad in cases {
            assert!(
                serde_json::from_str::<CoreRequest>(bad).is_err(),
                "expected error for: {bad}"
            );
        }
    }

    #[test]
    fn core_request_handles_empty_collections() {
        let req = CoreRequest {
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            system: vec![],
            messages: vec![],
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: RequestMetadata::default(),
            provider_hints: ProviderHints::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
        assert!(back.system.is_empty());
        assert!(back.messages.is_empty());
        assert!(back.tools.is_empty());
    }

    #[test]
    fn tool_result_with_is_error_true_round_trips() {
        let result = CoreContent::ToolResult {
            tool_use_id: "call_err".into(),
            content: vec![CoreContent::Text {
                text: "something went wrong".into(),
                cache: None,
            }],
            is_error: true,
            cache: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CoreContent = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn tool_result_with_empty_content_round_trips() {
        let result = CoreContent::ToolResult {
            tool_use_id: "call_empty".into(),
            content: vec![],
            is_error: true,
            cache: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: CoreContent = serde_json::from_str(&json).unwrap();
        assert_eq!(result, back);
    }

    #[test]
    fn nested_tool_result_round_trips() {
        let inner = CoreContent::ToolResult {
            tool_use_id: "inner".into(),
            content: vec![CoreContent::Text {
                text: "nested".into(),
                cache: None,
            }],
            is_error: false,
            cache: None,
        };
        let outer = CoreContent::ToolResult {
            tool_use_id: "outer".into(),
            content: vec![inner],
            is_error: false,
            cache: None,
        };
        let json = serde_json::to_string(&outer).unwrap();
        let back: CoreContent = serde_json::from_str(&json).unwrap();
        assert_eq!(outer, back);
    }

    #[test]
    fn all_core_content_variants_round_trip() {
        let contents: Vec<CoreContent> = vec![
            CoreContent::Text {
                text: "hello".into(),
                cache: None,
            },
            CoreContent::Image {
                source: serde_json::json!({"url": "https://example.com/img.png"}),
            },
            CoreContent::Document {
                source: serde_json::json!({"url": "https://example.com/doc.pdf"}),
            },
            CoreContent::Audio {
                source: serde_json::json!({"data": "base64..."}),
            },
            CoreContent::Video {
                source: serde_json::json!({"url": "https://example.com/vid.mp4"}),
            },
            CoreContent::ToolUse {
                id: "call_1".into(),
                name: "fn".into(),
                input: serde_json::json!({"x": 1}),
                signature: None,
            },
            CoreContent::ToolResult {
                tool_use_id: "call_1".into(),
                content: vec![CoreContent::Text {
                    text: "result".into(),
                    cache: None,
                }],
                is_error: false,
                cache: None,
            },
            CoreContent::Thinking {
                text: "hmm".into(),
                signature: Some("sig123".into()),
            },
            CoreContent::RedactedThinking {
                data: serde_json::json!({"redacted": true}),
            },
            CoreContent::Refusal {
                text: "I cannot".into(),
            },
            CoreContent::Opaque {
                block: serde_json::json!({
                    "type": "server_tool_use", "id": "srvtoolu_1",
                    "name": "web_search", "input": {"query": "q"}
                }),
            },
        ];
        for content in &contents {
            let json = serde_json::to_string(content).unwrap();
            let back: CoreContent = serde_json::from_str(&json).unwrap();
            assert_eq!(*content, back, "round-trip failed for variant");
        }
    }

    #[test]
    fn all_stop_reason_variants_round_trip() {
        let variants = vec![
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::ToolUse,
            StopReason::StopSequence,
            StopReason::Refusal,
            StopReason::Error,
            StopReason::Unknown("provider_specific".to_owned()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(&variant).unwrap();
            let back: StopReason = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, back);
        }
    }

    #[test]
    fn all_core_stream_error_kind_variants_round_trip() {
        let variants = vec![
            CoreStreamErrorKind::InvalidRequest,
            CoreStreamErrorKind::Authentication,
            CoreStreamErrorKind::Permission,
            CoreStreamErrorKind::RateLimit,
            CoreStreamErrorKind::Upstream,
            CoreStreamErrorKind::Internal,
        ];
        for kind in &variants {
            let err = CoreStreamError::new(*kind, format!("test error for {kind:?}"));
            let json = serde_json::to_string(&err).expect("serialize CoreStreamError");
            let back: CoreStreamError =
                serde_json::from_str(&json).expect("deserialize CoreStreamError");
            assert_eq!(err.kind, back.kind);
            assert!(back.message.is_empty());
        }
    }

    #[test]
    fn all_core_tool_choice_variants_round_trip() {
        let variants = vec![
            CoreToolChoice::Auto,
            CoreToolChoice::Any,
            CoreToolChoice::None,
            CoreToolChoice::Tool {
                name: "get_weather".into(),
            },
            CoreToolChoice::Raw(serde_json::json!({"custom": true})),
        ];
        for variant in &variants {
            let json = serde_json::to_string(&variant).unwrap();
            let back: CoreToolChoice = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, back);
        }
    }

    #[test]
    fn all_core_role_variants_round_trip() {
        let roles = vec![
            CoreRole::System,
            CoreRole::User,
            CoreRole::Assistant,
            CoreRole::Tool,
        ];
        for role in &roles {
            let msg = CoreMessage {
                role: *role,
                content: vec![CoreContent::Text {
                    text: format!("message as {role:?}"),
                    cache: None,
                }],
            };
            let json = serde_json::to_string(&msg).unwrap();
            let back: CoreMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg, back);
        }
    }

    #[test]
    fn all_content_kind_variants_round_trip() {
        let kinds = vec![
            ContentKind::Text,
            ContentKind::Thinking,
            ContentKind::ToolUse,
            ContentKind::ToolResult,
            ContentKind::Image,
            ContentKind::Document,
            ContentKind::Audio,
            ContentKind::Video,
            ContentKind::Refusal,
        ];
        for kind in &kinds {
            let event = CoreEvent::ContentStart {
                index: 0,
                kind: *kind,
            };
            let json = serde_json::to_string(&event).unwrap();
            let back: CoreEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event, back);
        }
    }

    #[test]
    fn provider_meta_with_complex_nested_json() {
        let mut meta = serde_json::Map::new();
        meta.insert(
            "deep".into(),
            serde_json::json!({
                "a": {"b": {"c": [1, 2, {"d": "value"}]}},
                "arr": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            }),
        );
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: meta,
            cost: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: CoreResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, back);
    }

    #[test]
    fn usage_provenance_synthetic_zero_round_trips() {
        let usage = Usage {
            input_tokens: 0,
            output_tokens: 0,
            reasoning_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            provenance: UsageProvenance::SyntheticZero,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let back: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provenance, UsageProvenance::SyntheticZero);
        assert_eq!(usage, back);
    }

    #[test]
    fn sampling_options_stop_as_vec_string_round_trips() {
        let opts = SamplingOptions {
            stop: Some(vec!["STOP".to_owned(), "END".to_owned()]),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: SamplingOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.stop, Some(vec!["STOP".to_owned(), "END".to_owned()]));
    }

    #[test]
    fn thinking_delta_event_round_trips() {
        let event = CoreEvent::ThinkingDelta {
            index: 5,
            text: "I am reasoning about this...".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: CoreEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, back);
    }

    #[test]
    fn content_start_thinking_event_round_trips() {
        let event = CoreEvent::ContentStart {
            index: 3,
            kind: ContentKind::Thinking,
        };
        let json = serde_json::to_string(&event).unwrap();
        let back: CoreEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, back);
    }

    #[test]
    fn core_request_round_trips_with_stop_sequences() {
        let req = CoreRequest {
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            system: vec![],
            messages: vec![],
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions {
                stop: Some(vec!["\n".to_owned(), "END".to_owned()]),
                ..Default::default()
            },
            stream: false,
            metadata: RequestMetadata::default(),
            provider_hints: ProviderHints::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn sampling_options_stop_deserializes_bare_string() {
        let json = r#"{"stop": "STOP"}"#;
        let opts: SamplingOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.stop, Some(vec!["STOP".to_owned()]));
    }

    #[test]
    fn sampling_options_stop_deserializes_null() {
        let json = r#"{"stop": null}"#;
        let opts: SamplingOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.stop, None);
    }

    #[test]
    fn sampling_options_stop_deserializes_array() {
        let json = r#"{"stop": ["STOP", "END"]}"#;
        let opts: SamplingOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.stop, Some(vec!["STOP".to_owned(), "END".to_owned()]));
    }

    #[test]
    fn sampling_options_stop_rejects_number() {
        let json = r#"{"stop": 42}"#;
        serde_json::from_str::<SamplingOptions>(json).unwrap_err();
    }

    #[test]
    fn sampling_options_rejects_out_of_range_sampling_values() {
        let bad_temp = r#"{"temperature": -0.5}"#;
        assert!(
            serde_json::from_str::<SamplingOptions>(bad_temp).is_err(),
            "negative temperature should be rejected at decode time"
        );
        let bad_top_p_low = r#"{"top_p": -0.1}"#;
        assert!(
            serde_json::from_str::<SamplingOptions>(bad_top_p_low).is_err(),
            "negative top_p should be rejected at decode time"
        );
        let bad_top_p_high = r#"{"top_p": 99.0}"#;
        assert!(
            serde_json::from_str::<SamplingOptions>(bad_top_p_high).is_err(),
            "top_p > 1 should be rejected at decode time"
        );
        let zero_max = r#"{"max_tokens": 0}"#;
        let opts: SamplingOptions =
            serde_json::from_str(zero_max).expect("max_tokens=0 must decode");
        assert_eq!(opts.max_tokens, Some(0));
    }

    #[test]
    fn sampling_options_nan_temperature_is_rejected_on_serialize_and_deserialize() {
        let opts = SamplingOptions {
            temperature: Some(f64::NAN),
            ..Default::default()
        };
        assert!(
            serde_json::to_string(&opts).is_err(),
            "serializing NaN temperature should error, not silently emit null"
        );
        let bad = r#"{"temperature": NaN}"#;
        assert!(
            serde_json::from_str::<SamplingOptions>(bad).is_err(),
            "NaN temperature should not decode"
        );
    }

    #[test]
    fn sampling_options_infinity_temperature_is_rejected_on_serialize() {
        let opts = SamplingOptions {
            temperature: Some(f64::INFINITY),
            ..Default::default()
        };
        assert!(
            serde_json::to_string(&opts).is_err(),
            "serializing Infinity temperature should error, not silently emit null"
        );
    }

    #[test]
    fn core_request_rejects_invalid_utf8_bytes() {
        let bad_bytes = b"{\"model\":{\"requested\":\"m\"},\"messages\":[],\"system\":[],\"tools\":[],\"sampling\":{},\"stream\":false,\"metadata\":{},\"provider_hints\":{},\"bad\xff_field\":1}";
        serde_json::from_slice::<CoreRequest>(bad_bytes).unwrap_err();
    }

    #[test]
    fn core_response_rejects_invalid_utf8_bytes() {
        let bad_bytes = b"{\"model\":{\"requested\":\"m\"},\"content\":[],\"stop_reason\":\"EndTurn\",\"usage\":{\"input_tokens\":0,\"output_tokens\":0},\"bad\xff_field\":1}";
        serde_json::from_slice::<CoreResponse>(bad_bytes).unwrap_err();
    }

    #[test]
    fn core_request_large_messages_round_trips() {
        let messages: Vec<CoreMessage> = (0..1000)
            .map(|i| CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: format!("message {i}"),
                    cache: None,
                }],
            })
            .collect();
        let req = CoreRequest {
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            system: vec![],
            messages,
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: RequestMetadata::default(),
            provider_hints: ProviderHints::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CoreRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.messages.len(), 1000);
    }

    #[test]
    fn deeply_nested_tool_result_round_trips() {
        let depth = 20;
        let mut content = CoreContent::Text {
            text: "leaf".into(),
            cache: None,
        };
        for i in (0..depth).rev() {
            content = CoreContent::ToolResult {
                tool_use_id: format!("level_{i}"),
                content: vec![content],
                is_error: false,
                cache: None,
            };
        }
        let json = serde_json::to_string(&content).unwrap();
        let back: CoreContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, back);
    }

    #[test]
    fn usage_synthetic_zero_constructor() {
        let usage = Usage::synthetic_zero();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.provenance, UsageProvenance::SyntheticZero);
        assert!(usage.reasoning_tokens.is_none());
    }

    #[test]
    fn usage_provider_reported_constructor() {
        let usage = Usage::provider_reported(100, 200);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 200);
        assert_eq!(usage.provenance, UsageProvenance::ProviderReported);
        assert!(usage.reasoning_tokens.is_none());
    }

    #[test]
    fn core_stream_error_display_and_error_traits() {
        let err = CoreStreamError::new(CoreStreamErrorKind::RateLimit, "too many requests".into());
        let display = format!("{err}");
        assert!(
            display.contains("rate_limit"),
            "Display should show kind: {display}"
        );
        assert!(
            !display.contains("too many requests"),
            "Display should not leak message"
        );

        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn core_stream_error_debug_redacts_message() {
        let err =
            CoreStreamError::new(CoreStreamErrorKind::Internal, "secret-api-key-12345".into());
        let debug = format!("{err:?}");
        assert!(
            !debug.contains("secret-api-key-12345"),
            "Debug should redact message content"
        );
        assert!(debug.contains("chars"), "Debug should show message length");
    }

    #[test]
    fn core_stream_error_kind_http_status_mapping() {
        assert_eq!(CoreStreamErrorKind::InvalidRequest.http_status(), 400);
        assert_eq!(CoreStreamErrorKind::Authentication.http_status(), 401);
        assert_eq!(CoreStreamErrorKind::Permission.http_status(), 403);
        assert_eq!(CoreStreamErrorKind::RateLimit.http_status(), 429);
        assert_eq!(CoreStreamErrorKind::Upstream.http_status(), 502);
        assert_eq!(CoreStreamErrorKind::Internal.http_status(), 500);
    }

    #[test]
    fn core_stream_error_kind_display() {
        assert_eq!(format!("{}", CoreStreamErrorKind::RateLimit), "rate_limit");
        assert_eq!(
            format!("{}", CoreStreamErrorKind::InvalidRequest),
            "invalid_request"
        );
        assert_eq!(format!("{}", CoreStreamErrorKind::Internal), "internal");
    }

    #[test]
    fn core_stream_error_message_accessor() {
        let err = CoreStreamError::new(
            CoreStreamErrorKind::Upstream,
            "provider error detail".into(),
        );
        assert_eq!(err.message(), "provider error detail");
        assert_eq!(err.kind, CoreStreamErrorKind::Upstream);
    }

    #[test]
    fn thinking_signature_redacted_in_debug() {
        let content = CoreContent::Thinking {
            text: "reasoning".into(),
            signature: Some("super-secret-sig".into()),
        };
        let debug = format!("{content:?}");
        assert!(
            !debug.contains("super-secret-sig"),
            "Debug should redact signature"
        );
        assert!(debug.contains("[REDACTED]"), "Debug should show [REDACTED]");
    }

    #[test]
    fn opaque_json_fields_redacted_in_debug() {
        let content = CoreContent::Image {
            source: serde_json::json!({"url": "secret-url"}),
        };
        let debug = format!("{content:?}");
        assert!(!debug.contains("secret-url"), "Debug should redact source");
        assert!(debug.contains("Object"), "Debug should show type");
    }

    #[test]
    fn opaque_block_debug_names_the_type_and_redacts_the_payload() {
        let content = CoreContent::Opaque {
            block: serde_json::json!({
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "encrypted_content": "secret-blob"}]
            }),
        };
        let debug = format!("{content:?}");
        assert!(
            debug.contains("web_search_tool_result"),
            "Debug should name the block type: {debug}"
        );
        assert!(
            !debug.contains("secret-blob") && !debug.contains("srvtoolu_1"),
            "Debug should redact the block payload: {debug}"
        );
    }

    #[test]
    fn provider_meta_redacted_in_debug() {
        let resp = CoreResponse {
            id: Some("r1".into()),
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::from_str(r#"{"secret":"value","count":42}"#).unwrap(),
            cost: None,
        };
        let debug = format!("{resp:?}");
        assert!(
            !debug.contains("secret"),
            "Debug should redact provider_meta contents"
        );
        assert!(
            !debug.contains("value"),
            "Debug should redact provider_meta contents"
        );
    }

    #[test]
    fn request_metadata_raw_redacted_in_debug() {
        let meta = RequestMetadata {
            user_id: Some("user-42".into()),
            raw: serde_json::from_str(r#"{"api_key":"sk-live-key-12345"}"#).unwrap(),
        };
        let debug = format!("{meta:?}");
        assert!(
            !debug.contains("sk-live-key-12345"),
            "Debug should redact raw map"
        );
    }

    #[test]
    fn provider_hints_raw_redacted_in_debug() {
        let hints = ProviderHints {
            raw: serde_json::from_str(r#"{"token":"bearer-abc123"}"#).unwrap(),
        };
        let debug = format!("{hints:?}");
        assert!(
            !debug.contains("bearer-abc123"),
            "Debug should redact raw map"
        );
    }

    #[test]
    fn sampling_options_thinking_redacted_in_debug() {
        let opts = SamplingOptions {
            thinking: Some(serde_json::json!({"budget_tokens": 9999})),
            ..Default::default()
        };
        let debug = format!("{opts:?}");
        assert!(
            !debug.contains("budget_tokens"),
            "Debug should redact thinking"
        );
    }

    #[test]
    fn core_tool_input_schema_redacted_in_debug() {
        let tool = CoreTool {
            name: "test".into(),
            description: Some("desc".into()),
            input_schema: serde_json::json!({"secret_field": "hidden"}),
        };
        let debug = format!("{tool:?}");
        assert!(
            !debug.contains("secret_field"),
            "Debug should redact input_schema"
        );
    }

    #[test]
    fn core_tool_choice_raw_redacted_in_debug() {
        let choice = CoreToolChoice::Raw(serde_json::json!({"secret": "value"}));
        let debug = format!("{choice:?}");
        assert!(!debug.contains("secret"), "Debug should redact Raw value");
    }

    #[test]
    fn core_request_rejects_unknown_fields() {
        let json = r#"{
            "model": {"requested": "m"},
            "messages": [],
            "system": [],
            "tools": [],
            "sampling": {},
            "stream": false,
            "metadata": {},
            "provider_hints": {},
            "bogus_field": true
        }"#;
        assert!(
            serde_json::from_str::<CoreRequest>(json).is_err(),
            "should reject unknown field bogus_field"
        );
    }

    #[test]
    fn core_response_rejects_unknown_fields() {
        let json = r#"{
            "model": {"requested": "m"},
            "content": [],
            "stop_reason": "EndTurn",
            "usage": {"input_tokens": 0, "output_tokens": 0, "provenance": "Unknown"},
            "bogus_field": true
        }"#;
        assert!(
            serde_json::from_str::<CoreResponse>(json).is_err(),
            "should reject unknown field bogus_field"
        );
    }

    #[test]
    fn model_ref_rejects_unknown_fields() {
        let json = r#"{"requested": "gpt-4o", "upstream": null, "bogus": true}"#;
        assert!(
            serde_json::from_str::<ModelRef>(json).is_err(),
            "should reject unknown field on ModelRef"
        );
    }

    #[test]
    fn cache_control_round_trips_through_json() {
        let cc = CacheControl {
            r#type: CacheControlType::Ephemeral,
            ttl: Some("1h".into()),
        };
        let json = serde_json::to_string(&cc).expect("serialize CacheControl");
        let back: CacheControl = serde_json::from_str(&json).expect("deserialize CacheControl");
        assert_eq!(cc, back);

        let cc_other = CacheControl {
            r#type: CacheControlType::Other("custom".into()),
            ttl: None,
        };
        let json = serde_json::to_string(&cc_other).expect("serialize CacheControl Other");
        let back: CacheControl =
            serde_json::from_str(&json).expect("deserialize CacheControl Other");
        assert_eq!(cc_other, back);
    }

    #[test]
    fn sampling_options_stop_rejects_array_with_non_strings() {
        let json = r#"{"stop": ["STOP", 42]}"#;
        assert!(
            serde_json::from_str::<SamplingOptions>(json).is_err(),
            "should reject array with non-string items"
        );
        let json2 = r#"{"stop": [true]}"#;
        assert!(
            serde_json::from_str::<SamplingOptions>(json2).is_err(),
            "should reject array with boolean items"
        );
    }

    #[test]
    fn sampling_options_stop_deserializes_empty_array() {
        let json = r#"{"stop": []}"#;
        let opts: SamplingOptions = serde_json::from_str(json).expect("deserialize");
        assert_eq!(opts.stop, None);
    }

    #[test]
    fn sampling_options_thinking_round_trips() {
        let opts = SamplingOptions {
            thinking: Some(serde_json::json!({"type": "enabled", "budget_tokens": 5000})),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let back: SamplingOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(opts, back);
        assert_eq!(
            back.thinking.as_ref().unwrap().get("budget_tokens"),
            Some(&serde_json::json!(5000))
        );
    }

    #[test]
    fn core_tool_round_trips_through_json() {
        let tool = CoreTool {
            name: "get_weather".into(),
            description: Some("Get the current weather".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        };
        let json = serde_json::to_string(&tool).expect("serialize CoreTool");
        let back: CoreTool = serde_json::from_str(&json).expect("deserialize CoreTool");
        assert_eq!(tool, back);
    }

    #[test]
    fn core_tool_rejects_non_object_input_schema_on_deserialize() {
        let json = r#"{"name":"f","description":null,"input_schema":"not-an-object"}"#;
        let err = serde_json::from_str::<CoreTool>(json);
        assert!(
            err.is_err(),
            "deserializing a non-object input_schema should fail"
        );
        let msg = err.unwrap_err().to_string();
        assert!(
            msg.contains("input_schema must be a JSON object"),
            "error should mention the object constraint, got: {msg}"
        );
    }

    #[test]
    fn bool_and_number_redacted_in_debug() {
        let content = CoreContent::Image {
            source: serde_json::json!({"flag": true, "count": 42}),
        };
        let debug = format!("{content:?}");
        assert!(
            debug.contains("Object"),
            "should show Object type for source"
        );
    }

    #[test]
    fn core_content_rejects_unknown_fields_in_variant() {
        let json = r#"{"Text": {"text": "hi", "secret_extra": "leaked"}}"#;
        assert!(
            serde_json::from_str::<CoreContent>(json).is_err(),
            "CoreContent::Text should reject unknown field"
        );
        let json2 = r#"{"ToolUse": {"id": "1", "name": "fn", "input": {}, "extra": true}}"#;
        assert!(
            serde_json::from_str::<CoreContent>(json2).is_err(),
            "CoreContent::ToolUse should reject unknown field"
        );
        let json3 =
            r#"{"ToolResult": {"tool_use_id": "1", "content": [], "is_error": false, "extra": 1}}"#;
        assert!(
            serde_json::from_str::<CoreContent>(json3).is_err(),
            "CoreContent::ToolResult should reject unknown field"
        );
    }

    #[test]
    fn core_event_rejects_unknown_fields_in_variant() {
        let json = r#"{"TextDelta": {"index": 0, "text": "hi", "extra": true}}"#;
        assert!(
            serde_json::from_str::<CoreEvent>(json).is_err(),
            "CoreEvent::TextDelta should reject unknown field"
        );
        let json2 = r#"{"MessageStart": {"id": null, "model": {"requested": "m"}, "extra": 1}}"#;
        assert!(
            serde_json::from_str::<CoreEvent>(json2).is_err(),
            "CoreEvent::MessageStart should reject unknown field"
        );
        let json3 =
            r#"{"MessageStop": {"stop_reason": "EndTurn", "stop_sequence": null, "extra": true}}"#;
        assert!(
            serde_json::from_str::<CoreEvent>(json3).is_err(),
            "CoreEvent::MessageStop should reject unknown field"
        );
    }
}
