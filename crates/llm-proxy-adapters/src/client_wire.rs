//! The client side of the wire at the provider edge: which dialect a request
//! arrived in, its raw body, and the cross-format field policy.
//!
//! Same-format requests (an Anthropic-dialect client reaching an Anthropic
//! Messages upstream, an OpenAI Chat client reaching a Chat Completions
//! upstream) are forwarded from the raw body with only surgical edits, so
//! unknown fields survive byte-for-byte. Cross-format requests go through the
//! normalized core encoding, and every client field the core does not model is
//! looked up in one table per format pair: **carried** verbatim, **mapped** by
//! adapter code, or **rejected** with an error naming it (a per-route `lossy`
//! opt-in turns a rejection into a drop with a logged warning). Server tools
//! and message blocks the cross-format encoders cannot carry are rejected the
//! same way.

use std::collections::HashMap;
use std::sync::Arc;

use llm_proxy_protocol::core::{CoreContent, CoreRequest, CoreToolChoice};
use serde_json::Value;

use crate::ProtocolName;
use crate::error::ProviderError;
use crate::protocols;

/// The wire dialect a client request arrived in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientWireFormat {
    /// Anthropic Messages (`/v1/messages`).
    AnthropicMessages,
    /// OpenAI Chat Completions (`/v1/chat/completions`).
    OpenAiChatCompletions,
    /// OpenAI Responses (`/v1/responses`; plan 08 step 1).
    OpenAiResponses,
    /// Gemini native `generateContent` (plan 08 step 2).
    GeminiGenerateContent,
    /// Ollama `/api/chat` (plan 08 step 3).
    OllamaChat,
    /// Ollama `/api/generate` (plan 08 step 3).
    OllamaGenerate,
}

impl ClientWireFormat {
    /// Canonical name, for logs.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::AnthropicMessages => "anthropic_messages",
            Self::OpenAiChatCompletions => "openai_chat_completions",
            Self::OpenAiResponses => "openai_responses",
            Self::GeminiGenerateContent => "gemini_generate_content",
            Self::OllamaChat => "ollama_chat",
            Self::OllamaGenerate => "ollama_generate",
        }
    }
}

/// The client's request as it arrived: dialect, parsed raw body, and the
/// inbound headers the per-provider allowlist selected for forwarding.
///
/// The routes construct it (empty `forwarded_headers`); the pipeline fills
/// `forwarded_headers` once the target and its allowlist are resolved.
#[derive(Clone)]
pub struct ClientWireRequest {
    /// Dialect of the raw body.
    pub format: ClientWireFormat,
    /// The parsed raw request body. Same-format passthrough edits this value
    /// instead of re-encoding the normalized core request.
    pub body: Arc<Value>,
    /// Allowlisted inbound headers, applied to the upstream request.
    pub forwarded_headers: HashMap<String, String>,
    /// Set when the engine rewrote the normalized request after decoding
    /// (a masking guardrail, plan 09 step 4): the redacted core — not the
    /// client's raw bytes — is the truth, so the same-format passthrough
    /// shortcut is refused and the adapter re-encodes from the core.
    pub force_core_encode: bool,
}

impl std::fmt::Debug for ClientWireRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientWireRequest")
            .field("format", &self.format)
            .field("body", &"<opaque client body>")
            .field(
                "forwarded_header_names",
                &self.forwarded_headers.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ClientWireRequest {
    /// Construct from the dialect and raw parsed body; `forwarded_headers`
    /// starts empty and is filled by the pipeline.
    #[must_use]
    pub fn new(format: ClientWireFormat, body: Value) -> Self {
        Self {
            format,
            body: Arc::new(body),
            forwarded_headers: HashMap::new(),
            force_core_encode: false,
        }
    }

    /// Whether this request's dialect matches the provider's protocol, i.e.
    /// same-format passthrough applies.
    #[must_use]
    pub fn same_format_as(&self, protocol: &ProtocolName) -> bool {
        if self.force_core_encode {
            return false;
        }
        matches!(
            (self.format, protocol.as_str()),
            (
                ClientWireFormat::AnthropicMessages,
                protocols::ANTHROPIC_MESSAGES
            ) | (
                ClientWireFormat::OpenAiChatCompletions,
                protocols::OPENAI_CHAT_COMPLETIONS
            ) | (
                ClientWireFormat::OpenAiResponses,
                protocols::OPENAI_RESPONSES
            ) | (
                ClientWireFormat::GeminiGenerateContent,
                protocols::GEMINI_GENERATE_CONTENT
            )
        )
    }

    /// The raw body as a JSON object, or `None` (with a warning) when the body
    /// is not an object; a non-object can never have passed typed decode, so
    /// this only guards against internal wiring mistakes.
    #[must_use]
    pub fn body_object(&self) -> Option<&serde_json::Map<String, Value>> {
        match self.body.as_object() {
            Some(obj) => Some(obj),
            None => {
                tracing::warn!(
                    format = self.format.name(),
                    "client raw body is not a JSON object; cannot passthrough"
                );
                None
            }
        }
    }
}

/// Top-level request keys the typed Anthropic Messages decode consumes.
/// Everything else on an Anthropic client body is an "extra" for the policy.
pub const ANTHROPIC_MESSAGES_REQUEST_KEYS: &[&str] = &[
    "model",
    "max_tokens",
    "system",
    "messages",
    "stream",
    "tools",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "metadata",
    "thinking",
    "tool_choice",
];

/// Top-level request keys the typed OpenAI Chat decode consumes.
pub const OPENAI_CHAT_REQUEST_KEYS: &[&str] = &[
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_completion_tokens",
    "max_tokens",
    "reasoning_effort",
    "thinking",
    "tools",
    "tool_choice",
    "stop",
    "stream_options",
    "user",
];

/// Top-level keys of `body` that the typed decode for `format` does not
/// consume; these are the fields the cross-format policy speaks about. The
/// key lists for the plan-08 dialects live with their client codecs in
/// `llm-proxy-protocol` (one definition, one path).
#[must_use]
pub fn extra_field_names(body: &Value, format: ClientWireFormat) -> Vec<&str> {
    let Some(obj) = body.as_object() else {
        return Vec::new();
    };
    let known = match format {
        ClientWireFormat::AnthropicMessages => ANTHROPIC_MESSAGES_REQUEST_KEYS,
        ClientWireFormat::OpenAiChatCompletions => OPENAI_CHAT_REQUEST_KEYS,
        ClientWireFormat::OpenAiResponses => {
            llm_proxy_protocol::client::responses::RESPONSES_REQUEST_KEYS
        }
        ClientWireFormat::GeminiGenerateContent => {
            llm_proxy_protocol::client::gemini::GEMINI_REQUEST_KEYS
        }
        ClientWireFormat::OllamaChat => {
            llm_proxy_protocol::client::ollama::OLLAMA_CHAT_REQUEST_KEYS
        }
        ClientWireFormat::OllamaGenerate => {
            llm_proxy_protocol::client::ollama::OLLAMA_GENERATE_REQUEST_KEYS
        }
    };
    let mut extras: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|key| !known.contains(key))
        .collect();
    extras.sort_unstable();
    extras
}

/// What happens to a client field when crossing formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldDisposition {
    /// Copied verbatim onto the provider request (the provider accepts the
    /// same field name and shape).
    Carried,
    /// Translated by adapter code into the provider's equivalent. Subfields
    /// the translation does not consume are rejected by name like unmapped
    /// fields (see [`check_output_config_subkeys`]).
    Mapped,
    /// The provider has no equivalent; the request fails naming the field
    /// (dropped with a warning under the per-route `lossy` opt-in).
    Rejected,
}

/// The cross-format field policy: one disposition per (client dialect,
/// provider protocol, client field) combination. Fields the typed decode
/// already consumed into the core (messages, sampling, tools, ...) are mapped
/// by the adapters and are not named here; this table answers for the
/// leftover client fields.
///
/// The per-provider arms below are the fallback for every client dialect; the
/// grid lists only the cells that differ (rows = provider protocol, columns =
/// client dialect):
///
/// | provider protocol | client dialect                       | field | disposition |
/// |-------------------|--------------------------------------|-------|-------------|
/// | Gemini            | OpenAI Responses, Anthropic Messages | seed  | Rejected    |
#[must_use]
pub fn field_disposition(
    client: ClientWireFormat,
    provider: &ProtocolName,
    field: &str,
) -> FieldDisposition {
    // The table covers the built-in protocols; a protocol added by an SDK
    // user has no policy table, so leftover client fields are rejected by
    // name unless the custom adapter's own same-format path handles them.
    match (client, provider.as_str()) {
        // `seed` maps only from the OpenAI Chat dialect (the Responses and
        // Anthropic APIs have no seed): their clients keep the named
        // rejection even on a Gemini target.
        (
            ClientWireFormat::OpenAiResponses | ClientWireFormat::AnthropicMessages,
            protocols::GEMINI_GENERATE_CONTENT,
        ) if field == "seed" => FieldDisposition::Rejected,
        (_, protocols::ANTHROPIC_MESSAGES) => match field {
            // Anthropic accepts these OpenAI Chat fields with the same shape.
            "service_tier" | "top_k" => FieldDisposition::Carried,
            // Mapped by the Anthropic adapter from the raw client body.
            "parallel_tool_calls" => FieldDisposition::Mapped,
            _ => FieldDisposition::Rejected,
        },
        (_, protocols::OPENAI_CHAT_COMPLETIONS) => match field {
            // The Chat Completions API accepts these with the same shape.
            "service_tier" | "parallel_tool_calls" | "store" | "prompt_cache_key" => {
                FieldDisposition::Carried
            }
            // Mapped by the chat adapter from the raw client body.
            "output_config" => FieldDisposition::Mapped,
            _ => FieldDisposition::Rejected,
        },
        (_, protocols::OPENAI_RESPONSES) => match field {
            // The Responses API accepts these with the same shape.
            "service_tier" | "parallel_tool_calls" | "store" | "prompt_cache_key" => {
                FieldDisposition::Carried
            }
            // Mapped by the Responses adapter from the raw client body:
            // `response_format` becomes the `text.format` structured-output
            // surface.
            "output_config" | "response_format" => FieldDisposition::Mapped,
            _ => FieldDisposition::Rejected,
        },
        (_, protocols::GEMINI_GENERATE_CONTENT) => match field {
            // Mapped by the Gemini adapter into generationConfig.topK.
            "top_k" => FieldDisposition::Mapped,
            // Mapped by the Gemini adapter from the raw client body:
            // `response_format` becomes generationConfig.responseMimeType
            // (plus responseJsonSchema for a schema).
            "response_format" => FieldDisposition::Mapped,
            // Mapped by the Gemini adapter into generationConfig.seed.
            "seed" => FieldDisposition::Mapped,
            _ => FieldDisposition::Rejected,
        },
        _ => FieldDisposition::Rejected,
    }
}

/// Applies the cross-format field policy to the client's extra fields.
///
/// Returns the `Carried` fields (the caller injects them verbatim); `Mapped`
/// fields are the caller's translation code. A `Rejected` field fails with an
/// error naming every rejected field, unless `lossy` drops them with a logged
/// warning instead.
pub fn apply_field_policy(
    client: &ClientWireRequest,
    provider: &ProtocolName,
    lossy: bool,
) -> Result<serde_json::Map<String, Value>, ProviderError> {
    let mut carried = serde_json::Map::new();
    let mut rejected: Vec<String> = Vec::new();
    let extras = extra_field_names(&client.body, client.format);
    let body_obj = client.body.as_object();

    for field in extras {
        match field_disposition(client.format, provider, field) {
            FieldDisposition::Carried => {
                if let Some(value) = body_obj.and_then(|obj| obj.get(field)) {
                    carried.insert(field.to_owned(), value.clone());
                }
            }
            FieldDisposition::Mapped => {}
            FieldDisposition::Rejected => rejected.push(field.to_owned()),
        }
    }

    check_cross_format_tools(client, provider, lossy)?;
    check_strict_function_tools(client, provider, lossy)?;
    check_output_config_subkeys(client, provider, lossy)?;

    if rejected.is_empty() {
        return Ok(carried);
    }
    if lossy {
        for field in &rejected {
            tracing::warn!(
                field,
                provider = %provider,
                "lossy route: dropping client field with no provider equivalent"
            );
        }
        return Ok(carried);
    }
    Err(ProviderError::InvalidRequest(format!(
        "field(s) {} have no {provider} equivalent; remove them or enable the route's lossy mode",
        rejected.join(", "),
    )))
}

/// Cross-format `tool_choice` policy shared by the Chat, Responses and
/// Anthropic encoders: each encoder supplies its per-variant `map`, and a
/// variant the map cannot represent (`CoreToolChoice` is `#[non_exhaustive]`,
/// so future variants decode fine and map to nothing) is rejected with an
/// error naming the field — or, under the per-route `lossy` opt-in, dropped
/// with a logged warning instead of being silently omitted.
pub(crate) fn resolve_tool_choice<F>(
    choice: &CoreToolChoice,
    lossy: bool,
    protocol: &str,
    map: F,
) -> Result<Option<Value>, ProviderError>
where
    F: FnOnce(&CoreToolChoice) -> Option<Value>,
{
    match map(choice) {
        Some(value) => Ok(Some(value)),
        None if lossy => {
            tracing::warn!(
                tool_choice = ?choice,
                provider = protocol,
                "lossy route: dropping tool_choice with no provider equivalent"
            );
            Ok(None)
        }
        None => Err(ProviderError::InvalidRequest(format!(
            "tool_choice {choice:?} has no {protocol} equivalent; \
             remove it or enable the route's lossy mode"
        ))),
    }
}

/// Server tools (Anthropic `{"type": "web_search_...", ...}`, Responses
/// `{"type": "web_search", ...}`, Gemini `{"googleSearch": {}}`) decode as
/// bare names in the core or stay out of the core tool list; forwarding them
/// cross-format would silently change what the model can do, so they are
/// rejected by name (or dropped whole under `lossy`, which the caller must
/// then also drop from the tools list).
fn check_cross_format_tools(
    client: &ClientWireRequest,
    provider: &ProtocolName,
    lossy: bool,
) -> Result<(), ProviderError> {
    let Some(obj) = client.body.as_object() else {
        return Ok(());
    };
    let Some(Value::Array(tools)) = obj.get("tools") else {
        return Ok(());
    };
    let mut server_tools: Vec<String> = Vec::new();
    for tool in tools.iter().filter_map(Value::as_object) {
        let label = match client.format {
            // Anthropic tools carry a `type`; function tools do not.
            ClientWireFormat::AnthropicMessages => {
                tool.get("type").and_then(Value::as_str).map(|kind| {
                    let name = tool
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("<unnamed>");
                    format!("type={kind} name={name}")
                })
            }
            // Responses function tools are `{"type": "function", ...}`;
            // every other type is a server tool.
            ClientWireFormat::OpenAiResponses => match tool.get("type").and_then(Value::as_str) {
                None | Some("function") => None,
                Some(kind) => {
                    let name = tool
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("<unnamed>");
                    Some(format!("type={kind} name={name}"))
                }
            },
            // Gemini tool objects are `{"functionDeclarations": [...]}`;
            // anything else (googleSearch, codeExecution) is a server tool.
            ClientWireFormat::GeminiGenerateContent => {
                if tool.contains_key("functionDeclarations")
                    || tool.contains_key("function_declarations")
                {
                    None
                } else {
                    tool.keys().next().cloned()
                }
            }
            _ => None,
        };
        if let Some(label) = label {
            server_tools.push(label);
        }
    }
    if server_tools.is_empty() {
        return Ok(());
    }
    if lossy {
        tracing::warn!(
            tools = ?server_tools,
            provider = %provider,
            "lossy route: dropping server tools with no provider equivalent"
        );
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(format!(
        "server tool(s) [{}] have no {provider} equivalent; remove them or enable the route's lossy mode",
        server_tools.join(", "),
    )))
}

/// A tool-level `strict: true` on a Responses function tool (Codex sends it)
/// carries the client's structured-output guarantee. The core tool models no
/// such field (`decode_tools` reads `name`/`description`/`parameters` only),
/// so crossing formats would silently drop the guarantee — these tools are
/// rejected by name (dropped with a warning under `lossy`), while
/// `strict: false`/absent pass (they are the default meaning and lose
/// nothing) and same-format passthrough carries the raw body untouched.
fn check_strict_function_tools(
    client: &ClientWireRequest,
    provider: &ProtocolName,
    lossy: bool,
) -> Result<(), ProviderError> {
    if client.format != ClientWireFormat::OpenAiResponses {
        return Ok(());
    }
    let Some(Value::Array(tools)) = client.body.as_object().and_then(|obj| obj.get("tools")) else {
        return Ok(());
    };
    let mut strict_tools: Vec<&str> = Vec::new();
    for tool in tools.iter().filter_map(Value::as_object) {
        // `type` absent or "function" is a function tool; every other type
        // is a server tool already handled by `check_cross_format_tools`.
        let is_function = matches!(
            tool.get("type").and_then(Value::as_str),
            None | Some("function")
        );
        if is_function
            && tool.get("strict") == Some(&Value::Bool(true))
            && let Some(name) = tool.get("name").and_then(Value::as_str)
        {
            strict_tools.push(name);
        }
    }
    if strict_tools.is_empty() {
        return Ok(());
    }
    if lossy {
        tracing::warn!(
            tools = ?strict_tools,
            provider = %provider,
            "lossy route: dropping function tools' strict flag with no provider equivalent"
        );
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(format!(
        "function tool(s) [{}] set strict: true, which has no {provider} equivalent; remove strict or enable the route's lossy mode",
        strict_tools.join(", "),
    )))
}

/// Whether the cross-format image mappings (OpenAI `image_url` parts, Gemini
/// `inlineData`/`fileData`) can read a neutral image source: `base64` with
/// `media_type` and `data`, or `url` with `url`.
fn image_source_is_mappable(source: &Value) -> bool {
    let Some(source) = source.as_object() else {
        return false;
    };
    let has_string = |key: &str| source.get(key).is_some_and(Value::is_string);
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => has_string("media_type") && has_string("data"),
        Some("url") => has_string("url"),
        _ => false,
    }
}

/// Labels a message block the cross-format encoders cannot carry, or `None`
/// when they can.
fn uncarried_block_label(content: &CoreContent) -> Option<String> {
    let wire_type = |value: &Value| {
        let found = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        llm_proxy_protocol::util::truncate_str_safe(found, 64).to_owned()
    };
    match content {
        CoreContent::Opaque { block } => Some(format!("type={}", wire_type(block))),
        CoreContent::Document { .. } => Some("type=document".to_owned()),
        CoreContent::Image { source } if !image_source_is_mappable(source) => {
            Some(format!("type=image source.type={}", wire_type(source)))
        }
        _ => None,
    }
}

/// Message blocks the cross-format encoders cannot carry: opaque blocks (a
/// client block the core does not model, e.g. a replayed Anthropic
/// `server_tool_use`), documents (no built-in provider encodes them) and
/// images without a readable `base64`/`url` source (e.g. a Files API `file`
/// source). Same-format passthrough forwards the raw client body instead;
/// every cross-format encode calls this so these blocks are rejected by name
/// rather than dropped by the encoder (dropped with a warning under `lossy`).
pub fn check_cross_format_content(
    core: &CoreRequest,
    provider: &ProtocolName,
    lossy: bool,
) -> Result<(), ProviderError> {
    let mut uncarried: Vec<String> = Vec::new();
    for (m, message) in core.messages.iter().enumerate() {
        for (b, content) in message.content.iter().enumerate() {
            if let Some(label) = uncarried_block_label(content) {
                uncarried.push(format!("messages[{m}].content[{b}] {label}"));
            }
        }
    }
    if uncarried.is_empty() {
        return Ok(());
    }
    if lossy {
        tracing::warn!(
            blocks = ?uncarried,
            provider = %provider,
            "lossy route: dropping content blocks with no provider equivalent"
        );
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(format!(
        "content block(s) [{}] have no {provider} equivalent; remove them or enable the route's lossy mode",
        uncarried.join(", "),
    )))
}

/// Subkeys of `output_config` the adapter mappings translate (the `effort`
/// level becomes the chat-family `reasoning_effort`/`reasoning.effort`).
/// Keep in sync with the mappings in the chat and Responses adapters.
const OUTPUT_CONFIG_MAPPED_SUBKEYS: &[&str] = &["effort"];

/// `output_config` is declared [`FieldDisposition::Mapped`] for the
/// chat-family providers, but the mappings translate only its `effort`
/// subkey — a structured-output `format` or any other subkey has no
/// translation. Those subkeys are rejected by name (dropped with a warning
/// under `lossy`) instead of silently vanishing, and a non-object
/// `output_config` (which no mapping can read) is treated the same way.
fn check_output_config_subkeys(
    client: &ClientWireRequest,
    provider: &ProtocolName,
    lossy: bool,
) -> Result<(), ProviderError> {
    if field_disposition(client.format, provider, "output_config") != FieldDisposition::Mapped {
        return Ok(());
    }
    let Some(config) = client
        .body
        .as_object()
        .and_then(|obj| obj.get("output_config"))
    else {
        return Ok(());
    };
    if config.is_null() {
        // `output_config: null` means absent, mirroring the `thinking` decode.
        return Ok(());
    }
    let Some(subkeys) = config.as_object() else {
        if lossy {
            tracing::warn!(
                field = "output_config",
                provider = %provider,
                "lossy route: dropping non-object output_config"
            );
            return Ok(());
        }
        return Err(ProviderError::InvalidRequest(format!(
            "output_config must be an object to be translated for {provider}"
        )));
    };
    let unmapped: Vec<&str> = subkeys
        .keys()
        .map(String::as_str)
        .filter(|key| !OUTPUT_CONFIG_MAPPED_SUBKEYS.contains(key))
        .collect();
    if unmapped.is_empty() {
        return Ok(());
    }
    if lossy {
        for key in &unmapped {
            tracing::warn!(
                field = %format!("output_config.{key}"),
                provider = %provider,
                "lossy route: dropping client field with no provider equivalent"
            );
        }
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(format!(
        "output_config subfield(s) {} have no {provider} equivalent; remove them or enable the route's lossy mode",
        unmapped.join(", "),
    )))
}

/// The cross-format reading of a client `response_format` object for a
/// provider the disposition table marks [`FieldDisposition::Mapped`]
/// (Responses `text.format`, Gemini `generationConfig.responseMimeType`).
pub(crate) enum ResponseFormat<'a> {
    /// `{"type": "text"}`: plain text is the provider default, so nothing is
    /// written to the wire.
    Text,
    /// `{"type": "json_object"}`: JSON mode without a schema.
    JsonObject,
    /// `{"type": "json_schema", ...}`: structured output. `spec` is the
    /// client's `json_schema` object (`name`, `schema`, `strict`,
    /// `description`), which each adapter flattens into its own shape.
    JsonSchema {
        /// The client's `json_schema` spec object.
        spec: &'a serde_json::Map<String, Value>,
    },
}

/// Reads the client's raw `response_format` for an adapter whose disposition
/// table marks the field `Mapped`. `Ok(None)` when the field is absent or
/// null. A shape no mapping can read (a non-object, a `json_schema` without a
/// spec object, an unknown `type`) is rejected naming the offending path — or,
/// under the per-route `lossy` opt-in, dropped with a logged warning instead
/// of being forwarded or silently lost.
pub(crate) fn parse_response_format<'a>(
    client: &'a ClientWireRequest,
    protocol: &str,
    lossy: bool,
) -> Result<Option<ResponseFormat<'a>>, ProviderError> {
    let Some(format) = client
        .body
        .as_object()
        .and_then(|obj| obj.get("response_format"))
    else {
        return Ok(None);
    };
    if format.is_null() {
        // `response_format: null` means absent, mirroring the `thinking` decode.
        return Ok(None);
    }
    let Some(spec) = format.as_object() else {
        return reject_unmappable_response_format("response_format", protocol, lossy)
            .map(|()| None);
    };
    match spec.get("type").and_then(Value::as_str) {
        Some("text") => Ok(Some(ResponseFormat::Text)),
        Some("json_object") => Ok(Some(ResponseFormat::JsonObject)),
        Some("json_schema") => match spec.get("json_schema").and_then(Value::as_object) {
            Some(schema_spec) => Ok(Some(ResponseFormat::JsonSchema { spec: schema_spec })),
            None => {
                reject_unmappable_response_format("response_format.json_schema", protocol, lossy)
                    .map(|()| None)
            }
        },
        _ => reject_unmappable_response_format("response_format.type", protocol, lossy)
            .map(|()| None),
    }
}

/// Rejects a `response_format` shape the target provider cannot carry,
/// naming the offending path; a lossy route drops it with a logged warning
/// instead of failing the request.
pub(crate) fn reject_unmappable_response_format(
    name: &str,
    protocol: &str,
    lossy: bool,
) -> Result<(), ProviderError> {
    if lossy {
        tracing::warn!(
            field = name,
            provider = protocol,
            "lossy route: dropping client field with no provider equivalent"
        );
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(format!(
        "{name} has no {protocol} equivalent; remove it or enable the route's lossy mode"
    )))
}

/// Whether the raw Anthropic client body contains at least one server tool;
/// the Anthropic adapter drops those tools from the normalized `tools` list on
/// lossy cross-format routes (they cannot be represented as functions).
#[must_use]
pub fn anthropic_body_has_server_tools(body: &Value) -> bool {
    let Some(Value::Array(tools)) = body.as_object().and_then(|obj| obj.get("tools")) else {
        return false;
    };
    tools
        .iter()
        .any(|tool| tool.as_object().is_some_and(|t| t.contains_key("type")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn wire(format: ClientWireFormat, body: Value) -> ClientWireRequest {
        ClientWireRequest::new(format, body)
    }

    #[test]
    fn same_format_detection() {
        let anthropic = wire(ClientWireFormat::AnthropicMessages, json!({}));
        let chat = wire(ClientWireFormat::OpenAiChatCompletions, json!({}));
        assert!(anthropic.same_format_as(&ProtocolName::new(protocols::ANTHROPIC_MESSAGES)));
        assert!(!anthropic.same_format_as(&ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS)));
        assert!(chat.same_format_as(&ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS)));
        assert!(!chat.same_format_as(&ProtocolName::new(protocols::ANTHROPIC_MESSAGES)));
    }

    #[test]
    fn extra_fields_are_the_unconsumed_keys() {
        let body = json!({
            "model": "m", "messages": [], "seed": 7, "response_format": {"type": "json_object"}
        });
        let extras = extra_field_names(&body, ClientWireFormat::OpenAiChatCompletions);
        assert_eq!(extras, vec!["response_format", "seed"]);
    }

    #[test]
    fn anthropic_extras_include_service_tier_and_output_config() {
        let body = json!({"model": "m", "messages": [], "service_tier": "auto"});
        let extras = extra_field_names(&body, ClientWireFormat::AnthropicMessages);
        assert_eq!(extras, vec!["service_tier"]);
    }

    #[test]
    fn carried_fields_survive_the_policy() {
        let client = wire(
            ClientWireFormat::OpenAiChatCompletions,
            json!({"model": "m", "messages": [], "service_tier": "flex"}),
        );
        let carried = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            false,
        )
        .expect("service_tier is carried");
        assert_eq!(carried.get("service_tier"), Some(&json!("flex")));

        // A Responses-dialect client's `parallel_tool_calls`, `store` and
        // `prompt_cache_key` ride verbatim to the Chat Completions provider:
        // same names, same shapes.
        let client = wire(
            ClientWireFormat::OpenAiResponses,
            json!({
                "model": "m",
                "input": [],
                "parallel_tool_calls": false,
                "store": true,
                "prompt_cache_key": "k"
            }),
        );
        let carried = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .expect("the three Responses extras are carried to Chat");
        assert_eq!(carried.get("parallel_tool_calls"), Some(&json!(false)));
        assert_eq!(carried.get("store"), Some(&json!(true)));
        assert_eq!(carried.get("prompt_cache_key"), Some(&json!("k")));
    }

    #[test]
    fn rejected_fields_are_named_in_the_error() {
        let client = wire(
            ClientWireFormat::OpenAiChatCompletions,
            json!({"model": "m", "messages": [], "seed": 7, "logprobs": true}),
        );
        // `seed` is mapped only for Gemini; Responses and Anthropic keep
        // rejecting it, as they do `logprobs`.
        for provider in [protocols::ANTHROPIC_MESSAGES, protocols::OPENAI_RESPONSES] {
            let err = apply_field_policy(&client, &ProtocolName::new(provider), false)
                .expect_err("seed and logprobs must be rejected");
            let msg = err.to_string();
            assert!(msg.contains("seed"), "error must name seed: {msg}");
            assert!(msg.contains("logprobs"), "error must name logprobs: {msg}");
        }
    }

    #[test]
    fn lossy_mode_drops_rejected_fields_instead_of_failing() {
        let client = wire(
            ClientWireFormat::OpenAiChatCompletions,
            json!({"model": "m", "messages": [], "seed": 7}),
        );
        let carried = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            true,
        )
        .expect("lossy drops instead of rejecting");
        assert!(carried.is_empty());
    }

    #[test]
    fn mapped_fields_do_not_fail_and_are_not_carried() {
        let client = wire(
            ClientWireFormat::OpenAiChatCompletions,
            json!({"model": "m", "messages": [], "parallel_tool_calls": false}),
        );
        let carried = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            false,
        )
        .expect("parallel_tool_calls is mapped, not rejected");
        assert!(carried.is_empty());
    }

    #[test]
    fn response_format_maps_cross_format_for_responses_and_gemini() {
        let client = wire(
            ClientWireFormat::OpenAiChatCompletions,
            json!({
                "model": "m", "messages": [],
                "response_format": {"type": "json_object"}
            }),
        );
        for provider in [
            protocols::OPENAI_RESPONSES,
            protocols::GEMINI_GENERATE_CONTENT,
        ] {
            let carried = apply_field_policy(&client, &ProtocolName::new(provider), false)
                .expect("response_format is mapped, not rejected, for these providers");
            assert!(
                !carried.contains_key("response_format"),
                "mapped response_format must not also be carried verbatim: {carried:?}"
            );
        }
        let err = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            false,
        )
        .expect_err("Anthropic has no structured-output mapping for response_format");
        assert!(
            err.to_string().contains("response_format"),
            "error must name response_format: {err}"
        );
    }

    #[test]
    fn output_config_effort_subkey_passes_the_policy() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({"model": "m", "max_tokens": 8, "messages": [], "output_config": {"effort": "high"}}),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .expect("the effort subkey is the mapped one");
    }

    #[test]
    fn unmapped_output_config_subkeys_are_rejected_by_name() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({
                "model": "m", "max_tokens": 8, "messages": [],
                "output_config": {"effort": "high", "format": {"type": "json_schema"}}
            }),
        );
        let err = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .expect_err("output_config.format has no chat translation");
        assert!(
            err.to_string().contains("format"),
            "error must name the subfield: {err}"
        );
        // Same contract on the Responses path.
        let err = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_RESPONSES),
            false,
        )
        .expect_err("output_config.format has no Responses translation");
        assert!(
            err.to_string().contains("format"),
            "error must name the subfield: {err}"
        );
    }

    #[test]
    fn lossy_mode_drops_unmapped_output_config_subkeys() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({
                "model": "m", "max_tokens": 8, "messages": [],
                "output_config": {"format": {"type": "json_schema"}}
            }),
        );
        let carried = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            true,
        )
        .expect("lossy drops the unmapped subkey instead of rejecting");
        assert!(carried.is_empty());
    }

    #[test]
    fn non_object_output_config_is_rejected() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({"model": "m", "max_tokens": 8, "messages": [], "output_config": "high"}),
        );
        let err = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .expect_err("a non-object output_config cannot be translated");
        assert!(
            err.to_string().contains("output_config"),
            "error must name the field: {err}"
        );
        // `null` means absent, as with the `thinking` decode.
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({"model": "m", "max_tokens": 8, "messages": [], "output_config": null}),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .expect("null output_config is absent");
    }

    #[test]
    fn server_tools_are_rejected_by_name_cross_format() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({
                "model": "m", "max_tokens": 8, "messages": [],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}]
            }),
        );
        let err = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .expect_err("server tools must be rejected cross-format");
        assert!(
            err.to_string().contains("web_search_20250305"),
            "error must name the server tool: {err}"
        );
    }

    #[test]
    fn server_tools_pass_on_lossy_routes() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({
                "model": "m", "max_tokens": 8, "messages": [],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}]
            }),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            true,
        )
        .unwrap();
        assert!(anthropic_body_has_server_tools(&client.body));
    }

    #[test]
    fn plain_function_tools_are_not_server_tools() {
        let client = wire(
            ClientWireFormat::AnthropicMessages,
            json!({
                "model": "m", "max_tokens": 8, "messages": [],
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
            }),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            false,
        )
        .unwrap();
        assert!(!anthropic_body_has_server_tools(&client.body));
    }

    // The Responses and Gemini dialects decode their server tools the same
    // way the Anthropic arm above does (`{"type": "web_search"}`,
    // `{"googleSearch": {}}`); crossing formats must reject them by name
    // rather than silently change what the model can do, and the lossy
    // variant must drop them instead of failing.
    #[test]
    fn responses_server_tools_are_rejected_by_name_cross_format() {
        let body = json!({
            "model": "m", "input": [],
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "keep", "parameters": {}}
            ]
        });
        for provider in [
            protocols::ANTHROPIC_MESSAGES,
            protocols::OPENAI_CHAT_COMPLETIONS,
            protocols::GEMINI_GENERATE_CONTENT,
        ] {
            let client = wire(ClientWireFormat::OpenAiResponses, body.clone());
            let err = apply_field_policy(&client, &ProtocolName::new(provider), false)
                .expect_err("a Responses server tool must be rejected cross-format");
            let text = err.to_string();
            assert!(
                text.contains("web_search"),
                "error must name the server tool for {provider}: {text}"
            );
            let client = wire(ClientWireFormat::OpenAiResponses, body.clone());
            apply_field_policy(&client, &ProtocolName::new(provider), true)
                .expect("lossy drops the server tool instead of failing");
        }
    }

    #[test]
    fn gemini_server_tools_are_rejected_by_name_cross_format() {
        let body = json!({
            "contents": [],
            "tools": [{"googleSearch": {}}]
        });
        for provider in [
            protocols::ANTHROPIC_MESSAGES,
            protocols::OPENAI_CHAT_COMPLETIONS,
            protocols::OPENAI_RESPONSES,
        ] {
            let client = wire(ClientWireFormat::GeminiGenerateContent, body.clone());
            let err = apply_field_policy(&client, &ProtocolName::new(provider), false)
                .expect_err("a Gemini server tool must be rejected cross-format");
            let text = err.to_string();
            assert!(
                text.contains("googleSearch"),
                "error must name the server tool for {provider}: {text}"
            );
            let client = wire(ClientWireFormat::GeminiGenerateContent, body.clone());
            apply_field_policy(&client, &ProtocolName::new(provider), true)
                .expect("lossy drops the server tool instead of failing");
        }
    }

    #[test]
    fn gemini_function_declarations_are_not_server_tools() {
        let client = wire(
            ClientWireFormat::GeminiGenerateContent,
            json!({
                "contents": [],
                "tools": [{"functionDeclarations": [
                    {"name": "get_weather", "parameters": {"type": "object"}}
                ]}]
            }),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            false,
        )
        .expect("function declarations are core tools, not server tools");
    }

    #[test]
    fn responses_function_tools_are_not_server_tools() {
        let client = wire(
            ClientWireFormat::OpenAiResponses,
            json!({
                "model": "m", "input": [],
                "tools": [{"type": "function", "name": "keep", "parameters": {}}]
            }),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            false,
        )
        .expect("function tools are core tools, not server tools");
    }

    // A tool-level `strict: true` on a Responses function tool carries the
    // client's structured-output guarantee; the core tool has no such field,
    // so crossing formats must reject it by name instead of silently
    // dropping the guarantee.
    #[test]
    fn strict_function_tools_are_rejected_by_name_cross_format() {
        let tool = json!({
            "type": "function", "name": "get_weather", "strict": true,
            "parameters": {"type": "object"}
        });
        for provider in [
            protocols::ANTHROPIC_MESSAGES,
            protocols::OPENAI_CHAT_COMPLETIONS,
            protocols::GEMINI_GENERATE_CONTENT,
        ] {
            let client = wire(
                ClientWireFormat::OpenAiResponses,
                json!({"model": "m", "input": [], "tools": [tool.clone()]}),
            );
            let err = apply_field_policy(&client, &ProtocolName::new(provider), false)
                .expect_err("strict: true must be rejected cross-format");
            let text = err.to_string();
            assert!(
                text.contains("strict") && text.contains("get_weather"),
                "error must name strict and the tool for {provider}: {text}"
            );
        }
    }

    #[test]
    fn strict_false_and_absent_pass_cross_format() {
        let body = json!({
            "model": "m", "input": [],
            "tools": [
                {"type": "function", "name": "with_false", "strict": false, "parameters": {}},
                {"name": "absent", "parameters": {}}
            ]
        });
        for provider in [
            protocols::ANTHROPIC_MESSAGES,
            protocols::OPENAI_CHAT_COMPLETIONS,
            protocols::GEMINI_GENERATE_CONTENT,
        ] {
            let client = wire(ClientWireFormat::OpenAiResponses, body.clone());
            apply_field_policy(&client, &ProtocolName::new(provider), false)
                .unwrap_or_else(|err| panic!("strict {provider} must pass: {err}"));
        }
    }

    #[test]
    fn strict_tools_pass_on_lossy_routes() {
        let client = wire(
            ClientWireFormat::OpenAiResponses,
            json!({
                "model": "m", "input": [],
                "tools": [
                    {"type": "function", "name": "get_weather", "strict": true, "parameters": {}}
                ]
            }),
        );
        apply_field_policy(
            &client,
            &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            true,
        )
        .expect("a lossy route drops strict with a warning instead of failing");
    }

    fn media_core_request() -> CoreRequest {
        use llm_proxy_protocol::core::{CoreMessage, CoreRole, ModelRef, SamplingOptions};
        CoreRequest {
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            system: vec![],
            messages: vec![CoreMessage {
                role: CoreRole::User,
                content: vec![
                    CoreContent::Image {
                        source: json!({"type": "base64", "media_type": "image/png", "data": "iVBOR"}),
                    },
                    CoreContent::Image {
                        source: json!({"type": "url", "url": "https://example.com/cat.jpg"}),
                    },
                    CoreContent::Image {
                        source: json!({"type": "file", "file_id": "file_fixture_01"}),
                    },
                    CoreContent::Document {
                        source: json!({"type": "url", "url": "https://example.com/report.pdf"}),
                    },
                    CoreContent::Opaque {
                        block: json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}),
                    },
                    CoreContent::Text {
                        text: "compare".into(),
                        cache: None,
                    },
                ],
            }],
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: Default::default(),
            provider_hints: Default::default(),
        }
    }

    #[test]
    fn content_the_cross_format_encoders_cannot_carry_is_rejected_by_name() {
        let core = media_core_request();
        for provider in [
            protocols::OPENAI_CHAT_COMPLETIONS,
            protocols::OPENAI_RESPONSES,
            protocols::GEMINI_GENERATE_CONTENT,
        ] {
            let err = check_cross_format_content(&core, &ProtocolName::new(provider), false)
                .expect_err("opaque, document and file-sourced image blocks cannot cross formats");
            let msg = err.to_string();
            for named in [
                "messages[0].content[2] type=image source.type=file",
                "messages[0].content[3] type=document",
                "messages[0].content[4] type=server_tool_use",
            ] {
                assert!(
                    msg.contains(named),
                    "error must name {named} for {provider}: {msg}"
                );
            }
            for carried in ["content[0]", "content[1]", "content[5]"] {
                assert!(
                    !msg.contains(carried),
                    "base64/url images and text cross formats, {carried} must not be named: {msg}"
                );
            }
        }
    }

    #[test]
    fn content_the_cross_format_encoders_cannot_carry_passes_on_lossy_routes() {
        check_cross_format_content(
            &media_core_request(),
            &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            true,
        )
        .expect("lossy drops uncarried blocks instead of rejecting");
    }

    #[test]
    fn disposition_is_per_format_pair() {
        let cases = [
            (
                ClientWireFormat::OpenAiResponses,
                protocols::OPENAI_CHAT_COMPLETIONS,
                "parallel_tool_calls",
                FieldDisposition::Carried,
            ),
            (
                ClientWireFormat::OpenAiChatCompletions,
                protocols::GEMINI_GENERATE_CONTENT,
                "seed",
                FieldDisposition::Mapped,
            ),
            (
                ClientWireFormat::OpenAiResponses,
                protocols::GEMINI_GENERATE_CONTENT,
                "seed",
                FieldDisposition::Rejected,
            ),
            (
                ClientWireFormat::AnthropicMessages,
                protocols::GEMINI_GENERATE_CONTENT,
                "seed",
                FieldDisposition::Rejected,
            ),
            (
                ClientWireFormat::OpenAiChatCompletions,
                protocols::ANTHROPIC_MESSAGES,
                "response_format",
                FieldDisposition::Rejected,
            ),
        ];
        for (client, provider, field, want) in cases {
            assert_eq!(
                field_disposition(client, &ProtocolName::new(provider), field),
                want,
                "{client:?} client -> {provider} must dispose of {field} as {want:?}"
            );
        }
    }

    #[test]
    fn responses_carries_its_native_knobs() {
        let client = wire(
            ClientWireFormat::OpenAiChatCompletions,
            json!({"model": "m", "messages": [], "store": false, "prompt_cache_key": "k"}),
        );
        let carried = apply_field_policy(
            &client,
            &ProtocolName::new(protocols::OPENAI_RESPONSES),
            false,
        )
        .expect("store and prompt_cache_key are native Responses fields");
        assert_eq!(carried.get("store"), Some(&json!(false)));
        assert_eq!(carried.get("prompt_cache_key"), Some(&json!("k")));
    }

    #[test]
    fn resolve_tool_choice_rejects_an_unmapped_variant_by_name() {
        let err = resolve_tool_choice(
            &CoreToolChoice::Auto,
            false,
            protocols::OPENAI_CHAT_COMPLETIONS,
            |_| None,
        )
        .expect_err("an unmapped tool_choice must reject on a strict route");
        let msg = err.to_string();
        assert!(
            msg.contains("tool_choice"),
            "error must name tool_choice: {msg}"
        );
        assert!(
            msg.contains("lossy"),
            "error must mention the lossy opt-in: {msg}"
        );
    }

    #[test]
    fn resolve_tool_choice_drops_an_unmapped_variant_on_a_lossy_route() {
        let resolved = resolve_tool_choice(
            &CoreToolChoice::Any,
            true,
            protocols::ANTHROPIC_MESSAGES,
            |_| None,
        )
        .expect("a lossy route drops the unmapped tool_choice instead of failing");
        assert_eq!(resolved, None);
    }

    #[test]
    fn resolve_tool_choice_carries_a_mapped_variant() {
        let resolved = resolve_tool_choice(
            &CoreToolChoice::Auto,
            false,
            protocols::OPENAI_RESPONSES,
            |tc| {
                if matches!(tc, CoreToolChoice::Auto) {
                    Some(json!("auto"))
                } else {
                    None
                }
            },
        )
        .expect("a mapped variant resolves without error");
        assert_eq!(resolved, Some(json!("auto")));
    }
}
