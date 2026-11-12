//! llm-proxy-adapters: provider codecs. Translate `llm_proxy_protocol::core`
//! request, response and event types to and from provider-specific wire
//! formats, with no network I/O — encoding produces a
//! [`request::ProxyRequest`] of plain bytes and headers that the transport
//! crate sends.
//!
//! The provider set is open: built-in adapters register themselves in
//! [`ProviderAdapterRegistry::builtin`], and an SDK user registers further
//! protocols with [`ProviderAdapterRegistry::register`].

#![deny(missing_docs)]

pub mod anthropic;
pub mod client_wire;
pub mod error;
pub mod gemini;
pub mod mutation;
pub mod openai_chat;
pub mod reasoning;
pub mod request;
pub mod responses;
pub mod sse;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use llm_proxy_core::AuthStyle;
use llm_proxy_protocol::core::{CoreEvent, CoreRequest, CoreResponse, ModelRef, StopReason, Usage};
use secrecy::SecretString;

pub use anthropic::{AnthropicAdapter, AnthropicStreamDecoder};
pub use client_wire::{ClientWireFormat, ClientWireRequest};
pub use error::ProviderError;
pub use gemini::{GeminiAdapter, GeminiStreamDecoder};
pub use openai_chat::{OpenAiChatAdapter, OpenAiChatStreamDecoder};
pub use request::{AuthHeaders, ProxyRequest};
pub use responses::{ResponsesAdapter, ResponsesStreamDecoder};
pub use sse::{SseFrame, SseFramer};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Canonical names of the built-in protocol families, matched
/// case-insensitively from provider TOML.
pub mod protocols {
    /// OpenAI Chat Completions (`/v1/chat/completions`).
    pub const OPENAI_CHAT_COMPLETIONS: &str = "openai_chat_completions";
    /// Anthropic Messages (`/v1/messages`).
    pub const ANTHROPIC_MESSAGES: &str = "anthropic_messages";
    /// OpenAI Responses (`/v1/responses`).
    pub const OPENAI_RESPONSES: &str = "openai_responses";
    /// Google Gemini GenerateContent
    /// (`/v1beta/models/{model}:generateContent`).
    pub const GEMINI_GENERATE_CONTENT: &str = "gemini_generate_content";

    /// All built-in protocol names.
    pub const BUILTIN: &[&str] = &[
        OPENAI_CHAT_COMPLETIONS,
        ANTHROPIC_MESSAGES,
        OPENAI_RESPONSES,
        GEMINI_GENERATE_CONTENT,
    ];
}

/// An open protocol-family identifier: the canonical (lowercase) name of a
/// protocol registered in a [`ProviderAdapterRegistry`]. Equality with a
/// `&str` compares against the canonical name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProtocolName(Arc<str>);

impl ProtocolName {
    /// Canonicalize a configured protocol name (lowercased, trimmed).
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self(Arc::from(name.trim().to_ascii_lowercase().as_str()))
    }

    /// The canonical name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProtocolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl PartialEq<str> for ProtocolName {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for ProtocolName {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Everything an adapter needs to encode a request for a specific provider.
/// The `api_key` field is redacted in [`fmt::Debug`] output so logs never leak it.
#[derive(Clone)]
pub struct ProviderAdapterTarget {
    /// Logical provider name (for logging/metrics).
    pub provider_name: String,
    /// Adapter name from config.
    pub adapter_name: String,
    /// Which protocol this adapter speaks.
    pub protocol: ProtocolName,
    /// Full upstream endpoint URL (possibly with a `{model}` placeholder).
    pub endpoint: String,
    /// How to authenticate with the upstream.
    pub auth_style: AuthStyle,
    /// API key for the upstream (wrapped in [`SecretString`]). Redacted in Debug output.
    pub api_key: SecretString,
    /// The model the client asked for.
    pub requested_model: String,
    /// The model to send upstream (may differ due to aliasing).
    pub upstream_model: String,
    /// Optional static headers from the adapter config (e.g. `anthropic-version`),
    /// `Arc`-shared so building a target is a refcount bump, not a map clone.
    pub headers: Arc<HashMap<String, String>>,
    /// Per-route `lossy` opt-in: cross-format fields with no provider
    /// equivalent are dropped with a logged warning instead of rejecting the
    /// request.
    pub lossy: bool,
    /// Inbound header names this provider allows to forward upstream;
    /// `Arc`-shared straight from provider config.
    pub forward_inbound_headers: Arc<Vec<String>>,
    /// Prompt-cache hygiene (plan 06 steps 6-7): the opt-in provider-edge
    /// edits — volatile-byte removal and marker injection — applied as the
    /// last mutation phase before serialization.
    pub prompt_cache: Option<Arc<llm_proxy_core::PromptCacheConfig>>,
}

impl fmt::Debug for ProviderAdapterTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderAdapterTarget")
            .field("provider_name", &self.provider_name)
            .field("adapter_name", &self.adapter_name)
            .field("protocol", &self.protocol)
            .field("endpoint", &self.endpoint_without_query())
            .field("auth_style", &self.auth_style)
            .field("api_key", &"[REDACTED]")
            .field("requested_model", &self.requested_model)
            .field("upstream_model", &self.upstream_model)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("lossy", &self.lossy)
            .finish()
    }
}

impl ProviderAdapterTarget {
    /// The endpoint with any query string removed, for logs.
    #[must_use]
    pub fn endpoint_without_query(&self) -> &str {
        request::endpoint_without_query(&self.endpoint)
    }
}

/// A provider codec: encodes core requests into provider wire bytes and
/// decodes provider responses and already-framed SSE events back into core
/// types. The built-in set lives in this crate; SDK users implement this trait
/// to add providers.
pub trait ProviderAdapter: fmt::Debug + Send + Sync {
    /// The canonical protocol name this adapter handles; must match the name
    /// it is registered under.
    fn protocol(&self) -> ProtocolName;

    /// Encode a `CoreRequest` into a transport-ready `ProxyRequest`.
    ///
    /// When the client's wire format matches this adapter's protocol the raw
    /// client body is forwarded with only the adapter's surgical edits
    /// (same-format passthrough, conflict 10); otherwise the normalized core
    /// request is encoded and every client field the core does not model is
    /// settled by the cross-format field policy (carried, mapped, or rejected
    /// by name).
    fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &ClientWireRequest,
    ) -> Result<ProxyRequest, ProviderError>;

    /// Decode a non-streaming provider response body into a `CoreResponse`.
    fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError>;

    /// Create a new decoder for one upstream stream.
    fn new_stream_decoder(
        &self,
        target: &ProviderAdapterTarget,
    ) -> Box<dyn ProviderStreamDecoder + Send>;
}

impl ProviderAdapter for anthropic::AnthropicAdapter {
    fn protocol(&self) -> ProtocolName {
        ProtocolName::new(protocols::ANTHROPIC_MESSAGES)
    }

    fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &ClientWireRequest,
    ) -> Result<ProxyRequest, ProviderError> {
        Self::encode_request(self, core, target, client)
    }

    fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        Self::decode_response(self, bytes, target)
    }

    fn new_stream_decoder(
        &self,
        target: &ProviderAdapterTarget,
    ) -> Box<dyn ProviderStreamDecoder + Send> {
        Box::new(Self::new_stream_decoder(self, target))
    }
}

impl ProviderAdapter for openai_chat::OpenAiChatAdapter {
    fn protocol(&self) -> ProtocolName {
        ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS)
    }

    fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &ClientWireRequest,
    ) -> Result<ProxyRequest, ProviderError> {
        Self::encode_request(self, core, target, client)
    }

    fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        Self::decode_response(self, bytes, target)
    }

    fn new_stream_decoder(
        &self,
        target: &ProviderAdapterTarget,
    ) -> Box<dyn ProviderStreamDecoder + Send> {
        Box::new(Self::new_stream_decoder(self, target))
    }
}

impl ProviderAdapter for responses::ResponsesAdapter {
    fn protocol(&self) -> ProtocolName {
        ProtocolName::new(protocols::OPENAI_RESPONSES)
    }

    fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &ClientWireRequest,
    ) -> Result<ProxyRequest, ProviderError> {
        Self::encode_request(self, core, target, client)
    }

    fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        Self::decode_response(self, bytes, target)
    }

    fn new_stream_decoder(
        &self,
        target: &ProviderAdapterTarget,
    ) -> Box<dyn ProviderStreamDecoder + Send> {
        Box::new(Self::new_stream_decoder(self, target))
    }
}

impl ProviderAdapter for gemini::GeminiAdapter {
    fn protocol(&self) -> ProtocolName {
        ProtocolName::new(protocols::GEMINI_GENERATE_CONTENT)
    }

    fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &ClientWireRequest,
    ) -> Result<ProxyRequest, ProviderError> {
        Self::encode_request(self, core, target, client)
    }

    fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        Self::decode_response(self, bytes, target)
    }

    fn new_stream_decoder(
        &self,
        target: &ProviderAdapterTarget,
    ) -> Box<dyn ProviderStreamDecoder + Send> {
        Box::new(Self::new_stream_decoder(self, target))
    }
}

/// Decodes already-framed SSE events from an upstream provider into core
/// events; must not parse raw network chunks.
pub trait ProviderStreamDecoder: fmt::Debug + Send {
    /// Decode a single SSE frame into zero or more core events.
    fn decode_frame(&mut self, frame: &SseFrame) -> Result<Vec<CoreEvent>, ProviderError>;

    /// Flush any remaining buffered state at stream end.
    fn finish(&mut self) -> Result<Vec<CoreEvent>, ProviderError>;

    /// Whether the provider itself signalled the end of the response (a
    /// finish reason, `message_stop`, or `response.completed`). `false` after
    /// a stream the upstream cut off, whose synthetic terminal from
    /// [`ProviderStreamDecoder::finish`] must be reported as a failure.
    fn saw_terminal(&self) -> bool;
}

/// Failures from registering an adapter.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RegistrationError {
    /// A different adapter is already registered under this protocol name.
    #[error("protocol \"{0}\" already has a registered adapter")]
    Duplicate(String),
}

/// Registry of provider adapters, keyed by canonical protocol name. Starts
/// from [`ProviderAdapterRegistry::builtin`] and accepts further protocols
/// via [`ProviderAdapterRegistry::register`], so the provider set is open.
#[derive(Debug, Clone, Default)]
pub struct ProviderAdapterRegistry {
    adapters: HashMap<String, Arc<dyn ProviderAdapter>>,
}

impl ProviderAdapterRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with all built-in adapters.
    #[must_use]
    pub fn builtin() -> Self {
        let builtins: [Arc<dyn ProviderAdapter>; 4] = [
            Arc::new(openai_chat::OpenAiChatAdapter),
            Arc::new(anthropic::AnthropicAdapter::new()),
            Arc::new(responses::ResponsesAdapter),
            Arc::new(gemini::GeminiAdapter),
        ];
        let mut registry = Self::new();
        for adapter in builtins {
            let name = adapter.protocol().as_str().to_owned();
            registry.adapters.insert(name, adapter);
        }
        registry
    }

    /// Register an adapter under its [`ProviderAdapter::protocol`] name; an
    /// SDK user adds providers here. Fails when the name is taken.
    pub fn register(&mut self, adapter: Arc<dyn ProviderAdapter>) -> Result<(), RegistrationError> {
        let name = adapter.protocol().as_str().to_owned();
        if self.adapters.contains_key(&name) {
            return Err(RegistrationError::Duplicate(name));
        }
        self.adapters.insert(name, adapter);
        Ok(())
    }

    /// The canonical protocol names, sorted for stable output.
    #[must_use]
    pub fn protocol_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.adapters.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// Returns `true` if the given protocol name (case-insensitive) is
    /// registered.
    #[must_use]
    pub fn has_protocol_name(&self, protocol: &str) -> bool {
        self.adapters
            .contains_key(&protocol.trim().to_ascii_lowercase())
    }

    /// Look up an adapter by protocol name (case-insensitive).
    #[must_use]
    pub fn get(&self, protocol: &str) -> Option<Arc<dyn ProviderAdapter>> {
        self.adapters
            .get(&protocol.trim().to_ascii_lowercase())
            .cloned()
    }
}

/// Expands a `{model}` placeholder in a provider endpoint template.
pub fn expand_url_template(
    template: &str,
    target: &ProviderAdapterTarget,
) -> Result<String, ProviderError> {
    let Some(model_pos) = template.find("{model}") else {
        return Ok(template.to_owned());
    };

    let configured_model = &target.upstream_model;
    // `model_pos` came from `find`, so the prefix always exists; the
    // checked forms keep the slicing rule's letter.
    let prefix = template.get(..model_pos).unwrap_or(template);
    let has_models_prefix = prefix.ends_with("models/");
    let model = configured_model
        .strip_prefix("models/")
        .filter(|_| has_models_prefix)
        .unwrap_or(configured_model);
    if model == ".." || model == "." {
        return Err(ProviderError::InvalidConfig(format!(
            "upstream_model {:?} is a path traversal pattern; refusing to interpolate into URL",
            model
        )));
    }
    let valid_resource_name = model
        .strip_prefix("models/")
        .is_some_and(is_safe_model_component);
    if !is_safe_model_component(model) && !valid_resource_name {
        return Err(ProviderError::InvalidConfig(format!(
            "upstream_model {:?} contains unsafe characters; refusing to interpolate into URL",
            model
        )));
    }
    let mut result = String::with_capacity(template.len() + model.len());
    result.push_str(prefix);
    result.push_str(model);
    if let Some(tail) = template.get(model_pos + "{model}".len()..) {
        result.push_str(tail);
    }
    Ok(result)
}

fn is_safe_model_component(model: &str) -> bool {
    !model.is_empty()
        && model != "."
        && model != ".."
        && model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
}

pub(crate) fn map_openai_finish_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" | "tool_use" => StopReason::ToolUse,
        "content_filter" => StopReason::Refusal,
        _ => StopReason::Unknown(reason.to_owned()),
    }
}

pub(crate) fn map_gemini_finish_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::EndTurn,
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY" | "RECITATION" => StopReason::Refusal,
        _ => StopReason::Unknown(reason.to_owned()),
    }
}

/// Model-visible marker put in front of a failed tool result's text on wire
/// formats whose tool-result item has no error flag (Chat Completions `tool`
/// messages, Responses `function_call_output` items).
pub(crate) const TOOL_ERROR_PREFIX: &str = "[tool error] ";

/// Marks `text` as a failed tool result when `is_error` is set; a failed
/// result with no text becomes the bare marker.
pub(crate) fn mark_tool_error(text: String, is_error: bool) -> String {
    if !is_error {
        return text;
    }
    if text.is_empty() {
        return TOOL_ERROR_PREFIX.trim_end().to_owned();
    }
    format!("{TOOL_ERROR_PREFIX}{text}")
}

/// Builds core usage from the OpenAI/DeepSeek chat usage shape.
///
/// `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` are DeepSeek fields
/// where `prompt_tokens = hit + miss`: hit tokens map to cache reads and the
/// miss tokens stay plain input — they are **not** cache creation. OpenAI's
/// own `prompt_tokens_details.cached_tokens` maps to cache reads the same way.
pub(crate) fn build_usage_from_openai(usage: &llm_proxy_protocol::openai::UsageInfo) -> Usage {
    let cache_read = usage.prompt_cache_hit_tokens.or_else(|| {
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
    });
    let cache_read_i64 = cache_read.unwrap_or(0) as i64;
    let input = (usage.prompt_tokens as i64) - cache_read_i64;
    let input_tokens = i32::try_from(input.max(0)).unwrap_or(i32::MAX);
    let reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens);

    Usage {
        input_tokens,
        output_tokens: usage.completion_tokens,
        reasoning_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cache_read,
        provenance: llm_proxy_protocol::core::UsageProvenance::ProviderReported,
    }
}

pub(crate) fn build_proxy_request(
    body: Vec<u8>,
    target: &ProviderAdapterTarget,
    stream: bool,
    url: String,
    client: &ClientWireRequest,
) -> ProxyRequest {
    let mut extra_headers = (*target.headers).clone();
    if target.protocol == protocols::ANTHROPIC_MESSAGES
        && !extra_headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("anthropic-version"))
    {
        extra_headers.insert("anthropic-version".to_owned(), ANTHROPIC_VERSION.to_owned());
    }

    // Static config headers win over forwarded inbound headers with the same
    // name (the transport applies these last).
    let forwarded_headers = client
        .forwarded_headers
        .iter()
        .filter(|(name, _)| {
            !extra_headers
                .keys()
                .any(|configured| configured.eq_ignore_ascii_case(name))
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();

    ProxyRequest {
        url,
        auth: AuthHeaders {
            style: target.auth_style,
            api_key: target.api_key.clone(),
        },
        body,
        stream,
        extra_headers,
        forwarded_headers,
    }
}

pub(crate) fn response_model_ref(target: &ProviderAdapterTarget) -> ModelRef {
    let upstream = if target.upstream_model != target.requested_model {
        Some(target.upstream_model.clone())
    } else {
        None
    };
    ModelRef {
        requested: target.requested_model.clone(),
        upstream,
    }
}

pub(crate) fn truncate_str_safe(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        let mut end = max_len;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        // `end` is boundary-adjusted below the length, so the head always
        // exists; the checked form keeps the slicing rule's letter.
        s.get(..end).unwrap_or(s)
    }
}

#[cfg(test)]
mod usage_tests {
    use llm_proxy_protocol::openai::{CompletionTokensDetails, PromptTokensDetails, UsageInfo};

    use super::build_usage_from_openai;

    #[test]
    fn deepseek_hit_and_miss_map_to_cache_read_and_plain_input() {
        // DeepSeek guarantees prompt_tokens = hit + miss: hit tokens are
        // cache reads, miss tokens are plain input, and nothing here is
        // cache creation.
        let usage = build_usage_from_openai(&UsageInfo {
            prompt_tokens: 30,
            completion_tokens: 7,
            total_tokens: 37,
            prompt_cache_hit_tokens: Some(10),
            prompt_cache_miss_tokens: Some(20),
            prompt_tokens_details: None,
            completion_tokens_details: None,
        });

        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.cache_read_input_tokens, Some(10));
        assert_eq!(
            usage.cache_creation_input_tokens, None,
            "miss tokens must never surface as cache creation"
        );
        assert_eq!(
            usage.provenance,
            llm_proxy_protocol::core::UsageProvenance::ProviderReported
        );
    }

    #[test]
    fn openai_cached_tokens_detail_maps_to_cache_read() {
        let usage = build_usage_from_openai(&UsageInfo {
            prompt_tokens: 50,
            completion_tokens: 5,
            total_tokens: 55,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(12),
            }),
            completion_tokens_details: None,
        });

        assert_eq!(usage.input_tokens, 38);
        assert_eq!(usage.cache_read_input_tokens, Some(12));
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn completion_reasoning_tokens_map_to_core() {
        let with_detail = build_usage_from_openai(&UsageInfo {
            prompt_tokens: 10,
            completion_tokens: 30,
            total_tokens: 40,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
            prompt_tokens_details: None,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(9),
            }),
        });
        assert_eq!(with_detail.reasoning_tokens, Some(9));

        let without_detail = build_usage_from_openai(&UsageInfo {
            completion_tokens_details: None,
            ..with_detail_producer()
        });
        assert_eq!(without_detail.reasoning_tokens, None);
    }

    fn with_detail_producer() -> UsageInfo {
        UsageInfo {
            prompt_tokens: 10,
            completion_tokens: 30,
            total_tokens: 40,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }

    #[test]
    fn cached_tokens_larger_than_prompt_does_not_underflow() {
        let usage = build_usage_from_openai(&UsageInfo {
            prompt_tokens: 5,
            completion_tokens: 1,
            total_tokens: 6,
            prompt_cache_hit_tokens: None,
            prompt_cache_miss_tokens: None,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(9),
            }),
            completion_tokens_details: None,
        });

        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, Some(9));
    }
}
