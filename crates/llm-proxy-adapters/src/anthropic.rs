//! Anthropic Messages API provider adapter: translates core requests,
//! responses, and event streams to and from the Anthropic wire format.

use std::sync::LazyLock;

use llm_proxy_protocol::anthropic::{MessageEvent, MessageResponse};
use llm_proxy_protocol::core::{
    ContentKind, CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreResponse, CoreRole,
    CoreTool, CoreToolChoice, ModelRef, StopReason, UsageProvenance,
};

use super::{
    ProviderAdapterTarget, ProviderStreamDecoder, build_proxy_request, expand_url_template,
    response_model_ref,
};
use crate::ProtocolName;
use crate::error::ProviderError;
use crate::protocols;
use crate::sse::SseFrame;

const ANTHROPIC_DEFAULT_MAX_TOKENS: i32 = 4096;

#[inline]
#[expect(
    clippy::expect_used,
    reason = "every caller passes a json! {{...}} literal, which is always an object; the \
              invariant is local to this module's call sites"
)]
fn json_object(val: &mut serde_json::Value) -> &mut serde_json::Map<String, serde_json::Value> {
    val.as_object_mut()
        .expect("json! macro with {{...}} always produces a JSON object")
}

#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time-constant regex literal; a failure is a programming \
              error caught the moment this static is first used, not a runtime input"
)]
static INVALID_TOOL_USE_ID_CHAR: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"[^A-Za-z0-9_]").expect("valid regex"));

fn sanitize_tool_name(name: &str) -> (String, bool) {
    use std::fmt::Write;
    let sentinel = "__llmp_";
    let max_len = 128;
    let mut result = String::with_capacity(name.len() * 3 + sentinel.len());
    let mut changed = false;

    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            result.push(ch);
        } else {
            let mut buf = [0u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            #[expect(
                clippy::expect_used,
                reason = "writing into a String cannot fail (no formatter errors on a String)"
            )]
            for &byte in encoded.as_bytes() {
                write!(result, "_0x{:02x}_", byte).expect("write! to String is infallible");
            }
            changed = true;
        }
    }

    if changed {
        result.insert_str(0, sentinel);
    }

    if result.len() > max_len {
        let mut end = max_len;
        while !result.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        if end >= 5 {
            // `end` is boundary-adjusted below the length, so the head
            // always exists; the checked form keeps the slicing rule's
            // letter.
            if let Some(pos) = result.get(..end).unwrap_or(&result).rfind("_0x") {
                let seq_end = pos + 6;
                if seq_end > end {
                    end = pos;
                }
            }
        }
        while !result.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        result.truncate(end);
        if result.is_empty() {
            return ("tool".to_owned(), true);
        }
    }

    if result.is_empty() {
        return ("tool".to_owned(), name != "tool");
    }

    (result, changed)
}

fn desanitize_tool_name(name: &str) -> std::borrow::Cow<'_, str> {
    const SENTINEL: &str = "__llmp_";

    if !name.starts_with(SENTINEL) {
        return std::borrow::Cow::Borrowed(name);
    }

    // `starts_with` above makes the strip always succeed; the checked form
    // keeps the slicing rule's letter.
    let Some(stripped) = name.get(SENTINEL.len()..) else {
        return std::borrow::Cow::Borrowed(name);
    };
    if !stripped.contains("_0x") {
        return std::borrow::Cow::Owned(stripped.to_owned());
    }

    let mut result = Vec::<u8>::with_capacity(stripped.len());
    let bytes = stripped.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // A `_0xHH_` escape decodes to one byte; anything else copies
        // through. Every access is bounds-checked by construction here.
        if bytes.get(i) == Some(&b'_')
            && bytes.get(i + 1) == Some(&b'0')
            && bytes.get(i + 2) == Some(&b'x')
            && bytes.get(i + 5) == Some(&b'_')
            && let (Some(&hi), Some(&lo)) = (bytes.get(i + 3), bytes.get(i + 4))
            && let Some(byte_val) = hex_byte(hi, lo)
        {
            result.push(byte_val);
            i += 6;
            continue;
        }
        if let Some(&byte) = bytes.get(i) {
            result.push(byte);
        }
        i += 1;
    }

    match String::from_utf8(result) {
        Ok(s) => std::borrow::Cow::Owned(s),
        Err(e) => std::borrow::Cow::Owned(String::from_utf8_lossy(e.as_bytes()).into_owned()),
    }
}

fn hex_byte(hi: u8, lo: u8) -> Option<u8> {
    let hi_val = hex_digit(hi)?;
    let lo_val = hex_digit(lo)?;
    Some(hi_val << 4 | lo_val)
}

const fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn sanitize_tool_use_id(id: &str) -> String {
    let sanitized: String = INVALID_TOOL_USE_ID_CHAR.replace_all(id, "_").into_owned();
    if sanitized.len() > 256 {
        let mut end = 256;
        while !sanitized.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        // `end` is boundary-adjusted below the length, so the head always
        // exists; the checked form keeps the slicing rule's letter.
        match sanitized.get(..end) {
            Some(head) => head.to_owned(),
            None => sanitized,
        }
    } else if sanitized.is_empty() {
        "tool_result".to_owned()
    } else {
        sanitized
    }
}

/// Adapter for the Anthropic Messages API.
#[derive(Debug, Clone, Default)]
pub struct AnthropicAdapter;

impl AnthropicAdapter {
    /// Create a new Anthropic adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// Stateful stream decoder translating Anthropic SSE frames into `CoreEvent`s.
#[derive(Debug)]
pub struct AnthropicStreamDecoder {
    model_ref: ModelRef,
    started: bool,
    current_block_kind: ContentKind,
    tool_blocks: Vec<usize>,
    tool_blocks_closed: Vec<bool>,
    stop_sent: bool,
    saw_terminal: bool,
    /// Input-side usage captured from `message_start`; `message_delta` usage
    /// objects carry only the cumulative `output_tokens`, so the input count
    /// must be merged in from here.
    start_usage: Option<llm_proxy_protocol::anthropic::Usage>,
}

fn tool_field_or_sentinel(value: Option<&str>, field: &str, sentinel: &str) -> String {
    match value {
        Some(v) => v.to_owned(),
        None => {
            tracing::warn!(
                field,
                "Anthropic: tool_use block missing field, using sentinel"
            );
            sentinel.to_owned()
        }
    }
}

fn push_content_block_delta(event: &MessageEvent, events: &mut Vec<CoreEvent>) {
    let idx = event.index.unwrap_or(0);
    let Some(ref delta) = event.delta else {
        return;
    };
    match delta.r#type.as_deref().unwrap_or("") {
        "text_delta" => {
            if let Some(ref text) = delta.text
                && !text.is_empty()
            {
                events.push(CoreEvent::TextDelta {
                    index: idx,
                    text: text.clone(),
                });
            }
        }
        "thinking_delta" => {
            if let Some(ref text) = delta.thinking
                && !text.is_empty()
            {
                events.push(CoreEvent::ThinkingDelta {
                    index: idx,
                    text: text.clone(),
                });
            }
        }
        "input_json_delta" => {
            if let Some(ref partial) = delta.partial_json
                && !partial.is_empty()
            {
                events.push(CoreEvent::ToolCallDelta {
                    index: idx,
                    args_delta: partial.clone(),
                });
            }
        }
        _ => {}
    }
}

fn push_stream_error(event: &MessageEvent, events: &mut Vec<CoreEvent>) {
    let Some(ref err) = event.error else {
        return;
    };
    let kind = match err.r#type.as_str() {
        "rate_limit_error" => llm_proxy_protocol::core::CoreStreamErrorKind::RateLimit,
        "authentication_error" => llm_proxy_protocol::core::CoreStreamErrorKind::Authentication,
        "invalid_request_error" => llm_proxy_protocol::core::CoreStreamErrorKind::InvalidRequest,
        "permission_error" => llm_proxy_protocol::core::CoreStreamErrorKind::Permission,
        _ => llm_proxy_protocol::core::CoreStreamErrorKind::Upstream,
    };
    events.push(CoreEvent::Error {
        error: llm_proxy_protocol::core::CoreStreamError::new(kind, err.message.clone()),
    });
}

impl AnthropicStreamDecoder {
    fn push_content_block_start(&mut self, event: &MessageEvent, events: &mut Vec<CoreEvent>) {
        let idx = event.index.unwrap_or(0);
        let Some(ref block) = event.content_block else {
            return;
        };
        match block.r#type.as_str() {
            "thinking" => {
                self.current_block_kind = ContentKind::Thinking;
                events.push(CoreEvent::ContentStart {
                    index: idx,
                    kind: ContentKind::Thinking,
                });
            }
            "tool_use" => {
                self.current_block_kind = ContentKind::ToolUse;
                self.tool_blocks.push(idx);
                self.tool_blocks_closed.push(false);
                let id = tool_field_or_sentinel(block.id.as_deref(), "id", "<unknown_tool_id>");
                let name = tool_field_or_sentinel(block.name.as_deref(), "name", "<unknown_tool>");
                let original_name = desanitize_tool_name(&name).into_owned();
                events.push(CoreEvent::ToolCallStart {
                    index: idx,
                    id,
                    name: original_name,
                });
            }
            _ => {
                self.current_block_kind = ContentKind::Text;
                events.push(CoreEvent::ContentStart {
                    index: idx,
                    kind: ContentKind::Text,
                });
            }
        }
    }

    fn push_content_block_stop(&mut self, event: &MessageEvent, events: &mut Vec<CoreEvent>) {
        let idx = event.index.unwrap_or(0);
        if self.current_block_kind == ContentKind::ToolUse {
            events.push(CoreEvent::ToolCallStop { index: idx });
            let closed_slot = self
                .tool_blocks
                .iter()
                .position(|&i| i == idx)
                .and_then(|tool_idx| self.tool_blocks_closed.get_mut(tool_idx));
            if let Some(closed) = closed_slot {
                *closed = true;
            } else {
                tracing::warn!(
                    idx,
                    "Anthropic: tool block index not found in content_block_stop; \
                     skipping close since tool call was never opened"
                );
            }
        }
        self.current_block_kind = ContentKind::Text;
    }

    fn push_message_delta(&mut self, event: &MessageEvent, events: &mut Vec<CoreEvent>) {
        if let Some(ref usage) = event.usage {
            // The documented message_delta usage carries only output_tokens;
            // input-side counts come from message_start.
            events.push(CoreEvent::UsageDelta {
                usage: merge_delta_usage(self.start_usage.as_ref(), usage),
            });
        }
        if let Some(ref delta) = event.delta
            && let Some(ref reason) = delta.stop_reason
            && !reason.is_empty()
            && !self.stop_sent
        {
            self.stop_sent = true;
            self.saw_terminal = true;
            let stop_reason = map_anthropic_stop_reason(reason);
            let stop_sequence = delta.stop_sequence.clone();
            events.push(CoreEvent::MessageStop {
                stop_reason,
                stop_sequence,
            });
        }
    }
}

impl ProviderStreamDecoder for AnthropicStreamDecoder {
    fn decode_frame(&mut self, frame: &SseFrame) -> Result<Vec<CoreEvent>, ProviderError> {
        let data = frame.data.trim();
        if data.is_empty() {
            return Ok(vec![]);
        }

        let event: MessageEvent = match serde_json::from_str(data) {
            Ok(e) => e,
            Err(_) => {
                let truncated = super::truncate_str_safe(data, 200);
                tracing::warn!(data = truncated, "malformed Anthropic event, skipping");
                return Ok(vec![]);
            }
        };

        let mut events = Vec::new();

        match event.r#type.as_str() {
            "message_start" => {
                if !self.started {
                    self.started = true;
                    let id = event
                        .message
                        .as_ref()
                        .filter(|m| !m.id.is_empty())
                        .map(|m| m.id.clone());
                    if let Some(message) = event.message.as_ref() {
                        self.start_usage = Some(message.usage.clone());
                    }
                    events.push(CoreEvent::MessageStart {
                        id,
                        model: self.model_ref.clone(),
                    });
                }
            }
            "content_block_start" => {
                self.push_content_block_start(&event, &mut events);
            }
            "content_block_delta" => {
                push_content_block_delta(&event, &mut events);
            }
            "content_block_stop" => {
                self.push_content_block_stop(&event, &mut events);
            }
            "message_delta" => {
                self.push_message_delta(&event, &mut events);
            }
            "message_stop" => {
                if !self.stop_sent {
                    self.stop_sent = true;
                    self.saw_terminal = true;
                    events.push(CoreEvent::MessageStop {
                        stop_reason: if !self.tool_blocks.is_empty() {
                            StopReason::ToolUse
                        } else {
                            StopReason::EndTurn
                        },
                        stop_sequence: None,
                    });
                }
            }
            "ping" => {
                events.push(CoreEvent::Ping);
            }
            "error" => {
                push_stream_error(&event, &mut events);
            }
            _ => {}
        }

        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<CoreEvent>, ProviderError> {
        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(CoreEvent::MessageStart {
                id: None,
                model: self.model_ref.clone(),
            });
        }

        for (i, &idx) in self.tool_blocks.iter().enumerate() {
            let closed = self.tool_blocks_closed.get(i).copied().unwrap_or(false);
            if !closed {
                events.push(CoreEvent::ToolCallStop { index: idx });
            }
        }

        if !self.stop_sent {
            self.stop_sent = true;
            events.push(CoreEvent::MessageStop {
                stop_reason: if !self.tool_blocks.is_empty() {
                    StopReason::ToolUse
                } else {
                    StopReason::EndTurn
                },
                stop_sequence: None,
            });
        }

        Ok(events)
    }

    fn saw_terminal(&self) -> bool {
        self.saw_terminal
    }
}

impl AnthropicAdapter {
    /// Encode a core request into an Anthropic Messages request.
    ///
    /// Same-format (Anthropic-dialect client) requests forward the raw client
    /// body with only surgical edits: the model name, reasoning translation,
    /// and stripping fields the Anthropic wire format does not define.
    /// Cross-format requests encode the normalized core request under the
    /// cross-format field policy.
    pub fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &super::client_wire::ClientWireRequest,
    ) -> Result<super::ProxyRequest, ProviderError> {
        let url = expand_url_template(&target.endpoint, target)?;
        let req = if client.same_format_as(&ProtocolName::new(protocols::ANTHROPIC_MESSAGES))
            && let Some(raw) = client.body_object()
        {
            edit_raw_anthropic_request(raw, core, target)?
        } else {
            encode_normalized_request(core, target, client)?
        };
        // Phase 3 (plan 06 step 7): operator hygiene at the provider edge —
        // the last mutation phase before serialization.
        let mut req = req;
        if let Some(hygiene) = &target.prompt_cache {
            super::mutation::apply_prompt_hygiene(&mut req, hygiene, hygiene.inject_markers);
        }
        let body = serde_json::to_vec(&req)?;
        Ok(build_proxy_request(body, target, core.stream, url, client))
    }

    /// Decode an Anthropic Messages response body.
    pub fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        let resp: MessageResponse = serde_json::from_slice(bytes)?;

        let mut content = Vec::new();

        for block in &resp.content {
            match block.r#type.as_str() {
                "text" => {
                    let text = block.text.clone().unwrap_or_default();
                    content.push(CoreContent::Text { text, cache: None });
                }
                "thinking" => {
                    let thinking = block.thinking.clone().unwrap_or_default();
                    content.push(CoreContent::Thinking {
                        text: thinking,
                        signature: block.signature.clone(),
                    });
                }
                "redacted_thinking" => {
                    content.push(CoreContent::RedactedThinking {
                        data: serde_json::json!(block.data.clone().unwrap_or_default()),
                    });
                }
                "tool_use" => {
                    let input = block
                        .input
                        .clone()
                        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                    let name = block.name.clone().unwrap_or_else(|| {
                        tracing::warn!("Anthropic: tool_use block missing name in response decode");
                        "<unknown_tool>".to_owned()
                    });
                    let original_name = desanitize_tool_name(&name).into_owned();
                    let tool_id = block.id.clone().unwrap_or_else(|| {
                        tracing::warn!("Anthropic: tool_use block missing id in response decode");
                        "<unknown_tool_id>".to_owned()
                    });
                    content.push(CoreContent::ToolUse {
                        id: tool_id,
                        name: original_name,
                        input,
                        signature: None,
                    });
                }
                _ => {
                    let text = block.text.clone().unwrap_or_default();
                    if !text.is_empty() {
                        content.push(CoreContent::Text { text, cache: None });
                    }
                }
            }
        }

        if content.is_empty() {
            content.push(CoreContent::Text {
                text: String::new(),
                cache: None,
            });
        }

        let stop_reason = resp
            .stop_reason
            .as_deref()
            .map(map_anthropic_stop_reason)
            .unwrap_or(StopReason::Unknown("absent".to_owned()));

        let usage = build_anthropic_usage(&resp.usage);

        Ok(CoreResponse {
            id: if resp.id.is_empty() {
                None
            } else {
                Some(resp.id)
            },
            model: response_model_ref(target),
            content,
            stop_reason,
            stop_sequence: resp.stop_sequence,
            usage,
            provider_meta: serde_json::Map::new(),
            cost: None,
        })
    }

    /// Create a new stream decoder.
    pub fn new_stream_decoder(&self, target: &ProviderAdapterTarget) -> AnthropicStreamDecoder {
        AnthropicStreamDecoder {
            model_ref: response_model_ref(target),
            started: false,
            current_block_kind: ContentKind::Text,
            tool_blocks: Vec::new(),
            tool_blocks_closed: Vec::new(),
            stop_sent: false,
            saw_terminal: false,
            start_usage: None,
        }
    }
}

/// Same-format passthrough: apply only the edits the proxy needs to the raw
/// Anthropic client body. Every other field — including fields this proxy
/// does not know — survives byte-for-byte.
fn edit_raw_anthropic_request(
    raw: &serde_json::Map<String, serde_json::Value>,
    core: &CoreRequest,
    target: &ProviderAdapterTarget,
) -> Result<serde_json::Value, ProviderError> {
    let mut obj = raw.clone();

    // Edit 1: model aliasing.
    obj.insert(
        "model".to_owned(),
        serde_json::Value::String(target.upstream_model.clone()),
    );

    // Edit 2: reasoning parameters are translated, never forwarded raw. An
    // OpenAI-style `reasoning_effort` is not an Anthropic field (the typed
    // decode leaves it out of the core, so read it from the raw body); a
    // `thinking` object is clamped to the model's limits.
    let mut sampling = core.sampling.clone();
    if sampling.reasoning_effort.is_none()
        && let Some(effort) = obj
            .get("reasoning_effort")
            .and_then(serde_json::Value::as_str)
    {
        sampling.reasoning_effort = Some(effort.to_owned());
    }
    obj.remove("reasoning_effort");
    obj.remove("thinking");
    if let Some(thinking) = super::reasoning::anthropic_thinking(
        &sampling,
        &target.upstream_model,
        sampling.max_tokens,
        target.lossy,
    )? {
        obj.insert("thinking".to_owned(), thinking);
    }

    // Edit 3: strip foreign fields the proxy itself emits on client-facing
    // tool_use blocks (Gemini thought signatures); Anthropic rejects unknown
    // block fields.
    if let Some(serde_json::Value::Array(messages)) = obj.get_mut("messages") {
        for message in messages
            .iter_mut()
            .filter_map(serde_json::Value::as_object_mut)
        {
            let Some(serde_json::Value::Array(blocks)) = message.get_mut("content") else {
                continue;
            };
            for block in blocks
                .iter_mut()
                .filter_map(serde_json::Value::as_object_mut)
            {
                if block.get("type").and_then(serde_json::Value::as_str) == Some("tool_use") {
                    block.remove("signature");
                }
            }
        }
    }

    Ok(serde_json::Value::Object(obj))
}

/// Cross-format encode: build the Anthropic request from the normalized core
/// request, settle the client's leftover fields through the field policy, and
/// translate reasoning parameters.
fn encode_normalized_request(
    core: &CoreRequest,
    target: &ProviderAdapterTarget,
    client: &super::client_wire::ClientWireRequest,
) -> Result<serde_json::Value, ProviderError> {
    let carried = super::client_wire::apply_field_policy(
        client,
        &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
        target.lossy,
    )?;
    super::client_wire::check_cross_format_content(
        core,
        &ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
        target.lossy,
    )?;

    let system = encode_system(&core.system, &core.messages);
    let messages = encode_messages(&core.messages);
    let tools = encode_tools_for_target(core, client, target);
    let tool_choice = encode_tool_choice_for_target(core, client, target)?;

    let mut req = serde_json::json!({
        "model": target.upstream_model,
        "max_tokens": core.sampling.max_tokens.unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS),
        "messages": messages,
    });

    let obj = json_object(&mut req);

    if let Some(sys) = system {
        obj.insert("system".to_owned(), sys);
    }
    if core.stream {
        obj.insert("stream".to_owned(), serde_json::json!(true));
    }
    if !tools.is_empty() {
        obj.insert("tools".to_owned(), serde_json::Value::Array(tools));
    }
    if let Some(temp) = core.sampling.temperature {
        obj.insert("temperature".to_owned(), serde_json::json!(temp));
    }
    if let Some(top_p) = core.sampling.top_p {
        obj.insert("top_p".to_owned(), serde_json::json!(top_p));
    }
    if let Some(top_k) = core.sampling.top_k {
        obj.insert("top_k".to_owned(), serde_json::json!(top_k));
    }
    if let Some(ref stop) = core.sampling.stop
        && !stop.is_empty()
    {
        let seqs: Vec<serde_json::Value> = stop
            .iter()
            .map(|s| serde_json::Value::String(s.clone()))
            .collect();
        obj.insert("stop_sequences".to_owned(), serde_json::Value::Array(seqs));
    }
    if let Some(ref user_id) = core.metadata.user_id {
        obj.insert(
            "metadata".to_owned(),
            serde_json::json!({"user_id": user_id}),
        );
    }
    if let Some(thinking) = super::reasoning::anthropic_thinking(
        &core.sampling,
        &target.upstream_model,
        // The clamp must run against the max_tokens the body actually emits:
        // a cross-format client that omitted max tokens still gets the
        // default ceiling below, so the budget can never exceed it.
        Some(
            core.sampling
                .max_tokens
                .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS),
        ),
        target.lossy,
    )? {
        obj.insert("thinking".to_owned(), thinking);
    }
    if let Some(tc) = tool_choice {
        obj.insert("tool_choice".to_owned(), tc);
    }
    for (name, value) in carried {
        obj.insert(name, value);
    }

    Ok(req)
}

/// Tools for the cross-format path: plain function tools encode as usual;
/// server tools only survive on lossy routes, where they are dropped (with a
/// warning from the field policy) rather than mis-encoded as functions.
fn encode_tools_for_target(
    core: &CoreRequest,
    client: &super::client_wire::ClientWireRequest,
    target: &ProviderAdapterTarget,
) -> Vec<serde_json::Value> {
    if !(target.lossy && super::client_wire::anthropic_body_has_server_tools(&client.body)) {
        return encode_tools(&core.tools);
    }
    let server_tool_names: Vec<String> = client
        .body
        .as_object()
        .and_then(|obj| obj.get("tools"))
        .and_then(serde_json::Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    let obj = tool.as_object()?;
                    if !obj.contains_key("type") {
                        return None;
                    }
                    obj.get("name")?.as_str().map(str::to_owned)
                })
                .collect()
        })
        .unwrap_or_default();
    encode_tools(
        &core
            .tools
            .iter()
            .filter(|tool| !server_tool_names.contains(&tool.name))
            .cloned()
            .collect::<Vec<_>>(),
    )
}

/// `tool_choice` for the cross-format path, mapping the OpenAI Chat
/// `parallel_tool_calls` knob onto Anthropic's `disable_parallel_tool_use`.
/// An unmapped core variant rejects (or drops under `target.lossy`) through
/// the shared `resolve_tool_choice` policy.
fn encode_tool_choice_for_target(
    core: &CoreRequest,
    client: &super::client_wire::ClientWireRequest,
    target: &ProviderAdapterTarget,
) -> Result<Option<serde_json::Value>, ProviderError> {
    let mut encoded = match core.tool_choice.as_ref() {
        Some(tc) => super::client_wire::resolve_tool_choice(
            tc,
            target.lossy,
            protocols::ANTHROPIC_MESSAGES,
            tool_choice_value,
        )?,
        None => None,
    };
    let parallel = client
        .body
        .as_object()
        .and_then(|obj| obj.get("parallel_tool_calls"))
        .and_then(serde_json::Value::as_bool);
    let Some(parallel) = parallel else {
        return Ok(encoded);
    };
    let disable = serde_json::json!(!parallel);
    match encoded.as_mut().and_then(serde_json::Value::as_object_mut) {
        Some(choice) => {
            choice.insert("disable_parallel_tool_use".to_owned(), disable);
        }
        None => {
            encoded = Some(serde_json::json!({
                "type": "auto",
                "disable_parallel_tool_use": disable,
            }));
        }
    }
    Ok(encoded)
}

fn map_anthropic_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        "stop_sequence" => StopReason::StopSequence,
        "refusal" => StopReason::Refusal,
        _ => StopReason::Unknown(reason.to_owned()),
    }
}

/// Builds core usage from resolved Anthropic usage fields; the cache-creation
/// total prefers the plain field and falls back to the 5m+1h split.
fn core_usage_from_fields(
    input_tokens: i32,
    output_tokens: i32,
    cache_creation_plain: Option<i32>,
    cache_creation_split: Option<&llm_proxy_protocol::anthropic::CacheCreation>,
    cache_read: Option<i32>,
) -> llm_proxy_protocol::core::Usage {
    let cache_creation = cache_creation_plain.or_else(|| {
        cache_creation_split
            .map(|split| split.total_input_tokens())
            .filter(|total| *total != 0)
    });
    llm_proxy_protocol::core::Usage {
        input_tokens,
        output_tokens,
        reasoning_tokens: None,
        cache_creation_input_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
        provenance: UsageProvenance::ProviderReported,
    }
}

/// Fills a `message_delta` usage object with the input-side counts captured
/// at `message_start`. The documented delta usage carries only the cumulative
/// `output_tokens`; every other field takes the `message_start` value unless
/// the provider explicitly repeated it.
fn merge_delta_usage(
    start: Option<&llm_proxy_protocol::anthropic::Usage>,
    delta: &llm_proxy_protocol::anthropic::Usage,
) -> llm_proxy_protocol::core::Usage {
    let Some(start) = start else {
        return build_anthropic_usage(delta);
    };
    core_usage_from_fields(
        if delta.input_tokens == 0 {
            start.input_tokens
        } else {
            delta.input_tokens
        },
        if delta.output_tokens == 0 {
            start.output_tokens
        } else {
            delta.output_tokens
        },
        delta
            .cache_creation_input_tokens
            .or(start.cache_creation_input_tokens),
        delta
            .cache_creation
            .as_ref()
            .or(start.cache_creation.as_ref()),
        delta
            .cache_read_input_tokens
            .or(start.cache_read_input_tokens),
    )
}

fn build_anthropic_usage(
    usage: &llm_proxy_protocol::anthropic::Usage,
) -> llm_proxy_protocol::core::Usage {
    core_usage_from_fields(
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_creation.as_ref(),
        usage.cache_read_input_tokens,
    )
}

/// Encodes a core cache-control directive into the Anthropic wire shape,
/// including the `ttl` window.
fn encode_cache_control_marker(
    cache: &llm_proxy_protocol::core::CacheControl,
) -> serde_json::Value {
    let mut marker = serde_json::json!({"type": cache.r#type.as_str()});
    if let Some(ttl) = &cache.ttl {
        json_object(&mut marker).insert("ttl".to_owned(), serde_json::Value::String(ttl.clone()));
    }
    marker
}

fn encode_content_blocks(content: &[CoreContent]) -> serde_json::Value {
    let blocks: Vec<serde_json::Value> = content
        .iter()
        .filter_map(|c| match c {
            CoreContent::Text { text, cache } => {
                if text.is_empty() {
                    return None;
                }
                let mut block = serde_json::json!({
                    "type": "text",
                    "text": text,
                });
                if let Some(cache) = cache {
                    json_object(&mut block).insert(
                        "cache_control".to_owned(),
                        encode_cache_control_marker(cache),
                    );
                }
                Some(block)
            }
            CoreContent::ToolResult {
                tool_use_id,
                content: result_content,
                is_error,
                cache,
            } => {
                let result_text = result_content
                    .iter()
                    .filter_map(|c| match c {
                        CoreContent::Text { text, .. } => Some(text.as_str()),
                        other => {
                            tracing::warn!(
                                ?other,
                                tool_use_id,
                                "Anthropic: dropping non-text content block in ToolResult encoding"
                            );
                            None
                        }
                    })
                    .collect::<String>();
                let sanitized_id = sanitize_tool_use_id(tool_use_id);
                let mut block = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": sanitized_id,
                    "content": result_text,
                });
                if *is_error {
                    json_object(&mut block).insert("is_error".to_owned(), serde_json::json!(true));
                }
                if let Some(cache) = cache {
                    json_object(&mut block).insert(
                        "cache_control".to_owned(),
                        encode_cache_control_marker(cache),
                    );
                }
                Some(block)
            }
            CoreContent::Image { source } => Some(serde_json::json!({
                "type": "image",
                "source": source,
            })),
            other => {
                tracing::warn!(
                    ?other,
                    "dropping unsupported content block during Anthropic encode"
                );
                None
            }
        })
        .collect();

    serde_json::Value::Array(blocks)
}

/// The top-level `system` blocks: the core system prompt, then the content of
/// every system-role message in conversation order. Anthropic has no system
/// role inside `messages`, so those turns are hoisted here rather than lost.
fn encode_system(system: &[CoreContent], messages: &[CoreMessage]) -> Option<serde_json::Value> {
    let hoisted = messages
        .iter()
        .filter(|msg| msg.role == CoreRole::System)
        .flat_map(|msg| &msg.content);
    let blocks: Vec<serde_json::Value> = system
        .iter()
        .chain(hoisted)
        .filter_map(|c| match c {
            CoreContent::Text { text, cache } => {
                if text.is_empty() {
                    return None;
                }
                let mut block = serde_json::json!({
                    "type": "text",
                    "text": text,
                });
                if let Some(cc) = cache {
                    json_object(&mut block)
                        .insert("cache_control".to_owned(), encode_cache_control_marker(cc));
                }
                Some(block)
            }
            other => {
                tracing::warn!(
                    ?other,
                    "dropping non-Text system content block during Anthropic encode"
                );
                None
            }
        })
        .collect();
    if blocks.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(blocks))
    }
}

fn encode_messages(core_messages: &[CoreMessage]) -> Vec<serde_json::Value> {
    let mut messages = Vec::with_capacity(core_messages.len());
    for msg in core_messages {
        match msg.role {
            CoreRole::User => {
                let content = encode_content_blocks(&msg.content);
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content,
                }));
            }
            CoreRole::Assistant => {
                let content = encode_assistant_content(&msg.content);
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": content,
                }));
            }
            // Hoisted into the top-level `system` by `encode_system`.
            CoreRole::System => {}
            CoreRole::Tool => {
                let content = encode_content_blocks(&msg.content);
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content,
                }));
            }
            _ => {
                let content = encode_content_blocks(&msg.content);
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": content,
                }));
            }
        }
    }
    messages
}

fn encode_tools(tools: &[CoreTool]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            let schema = if t.input_schema.is_object() {
                t.input_schema.clone()
            } else {
                serde_json::json!({"type": "object", "properties": {}})
            };
            let (sanitized_name, _changed) = sanitize_tool_name(&t.name);
            let mut tool = serde_json::json!({
                "name": sanitized_name,
                "input_schema": schema,
            });
            if let Some(ref desc) = t.description {
                json_object(&mut tool).insert(
                    "description".to_owned(),
                    serde_json::Value::String(desc.clone()),
                );
            }
            tool
        })
        .collect()
}

/// Anthropic takes the generic modes as `{"type": ...}` objects and a named
/// function as `{"type": "tool", "name": ...}`; a client `Raw` choice
/// forwards verbatim. A future core variant has no wired mapping yet
/// (`CoreToolChoice` is `#[non_exhaustive]`); the shared
/// `resolve_tool_choice` policy turns that missing mapping into a named
/// rejection (or a lossy drop) instead of a silent omission.
fn tool_choice_value(tc: &CoreToolChoice) -> Option<serde_json::Value> {
    match tc {
        CoreToolChoice::Auto => Some(serde_json::json!({"type": "auto"})),
        CoreToolChoice::Any => Some(serde_json::json!({"type": "any"})),
        CoreToolChoice::None => Some(serde_json::json!({"type": "none"})),
        CoreToolChoice::Tool { name } => {
            let (sanitized, _) = sanitize_tool_name(name);
            Some(serde_json::json!({
                "type": "tool",
                "name": sanitized
            }))
        }
        CoreToolChoice::Raw(v) => Some(v.clone()),
        _ => None,
    }
}

fn encode_assistant_content(content: &[CoreContent]) -> serde_json::Value {
    let blocks: Vec<serde_json::Value> = content
        .iter()
        .filter_map(|c| match c {
            CoreContent::Text { text, cache } => {
                if text.is_empty() {
                    return None;
                }
                let mut block = serde_json::json!({
                    "type": "text",
                    "text": text,
                });
                if let Some(cache) = cache {
                    json_object(&mut block).insert(
                        "cache_control".to_owned(),
                        encode_cache_control_marker(cache),
                    );
                }
                Some(block)
            }
            CoreContent::Thinking { text, signature } => {
                if text.is_empty() {
                    return None;
                }
                let mut block = serde_json::json!({
                    "type": "thinking",
                    "thinking": text,
                });
                if let Some(sig) = signature {
                    json_object(&mut block).insert(
                        "signature".to_owned(),
                        serde_json::Value::String(sig.clone()),
                    );
                }
                Some(block)
            }
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                let (sanitized_name, _) = sanitize_tool_name(name);
                let sanitized_id = sanitize_tool_use_id(id);
                Some(serde_json::json!({
                    "type": "tool_use",
                    "id": sanitized_id,
                    "name": sanitized_name,
                    "input": input,
                }))
            }
            other => {
                tracing::warn!(
                    ?other,
                    "dropping unsupported content block during Anthropic assistant encode"
                );
                None
            }
        })
        .collect();

    serde_json::Value::Array(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_wire::{ClientWireFormat, ClientWireRequest};
    use llm_proxy_protocol::core::SamplingOptions;

    /// An OpenAI-Chat-dialect client wire for the raw body: never the same
    /// format as the Anthropic adapter, so tests exercise the normalized
    /// cross-format encode (the body carries no policy-relevant extras).
    fn chat_client(core: &CoreRequest) -> ClientWireRequest {
        ClientWireRequest::new(
            ClientWireFormat::OpenAiChatCompletions,
            serde_json::json!({"model": core.model.requested, "messages": []}),
        )
    }

    fn make_target() -> ProviderAdapterTarget {
        ProviderAdapterTarget {
            provider_name: "test-anthropic".into(),
            adapter_name: "anthropic".into(),
            protocol: super::ProtocolName::new(protocols::ANTHROPIC_MESSAGES),
            endpoint: "https://api.anthropic.com/v1/messages".into(),
            auth_style: llm_proxy_core::AuthStyle::XApiKey,
            api_key: secrecy::SecretString::from("test-key"),
            requested_model: "claude-sonnet-4-20250514".into(),
            upstream_model: "claude-sonnet-4-20250514".into(),
            headers: std::sync::Arc::new(std::collections::HashMap::new()),
            lossy: false,
            forward_inbound_headers: std::sync::Arc::new(Vec::new()),
            prompt_cache: None,
        }
    }

    fn make_core_request(messages: Vec<CoreMessage>) -> CoreRequest {
        CoreRequest {
            model: ModelRef {
                requested: "claude-sonnet-4-20250514".into(),
                upstream: None,
            },
            system: vec![],
            messages,
            tools: vec![],
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: Default::default(),
            provider_hints: Default::default(),
        }
    }

    #[test]
    fn sanitize_tool_name_valid() {
        let (name, changed) = sanitize_tool_name("get_weather");
        assert_eq!(name, "get_weather");
        assert!(!changed);
        let (name, changed) = sanitize_tool_name("my-tool-123");
        assert_eq!(name, "my-tool-123");
        assert!(!changed);
    }

    #[test]
    fn sanitize_tool_name_replaces_dots_collision_safe() {
        let (name1, changed1) = sanitize_tool_name("my.tool.name");
        let (name2, changed2) = sanitize_tool_name("my_tool_name");
        assert!(changed1);
        assert!(!changed2);
        assert_ne!(
            name1, name2,
            "collision-safe: different names must produce different sanitized names"
        );
        assert!(name1.contains("0x2e"), "dot should be encoded as _0x2e_");
    }

    #[test]
    fn sanitize_tool_name_reversible() {
        let original = "my.tool+name";
        let (sanitized, _) = sanitize_tool_name(original);
        let restored = desanitize_tool_name(&sanitized);
        assert_eq!(restored, original);
    }

    #[test]
    fn sanitize_tool_name_truncates_long() {
        let long_name = "a".repeat(200);
        let (result, _) = sanitize_tool_name(&long_name);
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn sanitize_tool_name_empty_becomes_tool() {
        let (name, _) = sanitize_tool_name("");
        assert_eq!(name, "tool");
    }

    #[test]
    fn sanitize_tool_name_special_chars() {
        let (name, changed) = sanitize_tool_name("get weather!@#");
        assert!(changed);
        assert!(name.contains("0x"));
    }

    #[test]
    fn sanitize_tool_use_id_valid() {
        assert_eq!(sanitize_tool_use_id("toolu_123"), "toolu_123");
    }

    #[test]
    fn sanitize_tool_use_id_replaces_dashes() {
        assert_eq!(sanitize_tool_use_id("call-123"), "call_123");
    }

    #[test]
    fn sanitize_tool_use_id_empty_becomes_default() {
        assert_eq!(sanitize_tool_use_id(""), "tool_result");
    }

    #[test]
    fn documented_message_delta_usage_and_stop_reason_survive() {
        // Documented stream: message_start carries input usage, the final
        // message_delta carries only cumulative output_tokens plus the stop
        // reason. Both must reach core events.
        let target = make_target();
        let mut decoder = AnthropicAdapter::new().new_stream_decoder(&target);
        let frames = [
            SseFrame {
                event: None,
                id: None,
                data: r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}}"#.to_owned(),
            },
            SseFrame {
                event: None,
                id: None,
                data: r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":15}}"#.to_owned(),
            },
        ];
        let mut events = Vec::new();
        for frame in &frames {
            events.extend(decoder.decode_frame(frame).unwrap());
        }
        let usage = events
            .iter()
            .find_map(|e| match e {
                CoreEvent::UsageDelta { usage } => Some(usage.clone()),
                _ => None,
            })
            .expect("usage from message_delta");
        assert_eq!(usage.input_tokens, 25, "input comes from message_start");
        assert_eq!(usage.output_tokens, 15);
        assert_eq!(usage.cache_read_input_tokens, Some(7));
        assert_eq!(usage.cache_creation_input_tokens, Some(3));
        let stop = events
            .iter()
            .find_map(|e| match e {
                CoreEvent::MessageStop { stop_reason, .. } => Some(stop_reason.clone()),
                _ => None,
            })
            .expect("stop reason from message_delta");
        assert_eq!(stop, StopReason::MaxTokens);
    }

    #[test]
    fn anthropic_cache_creation_split_maps_to_total() {
        let target = make_target();
        let resp = AnthropicAdapter::new()
            .decode_response(
                br#"{"id":"msg_c","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2,"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":6}}}"#,
                &target,
            )
            .unwrap();
        assert_eq!(resp.usage.cache_creation_input_tokens, Some(10));
    }

    #[test]
    fn message_level_and_tool_result_cache_markers_reach_upstream() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![
                CoreContent::Text {
                    text: "KEEP ME".into(),
                    cache: None,
                },
                CoreContent::Text {
                    text: "MUST NOT LOSE ITS MARKER".into(),
                    cache: Some(llm_proxy_protocol::core::CacheControl {
                        r#type: llm_proxy_protocol::core::CacheControlType::Ephemeral,
                        ttl: Some("1h".to_owned()),
                    }),
                },
                CoreContent::ToolResult {
                    tool_use_id: "toolu_1".into(),
                    content: vec![CoreContent::Text {
                        text: "ok".into(),
                        cache: None,
                    }],
                    is_error: false,
                    cache: Some(llm_proxy_protocol::core::CacheControl {
                        r#type: llm_proxy_protocol::core::CacheControlType::Ephemeral,
                        ttl: None,
                    }),
                },
            ],
        }]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(
            content[1]["cache_control"],
            serde_json::json!({"type": "ephemeral", "ttl": "1h"}),
            "message-level text markers must survive: {content:?}"
        );
        assert_eq!(
            content[2]["cache_control"],
            serde_json::json!({"type": "ephemeral"}),
            "tool_result markers must survive: {content:?}"
        );
    }

    #[test]
    fn encode_text_request() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "Hello".into(),
                cache: None,
            }],
        }]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn encode_system_prompt() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.system = vec![CoreContent::Text {
            text: "You are helpful".into(),
            cache: None,
        }];
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(body.get("system").is_some());
    }

    #[test]
    fn system_role_messages_are_hoisted_into_the_system_prompt() {
        // Ollama `system` messages, Gemini `system` contents and Responses
        // `developer` items decode as inline system-role messages. Anthropic
        // has no system role in `messages`, so their text must reach the
        // top-level system (after the core prompt, in order), not vanish.
        let text = |text: &str| CoreContent::Text {
            text: text.into(),
            cache: None,
        };
        let mut core = make_core_request(vec![
            CoreMessage {
                role: CoreRole::System,
                content: vec![text("be terse")],
            },
            CoreMessage {
                role: CoreRole::User,
                content: vec![text("hi")],
            },
            CoreMessage {
                role: CoreRole::Assistant,
                content: vec![text("hello")],
            },
            CoreMessage {
                role: CoreRole::System,
                content: vec![text("answer in French")],
            },
            CoreMessage {
                role: CoreRole::User,
                content: vec![text("and now?")],
            },
        ]);
        core.system = vec![text("You are helpful")];
        let proxy_req = AnthropicAdapter::new()
            .encode_request(&core, &make_target(), &chat_client(&core))
            .expect("encode");

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).expect("json body");
        let system: Vec<&str> = body["system"]
            .as_array()
            .expect("system blocks")
            .iter()
            .filter_map(|block| block["text"].as_str())
            .collect();
        assert_eq!(system, ["You are helpful", "be terse", "answer in French"]);
        let roles: Vec<&str> = body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .filter_map(|message| message["role"].as_str())
            .collect();
        assert_eq!(roles, ["user", "assistant", "user"]);
    }

    #[test]
    fn encode_tool_declarations_sanitized() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tools = vec![CoreTool {
            name: "get.weather".into(),
            description: Some("Get weather".into()),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert!(tools[0]["name"].as_str().unwrap().contains("0x2e"));
    }

    #[test]
    fn encode_tool_choice() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tool_choice = Some(CoreToolChoice::Auto);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["tool_choice"], serde_json::json!({"type": "auto"}));
    }

    #[test]
    fn encode_stream_flag() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.stream = true;
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["stream"], true);
        assert!(proxy_req.stream);
    }

    #[test]
    fn encode_metadata_user_id() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.metadata.user_id = Some("user-42".into());
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(body.get("metadata").is_some());
        assert_eq!(body["metadata"]["user_id"], "user-42");
    }

    #[test]
    fn encode_thinking_passthrough() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.max_tokens = Some(8_192);
        core.sampling.thinking =
            Some(serde_json::json!({"type": "enabled", "budget_tokens": 5000}));
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(body.get("thinking").is_some());
        assert_eq!(body["thinking"]["budget_tokens"], 5000);
    }

    #[test]
    fn decode_text_response() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello world"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.id, Some("msg_test".to_owned()));
        assert_eq!(core_resp.model.requested, "claude-sonnet-4-20250514");
        assert_eq!(core_resp.stop_reason, StopReason::EndTurn);
        assert_eq!(core_resp.usage.input_tokens, 100);
        assert_eq!(core_resp.usage.output_tokens, 50);
    }

    #[test]
    fn decode_tool_use_response() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.stop_reason, StopReason::ToolUse);
        match &core_resp.content[0] {
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn decode_thinking_response() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm..."},
                {"type": "text", "text": "answer"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.content.len(), 2);
        match &core_resp.content[0] {
            CoreContent::Thinking { text, signature } => {
                assert_eq!(text, "hmm...");
                assert!(signature.is_none());
            }
            _ => panic!("expected Thinking"),
        }
    }

    #[test]
    fn decode_preserves_requested_model() {
        let mut target = make_target();
        target.upstream_model = "claude-sonnet-4-20250514".into();
        target.requested_model = "my-claude".into();

        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.model.requested, "my-claude");
        assert_eq!(
            core_resp.model.upstream.as_deref(),
            Some("claude-sonnet-4-20250514")
        );
    }

    #[test]
    fn stream_text_decoding() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#,
            ),
            make_frame(r#"{"type":"content_block_stop","index":0}"#),
            make_frame(
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":10}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. }))
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::ContentStart { .. }))
        );
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::TextDelta { text, .. } if text == "Hello"
        )));
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::TextDelta { text, .. } if text == " world"
        )));
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. }))
        );
    }

    #[test]
    fn stream_tool_call_decoding() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}"#,
            ),
            make_frame(r#"{"type":"content_block_stop","index":0}"#),
            make_frame(
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":20}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::ToolCallStart { name, .. } if name == "get_weather"
        )));
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallDelta { .. }))
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallStop { .. }))
        );
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
                ..
            }
        )));
    }

    #[test]
    fn stream_ping_event() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(r#"{"type":"ping"}"#);
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.iter().any(|e| matches!(e, CoreEvent::Ping)));
    }

    #[test]
    fn stream_error_event() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Too many requests"}}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.iter().any(|e| matches!(e, CoreEvent::Error { .. })));
    }

    #[test]
    fn stream_finish_emits_lifecycle() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let events = decoder.finish().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. }))
        );
    }

    #[test]
    fn stream_thinking_decoding() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}"#,
            ),
            make_frame(r#"{"type":"content_block_stop","index":0}"#),
            make_frame(
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}"#,
            ),
            make_frame(r#"{"type":"content_block_stop","index":1}"#),
            make_frame(
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":20}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::ThinkingDelta { text, .. } if text == "Let me think..."
        )));
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::TextDelta { text, .. } if text == "answer"
        )));
    }

    #[test]
    fn anthropic_stop_reason_mappings() {
        assert_eq!(map_anthropic_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(
            map_anthropic_stop_reason("max_tokens"),
            StopReason::MaxTokens
        );
        assert_eq!(map_anthropic_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(
            map_anthropic_stop_reason("stop_sequence"),
            StopReason::StopSequence
        );
        assert_eq!(map_anthropic_stop_reason("refusal"), StopReason::Refusal);
        assert_eq!(
            map_anthropic_stop_reason("unknown"),
            StopReason::Unknown("unknown".to_owned())
        );
    }

    #[test]
    fn encode_stop_sequences() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.stop = Some(vec!["END".into(), "STOP".into()]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["stop_sequences"], serde_json::json!(["END", "STOP"]));
    }

    #[test]
    fn encode_input_schema_null_coerced_to_object() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tools = vec![CoreTool {
            name: "my_tool".into(),
            description: Some("test".into()),
            input_schema: serde_json::Value::Null,
        }];
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let schema = &body["tools"].as_array().unwrap()[0]["input_schema"];
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn encode_input_schema_string_coerced_to_object() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tools = vec![CoreTool {
            name: "my_tool".into(),
            description: Some("test".into()),
            input_schema: serde_json::json!("not an object"),
        }];
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let schema = &body["tools"].as_array().unwrap()[0]["input_schema"];
        assert_eq!(schema["type"], "object");
    }

    #[test]
    fn decode_redacted_thinking_response() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "opaque"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        match &core_resp.content[0] {
            CoreContent::RedactedThinking { .. } => {}
            other => panic!("expected RedactedThinking, got {:?}", other),
        }
    }

    #[test]
    fn decode_stop_sequence_propagated() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "stop_sequence",
            "stop_sequence": "END",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.stop_reason, StopReason::StopSequence);
        assert_eq!(core_resp.stop_sequence, Some("END".to_owned()));
    }

    #[test]
    fn stream_malformed_frame_skipped() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("{not valid json}");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_usage_only_chunk() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":20}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::UsageDelta { .. }))
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. }))
        );
    }

    #[test]
    fn stream_full_lifecycle_ordering() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            ),
            make_frame(r#"{"type":"content_block_stop","index":0}"#),
            make_frame(
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":5}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        let msg_start_idx = all_events
            .iter()
            .position(|e| matches!(e, CoreEvent::MessageStart { .. }))
            .unwrap();
        let content_start_idx = all_events
            .iter()
            .position(|e| matches!(e, CoreEvent::ContentStart { .. }))
            .unwrap();
        let text_delta_idx = all_events
            .iter()
            .position(|e| matches!(e, CoreEvent::TextDelta { .. }))
            .unwrap();
        let msg_stop_idx = all_events
            .iter()
            .position(|e| matches!(e, CoreEvent::MessageStop { .. }))
            .unwrap();

        assert!(
            msg_start_idx < content_start_idx,
            "MessageStart must precede ContentStart"
        );
        assert!(
            content_start_idx < text_delta_idx,
            "ContentStart must precede TextDelta"
        );
        assert!(
            text_delta_idx < msg_stop_idx,
            "TextDelta must precede MessageStop"
        );
    }

    #[test]
    fn encode_tool_result_as_tool_result_block() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Tool,
            content: vec![CoreContent::ToolResult {
                tool_use_id: "toolu_123".into(),
                content: vec![CoreContent::Text {
                    text: "72F sunny".into(),
                    cache: None,
                }],
                is_error: false,
                cache: None,
            }],
        }]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_123");
        assert_eq!(content[0]["content"], "72F sunny");
    }

    #[test]
    fn encode_tool_result_error_flag() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Tool,
            content: vec![CoreContent::ToolResult {
                tool_use_id: "toolu_err".into(),
                content: vec![CoreContent::Text {
                    text: "error occurred".into(),
                    cache: None,
                }],
                is_error: true,
                cache: None,
            }],
        }]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let content = &body["messages"].as_array().unwrap()[0]["content"]
            .as_array()
            .unwrap()[0];
        assert_eq!(content["is_error"], true);
    }

    #[test]
    fn encode_cache_control_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.system = vec![CoreContent::Text {
            text: "You are helpful".into(),
            cache: Some(llm_proxy_protocol::core::CacheControl {
                r#type: "ephemeral".to_owned().into(),
                ttl: Some("1h".to_owned()),
            }),
        }];
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let system = body["system"].as_array().unwrap();
        assert_eq!(system[0]["text"], "You are helpful");
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(system[0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn encode_image_content_forwarded() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![
                CoreContent::Text {
                    text: "What is this?".into(),
                    cache: None,
                },
                CoreContent::Image {
                    source: serde_json::json!({
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBOR..."
                    }),
                },
            ],
        }]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let content = body["messages"].as_array().unwrap()[0]["content"]
            .as_array()
            .unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[1]["type"], "image");
    }

    #[test]
    fn sanitize_tool_use_id_long_truncated() {
        let long_id = "a".repeat(300);
        let sanitized = sanitize_tool_use_id(&long_id);
        assert!(sanitized.len() <= 256);
    }

    #[test]
    fn desanitize_no_pattern_returns_original() {
        let name = "get_weather";
        assert_eq!(desanitize_tool_name(name), name);
    }

    #[test]
    fn desanitize_roundtrip_unicode() {
        let original = "tool.中文";
        let (sanitized, _) = sanitize_tool_name(original);
        let restored = desanitize_tool_name(&sanitized);
        assert_eq!(restored, original);
    }

    #[test]
    fn encode_reasoning_effort_translated_to_thinking_not_forwarded() {
        // BUG-20: `reasoning_effort` is an OpenAI field; the Anthropic edge
        // translates it into a clamped `thinking.budget_tokens` instead of
        // forwarding it.
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.reasoning_effort = Some("high".into());
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(
            body.get("reasoning_effort").is_none(),
            "reasoning_effort must not reach the Anthropic wire: {body}"
        );
        assert_eq!(body["thinking"]["type"], "enabled");
        // The client omitted max tokens, so the body emits the 4096 default;
        // the budget must be clamped below that effective ceiling (Anthropic
        // requires budget_tokens < max_tokens).
        assert_eq!(body["max_tokens"], 4_096);
        assert_eq!(
            body["thinking"]["budget_tokens"], 4_095,
            "budget must stay under the emitted max_tokens"
        );
    }

    #[test]
    fn cross_format_thinking_that_cannot_fit_under_max_tokens_errors() {
        // A Chat client asking for reasoning with a `max_tokens` below the
        // model's minimum thinking budget gets a request error naming the
        // limit instead of a silently non-thinking answer (02-4).
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.reasoning_effort = Some("low".into());
        core.sampling.max_tokens = Some(512);
        let client = ClientWireRequest::new(
            ClientWireFormat::OpenAiChatCompletions,
            serde_json::json!({
                "model": core.model.requested,
                "max_tokens": 512,
                "messages": [],
                "reasoning_effort": "low",
            }),
        );
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let err = adapter
            .encode_request(&core, &target, &client)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot fit under max_tokens"), "{err}");
        assert!(err.contains("1024"), "{err}");
    }

    #[test]
    fn same_format_passthrough_translates_a_foreign_reasoning_effort() {
        // An Anthropic-dialect body carrying the OpenAI `reasoning_effort`
        // field: passthrough still strips it and translates it into a native
        // `thinking` object instead of dropping or forwarding it.
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let client = ClientWireRequest::new(
            ClientWireFormat::AnthropicMessages,
            serde_json::json!({
                "model": core.model.requested,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "low",
                "future_field": "kept verbatim",
            }),
        );
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter.encode_request(&core, &target, &client).unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(
            body.get("reasoning_effort").is_none(),
            "the foreign field must not reach Anthropic: {body}"
        );
        assert_eq!(body["thinking"]["budget_tokens"], 2_048);
        // Unknown fields survive same-format passthrough untouched.
        assert_eq!(body["future_field"], "kept verbatim");
    }

    #[test]
    fn encode_tool_choice_tool_sanitized() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tool_choice = Some(CoreToolChoice::Tool {
            name: "get.weather".into(),
        });
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let tc_name = body["tool_choice"]["name"].as_str().unwrap();
        assert!(
            tc_name.contains("0x2e"),
            "tool name in tool_choice must be sanitized"
        );
    }

    #[test]
    fn decode_empty_content_gets_default_text() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.content.len(), 1);
        assert_eq!(
            core_resp.content[0],
            CoreContent::Text {
                text: String::new(),
                cache: None
            }
        );
    }

    #[test]
    fn stream_unknown_event_skipped() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(r#"{"type":"some_new_event","data":"whatever"}"#);
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_redacted_thinking_decoded() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "opaque_blob"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": null,
                "cache_read_input_tokens": null,
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = AnthropicAdapter::new();
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        match &core_resp.content[0] {
            CoreContent::RedactedThinking { .. } => {}
            other => panic!("expected RedactedThinking, got {:?}", other),
        }
    }

    #[test]
    fn desanitize_sentinel_prevents_false_positive() {
        let natural_name = "parse_0xff_value";
        assert_eq!(desanitize_tool_name(natural_name), natural_name);
    }

    #[test]
    fn sanitize_desanitize_roundtrip_with_sentinel() {
        let original = "my.tool+name";
        let (sanitized, changed) = sanitize_tool_name(original);
        assert!(changed, "should have changed");
        assert!(
            sanitized.starts_with("__llmp_"),
            "sanitized name should have sentinel"
        );
        let restored = desanitize_tool_name(&sanitized);
        assert_eq!(restored, original);
    }

    #[test]
    fn sanitize_no_change_no_sentinel() {
        let original = "get_weather";
        let (sanitized, changed) = sanitize_tool_name(original);
        assert!(!changed, "should not have changed");
        assert!(
            !sanitized.starts_with("__llmp_"),
            "unchanged name should not have sentinel"
        );
        assert_eq!(sanitized, original);
    }

    #[test]
    fn encode_uses_upstream_model() {
        let mut target = make_target();
        target.upstream_model = "claude-sonnet-4-20250514-alias".into();
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let adapter = AnthropicAdapter::new();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-20250514-alias");
    }

    #[test]
    fn decode_malformed_json_returns_error() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let result = adapter.decode_response(b"not valid json {{{", &target);
        assert!(result.is_err(), "malformed JSON should produce an error");
    }

    #[test]
    fn decode_empty_bytes_returns_error() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let result = adapter.decode_response(b"", &target);
        assert!(result.is_err(), "empty bytes should produce an error");
    }

    #[test]
    fn encode_empty_messages() {
        let core = make_core_request(vec![]);
        let adapter = AnthropicAdapter::new();
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &chat_client(&core))
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn stream_error_maps_rate_limit() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Too many requests"}}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();
        let error_event = events.iter().find(|e| matches!(e, CoreEvent::Error { .. }));
        assert!(error_event.is_some());
        if let CoreEvent::Error { error } = error_event.unwrap() {
            assert_eq!(
                error.kind,
                llm_proxy_protocol::core::CoreStreamErrorKind::RateLimit
            );
        }
    }

    #[test]
    fn stream_error_maps_authentication() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();
        let error_event = events.iter().find(|e| matches!(e, CoreEvent::Error { .. }));
        assert!(error_event.is_some());
        if let CoreEvent::Error { error } = error_event.unwrap() {
            assert_eq!(
                error.kind,
                llm_proxy_protocol::core::CoreStreamErrorKind::Authentication
            );
        }
    }

    #[test]
    fn stream_finish_reason_only_chunk() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(r#"{"type":"message_stop"}"#),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. }))
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. })),
            "message_stop event must not be silently dropped"
        );
    }

    #[test]
    fn stream_no_duplicate_tool_call_stop() {
        let target = make_target();
        let adapter = AnthropicAdapter::new();
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            ),
            make_frame(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#,
            ),
            make_frame(r#"{"type":"content_block_stop","index":0}"#),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }
        all_events.extend(decoder.finish().unwrap());

        let tool_call_stop_count = all_events
            .iter()
            .filter(|e| matches!(e, CoreEvent::ToolCallStop { .. }))
            .count();
        assert_eq!(
            tool_call_stop_count, 1,
            "ToolCallStop should be emitted exactly once"
        );
    }

    #[test]
    fn adapter_source_no_forbidden_imports() {
        let source = include_str!("anthropic.rs");
        let prod = source
            .split_once("#[cfg(test)]")
            .map(|(p, _)| p)
            .unwrap_or(source);
        assert!(
            !prod.contains("llm_proxy_protocol::client"),
            "adapter must not import client protocol types"
        );
        assert!(
            !prod.contains("llm_proxy_server"),
            "adapter must not import server crate"
        );
    }

    fn make_frame(data: &str) -> SseFrame {
        SseFrame {
            event: None,
            id: None,
            data: data.to_owned(),
        }
    }
}
