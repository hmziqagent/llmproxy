//! OpenAI Chat Completions provider adapter: translates core requests,
//! responses, and event streams to and from the Chat Completions wire format.

use std::collections::HashMap;

use llm_proxy_protocol::core::{
    ContentKind, CoreContent, CoreEvent, CoreRequest, CoreResponse, CoreRole, CoreToolChoice,
    ModelRef, StopReason, Usage,
};
use llm_proxy_protocol::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatMessage, FunctionCall, FunctionDef,
    StreamOptions, ToolCall, ToolDef,
};

use super::{
    ProviderAdapterTarget, ProviderStreamDecoder, build_proxy_request, build_usage_from_openai,
    expand_url_template, map_openai_finish_reason, mark_tool_error, response_model_ref,
};
use crate::ProtocolName;
use crate::error::ProviderError;
use crate::protocols;
use crate::sse::SseFrame;

/// Adapter for the OpenAI Chat Completions API.
#[derive(Debug, Clone, Default)]
pub struct OpenAiChatAdapter;

/// Stateful stream decoder translating OpenAI Chat Completions SSE frames into `CoreEvent`s.
#[derive(Debug)]
pub struct OpenAiChatStreamDecoder {
    model_ref: ModelRef,
    started: bool,
    content_index: usize,
    content_started: bool,
    reasoning_started: bool,
    tool_blocks: HashMap<usize, usize>,
    stop_sent: bool,
    saw_terminal: bool,
}

/// Maps an OpenAI-compatible in-band `{"error": ...}` SSE frame to a core
/// error event. Returns `None` when the payload is not an error object.
fn decode_stream_error_frame(data: &str) -> Option<CoreEvent> {
    let err: llm_proxy_protocol::openai::ErrorResponse = serde_json::from_str(data).ok()?;
    let kind = match err.error.code.as_deref() {
        Some("invalid_api_key") | Some("invalid_request_error") | Some("400") => {
            llm_proxy_protocol::core::CoreStreamErrorKind::InvalidRequest
        }
        Some("rate_limit_exceeded") | Some("429") => {
            llm_proxy_protocol::core::CoreStreamErrorKind::RateLimit
        }
        Some("insufficient_quota") | Some("403") => {
            llm_proxy_protocol::core::CoreStreamErrorKind::Permission
        }
        _ => llm_proxy_protocol::core::CoreStreamErrorKind::Upstream,
    };
    Some(CoreEvent::Error {
        error: llm_proxy_protocol::core::CoreStreamError::new(kind, err.error.message),
    })
}

impl ProviderStreamDecoder for OpenAiChatStreamDecoder {
    fn decode_frame(&mut self, frame: &SseFrame) -> Result<Vec<CoreEvent>, ProviderError> {
        let data = frame.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(vec![]);
        }

        let chunk: ChatCompletionChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(_) => {
                if let Some(error_event) = decode_stream_error_frame(data) {
                    return Ok(vec![error_event]);
                }
                let truncated = super::truncate_str_safe(data, 200);
                tracing::warn!(data = truncated, "malformed OpenAI chunk, skipping");
                return Ok(vec![]);
            }
        };

        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(CoreEvent::MessageStart {
                id: Some(chunk.id.clone()),
                model: self.model_ref.clone(),
            });
        }

        // `first` keeps the empty-choices handling and the head access in one
        // checked form.
        let Some(choice) = chunk.choices.first() else {
            if let Some(ref usage) = chunk.usage {
                events.push(CoreEvent::UsageDelta {
                    usage: build_usage_from_openai(usage),
                });
            }
            return Ok(events);
        };

        if let Some(reasoning) = choice
            .delta
            .as_ref()
            .and_then(|d| d.reasoning_content.as_ref())
        {
            if !reasoning.is_empty() {
                self.close_text_if_open();
                if !self.reasoning_started {
                    self.reasoning_started = true;
                    events.push(CoreEvent::ContentStart {
                        index: self.content_index,
                        kind: ContentKind::Thinking,
                    });
                }
                events.push(CoreEvent::ThinkingDelta {
                    index: self.content_index,
                    text: reasoning.clone(),
                });
            }
        }

        if let Some(ref delta) = choice.delta {
            if !delta.content_is_empty() {
                self.close_reasoning_if_open();
                if !self.content_started {
                    self.content_started = true;
                    events.push(CoreEvent::ContentStart {
                        index: self.content_index,
                        kind: ContentKind::Text,
                    });
                }
                events.push(CoreEvent::TextDelta {
                    index: self.content_index,
                    text: delta.content_text(),
                });
            }
        }

        if let Some(ref delta) = choice.delta
            && !delta.tool_calls.is_empty()
        {
            self.close_content_if_open();
            self.push_tool_call_deltas(&delta.tool_calls, &mut events);
        }

        if let Some(ref reason) = choice.finish_reason {
            if !reason.is_empty() && !self.stop_sent {
                self.close_content_if_open();
                self.close_tool_blocks(&mut events);

                let stop_reason = map_openai_finish_reason(reason);

                if let Some(ref usage) = chunk.usage {
                    events.push(CoreEvent::UsageDelta {
                        usage: build_usage_from_openai(usage),
                    });
                }

                events.push(CoreEvent::MessageStop {
                    stop_reason,
                    stop_sequence: None,
                });
                self.stop_sent = true;
                self.saw_terminal = true;
            }
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

        self.close_content_if_open();
        self.close_tool_blocks(&mut events);

        if !self.stop_sent {
            let stop_reason = if !self.tool_blocks.is_empty() {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            };
            events.push(CoreEvent::MessageStop {
                stop_reason,
                stop_sequence: None,
            });
            self.stop_sent = true;
        }

        Ok(events)
    }

    fn saw_terminal(&self) -> bool {
        self.saw_terminal
    }
}

impl OpenAiChatStreamDecoder {
    fn close_text_if_open(&mut self) {
        if self.content_started {
            self.content_started = false;
            self.content_index += 1;
        }
    }

    fn close_reasoning_if_open(&mut self) {
        if self.reasoning_started {
            self.reasoning_started = false;
            self.content_index += 1;
        }
    }

    fn close_content_if_open(&mut self) {
        if self.content_started {
            self.content_started = false;
            self.content_index += 1;
        }
        if self.reasoning_started {
            self.reasoning_started = false;
            self.content_index += 1;
        }
    }

    fn push_tool_call_deltas(&mut self, tool_calls: &[ToolCall], events: &mut Vec<CoreEvent>) {
        for tc in tool_calls {
            let oi = tc
                .index
                .and_then(|i| usize::try_from(i).ok())
                .unwrap_or_else(|| {
                    if tc.index.is_some_and(|i| i < 0) {
                        tracing::warn!(
                            raw_index = tc.index,
                            "OpenAI: negative tool_call index, clamping to 0"
                        );
                    }
                    0
                });

            if !self.tool_blocks.contains_key(&oi) {
                let func_name = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.as_deref())
                    .unwrap_or("");
                if func_name.is_empty() {
                    tracing::warn!(
                        index = oi,
                        "OpenAI: tool_call without function name, skipping tool call"
                    );
                    continue;
                }

                let tool_id = tc
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4()));

                let block_idx = self.content_index;
                self.tool_blocks.insert(oi, block_idx);

                events.push(CoreEvent::ToolCallStart {
                    index: block_idx,
                    id: tool_id,
                    name: func_name.to_owned(),
                });

                self.content_index += 1;
            }

            if let Some(ref func) = tc.function
                && let Some(ref args) = func.arguments
                && !args.is_empty()
                && let Some(&block_idx) = self.tool_blocks.get(&oi)
            {
                events.push(CoreEvent::ToolCallDelta {
                    index: block_idx,
                    args_delta: args.clone(),
                });
            }
        }
    }

    fn close_tool_blocks(&self, events: &mut Vec<CoreEvent>) {
        let mut indices: Vec<_> = self.tool_blocks.values().copied().collect();
        indices.sort();
        for idx in indices {
            events.push(CoreEvent::ToolCallStop { index: idx });
        }
    }
}

/// Encodes user content: plain text when the message has no media, or an
/// array of `text` / `image_url` parts when it does. The neutral core image
/// source (`{"type":"url",...}` / `{"type":"base64",...}`) maps back onto the
/// OpenAI `image_url` shape (base64 sources become `data:` URIs).
fn encode_user_content(content: &[CoreContent]) -> serde_json::Value {
    let has_media = content
        .iter()
        .any(|c| matches!(c, CoreContent::Image { .. }));
    if !has_media {
        let text = collect_text(content);
        return serde_json::Value::String(text);
    }
    let mut parts = Vec::new();
    for c in content {
        match c {
            CoreContent::Text { text, .. } => {
                if !text.is_empty() {
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
            }
            CoreContent::Image { source } => {
                if let Some(part) = neutral_image_to_openai_part(source) {
                    parts.push(part);
                }
            }
            other => {
                tracing::warn!(
                    ?other,
                    "OpenAI Chat: dropping unsupported user content block during encode"
                );
            }
        }
    }
    serde_json::Value::Array(parts)
}

/// Maps a neutral core image source onto an OpenAI `image_url` content part.
fn neutral_image_to_openai_part(source: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = source.as_object()?;
    match obj.get("type").and_then(serde_json::Value::as_str) {
        Some("url") => {
            let url = obj.get("url").and_then(serde_json::Value::as_str)?;
            Some(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": url}
            }))
        }
        Some("base64") => {
            let media_type = obj.get("media_type").and_then(serde_json::Value::as_str)?;
            let data = obj.get("data").and_then(serde_json::Value::as_str)?;
            Some(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": format!("data:{media_type};base64,{data}")}
            }))
        }
        other => {
            tracing::warn!(
                source_type = ?other,
                "OpenAI Chat: unsupported neutral image source type; dropping"
            );
            None
        }
    }
}

fn encode_assistant_content(
    content: &CoreContent,
    tool_calls: &mut Vec<ToolCall>,
    reasoning_content: &mut Option<String>,
) {
    match content {
        // An empty Thinking block means the client explicitly replayed an
        // empty `reasoning_content`; providers that require the field on
        // replay (DeepSeek) must see it, so it is never omitted.
        CoreContent::Thinking { text, .. } => match reasoning_content {
            Some(existing) => existing.push_str(text),
            None => *reasoning_content = Some(text.clone()),
        },
        CoreContent::ToolUse {
            id, name, input, ..
        } => {
            let arguments = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_owned());
            tool_calls.push(ToolCall {
                index: None,
                id: (!id.is_empty()).then(|| id.clone()),
                r#type: Some("function".to_owned()),
                function: Some(FunctionCall {
                    name: (!name.is_empty()).then(|| name.clone()),
                    arguments: Some(arguments),
                }),
            });
        }
        other => {
            tracing::warn!(
                ?other,
                "OpenAI Chat: dropping unsupported content block in assistant message encoding"
            );
        }
    }
}

fn push_tool_result_messages(content: &[CoreContent], messages: &mut Vec<ChatMessage>) {
    for c in content {
        let CoreContent::ToolResult {
            tool_use_id,
            content: result_content,
            is_error,
            ..
        } = c
        else {
            continue;
        };
        messages.push(ChatMessage {
            role: "tool".to_owned(),
            content: serde_json::Value::String(mark_tool_error(
                collect_text(result_content),
                *is_error,
            )),
            reasoning_content: None,
            tool_calls: Vec::new(),
            name: None,
            tool_call_id: Some(tool_use_id.clone()),
            cache_control: None,
            refusal: None,
            // Not a Chat Completions field: a failure rides the content as
            // the tool-error marker instead.
            is_error: None,
        });
    }
}

impl OpenAiChatAdapter {
    /// Encode a core request into an OpenAI Chat Completions request.
    ///
    /// Same-format (Chat Completions client) requests forward the raw client
    /// body with only surgical edits: the model name, usage-on-streams, and
    /// reasoning translation. Cross-format requests encode the normalized
    /// core request under the cross-format field policy.
    pub fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &super::client_wire::ClientWireRequest,
    ) -> Result<super::ProxyRequest, ProviderError> {
        let url = expand_url_template(&target.endpoint, target)?;
        let req = if client.same_format_as(&ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS))
            && let Some(raw) = client.body_object()
        {
            edit_raw_chat_request(raw, core, target)?
        } else {
            encode_normalized_request(core, target, client)?
        };
        // Phase 3 (plan 06 step 7): operator hygiene at the provider edge.
        // The Chat Completions wire format has no cache markers, so only
        // volatile-byte stripping applies here.
        let mut req = req;
        if let Some(hygiene) = &target.prompt_cache {
            super::mutation::apply_prompt_hygiene(&mut req, hygiene, false);
        }
        let body = serde_json::to_vec(&req)?;
        Ok(build_proxy_request(body, target, core.stream, url, client))
    }
}

/// Same-format passthrough: apply only the edits the proxy needs to the raw
/// OpenAI Chat client body; every other field survives byte-for-byte.
fn edit_raw_chat_request(
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

    // Edit 2: an Anthropic-style `thinking` object is not a Chat field; the
    // reasoning module derives `reasoning_effort` from it instead.
    obj.remove("thinking");
    if let Some(effort) = super::reasoning::chat_effort(&core.sampling)? {
        obj.insert(
            "reasoning_effort".to_owned(),
            serde_json::Value::String(effort),
        );
    } else {
        obj.remove("reasoning_effort");
    }

    // Edit 3: the proxy needs real usage on streams for the ledger, whatever
    // the client asked to see (the client encoder decides what is exposed).
    if core.stream {
        let include_usage = serde_json::json!({"include_usage": true});
        match obj
            .get_mut("stream_options")
            .and_then(serde_json::Value::as_object_mut)
        {
            Some(options) => {
                options.insert(
                    "include_usage".to_owned(),
                    include_usage["include_usage"].clone(),
                );
            }
            None => {
                obj.insert("stream_options".to_owned(), include_usage);
            }
        }
    }

    Ok(serde_json::Value::Object(obj))
}

/// The Chat Completions API documents `tool_choice` as a string for the
/// generic modes and an object only for naming a function; a client `Raw`
/// choice forwards verbatim. A future core variant has no wired mapping yet
/// (`CoreToolChoice` is `#[non_exhaustive]`); the shared
/// `resolve_tool_choice` policy turns that missing mapping into a named
/// rejection (or a lossy drop) instead of a silent omission.
fn tool_choice_value(tc: &CoreToolChoice) -> Option<serde_json::Value> {
    match tc {
        CoreToolChoice::Auto => Some(serde_json::json!("auto")),
        CoreToolChoice::Any => Some(serde_json::json!("required")),
        CoreToolChoice::None => Some(serde_json::json!("none")),
        CoreToolChoice::Tool { name } => Some(serde_json::json!({
            "type": "function",
            "function": {"name": name}
        })),
        CoreToolChoice::Raw(v) => Some(v.clone()),
        _ => None,
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "field-by-field wire encode of the OpenAI Chat Completions request \
             shape, kept in one place so the mapping can be audited against \
             the API schema without jumping between helpers"
)]
/// Cross-format encode: build the Chat Completions request from the
/// normalized core request, settle the client's leftover fields through the
/// field policy, and translate reasoning parameters.
fn encode_normalized_request(
    core: &CoreRequest,
    target: &ProviderAdapterTarget,
    client: &super::client_wire::ClientWireRequest,
) -> Result<serde_json::Value, ProviderError> {
    let carried = super::client_wire::apply_field_policy(
        client,
        &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
        target.lossy,
    )?;
    super::client_wire::check_cross_format_content(
        core,
        &ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
        target.lossy,
    )?;
    let mut messages = Vec::new();

    for sys_content in &core.system {
        match sys_content {
            CoreContent::Text { text, cache } => {
                if cache.is_some() {
                    tracing::warn!(
                        "OpenAI Chat: cache_control is not supported, dropping cache marker on system message"
                    );
                }
                if !text.is_empty() {
                    messages.push(ChatMessage {
                        role: "system".to_owned(),
                        content: serde_json::Value::String(text.clone()),
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    });
                }
            }
            other => {
                tracing::warn!(
                    ?other,
                    "OpenAI Chat: dropping non-Text system content block during encode"
                );
            }
        }
    }

    for msg in &core.messages {
        match msg.role {
            CoreRole::User => {
                let content = encode_user_content(&msg.content);
                let is_empty = match &content {
                    serde_json::Value::String(text) => text.is_empty(),
                    serde_json::Value::Array(parts) => parts.is_empty(),
                    _ => false,
                };
                if !is_empty {
                    messages.push(ChatMessage {
                        role: "user".to_owned(),
                        content,
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    });
                }

                push_tool_result_messages(&msg.content, &mut messages);
            }
            CoreRole::Assistant => {
                let text = collect_text(&msg.content);
                let mut tool_calls = Vec::new();
                let mut reasoning_content: Option<String> = None;

                for content in &msg.content {
                    encode_assistant_content(content, &mut tool_calls, &mut reasoning_content);
                }

                let content = if text.is_empty() && !tool_calls.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(text)
                };

                messages.push(ChatMessage {
                    role: "assistant".to_owned(),
                    content,
                    reasoning_content,
                    tool_calls,
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                });
            }
            CoreRole::System => {
                let text = collect_text(&msg.content);
                if !text.is_empty() {
                    messages.push(ChatMessage {
                        role: "system".to_owned(),
                        content: serde_json::Value::String(text),
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    });
                }
            }
            CoreRole::Tool => {
                let text = collect_text(&msg.content);
                let tool_use_id = msg
                    .content
                    .iter()
                    .find_map(|c| match c {
                        CoreContent::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                        _ => None,
                    })
                    .unwrap_or_default();
                if tool_use_id.is_empty() {
                    tracing::warn!(
                        "OpenAI Chat: Tool message has empty tool_use_id; downstream protocols may reject this"
                    );
                }
                messages.push(ChatMessage {
                    role: "tool".to_owned(),
                    content: serde_json::Value::String(text),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    name: None,
                    tool_call_id: Some(tool_use_id),
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                });
            }
            _ => {
                let text = collect_text(&msg.content);
                if !text.is_empty() {
                    messages.push(ChatMessage {
                        role: "user".to_owned(),
                        content: serde_json::Value::String(text),
                        reasoning_content: None,
                        tool_calls: Vec::new(),
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    });
                }
            }
        }
    }

    let tools: Vec<ToolDef> = core
        .tools
        .iter()
        .map(|t| {
            let schema = if t.input_schema.is_null() {
                serde_json::json!({"type": "object", "properties": {}})
            } else {
                t.input_schema.clone()
            };
            ToolDef {
                r#type: "function".to_owned(),
                function: FunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: Some(schema),
                },
            }
        })
        .collect();

    let tool_choice = match core.tool_choice.as_ref() {
        Some(tc) => super::client_wire::resolve_tool_choice(
            tc,
            target.lossy,
            protocols::OPENAI_CHAT_COMPLETIONS,
            tool_choice_value,
        )?,
        None => None,
    };

    let mut req = ChatCompletionRequest {
        model: target.upstream_model.clone(),
        messages,
        stream: if core.stream { Some(true) } else { None },
        temperature: core.sampling.temperature,
        top_p: core.sampling.top_p,
        max_tokens: core.sampling.max_tokens,
        reasoning_effort: core.sampling.reasoning_effort.clone(),
        thinking: core.sampling.thinking.clone(),
        tools,
        tool_choice,
        stop: core.sampling.stop.as_ref().map(|v| {
            // A lone stop sequence is sent as a bare string per the API docs.
            if let [only] = v.as_slice() {
                serde_json::Value::String(only.clone())
            } else {
                serde_json::Value::Array(
                    v.iter()
                        .map(|s| serde_json::Value::String(s.clone()))
                        .collect(),
                )
            }
        }),
        stream_options: None,
        user: core.metadata.user_id.clone(),
        extra: serde_json::Map::new(),
    };

    if core.stream {
        // Always ask for usage on streams so the ledger records real
        // numbers; whether the client sees a usage chunk is decided by
        // the client encoder, not by this request.
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
    }

    // Reasoning translation: `thinking` never crosses to the Chat wire;
    // `reasoning_effort` is validated (or derived from a budget). An
    // Anthropic `output_config.effort` maps onto the same field.
    req.thinking = None;
    let mut effort = super::reasoning::chat_effort(&core.sampling)?;
    if effort.is_none()
        && let Some(config_effort) = client
            .body
            .as_object()
            .and_then(|obj| obj.get("output_config"))
            .and_then(serde_json::Value::as_object)
            .and_then(|config| config.get("effort"))
            .and_then(serde_json::Value::as_str)
    {
        super::reasoning::validate_effort(config_effort)?;
        effort = Some(config_effort.to_owned());
    }
    req.reasoning_effort = effort;

    // Carried extras (e.g. an Anthropic client's `service_tier`) ride the
    // flatten map of the request struct.
    req.extra = carried;

    Ok(serde_json::to_value(&req)?)
}

impl OpenAiChatAdapter {
    /// Decode an OpenAI Chat Completions response body.
    pub fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        let resp: llm_proxy_protocol::openai::ChatCompletionResponse =
            serde_json::from_slice(bytes)?;

        let Some(choice) = resp.choices.first() else {
            return Err(ProviderError::EmptyResponse(
                "no choices in response".to_owned(),
            ));
        };
        let msg = choice
            .message
            .as_ref()
            .ok_or_else(|| ProviderError::SseFraming("choice has no message".to_owned()))?;

        let mut content = Vec::new();

        if let Some(ref reasoning) = msg.reasoning_content {
            if !reasoning.is_empty() {
                content.push(CoreContent::Thinking {
                    text: reasoning.clone(),
                    signature: None,
                });
            }
        }

        if !msg.content_is_empty() {
            content.push(CoreContent::Text {
                text: msg.content_text(),
                cache: None,
            });
        }

        if let Some(ref refusal) = msg.refusal {
            if !refusal.is_empty() {
                content.push(CoreContent::Refusal {
                    text: refusal.clone(),
                });
            }
        }

        for tc in &msg.tool_calls {
            let raw_args = tc.function.as_ref().and_then(|f| f.arguments.as_ref());
            let input = raw_args
                .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
                .unwrap_or_else(|| {
                    if raw_args.is_some() {
                        tracing::debug!(
                            "OpenAI: tool_call.arguments was not valid JSON, using empty object"
                        );
                    }
                    serde_json::Value::Object(serde_json::Map::new())
                });

            content.push(CoreContent::ToolUse {
                id: tc.id.clone().unwrap_or_else(|| {
                    tracing::warn!("OpenAI: tool_call missing id in response decode");
                    "<unknown_tool_id>".to_owned()
                }),
                signature: None,
                name: tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.clone())
                    .unwrap_or_else(|| {
                        tracing::warn!("OpenAI: tool_call missing name in response decode");
                        "<unknown_tool>".to_owned()
                    }),
                input,
            });
        }

        if content.is_empty() {
            content.push(CoreContent::Text {
                text: String::new(),
                cache: None,
            });
        }

        let stop_reason = choice
            .finish_reason
            .as_deref()
            .map(map_openai_finish_reason)
            .unwrap_or(StopReason::Unknown("absent".to_owned()));

        // An absent/null usage decodes as None: label it missing rather than
        // booking provider-reported zeros.
        let usage = resp
            .usage
            .as_ref()
            .map(build_usage_from_openai)
            .unwrap_or_else(Usage::synthetic_zero);

        Ok(CoreResponse {
            id: Some(resp.id),
            model: response_model_ref(target),
            content,
            stop_reason,
            stop_sequence: None,
            usage,
            provider_meta: serde_json::Map::new(),
            cost: None,
        })
    }

    /// Create a new stream decoder.
    pub fn new_stream_decoder(&self, target: &ProviderAdapterTarget) -> OpenAiChatStreamDecoder {
        OpenAiChatStreamDecoder {
            model_ref: response_model_ref(target),
            started: false,
            content_index: 0,
            content_started: false,
            reasoning_started: false,
            tool_blocks: HashMap::new(),
            stop_sent: false,
            saw_terminal: false,
        }
    }
}

fn collect_text(content: &[CoreContent]) -> String {
    let mut text = String::new();
    for c in content {
        match c {
            CoreContent::Text { text: t, cache } => {
                if cache.is_some() {
                    tracing::warn!(
                        "OpenAI Chat: cache_control is not supported, dropping cache marker"
                    );
                }
                text.push_str(t);
            }
            other => {
                tracing::warn!(
                    ?other,
                    "OpenAI Chat: dropping non-text content block during collect_text"
                );
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::client_wire::{ClientWireFormat, ClientWireRequest};

    /// An Anthropic-dialect client wire: never the same format as the Chat
    /// adapter, so tests exercise the cross-format normalized encode.
    fn client_wire(core: &CoreRequest) -> ClientWireRequest {
        ClientWireRequest::new(
            ClientWireFormat::AnthropicMessages,
            serde_json::json!({"model": core.model.requested, "messages": []}),
        )
    }
    use llm_proxy_core::AuthStyle;
    use llm_proxy_protocol::core::{CoreMessage, CoreTool, SamplingOptions, UsageProvenance};
    use llm_proxy_protocol::openai::{Choice, UsageInfo};

    fn make_target() -> ProviderAdapterTarget {
        ProviderAdapterTarget {
            provider_name: "test-openai".into(),
            adapter_name: "openai-chat".into(),
            protocol: super::ProtocolName::new(protocols::OPENAI_CHAT_COMPLETIONS),
            endpoint: "https://api.openai.com/v1/chat/completions".into(),
            auth_style: AuthStyle::Bearer,
            api_key: secrecy::SecretString::from("test-key"),
            requested_model: "gpt-4o".into(),
            upstream_model: "gpt-4o-2024-08-06".into(),
            headers: std::sync::Arc::new(std::collections::HashMap::new()),
            lossy: false,
            forward_inbound_headers: std::sync::Arc::new(Vec::new()),
            prompt_cache: None,
        }
    }

    fn make_core_request(messages: Vec<CoreMessage>) -> CoreRequest {
        CoreRequest {
            model: ModelRef {
                requested: "gpt-4o".into(),
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
    fn encode_text_request() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "Hello".into(),
                cache: None,
            }],
        }]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.model, "gpt-4o-2024-08-06");
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
        assert_eq!(
            body.messages[0].content,
            serde_json::Value::String("Hello".into())
        );
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
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.messages[0].role, "system");
        assert_eq!(
            body.messages[0].content,
            serde_json::Value::String("You are helpful".into())
        );
    }

    #[test]
    fn encode_tool_declarations() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tools = vec![CoreTool {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            input_schema: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }];
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.tools.len(), 1);
        assert_eq!(body.tools[0].function.name, "get_weather");
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
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.tool_choice, Some(serde_json::json!("auto")));
    }

    #[test]
    fn encode_tool_result() {
        let core = make_core_request(vec![
            CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: "use tool".into(),
                    cache: None,
                }],
            },
            CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::ToolUse {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                    signature: None,
                }],
            },
            CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: vec![CoreContent::Text {
                        text: "72F sunny".into(),
                        cache: None,
                    }],
                    is_error: false,
                    cache: None,
                }],
            },
        ]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.messages.len(), 3);
        assert_eq!(body.messages[0].role, "user");
        assert_eq!(body.messages[1].role, "assistant");
        assert_eq!(body.messages[1].tool_calls.len(), 1);
        assert_eq!(body.messages[2].role, "tool");
        assert_eq!(body.messages[2].tool_call_id, Some("call_1".to_owned()));
        assert_eq!(
            body.messages[2].content,
            serde_json::Value::String("72F sunny".into())
        );
    }

    /// Encodes the tool replay turn an Anthropic-dialect client sends after a
    /// tool ran: the assistant's `tool_use`, then a user turn carrying its
    /// `tool_result`. Returns the encoded wire body.
    fn encode_tool_result_turn(result_text: &str, is_error: bool) -> serde_json::Value {
        let result_content = if result_text.is_empty() {
            Vec::new()
        } else {
            vec![CoreContent::Text {
                text: result_text.into(),
                cache: None,
            }]
        };
        let core = make_core_request(vec![
            CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::ToolUse {
                    id: "call_1".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({"path": "/tmp/out"}),
                    signature: None,
                }],
            },
            CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: result_content,
                    is_error,
                    cache: None,
                }],
            },
        ]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        serde_json::from_slice(&proxy_req.body).unwrap()
    }

    #[test]
    fn encode_failed_tool_result_marks_the_tool_message() {
        let body = encode_tool_result_turn("disk full", true);
        let tool_message = &body["messages"][1];
        assert_eq!(tool_message["role"], "tool");
        assert_eq!(tool_message["tool_call_id"], "call_1");
        let content = tool_message["content"].as_str().unwrap();
        assert!(
            content.starts_with("[tool error] "),
            "a failed tool result must reach the model marked as failed: {content:?}"
        );
        assert!(
            content.contains("disk full"),
            "the tool's own output must survive: {content:?}"
        );
        assert!(
            tool_message.get("is_error").is_none(),
            "`is_error` is not a Chat Completions field and must not be invented on the wire: {tool_message}"
        );
    }

    #[test]
    fn encode_successful_tool_result_has_no_error_marker() {
        let body = encode_tool_result_turn("disk full", false);
        assert_eq!(body["messages"][1]["content"], "disk full");
    }

    #[test]
    fn encode_failed_tool_result_without_output_still_says_it_failed() {
        let body = encode_tool_result_turn("", true);
        assert_eq!(body["messages"][1]["content"], "[tool error]");
    }

    #[test]
    fn encode_stream_sets_stream_options() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.stream = true;
        core.provider_hints.raw.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.stream, Some(true));
        assert!(body.stream_options.is_some());
        assert_eq!(body.stream_options.unwrap().include_usage, Some(true));
        assert!(proxy_req.stream);
    }

    #[test]
    fn encode_stream_always_requests_usage_from_upstream() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.stream = true;
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.stream, Some(true));
        // Usage is always requested from the upstream on streams; whether
        // the client sees a usage chunk is the client encoder's decision.
        assert!(body.stream_options.is_some());
        assert_eq!(body.stream_options.unwrap().include_usage, Some(true));
    }

    #[test]
    fn encode_uses_upstream_model() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.model, "gpt-4o-2024-08-06");
    }

    #[test]
    fn encode_sampling_options() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling = SamplingOptions {
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(1024),
            stop: Some(vec!["END".into()]),
            ..Default::default()
        };
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.temperature, Some(0.7));
        assert_eq!(body.top_p, Some(0.9));
        assert_eq!(body.max_tokens, Some(1024));
        assert_eq!(body.stop, Some(serde_json::json!("END")));
    }

    #[test]
    fn error_frame_decodes_to_core_error_event() {
        let target = make_target();
        let mut decoder = OpenAiChatAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"error":{"message":"upstream overloaded","type":"server_error","code":"overloaded"}}"#.to_owned(),
        };
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(
            matches!(
                &events[0],
                CoreEvent::Error { error } if error.message() == "upstream overloaded"
            ),
            "the error frame must surface as a core error event, got {events:?}"
        );
    }

    #[test]
    fn truncated_chat_stream_has_no_terminal() {
        let target = make_target();
        let mut decoder = OpenAiChatAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#.to_owned(),
        };
        decoder.decode_frame(&frame).unwrap();
        decoder.finish().unwrap();
        assert!(
            !decoder.saw_terminal(),
            "a stream cut before finish_reason must not count as terminal"
        );
    }

    #[test]
    fn decode_text_response() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion".into(),
            created: 12345,
            model: "gpt-4o-2024-08-06".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String("hello world".into()),
                    reasoning_content: None,
                    tool_calls: vec![],
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                }),
                finish_reason: Some("stop".into()),
                delta: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: 30,
                completion_tokens: 50,
                total_tokens: 80,
                prompt_cache_hit_tokens: Some(10),
                prompt_cache_miss_tokens: Some(20),
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.id, Some("chatcmpl-test".to_owned()));
        assert_eq!(core_resp.model.requested, "gpt-4o");
        assert_eq!(core_resp.stop_reason, StopReason::EndTurn);
        assert_eq!(core_resp.usage.input_tokens, 20);
        assert_eq!(core_resp.usage.output_tokens, 50);
        assert_eq!(
            core_resp.usage.cache_read_input_tokens,
            Some(10),
            "prompt_cache_hit_tokens must map to cache reads"
        );
        assert_eq!(
            core_resp.usage.cache_creation_input_tokens, None,
            "DeepSeek cache-miss tokens are plain input, never cache creation"
        );
        assert_eq!(core_resp.content.len(), 1);
        assert_eq!(
            core_resp.content[0],
            CoreContent::Text {
                text: "hello world".into(),
                cache: None,
            }
        );
    }

    #[test]
    fn decode_response_absent_or_null_usage_is_labelled_synthetic_zero() {
        // "The provider did not report usage" must be distinguishable from
        // "the provider reported zeros": only the latter is bookable as real.
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        for body in [
            r#"{"id":"c","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
            r#"{"id":"c","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":null}"#,
        ] {
            let core_resp = adapter.decode_response(body.as_bytes(), &target).unwrap();
            assert_eq!(
                core_resp.usage.provenance,
                UsageProvenance::SyntheticZero,
                "absent/null usage must be labelled synthetic, not provider-reported: {body}"
            );
            assert_eq!(core_resp.usage.input_tokens, 0, "body: {body}");
            assert_eq!(core_resp.usage.output_tokens, 0, "body: {body}");
        }

        let reported_zeros = r#"{"id":"c","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}"#;
        let core_resp = adapter
            .decode_response(reported_zeros.as_bytes(), &target)
            .unwrap();
        assert_eq!(
            core_resp.usage.provenance,
            UsageProvenance::ProviderReported,
            "explicit provider zeros are real reports, not synthetic: {reported_zeros}"
        );
    }

    #[test]
    fn decode_tool_call_response() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "gpt-4o".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String(String::new()),
                    reasoning_content: None,
                    tool_calls: vec![ToolCall {
                        index: None,
                        id: Some("call_123".into()),
                        r#type: Some("function".into()),
                        function: Some(FunctionCall {
                            name: Some("get_weather".into()),
                            arguments: Some(r#"{"city":"SF"}"#.into()),
                        }),
                    }],
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                }),
                finish_reason: Some("tool_calls".into()),
                delta: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: 200,
                completion_tokens: 80,
                total_tokens: 280,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.stop_reason, StopReason::ToolUse);
        let tool = &core_resp.content[0];
        match tool {
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            _ => panic!("expected ToolUse, got {:?}", tool),
        }
    }

    #[test]
    fn decode_response_preserves_requested_model() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "gpt-4o-2024-08-06".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String("hi".into()),
                    reasoning_content: None,
                    tool_calls: vec![],
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                }),
                finish_reason: Some("stop".into()),
                delta: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.model.requested, "gpt-4o");
    }

    #[test]
    fn stream_text_decoding() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }
        all_events.extend(decoder.finish().unwrap());

        assert_has_event(&all_events, "MessageStart");
        assert_has_event(&all_events, "ContentStart");
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::TextDelta { text, .. } if text == "Hello"
        )));
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::TextDelta { text, .. } if text == " world"
        )));
        assert_has_event(&all_events, "MessageStop");
    }

    #[test]
    fn stream_tool_call_decoding() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"calling"},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
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
        assert_has_event(&all_events, "ToolCallStop");
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
                ..
            }
        )));
    }

    #[test]
    fn stream_negative_tool_index_clamped() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":-1,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();

        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                CoreEvent::ToolCallStart { index, name, .. } => Some((*index, name.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(starts.len(), 1, "expected exactly one ToolCallStart");
        assert_eq!(starts[0].0, 0, "negative index must clamp to 0");
        assert_eq!(starts[0].1, "get_weather");
    }

    #[test]
    fn encode_assistant_only_tool_calls_emits_null_content() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Assistant,
            content: vec![CoreContent::ToolUse {
                id: "call_1".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"city": "SF"}),
                signature: None,
            }],
        }]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.messages.len(), 1);
        assert_eq!(
            body.messages[0].content,
            serde_json::Value::Null,
            "assistant-only-tool-calls must encode content as null"
        );
        assert_eq!(body.messages[0].tool_calls.len(), 1);
    }

    #[test]
    fn encode_assistant_with_text_keeps_string_content() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Assistant,
            content: vec![
                CoreContent::Text {
                    text: "thinking...".into(),
                    cache: None,
                },
                CoreContent::ToolUse {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                    signature: None,
                },
            ],
        }]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(
            body.messages[0].content,
            serde_json::Value::String("thinking...".into())
        );
    }

    #[test]
    fn stream_done_marker_ignored() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("[DONE]");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_malformed_frame_skipped() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("{not valid json}");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_finish_emits_lifecycle() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let events = decoder.finish().unwrap();
        assert_has_event(&events, "MessageStart");
        assert_has_event(&events, "MessageStop");
    }

    #[test]
    fn decode_thinking_response() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "deepseek-chat".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String("answer".into()),
                    reasoning_content: Some("Let me think...".into()),
                    tool_calls: vec![],
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                }),
                finish_reason: Some("stop".into()),
                delta: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        let thinking = core_resp
            .content
            .iter()
            .find(|c| matches!(c, CoreContent::Thinking { .. }));
        assert!(thinking.is_some(), "expected Thinking content");
    }

    #[test]
    fn decode_reasoning_alias_field() {
        let json = r#"{"role":"assistant","content":"hi","reasoning":"thinking tokens"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.reasoning_content, Some("thinking tokens".to_owned()));
    }

    #[test]
    fn decode_refusal_response() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "chatcmpl-test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "gpt-4o".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String(String::new()),
                    reasoning_content: None,
                    tool_calls: vec![],
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: Some("I cannot help with that.".into()),
                    is_error: None,
                }),
                finish_reason: Some("stop".into()),
                delta: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        let refusal = core_resp
            .content
            .iter()
            .find(|c| matches!(c, CoreContent::Refusal { .. }));
        assert!(refusal.is_some(), "expected Refusal content");
    }

    #[test]
    fn decode_empty_choices_returns_error() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "gpt-4o".into(),
            choices: vec![],
            usage: Some(UsageInfo {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let result = adapter.decode_response(&bytes, &target);
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::EmptyResponse(_) => {}
            other => panic!("expected EmptyResponse, got {:?}", other),
        }
    }

    #[test]
    fn stream_usage_only_chunk() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![make_frame(
            r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        )];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::UsageDelta { .. })),
            "usage-only chunk must emit UsageDelta, got: {:?}",
            all_events
        );
    }

    #[test]
    fn stream_reasoning_alias() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"reasoning":"thinking..."},"finish_reason":null}]}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::ThinkingDelta { .. })),
            "reasoning alias must produce ThinkingDelta"
        );
    }

    #[test]
    fn encode_reasoning_effort_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.reasoning_effort = Some("high".into());
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.reasoning_effort, Some("high".to_owned()));
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
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.stop, Some(serde_json::json!(["END", "STOP"])));
    }

    #[test]
    fn encode_stop_single_sequence_scalar() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.stop = Some(vec!["END".into()]);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.stop, Some(serde_json::json!("END")));
    }

    #[test]
    fn encode_thinking_translated_to_reasoning_effort_not_forwarded() {
        // BUG-20: an Anthropic-style `thinking` object must never reach a
        // chat API verbatim; it is translated into `reasoning_effort`.
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.thinking =
            Some(serde_json::json!({"type": "enabled", "budget_tokens": 5000}));
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(
            body.thinking.is_none(),
            "thinking must not be forwarded to a chat API: {:?}",
            body.thinking
        );
        assert_eq!(body.reasoning_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn encode_unknown_reasoning_effort_is_rejected_not_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.reasoning_effort = Some("extreme".into());
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let err = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap_err();
        assert!(err.to_string().contains("extreme"), "{err}");
    }

    #[test]
    fn encode_output_config_effort_maps_to_reasoning_effort() {
        // The field policy declares an Anthropic client's `output_config` as
        // Mapped for the Chat provider: its `effort` must become
        // `reasoning_effort` on the wire, not vanish silently.
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
                "max_tokens": 64,
                "messages": [],
                "output_config": {"effort": "high"},
            }),
        );
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter.encode_request(&core, &target, &client).unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert!(
            body.get("output_config").is_none(),
            "output_config is mapped, not carried: {body}"
        );
    }

    #[test]
    fn encode_output_config_unknown_effort_is_rejected() {
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
                "max_tokens": 64,
                "messages": [],
                "output_config": {"effort": "turbo"},
            }),
        );
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let err = adapter.encode_request(&core, &target, &client).unwrap_err();
        assert!(err.to_string().contains("turbo"), "{err}");
    }

    #[test]
    fn encode_output_config_unmapped_subkey_is_rejected() {
        // `output_config.format` (Anthropic structured outputs) has no Chat
        // translation: the request must fail naming the subfield instead of
        // silently dropping it next to the mapped `effort`.
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
                "max_tokens": 64,
                "messages": [],
                "output_config": {"effort": "high", "format": {"type": "json_schema"}},
            }),
        );
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let err = adapter.encode_request(&core, &target, &client).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("output_config"), "must name the field: {msg}");
        assert!(msg.contains("format"), "must name the subfield: {msg}");
    }

    #[test]
    fn responses_client_extras_reach_chat_upstream() {
        // The Chat Completions API accepts `parallel_tool_calls`, `store` and
        // `prompt_cache_key` with the same names and shapes the Responses API
        // uses, so a Responses-dialect client's fields ride verbatim instead
        // of being rejected by name.
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let client = ClientWireRequest::new(
            ClientWireFormat::OpenAiResponses,
            serde_json::json!({
                "model": core.model.requested,
                "input": [],
                "parallel_tool_calls": false,
                "store": true,
                "prompt_cache_key": "k",
            }),
        );
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter.encode_request(&core, &target, &client).unwrap();

        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(false));
        assert_eq!(body["store"], serde_json::json!(true));
        assert_eq!(body["prompt_cache_key"], serde_json::json!("k"));
    }

    #[test]
    fn encode_tool_choice_none_omitted() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tool_choice = Some(CoreToolChoice::Auto);
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(body.tool_choice.is_some());
    }

    #[test]
    fn encode_input_schema_null_coerced() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tools = vec![CoreTool {
            name: "my_tool".into(),
            description: None,
            input_schema: serde_json::Value::Null,
        }];
        let adapter = OpenAiChatAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ChatCompletionRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let schema = &body.tools[0].function.parameters;
        assert_eq!(schema.as_ref().unwrap()["type"], "object");
    }

    #[test]
    fn decode_preserves_content_order_thinking_before_text() {
        let target = make_target();
        let resp = llm_proxy_protocol::openai::ChatCompletionResponse {
            id: "test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "deepseek-chat".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String("answer".into()),
                    reasoning_content: Some("thoughts".into()),
                    tool_calls: vec![],
                    name: None,
                    tool_call_id: None,
                    cache_control: None,
                    refusal: None,
                    is_error: None,
                }),
                finish_reason: Some("stop".into()),
                delta: None,
            }],
            usage: Some(UsageInfo {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            extra: serde_json::Map::new(),
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = OpenAiChatAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        let thinking_idx = core_resp
            .content
            .iter()
            .position(|c| matches!(c, CoreContent::Thinking { .. }))
            .unwrap();
        let text_idx = core_resp
            .content
            .iter()
            .position(|c| matches!(c, CoreContent::Text { .. }))
            .unwrap();
        assert!(
            thinking_idx < text_idx,
            "Thinking must precede Text in content order"
        );
    }

    #[test]
    fn stream_full_lifecycle_ordering() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            ),
            make_frame(
                r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        let start_idx = all_events
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
        let stop_idx = all_events
            .iter()
            .rposition(|e| matches!(e, CoreEvent::MessageStop { .. }))
            .unwrap();

        assert!(
            start_idx < content_start_idx,
            "MessageStart must precede ContentStart"
        );
        assert!(
            content_start_idx < text_delta_idx,
            "ContentStart must precede TextDelta"
        );
        assert!(
            text_delta_idx < stop_idx,
            "TextDelta must precede MessageStop"
        );
    }

    #[test]
    fn stream_finish_reason_only_chunk() {
        let target = make_target();
        let adapter = OpenAiChatAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![make_frame(
            r#"{"id":"c","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        )];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. }))
        );
        assert!(all_events.iter().any(|e| matches!(
            e,
            CoreEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                ..
            }
        )));
    }

    #[test]
    fn adapter_source_no_forbidden_imports() {
        let source = include_str!("openai_chat.rs");
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

    fn make_frame(data: &str) -> crate::sse::SseFrame {
        crate::sse::SseFrame {
            event: None,
            id: None,
            data: data.to_owned(),
        }
    }

    fn assert_has_event(events: &[CoreEvent], name: &str) {
        let found = match name {
            "MessageStart" => events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. })),
            "ContentStart" => events
                .iter()
                .any(|e| matches!(e, CoreEvent::ContentStart { .. })),
            "MessageStop" => events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. })),
            "ToolCallStop" => events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallStop { .. })),
            _ => false,
        };
        assert!(found, "expected {} event, got: {:?}", name, events);
    }
}
