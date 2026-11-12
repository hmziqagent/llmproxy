//! OpenAI Chat Completions client protocol adapter: translates between the
//! OpenAI Chat wire format and the normalised core protocol types.

use std::sync::Arc;

use crate::client::ProtocolError;
use crate::core::{
    CacheControl, CacheControlType, ContentKind, CoreContent, CoreEvent, CoreMessage, CoreRequest,
    CoreResponse, CoreRole, CoreTool, CoreToolChoice, ModelRef, ProviderHints, RequestMetadata,
    SamplingOptions, StopReason, Usage, json_type_name,
};
#[cfg(test)]
use crate::core::{CoreStreamError, CoreStreamErrorKind};
#[cfg(test)]
use crate::openai::{CacheControl as OpenAICacheControl, StreamOptions};
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice,
    CompletionTokensDetails, FunctionCall, PromptTokensDetails, ToolCall, UsageInfo,
};

fn unix_timestamp_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|e| {
            tracing::warn!("system clock appears to be before UNIX epoch: {e}");
            std::time::Duration::ZERO
        })
        .as_secs() as i64
}

fn clamp_tool_index(index: usize) -> i32 {
    let capped = index.min(i32::MAX as usize);
    let clamped = capped as i32;
    if capped != index {
        tracing::warn!(
            original_index = index,
            clamped_index = clamped,
            "tool-call index exceeds i32::MAX; clamping"
        );
    }
    clamped
}

fn parse_tool_call_arguments(arguments: &str) -> Option<serde_json::Value> {
    match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(value) => Some(value),
        Err(_) => {
            tracing::warn!(
                arguments = %arguments,
                "malformed tool_call.function.arguments; replacing with empty object"
            );
            None
        }
    }
}

/// Decode an OpenAI [`ChatCompletionRequest`] into a normalised [`CoreRequest`].
#[expect(
    clippy::too_many_lines,
    reason = "field-by-field wire decode of the OpenAI Chat request shape, kept \
             in one place so the mapping can be audited against the API schema \
             without jumping between helpers"
)]
pub fn decode_request(req: ChatCompletionRequest) -> Result<CoreRequest, ProtocolError> {
    if req.model.is_empty() {
        return Err(ProtocolError::InvalidRequest("model is required".into()));
    }
    if req.messages.is_empty() {
        return Err(ProtocolError::InvalidRequest("messages is required".into()));
    }

    let model = ModelRef {
        requested: req.model,
        upstream: None,
    };

    let mut system = Vec::new();
    let mut messages = Vec::new();

    for msg in req.messages {
        match msg.role.as_str() {
            "system" => {
                let text = msg.content_text();
                let cache = msg.cache_control.map(|cc| CacheControl {
                    r#type: CacheControlType::from(cc.r#type),
                    ttl: cc.ttl,
                });
                if !text.is_empty() || cache.is_some() {
                    system.push(CoreContent::Text { text, cache });
                }
            }
            "user" => {
                let cache = msg.cache_control.map(|cc| CacheControl {
                    r#type: CacheControlType::from(cc.r#type),
                    ttl: cc.ttl,
                });
                let content = decode_user_content(&msg.content, cache);
                messages.push(CoreMessage {
                    role: CoreRole::User,
                    content,
                });
            }
            "assistant" => {
                let assistant_text = msg.content_text();
                let mut content = Vec::new();
                if let Some(thinking) = msg.reasoning_content {
                    // Preserve an explicitly empty `reasoning_content`: some
                    // providers (DeepSeek) require the field on replayed
                    // assistant turns even when it is empty.
                    content.push(CoreContent::Thinking {
                        text: thinking,
                        signature: None,
                    });
                }
                for tc in msg.tool_calls {
                    let tc_id = tc.id.ok_or_else(|| {
                        ProtocolError::InvalidRequest("tool_call.id is required".into())
                    })?;
                    let Some(tc_function) = tc.function else {
                        return Err(ProtocolError::InvalidRequest(
                            "tool_call.function is required".into(),
                        ));
                    };
                    let tc_name = tc_function.name.ok_or_else(|| {
                        ProtocolError::InvalidRequest("tool_call.function.name is required".into())
                    })?;
                    let tc_args: serde_json::Value = tc_function
                        .arguments
                        .and_then(|a| parse_tool_call_arguments(&a))
                        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                    content.push(CoreContent::ToolUse {
                        id: tc_id,
                        name: tc_name,
                        input: tc_args,
                        signature: None,
                    });
                }
                if !assistant_text.is_empty() {
                    content.push(CoreContent::Text {
                        text: assistant_text,
                        cache: None,
                    });
                }
                messages.push(CoreMessage {
                    role: CoreRole::Assistant,
                    content,
                });
            }
            "tool" => {
                let is_error = msg.is_error.unwrap_or(false);
                let tool_text = msg.content_text();
                let tool_use_id = msg.tool_call_id.ok_or_else(|| {
                    ProtocolError::InvalidRequest(
                        "tool_call_id is required on tool messages".into(),
                    )
                })?;
                let inner_content = if tool_text.is_empty() {
                    vec![]
                } else {
                    vec![CoreContent::Text {
                        text: tool_text,
                        cache: None,
                    }]
                };
                messages.push(CoreMessage {
                    role: CoreRole::Tool,
                    content: vec![CoreContent::ToolResult {
                        tool_use_id,
                        content: inner_content,
                        is_error,
                        cache: None,
                    }],
                });
            }
            other => {
                return Err(ProtocolError::Decode(format!("unknown role: {other}")));
            }
        }
    }

    let tools = req
        .tools
        .into_iter()
        .map(|t| CoreTool {
            name: t.function.name,
            description: t.function.description,
            input_schema: t
                .function
                .parameters
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
        })
        .collect();

    let tool_choice = req.tool_choice.map(decode_tool_choice);

    let mut stop = None;
    if let Some(s) = req.stop {
        match s {
            serde_json::Value::Null => {}
            serde_json::Value::String(st) => stop = Some(vec![st]),
            serde_json::Value::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        serde_json::Value::String(st) => result.push(st),
                        other => {
                            return Err(ProtocolError::InvalidRequest(format!(
                                "stop array must contain only strings, found {}",
                                json_type_name(&other)
                            )));
                        }
                    }
                }
                if result.is_empty() {
                    stop = None;
                } else {
                    stop = Some(result);
                }
            }
            other => {
                return Err(ProtocolError::InvalidRequest(format!(
                    "stop must be a string, array of strings, or null, found {}",
                    json_type_name(&other)
                )));
            }
        }
    }

    let sampling = SamplingOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stop,
        // The OpenAI Chat wire format has no top_k parameter; if a client
        // sends one it stays in `extra` and the cross-format policy decides.
        top_k: None,
        reasoning_effort: req.reasoning_effort,
        thinking: req.thinking,
    };

    let stream = req.stream.unwrap_or(false);

    let mut raw_hints = serde_json::Map::new();
    if let Some(so) = req.stream_options {
        if let Some(include) = so.include_usage {
            raw_hints.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": include}),
            );
        }
    }

    let metadata = RequestMetadata {
        user_id: req.user,
        raw: req.extra,
    };

    let provider_hints = ProviderHints { raw: raw_hints };

    Ok(CoreRequest {
        model,
        system,
        messages,
        tools,
        tool_choice,
        sampling,
        stream,
        metadata,
        provider_hints,
    })
}

/// Decodes a user message `content` field into core blocks: a bare string, or
/// an array of structured parts (`text`, `image_url`). `image_url` parts map
/// to the neutral core image source: `{"type":"url","url":...}` or
/// `{"type":"base64","media_type":...,"data":...}` (from a `data:` URI), the
/// shape the Anthropic wire format already uses.
fn decode_user_content(
    content: &serde_json::Value,
    message_cache: Option<CacheControl>,
) -> Vec<CoreContent> {
    match content {
        serde_json::Value::String(text) => {
            if text.is_empty() && message_cache.is_none() {
                return Vec::new();
            }
            vec![CoreContent::Text {
                text: text.clone(),
                cache: message_cache,
            }]
        }
        serde_json::Value::Array(parts) => {
            let mut blocks = Vec::new();
            let mut cache = message_cache;
            for part in parts {
                let Some(obj) = part.as_object() else {
                    tracing::warn!(
                        ?part,
                        "OpenAI Chat: unsupported content part shape; dropping"
                    );
                    continue;
                };
                match obj.get("type").and_then(|v| v.as_str()) {
                    Some("text") => {
                        let text = obj.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        if text.is_empty() && cache.is_none() {
                            continue;
                        }
                        blocks.push(CoreContent::Text {
                            text: text.to_owned(),
                            cache: cache.take(),
                        });
                    }
                    Some("image_url") => {
                        if let Some(image) = decode_image_url_part(obj) {
                            blocks.push(CoreContent::Image { source: image });
                        }
                    }
                    other => {
                        tracing::warn!(
                            part_type = ?other,
                            "OpenAI Chat: unsupported content part type in user message; dropping"
                        );
                    }
                }
            }
            blocks
        }
        serde_json::Value::Null => Vec::new(),
        other => {
            tracing::warn!(
                ?other,
                "OpenAI Chat: user content is neither a string nor an array of parts; dropping"
            );
            Vec::new()
        }
    }
}

/// Maps an OpenAI `{"type":"image_url","image_url":{"url":...}}` part to the
/// neutral core image source. `data:` URIs become base64 sources; http(s)
/// URLs become url sources. Anything else is dropped with a warning naming
/// the problem, never silently.
fn decode_image_url_part(
    part: &serde_json::Map<String, serde_json::Value>,
) -> Option<serde_json::Value> {
    let url = part
        .get("image_url")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("url"))
        .and_then(|v| v.as_str())?;
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',')?;
        let media_type = meta.split(';').next().unwrap_or_default().to_owned();
        let is_base64 = meta.to_ascii_lowercase().contains(";base64");
        if !is_base64 || media_type.is_empty() || data.is_empty() {
            tracing::warn!(
                data_uri_meta = meta,
                "OpenAI Chat: image_url data URI is not base64 or has no media type; dropping"
            );
            return None;
        }
        return Some(serde_json::json!({
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(serde_json::json!({"type": "url", "url": url}));
    }
    tracing::warn!(url, "OpenAI Chat: unsupported image_url scheme; dropping");
    None
}

fn decode_tool_choice(value: serde_json::Value) -> CoreToolChoice {
    if let Some(s) = value.as_str() {
        match s {
            "auto" => CoreToolChoice::Auto,
            "none" => CoreToolChoice::None,
            "required" => CoreToolChoice::Any,
            _ => CoreToolChoice::Raw(value),
        }
    } else if let Some(obj) = value.as_object() {
        match obj.get("type").and_then(|v| v.as_str()) {
            Some("auto") => CoreToolChoice::Auto,
            Some("none") => CoreToolChoice::None,
            Some("required") => CoreToolChoice::Any,
            Some("function") => {
                let name = obj
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                if name.is_empty() {
                    tracing::warn!(
                        "OpenAI Chat: tool_choice function has empty or missing name field"
                    );
                }
                CoreToolChoice::Tool { name }
            }
            _ => CoreToolChoice::Raw(value),
        }
    } else {
        CoreToolChoice::Raw(value)
    }
}

/// Encode a [`CoreResponse`] into an OpenAI [`ChatCompletionResponse`]:
/// text blocks concatenate without separator and thinking blocks join with
/// `\n`; `stop_sequence` and `provider_meta` are dropped.
#[expect(
    clippy::too_many_lines,
    reason = "field-by-field wire encode of the OpenAI Chat response shape, kept \
             in one place so the mapping can be audited against the API schema \
             without jumping between helpers"
)]
pub fn encode_response(resp: CoreResponse) -> Result<ChatCompletionResponse, ProtocolError> {
    if !resp.provider_meta.is_empty() {
        tracing::warn!(
            meta_keys = resp.provider_meta.len(),
            "provider_meta is non-empty during OpenAI client encode; \
             this data is for provider adapters only and will not be forwarded to the client"
        );
    }

    if let Some(seq) = resp.stop_sequence.as_deref() {
        if !seq.is_empty() {
            tracing::warn!(
                stop_sequence = seq,
                "OpenAI Chat Completions has no stop_sequence field on responses; dropping"
            );
        }
    }

    let id = resp
        .id
        .unwrap_or_else(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()));

    let mut content_text = String::new();
    let mut reasoning = None;
    let mut tool_calls = Vec::new();
    let mut refusal_text: Option<String> = None;

    for block in resp.content {
        match block {
            CoreContent::Text { text, .. } => {
                content_text.push_str(&text);
            }
            CoreContent::Thinking { text, .. } => {
                if reasoning.is_some() {
                    tracing::warn!(
                        "multiple Thinking blocks in response; concatenating with newline separator"
                    );
                }
                reasoning = Some(match reasoning {
                    Some(prev) => format!("{prev}\n{text}"),
                    None => text,
                });
            }
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                let args_str = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_owned());
                tool_calls.push(ToolCall {
                    index: Some(tool_calls.len() as i32),
                    id: Some(id),
                    r#type: Some("function".to_owned()),
                    function: Some(FunctionCall {
                        name: Some(name),
                        arguments: Some(args_str),
                    }),
                });
            }
            CoreContent::Image { .. } => {
                tracing::warn!(
                    "OpenAI Chat Completions does not support image blocks in assistant responses; dropping"
                );
            }
            CoreContent::Document { .. } => {
                tracing::warn!(
                    "OpenAI Chat Completions does not support document blocks in assistant responses; dropping"
                );
            }
            CoreContent::Audio { .. } => {
                tracing::warn!(
                    "OpenAI Chat Completions does not support audio blocks in assistant responses; dropping"
                );
            }
            CoreContent::Video { .. } => {
                tracing::warn!(
                    "OpenAI Chat Completions does not support video blocks in assistant responses; dropping"
                );
            }
            CoreContent::ToolResult { .. } => {
                tracing::warn!(
                    "ToolResult in response content is unexpected for OpenAI Chat encode; dropping"
                );
            }
            CoreContent::RedactedThinking { .. } => {
                tracing::warn!(
                    "OpenAI Chat Completions does not support redacted thinking; omitting block"
                );
            }
            CoreContent::Refusal { text } => {
                refusal_text = Some(text);
            }
            CoreContent::Opaque { .. } => {
                tracing::warn!(
                    "OpenAI Chat Completions has no encoding for opaque client blocks; dropping"
                );
            }
        }
    }

    let finish_reason = encode_finish_reason(resp.stop_reason);

    let message = ChatMessage {
        role: "assistant".to_owned(),
        content: serde_json::Value::String(content_text),
        reasoning_content: reasoning,
        tool_calls,
        name: None,
        tool_call_id: None,
        cache_control: None,
        refusal: refusal_text,
        is_error: None,
    };

    let usage = encode_usage(&resp.usage);

    let created = unix_timestamp_secs();

    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion".to_owned(),
        created,
        model: resp.model.requested,
        choices: vec![Choice {
            index: 0,
            message: Some(message),
            finish_reason: Some(finish_reason),
            delta: None,
        }],
        usage: Some(usage),
        extra: serde_json::Map::new(),
    })
}

fn encode_finish_reason(reason: StopReason) -> String {
    match reason {
        StopReason::EndTurn => "stop".to_owned(),
        StopReason::StopSequence => "stop".to_owned(),
        StopReason::ToolUse => "tool_calls".to_owned(),
        StopReason::MaxTokens => "length".to_owned(),
        StopReason::Refusal => "content_filter".to_owned(),
        StopReason::Error => {
            tracing::warn!("StopReason::Error mapped to 'stop' in OpenAI finish_reason");
            "stop".to_owned()
        }
        StopReason::Unknown(original) => {
            tracing::warn!(
                original_finish_reason = %original,
                "StopReason::Unknown mapped to 'stop' in OpenAI finish_reason"
            );
            "stop".to_owned()
        }
    }
}

fn encode_usage(usage: &Usage) -> UsageInfo {
    let cache_read = usage.cache_read_input_tokens;
    let cache_creation = usage.cache_creation_input_tokens;
    // Core `input_tokens` is net of cache everywhere, but OpenAI clients
    // read `prompt_tokens` as the full prompt: it must include both the
    // cached and the cache-creation tokens.
    let cached = cache_read
        .unwrap_or(0)
        .saturating_add(cache_creation.unwrap_or(0));
    let prompt_tokens = usage.input_tokens.saturating_add(cached);
    // Cache-creation tokens are uncached prompt tokens from the client's
    // view, so they land in "miss", not in "hit"; hit + miss then
    // reconstructs `prompt_tokens` (the DeepSeek invariant).
    let miss_tokens = (cache_read.is_some() || cache_creation.is_some()).then(|| {
        usage
            .input_tokens
            .saturating_add(cache_creation.unwrap_or(0))
    });
    UsageInfo {
        prompt_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: prompt_tokens.saturating_add(usage.output_tokens),
        prompt_cache_hit_tokens: cache_read,
        prompt_cache_miss_tokens: miss_tokens,
        prompt_tokens_details: cache_read.map(|cached_tokens| PromptTokensDetails {
            cached_tokens: Some(cached_tokens),
        }),
        completion_tokens_details: usage.reasoning_tokens.map(|reasoning_tokens| {
            CompletionTokensDetails {
                reasoning_tokens: Some(reasoning_tokens),
            }
        }),
    }
}

/// Stateful encoder that converts [`CoreEvent`] values into OpenAI
/// [`ChatCompletionChunk`] values, reusing a stable `id`/`model`/`created`
/// across chunks (stored as `Arc<str>` so cloning is a refcount bump).
/// The terminal `finish_reason` chunk is held back until
/// [`StreamEncoder::finish`] because providers send the usage chunk after
/// the finish chunk; flushing early would report zeros to clients that
/// asked for usage.
#[derive(Debug)]
pub struct StreamEncoder {
    id: Arc<str>,
    model: Arc<str>,
    created: i64,
    include_usage: bool,
    finished: bool,
    pending_usage: Option<UsageInfo>,
    pending_stop: Option<String>,
}

impl StreamEncoder {
    /// Create a new stream encoder; the route builds these args from the first
    /// `CoreEvent::MessageStart` and the decoded request.
    pub fn new(id: String, model: String, created: i64, include_usage: bool) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            created,
            include_usage,
            finished: false,
            pending_usage: None,
            pending_stop: None,
        }
    }

    /// Encode a single [`CoreEvent`] into zero or more [`ChatCompletionChunk`]s.
    #[expect(
        clippy::too_many_lines,
        reason = "one match arm per CoreEvent variant: the OpenAI Chat SSE chunk \
                 wire shape is a single translation table, and splitting the arms \
                 into helpers would separate each event's field mapping from the \
                 chunk layout it produces"
    )]
    pub fn encode_event(
        &mut self,
        event: CoreEvent,
    ) -> Result<Vec<ChatCompletionChunk>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }

        if self.pending_stop.is_some() {
            if let CoreEvent::Error { .. } = event {
                // The error itself cannot be a chunk — the route emits it as
                // a bare error-object data line — but the buffered finish
                // chunk is still owed to the client: flush it (with usage,
                // when the client asked) and stop encoding.
                return self.finish();
            }
            // The finish chunk is buffered; only a trailing usage update is
            // still meaningful (it becomes the final usage chunk).
            if let CoreEvent::UsageDelta { usage } = event {
                self.pending_usage = Some(encode_usage(&usage));
            } else {
                tracing::trace!(
                    event = ?event,
                    "StreamEncoder: dropping event after buffered MessageStop"
                );
            }
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();

        match event {
            CoreEvent::MessageStart { .. } => {
                chunks.push(self.make_chunk(Choice {
                    index: 0,
                    message: None,
                    finish_reason: None,
                    delta: Some(ChatMessage {
                        role: "assistant".to_owned(),
                        content: serde_json::Value::String(String::new()),
                        reasoning_content: None,
                        tool_calls: vec![],
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    }),
                }));
            }

            CoreEvent::ContentStart { index, kind } => {
                match kind {
                    ContentKind::ToolUse => {}
                    ContentKind::Text | ContentKind::Thinking => {}
                    _ => {
                        tracing::warn!(kind = ?kind, "unsupported ContentKind in OpenAI stream");
                    }
                }
                let _ = index;
            }

            CoreEvent::TextDelta { text, .. } => {
                chunks.push(self.make_chunk(Choice {
                    index: 0,
                    message: None,
                    finish_reason: None,
                    delta: Some(ChatMessage {
                        role: String::new(),
                        content: serde_json::Value::String(text),
                        reasoning_content: None,
                        tool_calls: vec![],
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    }),
                }));
            }

            CoreEvent::ThinkingDelta { text, .. } => {
                chunks.push(self.make_chunk(Choice {
                    index: 0,
                    message: None,
                    finish_reason: None,
                    delta: Some(ChatMessage {
                        role: String::new(),
                        content: serde_json::Value::String(String::new()),
                        reasoning_content: Some(text),
                        tool_calls: vec![],
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    }),
                }));
            }

            CoreEvent::ToolCallStart {
                index, id, name, ..
            } => {
                let clamped = clamp_tool_index(index);
                chunks.push(self.make_chunk(Choice {
                    index: 0,
                    message: None,
                    finish_reason: None,
                    delta: Some(ChatMessage {
                        role: String::new(),
                        content: serde_json::Value::String(String::new()),
                        reasoning_content: None,
                        tool_calls: vec![ToolCall {
                            index: Some(clamped),
                            id: Some(id),
                            r#type: Some("function".to_owned()),
                            function: Some(FunctionCall {
                                name: Some(name),
                                arguments: Some(String::new()),
                            }),
                        }],
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    }),
                }));
            }

            CoreEvent::ToolCallDelta { index, args_delta } => {
                let clamped = clamp_tool_index(index);
                chunks.push(self.make_chunk(Choice {
                    index: 0,
                    message: None,
                    finish_reason: None,
                    delta: Some(ChatMessage {
                        role: String::new(),
                        content: serde_json::Value::String(String::new()),
                        reasoning_content: None,
                        tool_calls: vec![ToolCall {
                            index: Some(clamped),
                            id: None,
                            r#type: None,
                            function: Some(FunctionCall {
                                name: None,
                                arguments: Some(args_delta),
                            }),
                        }],
                        name: None,
                        tool_call_id: None,
                        cache_control: None,
                        refusal: None,
                        is_error: None,
                    }),
                }));
            }

            CoreEvent::ToolCallStop { .. } => {}

            CoreEvent::UsageDelta { usage } => {
                self.pending_usage = Some(encode_usage(&usage));
            }

            CoreEvent::MessageStop { stop_reason, .. } => {
                // Hold the finish chunk back: the usage chunk routinely
                // arrives after it and must carry real numbers, not zeros.
                // `finish()` flushes both, in OpenAI's documented order.
                self.pending_stop = Some(encode_finish_reason(stop_reason));
            }

            CoreEvent::Error { error } => {
                self.finished = true;
                return Err(ProtocolError::Encode(format!(
                    "stream error: {}",
                    error.kind
                )));
            }

            CoreEvent::Ping => {}
        }

        Ok(chunks)
    }

    /// Flush remaining buffered events: a buffered `MessageStop` emits the
    /// finish chunk plus the final usage chunk when `include_usage` is set;
    /// a stream that ended without one emits a synthetic
    /// `finish_reason = "stop"` chunk (with a warning) instead.
    pub fn finish(&mut self) -> Result<Vec<ChatCompletionChunk>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;

        let finish_reason = match self.pending_stop.take() {
            Some(finish) => finish,
            None => {
                tracing::warn!(
                    "StreamEncoder::finish() called without prior MessageStop; emitting synthetic terminal chunk"
                );
                "stop".to_owned()
            }
        };

        let mut chunks = Vec::new();

        chunks.push(self.make_chunk(Choice {
            index: 0,
            message: None,
            finish_reason: Some(finish_reason),
            delta: Some(ChatMessage {
                role: String::new(),
                content: serde_json::Value::String(String::new()),
                reasoning_content: None,
                tool_calls: vec![],
                name: None,
                tool_call_id: None,
                cache_control: None,
                refusal: None,
                is_error: None,
            }),
        }));

        if self.include_usage {
            let usage = self.pending_usage.take().unwrap_or(UsageInfo {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            });
            chunks.push(self.make_usage_chunk(usage));
        }

        Ok(chunks)
    }

    fn make_chunk(&self, choice: Choice) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.to_string(),
            object: "chat.completion.chunk".to_owned(),
            created: self.created,
            model: self.model.to_string(),
            choices: vec![choice],
            usage: None,
        }
    }

    fn make_usage_chunk(&self, usage: UsageInfo) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.to_string(),
            object: "chat.completion.chunk".to_owned(),
            created: self.created,
            model: self.model.to_string(),
            choices: vec![],
            usage: Some(usage),
        }
    }

    /// Mark the encoder as finished so subsequent `finish()` calls return
    /// empty; used after emitting an in-band error. Any buffered finish
    /// chunk has already been flushed by the `Error` event (see
    /// [`StreamEncoder::encode_event`]).
    pub fn mark_finished(&mut self) {
        self.finished = true;
        self.pending_stop = None;
    }
}

/// Build the OpenAI stream-error payload as an error-object `data:` line so
/// a *leading* error crosses the first-byte boundary. Returns the JSON, or
/// `None` if serialization fails; the caller emits the `data: [DONE]`.
pub fn stream_error_sse_payload(error: &crate::core::CoreStreamError) -> Option<String> {
    let error_type = match error.kind {
        crate::core::CoreStreamErrorKind::InvalidRequest => "invalid_request_error",
        crate::core::CoreStreamErrorKind::Authentication => "authentication_error",
        crate::core::CoreStreamErrorKind::Permission => "permission_error",
        crate::core::CoreStreamErrorKind::RateLimit => "rate_limit_error",
        crate::core::CoreStreamErrorKind::Upstream | crate::core::CoreStreamErrorKind::Internal => {
            "server_error"
        }
    };
    #[derive(serde::Serialize)]
    #[serde(deny_unknown_fields)]
    struct ErrorBody {
        error: ErrorDetail,
    }
    #[derive(serde::Serialize)]
    #[serde(deny_unknown_fields)]
    struct ErrorDetail {
        message: String,
        r#type: String,
        code: serde_json::Value,
    }

    let body = ErrorBody {
        error: ErrorDetail {
            message: error.message().to_owned(),
            r#type: error_type.to_owned(),
            code: serde_json::Value::Null,
        },
    };
    serde_json::to_string(&body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_image_url_parts_decode_to_neutral_image_blocks() {
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: serde_json::json!([
                    { "type": "text", "text": "what is this?" },
                    { "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } },
                    { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,aGVsbG8=" } }
                ]),
                reasoning_content: None,
                tool_calls: vec![],
                name: None,
                tool_call_id: None,
                cache_control: None,
                refusal: None,
                is_error: None,
            }],
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            thinking: None,
            tools: vec![],
            tool_choice: None,
            stop: None,
            stream_options: None,
            user: None,
            extra: serde_json::Map::new(),
        };
        let core = decode_request(req).expect("decode");
        let content = &core.messages[0].content;
        assert_eq!(
            content.len(),
            3,
            "text and both images must decode: {content:?}"
        );
        match &content[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "what is this?"),
            other => panic!("expected text, got {other:?}"),
        }
        match &content[1] {
            CoreContent::Image { source } => {
                assert_eq!(source["type"], "url");
                assert_eq!(source["url"], "https://example.com/cat.png");
            }
            other => panic!("expected url image, got {other:?}"),
        }
        match &content[2] {
            CoreContent::Image { source } => {
                assert_eq!(source["type"], "base64");
                assert_eq!(source["media_type"], "image/jpeg");
                assert_eq!(source["data"], "aGVsbG8=");
            }
            other => panic!("expected base64 image, got {other:?}"),
        }
    }

    #[test]
    fn empty_reasoning_content_survives_decode_for_replay() {
        // DeepSeek-style providers require reasoning_content on replayed
        // assistant turns even when empty; an explicit empty string decodes
        // to an (empty) Thinking block instead of vanishing.
        let req = ChatCompletionRequest {
            model: "deepseek-reasoner".into(),
            messages: vec![ChatMessage {
                role: "assistant".into(),
                content: serde_json::json!(""),
                reasoning_content: Some(String::new()),
                tool_calls: vec![],
                name: None,
                tool_call_id: None,
                cache_control: None,
                refusal: None,
                is_error: None,
            }],
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            thinking: None,
            tools: vec![],
            tool_choice: None,
            stop: None,
            stream_options: None,
            user: None,
            extra: serde_json::Map::new(),
        };
        let core = decode_request(req).expect("decode");
        match &core.messages[0].content[0] {
            CoreContent::Thinking { text, .. } => assert_eq!(text, ""),
            other => panic!("expected an empty Thinking block, got {other:?}"),
        }
    }

    use crate::core::UsageProvenance;
    use crate::openai::{FunctionDef, ToolDef};

    fn make_openai_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String("hello".into()),
                reasoning_content: None,
                tool_calls: vec![],
                name: None,
                tool_call_id: None,
                cache_control: None,
                refusal: None,
                is_error: None,
            }],
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            reasoning_effort: None,
            thinking: None,
            tools: vec![],
            tool_choice: None,
            stop: None,
            stream_options: None,
            user: None,
            extra: serde_json::Map::new(),
        }
    }

    fn make_core_response() -> CoreResponse {
        CoreResponse {
            id: Some("chatcmpl-123".into()),
            model: ModelRef {
                requested: "gpt-4o".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: "hello there".into(),
                cache: None,
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::provider_reported(10, 20),
            provider_meta: serde_json::Map::new(),
            cost: None,
        }
    }

    #[test]
    fn plain_text_request_decodes_to_core() {
        let req = make_openai_request();
        let core = decode_request(req).unwrap();
        assert_eq!(core.model.requested, "gpt-4o");
        assert_eq!(core.messages.len(), 1);
        assert_eq!(core.messages[0].role, CoreRole::User);
        match &core.messages[0].content[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "hello"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn system_prompt_decodes_to_core() {
        let mut req = make_openai_request();
        req.messages.insert(
            0,
            ChatMessage {
                role: "system".into(),
                content: serde_json::Value::String("You are helpful".into()),
                reasoning_content: None,
                tool_calls: vec![],
                name: None,
                tool_call_id: None,
                cache_control: None,
                refusal: None,
                is_error: None,
            },
        );
        let core = decode_request(req).unwrap();
        assert_eq!(core.system.len(), 1);
        match &core.system[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "You are helpful"),
            _ => panic!("expected Text"),
        }
        assert_eq!(core.messages.len(), 1);
        assert_eq!(core.messages[0].role, CoreRole::User);
    }

    #[test]
    fn assistant_with_tool_calls_decodes_to_core() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String(String::new()),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                index: Some(0),
                id: Some("call_1".into()),
                r#type: Some("function".into()),
                function: Some(FunctionCall {
                    name: Some("get_weather".into()),
                    arguments: Some("{\"city\":\"SF\"}".into()),
                }),
            }],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.messages[1].role, CoreRole::Assistant);
        match &core.messages[1].content[0] {
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn tool_result_decodes_to_core() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "tool".into(),
            content: serde_json::Value::String("72F and sunny".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: Some("call_1".into()),
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.messages[1].role, CoreRole::Tool);
        match &core.messages[1].content[0] {
            CoreContent::ToolResult {
                tool_use_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn reasoning_content_decodes_to_thinking() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String(String::new()),
            reasoning_content: Some("let me think...".into()),
            tool_calls: vec![],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::Thinking { text, signature } => {
                assert_eq!(text, "let me think...");
                assert!(signature.is_none());
            }
            _ => panic!("expected Thinking"),
        }
    }

    #[test]
    fn tool_choice_decodes_to_core() {
        let mut req = make_openai_request();
        req.tool_choice = Some(serde_json::json!("auto"));
        let core = decode_request(req).unwrap();
        assert_eq!(core.tool_choice, Some(CoreToolChoice::Auto));

        let mut req2 = make_openai_request();
        req2.tool_choice =
            Some(serde_json::json!({"type": "function", "function": {"name": "get_weather"}}));
        let core2 = decode_request(req2).unwrap();
        match core2.tool_choice {
            Some(CoreToolChoice::Tool { name }) => assert_eq!(name, "get_weather"),
            _ => panic!("expected Tool"),
        }

        let mut req3 = make_openai_request();
        req3.tool_choice = Some(serde_json::json!("required"));
        let core3 = decode_request(req3).unwrap();
        assert_eq!(core3.tool_choice, Some(CoreToolChoice::Any));
    }

    #[test]
    fn tool_choice_none_decodes_to_core() {
        let mut req = make_openai_request();
        req.tool_choice = Some(serde_json::json!("none"));
        let core = decode_request(req).unwrap();
        assert_eq!(core.tool_choice, Some(CoreToolChoice::None));

        let mut req2 = make_openai_request();
        req2.tool_choice = Some(serde_json::json!({"type": "none"}));
        let core2 = decode_request(req2).unwrap();
        assert_eq!(core2.tool_choice, Some(CoreToolChoice::None));
    }

    #[test]
    fn tool_choice_raw_preserved_for_unrecognized() {
        let mut req = make_openai_request();
        req.tool_choice = Some(serde_json::json!({"type": "future_mode"}));
        let core = decode_request(req).unwrap();
        match core.tool_choice {
            Some(CoreToolChoice::Raw(_)) => {}
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn cache_control_decodes_to_core() {
        let mut req = make_openai_request();
        req.messages[0].cache_control = Some(OpenAICacheControl {
            r#type: "ephemeral".into(),
            ttl: None,
        });
        let core = decode_request(req).unwrap();
        match &core.messages[0].content[0] {
            CoreContent::Text { cache, .. } => {
                assert!(cache.is_some());
                assert_eq!(cache.as_ref().unwrap().r#type, CacheControlType::Ephemeral);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn stream_options_decode_to_provider_hints() {
        let mut req = make_openai_request();
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let core = decode_request(req).unwrap();
        assert!(core.provider_hints.raw.contains_key("stream_options"));
        assert_eq!(
            core.provider_hints.raw["stream_options"]["include_usage"],
            true
        );
    }

    #[test]
    fn multiple_messages_decode_to_ordered_core_messages() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String("hi there".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        req.messages.push(ChatMessage {
            role: "user".into(),
            content: serde_json::Value::String("how are you?".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.messages.len(), 3);
        assert_eq!(core.messages[0].role, CoreRole::User);
        assert_eq!(core.messages[1].role, CoreRole::Assistant);
        assert_eq!(core.messages[2].role, CoreRole::User);
    }

    #[test]
    fn malformed_request_returns_protocol_error() {
        let mut req = make_openai_request();
        req.model = String::new();
        let err = decode_request(req).unwrap_err();
        assert!(matches!(err, ProtocolError::InvalidRequest(_)));

        let mut req2 = make_openai_request();
        req2.messages = vec![];
        let err2 = decode_request(req2).unwrap_err();
        assert!(matches!(err2, ProtocolError::InvalidRequest(_)));
    }

    #[test]
    fn unknown_role_returns_decode_error() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "unknown_role".into(),
            content: serde_json::Value::String("test".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let err = decode_request(req).unwrap_err();
        assert!(matches!(err, ProtocolError::Decode(_)));
    }

    #[test]
    fn empty_user_content_with_cache_control_preserves_cache() {
        let mut req = make_openai_request();
        req.messages[0].content = serde_json::Value::String(String::new());
        req.messages[0].cache_control = Some(OpenAICacheControl {
            r#type: "ephemeral".into(),
            ttl: None,
        });
        let core = decode_request(req).unwrap();
        assert_eq!(
            core.messages[0].content.len(),
            1,
            "empty user message with cache_control should still produce a content block"
        );
        match &core.messages[0].content[0] {
            CoreContent::Text { text, cache } => {
                assert!(text.is_empty());
                assert!(
                    cache.is_some(),
                    "cache_control must be preserved even with empty text"
                );
                assert_eq!(cache.as_ref().unwrap().r#type, CacheControlType::Ephemeral);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn empty_system_content_with_cache_control_preserves_cache() {
        let mut req = make_openai_request();
        req.messages.insert(
            0,
            ChatMessage {
                role: "system".into(),
                content: serde_json::Value::String(String::new()),
                reasoning_content: None,
                tool_calls: vec![],
                name: None,
                tool_call_id: None,
                cache_control: Some(OpenAICacheControl {
                    r#type: "ephemeral".into(),
                    ttl: None,
                }),
                refusal: None,
                is_error: None,
            },
        );
        let core = decode_request(req).unwrap();
        assert_eq!(
            core.system.len(),
            1,
            "empty system message with cache_control should produce a system block"
        );
        match &core.system[0] {
            CoreContent::Text { text, cache } => {
                assert!(text.is_empty());
                assert!(
                    cache.is_some(),
                    "cache_control must be preserved even with empty text"
                );
                assert_eq!(cache.as_ref().unwrap().r#type, CacheControlType::Ephemeral);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn sampling_options_preserved_in_decode() {
        let mut req = make_openai_request();
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.max_tokens = Some(512);
        req.stop = Some(serde_json::json!(["STOP"]));
        req.reasoning_effort = Some("high".into());
        let core = decode_request(req).unwrap();
        assert_eq!(core.sampling.temperature, Some(0.7));
        assert_eq!(core.sampling.top_p, Some(0.9));
        assert_eq!(core.sampling.max_tokens, Some(512));
        assert_eq!(core.sampling.stop, Some(vec!["STOP".to_owned()]));
        assert_eq!(core.sampling.reasoning_effort, Some("high".into()));
    }

    #[test]
    fn tools_decoded_to_core() {
        let mut req = make_openai_request();
        req.tools = vec![ToolDef {
            r#type: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
            },
        }];
        let core = decode_request(req).unwrap();
        assert_eq!(core.tools.len(), 1);
        assert_eq!(core.tools[0].name, "get_weather");
    }

    #[test]
    fn core_text_response_encodes_to_route_response() {
        let resp = make_core_response();
        let out = encode_response(resp).unwrap();
        assert_eq!(out.id, "chatcmpl-123");
        assert_eq!(out.model, "gpt-4o");
        assert_eq!(out.object, "chat.completion");
        assert_eq!(out.choices.len(), 1);
        assert_eq!(out.choices[0].finish_reason.as_deref(), Some("stop"));
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.content, serde_json::Value::String("hello there".into()));
        assert_eq!(msg.role, "assistant");
    }

    #[test]
    fn core_tool_use_response_encodes_to_route_response() {
        let resp = CoreResponse {
            id: Some("chatcmpl-123".into()),
            model: ModelRef {
                requested: "gpt-4o".into(),
                upstream: None,
            },
            content: vec![CoreContent::ToolUse {
                id: "call_1".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"city": "SF"}),
                signature: None,
            }],
            stop_reason: StopReason::ToolUse,
            stop_sequence: None,
            usage: Usage::provider_reported(10, 20),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        assert_eq!(out.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            msg.tool_calls[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(encode_finish_reason(StopReason::ToolUse), "tool_calls");
        assert_eq!(encode_finish_reason(StopReason::MaxTokens), "length");
        assert_eq!(encode_finish_reason(StopReason::EndTurn), "stop");
        assert_eq!(encode_finish_reason(StopReason::StopSequence), "stop");
        assert_eq!(
            encode_finish_reason(StopReason::Unknown("test-unknown".to_owned())),
            "stop"
        );
        assert_eq!(encode_finish_reason(StopReason::Refusal), "content_filter");
        assert_eq!(encode_finish_reason(StopReason::Error), "stop");
    }

    #[test]
    fn usage_with_cache_sums_prompt_tokens_and_details() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 200,
            reasoning_tokens: None,
            cache_creation_input_tokens: Some(10),
            cache_read_input_tokens: Some(5),
            provenance: UsageProvenance::ProviderReported,
        };
        let out = encode_usage(&usage);
        assert_eq!(out.prompt_tokens, 115);
        assert_eq!(out.completion_tokens, 200);
        assert_eq!(out.total_tokens, 315);
        assert_eq!(out.prompt_cache_hit_tokens, Some(5));
        assert_eq!(out.prompt_cache_miss_tokens, Some(110));
        assert_eq!(
            out.prompt_tokens_details,
            Some(PromptTokensDetails {
                cached_tokens: Some(5)
            })
        );
    }

    #[test]
    fn cache_hit_plus_miss_equals_prompt_tokens() {
        // hit + miss must reconstruct prompt_tokens for every client that
        // reads the DeepSeek-style hit/miss fields.
        let tuples = [
            (100, Some(5), Some(10)),
            (100, Some(5), None),
            (50, Some(0), Some(0)),
            (30, Some(30), None),
        ];
        for (input, read, creation) in tuples {
            let usage = Usage {
                input_tokens: input,
                output_tokens: 3,
                reasoning_tokens: None,
                cache_creation_input_tokens: creation,
                cache_read_input_tokens: read,
                provenance: UsageProvenance::ProviderReported,
            };
            let out = encode_usage(&usage);
            if out.prompt_cache_hit_tokens.is_none() {
                continue;
            }
            assert_eq!(
                out.prompt_cache_hit_tokens.unwrap_or(0)
                    + out.prompt_cache_miss_tokens.unwrap_or(0),
                out.prompt_tokens,
                "hit + miss must equal prompt_tokens for input {input}, read {read:?}, creation {creation:?}"
            );
        }
    }

    #[test]
    fn usage_without_cache_reports_bare_prompt_tokens() {
        let usage = Usage::provider_reported(100, 200);
        let out = encode_usage(&usage);
        assert_eq!(out.prompt_tokens, 100);
        assert_eq!(out.prompt_cache_hit_tokens, None);
        assert_eq!(out.prompt_cache_miss_tokens, None);
        assert_eq!(out.prompt_tokens_details, None);
        assert_eq!(out.completion_tokens_details, None);
        // Clients with no cache must see exactly today's JSON: absent keys,
        // not nulls.
        let json = serde_json::to_value(&out).unwrap();
        assert!(json.get("prompt_cache_hit_tokens").is_none(), "{json}");
        assert!(json.get("prompt_cache_miss_tokens").is_none(), "{json}");
        assert!(json.get("prompt_tokens_details").is_none(), "{json}");
        assert!(json.get("completion_tokens_details").is_none(), "{json}");
    }

    #[test]
    fn reasoning_tokens_map_to_completion_tokens_details() {
        let mut usage = Usage::provider_reported(10, 30);
        usage.reasoning_tokens = Some(9);
        let out = encode_usage(&usage);
        assert_eq!(
            out.completion_tokens_details,
            Some(CompletionTokensDetails {
                reasoning_tokens: Some(9)
            })
        );
    }

    #[test]
    fn reasoning_content_encodes() {
        let resp = CoreResponse {
            id: Some("chatcmpl-123".into()),
            model: ModelRef {
                requested: "gpt-4o".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Thinking {
                    text: "let me think...".into(),
                    signature: None,
                },
                CoreContent::Text {
                    text: "answer".into(),
                    cache: None,
                },
            ],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.reasoning_content.as_deref(), Some("let me think..."));
        assert_eq!(msg.content, serde_json::Value::String("answer".into()));
    }

    #[test]
    fn streaming_text_event_mapping() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        let chunks = enc
            .encode_event(CoreEvent::TextDelta {
                index: 0,
                text: "hello".into(),
            })
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].object, "chat.completion.chunk");
        let delta = chunks[0].choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.content, serde_json::Value::String("hello".into()));
    }

    #[test]
    fn streaming_thinking_event_mapping() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        let chunks = enc
            .encode_event(CoreEvent::ThinkingDelta {
                index: 0,
                text: "hmm".into(),
            })
            .unwrap();
        assert_eq!(chunks.len(), 1);
        let delta = chunks[0].choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.reasoning_content.as_deref(), Some("hmm"));
    }

    #[test]
    fn streaming_tool_event_mapping() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);

        let start = enc
            .encode_event(CoreEvent::ToolCallStart {
                index: 0,
                id: "call_1".into(),
                name: "get_weather".into(),
                signature: None,
            })
            .unwrap();
        assert_eq!(start.len(), 1);
        let delta = start[0].choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            delta.tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .name
                .as_deref(),
            Some("get_weather")
        );

        let delta_chunks = enc
            .encode_event(CoreEvent::ToolCallDelta {
                index: 0,
                args_delta: "{\"city\":".into(),
            })
            .unwrap();
        assert_eq!(delta_chunks.len(), 1);
        let delta = delta_chunks[0].choices[0].delta.as_ref().unwrap();
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(
            delta.tool_calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref(),
            Some("{\"city\":")
        );
    }

    #[test]
    fn streaming_message_stop_emits_finish() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        let chunks = enc
            .encode_event(CoreEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                stop_sequence: None,
            })
            .unwrap();
        assert!(
            chunks.is_empty(),
            "finish chunk is held back until finish(): got {chunks:?}"
        );
        let chunks = enc.finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn streaming_error_after_message_stop_flushes_finish_chunk() {
        // The error itself is a route-level data line (the chunk schema cannot
        // carry it), but the buffered finish chunk must still reach the client
        // before the error ends the stream — not be dropped by it.
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, true);
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::ToolUse,
            stop_sequence: None,
        })
        .unwrap();
        let chunks = enc
            .encode_event(CoreEvent::Error {
                error: CoreStreamError::new(
                    CoreStreamErrorKind::Upstream,
                    "connection reset after terminal".into(),
                ),
            })
            .unwrap();
        let finish_pos = chunks
            .iter()
            .position(|c| {
                c.choices
                    .first()
                    .is_some_and(|ch| ch.finish_reason.is_some())
            })
            .expect("buffered finish chunk must flush ahead of the error");
        assert_eq!(
            chunks[finish_pos].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert!(
            chunks.iter().any(|c| c.usage.is_some()),
            "include_usage clients still get the usage chunk: {chunks:?}"
        );
        assert!(
            enc.finish().unwrap().is_empty(),
            "no synthetic second terminal after the error"
        );
    }

    #[test]
    fn streaming_message_stop_with_usage_when_include() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, true);
        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(50, 100),
        })
        .unwrap();
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let chunks = enc.finish().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices.len(), 1);
        assert!(chunks[1].usage.is_some());
        assert_eq!(chunks[1].usage.as_ref().unwrap().prompt_tokens, 50);
    }

    #[test]
    fn usage_after_finish_chunk_reaches_include_usage_clients() {
        // OpenAI's documented chunk order: finish_reason chunk first, then
        // the usage chunk. A client that asked for usage must see the real
        // numbers, not the zeros buffered at finish time.
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, true);
        enc.encode_event(CoreEvent::TextDelta {
            index: 0,
            text: "hi".into(),
        })
        .unwrap();
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(100, 5),
        })
        .unwrap();

        let chunks = enc.finish().unwrap();
        let usage_chunk = chunks
            .iter()
            .find(|c| c.usage.is_some())
            .expect("usage chunk after finish()");
        let usage = usage_chunk.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 5);
        let finish_pos = chunks
            .iter()
            .position(|c| {
                c.choices
                    .first()
                    .is_some_and(|ch| ch.finish_reason.is_some())
            })
            .expect("finish chunk");
        let usage_pos = chunks
            .iter()
            .position(|c| c.usage.is_some())
            .expect("usage chunk");
        assert!(
            finish_pos < usage_pos,
            "finish chunk must precede the usage chunk: {chunks:?}"
        );
    }

    #[test]
    fn streaming_no_usage_chunk_when_client_did_not_ask() {
        // The proxy always asks the upstream for usage, but a client that did
        // not set stream_options.include_usage must not receive a usage chunk.
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(50, 100),
        })
        .unwrap();
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let chunks = enc.finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].usage.is_none());
    }

    #[test]
    fn streaming_error_event_returns_encode_error() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        let result = enc.encode_event(CoreEvent::Error {
            error: CoreStreamError::new(CoreStreamErrorKind::RateLimit, "too many requests".into()),
        });
        assert!(result.is_err());
        assert!(matches!(result, Err(ProtocolError::Encode(_))));
    }

    #[test]
    fn stream_error_sse_payload_builds_typed_error_object() {
        let payload = stream_error_sse_payload(&CoreStreamError::new(
            CoreStreamErrorKind::RateLimit,
            "too many requests".into(),
        ))
        .expect("payload serializes");
        let json: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(json["error"]["message"], "too many requests");
        assert_eq!(json["error"]["type"], "rate_limit_error");
        assert!(json["error"]["code"].is_null());
        assert!(json.get("type").is_none() || json["type"].is_null());
    }

    #[test]
    fn stream_error_sse_payload_upstream_is_server_error() {
        let payload = stream_error_sse_payload(&CoreStreamError::new(
            CoreStreamErrorKind::Upstream,
            "provider failed".into(),
        ))
        .expect("payload serializes");
        let json: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["message"], "provider failed");
    }

    #[test]
    fn streaming_ping_is_noop() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        let chunks = enc.encode_event(CoreEvent::Ping).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn streaming_events_after_stop_are_ignored() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let chunks = enc
            .encode_event(CoreEvent::TextDelta {
                index: 0,
                text: "ignored".into(),
            })
            .unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn streaming_finish_after_stop_flushes_buffered_terminal_once() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let remaining = enc.finish().unwrap();
        assert_eq!(
            remaining.len(),
            1,
            "buffered finish chunk flushes on finish()"
        );
        let again = enc.finish().unwrap();
        assert!(again.is_empty(), "second finish() must be empty");
    }

    #[test]
    fn streaming_finish_without_prior_events_emits_synthetic_terminal() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, false);
        let remaining = enc.finish().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(
            remaining[0].choices[0].finish_reason.as_deref(),
            Some("stop")
        );
    }

    #[test]
    fn every_core_event_variant_maps_or_errors_intentionally() {
        let content_kinds: Vec<ContentKind> = vec![
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
        for (i, kind) in content_kinds.iter().enumerate() {
            let mut enc = StreamEncoder::new("chatcmpl-1".into(), "m".into(), 1000, false);
            let result = enc.encode_event(CoreEvent::ContentStart {
                index: i,
                kind: *kind,
            });
            assert!(result.is_ok(), "ContentStart {:?} should not error", kind);
        }

        let variants: Vec<CoreEvent> = vec![
            CoreEvent::MessageStart {
                id: None,
                model: ModelRef {
                    requested: "m".into(),
                    upstream: None,
                },
            },
            CoreEvent::TextDelta {
                index: 0,
                text: "hello".into(),
            },
            CoreEvent::ThinkingDelta {
                index: 1,
                text: "hmm".into(),
            },
            CoreEvent::ToolCallStart {
                index: 2,
                id: "call_1".into(),
                name: "fn".into(),
                signature: None,
            },
            CoreEvent::ToolCallDelta {
                index: 2,
                args_delta: "{}".into(),
            },
            CoreEvent::ToolCallStop { index: 2 },
            CoreEvent::UsageDelta {
                usage: Usage::default(),
            },
            CoreEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                stop_sequence: None,
            },
            CoreEvent::Error {
                error: CoreStreamError::new(CoreStreamErrorKind::Internal, "test".into()),
            },
            CoreEvent::Ping,
        ];
        for event in &variants {
            let mut enc = StreamEncoder::new("chatcmpl-1".into(), "m".into(), 1000, false);
            let result = enc.encode_event(event.clone());
            match event {
                CoreEvent::Error { .. } => {
                    assert!(result.is_err(), "Error event should return Err");
                }
                _ => {
                    assert!(result.is_ok(), "event {event:?} should not error");
                }
            }
        }
    }

    #[test]
    fn encode_image_drops_with_warning() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Image {
                source: serde_json::json!({"url": "http://example.com/img.png"}),
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
        assert!(msg.tool_calls.is_empty());
    }

    #[test]
    fn encode_refusal_uses_native_refusal_field() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Refusal {
                text: "I cannot help".into(),
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.refusal.as_deref(), Some("I cannot help"));
        assert!(msg.content_is_empty());
    }

    #[test]
    fn encode_document_drops_with_warning() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Document {
                source: serde_json::json!({"url": "http://example.com/doc.pdf"}),
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
    }

    #[test]
    fn encode_audio_drops_with_warning() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Audio {
                source: serde_json::json!({"data": "base64..."}),
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
    }

    #[test]
    fn encode_video_drops_with_warning() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Video {
                source: serde_json::json!({"url": "http://example.com/vid.mp4"}),
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
    }

    #[test]
    fn encode_tool_result_in_response_drops_with_warning() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::ToolResult {
                tool_use_id: "call_1".into(),
                content: vec![],
                is_error: false,
                cache: None,
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
    }

    #[test]
    fn provider_hints_raw_preserved_in_decode() {
        let mut req = make_openai_request();
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let core = decode_request(req).unwrap();
        assert!(core.provider_hints.raw.contains_key("stream_options"));
    }

    #[test]
    fn user_field_maps_to_metadata_user_id() {
        let mut req = make_openai_request();
        req.user = Some("user-abc".into());
        let core = decode_request(req).unwrap();
        assert_eq!(core.metadata.user_id.as_deref(), Some("user-abc"));
    }

    #[test]
    fn extra_fields_preserved_in_metadata_raw() {
        let mut req = make_openai_request();
        req.extra
            .insert("custom_field".into(), serde_json::json!("custom_value"));
        let core = decode_request(req).unwrap();
        assert_eq!(
            core.metadata.raw.get("custom_field").unwrap(),
            "custom_value"
        );
    }

    #[test]
    fn stream_flag_preserved_in_decode() {
        let mut req = make_openai_request();
        req.stream = Some(true);
        let core = decode_request(req).unwrap();
        assert!(core.stream);

        let req2 = make_openai_request();
        let core2 = decode_request(req2).unwrap();
        assert!(!core2.stream);
    }

    #[test]
    fn encode_redacted_thinking_drops_with_warning() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::RedactedThinking {
                data: serde_json::json!({"redacted": true}),
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
    }

    #[test]
    fn encode_redacted_thinking_preserves_prior_blocks() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Text {
                    text: "hello".into(),
                    cache: None,
                },
                CoreContent::RedactedThinking {
                    data: serde_json::json!({"redacted": true}),
                },
            ],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.content, serde_json::Value::String("hello".into()));
    }

    #[test]
    fn empty_content_produces_valid_response() {
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
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert!(msg.content_is_empty());
        assert!(msg.tool_calls.is_empty());
    }

    #[test]
    fn id_generated_when_missing() {
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
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        assert!(out.id.starts_with("chatcmpl-"));
    }

    #[test]
    fn client_openai_chat_does_not_import_forbidden_modules() {
        let source = include_str!("openai_chat.rs");
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") {
                continue;
            }
            if trimmed.starts_with("use ") {
                assert!(
                    !trimmed.contains("llm_proxy_provider"),
                    "client/openai_chat.rs must not import llm_proxy_provider"
                );
                assert!(
                    !trimmed.contains("llm_proxy_server"),
                    "client/openai_chat.rs must not import llm_proxy_server"
                );
                assert!(
                    !trimmed.contains("crate::config"),
                    "client/openai_chat.rs must not import crate::config"
                );
                assert!(
                    !trimmed.contains("crate::routing"),
                    "client/openai_chat.rs must not import crate::routing"
                );
                assert!(
                    !trimmed.contains("crate::transformer"),
                    "client/openai_chat.rs must not import crate::transformer"
                );
            }
        }
    }

    #[test]
    fn decode_empty_model_returns_error() {
        let mut req = make_openai_request();
        req.model = String::new();
        assert!(matches!(
            decode_request(req),
            Err(ProtocolError::InvalidRequest(_))
        ));
    }

    #[test]
    fn decode_empty_messages_returns_error() {
        let mut req = make_openai_request();
        req.messages = vec![];
        assert!(matches!(
            decode_request(req),
            Err(ProtocolError::InvalidRequest(_))
        ));
    }

    #[test]
    fn decode_null_optional_fields() {
        let req = make_openai_request();
        let core = decode_request(req).unwrap();
        assert!(core.sampling.temperature.is_none());
        assert!(core.sampling.top_p.is_none());
        assert!(core.sampling.max_tokens.is_none());
        assert!(!core.stream);
    }

    #[test]
    fn decode_max_tokens_zero() {
        let mut req = make_openai_request();
        req.max_tokens = Some(0);
        let core = decode_request(req).unwrap();
        assert_eq!(core.sampling.max_tokens, Some(0));
    }

    #[test]
    fn encode_response_with_stop_sequence_drops_with_warning() {
        let resp = CoreResponse {
            id: Some("chatcmpl-123".into()),
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: "hello".into(),
                cache: None,
            }],
            stop_reason: StopReason::StopSequence,
            stop_sequence: Some("\n".into()),
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        assert_eq!(out.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn encode_multiple_thinking_blocks_concatenates() {
        let resp = CoreResponse {
            id: Some("chatcmpl-123".into()),
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Thinking {
                    text: "first".into(),
                    signature: None,
                },
                CoreContent::Thinking {
                    text: "second".into(),
                    signature: None,
                },
            ],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.reasoning_content.as_deref(), Some("first\nsecond"));
    }

    #[test]
    fn large_content_string_encodes() {
        let large_text = "x".repeat(100_000);
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: large_text.clone(),
                cache: None,
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        let msg = out.choices[0].message.as_ref().unwrap();
        assert_eq!(msg.content, serde_json::Value::String(large_text));
    }

    #[test]
    fn streaming_finish_with_include_usage_emits_usage_chunk() {
        let mut enc = StreamEncoder::new("chatcmpl-1".into(), "gpt-4o".into(), 1000, true);
        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(50, 100),
        })
        .unwrap();
        let remaining = enc.finish().unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(
            remaining[0].choices[0].finish_reason.as_deref(),
            Some("stop")
        );
        assert!(remaining[1].usage.is_some());
        assert_eq!(remaining[1].usage.as_ref().unwrap().prompt_tokens, 50);
    }

    #[test]
    fn tool_call_missing_id_returns_error() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String(String::new()),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                index: Some(0),
                id: None,
                r#type: Some("function".into()),
                function: Some(FunctionCall {
                    name: Some("get_weather".into()),
                    arguments: Some("{}".into()),
                }),
            }],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let err = decode_request(req).unwrap_err();
        match &err {
            ProtocolError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("tool_call.id"),
                    "expected tool_call.id error, got: {msg}"
                );
            }
            _ => panic!("expected InvalidRequest, got: {:?}", err),
        }
    }

    #[test]
    fn tool_call_missing_function_returns_error() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String(String::new()),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                index: Some(0),
                id: Some("call_1".into()),
                r#type: Some("function".into()),
                function: None,
            }],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let err = decode_request(req).unwrap_err();
        match &err {
            ProtocolError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("tool_call.function"),
                    "expected tool_call.function error, got: {msg}"
                );
            }
            _ => panic!("expected InvalidRequest, got: {:?}", err),
        }
    }

    #[test]
    fn tool_call_missing_function_name_returns_error() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String(String::new()),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                index: Some(0),
                id: Some("call_1".into()),
                r#type: Some("function".into()),
                function: Some(FunctionCall {
                    name: None,
                    arguments: Some("{}".into()),
                }),
            }],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let err = decode_request(req).unwrap_err();
        match &err {
            ProtocolError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("tool_call.function.name"),
                    "expected tool_call.function.name error, got: {msg}"
                );
            }
            _ => panic!("expected InvalidRequest, got: {:?}", err),
        }
    }

    #[test]
    fn tool_message_missing_tool_call_id_returns_error() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "tool".into(),
            content: serde_json::Value::String("result".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let err = decode_request(req).unwrap_err();
        match &err {
            ProtocolError::InvalidRequest(msg) => {
                assert!(
                    msg.contains("tool_call_id"),
                    "expected tool_call_id error, got: {msg}"
                );
            }
            _ => panic!("expected InvalidRequest, got: {:?}", err),
        }
    }

    #[test]
    fn malformed_tool_call_arguments_replaced_with_empty_object() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String(String::new()),
            reasoning_content: None,
            tool_calls: vec![ToolCall {
                index: Some(0),
                id: Some("call_1".into()),
                r#type: Some("function".into()),
                function: Some(FunctionCall {
                    name: Some("get_weather".into()),
                    arguments: Some("not valid json{}".into()),
                }),
            }],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::ToolUse { input, .. } => {
                assert_eq!(
                    input,
                    &serde_json::json!({}),
                    "malformed args should become empty object"
                );
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn tool_message_is_error_decodes_to_core() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "tool".into(),
            content: serde_json::Value::String("boom".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: Some("call_1".into()),
            cache_control: None,
            refusal: None,
            is_error: Some(true),
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::ToolResult { is_error, .. } => assert!(is_error),
            _ => panic!("expected ToolResult"),
        }

        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "tool".into(),
            content: serde_json::Value::String("ok".into()),
            reasoning_content: None,
            tool_calls: vec![],
            name: None,
            tool_call_id: Some("call_2".into()),
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::ToolResult { is_error, .. } => assert!(!is_error),
            _ => panic!("expected ToolResult"),
        }

        let json = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "tool",
                    "content": "boom",
                    "tool_call_id": "call_3",
                    "status": true,
                },
            ],
        });
        let req: ChatCompletionRequest = serde_json::from_value(json).unwrap();
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::ToolResult { is_error, .. } => assert!(is_error),
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn assistant_with_reasoning_tool_calls_and_text_preserves_order() {
        let mut req = make_openai_request();
        req.messages.push(ChatMessage {
            role: "assistant".into(),
            content: serde_json::Value::String("after thinking".into()),
            reasoning_content: Some("let me think...".into()),
            tool_calls: vec![ToolCall {
                index: Some(0),
                id: Some("call_1".into()),
                r#type: Some("function".into()),
                function: Some(FunctionCall {
                    name: Some("get_weather".into()),
                    arguments: Some("{\"city\":\"SF\"}".into()),
                }),
            }],
            name: None,
            tool_call_id: None,
            cache_control: None,
            refusal: None,
            is_error: None,
        });
        let core = decode_request(req).unwrap();
        let content = &core.messages[1].content;
        assert_eq!(content.len(), 3, "expected Thinking, ToolUse, Text");
        assert!(
            matches!(content[0], CoreContent::Thinking { .. }),
            "first block must be Thinking"
        );
        assert!(
            matches!(content[1], CoreContent::ToolUse { .. }),
            "second block must be ToolUse"
        );
        assert!(
            matches!(content[2], CoreContent::Text { .. }),
            "third block must be Text"
        );
    }

    #[test]
    fn stop_as_string_decodes_to_single_element_vec() {
        let mut req = make_openai_request();
        req.stop = Some(serde_json::json!("END"));
        let core = decode_request(req).unwrap();
        assert_eq!(core.sampling.stop, Some(vec!["END".to_owned()]));
    }

    #[test]
    fn stop_as_null_decodes_to_none() {
        let mut req = make_openai_request();
        req.stop = Some(serde_json::Value::Null);
        let core = decode_request(req).unwrap();
        assert_eq!(core.sampling.stop, None);
    }

    #[test]
    fn stop_as_empty_array_decodes_to_none() {
        let mut req = make_openai_request();
        req.stop = Some(serde_json::json!([]));
        let core = decode_request(req).unwrap();
        assert_eq!(core.sampling.stop, None);
    }

    #[test]
    fn stop_as_non_string_non_array_is_rejected() {
        let mut req = make_openai_request();
        req.stop = Some(serde_json::json!(42));
        let err = decode_request(req).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("stop must be a string, array of strings, or null"),
            "expected stop-type rejection, got: {msg}"
        );
        assert!(
            msg.contains("number"),
            "expected type name in message: {msg}"
        );
    }

    #[test]
    fn stop_array_with_non_string_items_is_rejected() {
        let mut req = make_openai_request();
        req.stop = Some(serde_json::json!(["STOP", 123, "END"]));
        let err = decode_request(req).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("stop array must contain only strings"),
            "expected stop-array rejection, got: {msg}"
        );
        assert!(
            msg.contains("number"),
            "expected type name in message: {msg}"
        );
    }
}
