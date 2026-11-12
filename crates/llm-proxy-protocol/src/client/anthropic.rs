//! Anthropic Messages client protocol adapter: translates between the
//! Anthropic wire format and the normalised core protocol types.

use crate::anthropic::{
    self, ApiError, CacheControl as AnthropicCacheControl, ContentBlock, Delta, Message,
    MessageEvent, MessageRequest, MessageResponse, SystemContentBlock,
};
use crate::client::ProtocolError;
use crate::core::{
    CacheControl, CacheControlType, ContentKind, CoreContent, CoreEvent, CoreMessage, CoreRequest,
    CoreResponse, CoreRole, CoreStreamError, CoreStreamErrorKind, CoreTool, CoreToolChoice,
    ModelRef, ProviderHints, RequestMetadata, SamplingOptions, StopReason, Usage,
};

/// Decode an Anthropic [`MessageRequest`] into a normalised [`CoreRequest`].
pub fn decode_request(req: MessageRequest) -> Result<CoreRequest, ProtocolError> {
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

    let system = decode_system(req.system)?;

    let messages = req
        .messages
        .iter()
        .map(decode_message)
        .collect::<Result<Vec<_>, _>>()?;

    let tools = req
        .tools
        .into_iter()
        .map(|t| CoreTool {
            name: t.name,
            description: t.description,
            input_schema: t
                .input_schema
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
        })
        .collect();

    let tool_choice = req.tool_choice.map(decode_tool_choice);

    let sampling = SamplingOptions {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: Some(req.max_tokens),
        stop: req.stop_sequences.filter(|s| !s.is_empty()),
        top_k: req.top_k,
        reasoning_effort: None,
        thinking: req.thinking,
    };

    let stream = req.stream.unwrap_or(false);

    let (user_id, raw_meta) = if let Some(meta) = req.metadata {
        (meta.user_id, meta.extra)
    } else {
        (None, serde_json::Map::new())
    };

    let metadata = RequestMetadata {
        user_id,
        raw: raw_meta,
    };

    let provider_hints = ProviderHints::default();

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

fn decode_cache_control(cache_control: AnthropicCacheControl) -> CacheControl {
    CacheControl {
        r#type: CacheControlType::from(cache_control.r#type),
        ttl: cache_control.ttl,
    }
}

fn malformed_block(context: &str, error: &serde_json::Error) -> ProtocolError {
    let detail = error.to_string();
    let detail = crate::util::truncate_str_safe(&detail, 120);
    ProtocolError::InvalidRequest(format!("malformed {context}: {detail}"))
}

fn decode_system(system: Option<serde_json::Value>) -> Result<Vec<CoreContent>, ProtocolError> {
    match system {
        None => Ok(Vec::new()),
        Some(serde_json::Value::String(s)) => {
            if s.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![CoreContent::Text {
                text: s,
                cache: None,
            }])
        }
        Some(serde_json::Value::Array(arr)) => {
            let mut result = Vec::with_capacity(arr.len());
            for item in arr {
                let block = serde_json::from_value::<SystemContentBlock>(item)
                    .map_err(|error| malformed_block("system block", &error))?;
                if block.r#type != "text" {
                    let block_type = crate::util::truncate_str_safe(&block.r#type, 64);
                    return Err(ProtocolError::InvalidRequest(format!(
                        "system blocks must have type \"text\", got \"{block_type}\""
                    )));
                }
                let Some(text) = block.text else {
                    return Err(ProtocolError::InvalidRequest(
                        "system block of type \"text\" is missing \"text\"".into(),
                    ));
                };
                let cache = block.cache_control.map(decode_cache_control);
                result.push(CoreContent::Text { text, cache });
            }
            Ok(result)
        }
        Some(_) => Err(ProtocolError::InvalidRequest(
            "system must be a string or an array of text blocks".into(),
        )),
    }
}

fn decode_message(msg: &Message) -> Result<CoreMessage, ProtocolError> {
    let role = match msg.role.as_str() {
        "user" => CoreRole::User,
        "assistant" => CoreRole::Assistant,
        other => {
            return Err(ProtocolError::Decode(format!("unknown role: {other}")));
        }
    };

    let content = match msg.content.as_array() {
        Some(items) => items
            .iter()
            .map(|item| decode_wire_block(item, "message content"))
            .collect::<Result<Vec<_>, _>>()?,
        // A string (one text block), null (no blocks) or an invalid shape.
        None => msg
            .content_blocks()
            .map_err(|error| malformed_block("message content", &error))?
            .into_iter()
            .map(decode_content_block)
            .collect::<Result<Vec<_>, _>>()?,
    };

    Ok(CoreMessage { role, content })
}

/// Block types the decoder models in the core. Every other type is carried
/// verbatim as [`CoreContent::Opaque`] so same-format passthrough can forward
/// it: Anthropic requires replayed server-tool blocks back unmodified.
const MODELED_BLOCK_TYPES: &[&str] = &[
    "text",
    "image",
    "document",
    "tool_use",
    "tool_result",
    "thinking",
    "redacted_thinking",
];

/// Decodes one wire block. Unmodeled types bypass the typed [`ContentBlock`]
/// decode, so fields the proxy has no model for can never fail the request.
fn decode_wire_block(
    item: &serde_json::Value,
    context: &str,
) -> Result<CoreContent, ProtocolError> {
    let unmodeled = item
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|block_type| !MODELED_BLOCK_TYPES.contains(&block_type));
    if unmodeled {
        return Ok(CoreContent::Opaque {
            block: item.clone(),
        });
    }
    let block = serde_json::from_value::<ContentBlock>(item.clone())
        .map_err(|error| malformed_block(context, &error))?;
    decode_content_block(block)
}

fn decode_tool_result_blocks(arr: &[serde_json::Value]) -> Result<Vec<CoreContent>, ProtocolError> {
    let mut blocks = Vec::with_capacity(arr.len());
    for item in arr {
        match decode_wire_block(item, "tool_result content") {
            Ok(core) => blocks.push(core),
            Err(ProtocolError::Decode(msg)) => {
                let detail = crate::util::truncate_str_safe(&msg, 120);
                return Err(ProtocolError::InvalidRequest(format!(
                    "malformed tool_result content: {detail}"
                )));
            }
            Err(other) => return Err(other),
        }
    }
    if arr.is_empty() {
        blocks.push(CoreContent::Text {
            text: String::new(),
            cache: None,
        });
    }
    Ok(blocks)
}

fn decode_tool_result_block(block: ContentBlock) -> Result<CoreContent, ProtocolError> {
    let is_error = block.is_error.unwrap_or(false);
    let inner_content = if let Some(ref content_val) = block.content {
        match content_val {
            serde_json::Value::String(s) => {
                vec![CoreContent::Text {
                    text: s.clone(),
                    cache: None,
                }]
            }
            serde_json::Value::Array(arr) => decode_tool_result_blocks(arr)?,
            _ => {
                let text = block.text_content();
                if text.is_empty() {
                    vec![]
                } else {
                    vec![CoreContent::Text { text, cache: None }]
                }
            }
        }
    } else {
        let text = block.text_content();
        if text.is_empty() {
            vec![]
        } else {
            vec![CoreContent::Text { text, cache: None }]
        }
    };
    let tool_use_id = block.tool_use_id.unwrap_or_default();
    if tool_use_id.is_empty() {
        return Err(ProtocolError::Decode(
            "tool_result block missing required 'tool_use_id' field".into(),
        ));
    }
    let cache = block.cache_control.map(decode_cache_control);
    Ok(CoreContent::ToolResult {
        tool_use_id,
        content: inner_content,
        is_error,
        cache,
    })
}

fn decode_content_block(block: ContentBlock) -> Result<CoreContent, ProtocolError> {
    match block.r#type.as_str() {
        "text" => {
            let cache = block.cache_control.map(decode_cache_control);
            Ok(CoreContent::Text {
                text: block.text.unwrap_or_default(),
                cache,
            })
        }
        "image" => {
            if block.cache_control.is_some() {
                tracing::warn!(
                    "Anthropic client: cache_control on image block is not supported; dropping"
                );
            }
            let source = block.source.unwrap_or(serde_json::Value::Null);
            Ok(CoreContent::Image { source })
        }
        "document" => Ok(CoreContent::Document {
            source: block.source.unwrap_or(serde_json::Value::Null),
        }),
        "tool_use" => {
            let id = block.id.unwrap_or_default();
            if id.is_empty() {
                return Err(ProtocolError::Decode(
                    "tool_use block missing required 'id' field".into(),
                ));
            }
            let name = block.name.unwrap_or_default();
            if name.is_empty() {
                return Err(ProtocolError::Decode(
                    "tool_use block missing required 'name' field".into(),
                ));
            }
            Ok(CoreContent::ToolUse {
                id,
                name,
                input: block
                    .input
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new())),
                signature: block.signature,
            })
        }
        "tool_result" => decode_tool_result_block(block),
        "thinking" => Ok(CoreContent::Thinking {
            text: block.thinking.unwrap_or_default(),
            signature: block.signature,
        }),
        "redacted_thinking" => Ok(CoreContent::RedactedThinking {
            data: serde_json::Value::String(block.data.unwrap_or_default()),
        }),
        other => {
            let truncated = crate::util::truncate_str_safe(other, 64);
            tracing::warn!(
                block_type = truncated,
                "unknown Anthropic content block type during decode"
            );
            Err(ProtocolError::Decode(format!(
                "unknown Anthropic content block type: {other}"
            )))
        }
    }
}

fn decode_tool_choice(value: serde_json::Value) -> CoreToolChoice {
    if let Some(obj) = value.as_object() {
        match obj.get("type").and_then(|v| v.as_str()) {
            Some("auto") => CoreToolChoice::Auto,
            Some("any") => CoreToolChoice::Any,
            Some("none") => CoreToolChoice::None,
            Some("tool") => {
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                if name.is_empty() {
                    tracing::warn!(
                        "Anthropic tool_choice type 'tool' has empty or missing name field"
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

/// Encode a [`CoreResponse`] into an Anthropic [`MessageResponse`]. Blocks
/// the format cannot represent are dropped with a warning
/// (`EncodeSkippable`); a `CoreContent::Refusal` is a hard `Encode` error.
pub fn encode_response(resp: CoreResponse) -> Result<MessageResponse, ProtocolError> {
    if !resp.provider_meta.is_empty() {
        tracing::warn!(
            meta_keys = resp.provider_meta.len(),
            "provider_meta is non-empty during Anthropic client encode; \
             this data is for provider adapters only and will not be forwarded to the client"
        );
    }

    let id = resp
        .id
        .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4()));

    let mut content = Vec::new();
    for block in resp.content {
        match encode_content_block(block) {
            Ok(encoded) => content.push(encoded),
            Err(ProtocolError::EncodeSkippable(_msg)) => {}
            Err(other) => return Err(other),
        }
    }
    if content.is_empty() {
        content.push(ContentBlock::new_text(String::new()));
    }

    let stop_reason = encode_stop_reason(resp.stop_reason);

    let usage = encode_usage(&resp.usage);

    Ok(MessageResponse {
        id,
        r#type: "message".to_owned(),
        role: "assistant".to_owned(),
        content,
        model: resp.model.requested,
        stop_reason: Some(stop_reason),
        stop_sequence: resp.stop_sequence,
        usage,
    })
}

fn encode_content_block(content: CoreContent) -> Result<ContentBlock, ProtocolError> {
    match content {
        CoreContent::Text { text, cache } => {
            let cc = cache.map(|c| AnthropicCacheControl {
                r#type: c.r#type.as_str().to_owned(),
                ttl: c.ttl,
            });
            let mut block = ContentBlock::new_text(text);
            block.cache_control = cc;
            Ok(block)
        }
        CoreContent::Image { source } => Ok(ContentBlock::new_image(source)),
        CoreContent::ToolUse {
            id,
            name,
            input,
            signature,
        } => {
            let mut block = ContentBlock::new_tool_use(id, name, input);
            // Tool-call signatures (e.g. Gemini thought signatures routed
            // through an Anthropic-dialect client) ride the `signature` field
            // so clients that replay blocks verbatim return them to us.
            block.signature = signature;
            Ok(block)
        }
        CoreContent::ToolResult {
            tool_use_id,
            content,
            is_error,
            cache,
        } => {
            let content_val = if content.len() == 1 {
                if let CoreContent::Text { ref text, .. } = content[0] {
                    Some(serde_json::Value::String(text.clone()))
                } else {
                    Some(serde_json::to_value(&content).unwrap_or(serde_json::Value::Null))
                }
            } else if content.is_empty() {
                None
            } else {
                Some(serde_json::to_value(&content).unwrap_or(serde_json::Value::Null))
            };
            let mut block = ContentBlock::new_tool_result(tool_use_id, content_val, Some(is_error));
            block.cache_control = cache.map(|c| AnthropicCacheControl {
                r#type: c.r#type.as_str().to_owned(),
                ttl: c.ttl,
            });
            Ok(block)
        }
        CoreContent::Thinking { text, signature } => {
            let mut block = ContentBlock::new_thinking(text);
            block.signature = signature;
            Ok(block)
        }
        CoreContent::Document { .. } => {
            tracing::warn!(
                "Anthropic Messages protocol does not natively support document blocks in responses; omitting"
            );
            Err(ProtocolError::EncodeSkippable(
                "Anthropic Messages does not support Document blocks in responses".into(),
            ))
        }
        CoreContent::Audio { .. } => {
            tracing::warn!(
                "Anthropic Messages protocol does not natively support audio blocks in responses; omitting"
            );
            Err(ProtocolError::EncodeSkippable(
                "Anthropic Messages does not support Audio blocks in responses".into(),
            ))
        }
        CoreContent::Video { .. } => {
            tracing::warn!(
                "Anthropic Messages protocol does not natively support video blocks in responses; omitting"
            );
            Err(ProtocolError::EncodeSkippable(
                "Anthropic Messages does not support Video blocks in responses".into(),
            ))
        }
        CoreContent::RedactedThinking { data } => {
            let data_str = match data {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            Ok(ContentBlock::new_redacted_thinking(data_str))
        }
        CoreContent::Refusal { text } => {
            tracing::warn!(
                refusal_text_len = text.len(),
                "Anthropic Messages protocol has no refusal field; cannot encode"
            );
            Err(ProtocolError::Encode(
                "Anthropic Messages does not support Refusal blocks".into(),
            ))
        }
        CoreContent::Opaque { .. } => {
            tracing::warn!("opaque client block has no Anthropic response encoding; omitting");
            Err(ProtocolError::EncodeSkippable(
                "opaque client blocks have no Anthropic response encoding".into(),
            ))
        }
    }
}

fn encode_stop_reason(reason: StopReason) -> String {
    match reason {
        StopReason::EndTurn => "end_turn".to_owned(),
        StopReason::MaxTokens => "max_tokens".to_owned(),
        StopReason::ToolUse => "tool_use".to_owned(),
        StopReason::StopSequence => "stop_sequence".to_owned(),
        StopReason::Refusal => {
            tracing::warn!(
                "StopReason::Refusal has no native Anthropic stop_reason; mapping to end_turn"
            );
            "end_turn".to_owned()
        }
        StopReason::Error => {
            tracing::warn!(
                "StopReason::Error has no native Anthropic stop_reason; mapping to end_turn"
            );
            "end_turn".to_owned()
        }
        StopReason::Unknown(original) => {
            tracing::warn!(
                original_stop_reason = %original,
                "StopReason::Unknown mapped to 'end_turn' in Anthropic stop_reason"
            );
            "end_turn".to_owned()
        }
    }
}

fn encode_usage(usage: &Usage) -> anthropic::Usage {
    if let Some(rt) = usage.reasoning_tokens {
        if rt != 0 {
            tracing::warn!(
                reasoning_tokens = rt,
                "Anthropic Usage does not support reasoning_tokens; dropping"
            );
        }
    }
    anthropic::Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_creation_input_tokens: usage.cache_creation_input_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
        cache_creation: None,
    }
}

fn encode_content_start_block(index: usize, kind: ContentKind) -> Option<ContentBlock> {
    match kind {
        ContentKind::Text => Some(ContentBlock::new_text(String::new())),
        ContentKind::Thinking => Some(ContentBlock::new_thinking(String::new())),
        ContentKind::ToolUse => {
            tracing::trace!(
                index,
                "ContentStart ToolUse skipped; ToolCallStart will emit content_block_start"
            );
            None
        }
        other => {
            tracing::warn!(
                kind = ?other,
                "unsupported ContentKind in Anthropic stream encode; skipping content_block_start"
            );
            None
        }
    }
}

fn encode_error_event(error: &CoreStreamError) -> MessageEvent {
    let msg = error.message();
    let capped_msg = if msg.len() > 1024 {
        tracing::warn!(
            original_len = msg.len(),
            "stream error message exceeds 1024 chars; truncating for client-facing SSE"
        );
        crate::util::truncate_str_safe(msg, 1024).to_owned()
    } else {
        msg.to_owned()
    };
    MessageEvent {
        r#type: "error".to_owned(),
        message: None,
        index: None,
        content_block: None,
        delta: None,
        usage: None,
        error: Some(ApiError {
            r#type: encode_error_kind(&error.kind),
            message: capped_msg,
        }),
    }
}

/// Stateful encoder that converts [`CoreEvent`] values into Anthropic SSE
/// [`MessageEvent`] values, buffering the latest `UsageDelta` and coalescing
/// it into the terminal `message_delta` + `message_stop` pair. The terminal
/// pair is held back until [`StreamEncoder::finish`] because providers send
/// usage after the finish event; flushing on `MessageStop` would report the
/// buffered (zero) usage instead of the real numbers.
#[derive(Debug)]
pub struct StreamEncoder {
    msg_id: String,
    model: String,
    pending_usage: Option<anthropic::Usage>,
    pending_stop: Option<PendingStop>,
    started: bool,
    finished: bool,
    open_text_blocks: Vec<usize>,
}

/// A `MessageStop` waiting for the trailing usage event before it is flushed.
#[derive(Debug)]
struct PendingStop {
    stop_reason: StopReason,
    stop_sequence: Option<String>,
}

impl StreamEncoder {
    /// Create a new stream encoder; `msg_id` and `model` are typically taken
    /// from the first `CoreEvent::MessageStart`.
    pub fn new(msg_id: String, model: String) -> Self {
        Self {
            msg_id,
            model,
            pending_usage: None,
            pending_stop: None,
            started: false,
            finished: false,
            open_text_blocks: Vec::new(),
        }
    }

    fn encode_message_start(&mut self, id: Option<String>, model: String) -> MessageEvent {
        if let Some(id) = id {
            self.msg_id = id;
        }
        self.model = model;
        self.started = true;

        MessageEvent {
            r#type: "message_start".to_owned(),
            message: Some(MessageResponse {
                id: self.msg_id.clone(),
                r#type: "message".to_owned(),
                role: "assistant".to_owned(),
                content: vec![],
                model: self.model.clone(),
                stop_reason: None,
                stop_sequence: None,
                usage: anthropic::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                    cache_creation: None,
                },
            }),
            index: None,
            content_block: None,
            delta: None,
            usage: None,
            error: None,
        }
    }

    fn encode_message_stop(
        &mut self,
        stop_reason: StopReason,
        stop_sequence: Option<String>,
        events: &mut Vec<MessageEvent>,
    ) {
        let anth_stop = encode_stop_reason(stop_reason);
        let usage = self.pending_usage.take().unwrap_or(anthropic::Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_creation: None,
        });

        self.open_text_blocks.sort_unstable();
        for idx in self.open_text_blocks.drain(..) {
            events.push(MessageEvent {
                r#type: "content_block_stop".to_owned(),
                message: None,
                index: Some(idx),
                content_block: None,
                delta: None,
                usage: None,
                error: None,
            });
        }

        events.push(MessageEvent {
            r#type: "message_delta".to_owned(),
            message: None,
            index: None,
            content_block: None,
            delta: Some(Delta {
                r#type: None,
                text: None,
                thinking: None,
                partial_json: None,
                stop_reason: Some(anth_stop),
                stop_sequence,
            }),
            usage: Some(usage),
            error: None,
        });

        events.push(MessageEvent {
            r#type: "message_stop".to_owned(),
            message: None,
            index: None,
            content_block: None,
            delta: None,
            usage: None,
            error: None,
        });

        self.finished = true;
    }

    /// Encode a single [`CoreEvent`] into zero or more Anthropic [`MessageEvent`]s.
    #[expect(
        clippy::too_many_lines,
        reason = "one match arm per CoreEvent variant: the Anthropic Messages SSE \
                 wire shape is a single translation table, and splitting the arms \
                 into helpers would separate each event's field mapping from the \
                 frame layout it produces"
    )]
    pub fn encode_event(&mut self, event: CoreEvent) -> Result<Vec<MessageEvent>, ProtocolError> {
        if self.finished {
            tracing::trace!(
                event = ?event,
                "StreamEncoder: dropping event after stream finished"
            );
            return Ok(Vec::new());
        }

        if self.pending_stop.is_some() {
            if let CoreEvent::Error { error } = event {
                // An error after the buffered terminal must neither swallow
                // the terminal nor be swallowed by it: flush the buffered
                // pair ahead of the error event, then stop encoding. This is
                // the wire shape clients saw before the terminal was buffered.
                let mut events = self.finish()?;
                events.push(encode_error_event(&error));
                return Ok(events);
            }
            // The terminal pair is buffered; only a trailing usage update is
            // still meaningful (it lands in the terminal message_delta).
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

        let mut events = Vec::new();

        match event {
            CoreEvent::MessageStart { id, model } => {
                events.push(self.encode_message_start(id, model.requested));
            }

            CoreEvent::ContentStart { index, kind } => {
                let Some(block) = encode_content_start_block(index, kind) else {
                    return Ok(events);
                };
                events.push(MessageEvent {
                    r#type: "content_block_start".to_owned(),
                    message: None,
                    index: Some(index),
                    content_block: Some(block),
                    delta: None,
                    usage: None,
                    error: None,
                });
                self.open_text_blocks.push(index);
            }

            CoreEvent::TextDelta { index, text } => {
                events.push(MessageEvent {
                    r#type: "content_block_delta".to_owned(),
                    message: None,
                    index: Some(index),
                    content_block: None,
                    delta: Some(Delta {
                        r#type: Some("text_delta".to_owned()),
                        text: Some(text),
                        thinking: None,
                        partial_json: None,
                        stop_reason: None,
                        stop_sequence: None,
                    }),
                    usage: None,
                    error: None,
                });
            }

            CoreEvent::ThinkingDelta { index, text } => {
                events.push(MessageEvent {
                    r#type: "content_block_delta".to_owned(),
                    message: None,
                    index: Some(index),
                    content_block: None,
                    delta: Some(Delta {
                        r#type: Some("thinking_delta".to_owned()),
                        text: None,
                        thinking: Some(text),
                        partial_json: None,
                        stop_reason: None,
                        stop_sequence: None,
                    }),
                    usage: None,
                    error: None,
                });
            }

            CoreEvent::ToolCallStart {
                index,
                id,
                name,
                signature,
            } => {
                let mut block = ContentBlock::new_tool_use(
                    id,
                    name,
                    serde_json::Value::Object(serde_json::Map::new()),
                );
                // Tool-call signatures (e.g. Gemini thought signatures routed
                // through an Anthropic-dialect client) ride the `signature`
                // field so clients that replay blocks verbatim return them.
                block.signature = signature;
                events.push(MessageEvent {
                    r#type: "content_block_start".to_owned(),
                    message: None,
                    index: Some(index),
                    content_block: Some(block),
                    delta: None,
                    usage: None,
                    error: None,
                });
            }

            CoreEvent::ToolCallDelta { index, args_delta } => {
                events.push(MessageEvent {
                    r#type: "content_block_delta".to_owned(),
                    message: None,
                    index: Some(index),
                    content_block: None,
                    delta: Some(Delta {
                        r#type: Some("input_json_delta".to_owned()),
                        text: None,
                        thinking: None,
                        partial_json: Some(args_delta),
                        stop_reason: None,
                        stop_sequence: None,
                    }),
                    usage: None,
                    error: None,
                });
            }

            CoreEvent::ToolCallStop { index } => {
                events.push(MessageEvent {
                    r#type: "content_block_stop".to_owned(),
                    message: None,
                    index: Some(index),
                    content_block: None,
                    delta: None,
                    usage: None,
                    error: None,
                });
            }

            CoreEvent::UsageDelta { usage } => {
                self.pending_usage = Some(encode_usage(&usage));
            }

            CoreEvent::MessageStop {
                stop_reason,
                stop_sequence,
            } => {
                // Hold the terminal pair back: usage routinely arrives after
                // the finish event and belongs in the terminal message_delta.
                // Nothing is lost — `finish()` flushes it with the freshest
                // buffered usage.
                self.pending_stop = Some(PendingStop {
                    stop_reason,
                    stop_sequence,
                });
            }

            CoreEvent::Error { error } => {
                events.push(encode_error_event(&error));
            }

            CoreEvent::Ping => {
                events.push(MessageEvent {
                    r#type: "ping".to_owned(),
                    message: None,
                    index: None,
                    content_block: None,
                    delta: None,
                    usage: None,
                    error: None,
                });
            }
        }

        Ok(events)
    }

    /// Mark the encoder as finished without emitting synthetic terminal
    /// events; later `finish()` calls return an empty vec. A buffered
    /// `MessageStop` is dropped: callers are the error paths, which have
    /// already delivered the buffered terminal ahead of the error event (an
    /// `Error` event flushes it) or a terminal error event on its own.
    pub fn mark_finished(&mut self) {
        self.finished = true;
        self.pending_stop = None;
    }

    /// Flush remaining buffered events: a buffered `MessageStop` is emitted
    /// with the freshest usage; a stream that ended without one emits a
    /// synthetic `message_delta` (stop reason `"end_turn"`, with a warning) +
    /// `message_stop` pair.
    pub fn finish(&mut self) -> Result<Vec<MessageEvent>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;

        if let Some(pending) = self.pending_stop.take() {
            let mut events = Vec::new();
            self.encode_message_stop(pending.stop_reason, pending.stop_sequence, &mut events);
            return Ok(events);
        }

        tracing::warn!(
            "StreamEncoder::finish() called without prior MessageStop; emitting synthetic terminal events"
        );

        let usage = self.pending_usage.take().unwrap_or(anthropic::Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            cache_creation: None,
        });

        let events = vec![
            MessageEvent {
                r#type: "message_delta".to_owned(),
                message: None,
                index: None,
                content_block: None,
                delta: Some(Delta {
                    r#type: None,
                    text: None,
                    thinking: None,
                    partial_json: None,
                    stop_reason: Some("end_turn".to_owned()),
                    stop_sequence: None,
                }),
                usage: Some(usage),
                error: None,
            },
            MessageEvent {
                r#type: "message_stop".to_owned(),
                message: None,
                index: None,
                content_block: None,
                delta: None,
                usage: None,
                error: None,
            },
        ];

        Ok(events)
    }
}

fn encode_error_kind(kind: &CoreStreamErrorKind) -> String {
    match kind {
        CoreStreamErrorKind::InvalidRequest => "invalid_request_error".to_owned(),
        CoreStreamErrorKind::Authentication => "authentication_error".to_owned(),
        CoreStreamErrorKind::Permission => "permission_error".to_owned(),
        CoreStreamErrorKind::RateLimit => "rate_limit_error".to_owned(),
        CoreStreamErrorKind::Upstream => "api_error".to_owned(),
        CoreStreamErrorKind::Internal => "api_error".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::Tool;
    use crate::core::UsageProvenance;

    fn make_anthropic_request() -> MessageRequest {
        MessageRequest {
            model: "claude-sonnet-4-20250514".into(),
            max_tokens: 1024,
            system: None,
            messages: vec![Message {
                role: "user".into(),
                content: serde_json::Value::String("hi".into()),
            }],
            stream: None,
            tools: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None,
            thinking: None,
            tool_choice: None,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn plain_text_request_decodes_to_core() {
        let req = make_anthropic_request();
        let core = decode_request(req).unwrap();
        assert_eq!(core.model.requested, "claude-sonnet-4-20250514");
        assert_eq!(core.messages.len(), 1);
        assert_eq!(core.messages[0].role, CoreRole::User);
        assert_eq!(core.messages[0].content.len(), 1);
        match &core.messages[0].content[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "hi"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn system_prompt_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.system = Some(serde_json::Value::String("You are helpful".into()));
        let core = decode_request(req).unwrap();
        assert_eq!(core.system.len(), 1);
        match &core.system[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "You are helpful"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn system_prompt_array_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.system = Some(serde_json::json!([
            { "type": "text", "text": "You are helpful." },
            { "type": "text", "text": "Be concise." }
        ]));
        let core = decode_request(req).unwrap();
        assert_eq!(core.system.len(), 2);
    }

    #[test]
    fn tool_call_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "assistant".into(),
            content: serde_json::json!([
                { "type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "SF"} }
            ]),
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.messages.len(), 2);
        match &core.messages[1].content[0] {
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn tool_result_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "tool_result", "tool_use_id": "tu_1", "content": "72F and sunny" }
            ]),
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.messages.len(), 2);
        match &core.messages[1].content[0] {
            CoreContent::ToolResult {
                tool_use_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "tu_1");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn thinking_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "assistant".into(),
            content: serde_json::json!([
                { "type": "thinking", "thinking": "hmm...", "signature": "sig_abc" }
            ]),
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::Thinking { text, signature } => {
                assert_eq!(text, "hmm...");
                assert_eq!(signature.as_deref(), Some("sig_abc"));
            }
            _ => panic!("expected Thinking"),
        }
    }

    #[test]
    fn tool_result_with_is_error_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "tool_result", "tool_use_id": "tu_1", "is_error": true, "content": "something went wrong" }
            ]),
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::ToolResult {
                tool_use_id,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "tu_1");
                assert!(is_error);
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn image_block_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "iVBOR..." } }
            ]),
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::Image { source } => {
                assert!(source.is_object());
                assert_eq!(source["type"], "base64");
            }
            _ => panic!("expected Image"),
        }
    }

    #[test]
    fn every_documented_image_source_decodes_with_the_source_intact() {
        let sources = [
            serde_json::json!({ "type": "base64", "media_type": "image/png", "data": "iVBOR..." }),
            serde_json::json!({ "type": "url", "url": "https://example.com/cat.jpg" }),
            serde_json::json!({ "type": "file", "file_id": "file_fixture_01" }),
        ];
        for source in sources {
            let mut req = make_anthropic_request();
            req.messages.push(Message {
                role: "user".into(),
                content: serde_json::json!([{ "type": "image", "source": source }]),
            });
            let core = decode_request(req)
                .unwrap_or_else(|e| panic!("image source {source} must decode: {e}"));
            match &core.messages[1].content[..] {
                [CoreContent::Image { source: decoded }] => assert_eq!(decoded, &source),
                other => panic!("expected one Image for {source}, got {other:?}"),
            }
        }
    }

    #[test]
    fn document_blocks_decode_to_core_documents_in_order() {
        let url_source =
            serde_json::json!({ "type": "url", "url": "https://example.com/report.pdf" });
        let base64_source = serde_json::json!({
            "type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjcK"
        });
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "document", "source": url_source },
                {
                    "type": "document",
                    "source": base64_source,
                    "cache_control": { "type": "ephemeral" }
                },
                { "type": "text", "text": "Summarize both." }
            ]),
        });
        let core = decode_request(req).expect("document blocks must decode");
        match &core.messages[1].content[..] {
            [
                CoreContent::Document { source: first },
                CoreContent::Document { source: second },
                CoreContent::Text { text, .. },
            ] => {
                assert_eq!(first, &url_source);
                assert_eq!(second, &base64_source);
                assert_eq!(text, "Summarize both.");
            }
            other => panic!("expected two Documents then Text, got {other:?}"),
        }
    }

    #[test]
    fn tool_choice_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.tool_choice = Some(serde_json::json!({"type": "auto"}));
        let core = decode_request(req).unwrap();
        assert_eq!(core.tool_choice, Some(CoreToolChoice::Auto));

        let mut req2 = make_anthropic_request();
        req2.tool_choice = Some(serde_json::json!({"type": "tool", "name": "get_weather"}));
        let core2 = decode_request(req2).unwrap();
        match core2.tool_choice {
            Some(CoreToolChoice::Tool { name }) => assert_eq!(name, "get_weather"),
            _ => panic!("expected Tool"),
        }
    }

    #[test]
    fn tool_choice_any_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.tool_choice = Some(serde_json::json!({"type": "any"}));
        let core = decode_request(req).unwrap();
        assert_eq!(core.tool_choice, Some(CoreToolChoice::Any));
    }

    #[test]
    fn tool_choice_none_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.tool_choice = Some(serde_json::json!({"type": "none"}));
        let core = decode_request(req).unwrap();
        assert_eq!(core.tool_choice, Some(CoreToolChoice::None));
    }

    #[test]
    fn cache_control_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.system = Some(serde_json::json!([
            { "type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"} }
        ]));
        let core = decode_request(req).unwrap();
        match &core.system[0] {
            CoreContent::Text { text, cache } => {
                assert_eq!(text, "You are helpful.");
                assert!(cache.is_some());
                assert_eq!(cache.as_ref().unwrap().r#type, CacheControlType::Ephemeral);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn system_block_with_cache_ttl_keeps_its_text_and_ttl() {
        let mut req = make_anthropic_request();
        req.system = Some(serde_json::json!([
            { "type": "text", "text": "Long prompt", "cache_control": {"type": "ephemeral", "ttl": "1h"} }
        ]));
        let core = decode_request(req).unwrap();
        match core.system.as_slice() {
            [CoreContent::Text { text, cache }] => {
                assert_eq!(text, "Long prompt");
                assert_eq!(cache.as_ref().unwrap().ttl.as_deref(), Some("1h"));
            }
            other => panic!("expected one text block, got {other:?}"),
        }
    }

    #[test]
    fn message_block_with_cache_ttl_keeps_its_text() {
        let mut req = make_anthropic_request();
        req.messages = vec![Message {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "text", "text": "Long question", "cache_control": {"type": "ephemeral", "ttl": "1h"} }
            ]),
        }];
        let core = decode_request(req).unwrap();
        match core.messages[0].content.as_slice() {
            [CoreContent::Text { text, cache }] => {
                assert_eq!(text, "Long question");
                assert_eq!(cache.as_ref().unwrap().ttl.as_deref(), Some("1h"));
            }
            other => panic!("expected one text block, got {other:?}"),
        }
    }

    #[test]
    fn system_content_the_proxy_cannot_read_is_rejected_not_dropped() {
        let unreadable = [
            serde_json::json!([{ "type": "text", "text": "ok" }, { "type": "text", "text": 42 }]),
            serde_json::json!([{ "type": "image" }]),
            serde_json::json!([{ "type": "text" }]),
            serde_json::json!(42),
        ];
        for system in unreadable {
            let mut req = make_anthropic_request();
            req.system = Some(system.clone());
            assert!(
                matches!(decode_request(req), Err(ProtocolError::InvalidRequest(_))),
                "expected a rejection for system = {system}"
            );
        }
    }

    #[test]
    fn system_blocks_with_unknown_fields_decode_instead_of_failing() {
        // Plan 02: the request path tolerates unknown fields so same-format
        // passthrough can forward them; unreadable content is still rejected
        // (previous test).
        let mut req = make_anthropic_request();
        req.system = Some(serde_json::json!([
            { "type": "text", "text": "ok", "not_a_field": true }
        ]));
        let core = decode_request(req).expect("unknown system-block fields must not 400");
        assert_eq!(core.system.len(), 1);
        match &core.system[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "ok"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn stop_sequences_and_top_k_decode_into_sampling() {
        let mut req = make_anthropic_request();
        req.stop_sequences = Some(vec!["END".into(), "STOP".into()]);
        req.top_k = Some(5);
        let core = decode_request(req).unwrap();
        assert_eq!(
            core.sampling.stop.as_deref(),
            Some(["END".to_owned(), "STOP".to_owned()].as_slice())
        );
        assert_eq!(core.sampling.top_k, Some(5));
    }

    #[test]
    fn malformed_message_and_tool_result_blocks_are_rejected_not_dropped() {
        let unreadable = [
            serde_json::json!([{ "type": "text", "text": 42 }]),
            serde_json::json!([{
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [{ "type": "text", "text": 42 }]
            }]),
        ];
        for content in unreadable {
            let mut req = make_anthropic_request();
            req.messages = vec![Message {
                role: "user".into(),
                content: content.clone(),
            }];
            assert!(
                matches!(decode_request(req), Err(ProtocolError::InvalidRequest(_))),
                "expected a rejection for content = {content}"
            );
        }
    }

    #[test]
    fn tool_result_with_undecodable_nested_block_is_rejected() {
        let mut req = make_anthropic_request();
        req.messages = vec![Message {
            role: "user".into(),
            content: serde_json::json!([{
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [
                    { "type": "text", "text": "ok" },
                    { "type": "tool_use", "name": "no_id" }
                ]
            }]),
        }];
        let err = decode_request(req).unwrap_err();
        assert!(
            matches!(&err, ProtocolError::InvalidRequest(m) if m.contains("tool_result content")),
            "expected InvalidRequest naming tool_result content, got {err:?}"
        );

        // An explicitly empty content array stays valid: it must decode to a
        // ToolResult whose content is exactly one empty text block.
        let mut req = make_anthropic_request();
        req.messages = vec![Message {
            role: "user".into(),
            content: serde_json::json!([
                { "type": "tool_result", "tool_use_id": "tu_1", "content": [] }
            ]),
        }];
        let core = decode_request(req).expect("empty tool_result content array is valid");
        match &core.messages[0].content[0] {
            CoreContent::ToolResult { content, .. } => match content.as_slice() {
                [CoreContent::Text { text, cache: None }] => assert_eq!(text, ""),
                other => panic!("expected exactly one empty Text block, got {other:?}"),
            },
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn multiple_messages_decode_to_ordered_core_messages() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "assistant".into(),
            content: serde_json::Value::String("hello".into()),
        });
        req.messages.push(Message {
            role: "user".into(),
            content: serde_json::Value::String("how are you?".into()),
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.messages.len(), 3);
        assert_eq!(core.messages[0].role, CoreRole::User);
        assert_eq!(core.messages[1].role, CoreRole::Assistant);
        assert_eq!(core.messages[2].role, CoreRole::User);
    }

    #[test]
    fn malformed_request_returns_protocol_error() {
        let mut req = make_anthropic_request();
        req.model = String::new();
        let err = decode_request(req).unwrap_err();
        assert!(matches!(err, ProtocolError::InvalidRequest(_)));

        let mut req2 = make_anthropic_request();
        req2.messages = vec![];
        let err2 = decode_request(req2).unwrap_err();
        assert!(matches!(err2, ProtocolError::InvalidRequest(_)));
    }

    #[test]
    fn unknown_role_returns_decode_error() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "system".into(),
            content: serde_json::Value::String("test".into()),
        });
        let err = decode_request(req).unwrap_err();
        assert!(matches!(err, ProtocolError::Decode(_)));
    }

    #[test]
    fn metadata_preserved_in_decode() {
        let mut req = make_anthropic_request();
        req.metadata = Some(anthropic::Metadata {
            user_id: Some("user-42".into()),
            extra: serde_json::Map::new(),
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.metadata.user_id.as_deref(), Some("user-42"));
    }

    #[test]
    fn sampling_options_preserved_in_decode() {
        let mut req = make_anthropic_request();
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.thinking = Some(serde_json::json!({"type": "enabled", "budget_tokens": 5000}));
        let core = decode_request(req).unwrap();
        assert_eq!(core.sampling.temperature, Some(0.7));
        assert_eq!(core.sampling.top_p, Some(0.9));
        assert_eq!(core.sampling.max_tokens, Some(1024));
        assert!(core.sampling.thinking.is_some());
    }

    #[test]
    fn stream_flag_preserved_in_decode() {
        let mut req = make_anthropic_request();
        req.stream = Some(true);
        let core = decode_request(req).unwrap();
        assert!(core.stream);

        let req2 = make_anthropic_request();
        let core2 = decode_request(req2).unwrap();
        assert!(!core2.stream);
    }

    #[test]
    fn core_text_response_encodes_to_route_response() {
        let resp = CoreResponse {
            id: Some("msg_123".into()),
            model: ModelRef {
                requested: "claude-sonnet-4-20250514".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: "hello".into(),
                cache: None,
            }],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::provider_reported(10, 20),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        assert_eq!(out.id, "msg_123");
        assert_eq!(out.model, "claude-sonnet-4-20250514");
        assert_eq!(out.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(out.content.len(), 1);
        assert_eq!(out.content[0].r#type, "text");
        assert_eq!(out.content[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn core_tool_use_response_encodes_to_route_response() {
        let resp = CoreResponse {
            id: Some("msg_123".into()),
            model: ModelRef {
                requested: "claude-sonnet-4-20250514".into(),
                upstream: None,
            },
            content: vec![CoreContent::ToolUse {
                id: "tu_1".into(),
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
        assert_eq!(out.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(out.content.len(), 1);
        assert_eq!(out.content[0].r#type, "tool_use");
    }

    #[test]
    fn stop_reason_mapping() {
        let cases = vec![
            (StopReason::EndTurn, "end_turn"),
            (StopReason::MaxTokens, "max_tokens"),
            (StopReason::ToolUse, "tool_use"),
            (StopReason::StopSequence, "stop_sequence"),
            (StopReason::Refusal, "end_turn"),
            (StopReason::Error, "end_turn"),
            (StopReason::Unknown("test-unknown".to_owned()), "end_turn"),
        ];
        for (reason, expected) in cases {
            assert_eq!(encode_stop_reason(reason), expected);
        }
    }

    #[test]
    fn usage_mapping() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 200,
            reasoning_tokens: None,
            cache_creation_input_tokens: Some(10),
            cache_read_input_tokens: Some(5),
            provenance: UsageProvenance::ProviderReported,
        };
        let out = encode_usage(&usage);
        assert_eq!(out.input_tokens, 100);
        assert_eq!(out.output_tokens, 200);
        assert_eq!(out.cache_creation_input_tokens, Some(10));
        assert_eq!(out.cache_read_input_tokens, Some(5));
    }

    #[test]
    fn stop_sequence_mapping() {
        let resp = CoreResponse {
            id: Some("msg_123".into()),
            model: ModelRef {
                requested: "m".into(),
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
        let out = encode_response(resp).unwrap();
        assert_eq!(out.stop_sequence.as_deref(), Some("\n"));
    }

    #[test]
    fn empty_content_produces_empty_text_block() {
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
        assert_eq!(out.content.len(), 1);
        assert_eq!(out.content[0].r#type, "text");
        assert_eq!(out.content[0].text.as_deref(), Some(""));
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
        assert!(out.id.starts_with("msg_"));
    }

    #[test]
    fn streaming_text_event_mapping() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::TextDelta {
                index: 0,
                text: "hello".into(),
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "content_block_delta");
        assert_eq!(
            events[0].delta.as_ref().unwrap().r#type.as_deref(),
            Some("text_delta")
        );
        assert_eq!(
            events[0].delta.as_ref().unwrap().text.as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn streaming_thinking_event_mapping() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::ThinkingDelta {
                index: 0,
                text: "hmm".into(),
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "content_block_delta");
        assert_eq!(
            events[0].delta.as_ref().unwrap().r#type.as_deref(),
            Some("thinking_delta")
        );
        assert_eq!(
            events[0].delta.as_ref().unwrap().thinking.as_deref(),
            Some("hmm")
        );
    }

    #[test]
    fn streaming_tool_event_mapping() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());

        let start_events = enc
            .encode_event(CoreEvent::ToolCallStart {
                index: 0,
                id: "tu_1".into(),
                name: "get_weather".into(),
                signature: None,
            })
            .unwrap();
        assert_eq!(start_events.len(), 1);
        assert_eq!(start_events[0].r#type, "content_block_start");
        assert_eq!(
            start_events[0].content_block.as_ref().unwrap().r#type,
            "tool_use"
        );

        let delta_events = enc
            .encode_event(CoreEvent::ToolCallDelta {
                index: 0,
                args_delta: "{\"city\":".into(),
            })
            .unwrap();
        assert_eq!(delta_events.len(), 1);
        assert_eq!(delta_events[0].r#type, "content_block_delta");
        assert_eq!(
            delta_events[0].delta.as_ref().unwrap().r#type.as_deref(),
            Some("input_json_delta")
        );

        let stop_events = enc
            .encode_event(CoreEvent::ToolCallStop { index: 0 })
            .unwrap();
        assert_eq!(stop_events.len(), 1);
        assert_eq!(stop_events[0].r#type, "content_block_stop");
    }

    #[test]
    fn stream_tool_call_start_with_signature_emits_signature_on_block() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::ToolCallStart {
                index: 0,
                id: "tu_1".into(),
                name: "get_weather".into(),
                signature: Some("sig-1".into()),
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "content_block_start");
        let json = serde_json::to_value(&events[0]).unwrap();
        assert_eq!(
            json["content_block"]["signature"], "sig-1",
            "the signature must ride the content_block_start tool_use block: {json}"
        );
    }

    #[test]
    fn stream_tool_call_start_without_signature_omits_the_key() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::ToolCallStart {
                index: 0,
                id: "tu_1".into(),
                name: "get_weather".into(),
                signature: None,
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "content_block_start");
        let json = serde_json::to_value(&events[0]).unwrap();
        assert!(
            json["content_block"].get("signature").is_none(),
            "no signature on the event must mean no signature key on the block: {json}"
        );
    }

    #[test]
    fn streaming_content_start_tool_use_defers_to_tool_call_start() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());

        let content_start_events = enc
            .encode_event(CoreEvent::ContentStart {
                index: 0,
                kind: ContentKind::ToolUse,
            })
            .unwrap();
        assert!(
            content_start_events.is_empty(),
            "ContentStart ToolUse should produce zero events, got {}",
            content_start_events.len()
        );

        let tool_start_events = enc
            .encode_event(CoreEvent::ToolCallStart {
                index: 0,
                id: "tu_1".into(),
                name: "get_weather".into(),
                signature: None,
            })
            .unwrap();
        assert_eq!(tool_start_events.len(), 1);
        assert_eq!(tool_start_events[0].r#type, "content_block_start");
        assert_eq!(
            tool_start_events[0]
                .content_block
                .as_ref()
                .unwrap()
                .id
                .as_deref(),
            Some("tu_1")
        );
    }

    #[test]
    fn streaming_text_block_emits_content_block_stop() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        enc.encode_event(CoreEvent::MessageStart {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
        })
        .unwrap();
        enc.encode_event(CoreEvent::ContentStart {
            index: 0,
            kind: ContentKind::Text,
        })
        .unwrap();
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
        let stop = enc.finish().unwrap();

        assert!(
            stop.iter()
                .any(|e| e.r#type == "content_block_stop" && e.index == Some(0))
        );

        let md = stop
            .iter()
            .position(|e| e.r#type == "message_delta")
            .expect("message_delta present");
        let cbs = stop
            .iter()
            .position(|e| e.r#type == "content_block_stop")
            .expect("content_block_stop present");
        assert!(cbs < md, "content_block_stop must precede message_delta");
    }

    #[test]
    fn streaming_usage_coalesced_with_message_stop() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());

        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(50, 100),
        })
        .unwrap();

        let events = enc
            .encode_event(CoreEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                stop_sequence: None,
            })
            .unwrap();
        assert!(
            events.is_empty(),
            "terminal pair is held back until finish(): got {events:?}"
        );

        let events = enc.finish().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].r#type, "message_delta");
        assert!(events[0].usage.is_some());
        assert_eq!(events[0].usage.as_ref().unwrap().input_tokens, 50);
        assert_eq!(
            events[0].delta.as_ref().unwrap().stop_reason.as_deref(),
            Some("end_turn")
        );
        assert_eq!(events[1].r#type, "message_stop");
    }

    #[test]
    fn usage_arriving_after_message_stop_reaches_the_client() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());

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
        // Standard OpenAI-style ordering: usage arrives after the finish
        // event. It must land in the terminal message_delta the client sees.
        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(100, 5),
        })
        .unwrap();

        let events = enc.finish().unwrap();
        let delta = events
            .iter()
            .find(|e| e.r#type == "message_delta")
            .expect("terminal message_delta after finish");
        let usage = delta.usage.as_ref().expect("usage on terminal delta");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn streaming_message_start_event() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::MessageStart {
                id: Some("msg_abc".into()),
                model: ModelRef {
                    requested: "claude-sonnet-4-20250514".into(),
                    upstream: None,
                },
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "message_start");
        assert_eq!(events[0].message.as_ref().unwrap().id, "msg_abc");
    }

    #[test]
    fn streaming_error_event() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::Error {
                error: CoreStreamError::new(
                    CoreStreamErrorKind::RateLimit,
                    "too many requests".into(),
                ),
            })
            .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "error");
        assert!(events[0].error.is_some());
    }

    #[test]
    fn streaming_ping_event() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc.encode_event(CoreEvent::Ping).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].r#type, "ping");
    }

    #[test]
    fn streaming_finish_after_message_stop_is_empty() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let remaining = enc.finish().unwrap();
        assert_eq!(remaining.len(), 2, "buffered stop flushes on finish()");
        assert_eq!(remaining[0].r#type, "message_delta");
        assert_eq!(remaining[1].r#type, "message_stop");
        let again = enc.finish().unwrap();
        assert!(again.is_empty(), "second finish() must be empty");
    }

    #[test]
    fn streaming_error_after_message_stop_flushes_terminal_then_error() {
        // A transport error or in-band error frame arriving after the
        // upstream terminal must neither be dropped nor drop the buffered
        // terminal pair: the client sees the terminal, then the error.
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let events = enc
            .encode_event(CoreEvent::Error {
                error: CoreStreamError::new(
                    CoreStreamErrorKind::Upstream,
                    "connection reset after terminal".into(),
                ),
            })
            .unwrap();
        let types: Vec<&str> = events.iter().map(|e| e.r#type.as_str()).collect();
        assert_eq!(
            types,
            vec!["message_delta", "message_stop", "error"],
            "terminal pair must flush ahead of the error event: {events:?}"
        );
        assert!(
            events.last().is_some_and(|e| e.error.is_some()),
            "the error event must carry the failure: {events:?}"
        );
        assert!(
            enc.finish().unwrap().is_empty(),
            "no synthetic second terminal after the error"
        );
    }

    #[test]
    fn streaming_finish_with_pending_usage() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        enc.encode_event(CoreEvent::UsageDelta {
            usage: Usage::provider_reported(10, 20),
        })
        .unwrap();
        let remaining = enc.finish().unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].r#type, "message_delta");
        assert_eq!(
            remaining[0].delta.as_ref().unwrap().stop_reason.as_deref(),
            Some("end_turn")
        );
        assert_eq!(remaining[1].r#type, "message_stop");
    }

    #[test]
    fn streaming_finish_without_pending_usage_still_emits_terminal() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let remaining = enc.finish().unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].r#type, "message_delta");
        assert_eq!(
            remaining[0].delta.as_ref().unwrap().stop_reason.as_deref(),
            Some("end_turn")
        );
        assert!(remaining[0].usage.is_some());
        assert_eq!(remaining[0].usage.as_ref().unwrap().input_tokens, 0);
        assert_eq!(remaining[0].usage.as_ref().unwrap().output_tokens, 0);
        assert_eq!(remaining[1].r#type, "message_stop");
    }

    #[test]
    fn streaming_events_after_stop_are_ignored() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        enc.encode_event(CoreEvent::MessageStop {
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
        })
        .unwrap();
        let events = enc
            .encode_event(CoreEvent::TextDelta {
                index: 0,
                text: "ignored".into(),
            })
            .unwrap();
        assert!(events.is_empty());
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
            let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
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
                id: "tu_1".into(),
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
            let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
            let result = enc.encode_event(event.clone());
            assert!(result.is_ok(), "event {event:?} should not error");
        }
    }

    #[test]
    fn encode_image_produces_image_block() {
        let content = CoreContent::Image {
            source: serde_json::json!({"type": "base64", "media_type": "image/png", "data": "iVBOR..."}),
        };
        let block = encode_content_block(content).unwrap();
        assert_eq!(block.r#type, "image");
        assert!(block.source.is_some());
    }

    #[test]
    fn encode_image_forwards_a_url_source_verbatim() {
        let source = serde_json::json!({"type": "url", "url": "https://example.com/cat.jpg"});
        let block = encode_content_block(CoreContent::Image {
            source: source.clone(),
        })
        .unwrap();
        let wire = serde_json::to_value(&block).unwrap();
        assert_eq!(wire, serde_json::json!({"type": "image", "source": source}));
    }

    #[test]
    fn encode_refusal_returns_encode_error() {
        let content = CoreContent::Refusal {
            text: "I cannot help with that".into(),
        };
        let result = encode_content_block(content);
        assert!(matches!(result, Err(ProtocolError::Encode(_))));
    }

    #[test]
    fn encode_redacted_thinking_produces_redacted_thinking_block() {
        let content = CoreContent::RedactedThinking {
            data: serde_json::json!({"redacted": true}),
        };
        let block = encode_content_block(content).unwrap();
        assert_eq!(block.r#type, "redacted_thinking");
        assert!(block.data.is_some());
    }

    #[test]
    fn encode_document_returns_encode_skippable() {
        let content = CoreContent::Document {
            source: serde_json::json!({"url": "http://example.com/doc.pdf"}),
        };
        let result = encode_content_block(content);
        assert!(matches!(result, Err(ProtocolError::EncodeSkippable(_))));
    }

    #[test]
    fn encode_audio_returns_encode_skippable() {
        let content = CoreContent::Audio {
            source: serde_json::json!({"data": "base64..."}),
        };
        let result = encode_content_block(content);
        assert!(matches!(result, Err(ProtocolError::EncodeSkippable(_))));
    }

    #[test]
    fn encode_video_returns_encode_skippable() {
        let content = CoreContent::Video {
            source: serde_json::json!({"url": "http://example.com/vid.mp4"}),
        };
        let result = encode_content_block(content);
        assert!(matches!(result, Err(ProtocolError::EncodeSkippable(_))));
    }

    #[test]
    fn encode_response_omits_unsupported_blocks() {
        let resp = CoreResponse {
            id: Some("msg_123".into()),
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Text {
                    text: "hello".into(),
                    cache: None,
                },
                CoreContent::Document {
                    source: serde_json::json!({"url": "http://example.com/doc.pdf"}),
                },
            ],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let out = encode_response(resp).unwrap();
        assert_eq!(out.content.len(), 1);
        assert_eq!(out.content[0].r#type, "text");
    }

    #[test]
    fn metadata_raw_preserved_in_decode() {
        let mut req = make_anthropic_request();
        req.metadata = Some(anthropic::Metadata {
            user_id: Some("user-42".into()),
            extra: serde_json::Map::new(),
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.metadata.user_id.as_deref(), Some("user-42"));
    }

    #[test]
    fn metadata_extra_fields_preserved_in_raw() {
        let mut req = make_anthropic_request();
        let mut extra = serde_json::Map::new();
        extra.insert("trace_id".into(), serde_json::json!("abc-123"));
        extra.insert("session_id".into(), serde_json::json!("sess-456"));
        req.metadata = Some(anthropic::Metadata {
            user_id: Some("user-42".into()),
            extra: extra.clone(),
        });
        let core = decode_request(req).unwrap();
        assert_eq!(core.metadata.user_id.as_deref(), Some("user-42"));
        assert_eq!(core.metadata.raw.get("trace_id").unwrap(), "abc-123");
        assert_eq!(core.metadata.raw.get("session_id").unwrap(), "sess-456");
    }

    #[test]
    fn redacted_thinking_decodes_to_core() {
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "assistant".into(),
            content: serde_json::json!([
                { "type": "redacted_thinking", "data": "base64encodeddata" }
            ]),
        });
        let core = decode_request(req).unwrap();
        match &core.messages[1].content[0] {
            CoreContent::RedactedThinking { data } => {
                assert_eq!(data.as_str(), Some("base64encodeddata"));
            }
            _ => panic!("expected RedactedThinking"),
        }
    }

    #[test]
    fn unmodeled_block_types_are_carried_verbatim_as_opaque() {
        // `data` is a string on the typed block: an unmodeled block's own field
        // shapes must never reach the typed decode, at top level or nested.
        let unmodeled =
            serde_json::json!({ "type": "future_unknown_block", "data": { "nested": true } });
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "user".into(),
            content: serde_json::json!([
                unmodeled,
                { "type": "tool_result", "tool_use_id": "tu_1", "content": [unmodeled] }
            ]),
        });
        let core = decode_request(req).expect("unmodeled blocks must not fail the decode");
        match &core.messages[1].content[..] {
            [
                CoreContent::Opaque { block },
                CoreContent::ToolResult { content, .. },
            ] => {
                assert_eq!(block, &unmodeled);
                assert_eq!(content, &vec![CoreContent::Opaque { block: unmodeled }]);
            }
            other => panic!("expected Opaque then ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn web_search_replay_blocks_decode_as_opaque_blocks_in_order() {
        let server_tool_use = serde_json::json!({
            "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
            "input": { "query": "claude shannon birth date" }
        });
        let search_result = serde_json::json!({
            "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
            "content": [{
                "type": "web_search_result", "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                "title": "Claude Shannon - Wikipedia", "encrypted_content": "EqgfCioIARgB",
                "page_age": "April 30, 2025"
            }]
        });
        let mut req = make_anthropic_request();
        req.messages.push(Message {
            role: "assistant".into(),
            content: serde_json::json!([
                { "type": "text", "text": "Searching." },
                server_tool_use,
                search_result,
                { "type": "text", "text": "Born April 30, 1916." }
            ]),
        });
        let core = decode_request(req).expect("server-tool replay blocks must decode");
        match &core.messages[1].content[..] {
            [
                CoreContent::Text { text: before, .. },
                CoreContent::Opaque { block: tool_use },
                CoreContent::Opaque { block: tool_result },
                CoreContent::Text { text: after, .. },
            ] => {
                assert_eq!(before, "Searching.");
                assert_eq!(tool_use, &server_tool_use);
                assert_eq!(tool_result, &search_result);
                assert_eq!(after, "Born April 30, 1916.");
            }
            other => panic!("expected Text, two Opaque blocks, Text; got {other:?}"),
        }
    }

    #[test]
    fn tools_decoded_to_core() {
        let mut req = make_anthropic_request();
        req.tools = vec![Tool {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            input_schema: Some(
                serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            ),
            extra: serde_json::Map::new(),
        }];
        let core = decode_request(req).unwrap();
        assert_eq!(core.tools.len(), 1);
        assert_eq!(core.tools[0].name, "get_weather");
        assert_eq!(core.tools[0].description.as_deref(), Some("Get weather"));
    }

    #[test]
    fn client_anthropic_does_not_import_forbidden_modules() {
        let source = include_str!("anthropic.rs");
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") {
                continue;
            }
            if trimmed.starts_with("use ") {
                assert!(
                    !trimmed.contains("llm_proxy_provider"),
                    "client/anthropic.rs must not import llm_proxy_provider"
                );
                assert!(
                    !trimmed.contains("llm_proxy_server"),
                    "client/anthropic.rs must not import llm_proxy_server"
                );
                assert!(
                    !trimmed.contains("crate::config"),
                    "client/anthropic.rs must not import crate::config"
                );
                assert!(
                    !trimmed.contains("crate::routing"),
                    "client/anthropic.rs must not import crate::routing"
                );
                assert!(
                    !trimmed.contains("crate::transformer"),
                    "client/anthropic.rs must not import crate::transformer"
                );
            }
        }
    }

    #[test]
    fn decode_empty_model_returns_error() {
        let mut req = make_anthropic_request();
        req.model = String::new();
        assert!(matches!(
            decode_request(req),
            Err(ProtocolError::InvalidRequest(_))
        ));
    }

    #[test]
    fn decode_empty_messages_returns_error() {
        let mut req = make_anthropic_request();
        req.messages = vec![];
        assert!(matches!(
            decode_request(req),
            Err(ProtocolError::InvalidRequest(_))
        ));
    }

    #[test]
    fn decode_null_optional_fields() {
        let mut req = make_anthropic_request();
        req.system = None;
        req.temperature = None;
        req.top_p = None;
        req.thinking = None;
        req.stream = None;
        let core = decode_request(req).unwrap();
        assert!(core.system.is_empty());
        assert!(core.sampling.temperature.is_none());
        assert!(core.sampling.top_p.is_none());
        assert!(core.sampling.thinking.is_none());
        assert!(!core.stream);
    }

    #[test]
    fn encode_response_with_refusal_propagates_error() {
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
                CoreContent::Refusal {
                    text: "I cannot".into(),
                },
            ],
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            usage: Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let result = encode_response(resp);
        assert!(matches!(result, Err(ProtocolError::Encode(_))));
    }

    #[test]
    fn encode_response_skips_safe_blocks_preserves_text() {
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
                CoreContent::Document {
                    source: serde_json::json!({"url": "x"}),
                },
                CoreContent::Text {
                    text: " world".into(),
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
        assert_eq!(out.content.len(), 2);
        assert_eq!(out.content[0].text.as_deref(), Some("hello"));
        assert_eq!(out.content[1].text.as_deref(), Some(" world"));
    }

    #[test]
    fn encode_error_does_not_leak_secret_in_message() {
        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let secret_msg = "api_key=sk-12345-secret";
        let events = enc
            .encode_event(CoreEvent::Error {
                error: CoreStreamError::new(CoreStreamErrorKind::RateLimit, secret_msg.into()),
            })
            .unwrap();
        assert_eq!(events[0].error.as_ref().unwrap().message, secret_msg);
    }

    #[test]
    fn encode_error_truncates_multibyte_message_without_panicking() {
        let emoji = "🚀";
        assert_eq!(emoji.len(), 4);
        let msg: String = std::iter::once('_')
            .chain(std::iter::repeat_n(emoji, 257).flat_map(|s| s.chars()))
            .collect();
        assert!(
            msg.len() > 1024,
            "precondition: message must exceed the cap"
        );
        assert!(
            !msg.is_char_boundary(1024),
            "precondition: byte 1024 must fall mid-code-point"
        );

        let mut enc = StreamEncoder::new("msg_1".into(), "m".into());
        let events = enc
            .encode_event(CoreEvent::Error {
                error: CoreStreamError::new(CoreStreamErrorKind::Upstream, msg.clone()),
            })
            .unwrap();

        let capped = &events[0].error.as_ref().unwrap().message;
        assert!(
            capped.len() <= 1024,
            "truncated message must not exceed the cap"
        );
        assert!(
            msg.starts_with(capped.as_str()),
            "truncated message must be a prefix of the original"
        );
        assert!(capped.len() < msg.len());
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
        assert_eq!(out.content[0].text.as_deref(), Some(large_text.as_str()));
    }
}
