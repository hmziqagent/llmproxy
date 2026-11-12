//! OpenAI Responses client protocol adapter (plan 08, step 1): translates
//! between the Responses API wire format (`/v1/responses`, the dialect Codex
//! speaks) and the normalised core protocol types.
//!
//! Wire shapes follow the Responses API reference
//! (<https://platform.openai.com/docs/api-reference/responses>) and the SSE
//! event sequence the Codex CLI consumes
//! (<https://github.com/openai/codex>): `response.output_item.added` before
//! any delta for the item, `response.function_call_arguments.delta` per
//! argument fragment, `response.output_item.done` with the complete item, and
//! one terminal `response.completed`/`response.incomplete`/`response.failed`
//! event whose `response.usage` carries the exact documented fields —
//! `input_tokens_details.cached_tokens` and
//! `output_tokens_details.reasoning_tokens` are always present as explicit
//! numbers (never omitted via omitempty), because Responses clients break on
//! their absence.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::client::ProtocolError;
use crate::core::{
    ContentKind, CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreResponse, CoreRole,
    CoreTool, CoreToolChoice, ModelRef, RequestMetadata, SamplingOptions, StopReason, Usage,
    json_type_name,
};

/// Top-level request keys the Responses client decode consumes; everything
/// else is an "extra" for the cross-format field policy (and survives
/// same-format passthrough untouched).
pub const RESPONSES_REQUEST_KEYS: &[&str] = &[
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_output_tokens",
    "reasoning",
    "stream",
    "metadata",
];

/// Decode a raw Responses API request body into a normalised [`CoreRequest`].
///
/// Unknown top-level fields are kept in `metadata.raw` (visible to the
/// cross-format policy, which names them) rather than dropped.
pub fn decode_request(body: &Value) -> Result<CoreRequest, ProtocolError> {
    let obj = body.as_object().ok_or_else(|| {
        ProtocolError::InvalidRequest("request body must be a JSON object".into())
    })?;

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ProtocolError::InvalidRequest("model is required".into()))?;

    let mut system = Vec::new();
    if let Some(instructions) = obj.get("instructions")
        && let Some(text) = decode_instructions(instructions)?.filter(|text| !text.is_empty())
    {
        system.push(CoreContent::Text { text, cache: None });
    }

    let input = obj
        .get("input")
        .ok_or_else(|| ProtocolError::InvalidRequest("input is required".into()))?;
    let messages = decode_input(input)?;

    let tools = decode_tools(obj.get("tools"))?;
    let tool_choice = obj.get("tool_choice").map(decode_tool_choice);
    let sampling = decode_sampling(obj)?;

    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let user_id = obj
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("user_id"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    let mut raw = Map::new();
    for (key, value) in obj {
        if !RESPONSES_REQUEST_KEYS.contains(&key.as_str()) {
            raw.insert(key.clone(), value.clone());
        }
    }

    Ok(CoreRequest {
        model: ModelRef {
            requested: model.to_owned(),
            upstream: None,
        },
        system,
        messages,
        tools,
        tool_choice,
        sampling,
        stream,
        metadata: RequestMetadata { user_id, raw },
        provider_hints: Default::default(),
    })
}

/// `instructions` is a string or an array of `{type, text}` parts; parts
/// join with newlines like the Anthropic system encode.
fn decode_instructions(instructions: &Value) -> Result<Option<String>, ProtocolError> {
    match instructions {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        Value::Array(parts) => {
            let mut joined = String::new();
            for part in parts {
                let Some(text) = part
                    .as_object()
                    .and_then(|p| p.get("text"))
                    .and_then(Value::as_str)
                else {
                    return Err(ProtocolError::InvalidRequest(format!(
                        "instructions parts must be objects with a \"text\" field, found {}",
                        json_type_name(part)
                    )));
                };
                if !joined.is_empty() {
                    joined.push('\n');
                }
                joined.push_str(text);
            }
            Ok(Some(joined))
        }
        other => Err(ProtocolError::InvalidRequest(format!(
            "instructions must be a string or an array of text parts, found {}",
            json_type_name(other)
        ))),
    }
}

/// The sampling knobs a Responses request carries: temperature, top_p,
/// `max_output_tokens` and the `reasoning.effort` level.
fn decode_sampling(obj: &Map<String, Value>) -> Result<SamplingOptions, ProtocolError> {
    Ok(SamplingOptions {
        temperature: get_f64(obj, "temperature")?,
        top_p: get_f64(obj, "top_p")?,
        max_tokens: obj
            .get("max_output_tokens")
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok()),
        stop: None,
        top_k: None,
        reasoning_effort: obj
            .get("reasoning")
            .and_then(Value::as_object)
            .and_then(|r| r.get("effort"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        thinking: None,
    })
}

fn get_f64(obj: &Map<String, Value>, key: &str) -> Result<Option<f64>, ProtocolError> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .map(Some)
            .ok_or_else(|| ProtocolError::InvalidRequest(format!("{key} must be a finite number"))),
        Some(other) => Err(ProtocolError::InvalidRequest(format!(
            "{key} must be a number, found {}",
            json_type_name(other)
        ))),
    }
}

/// Decode the `input` field: a bare string, or an array of typed items
/// (messages, `function_call`, `function_call_output`, `reasoning`).
fn decode_input(input: &Value) -> Result<Vec<CoreMessage>, ProtocolError> {
    match input {
        Value::String(text) => Ok(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: text.clone(),
                cache: None,
            }],
        }]),
        Value::Array(items) => {
            let mut messages = Vec::new();
            for item in items {
                decode_input_item(item, &mut messages)?;
            }
            Ok(messages)
        }
        Value::Null => Err(ProtocolError::InvalidRequest("input is required".into())),
        other => Err(ProtocolError::InvalidRequest(format!(
            "input must be a string or an array of items, found {}",
            json_type_name(other)
        ))),
    }
}

fn decode_input_item(item: &Value, messages: &mut Vec<CoreMessage>) -> Result<(), ProtocolError> {
    let obj = item.as_object().ok_or_else(|| {
        ProtocolError::InvalidRequest(format!(
            "input items must be objects, found {}",
            json_type_name(item)
        ))
    })?;
    let item_type = obj.get("type").and_then(Value::as_str);
    match item_type {
        None | Some("message") => decode_input_message(obj, messages),
        Some("function_call") => {
            messages.push(decode_function_call(obj)?);
            Ok(())
        }
        Some("function_call_output") => {
            messages.push(decode_function_call_output(obj)?);
            Ok(())
        }
        Some("reasoning") => {
            // A replayed reasoning item: the summary text becomes thinking
            // content and `encrypted_content` rides as the opaque signature
            // (Anthropic-style) so same-shape replay stays possible.
            let summary_text: Vec<&str> = obj
                .get("summary")
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|e| {
                            e.as_object()
                                .and_then(|o| o.get("text"))
                                .and_then(Value::as_str)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let signature = obj
                .get("encrypted_content")
                .and_then(Value::as_str)
                .map(str::to_owned);
            messages.push(CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::Thinking {
                    text: summary_text.join("\n"),
                    signature,
                }],
            });
            Ok(())
        }
        Some("item_reference") => Err(ProtocolError::InvalidRequest(
            "item_reference input items cannot be resolved by the proxy; resend the full item"
                .into(),
        )),
        Some(other) => Err(ProtocolError::InvalidRequest(format!(
            "unknown input item type: {other}"
        ))),
    }
}

/// A replayed `function_call` input item.
fn decode_function_call(obj: &Map<String, Value>) -> Result<CoreMessage, ProtocolError> {
    let call_id = required_str(obj, "call_id", "function_call item")?;
    let name = required_str(obj, "name", "function_call item")?;
    let arguments = obj.get("arguments").and_then(Value::as_str).unwrap_or("{}");
    let input = serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| {
        tracing::warn!(
            arguments,
            "Responses: malformed function_call arguments; replacing with empty object"
        );
        Value::Object(Map::new())
    });
    Ok(CoreMessage {
        role: CoreRole::Assistant,
        content: vec![CoreContent::ToolUse {
            id: call_id,
            name,
            input,
            signature: None,
        }],
    })
}

/// A `function_call_output` item; `output` is a string, or an object whose
/// `content` is the string (or the serialized object itself).
fn decode_function_call_output(obj: &Map<String, Value>) -> Result<CoreMessage, ProtocolError> {
    let call_id = required_str(obj, "call_id", "function_call_output item")?;
    let output = match obj.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(map)) => match map.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => {
                return Err(ProtocolError::InvalidRequest(format!(
                    "function_call_output output.content must be a string, found {}",
                    json_type_name(other)
                )));
            }
            None => serde_json::to_string(map)
                .map_err(|e| ProtocolError::Decode(format!("function_call_output output: {e}")))?,
        },
        None | Some(Value::Null) => String::new(),
        Some(other) => {
            return Err(ProtocolError::InvalidRequest(format!(
                "function_call_output output must be a string or an object, found {}",
                json_type_name(other)
            )));
        }
    };
    Ok(CoreMessage {
        role: CoreRole::Tool,
        content: vec![CoreContent::ToolResult {
            tool_use_id: call_id,
            content: if output.is_empty() {
                Vec::new()
            } else {
                vec![CoreContent::Text {
                    text: output,
                    cache: None,
                }]
            },
            is_error: false,
            cache: None,
        }],
    })
}

fn decode_input_message(
    obj: &Map<String, Value>,
    messages: &mut Vec<CoreMessage>,
) -> Result<(), ProtocolError> {
    let role = obj
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| ProtocolError::InvalidRequest("input message items need a role".into()))?;
    let role = match role {
        "user" => CoreRole::User,
        "assistant" => CoreRole::Assistant,
        "system" | "developer" => CoreRole::System,
        other => {
            return Err(ProtocolError::InvalidRequest(format!(
                "unknown input message role: {other}"
            )));
        }
    };
    let content = match obj.get("content") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![CoreContent::Text {
            text: text.clone(),
            cache: None,
        }],
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::new();
            for part in parts {
                let part_obj = part.as_object().ok_or_else(|| {
                    ProtocolError::InvalidRequest(format!(
                        "content parts must be objects, found {}",
                        json_type_name(part)
                    ))
                })?;
                match part_obj.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") | Some("text") => {
                        let text = part_obj
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        blocks.push(CoreContent::Text {
                            text: text.to_owned(),
                            cache: None,
                        });
                    }
                    Some("input_image") => {
                        if let Some(source) = decode_input_image(part_obj)? {
                            blocks.push(CoreContent::Image { source });
                        }
                    }
                    Some("summary_text") => {
                        let text = part_obj
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        blocks.push(CoreContent::Thinking {
                            text: text.to_owned(),
                            signature: None,
                        });
                    }
                    Some(other) => {
                        return Err(ProtocolError::InvalidRequest(format!(
                            "unknown content part type: {other}"
                        )));
                    }
                    None => {
                        return Err(ProtocolError::InvalidRequest(
                            "content parts need a type field".into(),
                        ));
                    }
                }
            }
            blocks
        }
        Some(other) => {
            return Err(ProtocolError::InvalidRequest(format!(
                "message content must be a string or an array of parts, found {}",
                json_type_name(other)
            )));
        }
    };
    messages.push(CoreMessage { role, content });
    Ok(())
}

/// An `input_image` part maps onto the neutral core image source
/// (`{"type":"url","url":...}` or `{"type":"base64",...}` from a `data:` URI).
fn decode_input_image(part: &Map<String, Value>) -> Result<Option<Value>, ProtocolError> {
    let url = part
        .get("image_url")
        .and_then(Value::as_str)
        .or_else(|| {
            part.get("image_url")
                .and_then(Value::as_object)
                .and_then(|o| o.get("url"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| {
            ProtocolError::InvalidRequest("input_image parts need an image_url".into())
        })?;
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or_else(|| {
            ProtocolError::InvalidRequest("input_image data URI has no comma".into())
        })?;
        let media_type = meta.split(';').next().unwrap_or_default().to_owned();
        let is_base64 = meta.to_ascii_lowercase().contains(";base64");
        if !is_base64 || media_type.is_empty() || data.is_empty() {
            return Err(ProtocolError::InvalidRequest(
                "input_image data URI must be base64 with a media type".into(),
            ));
        }
        return Ok(Some(
            serde_json::json!({"type": "base64", "media_type": media_type, "data": data}),
        ));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(Some(serde_json::json!({"type": "url", "url": url})));
    }
    Err(ProtocolError::InvalidRequest(format!(
        "unsupported input_image url: {url}"
    )))
}

/// Function tools decode into core tools; every other tool type is a server
/// tool the core cannot represent — it stays out of the core list (the
/// cross-format policy rejects it by name; same-format passthrough carries
/// the raw body untouched).
fn decode_tools(tools: Option<&Value>) -> Result<Vec<CoreTool>, ProtocolError> {
    let Some(Value::Array(items)) = tools else {
        return Ok(Vec::new());
    };
    let mut decoded = Vec::new();
    for tool in items {
        let obj = tool
            .as_object()
            .ok_or_else(|| ProtocolError::InvalidRequest("tools must be objects".into()))?;
        let tool_type = obj
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function");
        if tool_type != "function" {
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("<unnamed>");
            tracing::debug!(
                tool_type,
                name,
                "Responses: server tool left out of the core tool list (cross-format policy names it)"
            );
            continue;
        }
        let name = required_str(obj, "name", "function tool")?;
        let description = obj
            .get("description")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let input_schema = match obj.get("parameters") {
            Some(value) if value.is_object() => value.clone(),
            None | Some(Value::Null) => Value::Object(Map::new()),
            Some(other) => {
                return Err(ProtocolError::InvalidRequest(format!(
                    "tool parameters must be an object, found {}",
                    json_type_name(other)
                )));
            }
        };
        decoded.push(CoreTool {
            name,
            description,
            input_schema,
        });
    }
    Ok(decoded)
}

fn decode_tool_choice(value: &Value) -> CoreToolChoice {
    if let Some(s) = value.as_str() {
        return match s {
            "auto" => CoreToolChoice::Auto,
            "none" => CoreToolChoice::None,
            "required" => CoreToolChoice::Any,
            _ => CoreToolChoice::Raw(value.clone()),
        };
    }
    if let Some(obj) = value.as_object() {
        return match obj.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                CoreToolChoice::Tool { name }
            }
            Some("allowed_tools") => CoreToolChoice::Any,
            _ => CoreToolChoice::Raw(value.clone()),
        };
    }
    CoreToolChoice::Raw(value.clone())
}

fn required_str(
    obj: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<String, ProtocolError> {
    obj.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ProtocolError::InvalidRequest(format!("{context} needs a non-empty {key}")))
}

/// The exact documented usage object. `input_tokens_details.cached_tokens`
/// and `output_tokens_details.reasoning_tokens` are always present as
/// explicit numbers — omitting a zero via omitempty is the exact bug that
/// broke Codex clients against one-hub (backlog 9).
pub(crate) fn encode_usage(usage: &Usage) -> Value {
    // The Responses wire's `input_tokens` includes the cached subset; the
    // core's `input_tokens` excludes cache reads (Anthropic wire semantics),
    // so the cache read is added back here — a same-format Responses hop
    // must round-trip the exact numbers the upstream reported.
    let cached = i64::from(usage.cache_read_input_tokens.unwrap_or(0));
    let input = i64::from(usage.input_tokens).saturating_add(cached);
    let output = i64::from(usage.output_tokens);
    serde_json::json!({
        "input_tokens": i32::try_from(input).unwrap_or(i32::MAX),
        "input_tokens_details": {
            "cached_tokens": usage.cache_read_input_tokens.unwrap_or(0),
        },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": usage.reasoning_tokens.unwrap_or(0),
        },
        "total_tokens": i32::try_from(input.saturating_add(output)).unwrap_or(i32::MAX),
    })
}

/// Map a core stop reason onto the Responses status/incomplete shape.
fn stop_status(reason: &StopReason) -> (&'static str, Option<Value>) {
    match reason {
        StopReason::MaxTokens => (
            "incomplete",
            Some(serde_json::json!({"reason": "max_output_tokens"})),
        ),
        StopReason::Error => ("failed", None),
        StopReason::Refusal => (
            "incomplete",
            Some(serde_json::json!({"reason": "content_filter"})),
        ),
        _ => ("completed", None),
    }
}

/// The terminal event name for a response status: `response.completed` or
/// `response.incomplete` (a failure goes through `response.failed`).
fn finish_from_status(status: &str) -> &'static str {
    if status == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    }
}

/// One output item for the response object, shared by the non-streaming
/// encode and the terminal stream events.
fn encode_output_item(item: &EncodedItem) -> Value {
    match &item.kind {
        ItemKind::Message { text, .. } => serde_json::json!({
            "type": "message",
            "id": item.item_id,
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        }),
        ItemKind::Reasoning { text, .. } => serde_json::json!({
            "type": "reasoning",
            "id": item.item_id,
            "summary": if text.is_empty() { Vec::<Value>::new() } else {
                vec![serde_json::json!({"type": "summary_text", "text": text})]
            },
        }),
        ItemKind::FunctionCall {
            call_id,
            name,
            arguments,
        } => serde_json::json!({
            "type": "function_call",
            "id": item.item_id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        }),
    }
}

#[derive(Debug, Clone)]
enum ItemKind {
    Message {
        text: String,
    },
    Reasoning {
        text: String,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Clone)]
struct EncodedItem {
    item_id: String,
    output_index: usize,
    kind: ItemKind,
}

/// Encode a [`CoreResponse`] into a Responses API response object.
pub fn encode_response(resp: &CoreResponse) -> Result<Value, ProtocolError> {
    if let Some(seq) = resp.stop_sequence.as_deref()
        && !seq.is_empty()
    {
        tracing::warn!(
            stop_sequence = seq,
            "the Responses API has no stop_sequence field on responses; dropping"
        );
    }
    let id = resp
        .id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", uuid::Uuid::new_v4().simple()));
    let mut items: Vec<EncodedItem> = Vec::new();
    for block in &resp.content {
        let output_index = items.len();
        match block {
            CoreContent::Text { text, .. } => {
                items.push(EncodedItem {
                    item_id: format!("msg_{output_index}"),
                    output_index,
                    kind: ItemKind::Message { text: text.clone() },
                });
            }
            CoreContent::Thinking { text, .. } => {
                items.push(EncodedItem {
                    item_id: format!("rs_{output_index}"),
                    output_index,
                    kind: ItemKind::Reasoning { text: text.clone() },
                });
            }
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                let arguments = serde_json::to_string(input)
                    .map_err(|e| ProtocolError::Encode(format!("tool arguments: {e}")))?;
                items.push(EncodedItem {
                    item_id: format!("fc_{output_index}"),
                    output_index,
                    kind: ItemKind::FunctionCall {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments,
                    },
                });
            }
            CoreContent::Refusal { text } => {
                items.push(EncodedItem {
                    item_id: format!("msg_{output_index}"),
                    output_index,
                    kind: ItemKind::Message { text: text.clone() },
                });
            }
            other => {
                tracing::warn!(
                    kind = %core_kind_name(other),
                    "Responses responses cannot carry this block; dropping"
                );
            }
        }
    }
    let (status, incomplete_details) = stop_status(&resp.stop_reason);
    let mut response = serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": unix_timestamp_secs(),
        "model": resp.model.requested,
        "output": items.iter().map(encode_output_item).collect::<Vec<_>>(),
        "usage": encode_usage(&resp.usage),
        "status": status,
        "error": Value::Null,
    });
    if let Some(details) = incomplete_details {
        response["incomplete_details"] = details;
    }
    Ok(response)
}

fn core_kind_name(block: &CoreContent) -> &'static str {
    match block {
        CoreContent::Text { .. } => "text",
        CoreContent::Image { .. } => "image",
        CoreContent::Document { .. } => "document",
        CoreContent::Audio { .. } => "audio",
        CoreContent::Video { .. } => "video",
        CoreContent::ToolUse { .. } => "tool_use",
        CoreContent::ToolResult { .. } => "tool_result",
        CoreContent::Thinking { .. } => "thinking",
        CoreContent::RedactedThinking { .. } => "redacted_thinking",
        CoreContent::Refusal { .. } => "refusal",
        CoreContent::Opaque { .. } => "opaque",
    }
}

fn unix_timestamp_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|e| {
            tracing::warn!("system clock appears to be before UNIX epoch: {e}");
            std::time::Duration::ZERO
        })
        .as_secs() as i64
}

/// One synthesized SSE event: the `event:` name and the JSON payload.
#[derive(Debug, Clone)]
pub struct ResponsesSseEvent {
    /// The `event:` line value (e.g. `response.output_text.delta`).
    pub event_type: String,
    /// The `data:` payload.
    pub payload: Value,
}

/// Stateful encoder that converts [`CoreEvent`] values into the documented
/// Responses SSE event sequence.
///
/// Tool-call ordering (the Codex requirement, backlog 9): `output_item.added`
/// opens a function_call item with empty `arguments`, every argument
/// fragment is a `function_call_arguments.delta`, and `output_item.done`
/// closes the item with the complete arguments — deltas for an item never
/// appear before its `added` event or after its `done` event.
#[derive(Debug)]
pub struct StreamEncoder {
    id: Arc<str>,
    model: Arc<str>,
    sequence: u64,
    output_index: usize,
    /// The open message item (text), closed on the next item or at the end.
    open_text: Option<OpenText>,
    /// The open reasoning item.
    open_reasoning: Option<OpenText>,
    /// Tool calls keyed by core content index.
    open_tool_calls: std::collections::HashMap<usize, EncodedItem>,
    finished_items: Vec<EncodedItem>,
    pending_usage: Option<Usage>,
    pending_stop: Option<StopReason>,
    failed: Option<crate::core::CoreStreamError>,
    finished: bool,
}

#[derive(Debug)]
struct OpenText {
    item: EncodedItem,
    text: String,
}

impl StreamEncoder {
    /// Create a stream encoder; `id` and `model` come from the first
    /// `CoreEvent::MessageStart` (or are generated by the route).
    pub fn new(id: String, model: String) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            sequence: 0,
            output_index: 0,
            open_text: None,
            open_reasoning: None,
            open_tool_calls: std::collections::HashMap::new(),
            finished_items: Vec::new(),
            pending_usage: None,
            pending_stop: None,
            failed: None,
            finished: false,
        }
    }

    fn next_seq(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    fn base_payload(&mut self) -> Map<String, Value> {
        let mut payload = Map::new();
        payload.insert("type".into(), Value::Null);
        payload.insert("sequence_number".into(), Value::from(self.next_seq()));
        payload
    }

    fn event(&self, event_type: &str, mut payload: Map<String, Value>) -> ResponsesSseEvent {
        payload.insert("type".into(), Value::String(event_type.to_owned()));
        ResponsesSseEvent {
            event_type: event_type.to_owned(),
            payload: Value::Object(payload),
        }
    }

    /// The response skeleton every terminal event embeds; `incomplete_details`
    /// comes from the same [`stop_status`] mapping as the non-streaming
    /// encode.
    fn response_object(
        &self,
        status: &str,
        incomplete_details: Option<Value>,
        error: Value,
    ) -> Value {
        // Output items in output order: the finished ones, the open text or
        // reasoning item, then any tool calls still open.
        let mut items: Vec<(usize, Value)> = self
            .finished_items
            .iter()
            .map(|item| (item.output_index, encode_output_item(item)))
            .collect();
        if let Some(open) = &self.open_text {
            let item = OpenText::as_item(open);
            items.push((item.output_index, encode_output_item(&item)));
        }
        if let Some(open) = &self.open_reasoning {
            let item = OpenText::as_item(open);
            items.push((item.output_index, encode_output_item(&item)));
        }
        for tool in self.open_tool_calls.values() {
            items.push((tool.output_index, encode_output_item(tool)));
        }
        items.sort_by_key(|(index, _)| *index);
        let usage = self
            .pending_usage
            .clone()
            .unwrap_or_else(Usage::synthetic_zero);
        let mut response = serde_json::json!({
            "id": self.id.to_string(),
            "object": "response",
            "created_at": unix_timestamp_secs(),
            "model": self.model.to_string(),
            "output": items.into_iter().map(|(_, value)| value).collect::<Vec<_>>(),
            "usage": encode_usage(&usage),
            "status": status,
        });
        response["error"] = error;
        if let Some(details) = incomplete_details
            && let Some(obj) = response.as_object_mut()
        {
            obj.insert("incomplete_details".to_owned(), details);
        }
        response
    }

    /// Close the open text or reasoning item, emitting its terminal events.
    fn close_open_items(&mut self, events: &mut Vec<ResponsesSseEvent>) {
        if let Some(open) = self.open_text.take() {
            let item = OpenText::as_item(&open);
            let item_id = item.item_id.clone();
            let output_index = item.output_index;
            let text = match &item.kind {
                ItemKind::Message { text } => text.clone(),
                _ => String::new(),
            };
            let mut done = self.base_payload();
            done.insert("item_id".into(), Value::String(item_id.clone()));
            done.insert("output_index".into(), Value::from(output_index));
            done.insert("content_index".into(), Value::from(0));
            done.insert("text".into(), Value::String(text.clone()));
            events.push(self.event("response.output_text.done", done));

            let mut part_done = self.base_payload();
            part_done.insert("item_id".into(), Value::String(item_id));
            part_done.insert("output_index".into(), Value::from(output_index));
            part_done.insert("content_index".into(), Value::from(0));
            part_done.insert(
                "part".into(),
                serde_json::json!({"type": "output_text", "text": text, "annotations": []}),
            );
            events.push(self.event("response.content_part.done", part_done));

            let mut item_payload = self.base_payload();
            item_payload.insert("output_index".into(), Value::from(output_index));
            item_payload.insert("item".into(), encode_output_item(&item));
            events.push(self.event("response.output_item.done", item_payload));
            self.finished_items.push(item);
        }
        if let Some(open) = self.open_reasoning.take() {
            let item = OpenText::as_item(&open);
            let item_id = item.item_id.clone();
            let output_index = item.output_index;
            let text = match &item.kind {
                ItemKind::Reasoning { text } => text.clone(),
                _ => String::new(),
            };
            let mut done = self.base_payload();
            done.insert("item_id".into(), Value::String(item_id.clone()));
            done.insert("output_index".into(), Value::from(output_index));
            done.insert("summary_index".into(), Value::from(0));
            done.insert("text".into(), Value::String(text.clone()));
            events.push(self.event("response.reasoning_summary_text.done", done));

            let mut part_done = self.base_payload();
            part_done.insert("item_id".into(), Value::String(item_id));
            part_done.insert("output_index".into(), Value::from(output_index));
            part_done.insert("summary_index".into(), Value::from(0));
            part_done.insert(
                "part".into(),
                serde_json::json!({"type": "summary_text", "text": text}),
            );
            events.push(self.event("response.reasoning_summary_part.done", part_done));

            let mut item_payload = self.base_payload();
            item_payload.insert("output_index".into(), Value::from(output_index));
            item_payload.insert("item".into(), encode_output_item(&item));
            events.push(self.event("response.output_item.done", item_payload));
            self.finished_items.push(item);
        }
    }

    /// Encode a single [`CoreEvent`] into zero or more SSE events.
    #[expect(
        clippy::too_many_lines,
        reason = "one match arm per CoreEvent variant: the Responses SSE event shape is a \
                 single translation table, and splitting the arms into helpers would \
                 separate each event's field mapping from the item lifecycle it touches"
    )]
    pub fn encode_event(
        &mut self,
        event: CoreEvent,
    ) -> Result<Vec<ResponsesSseEvent>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        match event {
            CoreEvent::MessageStart { .. } => {
                let mut payload = self.base_payload();
                payload.insert(
                    "response".into(),
                    self.response_object("in_progress", None, Value::Null),
                );
                events.push(self.event("response.created", payload));
            }
            CoreEvent::ContentStart { index, kind } => {
                // A new content block closes the open text/reasoning item;
                // tool items open on their own ToolCallStart event.
                match kind {
                    ContentKind::ToolUse => {
                        let _ = index;
                    }
                    ContentKind::Text => {
                        self.close_open_items(&mut events);
                        let open = OpenText {
                            item: EncodedItem {
                                item_id: format!("msg_{}", self.output_index),
                                output_index: self.output_index,
                                kind: ItemKind::Message {
                                    text: String::new(),
                                },
                            },
                            text: String::new(),
                        };
                        self.output_index += 1;
                        let item = OpenText::as_item(&open);
                        let mut added = self.base_payload();
                        let output_index = item.output_index;
                        added.insert("output_index".into(), Value::from(output_index));
                        added.insert("item".into(), encode_output_item(&item));
                        events.push(self.event("response.output_item.added", added));
                        let mut part = self.base_payload();
                        part.insert("item_id".into(), Value::String(item.item_id));
                        part.insert("output_index".into(), Value::from(output_index));
                        part.insert("content_index".into(), Value::from(0));
                        part.insert(
                            "part".into(),
                            serde_json::json!({"type": "output_text", "text": "", "annotations": []}),
                        );
                        events.push(self.event("response.content_part.added", part));
                        self.open_text = Some(open);
                    }
                    ContentKind::Thinking => {
                        self.close_open_items(&mut events);
                        let open = OpenText {
                            item: EncodedItem {
                                item_id: format!("rs_{}", self.output_index),
                                output_index: self.output_index,
                                kind: ItemKind::Reasoning {
                                    text: String::new(),
                                },
                            },
                            text: String::new(),
                        };
                        self.output_index += 1;
                        let item = OpenText::as_item(&open);
                        let mut added = self.base_payload();
                        let output_index = item.output_index;
                        added.insert("output_index".into(), Value::from(output_index));
                        added.insert("item".into(), encode_output_item(&item));
                        events.push(self.event("response.output_item.added", added));
                        let mut part = self.base_payload();
                        part.insert("item_id".into(), Value::String(item.item_id));
                        part.insert("output_index".into(), Value::from(output_index));
                        part.insert("summary_index".into(), Value::from(0));
                        part.insert(
                            "part".into(),
                            serde_json::json!({"type": "summary_text", "text": ""}),
                        );
                        events.push(self.event("response.reasoning_summary_part.added", part));
                        self.open_reasoning = Some(open);
                    }
                    other => {
                        tracing::warn!(kind = ?other, "Responses: unsupported ContentKind");
                    }
                }
            }
            CoreEvent::TextDelta { text, .. } => {
                let Some(open) = self.open_text.as_mut() else {
                    tracing::trace!("Responses: TextDelta without an open text item; dropping");
                    return Ok(events);
                };
                open.text.push_str(&text);
                let item_id = open.item.item_id.clone();
                let output_index = open.item.output_index;
                let mut payload = self.base_payload();
                payload.insert("item_id".into(), Value::String(item_id));
                payload.insert("output_index".into(), Value::from(output_index));
                payload.insert("content_index".into(), Value::from(0));
                payload.insert("delta".into(), Value::String(text));
                events.push(self.event("response.output_text.delta", payload));
            }
            CoreEvent::ThinkingDelta { text, .. } => {
                let Some(open) = self.open_reasoning.as_mut() else {
                    tracing::trace!(
                        "Responses: ThinkingDelta without an open reasoning item; dropping"
                    );
                    return Ok(events);
                };
                open.text.push_str(&text);
                let item_id = open.item.item_id.clone();
                let output_index = open.item.output_index;
                let mut payload = self.base_payload();
                payload.insert("item_id".into(), Value::String(item_id));
                payload.insert("output_index".into(), Value::from(output_index));
                payload.insert("summary_index".into(), Value::from(0));
                payload.insert("delta".into(), Value::String(text));
                events.push(self.event("response.reasoning_summary_text.delta", payload));
            }
            CoreEvent::ToolCallStart {
                index, id, name, ..
            } => {
                self.close_open_items(&mut events);
                let item = EncodedItem {
                    item_id: format!("fc_{}", self.output_index),
                    output_index: self.output_index,
                    kind: ItemKind::FunctionCall {
                        call_id: id,
                        name,
                        arguments: String::new(),
                    },
                };
                self.output_index += 1;
                let mut payload = self.base_payload();
                payload.insert("output_index".into(), Value::from(item.output_index));
                payload.insert("item".into(), encode_output_item(&item));
                events.push(self.event("response.output_item.added", payload));
                self.open_tool_calls.insert(index, item);
            }
            CoreEvent::ToolCallDelta { index, args_delta } => {
                let Some(item) = self.open_tool_calls.get_mut(&index) else {
                    tracing::warn!(
                        core_index = index,
                        "Responses: ToolCallDelta without an open tool item; dropping"
                    );
                    return Ok(events);
                };
                if let ItemKind::FunctionCall { arguments, .. } = &mut item.kind {
                    arguments.push_str(&args_delta);
                }
                let item_id = item.item_id.clone();
                let output_index = item.output_index;
                let mut payload = self.base_payload();
                payload.insert("item_id".into(), Value::String(item_id));
                payload.insert("output_index".into(), Value::from(output_index));
                payload.insert("delta".into(), Value::String(args_delta));
                events.push(self.event("response.function_call_arguments.delta", payload));
            }
            CoreEvent::ToolCallStop { index } => {
                if let Some(item) = self.open_tool_calls.remove(&index) {
                    let arguments = match &item.kind {
                        ItemKind::FunctionCall { arguments, .. } => arguments.clone(),
                        _ => String::new(),
                    };
                    let item_id = item.item_id.clone();
                    let output_index = item.output_index;
                    let mut payload = self.base_payload();
                    payload.insert("item_id".into(), Value::String(item_id));
                    payload.insert("output_index".into(), Value::from(output_index));
                    payload.insert("arguments".into(), Value::String(arguments));
                    events.push(self.event("response.function_call_arguments.done", payload));

                    let mut done = self.base_payload();
                    done.insert("output_index".into(), Value::from(output_index));
                    done.insert("item".into(), encode_output_item(&item));
                    events.push(self.event("response.output_item.done", done));
                    self.finished_items.push(item);
                }
            }
            CoreEvent::UsageDelta { usage } => {
                self.pending_usage = Some(usage);
            }
            CoreEvent::MessageStop { stop_reason, .. } => {
                self.close_open_items(&mut events);
                self.pending_stop = Some(stop_reason);
                events.extend(self.terminal_events());
            }
            CoreEvent::Error { error } => {
                self.failed = Some(error);
                events.extend(self.terminal_events());
            }
            CoreEvent::Ping => {}
        }
        Ok(events)
    }

    fn terminal_events(&mut self) -> Vec<ResponsesSseEvent> {
        self.finished = true;
        let mut events = Vec::new();
        if let Some(error) = self.failed.take() {
            let mut payload = self.base_payload();
            payload.insert(
                "response".into(),
                self.response_object(
                    "failed",
                    None,
                    serde_json::json!({
                        "code": Value::Null,
                        "message": error.message(),
                    }),
                ),
            );
            events.push(self.event("response.failed", payload));
            return events;
        }
        let stop = self.pending_stop.clone().unwrap_or_else(|| {
            tracing::warn!(
                "Responses: stream ended without a MessageStop; synthesizing a completed terminal"
            );
            StopReason::EndTurn
        });
        let (status, incomplete_details) = stop_status(&stop);
        let mut payload = self.base_payload();
        payload.insert(
            "response".into(),
            self.response_object(status, incomplete_details, Value::Null),
        );
        let event_type = finish_from_status(status);
        events.push(self.event(event_type, payload));
        events
    }

    /// Flush at the natural end of the stream; a stream that ended without a
    /// terminal event gets a synthetic one (with a warning).
    pub fn finish(&mut self) -> Result<Vec<ResponsesSseEvent>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        Ok(self.terminal_events())
    }

    /// Mark the encoder as finished so subsequent `finish()` calls return
    /// empty; used after a terminal error has been emitted.
    pub fn mark_finished(&mut self) {
        self.finished = true;
    }
}

impl OpenText {
    fn as_item(&self) -> EncodedItem {
        let kind = match &self.item.kind {
            ItemKind::Message { .. } => ItemKind::Message {
                text: self.text.clone(),
            },
            ItemKind::Reasoning { .. } => ItemKind::Reasoning {
                text: self.text.clone(),
            },
            other => other.clone(),
        };
        EncodedItem {
            item_id: self.item.item_id.clone(),
            output_index: self.item.output_index,
            kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::UsageProvenance;

    fn stream_events(encoder: &mut StreamEncoder, events: Vec<CoreEvent>) -> Vec<Value> {
        let mut out = Vec::new();
        for event in events {
            for encoded in encoder.encode_event(event).expect("encode") {
                out.push(encoded.payload);
            }
        }
        out
    }

    #[test]
    fn decode_request_maps_the_documented_codex_shape() {
        let body = serde_json::json!({
            "model": "gpt-5.1-codex",
            "instructions": "You are a coding agent.",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "list the files"}
                ]},
                {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.txt\nb.txt"},
                {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "run ls"}], "encrypted_content": "ENC"}
            ],
            "tools": [{
                "type": "function",
                "name": "shell",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"], "strict": true}
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "reasoning": {"effort": "medium", "summary": "auto"},
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "temperature": 0.2,
            "max_output_tokens": 4096,
            "metadata": {"user_id": "user_abc"}
        });
        let core = decode_request(&body).expect("decode");
        assert_eq!(core.model.requested, "gpt-5.1-codex");
        assert!(core.stream);
        assert_eq!(
            core.system,
            vec![CoreContent::Text {
                text: "You are a coding agent.".into(),
                cache: None
            }]
        );
        assert_eq!(core.sampling.reasoning_effort.as_deref(), Some("medium"));
        assert_eq!(core.sampling.temperature, Some(0.2));
        assert_eq!(core.sampling.max_tokens, Some(4096));
        assert_eq!(core.metadata.user_id.as_deref(), Some("user_abc"));
        // Unconsumed top-level fields are captured for the policy, not dropped.
        for carried in ["store", "include", "parallel_tool_calls"] {
            assert!(
                core.metadata.raw.contains_key(carried),
                "{carried} must survive into metadata.raw"
            );
        }
        // The conversation: user text, assistant tool_use, tool result, replayed reasoning.
        assert_eq!(core.messages.len(), 4);
        assert!(matches!(
            core.messages[0].content[0],
            CoreContent::Text { .. }
        ));
        match &core.messages[1].content[0] {
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "shell");
                assert_eq!(input["cmd"], "ls");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &core.messages[2].content[0] {
            CoreContent::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                assert_eq!(tool_use_id, "call_1");
                assert_eq!(content.len(), 1);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        match &core.messages[3].content[0] {
            CoreContent::Thinking { text, signature } => {
                assert_eq!(text, "run ls");
                assert_eq!(signature.as_deref(), Some("ENC"));
            }
            other => panic!("expected Thinking, got {other:?}"),
        }
        assert_eq!(core.tools.len(), 1);
        assert_eq!(core.tools[0].name, "shell");
        assert!(matches!(core.tool_choice, Some(CoreToolChoice::Auto)));
    }

    #[test]
    fn decode_rejects_unknown_item_and_part_types_by_name() {
        let unknown_item = decode_request(&serde_json::json!({
            "model": "m", "input": [{"type": "web_search_call", "id": "ws_1"}]
        }))
        .expect_err("unknown item type must fail");
        assert!(
            unknown_item.to_string().contains("web_search_call"),
            "{unknown_item}"
        );
        let unknown_part = decode_request(&serde_json::json!({
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]
        }))
        .expect_err("unknown part type must fail");
        assert!(
            unknown_part.to_string().contains("input_file"),
            "{unknown_part}"
        );
    }

    #[test]
    fn non_function_tools_stay_out_of_the_core_list() {
        let core = decode_request(&serde_json::json!({
            "model": "m",
            "input": "hi",
            "tools": [
                {"type": "web_search", "name": "web_search"},
                {"type": "function", "name": "shell", "parameters": {"type": "object"}}
            ]
        }))
        .expect("decode");
        assert_eq!(
            core.tools.len(),
            1,
            "server tools must not be mis-encoded as functions: {:?}",
            core.tools
        );
        assert_eq!(core.tools[0].name, "shell");
    }

    #[test]
    fn wire_input_tokens_include_the_cached_subset() {
        // A same-format Responses hop: the provider decoder normalized
        // input_tokens 21 / cached 8 into core 13 + cache_read 8; the client
        // encode must restore the wire-exact 21 with cached_tokens 8.
        let usage = Usage {
            input_tokens: 13,
            output_tokens: 6,
            reasoning_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(8),
            provenance: UsageProvenance::ProviderReported,
        };
        let encoded = encode_usage(&usage);
        assert_eq!(encoded["input_tokens"], 21);
        assert_eq!(encoded["input_tokens_details"]["cached_tokens"], 8);
        assert_eq!(encoded["total_tokens"], 27);
    }

    #[test]
    fn response_usage_fields_are_always_explicit() {
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
            reasoning_tokens: None,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
            provenance: UsageProvenance::ProviderReported,
        };
        let encoded = encode_usage(&usage);
        assert_eq!(encoded["input_tokens_details"]["cached_tokens"], 0);
        assert_eq!(encoded["output_tokens_details"]["reasoning_tokens"], 0);
        assert_eq!(encoded["total_tokens"], 15);
        let text = serde_json::to_string(&encoded).expect("serialize");
        assert!(
            text.contains("\"cached_tokens\":0"),
            "a zero cached_tokens must be present, not omitted: {text}"
        );
    }

    #[test]
    fn encode_response_maps_blocks_to_output_items() {
        let resp = CoreResponse {
            id: Some("resp_9".into()),
            model: ModelRef {
                requested: "gpt-5.1-codex".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Text {
                    text: "running".into(),
                    cache: None,
                },
                CoreContent::ToolUse {
                    id: "call_2".into(),
                    name: "shell".into(),
                    input: serde_json::json!({"cmd": "ls"}),
                    signature: None,
                },
            ],
            stop_reason: StopReason::ToolUse,
            stop_sequence: None,
            usage: Usage::provider_reported(3, 4),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let value = encode_response(&resp).expect("encode");
        assert_eq!(value["status"], "completed");
        assert_eq!(value["output"][0]["type"], "message");
        assert_eq!(value["output"][0]["content"][0]["text"], "running");
        assert_eq!(value["output"][1]["type"], "function_call");
        assert_eq!(value["output"][1]["call_id"], "call_2");
        assert_eq!(value["output"][1]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(value["usage"]["input_tokens"], 3);
    }

    #[test]
    fn max_tokens_maps_to_an_incomplete_response() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "m".into(),
                upstream: None,
            },
            content: vec![CoreContent::Text {
                text: "half".into(),
                cache: None,
            }],
            stop_reason: StopReason::MaxTokens,
            stop_sequence: None,
            usage: Usage::synthetic_zero(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let value = encode_response(&resp).expect("encode");
        assert_eq!(value["status"], "incomplete");
        assert_eq!(value["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn stream_terminal_reports_the_same_incomplete_reason_as_the_response() {
        for (stop_reason, reason) in [
            (StopReason::MaxTokens, "max_output_tokens"),
            (StopReason::Refusal, "content_filter"),
        ] {
            let resp = CoreResponse {
                id: None,
                model: ModelRef {
                    requested: "m".into(),
                    upstream: None,
                },
                content: vec![CoreContent::Text {
                    text: "partial".into(),
                    cache: None,
                }],
                stop_reason: stop_reason.clone(),
                stop_sequence: None,
                usage: Usage::synthetic_zero(),
                provider_meta: serde_json::Map::new(),
                cost: None,
            };
            let non_streaming = encode_response(&resp).expect("encode");
            let mut encoder = StreamEncoder::new("resp_1".into(), "m".into());
            let events = stream_events(
                &mut encoder,
                vec![
                    CoreEvent::MessageStart {
                        id: Some("resp_1".into()),
                        model: ModelRef {
                            requested: "m".into(),
                            upstream: None,
                        },
                    },
                    CoreEvent::ContentStart {
                        index: 0,
                        kind: ContentKind::Text,
                    },
                    CoreEvent::TextDelta {
                        index: 0,
                        text: "partial".into(),
                    },
                    CoreEvent::MessageStop {
                        stop_reason: stop_reason.clone(),
                        stop_sequence: None,
                    },
                ],
            );
            let terminal = events.last().cloned().unwrap_or_default();
            assert_eq!(terminal["type"], "response.incomplete", "{stop_reason:?}");
            assert_eq!(
                terminal["response"]["incomplete_details"]["reason"], reason,
                "{stop_reason:?}"
            );
            assert_eq!(
                terminal["response"]["incomplete_details"], non_streaming["incomplete_details"],
                "{stop_reason:?}: streaming and non-streaming must agree"
            );
        }
    }

    #[test]
    fn stream_event_order_follows_the_documented_sequence() {
        let mut encoder = StreamEncoder::new("resp_1".into(), "m".into());
        let events = stream_events(
            &mut encoder,
            vec![
                CoreEvent::MessageStart {
                    id: Some("resp_1".into()),
                    model: ModelRef {
                        requested: "m".into(),
                        upstream: None,
                    },
                },
                CoreEvent::ContentStart {
                    index: 0,
                    kind: ContentKind::Text,
                },
                CoreEvent::TextDelta {
                    index: 0,
                    text: "he".into(),
                },
                CoreEvent::TextDelta {
                    index: 0,
                    text: "llo".into(),
                },
                CoreEvent::ContentStart {
                    index: 1,
                    kind: ContentKind::ToolUse,
                },
                CoreEvent::ToolCallStart {
                    index: 1,
                    id: "call_9".into(),
                    name: "shell".into(),
                    signature: None,
                },
                CoreEvent::ToolCallDelta {
                    index: 1,
                    args_delta: "{\"cmd\":".into(),
                },
                CoreEvent::ToolCallDelta {
                    index: 1,
                    args_delta: "\"ls\"}".into(),
                },
                CoreEvent::ToolCallStop { index: 1 },
                CoreEvent::UsageDelta {
                    usage: Usage::provider_reported(7, 8),
                },
                CoreEvent::MessageStop {
                    stop_reason: StopReason::ToolUse,
                    stop_sequence: None,
                },
            ],
        );
        let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
        let expected = [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            // Opening the tool item closes the text item first.
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ];
        assert_eq!(types, expected, "event order: {events:#?}");
        // The completed event embeds exact usage.
        let completed = events.last().cloned().unwrap_or_default();
        assert_eq!(completed["response"]["usage"]["input_tokens"], 7);
        assert_eq!(completed["response"]["usage"]["output_tokens"], 8);
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            0
        );
        // The function_call item is complete in the terminal response.
        assert_eq!(
            completed["response"]["output"][1]["arguments"],
            "{\"cmd\":\"ls\"}"
        );
        // Sequence numbers strictly increase.
        let seqs: Vec<u64> = events
            .iter()
            .filter_map(|e| e["sequence_number"].as_u64())
            .collect();
        let mut sorted = seqs.clone();
        sorted.sort_unstable();
        assert_eq!(seqs, sorted, "sequence numbers must increase");
    }

    #[test]
    fn stream_error_emits_response_failed_with_the_message() {
        let mut encoder = StreamEncoder::new("resp_1".into(), "m".into());
        let error = crate::core::CoreStreamError::new(
            crate::core::CoreStreamErrorKind::Upstream,
            "upstream exploded".to_owned(),
        );
        let events = stream_events(&mut encoder, vec![CoreEvent::Error { error }]);
        let types: Vec<&str> = events.iter().filter_map(|e| e["type"].as_str()).collect();
        assert_eq!(types, vec!["response.failed"]);
        assert_eq!(
            events[0]["response"]["error"]["message"],
            "upstream exploded"
        );
        assert_eq!(events[0]["response"]["status"], "failed");
    }

    #[test]
    fn finish_without_a_terminal_synthesizes_completed() {
        let mut encoder = StreamEncoder::new("resp_1".into(), "m".into());
        let _ = stream_events(
            &mut encoder,
            vec![CoreEvent::ContentStart {
                index: 0,
                kind: ContentKind::Text,
            }],
        );
        let events = encoder.finish().expect("finish");
        let types: Vec<&str> = events
            .iter()
            .map(|event| event.event_type.as_str())
            .collect();
        assert!(types.contains(&"response.completed"), "{types:?}");
    }
}
