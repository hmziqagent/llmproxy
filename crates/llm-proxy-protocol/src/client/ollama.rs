//! Ollama native client protocol adapter (plan 08, step 3): translates
//! between the Ollama wire formats (`/api/chat`, `/api/generate`) and the
//! normalised core protocol types, plus the `/api/tags` model-list shape.
//!
//! Wire shapes follow the Ollama API reference
//! (<https://github.com/ollama/ollama/blob/main/docs/api.md>): NDJSON
//! streaming (one JSON object per line, no SSE framing), `message.content`
//! deltas for `/api/chat`, `response` deltas for `/api/generate`, `tools`
//! with `{name, description, parameters}`, sampling knobs under `options`,
//! and usage as `prompt_eval_count`/`eval_count` on the final object.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::client::ProtocolError;
use crate::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreTool, ModelRef,
    RequestMetadata, SamplingOptions, StopReason, Usage, json_type_name,
};

/// Top-level request keys the `/api/chat` decode consumes.
pub const OLLAMA_CHAT_REQUEST_KEYS: &[&str] = &[
    "model", "messages", "stream", "tools", "options", "format", "think",
];

/// Top-level request keys the `/api/generate` decode consumes.
pub const OLLAMA_GENERATE_REQUEST_KEYS: &[&str] = &[
    "model", "prompt", "system", "stream", "options", "format", "think",
];

/// Option keys the decode maps onto core sampling; any other `options` key is
/// an engine knob with no cross-format equivalent and is rejected by name.
const MAPPED_OPTION_KEYS: &[&str] = &["temperature", "top_p", "top_k", "num_predict", "stop"];

fn decode_options(options: Option<&Value>) -> Result<SamplingOptions, ProtocolError> {
    let mut sampling = SamplingOptions::default();
    let Some(obj) = options.and_then(Value::as_object) else {
        return Ok(sampling);
    };
    if let Some(temp) = obj.get("temperature") {
        sampling.temperature = as_finite(temp, "options.temperature")?;
    }
    if let Some(top_p) = obj.get("top_p") {
        sampling.top_p = as_finite(top_p, "options.top_p")?;
    }
    sampling.top_k = obj
        .get("top_k")
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok());
    sampling.max_tokens = obj
        .get("num_predict")
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok());
    if let Some(Value::Array(stops)) = obj.get("stop") {
        let mut decoded = Vec::new();
        for stop in stops {
            let Some(s) = stop.as_str() else {
                return Err(ProtocolError::InvalidRequest(
                    "options.stop entries must be strings".into(),
                ));
            };
            decoded.push(s.to_owned());
        }
        if !decoded.is_empty() {
            sampling.stop = Some(decoded);
        }
    }
    Ok(sampling)
}

fn as_finite(value: &Value, field: &str) -> Result<Option<f64>, ProtocolError> {
    match value {
        Value::Null => Ok(None),
        Value::Number(n) => n
            .as_f64()
            .filter(|v| v.is_finite())
            .map(Some)
            .ok_or_else(|| ProtocolError::InvalidRequest(format!("{field} must be finite"))),
        other => Err(ProtocolError::InvalidRequest(format!(
            "{field} must be a number, found {}",
            json_type_name(other)
        ))),
    }
}

/// Reject unmapped `options` keys by name (nothing is lost silently); the
/// known-mapped keys are consumed by [`decode_options`].
fn check_unmapped_options(options: Option<&Value>) -> Result<(), ProtocolError> {
    let Some(obj) = options.and_then(Value::as_object) else {
        return Ok(());
    };
    let unmapped: Vec<&str> = obj
        .keys()
        .map(String::as_str)
        .filter(|key| !MAPPED_OPTION_KEYS.contains(key))
        .collect();
    if unmapped.is_empty() {
        return Ok(());
    }
    Err(ProtocolError::InvalidRequest(format!(
        "options key(s) {} have no equivalent outside the Ollama engine; remove them or pick a \
         provider that accepts them",
        unmapped.join(", "),
    )))
}

fn decode_think(value: Option<&Value>) -> Result<Option<String>, ProtocolError> {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => Ok(None),
        Some(Value::Bool(true)) => Ok(Some("medium".to_owned())),
        Some(Value::String(s)) if matches!(s.as_str(), "low" | "medium" | "high") => {
            Ok(Some(s.clone()))
        }
        Some(other) => Err(ProtocolError::InvalidRequest(format!(
            "think must be a boolean or \"low\"/\"medium\"/\"high\", found {}",
            json_type_name(other)
        ))),
    }
}

fn decode_tools(tools: Option<&Value>) -> Result<Vec<CoreTool>, ProtocolError> {
    let Some(Value::Array(items)) = tools else {
        return Ok(Vec::new());
    };
    let mut decoded = Vec::new();
    for tool in items {
        let obj = tool
            .as_object()
            .ok_or_else(|| ProtocolError::InvalidRequest("tools must be objects".into()))?;
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .ok_or_else(|| {
                ProtocolError::InvalidRequest("tools entries need a non-empty name".into())
            })?
            .to_owned();
        decoded.push(CoreTool {
            name,
            description: obj
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_owned),
            input_schema: obj
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
        });
    }
    Ok(decoded)
}

fn decode_messages(messages: &Value) -> Result<Vec<CoreMessage>, ProtocolError> {
    let Value::Array(items) = messages else {
        return Err(ProtocolError::InvalidRequest(format!(
            "messages must be an array, found {}",
            json_type_name(messages)
        )));
    };
    let mut decoded = Vec::new();
    for (message_index, message) in items.iter().enumerate() {
        let obj = message
            .as_object()
            .ok_or_else(|| ProtocolError::InvalidRequest("messages must be objects".into()))?;
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
        let role = match role {
            "user" => CoreRole::User,
            "assistant" => CoreRole::Assistant,
            "system" => CoreRole::System,
            "tool" => CoreRole::Tool,
            other => {
                return Err(ProtocolError::InvalidRequest(format!(
                    "unknown message role: {other}"
                )));
            }
        };
        // A `tool` role message is the tool result for the n-th outstanding
        // call; Ollama correlates by order/name, not id.
        if role == CoreRole::Tool {
            let tool_name = obj
                .get("tool_name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let text = obj.get("content").and_then(Value::as_str).unwrap_or("");
            decoded.push(CoreMessage {
                role: CoreRole::Tool,
                content: vec![CoreContent::ToolResult {
                    tool_use_id: tool_name,
                    content: if text.is_empty() {
                        Vec::new()
                    } else {
                        vec![CoreContent::Text {
                            text: text.to_owned(),
                            cache: None,
                        }]
                    },
                    is_error: false,
                    cache: None,
                }],
            });
            continue;
        }
        let content = decode_message_content(obj, message_index)?;
        decoded.push(CoreMessage { role, content });
    }
    // Ollama correlates tool results by order/name, not id: match each result
    // to the earliest unmatched call with that name.
    super::correlate_tool_results_by_name(&mut decoded);
    Ok(decoded)
}

/// The content blocks of the user/assistant message at `message_index`:
/// text, `images` and `tool_calls` in order.
fn decode_message_content(
    obj: &Map<String, Value>,
    message_index: usize,
) -> Result<Vec<CoreContent>, ProtocolError> {
    let mut content = Vec::new();
    if let Some(text) = obj.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        content.push(CoreContent::Text {
            text: text.to_owned(),
            cache: None,
        });
    }
    if let Some(Value::Array(images)) = obj.get("images") {
        for image in images {
            let Some(data) = image.as_str() else {
                return Err(ProtocolError::InvalidRequest(
                    "images entries must be base64 strings".into(),
                ));
            };
            // Ollama sends bare base64 with no MIME hint: label the image by
            // its file signature. An unrecognised signature keeps the PNG
            // label (the upstream validates the bytes), with a warning.
            let media_type = sniff_image_media_type(data).unwrap_or_else(|| {
                tracing::warn!("Ollama: unrecognised image signature; labelling it image/png");
                "image/png"
            });
            content.push(CoreContent::Image {
                source: serde_json::json!({
                    "type": "base64", "media_type": media_type, "data": data,
                }),
            });
        }
    }
    if let Some(Value::Array(calls)) = obj.get("tool_calls") {
        for (call_index, call) in calls.iter().enumerate() {
            let function = call
                .as_object()
                .and_then(|c| c.get("function"))
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    ProtocolError::InvalidRequest(
                        "tool_calls entries need a function object".into(),
                    )
                })?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| {
                    ProtocolError::InvalidRequest(
                        "tool_calls function needs a non-empty name".into(),
                    )
                })?
                .to_owned();
            let input = function
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            // Ollama tool calls carry no id; the message and call positions
            // give a synthetic id that is unique across the request (and
            // stable as the conversation grows), so adapters can correlate
            // results to calls.
            let id = format!("ollama-call-{message_index}-{call_index}");
            content.push(CoreContent::ToolUse {
                id,
                name,
                input,
                signature: None,
            });
        }
    }
    Ok(content)
}

/// The media type of base64 image data, read from its file signature.
/// Covers the formats the cross-format upstreams accept; `None` otherwise.
fn sniff_image_media_type(data: &str) -> Option<&'static str> {
    match decode_base64_prefix(data)? {
        [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, ..] => Some("image/png"),
        [0xFF, 0xD8, 0xFF, ..] => Some("image/jpeg"),
        [b'G', b'I', b'F', b'8', b'7' | b'9', b'a', ..] => Some("image/gif"),
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P'] => Some("image/webp"),
        _ => None,
    }
}

/// The first 12 bytes of base64 `data`, decoded from its first 16
/// characters: enough for a file signature, without a base64 dependency.
fn decode_base64_prefix(data: &str) -> Option<[u8; 12]> {
    let mut head = [0u8; 12];
    let chars = data.as_bytes().get(..16)?;
    for (quad, out) in chars.chunks_exact(4).zip(head.chunks_exact_mut(3)) {
        let mut word = 0u32;
        for &c in quad {
            word = (word << 6) | base64_sextet(c)?;
        }
        let [_, b0, b1, b2] = word.to_be_bytes();
        for (slot, byte) in out.iter_mut().zip([b0, b1, b2]) {
            *slot = byte;
        }
    }
    Some(head)
}

/// The 6-bit value of one base64 character (standard or URL-safe alphabet).
fn base64_sextet(c: u8) -> Option<u32> {
    let value = match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' | b'-' => 62,
        b'/' | b'_' => 63,
        _ => return None,
    };
    Some(u32::from(value))
}

/// Decode a raw `/api/chat` request body into a normalised [`CoreRequest`].
pub fn decode_chat_request(body: &Value) -> Result<CoreRequest, ProtocolError> {
    let obj = body.as_object().ok_or_else(|| {
        ProtocolError::InvalidRequest("request body must be a JSON object".into())
    })?;
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ProtocolError::InvalidRequest("model is required".into()))?;
    let messages = obj
        .get("messages")
        .ok_or_else(|| ProtocolError::InvalidRequest("messages is required".into()))?;
    let messages = decode_messages(messages)?;
    let mut sampling = decode_options(obj.get("options"))?;
    check_unmapped_options(obj.get("options"))?;
    sampling.reasoning_effort = decode_think(obj.get("think"))?;

    // `format` requests structured output; a JSON-schema object maps onto the
    // chat family's response_format (carried in metadata for the adapters).
    let mut raw = Map::new();
    for (key, value) in obj {
        if !OLLAMA_CHAT_REQUEST_KEYS.contains(&key.as_str()) {
            raw.insert(key.clone(), value.clone());
        }
    }

    Ok(CoreRequest {
        model: ModelRef {
            requested: model.to_owned(),
            upstream: None,
        },
        system: Vec::new(),
        messages,
        tools: decode_tools(obj.get("tools"))?,
        // Ollama's tool semantics are auto-only; there is no tool_choice knob.
        tool_choice: None,
        sampling,
        stream: obj.get("stream").and_then(Value::as_bool).unwrap_or(false),
        metadata: RequestMetadata { user_id: None, raw },
        provider_hints: Default::default(),
    })
}

/// Decode a raw `/api/generate` request body into a normalised
/// [`CoreRequest`]: the prompt becomes one user turn, `system` the system
/// prompt.
pub fn decode_generate_request(body: &Value) -> Result<CoreRequest, ProtocolError> {
    let obj = body.as_object().ok_or_else(|| {
        ProtocolError::InvalidRequest("request body must be a JSON object".into())
    })?;
    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ProtocolError::InvalidRequest("model is required".into()))?;
    let prompt = obj
        .get("prompt")
        .and_then(Value::as_str)
        .ok_or_else(|| ProtocolError::InvalidRequest("prompt is required".into()))?;
    let mut sampling = decode_options(obj.get("options"))?;
    check_unmapped_options(obj.get("options"))?;
    sampling.reasoning_effort = decode_think(obj.get("think"))?;

    let mut raw = Map::new();
    for (key, value) in obj {
        if !OLLAMA_GENERATE_REQUEST_KEYS.contains(&key.as_str()) {
            raw.insert(key.clone(), value.clone());
        }
    }

    let mut system = Vec::new();
    if let Some(system_text) = obj.get("system").and_then(Value::as_str)
        && !system_text.is_empty()
    {
        system.push(CoreContent::Text {
            text: system_text.to_owned(),
            cache: None,
        });
    }

    Ok(CoreRequest {
        model: ModelRef {
            requested: model.to_owned(),
            upstream: None,
        },
        system,
        messages: vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: prompt.to_owned(),
                cache: None,
            }],
        }],
        tools: Vec::new(),
        tool_choice: None,
        sampling,
        stream: obj.get("stream").and_then(Value::as_bool).unwrap_or(false),
        metadata: RequestMetadata { user_id: None, raw },
        provider_hints: Default::default(),
    })
}

fn done_reason(stop: &StopReason) -> &'static str {
    match stop {
        StopReason::MaxTokens => "length",
        _ => "stop",
    }
}

fn created_at() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format_unix_millis(now.as_secs() as i64, now.subsec_millis())
}

/// RFC 3339 UTC with millisecond precision (the shape Ollama emits) for a
/// unix timestamp, via the civil-from-days algorithm (Hinnant,
/// <https://howardhinnant.github.io/date_algorithms.html>); this crate has no
/// date dependency and the input is our own clock.
fn format_unix_millis(secs: i64, millis: u32) -> String {
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    let month = u32::try_from(m).unwrap_or(1);
    let day = u32::try_from(d).unwrap_or(1);
    (year, month, day)
}

fn usage_counts(usage: &Usage) -> Value {
    serde_json::json!({
        "prompt_eval_count": usage.input_tokens,
        "eval_count": usage.output_tokens,
    })
}

/// Encode a [`CoreResponse`] as an `/api/chat` non-streaming response.
pub fn encode_chat_response(resp: &CoreResponse) -> Result<Value, ProtocolError> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls = Vec::new();
    for block in &resp.content {
        match block {
            CoreContent::Text { text: t, .. } => text.push_str(t),
            CoreContent::Thinking { text: t, .. } => thinking.push_str(t),
            CoreContent::ToolUse { name, input, .. } => {
                tool_calls.push(serde_json::json!({
                    "function": {"name": name, "arguments": input},
                }));
            }
            CoreContent::Refusal { text: t } => text.push_str(t),
            other => {
                tracing::warn!(kind = ?other, "Ollama chat responses cannot carry this block; dropping");
            }
        }
    }
    if !thinking.is_empty() {
        tracing::warn!(
            "Ollama /api/chat responses have no thinking field unless think was enabled; including it in content"
        );
        text = format!("{thinking}\n{text}");
    }
    let mut message = serde_json::json!({"role": "assistant", "content": text});
    if !tool_calls.is_empty()
        && let Some(obj) = message.as_object_mut()
    {
        obj.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    let mut response = serde_json::json!({
        "model": resp.model.requested,
        "created_at": created_at(),
        "message": message,
        "done": true,
        "done_reason": done_reason(&resp.stop_reason),
    });
    if let Some(obj) = response.as_object_mut() {
        for (key, value) in usage_counts(&resp.usage)
            .as_object()
            .cloned()
            .unwrap_or_default()
        {
            obj.insert(key, value);
        }
    }
    Ok(response)
}

/// Encode a [`CoreResponse`] as an `/api/generate` non-streaming response.
pub fn encode_generate_response(resp: &CoreResponse) -> Result<Value, ProtocolError> {
    let mut text = String::new();
    let mut thinking = String::new();
    for block in &resp.content {
        match block {
            CoreContent::Text { text: t, .. } => text.push_str(t),
            CoreContent::Thinking { text: t, .. } => thinking.push_str(t),
            CoreContent::Refusal { text: t } => text.push_str(t),
            other => {
                tracing::warn!(
                    kind = ?other,
                    "Ollama /api/generate responses cannot carry this block; dropping"
                );
            }
        }
    }
    if !thinking.is_empty() {
        text = format!("{thinking}\n{text}");
    }
    let mut response = serde_json::json!({
        "model": resp.model.requested,
        "created_at": created_at(),
        "response": text,
        "done": true,
        "done_reason": done_reason(&resp.stop_reason),
    });
    if let Some(obj) = response.as_object_mut() {
        for (key, value) in usage_counts(&resp.usage)
            .as_object()
            .cloned()
            .unwrap_or_default()
        {
            obj.insert(key, value);
        }
    }
    Ok(response)
}

/// The `/api/tags` model list over the proxy catalog's model ids.
pub fn encode_tags_response(models: &[String]) -> Value {
    serde_json::json!({
        "models": models
            .iter()
            .map(|id| serde_json::json!({"name": id, "model": id}))
            .collect::<Vec<_>>(),
    })
}

/// The Ollama error body shape (`{"error": "<message>"}`).
pub fn error_payload(message: &str) -> Value {
    serde_json::json!({"error": message})
}

/// Stateful encoder that converts [`CoreEvent`] values into NDJSON stream
/// objects (chat or generate framing).
#[derive(Debug)]
pub struct StreamEncoder {
    model: Arc<str>,
    generate: bool,
    /// Open tool calls keyed by core content index; Ollama emits complete
    /// function objects, so arguments buffer until `ToolCallStop`.
    open_tool_calls: std::collections::HashMap<usize, (String, String)>,
    pending_usage: Option<Usage>,
    pending_stop: Option<StopReason>,
    finished: bool,
}

impl StreamEncoder {
    /// Create a stream encoder; `generate` selects the `/api/generate`
    /// framing (`response` deltas) over `/api/chat` (`message.content`).
    pub fn new(model: String, generate: bool) -> Self {
        Self {
            model: model.into(),
            generate,
            open_tool_calls: std::collections::HashMap::new(),
            pending_usage: None,
            pending_stop: None,
            finished: false,
        }
    }

    fn delta_object(&self, delta: &str) -> Value {
        if self.generate {
            serde_json::json!({
                "model": self.model.to_string(),
                "created_at": created_at(),
                "response": delta,
                "done": false,
            })
        } else {
            serde_json::json!({
                "model": self.model.to_string(),
                "created_at": created_at(),
                "message": {"role": "assistant", "content": delta},
                "done": false,
            })
        }
    }

    /// Encode a single [`CoreEvent`] into zero or more NDJSON objects.
    pub fn encode_event(&mut self, event: CoreEvent) -> Result<Vec<Value>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut chunks = Vec::new();
        match event {
            CoreEvent::MessageStart { .. } | CoreEvent::ContentStart { .. } => {}
            CoreEvent::TextDelta { text, .. } => {
                chunks.push(self.delta_object(&text));
            }
            CoreEvent::ThinkingDelta { text, .. } => {
                chunks.push(self.delta_object(&text));
            }
            CoreEvent::ToolCallStart { index, name, .. } => {
                self.open_tool_calls.insert(index, (name, String::new()));
            }
            CoreEvent::ToolCallDelta { index, args_delta } => {
                if let Some((_, arguments)) = self.open_tool_calls.get_mut(&index) {
                    arguments.push_str(&args_delta);
                } else {
                    tracing::warn!(
                        core_index = index,
                        "Ollama: ToolCallDelta without an open tool call; dropping"
                    );
                }
            }
            CoreEvent::ToolCallStop { index } => {
                if let Some((name, arguments)) = self.open_tool_calls.remove(&index) {
                    let input: Value = serde_json::from_str(&arguments).unwrap_or_else(|_| {
                        tracing::warn!(
                            arguments = %arguments,
                            "Ollama: malformed buffered tool arguments; sending an empty object"
                        );
                        Value::Object(Map::new())
                    });
                    let mut object = serde_json::json!({
                        "model": self.model.to_string(),
                        "created_at": created_at(),
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"function": {"name": name, "arguments": input}}],
                        },
                        "done": false,
                    });
                    if self.generate {
                        // /api/generate has no tool-call channel; the call is
                        // rendered as text so nothing vanishes.
                        object = serde_json::json!({
                            "model": self.model.to_string(),
                            "created_at": created_at(),
                            "response": format!("{name}({input})"),
                            "done": false,
                        });
                    }
                    chunks.push(object);
                }
            }
            CoreEvent::UsageDelta { usage } => {
                self.pending_usage = Some(usage);
            }
            CoreEvent::MessageStop { stop_reason, .. } => {
                self.pending_stop = Some(stop_reason);
                chunks.extend(self.terminal_objects());
            }
            CoreEvent::Error { .. } => {
                // Rendered by the route as an Ollama error object; nothing to
                // synthesize here.
                self.finished = true;
            }
            CoreEvent::Ping => {}
        }
        Ok(chunks)
    }

    fn terminal_objects(&mut self) -> Vec<Value> {
        self.finished = true;
        let stop = self.pending_stop.clone().unwrap_or_else(|| {
            tracing::warn!("Ollama: stream ended without a MessageStop; synthesizing stop");
            StopReason::EndTurn
        });
        let usage = self
            .pending_usage
            .clone()
            .unwrap_or_else(Usage::synthetic_zero);
        let mut object = if self.generate {
            serde_json::json!({
                "model": self.model.to_string(),
                "created_at": created_at(),
                "response": "",
                "done": true,
                "done_reason": done_reason(&stop),
            })
        } else {
            serde_json::json!({
                "model": self.model.to_string(),
                "created_at": created_at(),
                "message": {"role": "assistant", "content": ""},
                "done": true,
                "done_reason": done_reason(&stop),
            })
        };
        if let Some(obj) = object.as_object_mut() {
            for (key, value) in usage_counts(&usage)
                .as_object()
                .cloned()
                .unwrap_or_default()
            {
                obj.insert(key, value);
            }
        }
        vec![object]
    }

    /// Flush at the natural end of the stream.
    pub fn finish(&mut self) -> Result<Vec<Value>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        Ok(self.terminal_objects())
    }

    /// Mark the encoder as finished so subsequent `finish()` calls return
    /// empty; used after a terminal error object has been emitted.
    pub fn mark_finished(&mut self) {
        self.finished = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core_response(content: Vec<CoreContent>, stop: StopReason) -> CoreResponse {
        CoreResponse {
            id: None,
            model: ModelRef {
                requested: "llama3".into(),
                upstream: None,
            },
            content,
            stop_reason: stop,
            stop_sequence: None,
            usage: Usage::provider_reported(11, 7),
            provider_meta: serde_json::Map::new(),
            cost: None,
        }
    }

    #[test]
    fn chat_decode_maps_messages_tools_and_options() {
        let body = serde_json::json!({
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "weather?", "images": ["aGVsbG8="]},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "SF"}}}
                ]},
                {"role": "tool", "content": "72F", "tool_name": "get_weather"}
            ],
            "tools": [{"name": "get_weather", "parameters": {"type": "object"}}],
            "stream": true,
            "options": {"temperature": 0.4, "top_k": 5, "num_predict": 100, "stop": ["END"]},
            "think": "low",
            "keep_alive": "5m"
        });
        let core = decode_chat_request(&body).expect("decode");
        assert_eq!(core.model.requested, "llama3.1");
        assert!(core.stream);
        assert_eq!(core.messages.len(), 4);
        assert_eq!(core.sampling.temperature, Some(0.4));
        assert_eq!(core.sampling.top_k, Some(5));
        assert_eq!(core.sampling.max_tokens, Some(100));
        assert_eq!(core.sampling.stop.as_deref(), Some(&["END".to_owned()][..]));
        assert_eq!(core.sampling.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(core.tools.len(), 1);
        assert!(core.metadata.raw.contains_key("keep_alive"));
        match &core.messages[2].content[0] {
            CoreContent::ToolUse { name, input, .. } => {
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &core.messages[3].content[0] {
            CoreContent::ToolResult { tool_use_id, .. } => {
                // The result is correlated to the call by tool name.
                assert_eq!(tool_use_id, "ollama-call-2-0");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn parallel_tool_calls_in_one_message_get_distinct_ids() {
        // Two calls of the same tool in one assistant turn, answered in
        // order: each call needs its own id (upstreams reject or mis-pair
        // duplicates) and each result must land on its own call.
        let body = serde_json::json!({
            "model": "llama3.1",
            "messages": [
                {"role": "user", "content": "weather in SF and NYC?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "SF"}}},
                    {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
                ]},
                {"role": "tool", "content": "72F", "tool_name": "get_weather"},
                {"role": "tool", "content": "60F", "tool_name": "get_weather"}
            ]
        });
        let core = decode_chat_request(&body).expect("decode");
        let call_ids: Vec<&str> = core
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|block| match block {
                CoreContent::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let result_ids: Vec<&str> = core
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|block| match block {
                CoreContent::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(call_ids.len(), 2, "{call_ids:?}");
        assert_ne!(call_ids[0], call_ids[1], "each call needs its own id");
        assert_eq!(result_ids, call_ids, "results answer the calls in order");
    }

    #[test]
    fn images_are_labelled_by_their_file_signature() {
        // Ollama `images` are bare base64; upstreams that validate media
        // types reject (or mis-decode) a JPEG labelled image/png. Each data
        // string is the base64 of a real file header.
        for (data, media_type) in [
            ("iVBORw0KGgoAAAANSUhEUg==", "image/png"),
            ("/9j/4AAQSkZJRgABAQAAAQ==", "image/jpeg"),
            ("R0lGODlhAQABAIAAAAAAAA==", "image/gif"),
            ("UklGRhoAAABXRUJQVlA4TA==", "image/webp"),
        ] {
            let body = serde_json::json!({
                "model": "llava",
                "messages": [{"role": "user", "content": "what is this?", "images": [data]}]
            });
            let core = decode_chat_request(&body).expect("decode");
            match &core.messages[0].content[1] {
                CoreContent::Image { source } => {
                    assert_eq!(source["media_type"], media_type, "{data}");
                    assert_eq!(source["data"], data, "the bytes ride unchanged");
                }
                other => panic!("expected Image, got {other:?}"),
            }
        }
    }

    #[test]
    fn unmapped_option_keys_are_rejected_by_name() {
        let body = serde_json::json!({
            "model": "m", "messages": [], "options": {"num_ctx": 4096, "temperature": 0.1}
        });
        let error = decode_chat_request(&body).expect_err("num_ctx has no equivalent");
        assert!(error.to_string().contains("num_ctx"), "{error}");
    }

    #[test]
    fn generate_decode_maps_prompt_and_system() {
        let body = serde_json::json!({
            "model": "llama3",
            "prompt": "why is the sky blue?",
            "system": "you are an astronomer",
            "stream": false,
            "options": {"temperature": 0.2}
        });
        let core = decode_generate_request(&body).expect("decode");
        assert_eq!(core.messages.len(), 1);
        assert_eq!(core.system.len(), 1);
        match &core.messages[0].content[0] {
            CoreContent::Text { text, .. } => assert_eq!(text, "why is the sky blue?"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn chat_response_encodes_tools_and_usage() {
        let resp = core_response(
            vec![
                CoreContent::Text {
                    text: "checking".into(),
                    cache: None,
                },
                CoreContent::ToolUse {
                    id: "ollama-call-0".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                    signature: None,
                },
            ],
            StopReason::ToolUse,
        );
        let value = encode_chat_response(&resp).expect("encode");
        assert_eq!(value["done"], true);
        assert_eq!(value["message"]["role"], "assistant");
        assert_eq!(value["message"]["content"], "checking");
        assert_eq!(
            value["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(value["prompt_eval_count"], 11);
        assert_eq!(value["eval_count"], 7);
        assert_eq!(value["done_reason"], "stop");
    }

    #[test]
    fn max_tokens_maps_to_length_done_reason() {
        let resp = core_response(
            vec![CoreContent::Text {
                text: "half".into(),
                cache: None,
            }],
            StopReason::MaxTokens,
        );
        let value = encode_generate_response(&resp).expect("encode");
        assert_eq!(value["done_reason"], "length");
        assert_eq!(value["response"], "half");
    }

    #[test]
    fn stream_chat_objects_carry_content_then_terminal_usage() {
        let mut encoder = StreamEncoder::new("llama3".into(), false);
        let mut objects = Vec::new();
        for event in [
            CoreEvent::TextDelta {
                index: 0,
                text: "hi".into(),
            },
            CoreEvent::ToolCallStart {
                index: 1,
                id: "x".into(),
                name: "shell".into(),
                signature: None,
            },
            CoreEvent::ToolCallDelta {
                index: 1,
                args_delta: "{\"cmd\":\"ls\"}".into(),
            },
            CoreEvent::ToolCallStop { index: 1 },
            CoreEvent::UsageDelta {
                usage: Usage::provider_reported(9, 3),
            },
            CoreEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
                stop_sequence: None,
            },
        ] {
            objects.extend(encoder.encode_event(event).expect("encode"));
        }
        assert_eq!(objects.len(), 3, "{objects:#?}");
        assert_eq!(objects[0]["message"]["content"], "hi");
        assert_eq!(objects[0]["done"], false);
        assert_eq!(
            objects[1]["message"]["tool_calls"][0]["function"]["arguments"]["cmd"],
            "ls"
        );
        let terminal = objects.last().cloned().unwrap_or_default();
        assert_eq!(terminal["done"], true);
        assert_eq!(terminal["prompt_eval_count"], 9);
        assert_eq!(terminal["eval_count"], 3);
    }

    #[test]
    fn tags_response_lists_catalog_models() {
        let value = encode_tags_response(&["llama3:latest".to_owned(), "qwen:7b".to_owned()]);
        assert_eq!(value["models"][0]["name"], "llama3:latest");
        assert_eq!(value["models"][1]["model"], "qwen:7b");
    }
}
