//! Gemini native client protocol adapter (plan 08, step 2): translates
//! between the Gemini `generateContent` wire format
//! (`/v1beta/models/{model}:generateContent` and `:streamGenerateContent`)
//! and the normalised core protocol types.
//!
//! Wire shapes follow the Gemini API reference
//! (<https://ai.google.dev/api/generate-content>): camelCase
//! `functionCall`/`functionResponse` parts, `usageMetadata` token counts,
//! `candidates[].finishReason`, and SSE streaming (`?alt=sse`) where each
//! `data:` line is one `GenerateContentResponse` chunk.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::client::ProtocolError;
use crate::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreTool,
    CoreToolChoice, ModelRef, RequestMetadata, SamplingOptions, StopReason, Usage, json_type_name,
};

/// Top-level request keys the Gemini client decode consumes; everything else
/// is an "extra" for the cross-format field policy (and survives same-format
/// passthrough untouched).
pub const GEMINI_REQUEST_KEYS: &[&str] = &[
    "contents",
    "systemInstruction",
    "system_instruction",
    "generationConfig",
    "generation_config",
    "tools",
    "toolConfig",
    "tool_config",
];

fn field<'a>(obj: &'a Map<String, Value>, camel: &str, snake: &str) -> Option<&'a Value> {
    obj.get(camel).or_else(|| obj.get(snake))
}

/// Decode a raw Gemini request body into a normalised [`CoreRequest`]. The
/// model arrives in the URL (`/v1beta/models/{model}:...`), not the body, so
/// the caller passes it in; `stream` likewise comes from which method the
/// client called.
pub fn decode_request(
    body: &Value,
    model: &str,
    stream: bool,
) -> Result<CoreRequest, ProtocolError> {
    if model.is_empty() {
        return Err(ProtocolError::InvalidRequest(
            "the model is required in the URL path".into(),
        ));
    }
    let obj = body.as_object().ok_or_else(|| {
        ProtocolError::InvalidRequest("request body must be a JSON object".into())
    })?;

    let mut system = Vec::new();
    if let Some(instructions) = field(obj, "systemInstruction", "system_instruction") {
        let parts = instructions
            .as_object()
            .and_then(|o| o.get("parts"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut text = String::new();
        for part in parts {
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
        }
        if !text.is_empty() {
            system.push(CoreContent::Text { text, cache: None });
        }
    }

    let mut messages = Vec::new();
    if let Some(contents) = obj.get("contents") {
        let Value::Array(items) = contents else {
            return Err(ProtocolError::InvalidRequest(format!(
                "contents must be an array, found {}",
                json_type_name(contents)
            )));
        };
        if items.is_empty() {
            return Err(ProtocolError::InvalidRequest(
                "contents must not be empty".into(),
            ));
        }
        for (content_index, content) in items.iter().enumerate() {
            let content_obj = content.as_object().ok_or_else(|| {
                ProtocolError::InvalidRequest(format!(
                    "contents entries must be objects, found {}",
                    json_type_name(content)
                ))
            })?;
            let role = content_obj
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user");
            let role = match role {
                "user" => CoreRole::User,
                "model" => CoreRole::Assistant,
                "system" => CoreRole::System,
                other => {
                    return Err(ProtocolError::InvalidRequest(format!(
                        "unknown contents role: {other}"
                    )));
                }
            };
            let parts = content_obj
                .get("parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut blocks = Vec::new();
            for (part_index, part) in parts.iter().enumerate() {
                decode_part(part, content_index, part_index, &mut blocks)?;
            }
            messages.push(CoreMessage {
                role,
                content: blocks,
            });
        }
    }

    let tools = decode_tools(obj.get("tools"))?;
    let tool_choice = decode_tool_config(field(obj, "toolConfig", "tool_config"));
    let sampling = decode_generation_config(field(obj, "generationConfig", "generation_config"))?;

    // Gemini correlates function responses by name, not id: match each
    // result to the earliest unmatched call with that name.
    super::correlate_tool_results_by_name(&mut messages);

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
        metadata: RequestMetadata::default(),
        provider_hints: Default::default(),
    })
}

/// Decode the part at `parts[part_index]` of `contents[content_index]`.
fn decode_part(
    part: &Value,
    content_index: usize,
    part_index: usize,
    blocks: &mut Vec<CoreContent>,
) -> Result<(), ProtocolError> {
    let obj = part.as_object().ok_or_else(|| {
        ProtocolError::InvalidRequest(format!(
            "parts must be objects, found {}",
            json_type_name(part)
        ))
    })?;
    if let Some(text) = obj.get("text").and_then(Value::as_str) {
        blocks.push(CoreContent::Text {
            text: text.to_owned(),
            cache: None,
        });
    }
    if let Some(call) = obj.get("functionCall").and_then(Value::as_object) {
        let name = call
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| ProtocolError::InvalidRequest("functionCall needs a name".into()))?;
        let args = call
            .get("args")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        // Gemini tool calls carry no id; the content and part positions give
        // a synthetic id that is unique across the request (and stable as
        // the conversation grows), so the core and any provider adapter can
        // correlate results to calls.
        let id = format!("gemini_call_{content_index}_{part_index}");
        blocks.push(CoreContent::ToolUse {
            id,
            name: name.to_owned(),
            input: args,
            signature: obj
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .map(str::to_owned),
        });
    }
    if let Some(result) = obj.get("functionResponse").and_then(Value::as_object) {
        let name = result
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let response = result
            .get("response")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        // The result is correlated to its call by name in a second pass (the
        // name stands in as the id until then).
        let tool_use_id = name;
        let text = serde_json::to_string(&response)
            .map_err(|e| ProtocolError::Decode(format!("functionResponse response: {e}")))?;
        blocks.push(CoreContent::ToolResult {
            tool_use_id,
            content: vec![CoreContent::Text { text, cache: None }],
            is_error: false,
            cache: None,
        });
    }
    if let Some(inline) = obj.get("inlineData").and_then(Value::as_object) {
        let mime_type = inline.get("mimeType").and_then(Value::as_str).unwrap_or("");
        let data = inline.get("data").and_then(Value::as_str).unwrap_or("");
        blocks.push(CoreContent::Image {
            source: serde_json::json!({
                "type": "base64", "media_type": mime_type, "data": data,
            }),
        });
    }
    if let Some(file) = obj.get("fileData").and_then(Value::as_object) {
        let mime_type = file.get("mimeType").and_then(Value::as_str).unwrap_or("");
        let uri = file.get("fileUri").and_then(Value::as_str).unwrap_or("");
        blocks.push(CoreContent::Image {
            source: serde_json::json!({"type": "url", "url": uri, "media_type": mime_type}),
        });
    }
    Ok(())
}

fn decode_tools(tools: Option<&Value>) -> Result<Vec<CoreTool>, ProtocolError> {
    let Some(Value::Array(items)) = tools else {
        return Ok(Vec::new());
    };
    let mut decoded = Vec::new();
    for tool in items {
        let Some(declarations) = tool
            .as_object()
            .and_then(|o| o.get("functionDeclarations"))
            .or_else(|| {
                tool.as_object()
                    .and_then(|o| o.get("function_declarations"))
            })
            .and_then(Value::as_array)
        else {
            // googleSearch and codeExecution tools have no function
            // declarations; they stay out of the core list (cross-format
            // policy names them; same-format passthrough carries them).
            let kind = tool
                .as_object()
                .and_then(|o| o.keys().next())
                .cloned()
                .unwrap_or_else(|| "<empty>".to_owned());
            tracing::debug!(
                tool = %kind,
                "Gemini: non-function tool left out of the core tool list"
            );
            continue;
        };
        for declaration in declarations {
            let obj = declaration.as_object().ok_or_else(|| {
                ProtocolError::InvalidRequest("functionDeclarations must be objects".into())
            })?;
            let name = obj
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| {
                    ProtocolError::InvalidRequest(
                        "functionDeclarations entries need a non-empty name".into(),
                    )
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
    }
    Ok(decoded)
}

fn decode_tool_config(value: Option<&Value>) -> Option<CoreToolChoice> {
    let config = value.and_then(Value::as_object)?;
    let calling = config
        .get("functionCallingConfig")
        .or_else(|| config.get("function_calling_config"))?
        .as_object()?;
    let mode = calling.get("mode")?.as_str()?;
    let allowed_names = || {
        calling
            .get("allowedFunctionNames")
            .or_else(|| calling.get("allowed_function_names"))
            .and_then(Value::as_array)
            .map(|names| names.iter().filter_map(Value::as_str).collect::<Vec<_>>())
    };
    match mode {
        "AUTO" => Some(CoreToolChoice::Auto),
        "NONE" => Some(CoreToolChoice::None),
        // With exactly one allowed function, Gemini's semantics are the
        // core's forced named tool; otherwise any tool.
        "ANY" => match allowed_names().as_deref() {
            Some([single]) => Some(CoreToolChoice::Tool {
                name: (*single).to_owned(),
            }),
            _ => Some(CoreToolChoice::Any),
        },
        _ => None,
    }
}

fn decode_generation_config(value: Option<&Value>) -> Result<SamplingOptions, ProtocolError> {
    let mut sampling = SamplingOptions::default();
    let Some(config) = value.and_then(Value::as_object) else {
        return Ok(sampling);
    };
    let num = |key_camel: &str, key_snake: &str| -> Result<Option<f64>, ProtocolError> {
        match field(config, key_camel, key_snake) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => n.as_f64().map(Some).ok_or_else(|| {
                ProtocolError::InvalidRequest(format!("{key_camel} must be a finite number"))
            }),
            Some(other) => Err(ProtocolError::InvalidRequest(format!(
                "{key_camel} must be a number, found {}",
                json_type_name(other)
            ))),
        }
    };
    sampling.temperature = num("temperature", "temperature")?;
    sampling.top_p = num("topP", "top_p")?;
    sampling.top_k = field(config, "topK", "top_k")
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok());
    sampling.max_tokens = field(config, "maxOutputTokens", "max_output_tokens")
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok());
    if let Some(stop) = field(config, "stopSequences", "stop_sequences").and_then(Value::as_array) {
        let mut stops = Vec::new();
        for entry in stop {
            let Some(s) = entry.as_str() else {
                return Err(ProtocolError::InvalidRequest(
                    "stopSequences entries must be strings".into(),
                ));
            };
            stops.push(s.to_owned());
        }
        if !stops.is_empty() {
            sampling.stop = Some(stops);
        }
    }
    if let Some(thinking) =
        field(config, "thinkingConfig", "thinking_config").and_then(Value::as_object)
    {
        // `thinkingBudget` (0 disables, -1 is dynamic); a positive budget
        // maps onto the core thinking hint, which each provider edge
        // translates with its per-model clamps.
        let budget = thinking
            .get("thinkingBudget")
            .or_else(|| thinking.get("thinking_budget"))
            .and_then(Value::as_i64);
        if let Some(budget) = budget {
            if budget > 0 {
                sampling.thinking =
                    Some(serde_json::json!({"type": "enabled", "budget_tokens": budget}));
            }
        }
    }
    Ok(sampling)
}

fn core_to_finish_reason(reason: &StopReason) -> String {
    match reason {
        StopReason::EndTurn
        | StopReason::ToolUse
        | StopReason::StopSequence
        | StopReason::Error => "STOP".to_owned(),
        StopReason::MaxTokens => "MAX_TOKENS".to_owned(),
        StopReason::Refusal => "SAFETY".to_owned(),
        StopReason::Unknown(original) => original.clone(),
    }
}

/// Google documents `promptTokenCount` as the total prompt size including
/// cached content, with `cachedContentTokenCount` naming the cached subset.
/// The core's `input_tokens` excludes cache reads (Anthropic wire semantics),
/// so the cache read is added back here, mirroring the Responses encoder.
/// Cache-creation tokens have no Gemini wire field and are not reported.
fn encode_usage(usage: &Usage) -> Value {
    let cached = i64::from(usage.cache_read_input_tokens.unwrap_or(0));
    let prompt = i64::from(usage.input_tokens).saturating_add(cached);
    let total = prompt.saturating_add(i64::from(usage.output_tokens));
    let mut metadata = Map::new();
    metadata.insert(
        "promptTokenCount".to_owned(),
        Value::from(i32::try_from(prompt).unwrap_or(i32::MAX)),
    );
    if let Some(read) = usage.cache_read_input_tokens.filter(|read| *read > 0) {
        metadata.insert("cachedContentTokenCount".to_owned(), Value::from(read));
    }
    metadata.insert(
        "candidatesTokenCount".to_owned(),
        Value::from(usage.output_tokens),
    );
    if let Some(reasoning) = usage.reasoning_tokens {
        metadata.insert("thoughtsTokenCount".to_owned(), Value::from(reasoning));
    }
    metadata.insert(
        "totalTokenCount".to_owned(),
        Value::from(i32::try_from(total).unwrap_or(i32::MAX)),
    );
    Value::Object(metadata)
}

fn content_parts(content: &[CoreContent]) -> Vec<Value> {
    let mut parts = Vec::new();
    for block in content {
        match block {
            CoreContent::Text { text, .. } => parts.push(serde_json::json!({"text": text})),
            CoreContent::Thinking { text, .. } => {
                parts.push(serde_json::json!({"text": text, "thought": true}));
            }
            CoreContent::ToolUse {
                name,
                input,
                signature,
                ..
            } => {
                let mut part = serde_json::json!({
                    "functionCall": {"name": name, "args": input},
                });
                if let Some(signature) = signature
                    && let Some(obj) = part.as_object_mut()
                {
                    obj.insert(
                        "thoughtSignature".to_owned(),
                        Value::String(signature.clone()),
                    );
                }
                parts.push(part);
            }
            CoreContent::Refusal { text } => parts.push(serde_json::json!({"text": text})),
            other => {
                tracing::warn!(
                    kind = ?other,
                    "Gemini responses cannot carry this block; dropping"
                );
            }
        }
    }
    parts
}

/// Encode a [`CoreResponse`] into a Gemini `GenerateContentResponse`.
pub fn encode_response(resp: &CoreResponse) -> Result<Value, ProtocolError> {
    let parts = content_parts(&resp.content);
    let usage_metadata = encode_usage(&resp.usage);
    Ok(serde_json::json!({
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": core_to_finish_reason(&resp.stop_reason),
        }],
        "usageMetadata": usage_metadata,
        "modelVersion": resp.model.requested,
    }))
}

/// The error body shape the Gemini API returns
/// (`{"error": {"code", "message", "status"}}`).
pub fn error_payload(status: u16, message: &str) -> Value {
    let canonical = match status {
        400 => "INVALID_ARGUMENT",
        401 | 403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        409 => "ABORTED",
        429 => "RESOURCE_EXHAUSTED",
        499 => "CANCELLED",
        500 => "INTERNAL",
        503 => "UNAVAILABLE",
        504 => "DEADLINE_EXCEEDED",
        _ => "UNKNOWN",
    };
    serde_json::json!({
        "error": {"code": status, "message": message, "status": canonical}
    })
}

/// Stateful encoder that converts [`CoreEvent`] values into Gemini stream
/// chunks (one `GenerateContentResponse` per event; SSE framing sends each as
/// one `data:` line).
#[derive(Debug)]
pub struct StreamEncoder {
    model: Arc<str>,
    /// Open tool calls keyed by core content index: Gemini emits function
    /// calls as complete parts, so arguments buffer until `ToolCallStop`.
    open_tool_calls: std::collections::HashMap<usize, (String, String)>,
    pending_usage: Option<Usage>,
    pending_finish: Option<StopReason>,
    finished: bool,
}

impl StreamEncoder {
    /// Create a stream encoder for `model`.
    pub fn new(model: String) -> Self {
        Self {
            model: model.into(),
            open_tool_calls: std::collections::HashMap::new(),
            pending_usage: None,
            pending_finish: None,
            finished: false,
        }
    }

    fn chunk(&self, parts: Vec<Value>, finish_reason: Option<String>) -> Value {
        let mut content = Map::new();
        content.insert("role".to_owned(), Value::String("model".to_owned()));
        content.insert("parts".to_owned(), Value::Array(parts));
        let mut candidate = Map::new();
        candidate.insert("content".to_owned(), Value::Object(content));
        if let Some(reason) = finish_reason {
            candidate.insert("finishReason".to_owned(), Value::String(reason));
        }
        let mut chunk = Map::new();
        chunk.insert(
            "candidates".to_owned(),
            Value::Array(vec![Value::Object(candidate)]),
        );
        chunk.insert(
            "modelVersion".to_owned(),
            Value::String(self.model.to_string()),
        );
        Value::Object(chunk)
    }

    /// Encode a single [`CoreEvent`] into zero or more Gemini chunks.
    pub fn encode_event(&mut self, event: CoreEvent) -> Result<Vec<Value>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut chunks = Vec::new();
        match event {
            CoreEvent::MessageStart { .. } | CoreEvent::ContentStart { .. } => {}
            CoreEvent::TextDelta { text, .. } => {
                chunks.push(self.chunk(vec![serde_json::json!({"text": text})], None));
            }
            CoreEvent::ThinkingDelta { text, .. } => {
                chunks.push(self.chunk(
                    vec![serde_json::json!({"text": text, "thought": true})],
                    None,
                ));
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
                        "Gemini: ToolCallDelta without an open tool call; dropping"
                    );
                }
            }
            CoreEvent::ToolCallStop { index } => {
                if let Some((name, arguments)) = self.open_tool_calls.remove(&index) {
                    let args: Value = serde_json::from_str(&arguments).unwrap_or_else(|_| {
                        tracing::warn!(
                            arguments = %arguments,
                            "Gemini: malformed buffered tool arguments; sending an empty object"
                        );
                        Value::Object(Map::new())
                    });
                    let part = serde_json::json!({
                        "functionCall": {"name": name, "args": args},
                    });
                    chunks.push(self.chunk(vec![part], None));
                }
            }
            CoreEvent::UsageDelta { usage } => {
                self.pending_usage = Some(usage);
            }
            CoreEvent::MessageStop { stop_reason, .. } => {
                self.pending_finish = Some(stop_reason);
                chunks.extend(self.terminal_chunks());
            }
            CoreEvent::Error { .. } => {
                // Rendered by the route as a Gemini error chunk; nothing to
                // synthesize here.
                self.finished = true;
            }
            CoreEvent::Ping => {}
        }
        Ok(chunks)
    }

    fn terminal_chunks(&mut self) -> Vec<Value> {
        self.finished = true;
        let finish = self.pending_finish.clone().unwrap_or_else(|| {
            tracing::warn!("Gemini: stream ended without a MessageStop; synthesizing STOP");
            StopReason::EndTurn
        });
        let mut chunk = self.chunk(Vec::new(), Some(core_to_finish_reason(&finish)));
        let usage = self
            .pending_usage
            .clone()
            .unwrap_or_else(Usage::synthetic_zero);
        if let Some(obj) = chunk.as_object_mut() {
            obj.insert("usageMetadata".to_owned(), encode_usage(&usage));
        }
        vec![chunk]
    }

    /// Flush at the natural end of the stream.
    pub fn finish(&mut self) -> Result<Vec<Value>, ProtocolError> {
        if self.finished {
            return Ok(Vec::new());
        }
        Ok(self.terminal_chunks())
    }

    /// Mark the encoder as finished so subsequent `finish()` calls return
    /// empty; used after a terminal error chunk has been emitted.
    pub fn mark_finished(&mut self) {
        self.finished = true;
    }

    /// Whether a terminal chunk has been emitted.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::UsageProvenance;

    #[test]
    fn decode_maps_the_documented_request_shape() {
        let body = serde_json::json!({
            "systemInstruction": {"parts": [{"text": "be terse"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "weather in SF?"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}},
                     "thoughtSignature": "c2ln"}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"temp": 72}}}
                ]}
            ],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "weather", "parameters": {"type": "object"}}
            ]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}},
            "generationConfig": {
                "temperature": 0.3, "topP": 0.9, "topK": 20,
                "maxOutputTokens": 512, "stopSequences": ["END"],
                "thinkingConfig": {"thinkingBudget": 1024}
            },
            "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT"}]
        });
        let core = decode_request(&body, "gemini-2.5-pro", false).expect("decode");
        assert_eq!(core.model.requested, "gemini-2.5-pro");
        assert!(!core.stream);
        assert_eq!(
            core.system,
            vec![CoreContent::Text {
                text: "be terse".into(),
                cache: None
            }]
        );
        assert_eq!(core.messages.len(), 3);
        assert_eq!(core.sampling.temperature, Some(0.3));
        assert_eq!(core.sampling.top_p, Some(0.9));
        assert_eq!(core.sampling.top_k, Some(20));
        assert_eq!(core.sampling.max_tokens, Some(512));
        assert_eq!(core.sampling.stop.as_deref(), Some(&["END".to_owned()][..]));
        assert_eq!(
            core.sampling
                .thinking
                .as_ref()
                .and_then(|t| t.get("budget_tokens")),
            Some(&serde_json::json!(1024))
        );
        match &core.messages[1].content[0] {
            CoreContent::ToolUse {
                name, signature, ..
            } => {
                assert_eq!(name, "get_weather");
                assert_eq!(signature.as_deref(), Some("c2ln"));
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
        match &core.messages[2].content[0] {
            CoreContent::ToolResult { tool_use_id, .. } => {
                // The result is correlated to the replayed call by name.
                assert_eq!(tool_use_id, "gemini_call_1_0");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        assert!(matches!(
            core.tool_choice,
            Some(CoreToolChoice::Tool { ref name }) if name == "get_weather"
        ));
        assert_eq!(core.tools.len(), 1);
    }

    #[test]
    fn function_calls_in_different_turns_get_distinct_ids() {
        // Normal multi-turn tool traffic: every model turn's call sits at
        // part 0. The calls need distinct ids across the request (upstreams
        // reject or mis-pair duplicates), and each functionResponse must
        // land on the call of its own turn.
        let body = serde_json::json!({
            "contents": [
                {"role": "user", "parts": [{"text": "weather in SF, then the time?"}]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"temp": 72}}}
                ]},
                {"role": "model", "parts": [
                    {"functionCall": {"name": "get_time", "args": {"city": "SF"}}}
                ]},
                {"role": "user", "parts": [
                    {"functionResponse": {"name": "get_time", "response": {"time": "09:00"}}}
                ]}
            ]
        });
        let core = decode_request(&body, "gemini-2.5-pro", false).expect("decode");
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
        assert_eq!(
            result_ids, call_ids,
            "each result answers its own turn's call"
        );
    }

    #[test]
    fn encode_response_carries_tool_calls_thoughts_and_usage() {
        let resp = CoreResponse {
            id: None,
            model: ModelRef {
                requested: "gemini-2.5-pro".into(),
                upstream: None,
            },
            content: vec![
                CoreContent::Thinking {
                    text: "thinking...".into(),
                    signature: None,
                },
                CoreContent::Text {
                    text: "it is 72F".into(),
                    cache: None,
                },
                CoreContent::ToolUse {
                    id: "gemini_call_0".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                    signature: Some("c2ln".into()),
                },
            ],
            stop_reason: StopReason::ToolUse,
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                reasoning_tokens: Some(3),
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
                provenance: UsageProvenance::ProviderReported,
            },
            provider_meta: serde_json::Map::new(),
            cost: None,
        };
        let value = encode_response(&resp).expect("encode");
        let parts = value["candidates"][0]["content"]["parts"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[1]["text"], "it is 72F");
        assert_eq!(parts[2]["functionCall"]["name"], "get_weather");
        assert_eq!(parts[2]["thoughtSignature"], "c2ln");
        assert_eq!(value["candidates"][0]["finishReason"], "STOP");
        assert_eq!(value["usageMetadata"]["promptTokenCount"], 10);
        assert_eq!(value["usageMetadata"]["candidatesTokenCount"], 5);
        assert_eq!(value["usageMetadata"]["thoughtsTokenCount"], 3);
    }

    #[test]
    fn stream_text_and_tool_call_order() {
        let mut encoder = StreamEncoder::new("gemini-2.5-pro".into());
        let mut chunks = Vec::new();
        for event in [
            CoreEvent::MessageStart {
                id: None,
                model: ModelRef {
                    requested: "m".into(),
                    upstream: None,
                },
            },
            CoreEvent::TextDelta {
                index: 0,
                text: "hi".into(),
            },
            CoreEvent::UsageDelta {
                usage: Usage::provider_reported(4, 2),
            },
            CoreEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                stop_sequence: None,
            },
        ] {
            chunks.extend(encoder.encode_event(event).expect("encode"));
        }
        assert_eq!(chunks.len(), 2, "{chunks:#?}");
        assert_eq!(
            chunks[0]["candidates"][0]["content"]["parts"][0]["text"],
            "hi"
        );
        let terminal = chunks.last().cloned().unwrap_or_default();
        assert_eq!(terminal["candidates"][0]["finishReason"], "STOP");
        assert_eq!(terminal["usageMetadata"]["promptTokenCount"], 4);
        assert_eq!(terminal["usageMetadata"]["totalTokenCount"], 6);
    }

    /// The streamed terminal chunk must carry the same `thoughtsTokenCount`
    /// the non-streaming response carries, so Gemini clients see reasoning
    /// usage in both paths.
    #[test]
    fn stream_terminal_chunk_carries_thoughts_token_count() {
        let mut encoder = StreamEncoder::new("gemini-2.5-pro".into());
        let mut usage = Usage::provider_reported(4, 2);
        usage.reasoning_tokens = Some(3);
        let mut chunks = Vec::new();
        for event in [
            CoreEvent::UsageDelta { usage },
            CoreEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
                stop_sequence: None,
            },
        ] {
            chunks.extend(encoder.encode_event(event).expect("encode"));
        }
        let terminal = chunks.last().cloned().unwrap_or_default();
        assert_eq!(terminal["candidates"][0]["finishReason"], "STOP");
        assert_eq!(terminal["usageMetadata"]["thoughtsTokenCount"], 3);
    }

    /// A usage with no cache read must not synthesize a
    /// `cachedContentTokenCount` key: Google clients read its absence as "no
    /// cached content", which is the truth here.
    #[test]
    fn usage_without_cache_reads_has_no_cached_content_token_count() {
        let metadata = encode_usage(&Usage::provider_reported(10, 5));
        assert_eq!(metadata["promptTokenCount"], 10);
        assert_eq!(metadata["candidatesTokenCount"], 5);
        assert_eq!(metadata["totalTokenCount"], 15);
        assert!(
            metadata.get("cachedContentTokenCount").is_none(),
            "unexpected key: {metadata}"
        );
    }

    #[test]
    fn error_payload_uses_google_status_codes() {
        let payload = error_payload(429, "slow down");
        assert_eq!(payload["error"]["status"], "RESOURCE_EXHAUSTED");
        assert_eq!(payload["error"]["code"], 429);
    }
}
