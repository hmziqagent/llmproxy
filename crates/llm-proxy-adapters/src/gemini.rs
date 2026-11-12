//! Google Gemini GenerateContent API provider adapter: translates core
//! requests, responses, and event streams to and from the Gemini wire format.

use std::collections::HashMap;

use llm_proxy_protocol::core::{
    ContentKind, CoreContent, CoreEvent, CoreRequest, CoreResponse, CoreRole, CoreToolChoice,
    ModelRef, StopReason, Usage, UsageProvenance,
};
use llm_proxy_protocol::zen::{
    GeminiFunctionCall, GeminiFunctionDeclaration, GeminiGenerationConfig, GeminiPart,
    GeminiRequest, GeminiResponse, GeminiStreamChunk, GeminiThinkingConfig, GeminiTool,
    GeminiUsage,
};

use super::{
    ProviderAdapterTarget, ProviderStreamDecoder, build_proxy_request, expand_url_template,
    map_gemini_finish_reason, response_model_ref,
};
use crate::ProtocolName;
use crate::error::ProviderError;
use crate::protocols;
use crate::sse::SseFrame;

/// Adapter for the Google Gemini GenerateContent API.
#[derive(Debug, Clone, Default)]
pub struct GeminiAdapter;

/// Stateful stream decoder translating Gemini SSE frames into `CoreEvent`s.
#[derive(Debug)]
pub struct GeminiStreamDecoder {
    model_ref: ModelRef,
    started: bool,
    content_index: usize,
    content_started: bool,
    tool_blocks: Vec<usize>,
    tool_blocks_closed: Vec<bool>,
    tool_id_counter: usize,
    stop_sent: bool,
    saw_terminal: bool,
}

impl ProviderStreamDecoder for GeminiStreamDecoder {
    fn decode_frame(&mut self, frame: &SseFrame) -> Result<Vec<CoreEvent>, ProviderError> {
        let data = frame.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(vec![]);
        }

        let chunk: GeminiStreamChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(_) => {
                let truncated = super::truncate_str_safe(data, 200);
                tracing::warn!(data = truncated, "malformed Gemini chunk, skipping");
                return Ok(vec![]);
            }
        };

        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            events.push(CoreEvent::MessageStart {
                id: None,
                model: self.model_ref.clone(),
            });
        }

        // `first` keeps the empty-candidates handling and the head access in
        // one checked form.
        let Some(candidate) = chunk.candidates.first() else {
            if let Some(ref usage) = chunk.usage_metadata {
                events.push(CoreEvent::UsageDelta {
                    usage: build_gemini_usage(usage),
                });
            }
            return Ok(events);
        };

        let parts: &[GeminiPart] = candidate
            .content
            .as_ref()
            .map(|c| c.parts.as_slice())
            .unwrap_or(&[]);
        for part in parts {
            if let Some(ref text) = part.text
                && !text.is_empty()
            {
                self.push_text_delta(text, &mut events);
            }

            if let Some(ref function_call) = part.function_call {
                self.push_function_call(function_call, &mut events);
            }
        }

        if let Some(ref reason) = candidate.finish_reason {
            if !reason.is_empty() && !self.stop_sent {
                self.close_content_if_open();
                self.saw_terminal = true;

                let stop_reason = map_gemini_finish_reason(reason);

                if let Some(ref usage) = chunk.usage_metadata {
                    events.push(CoreEvent::UsageDelta {
                        usage: build_gemini_usage(usage),
                    });
                }

                events.push(CoreEvent::MessageStop {
                    stop_reason,
                    stop_sequence: None,
                });
                self.stop_sent = true;
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

        for (&idx, &closed) in self.tool_blocks.iter().zip(&self.tool_blocks_closed) {
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

impl GeminiStreamDecoder {
    fn push_text_delta(&mut self, text: &str, events: &mut Vec<CoreEvent>) {
        if !self.content_started {
            self.content_started = true;
            events.push(CoreEvent::ContentStart {
                index: self.content_index,
                kind: ContentKind::Text,
            });
        }
        events.push(CoreEvent::TextDelta {
            index: self.content_index,
            text: text.to_owned(),
        });
    }

    fn push_function_call(
        &mut self,
        function_call: &GeminiFunctionCall,
        events: &mut Vec<CoreEvent>,
    ) {
        self.close_content_if_open();

        let block_idx = self.content_index;
        self.tool_blocks.push(block_idx);
        self.tool_blocks_closed.push(false);

        events.push(CoreEvent::ToolCallStart {
            index: block_idx,
            id: format!("gemini_call_{}", self.tool_id_counter),
            name: function_call.name.clone(),
        });
        self.tool_id_counter += 1;

        if let Some(ref args) = function_call.args {
            let args_str = serde_json::to_string(args).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    "Gemini: failed to serialize function_call args, falling back to empty object"
                );
                "{}".to_owned()
            });
            if args_str != "null" {
                events.push(CoreEvent::ToolCallDelta {
                    index: block_idx,
                    args_delta: args_str,
                });
            }
        }

        events.push(CoreEvent::ToolCallStop { index: block_idx });
        if let Some(last) = self.tool_blocks_closed.last_mut() {
            *last = true;
        }
        self.content_index += 1;
    }

    fn close_content_if_open(&mut self) {
        if self.content_started {
            self.content_started = false;
            self.content_index += 1;
        }
    }
}

fn encode_content_part(
    c: &CoreContent,
    tool_name_map: &HashMap<String, String>,
    tool_signature_map: &HashMap<String, String>,
    parts: &mut Vec<GeminiPart>,
) -> Result<(), ProviderError> {
    match c {
        CoreContent::Text { text, cache } => {
            if cache.is_some() {
                tracing::warn!("Gemini: cache_control is not supported, dropping cache marker");
            }
            if !text.is_empty() {
                parts.push(GeminiPart::text(text.clone()));
            }
        }
        CoreContent::ToolUse {
            name,
            input,
            signature,
            ..
        } => {
            let mut part = GeminiPart::function_call(name.clone(), Some(input.clone()));
            // Thought signatures captured from a previous Gemini response ride
            // the replayed functionCall part back to the model.
            if let Some(signature) = signature {
                part = part.with_thought_signature(signature.clone());
            }
            parts.push(part);
        }
        CoreContent::Image { source } => {
            if let Some(part) = neutral_image_to_gemini_part(source) {
                parts.push(part);
            }
        }
        CoreContent::ToolResult {
            tool_use_id,
            content: result_content,
            is_error,
            ..
        } => {
            let response_val = encode_tool_result_payload(tool_use_id, result_content, *is_error);

            let Some(fn_name) = tool_name_map.get(tool_use_id) else {
                return Err(ProviderError::InvalidConfig(format!(
                    "Gemini: cannot resolve function name for tool_use_id \
                     \"{tool_use_id}\"; the corresponding ToolUse block was \
                     not found in the conversation history"
                )));
            };
            // Gemini requires the original thoughtSignature on the
            // functionResponse part when replaying signed tool state.
            let mut part = GeminiPart::function_response(fn_name.clone(), response_val);
            if let Some(signature) = tool_signature_map.get(tool_use_id) {
                part = part.with_thought_signature(signature.clone());
            }
            parts.push(part);
        }
        _ => {
            tracing::warn!(
                ?c,
                "Gemini: dropping unsupported content type in message encoding"
            );
        }
    }
    Ok(())
}

/// Maps a neutral core image source onto a Gemini media part: base64 sources
/// become `inlineData`, url sources become `fileData`.
fn neutral_image_to_gemini_part(source: &serde_json::Value) -> Option<GeminiPart> {
    let obj = source.as_object()?;
    match obj.get("type").and_then(serde_json::Value::as_str) {
        Some("base64") => {
            let mime_type = obj.get("media_type").and_then(serde_json::Value::as_str)?;
            let data = obj.get("data").and_then(serde_json::Value::as_str)?;
            Some(GeminiPart::inline_data(
                mime_type.to_owned(),
                data.to_owned(),
            ))
        }
        Some("url") => {
            let url = obj.get("url").and_then(serde_json::Value::as_str)?;
            let mime_type = obj
                .get("media_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("image/png");
            Some(GeminiPart::file_data(mime_type.to_owned(), url.to_owned()))
        }
        other => {
            tracing::warn!(
                source_type = ?other,
                "Gemini: unsupported neutral image source type; dropping"
            );
            None
        }
    }
}

fn encode_tool_result_payload(
    tool_use_id: &str,
    result_content: &[CoreContent],
    is_error: bool,
) -> serde_json::Value {
    // A `functionResponse` reports a failed call under an `error` key, where
    // the Gemini API reads the error details for the model.
    let key = if is_error { "error" } else { "result" };
    if result_content.is_empty() {
        return serde_json::json!({ key: "" });
    }
    let texts: Vec<&str> = result_content
        .iter()
        .filter_map(|rc| match rc {
            CoreContent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if texts.len() != result_content.len() {
        tracing::warn!(
            tool_use_id,
            "Gemini: dropping non-text content in tool result"
        );
    }
    serde_json::json!({ key: texts.join("\n") })
}

/// Maps the core tool choice onto Gemini's `toolConfig.functionCallingConfig`.
/// The config is emitted only when the request carries function declarations
/// (Gemini rejects `toolConfig` in ANY mode without `tools`), so `None`/`Auto`
/// with no tools are omitted entirely. A `Raw` choice (or any future core
/// variant) has no Gemini representation: strict routes fail with the field
/// named, lossy routes warn and drop it.
fn encode_tool_choice(
    choice: Option<&CoreToolChoice>,
    has_tools: bool,
    lossy: bool,
) -> Result<Option<serde_json::Value>, ProviderError> {
    let Some(choice) = choice else {
        return Ok(None);
    };
    let config = match choice {
        CoreToolChoice::Auto => serde_json::json!({"functionCallingConfig": {"mode": "AUTO"}}),
        CoreToolChoice::Any => serde_json::json!({"functionCallingConfig": {"mode": "ANY"}}),
        CoreToolChoice::None => serde_json::json!({"functionCallingConfig": {"mode": "NONE"}}),
        // The forced name must be the same string the `functionDeclarations`
        // entry uses, which is the core tool's `name` verbatim.
        CoreToolChoice::Tool { name } => serde_json::json!({
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name],
            }
        }),
        CoreToolChoice::Raw(_) | _ => {
            if lossy {
                tracing::warn!(
                    tool_choice = ?choice,
                    "Gemini: tool_choice has no Gemini equivalent; \
                     dropping because the route allows lossy conversion"
                );
                return Ok(None);
            }
            return Err(ProviderError::InvalidRequest(format!(
                "tool_choice {choice:?} has no gemini equivalent; \
                 remove it or enable the route's lossy mode"
            )));
        }
    };
    if !has_tools {
        return Ok(None);
    }
    Ok(Some(config))
}

/// The client's `generationConfig` object under either spelling; the camelCase
/// spelling wins when both are present, matching the client decode.
fn generation_config_mut(
    body: &mut serde_json::Value,
) -> Option<&mut serde_json::Map<String, serde_json::Value>> {
    let key = if body.get("generationConfig").is_some() {
        "generationConfig"
    } else {
        "generation_config"
    };
    body.get_mut(key).and_then(serde_json::Value::as_object_mut)
}

/// The `thinkingConfig` object inside a `generationConfig`, under either
/// spelling.
fn thinking_config_mut(
    config: &mut serde_json::Map<String, serde_json::Value>,
) -> Option<&mut serde_json::Map<String, serde_json::Value>> {
    let key = if config.contains_key("thinkingConfig") {
        "thinkingConfig"
    } else {
        "thinking_config"
    };
    config
        .get_mut(key)
        .and_then(serde_json::Value::as_object_mut)
}

/// Same-format passthrough edit: the client's `thinkingBudget` is clamped to
/// the upstream model's limits through the same filter every route into Gemini
/// uses, instead of reaching the wire unclamped. The `-1` (dynamic) and `0`
/// (disabled) sentinels pass through untouched; both the camelCase and
/// snake_case spellings are located and edited in place, and every other field
/// survives untouched. A non-integer budget is rejected by name rather than
/// forwarded.
fn clamp_same_format_thinking_budget(
    body: &mut serde_json::Value,
    upstream_model: &str,
) -> Result<(), ProviderError> {
    let Some(config) = generation_config_mut(body) else {
        return Ok(());
    };
    let Some(thinking) = thinking_config_mut(config) else {
        return Ok(());
    };
    let budget_key = if thinking.contains_key("thinkingBudget") {
        "thinkingBudget"
    } else if thinking.contains_key("thinking_budget") {
        "thinking_budget"
    } else {
        return Ok(());
    };
    let Some(budget) = thinking.get(budget_key).and_then(serde_json::Value::as_i64) else {
        return Err(ProviderError::InvalidRequest(
            "thinkingBudget must be an integer".to_owned(),
        ));
    };
    if budget == -1 || budget == 0 {
        return Ok(());
    }
    let limits = super::reasoning::thinking_limits(upstream_model);
    let clamped = super::reasoning::clamp_budget(budget, limits);
    if clamped != budget {
        tracing::warn!(
            budget,
            clamped,
            upstream_model,
            "clamping the client's thinkingBudget to the upstream model's limits"
        );
        thinking.insert(budget_key.to_owned(), serde_json::Value::from(clamped));
    }
    Ok(())
}

impl GeminiAdapter {
    /// Encode a core request into a Gemini GenerateContent request.
    #[expect(
        clippy::too_many_lines,
        reason = "field-by-field wire encode of the Gemini GenerateContent request \
                 shape, kept in one place so the mapping can be audited against \
                 the API schema without jumping between helpers"
    )]
    pub fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &super::client_wire::ClientWireRequest,
    ) -> Result<super::ProxyRequest, ProviderError> {
        let url = gemini_stream_url(expand_url_template(&target.endpoint, target)?, core.stream);

        // Same-format passthrough (plan 08 step 2): a Gemini-dialect client
        // forwards its raw body with only surgical edits (the model lives in
        // the URL, which the alias resolution above already rewrote; streaming
        // lives in the method, which the URL rewrite above already selected).
        if client.same_format_as(&ProtocolName::new(protocols::GEMINI_GENERATE_CONTENT))
            && client.body_object().is_some()
        {
            let mut value = (*client.body).clone();
            clamp_same_format_thinking_budget(&mut value, &target.upstream_model)?;
            if let Some(hygiene) = &target.prompt_cache {
                super::mutation::apply_prompt_hygiene(&mut value, hygiene, false);
            }
            let body = serde_json::to_vec(&value)?;
            return Ok(build_proxy_request(body, target, core.stream, url, client));
        }

        // Every other client dialect is cross-format here: settle the
        // leftover client fields first.
        let carried = super::client_wire::apply_field_policy(
            client,
            &ProtocolName::new(protocols::GEMINI_GENERATE_CONTENT),
            target.lossy,
        )?;
        super::client_wire::check_cross_format_content(
            core,
            &ProtocolName::new(protocols::GEMINI_GENERATE_CONTENT),
            target.lossy,
        )?;
        if !carried.is_empty() {
            tracing::warn!(
                keys = ?carried.keys().collect::<Vec<_>>(),
                "Gemini: carried client fields have no Gemini request slot; dropping"
            );
        }
        let mut contents = Vec::new();

        let tool_name_map = build_tool_name_map(core);
        let tool_signature_map = build_tool_signature_map(core);

        for msg in &core.messages {
            let role = match msg.role {
                CoreRole::User => "user",
                CoreRole::Assistant => "model",
                CoreRole::System => "user",
                CoreRole::Tool => "user",
                _ => "user",
            };

            let mut parts: Vec<GeminiPart> = Vec::new();
            for c in &msg.content {
                encode_content_part(c, &tool_name_map, &tool_signature_map, &mut parts)?;
            }

            if !parts.is_empty() {
                contents.push(llm_proxy_protocol::zen::GeminiContent {
                    role: role.to_owned(),
                    parts,
                });
            }
        }

        if !core.system.is_empty() {
            for c in &core.system {
                if !matches!(c, CoreContent::Text { .. }) {
                    tracing::warn!(
                        ?c,
                        "Gemini: dropping non-Text system content block during encode"
                    );
                }
            }
            let system_text: String = core
                .system
                .iter()
                .filter_map(|c| match c {
                    CoreContent::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .fold(String::new(), |mut acc, s| {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(s);
                    acc
                });
            if !system_text.is_empty() {
                contents.insert(
                    0,
                    llm_proxy_protocol::zen::GeminiContent {
                        role: "user".to_owned(),
                        parts: vec![GeminiPart::text(system_text)],
                    },
                );
                let next_is_model = contents.get(1).is_some_and(|c| c.role == "model");
                if !next_is_model {
                    contents.insert(
                        1,
                        llm_proxy_protocol::zen::GeminiContent {
                            role: "model".to_owned(),
                            parts: vec![GeminiPart::text("Understood.".to_owned())],
                        },
                    );
                }
            }
        }

        let tools: Vec<GeminiTool> = if core.tools.is_empty() {
            vec![]
        } else {
            vec![GeminiTool {
                function_declarations: core
                    .tools
                    .iter()
                    .map(|t| {
                        let schema = if !t.input_schema.is_object() {
                            serde_json::json!({"type": "object", "properties": {}})
                        } else {
                            t.input_schema.clone()
                        };
                        GeminiFunctionDeclaration {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: Some(schema),
                        }
                    })
                    .collect(),
            }]
        };

        let thinking_config =
            super::reasoning::gemini_budget(&core.sampling, &target.upstream_model)?.map(
                |budget| GeminiThinkingConfig {
                    thinking_budget: budget,
                },
            );

        // `top_k` arrives via the core (Anthropic clients) or as a carried
        // extra (OpenAI Chat clients send it as an unknown field).
        let top_k = core.sampling.top_k.or_else(|| {
            client
                .body
                .as_object()
                .and_then(|obj| obj.get("top_k"))
                .and_then(serde_json::Value::as_i64)
                .and_then(|v| i32::try_from(v).ok())
        });

        // `seed` is an OpenAI Chat extra; Gemini carries it verbatim as
        // generationConfig.seed. A present non-integer value has no mapping:
        // it is rejected naming the field (dropped with a warning under the
        // per-route `lossy` opt-in) instead of silently vanishing.
        let seed = match client.body.as_object().and_then(|obj| obj.get("seed")) {
            None | Some(serde_json::Value::Null) => None,
            Some(value) => match value.as_i64() {
                Some(seed) => Some(seed),
                None => {
                    if target.lossy {
                        tracing::warn!(
                            field = "seed",
                            provider = protocols::GEMINI_GENERATE_CONTENT,
                            "lossy route: dropping non-integer seed with no Gemini equivalent"
                        );
                        None
                    } else {
                        return Err(ProviderError::InvalidRequest(
                            "seed must be an integer to map onto generationConfig.seed; \
                             remove it or enable the route's lossy mode"
                                .to_owned(),
                        ));
                    }
                }
            },
        };

        // `response_format` maps onto generationConfig.responseMimeType
        // (plus responseJsonSchema for a schema) — structured output is a
        // native Gemini capability (plan step 4); `{"type": "text"}` maps to
        // omission.
        use super::client_wire::ResponseFormat;
        let mut response_mime_type = None;
        let mut response_json_schema = None;
        match super::client_wire::parse_response_format(
            client,
            protocols::GEMINI_GENERATE_CONTENT,
            target.lossy,
        )? {
            None | Some(ResponseFormat::Text) => {}
            Some(ResponseFormat::JsonObject) => {
                response_mime_type = Some("application/json".to_owned());
            }
            Some(ResponseFormat::JsonSchema { spec }) => {
                response_mime_type = Some("application/json".to_owned());
                response_json_schema = spec.get("schema").cloned();
            }
        }

        let generation_config = GeminiGenerationConfig {
            temperature: core.sampling.temperature,
            top_p: core.sampling.top_p,
            top_k,
            seed,
            max_output_tokens: core.sampling.max_tokens,
            stop_sequences: core
                .sampling
                .stop
                .as_ref()
                .and_then(|s| if s.is_empty() { None } else { Some(s.clone()) }),
            thinking_config,
            response_mime_type,
            response_json_schema,
        };

        let tool_config =
            encode_tool_choice(core.tool_choice.as_ref(), !tools.is_empty(), target.lossy)?;

        if core.metadata.user_id.is_some() {
            tracing::warn!(
                "Gemini: user_id specified but not forwarded; \
                 Gemini does not support this parameter"
            );
        }

        let req = GeminiRequest {
            contents,
            generation_config: Some(generation_config),
            tools,
            tool_config,
            stream: if core.stream { Some(true) } else { None },
        };

        // Phase 3 (plan 06 step 7): operator hygiene at the provider edge —
        // volatile-byte stripping only (this wire format has no cache
        // markers). The round-trip through `Value` happens only when the
        // operator configured hygiene.
        let body = match &target.prompt_cache {
            None => serde_json::to_vec(&req)?,
            Some(hygiene) => {
                let mut value = serde_json::to_value(&req)?;
                super::mutation::apply_prompt_hygiene(&mut value, hygiene, false);
                serde_json::to_vec(&value)?
            }
        };

        Ok(build_proxy_request(body, target, core.stream, url, client))
    }

    /// Decode a Gemini GenerateContent response body.
    pub fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        let resp: GeminiResponse = serde_json::from_slice(bytes)?;

        let Some(candidate) = resp.candidates.first() else {
            return Err(ProviderError::EmptyResponse(
                "no candidates in response".to_owned(),
            ));
        };
        let mut content = Vec::new();
        let mut tool_id_counter = 0usize;

        let parts: &[GeminiPart] = candidate
            .content
            .as_ref()
            .map(|c| c.parts.as_slice())
            .unwrap_or(&[]);
        for part in parts {
            if let Some(ref text) = part.text {
                if !text.is_empty() {
                    content.push(CoreContent::Text {
                        text: text.clone(),
                        cache: None,
                    });
                }
            }
            if let Some(ref function_call) = part.function_call {
                let input = function_call
                    .args
                    .clone()
                    .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
                content.push(CoreContent::ToolUse {
                    id: format!("gemini_call_{}", tool_id_counter),
                    name: function_call.name.clone(),
                    input,
                    // Thought signatures must be replayed on the function
                    // response part, so they ride the ToolUse block.
                    signature: part.thought_signature.clone(),
                });
                tool_id_counter += 1;
            }
        }

        if content.is_empty() {
            content.push(CoreContent::Text {
                text: String::new(),
                cache: None,
            });
        }

        let has_tool_use = content
            .iter()
            .any(|c| matches!(c, CoreContent::ToolUse { .. }));
        let stop_reason = candidate
            .finish_reason
            .as_deref()
            .map(map_gemini_finish_reason)
            .unwrap_or(if has_tool_use {
                StopReason::ToolUse
            } else {
                StopReason::Unknown("absent".to_owned())
            });

        let usage = resp
            .usage_metadata
            .as_ref()
            .map(build_gemini_usage)
            .unwrap_or_else(Usage::synthetic_zero);

        Ok(CoreResponse {
            id: None,
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
    pub fn new_stream_decoder(&self, target: &ProviderAdapterTarget) -> GeminiStreamDecoder {
        GeminiStreamDecoder {
            model_ref: response_model_ref(target),
            started: false,
            content_index: 0,
            content_started: false,
            tool_blocks: Vec::new(),
            tool_blocks_closed: Vec::new(),
            tool_id_counter: 0,
            stop_sent: false,
            saw_terminal: false,
        }
    }
}

/// Selects the Gemini streaming endpoint for stream requests: streaming is
/// chosen by calling `:streamGenerateContent?alt=sse` instead of
/// `:generateContent`; the request body has no `stream` field.
fn gemini_stream_url(url: String, stream: bool) -> String {
    if !stream {
        return url;
    }
    let mut url = url;
    if let Some(base) = url.strip_suffix(":generateContent") {
        url = format!("{base}:streamGenerateContent");
    }
    if url.contains('?') {
        url.push_str("&alt=sse");
    } else {
        url.push_str("?alt=sse");
    }
    url
}

/// Maps tool_use_id -> thought signature across the conversation so a
/// functionResponse part can replay the signature of the call it answers.
fn build_tool_signature_map(core: &CoreRequest) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in &core.messages {
        for content in &msg.content {
            if let CoreContent::ToolUse {
                id,
                signature: Some(signature),
                ..
            } = content
            {
                map.insert(id.clone(), signature.clone());
            }
        }
    }
    map
}

fn build_tool_name_map(core: &CoreRequest) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in &core.messages {
        for content in &msg.content {
            if let CoreContent::ToolUse { id, name, .. } = content {
                map.entry(id.clone()).or_insert_with(|| name.clone());
            }
        }
    }
    map
}

fn build_gemini_usage(usage: &GeminiUsage) -> Usage {
    Usage {
        input_tokens: usage.prompt_token_count,
        output_tokens: usage.candidates_token_count,
        reasoning_tokens: None,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
        provenance: UsageProvenance::ProviderReported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_wire::{ClientWireFormat, ClientWireRequest};
    use llm_proxy_protocol::core::{CoreMessage, CoreTool, SamplingOptions};

    /// An Anthropic-dialect client wire: never the same format as the Gemini
    /// adapter, so tests exercise the cross-format encode.
    fn client_wire(core: &CoreRequest) -> ClientWireRequest {
        ClientWireRequest::new(
            ClientWireFormat::AnthropicMessages,
            serde_json::json!({"model": core.model.requested, "messages": []}),
        )
    }

    fn make_target() -> ProviderAdapterTarget {
        ProviderAdapterTarget {
            provider_name: "test-gemini".into(),
            adapter_name: "gemini".into(),
            protocol: super::ProtocolName::new(protocols::GEMINI_GENERATE_CONTENT),
            endpoint:
                "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    .into(),
            auth_style: llm_proxy_core::AuthStyle::Bearer,
            api_key: secrecy::SecretString::from("test-key"),
            requested_model: "gemini-2.5-pro".into(),
            upstream_model: "gemini-2.5-pro".into(),
            headers: std::sync::Arc::new(std::collections::HashMap::new()),
            lossy: false,
            forward_inbound_headers: std::sync::Arc::new(Vec::new()),
            prompt_cache: None,
        }
    }

    fn make_core_request(messages: Vec<CoreMessage>) -> CoreRequest {
        CoreRequest {
            model: ModelRef {
                requested: "gemini-2.5-pro".into(),
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

    /// Encodes a core request with the given tool choice through the
    /// cross-format path and returns the parsed Gemini wire body.
    fn encode_body_with_tool_choice(
        tool_choice: Option<CoreToolChoice>,
        tools: Vec<CoreTool>,
        lossy: bool,
    ) -> Result<serde_json::Value, ProviderError> {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "What is the weather?".into(),
                cache: None,
            }],
        }]);
        core.tools = tools;
        core.tool_choice = tool_choice;
        let mut target = make_target();
        target.lossy = lossy;
        let adapter = GeminiAdapter;
        let proxy_req = adapter.encode_request(&core, &target, &client_wire(&core))?;
        Ok(serde_json::from_slice(&proxy_req.body)?)
    }

    fn weather_tool() -> CoreTool {
        CoreTool {
            name: "get_weather".into(),
            description: Some("Get current weather".into()),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    #[test]
    fn encode_tool_choice_maps_to_tool_config() {
        let cases = [
            (
                CoreToolChoice::Auto,
                serde_json::json!({"functionCallingConfig": {"mode": "AUTO"}}),
            ),
            (
                CoreToolChoice::Any,
                serde_json::json!({"functionCallingConfig": {"mode": "ANY"}}),
            ),
            (
                CoreToolChoice::None,
                serde_json::json!({"functionCallingConfig": {"mode": "NONE"}}),
            ),
            (
                CoreToolChoice::Tool {
                    name: "get_weather".into(),
                },
                serde_json::json!({
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["get_weather"],
                    }
                }),
            ),
        ];
        for (choice, expected_config) in cases {
            let body =
                encode_body_with_tool_choice(Some(choice.clone()), vec![weather_tool()], false)
                    .unwrap();
            assert_eq!(
                body.get("toolConfig"),
                Some(&expected_config),
                "tool_choice {choice:?} must map onto toolConfig.functionCallingConfig; body: {body}"
            );
        }
    }

    #[test]
    fn encode_tool_choice_none_without_tools_omits_tool_config() {
        let body = encode_body_with_tool_choice(Some(CoreToolChoice::None), vec![], false).unwrap();
        assert!(
            body.get("toolConfig").is_none(),
            "toolConfig must be omitted when the request carries no tools: {body}"
        );
    }

    #[test]
    fn encode_raw_tool_choice_is_rejected_by_name() {
        let raw = CoreToolChoice::Raw(serde_json::json!({"type": "custom"}));
        let err = encode_body_with_tool_choice(Some(raw), vec![weather_tool()], false)
            .expect_err("Raw tool_choice must be rejected when the route is not lossy");
        assert!(
            err.to_string().contains("tool_choice"),
            "the rejection must name tool_choice, got: {err}"
        );
    }

    #[test]
    fn encode_raw_tool_choice_lossy_is_omitted() {
        let raw = CoreToolChoice::Raw(serde_json::json!({"type": "custom"}));
        let body = encode_body_with_tool_choice(Some(raw), vec![weather_tool()], true).unwrap();
        assert!(
            body.get("toolConfig").is_none(),
            "a lossy route drops the unmappable tool_choice instead of failing: {body}"
        );
    }

    #[test]
    fn encode_tool_choice_round_trips_through_client_decode() {
        let body = encode_body_with_tool_choice(
            Some(CoreToolChoice::Tool {
                name: "get_weather".into(),
            }),
            vec![weather_tool()],
            false,
        )
        .unwrap();
        let decoded =
            llm_proxy_protocol::client::gemini::decode_request(&body, "gemini-2.5-pro", false)
                .unwrap();
        assert_eq!(
            decoded.tool_choice,
            Some(CoreToolChoice::Tool {
                name: "get_weather".into(),
            }),
            "a forced named tool must survive the Gemini wire round trip"
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
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].role, "user");
        assert_eq!(body.contents[0].parts[0].text, Some("Hello".to_owned()));
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
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.contents[0].role, "user");
        assert_eq!(
            body.contents[0].parts[0].text,
            Some("You are helpful".to_owned())
        );
        assert_eq!(body.contents[1].role, "model");
    }

    #[test]
    fn encode_system_prompt_no_double_model_when_assistant_first() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::Assistant,
            content: vec![CoreContent::Text {
                text: "I will help.".into(),
                cache: None,
            }],
        }]);
        core.system = vec![CoreContent::Text {
            text: "You are helpful".into(),
            cache: None,
        }];
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.contents[0].role, "user");
        assert_eq!(body.contents[1].role, "model");
        let has_understood = body.contents.iter().any(|c| {
            c.parts
                .iter()
                .any(|p| p.text.as_deref() == Some("Understood."))
        });
        assert!(
            !has_understood,
            "synthetic Understood. turn must be suppressed when the next turn is already model"
        );
        for pair in body.contents.windows(2) {
            assert!(
                !(pair[0].role == "model" && pair[1].role == "model"),
                "consecutive model turns are not allowed: {:?} -> {:?}",
                pair[0].role,
                pair[1].role
            );
        }
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
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.tools.len(), 1);
        assert_eq!(body.tools[0].function_declarations[0].name, "get_weather");
    }

    /// A Chat-dialect client wire carrying `response_format`: never the same
    /// format as the Gemini adapter, so these tests exercise the cross-format
    /// structured-output mapping.
    fn chat_client_with_response_format(response_format: &serde_json::Value) -> ClientWireRequest {
        ClientWireRequest::new(
            ClientWireFormat::OpenAiChatCompletions,
            serde_json::json!({
                "model": "gemini-2.5-pro",
                "messages": [],
                "response_format": response_format,
            }),
        )
    }

    #[test]
    fn chat_json_object_sets_response_mime_type() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "Return JSON.".into(),
                cache: None,
            }],
        }]);
        let client = chat_client_with_response_format(&serde_json::json!({"type": "json_object"}));
        let adapter = GeminiAdapter;
        let proxy_req = adapter
            .encode_request(&core, &make_target(), &client)
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(
            body["generation_config"]["responseMimeType"], "application/json",
            "json_object must select Gemini JSON mode via responseMimeType: {body}"
        );
        assert!(
            body.get("response_format").is_none(),
            "the client field must not ride the wire verbatim: {body}"
        );
    }

    #[test]
    fn chat_json_schema_sets_response_json_schema() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "Return JSON.".into(),
                cache: None,
            }],
        }]);
        let client = chat_client_with_response_format(&serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_total",
                "strict": true,
                "schema": {"type": "object", "properties": {"total": {"type": "number"}}}
            }
        }));
        let adapter = GeminiAdapter;
        let proxy_req = adapter
            .encode_request(&core, &make_target(), &client)
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(
            body["generation_config"]["responseMimeType"], "application/json",
            "json_schema must also select JSON mode: {body}"
        );
        assert_eq!(
            body["generation_config"]["responseJsonSchema"],
            serde_json::json!({"type": "object", "properties": {"total": {"type": "number"}}}),
            "the standard JSON Schema must ride responseJsonSchema: {body}"
        );
        assert!(
            body.get("response_format").is_none(),
            "the client field must not ride the wire verbatim: {body}"
        );
    }

    #[test]
    fn chat_seed_maps_to_generation_config_seed() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let client = ClientWireRequest::new(
            ClientWireFormat::OpenAiChatCompletions,
            serde_json::json!({
                "model": "gemini-2.5-pro",
                "messages": [],
                "seed": 7,
            }),
        );
        let adapter = GeminiAdapter;
        let proxy_req = adapter
            .encode_request(&core, &make_target(), &client)
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(
            body["generation_config"]["seed"], 7,
            "the Chat seed must map onto generationConfig.seed: {body}"
        );
        assert!(
            body.get("seed").is_none(),
            "the client field must not ride the wire verbatim: {body}"
        );
    }

    #[test]
    fn chat_non_integer_seed_is_rejected_naming_seed_and_dropped_on_lossy() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let client = ClientWireRequest::new(
            ClientWireFormat::OpenAiChatCompletions,
            serde_json::json!({
                "model": "gemini-2.5-pro",
                "messages": [],
                "seed": "seven",
            }),
        );
        let err = GeminiAdapter
            .encode_request(&core, &make_target(), &client)
            .expect_err("a non-integer seed cannot map onto generationConfig.seed");
        assert!(
            err.to_string().contains("seed"),
            "the error must name seed: {err}"
        );

        let mut lossy_target = make_target();
        lossy_target.lossy = true;
        let proxy_req = GeminiAdapter
            .encode_request(&core, &lossy_target, &client)
            .expect("a lossy route drops the unmappable seed instead of failing");
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(
            body["generation_config"].get("seed").is_none(),
            "a lossy route must not send the unmappable seed: {body}"
        );
    }

    #[test]
    fn encode_generation_config() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling = SamplingOptions {
            temperature: Some(0.7),
            max_tokens: Some(1024),
            ..Default::default()
        };
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let config = body.generation_config.unwrap();
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_output_tokens, Some(1024));
    }

    #[test]
    fn encode_url_template_expansion() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        assert_eq!(
            proxy_req.url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
        );
    }

    #[test]
    fn stream_requests_use_stream_generate_content_with_alt_sse() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.stream = true;
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        assert!(
            proxy_req.url.ends_with(":streamGenerateContent?alt=sse"),
            "streaming URL must select streamGenerateContent with alt=sse: {}",
            proxy_req.url
        );
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(
            body.get("stream").is_none(),
            "the Gemini request body must not carry a stream field: {body}"
        );
    }

    #[test]
    fn non_stream_requests_keep_generate_content_url() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        assert!(
            proxy_req.url.ends_with(":generateContent"),
            "non-streaming URL must keep generateContent: {}",
            proxy_req.url
        );
        assert!(!proxy_req.url.contains("alt=sse"));
    }

    #[test]
    fn camel_case_function_call_parts_decode_to_tool_use() {
        let target = make_target();
        let resp = GeminiAdapter
            .decode_response(
                br#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
                &target,
            )
            .unwrap();
        let tool = resp
            .content
            .iter()
            .find_map(|c| match c {
                CoreContent::ToolUse { name, input, .. } => Some((name.clone(), input.clone())),
                _ => None,
            })
            .expect("functionCall part must decode to ToolUse");
        assert_eq!(tool.0, "get_weather");
        assert_eq!(tool.1["city"], "SF");
    }

    #[test]
    fn truncated_gemini_stream_has_no_terminal() {
        let target = make_target();
        let mut decoder = GeminiAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}"#
                .to_owned(),
        };
        decoder.decode_frame(&frame).unwrap();
        decoder.finish().unwrap();
        assert!(
            !decoder.saw_terminal(),
            "a stream cut before finishReason must not count as terminal"
        );
    }

    #[test]
    fn decode_text_response() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "hello world"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.model.requested, "gemini-2.5-pro");
        assert_eq!(core_resp.stop_reason, StopReason::EndTurn);
        assert_eq!(core_resp.usage.input_tokens, 100);
        assert_eq!(core_resp.usage.output_tokens, 50);
        assert_eq!(core_resp.content.len(), 1);
    }

    #[test]
    fn decode_function_call_response() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 200,
                "candidatesTokenCount": 80,
                "totalTokenCount": 280
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        match &core_resp.content[0] {
            CoreContent::ToolUse { name, input, .. } => {
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn decode_no_candidates_error() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "candidates": [],
            "usageMetadata": null
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
        let result = adapter.decode_response(&bytes, &target);
        assert!(result.is_err());
        match result.unwrap_err() {
            ProviderError::EmptyResponse(_) => {}
            other => panic!("expected EmptyResponse, got {:?}", other),
        }
    }

    #[test]
    fn decode_preserves_requested_model() {
        let mut target = make_target();
        target.requested_model = "my-gemini".into();

        let resp_json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "hi"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.model.requested, "my-gemini");
    }

    #[test]
    fn stream_text_decoding() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":null}],"usageMetadata":null}"#,
            ),
            make_frame(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
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
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. }))
        );
    }

    #[test]
    fn stream_done_marker_ignored() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("[DONE]");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_finish_emits_lifecycle() {
        let target = make_target();
        let adapter = GeminiAdapter;
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
    fn stream_tool_call_decoding() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
        );

        let events = decoder.decode_frame(&frame).unwrap();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallStart { .. })),
            "expected ToolCallStart"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallDelta { .. })),
            "expected ToolCallDelta"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallStop { .. })),
            "expected ToolCallStop"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. })),
            "expected MessageStop"
        );
    }

    #[test]
    fn stream_malformed_frame_skipped() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("not valid json {{{");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(
            events.is_empty(),
            "malformed frame should produce no events"
        );
    }

    #[test]
    fn stream_usage_only_chunk() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"candidates":[],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":50,"totalTokenCount":150}}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::UsageDelta { .. })),
            "usage-only chunk must emit UsageDelta"
        );
    }

    #[test]
    fn encode_tool_result_as_function_response() {
        let core = make_core_request(vec![
            CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::ToolUse {
                    id: "gemini_call_0".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                    signature: None,
                }],
            },
            CoreMessage {
                role: CoreRole::Tool,
                content: vec![CoreContent::ToolResult {
                    tool_use_id: "gemini_call_0".into(),
                    content: vec![CoreContent::Text {
                        text: "72F, sunny".into(),
                        cache: None,
                    }],
                    is_error: false,
                    cache: None,
                }],
            },
        ]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let fr = body
            .contents
            .iter()
            .find_map(|c| c.parts.iter().find_map(|p| p.function_response.clone()))
            .expect("ToolResult must be encoded as a functionResponse");
        assert_eq!(fr.name, "get_weather");
    }

    /// Encodes a tool call followed by its result and returns the raw wire
    /// `functionResponse` payload object.
    fn encode_tool_result_response(
        result_content: Vec<CoreContent>,
        is_error: bool,
    ) -> serde_json::Value {
        let core = make_core_request(vec![
            CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::ToolUse {
                    id: "gemini_call_0".into(),
                    name: "write_file".into(),
                    input: serde_json::json!({"path": "/tmp/out"}),
                    signature: None,
                }],
            },
            CoreMessage {
                role: CoreRole::Tool,
                content: vec![CoreContent::ToolResult {
                    tool_use_id: "gemini_call_0".into(),
                    content: result_content,
                    is_error,
                    cache: None,
                }],
            },
        ]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        let response_part = &body["contents"][1]["parts"][0];
        assert_eq!(response_part["functionResponse"]["name"], "write_file");
        response_part["functionResponse"]["response"].clone()
    }

    #[test]
    fn encode_failed_tool_result_uses_the_error_key() {
        let response = encode_tool_result_response(
            vec![CoreContent::Text {
                text: "disk full".into(),
                cache: None,
            }],
            true,
        );
        assert_eq!(response, serde_json::json!({"error": "disk full"}));
        assert!(
            response.get("result").is_none(),
            "a failed tool result must not be sent as a success payload: {response}"
        );
    }

    #[test]
    fn encode_successful_tool_result_uses_the_result_key() {
        let response = encode_tool_result_response(
            vec![CoreContent::Text {
                text: "disk full".into(),
                cache: None,
            }],
            false,
        );
        assert_eq!(response, serde_json::json!({"result": "disk full"}));
        assert!(
            response.get("error").is_none(),
            "a successful tool result must not be marked failed: {response}"
        );
    }

    #[test]
    fn encode_failed_tool_result_without_output_still_uses_the_error_key() {
        let response = encode_tool_result_response(Vec::new(), true);
        assert_eq!(response, serde_json::json!({"error": ""}));
    }

    #[test]
    fn encode_tool_use_as_function_call() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Assistant,
            content: vec![CoreContent::ToolUse {
                id: "gemini_call_0".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"city": "SF"}),
                signature: None,
            }],
        }]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.contents.len(), 1);
        let part = &body.contents[0].parts[0];
        assert!(
            part.function_call.is_some(),
            "ToolUse must be encoded as functionCall"
        );
        let fc = part.function_call.as_ref().unwrap();
        assert_eq!(fc.name, "get_weather");
    }

    #[test]
    fn encode_top_p_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.top_p = Some(0.9);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let config = body.generation_config.unwrap();
        assert_eq!(config.top_p, Some(0.9));
    }

    #[test]
    fn encode_input_schema_coercion() {
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
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let params = body.tools[0].function_declarations[0]
            .parameters
            .as_ref()
            .unwrap();
        assert_eq!(params["type"], "object");
    }

    #[test]
    fn usage_mapping() {
        let usage = GeminiUsage {
            prompt_token_count: 100,
            candidates_token_count: 50,
            total_token_count: 150,
        };
        let core_usage = build_gemini_usage(&usage);
        assert_eq!(core_usage.input_tokens, 100);
        assert_eq!(core_usage.output_tokens, 50);
        assert_eq!(core_usage.provenance, UsageProvenance::ProviderReported);
    }

    #[test]
    fn stream_full_lifecycle_ordering() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":null}],"usageMetadata":null}"#,
            ),
            make_frame(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
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
    fn encode_stop_sequences_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.stop = Some(vec!["END".into(), "STOP".into()]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let config = body.generation_config.unwrap();
        assert_eq!(
            config.stop_sequences,
            Some(vec!["END".into(), "STOP".into()])
        );
    }

    #[test]
    fn encode_tool_result_lookup_function_name() {
        let core = make_core_request(vec![
            CoreMessage {
                role: CoreRole::Assistant,
                content: vec![CoreContent::ToolUse {
                    id: "gemini_call_0".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "SF"}),
                    signature: None,
                }],
            },
            CoreMessage {
                role: CoreRole::Tool,
                content: vec![CoreContent::ToolResult {
                    tool_use_id: "gemini_call_0".into(),
                    content: vec![CoreContent::Text {
                        text: "72F".into(),
                        cache: None,
                    }],
                    is_error: false,
                    cache: None,
                }],
            },
        ]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let fr = body
            .contents
            .iter()
            .find_map(|c| c.parts.iter().find_map(|p| p.function_response.clone()));
        assert!(fr.is_some(), "expected function_response part");
        assert_eq!(fr.unwrap().name, "get_weather");
    }

    #[test]
    fn stream_unknown_event_skipped() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(r#"{"candidates":[],"usageMetadata":null}"#);
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStart { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, CoreEvent::TextDelta { .. }))
        );
    }

    #[test]
    fn decode_stop_reason_max_tokens() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "truncated"}]
                },
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn decode_empty_content_gets_default_text() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": []
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
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
    fn stream_finish_emits_tool_use_for_unclosed_blocks() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},"finishReason":null}],"usageMetadata":null}"#,
        );
        decoder.decode_frame(&frame).unwrap();

        let events = decoder.finish().unwrap();
        let msg_stop = events
            .iter()
            .find(|e| matches!(e, CoreEvent::MessageStop { .. }));
        let Some(CoreEvent::MessageStop { stop_reason, .. }) = msg_stop else {
            panic!("expected a MessageStop event");
        };
        assert_eq!(*stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn encode_reasoning_effort_omitted() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.reasoning_effort = Some("high".into());
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: GeminiRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let config = body.generation_config.unwrap();
        assert_eq!(config.temperature, None);
    }

    #[test]
    fn decode_function_call_tool_use_id_format() {
        let target = make_target();
        let resp_json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "search", "args": {"q": "test"}}}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });
        let bytes = serde_json::to_vec(&resp_json).unwrap();
        let adapter = GeminiAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        match &core_resp.content[0] {
            CoreContent::ToolUse { id, name, .. } => {
                assert!(
                    id.starts_with("gemini_call_"),
                    "Gemini tool use IDs should start with gemini_call_"
                );
                assert_eq!(name, "search");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn encode_uses_upstream_model() {
        let mut target = make_target();
        target.upstream_model = "gemini-2.5-flash-preview-05-20".into();
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let adapter = GeminiAdapter;
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        assert!(proxy_req.url.contains("gemini-2.5-flash-preview-05-20"));
    }

    #[test]
    fn encode_stream_flag_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.stream = true;
        let target = make_target();
        let adapter = GeminiAdapter;
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        assert!(
            proxy_req.stream,
            "stream flag should be forwarded to ProxyRequest"
        );
    }

    #[test]
    fn encode_empty_messages() {
        let core = make_core_request(vec![]);
        let adapter = GeminiAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["contents"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn decode_malformed_json_returns_error() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let result = adapter.decode_response(b"not valid json {{{", &target);
        assert!(result.is_err(), "malformed JSON should produce an error");
    }

    #[test]
    fn decode_empty_bytes_returns_error() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let result = adapter.decode_response(b"", &target);
        assert!(result.is_err(), "empty bytes should produce an error");
    }

    #[test]
    fn stream_no_duplicate_tool_call_stop() {
        let target = make_target();
        let adapter = GeminiAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![make_frame(
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"search","args":{"q":"rust"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#,
        )];

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

    /// A Gemini-dialect client wire, so these tests exercise the same-format
    /// passthrough branch rather than the cross-format encode.
    fn gemini_client_wire(body: serde_json::Value) -> ClientWireRequest {
        ClientWireRequest::new(ClientWireFormat::GeminiGenerateContent, body)
    }

    fn same_format_core_request() -> CoreRequest {
        make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }])
    }

    #[test]
    fn same_format_thinking_budget_is_clamped_to_model_max() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 100_000}},
        });
        let target = make_target();
        let client = gemini_client_wire(body);
        let proxy_req = GeminiAdapter
            .encode_request(&same_format_core_request(), &target, &client)
            .unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(
            wire["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            serde_json::json!(32_768),
            "a Gemini client's thinkingBudget must be clamped to gemini-2.5-pro's max: {wire}"
        );
    }

    #[test]
    fn same_format_thinking_budget_minus_one_and_zero_pass_through() {
        for budget in [-1i64, 0] {
            let body = serde_json::json!({
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"thinkingConfig": {"thinkingBudget": budget}},
            });
            let target = make_target();
            let client = gemini_client_wire(body);
            let proxy_req = GeminiAdapter
                .encode_request(&same_format_core_request(), &target, &client)
                .unwrap();
            let wire: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
            assert_eq!(
                wire["generationConfig"]["thinkingConfig"]["thinkingBudget"],
                serde_json::json!(budget),
                "the Gemini sentinels -1 (dynamic) and 0 (disabled) must not be clamped: {wire}"
            );
        }
    }

    #[test]
    fn same_format_without_thinking_config_is_byte_identical() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"temperature": 0.5, "topP": 0.9},
        });
        let target = make_target();
        let client = gemini_client_wire(body);
        let proxy_req = GeminiAdapter
            .encode_request(&same_format_core_request(), &target, &client)
            .unwrap();
        assert_eq!(
            proxy_req.body,
            serde_json::to_vec(&*client.body).unwrap(),
            "same-format passthrough without a thinkingConfig must forward the client bytes unchanged"
        );
    }

    #[test]
    fn same_format_snake_case_thinking_budget_is_clamped() {
        let body = serde_json::json!({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generation_config": {"thinking_config": {"thinking_budget": 100_000}},
        });
        let target = make_target();
        let client = gemini_client_wire(body);
        let proxy_req = GeminiAdapter
            .encode_request(&same_format_core_request(), &target, &client)
            .unwrap();
        let wire: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(
            wire["generation_config"]["thinking_config"]["thinking_budget"],
            serde_json::json!(32_768),
            "the snake_case spelling must be located and clamped in place: {wire}"
        );
        assert!(
            wire["generation_config"]["thinking_config"]
                .get("thinkingBudget")
                .is_none(),
            "clamping must not invent a second camelCase key: {wire}"
        );
    }

    #[test]
    fn same_format_non_integer_thinking_budget_is_rejected() {
        for budget in [
            serde_json::json!("100000"),
            serde_json::json!(100_000.5),
            serde_json::json!({"budget": 100_000}),
        ] {
            let body = serde_json::json!({
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"thinkingConfig": {"thinkingBudget": budget}},
            });
            let target = make_target();
            let client = gemini_client_wire(body);
            let err = GeminiAdapter
                .encode_request(&same_format_core_request(), &target, &client)
                .expect_err("a non-integer thinkingBudget must be rejected, not forwarded");
            assert!(
                matches!(err, ProviderError::InvalidRequest(_)),
                "expected InvalidRequest, got: {err:?}"
            );
            assert!(
                err.to_string()
                    .contains("thinkingBudget must be an integer"),
                "the rejection must name the field and the expectation, got: {err}"
            );
        }
    }

    #[test]
    fn adapter_source_no_forbidden_imports() {
        let source = include_str!("gemini.rs");
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
