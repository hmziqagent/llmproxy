//! OpenAI Responses API provider adapter: translates core requests, responses,
//! and event streams to and from the OpenAI Responses wire format.

use std::collections::HashMap;

use llm_proxy_protocol::core::{
    ContentKind, CoreContent, CoreEvent, CoreRequest, CoreResponse, CoreRole, CoreToolChoice,
    ModelRef, StopReason, Usage, UsageProvenance,
};
use llm_proxy_protocol::zen::{
    ResponsesChunk, ResponsesInput, ResponsesOutput, ResponsesReasoning, ResponsesRequest,
    ResponsesResponse, ResponsesTool, ResponsesUsage,
};

use super::{
    ProviderAdapterTarget, ProviderStreamDecoder, build_proxy_request, expand_url_template,
    mark_tool_error, response_model_ref,
};
use crate::ProtocolName;
use crate::error::ProviderError;
use crate::protocols;
use crate::sse::SseFrame;

/// Adapter for the OpenAI Responses API.
#[derive(Debug, Clone, Default)]
pub struct ResponsesAdapter;

/// Stateful stream decoder translating OpenAI Responses API SSE frames into `CoreEvent`s.
#[derive(Debug)]
pub struct ResponsesStreamDecoder {
    model_ref: ModelRef,
    started: bool,
    content_index: usize,
    content_started: bool,
    tool_blocks: HashMap<usize, usize>,
    saw_tool_call: bool,
    stop_sent: bool,
    saw_terminal: bool,
}

impl ProviderStreamDecoder for ResponsesStreamDecoder {
    #[expect(
        clippy::too_many_lines,
        reason = "one match arm per Responses API stream event: the wire shape \
                 is a single translation table, and splitting the arms into \
                 helpers would separate each event's field mapping from the \
                 frame layout it produces"
    )]
    fn decode_frame(&mut self, frame: &SseFrame) -> Result<Vec<CoreEvent>, ProviderError> {
        let data = frame.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(vec![]);
        }

        let chunk: ResponsesChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(_) => {
                let truncated = super::truncate_str_safe(data, 200);
                tracing::warn!(data = truncated, "malformed Responses chunk, skipping");
                return Ok(vec![]);
            }
        };

        let mut events = Vec::new();

        match chunk.r#type.as_str() {
            "response.created" | "response.in_progress" => {
                if !self.started {
                    self.started = true;
                    events.push(CoreEvent::MessageStart {
                        id: chunk.id.clone(),
                        model: self.model_ref.clone(),
                    });
                }
            }
            "response.output_item.added" => {
                let oi = chunk.output_index.unwrap_or(0);
                // Documented shape: the added item arrives as `item`.
                if let Some(item) = chunk.item.as_ref() {
                    if item.r#type == "message" {
                        self.start_text_block_if_needed(item, &mut events);
                    } else if item.r#type == "function_call" {
                        self.start_function_call(oi, item, &mut events);
                    }
                }
                for output in chunk.output.iter().flatten() {
                    if output.r#type == "message" {
                        self.start_text_block_if_needed(output, &mut events);
                    } else if output.r#type == "function_call" {
                        self.start_function_call(oi, output, &mut events);
                    }
                }
            }
            "response.output_text.delta" => {
                if let Some(ref delta) = chunk.delta
                    && !delta.is_empty()
                {
                    if !self.content_started {
                        self.content_started = true;
                        events.push(CoreEvent::ContentStart {
                            index: self.content_index,
                            kind: ContentKind::Text,
                        });
                    }
                    events.push(CoreEvent::TextDelta {
                        index: self.content_index,
                        text: delta.clone(),
                    });
                }
            }
            "response.output_text.done" => {
                if self.content_started {
                    self.content_started = false;
                    self.content_index += 1;
                }
            }
            "response.function_call_arguments.delta" => {
                if let Some(ref delta) = chunk.delta
                    && !delta.is_empty()
                {
                    let oi = chunk.output_index.unwrap_or(0);
                    let block_idx = self.ensure_tool_block(
                        oi,
                        "no prior response.output_item.added event",
                        &mut events,
                    );
                    events.push(CoreEvent::ToolCallDelta {
                        index: block_idx,
                        args_delta: delta.clone(),
                    });
                }
            }
            "response.function_call_arguments.done" => {
                let oi = chunk.output_index.unwrap_or(0);
                self.ensure_tool_block(
                    oi,
                    "no prior response.output_item.added event before arguments.done",
                    &mut events,
                );
                if let Some(block_idx) = self.tool_blocks.remove(&oi) {
                    events.push(CoreEvent::ToolCallStop { index: block_idx });
                }
            }
            "response.completed" | "response.done" => {
                self.close_content_if_open();
                self.close_tool_blocks(&mut events);

                // Documented shape: terminal events carry the whole response
                // object, with usage and output inside it.
                let usage = chunk
                    .response
                    .as_ref()
                    .and_then(|r| r.usage.as_ref())
                    .or(chunk.usage.as_ref());
                if let Some(usage) = usage {
                    events.push(CoreEvent::UsageDelta {
                        usage: build_responses_usage(usage),
                    });
                }

                if !self.stop_sent {
                    self.stop_sent = true;
                    self.saw_terminal = true;
                    let outputs = chunk
                        .response
                        .as_ref()
                        .map(|r| r.output.as_slice())
                        .or(chunk.output.as_deref());
                    let stop_reason = self.infer_stop_reason(outputs);
                    events.push(CoreEvent::MessageStop {
                        stop_reason,
                        stop_sequence: None,
                    });
                }
            }
            "response.failed" => {
                let error_obj = chunk
                    .error
                    .as_ref()
                    .or(chunk.response.as_ref().and_then(|r| r.error.as_ref()));
                let message = error_obj
                    .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                    .unwrap_or("response failed")
                    .to_owned();
                let kind = error_obj
                    .and_then(|e| e.get("code").and_then(|c| c.as_str()))
                    .map(|code| match code {
                        "rate_limit_exceeded" | "429" => {
                            llm_proxy_protocol::core::CoreStreamErrorKind::RateLimit
                        }
                        "invalid_request_error" | "400" => {
                            llm_proxy_protocol::core::CoreStreamErrorKind::InvalidRequest
                        }
                        "authentication_error" | "401" => {
                            llm_proxy_protocol::core::CoreStreamErrorKind::Authentication
                        }
                        "server_error" | "502" | "503" => {
                            llm_proxy_protocol::core::CoreStreamErrorKind::Upstream
                        }
                        _ => llm_proxy_protocol::core::CoreStreamErrorKind::Upstream,
                    })
                    .unwrap_or(llm_proxy_protocol::core::CoreStreamErrorKind::Upstream);
                events.push(CoreEvent::Error {
                    error: llm_proxy_protocol::core::CoreStreamError::new(kind, message),
                });
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

        self.close_content_if_open();

        self.close_tool_blocks(&mut events);

        if !self.stop_sent {
            self.stop_sent = true;
            let stop_reason = if self.saw_tool_call {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            };
            events.push(CoreEvent::MessageStop {
                stop_reason,
                stop_sequence: None,
            });
        }

        Ok(events)
    }

    fn saw_terminal(&self) -> bool {
        self.saw_terminal
    }
}

impl ResponsesStreamDecoder {
    fn close_content_if_open(&mut self) {
        if self.content_started {
            self.content_started = false;
            self.content_index += 1;
        }
    }

    fn start_text_block_if_needed(
        &mut self,
        output: &ResponsesOutput,
        events: &mut Vec<CoreEvent>,
    ) {
        if self.content_started {
            return;
        }
        let has_text = output
            .content
            .iter()
            .flatten()
            .any(|content| content.r#type == "output_text");
        if has_text {
            self.content_started = true;
            events.push(CoreEvent::ContentStart {
                index: self.content_index,
                kind: ContentKind::Text,
            });
        }
    }

    fn start_function_call(
        &mut self,
        oi: usize,
        output: &ResponsesOutput,
        events: &mut Vec<CoreEvent>,
    ) {
        self.saw_tool_call = true;
        if self.tool_blocks.contains_key(&oi) {
            return;
        }
        self.close_content_if_open();
        let call_id = output.call_id.clone().unwrap_or_else(|| {
            tracing::warn!("Responses: function_call output missing call_id");
            "<unknown_tool_id>".to_owned()
        });
        let name = output.name.clone().unwrap_or_else(|| {
            tracing::warn!("Responses: function_call output missing name");
            "<unknown_tool>".to_owned()
        });
        let block_idx = self.content_index;
        self.tool_blocks.insert(oi, block_idx);
        self.content_index += 1;
        events.push(CoreEvent::ToolCallStart {
            index: block_idx,
            id: call_id,
            name,
        });
    }

    fn ensure_tool_block(
        &mut self,
        oi: usize,
        warn_message: &'static str,
        events: &mut Vec<CoreEvent>,
    ) -> usize {
        if let Some(&block_idx) = self.tool_blocks.get(&oi) {
            return block_idx;
        }
        self.close_content_if_open();
        let block_idx = self.content_index;
        let synthetic_id = format!("__responses_missing_{block_idx}__");
        tracing::warn!(
            synthetic_id = synthetic_id,
            output_index = oi,
            reason = warn_message,
            "Responses: emitting synthetic ToolCallStart"
        );
        self.tool_blocks.insert(oi, block_idx);
        self.content_index += 1;
        events.push(CoreEvent::ToolCallStart {
            index: block_idx,
            id: synthetic_id,
            name: "<missing_function_name>".to_owned(),
        });
        block_idx
    }

    fn close_tool_blocks(&mut self, events: &mut Vec<CoreEvent>) {
        let mut indices: Vec<_> = self.tool_blocks.values().copied().collect();
        indices.sort_unstable();
        for idx in indices {
            events.push(CoreEvent::ToolCallStop { index: idx });
        }
        self.tool_blocks.clear();
    }

    fn infer_stop_reason(&self, outputs: Option<&[ResponsesOutput]>) -> StopReason {
        if let Some(outs) = outputs {
            if outs.iter().any(|o| o.r#type == "function_call") {
                return StopReason::ToolUse;
            }
        }
        if self.saw_tool_call {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        }
    }
}

fn encode_content_item<'a>(
    c: &'a CoreContent,
    role: &str,
    text_parts: &mut Vec<&'a str>,
    input: &mut Vec<ResponsesInput>,
) {
    match c {
        CoreContent::Text { text, cache } => {
            if cache.is_some() {
                tracing::warn!("Responses: cache_control is not supported, dropping cache marker");
            }
            if !text.is_empty() {
                text_parts.push(text.as_str());
            }
        }
        CoreContent::ToolUse {
            id,
            name,
            input: tool_input,
            signature: None,
        } => {
            if !text_parts.is_empty() {
                input.push(ResponsesInput::Message {
                    role: role.to_owned(),
                    content: Some(serde_json::Value::String(text_parts.join(""))),
                });
                text_parts.clear();
            }

            let arguments = serde_json::to_string(tool_input).unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    "Responses: failed to serialize tool input, falling back to empty object"
                );
                "{}".to_owned()
            });
            input.push(ResponsesInput::FunctionCall {
                r#type: "function_call".to_owned(),
                call_id: id.clone(),
                name: name.clone(),
                arguments,
            });
        }
        CoreContent::ToolResult {
            tool_use_id,
            content: result_content,
            is_error,
            ..
        } => {
            let result_text: String = result_content
                .iter()
                .filter_map(|rc| match rc {
                    CoreContent::Text { text, .. } => Some(text.as_str()),
                    other => {
                        tracing::warn!(
                            ?other,
                            tool_use_id,
                            "Responses: dropping non-text content block in ToolResult encoding"
                        );
                        None
                    }
                })
                .collect();
            // `function_call_output` has no error flag: a failure rides the
            // output as the tool-error marker.
            input.push(ResponsesInput::FunctionCallOutput {
                r#type: "function_call_output".to_owned(),
                call_id: tool_use_id.clone(),
                output: mark_tool_error(result_text, *is_error),
            });
        }
        _ => {
            tracing::warn!(
                role,
                ?c,
                "dropping unsupported content block during Responses encode"
            );
        }
    }
}

fn push_message_text(output: &ResponsesOutput, content: &mut Vec<CoreContent>) {
    for block in output.content.iter().flatten() {
        if block.r#type != "output_text" {
            tracing::warn!(
                block_type = block.r#type,
                "Responses: unknown content block type in message output, extracting text if present"
            );
        }
        if let Some(ref text) = block.text {
            content.push(CoreContent::Text {
                text: text.clone(),
                cache: None,
            });
        }
    }
}

impl ResponsesAdapter {
    /// Encode a core request into a Responses API request.
    #[expect(
        clippy::too_many_lines,
        reason = "field-by-field wire encode of the Responses API request shape, \
                 kept in one place so the mapping can be audited against the \
                 API schema without jumping between helpers"
    )]
    pub fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        client: &super::client_wire::ClientWireRequest,
    ) -> Result<super::ProxyRequest, ProviderError> {
        // Same-format passthrough (plan 08 step 1): a Responses-dialect
        // client forwards its raw body with only the surgical edits — the
        // model alias; everything else (`instructions`, `include`,
        // `previous_response_id`, `store`, ...) survives byte-for-byte.
        let url = expand_url_template(&target.endpoint, target)?;
        if client.same_format_as(&ProtocolName::new(protocols::OPENAI_RESPONSES))
            && let Some(raw) = client.body_object()
        {
            let mut obj = raw.clone();
            obj.insert(
                "model".to_owned(),
                serde_json::Value::String(target.upstream_model.clone()),
            );
            let mut value = serde_json::Value::Object(obj);
            if let Some(hygiene) = &target.prompt_cache {
                super::mutation::apply_prompt_hygiene(&mut value, hygiene, false);
            }
            let body = serde_json::to_vec(&value)?;
            return Ok(super::build_proxy_request(
                body,
                target,
                core.stream,
                url,
                client,
            ));
        }

        // Every other client dialect is cross-format here: settle the
        // leftover client fields first.
        let carried = super::client_wire::apply_field_policy(
            client,
            &ProtocolName::new(protocols::OPENAI_RESPONSES),
            target.lossy,
        )?;
        super::client_wire::check_cross_format_content(
            core,
            &ProtocolName::new(protocols::OPENAI_RESPONSES),
            target.lossy,
        )?;
        let mut input = Vec::new();

        if !core.system.is_empty() {
            for c in &core.system {
                if !matches!(c, CoreContent::Text { .. }) {
                    tracing::warn!(
                        ?c,
                        "Responses: dropping non-Text system content block during encode"
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
                input.push(ResponsesInput::Message {
                    role: "developer".to_owned(),
                    content: Some(serde_json::Value::String(system_text)),
                });
            }
        }

        for msg in &core.messages {
            let role = match msg.role {
                CoreRole::User => "user",
                CoreRole::Assistant => "assistant",
                CoreRole::System => "developer",
                CoreRole::Tool => "user",
                _ => "user",
            };

            let mut text_parts = Vec::new();

            for c in &msg.content {
                encode_content_item(c, role, &mut text_parts, &mut input);
            }

            if !text_parts.is_empty() {
                let text: String = text_parts.join("");
                input.push(ResponsesInput::Message {
                    role: role.to_owned(),
                    content: Some(serde_json::Value::String(text)),
                });
            }
        }

        let tools: Vec<ResponsesTool> = core
            .tools
            .iter()
            .map(|t| ResponsesTool {
                r#type: "function".to_owned(),
                name: Some(t.name.clone()),
                description: t.description.clone(),
                parameters: Some(if !t.input_schema.is_object() {
                    serde_json::json!({"type": "object", "properties": {}})
                } else {
                    t.input_schema.clone()
                }),
            })
            .collect();

        // Reasoning translation: an effort level is validated (or derived
        // from an Anthropic `thinking` budget); `thinking` itself never
        // crosses to the Responses wire. An Anthropic `output_config.effort`
        // maps onto the same field.
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
        let reasoning = effort.as_ref().map(|effort| ResponsesReasoning {
            effort: Some(effort.clone()),
        });

        let mut req = ResponsesRequest {
            model: target.upstream_model.clone(),
            input,
            stream: if core.stream { Some(true) } else { None },
            tools,
            reasoning,
            tool_choice: None,
            temperature: core.sampling.temperature,
            top_p: core.sampling.top_p,
            max_output_tokens: core.sampling.max_tokens,
            extra: carried,
        };

        if core.sampling.stop.as_ref().is_some_and(|s| !s.is_empty()) {
            tracing::warn!(
                "Responses: stop sequences specified but not forwarded; \
                 the Responses API does not support this parameter"
            );
        }
        if core.metadata.user_id.is_some() {
            tracing::warn!(
                "Responses: user_id specified but not forwarded; \
                 the Responses API does not support this parameter"
            );
        }

        if let Some(ref tc) = core.tool_choice {
            req.tool_choice = match tc {
                CoreToolChoice::Auto => Some(serde_json::json!("auto")),
                CoreToolChoice::Any => Some(serde_json::json!("required")),
                CoreToolChoice::None => Some(serde_json::json!("none")),
                CoreToolChoice::Tool { name } => Some(serde_json::json!({
                    "type": "function",
                    "name": name
                })),
                CoreToolChoice::Raw(v) => Some(v.clone()),
                _ => {
                    tracing::warn!(?tc, "Responses: unknown tool_choice variant, omitting");
                    None
                }
            };
        }

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

    /// Decode a Responses API response body.
    pub fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, ProviderError> {
        let resp: ResponsesResponse = serde_json::from_slice(bytes)?;

        let mut content = Vec::new();

        for output in &resp.output {
            match output.r#type.as_str() {
                "message" => {
                    push_message_text(output, &mut content);
                }
                "function_call" => {
                    let input = output
                        .arguments
                        .as_ref()
                        .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
                        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

                    content.push(CoreContent::ToolUse {
                        id: output.call_id.clone().unwrap_or_default(),
                        name: output.name.clone().unwrap_or_default(),
                        input,
                        signature: None,
                    });
                }
                _ => {
                    tracing::warn!(
                        output_type = output.r#type,
                        "Responses: skipping unknown output type in decode_response"
                    );
                }
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
        let stop_reason = match resp.status.as_deref() {
            Some("failed") | Some("expired") => StopReason::Error,
            Some("incomplete") => StopReason::MaxTokens,
            _ if has_tool_use => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        };

        let usage = resp
            .usage
            .as_ref()
            .map(build_responses_usage)
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
    pub fn new_stream_decoder(&self, target: &ProviderAdapterTarget) -> ResponsesStreamDecoder {
        ResponsesStreamDecoder {
            model_ref: response_model_ref(target),
            started: false,
            content_index: 0,
            content_started: false,
            tool_blocks: HashMap::new(),
            saw_tool_call: false,
            stop_sent: false,
            saw_terminal: false,
        }
    }
}

fn build_responses_usage(usage: &ResponsesUsage) -> Usage {
    let cache_read = usage
        .input_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens);
    let cache_read_i64 = cache_read.unwrap_or(0) as i64;
    let input = (usage.input_tokens as i64) - cache_read_i64;
    Usage {
        input_tokens: i32::try_from(input.max(0)).unwrap_or(i32::MAX),
        output_tokens: usage.output_tokens,
        reasoning_tokens: usage
            .output_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cache_read,
        provenance: UsageProvenance::ProviderReported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_wire::{ClientWireFormat, ClientWireRequest};

    /// An Anthropic-dialect client wire: never the same format as the
    /// Responses adapter, so tests exercise the cross-format encode.
    fn client_wire(core: &CoreRequest) -> ClientWireRequest {
        ClientWireRequest::new(
            ClientWireFormat::AnthropicMessages,
            serde_json::json!({"model": core.model.requested, "messages": []}),
        )
    }
    use llm_proxy_core::AuthStyle;
    use llm_proxy_protocol::core::{CoreMessage, CoreTool, SamplingOptions};
    use llm_proxy_protocol::zen::ResponsesContent;

    fn make_target() -> ProviderAdapterTarget {
        ProviderAdapterTarget {
            provider_name: "test-responses".into(),
            adapter_name: "openai-responses".into(),
            protocol: super::ProtocolName::new(protocols::OPENAI_RESPONSES),
            endpoint: "https://api.openai.com/v1/responses".into(),
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
    fn documented_output_item_added_starts_tool_call_with_real_identity() {
        // The documented SSE shape puts the added item under `item`; the
        // decoder must not synthesize "<missing_function_name>" identities.
        let target = make_target();
        let mut decoder = ResponsesAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_9","name":"get_weather","arguments":""}}"#.to_owned(),
        };
        let events = decoder.decode_frame(&frame).unwrap();
        let start = events
            .iter()
            .find_map(|e| match e {
                CoreEvent::ToolCallStart { id, name, .. } => Some((id.clone(), name.clone())),
                _ => None,
            })
            .expect("ToolCallStart from output_item.added item");
        assert_eq!(start.0, "call_9");
        assert_eq!(start.1, "get_weather");
    }

    #[test]
    fn documented_completed_event_carries_usage_inside_response() {
        let target = make_target();
        let mut decoder = ResponsesAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"type":"response.completed","response":{"id":"resp_1","object":"response","created":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"input_tokens_details":{"cached_tokens":3}}}}"#.to_owned(),
        };
        let events = decoder.decode_frame(&frame).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                CoreEvent::UsageDelta { usage } => Some(usage.clone()),
                _ => None,
            })
            .expect("usage from response.completed");
        assert_eq!(usage.input_tokens, 7);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cache_read_input_tokens, Some(3));
        assert!(decoder.saw_terminal());
    }

    #[test]
    fn documented_failed_event_carries_error_inside_response() {
        let target = make_target();
        let mut decoder = ResponsesAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"type":"response.failed","response":{"id":"resp_f","error":{"code":"rate_limit_exceeded","message":"slow down"}}}"#.to_owned(),
        };
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(matches!(
            &events[0],
            CoreEvent::Error { error } if error.message() == "slow down"
        ));
    }

    #[test]
    fn truncated_responses_stream_has_no_terminal() {
        let target = make_target();
        let mut decoder = ResponsesAdapter.new_stream_decoder(&target);
        let frame = SseFrame {
            event: None,
            id: None,
            data: r#"{"type":"response.output_text.delta","delta":"hi"}"#.to_owned(),
        };
        decoder.decode_frame(&frame).unwrap();
        decoder.finish().unwrap();
        assert!(
            !decoder.saw_terminal(),
            "a stream cut before response.completed must not count as terminal"
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
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.model, "gpt-4o-2024-08-06");
        assert_eq!(body.input.len(), 1);
        match &body.input[0] {
            ResponsesInput::Message { role, .. } => assert_eq!(role, "user"),
            other => panic!("expected Message input item, got {other:?}"),
        }
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
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        match &body.input[0] {
            ResponsesInput::Message { role, .. } => assert_eq!(role, "developer"),
            other => panic!("expected Message input item, got {other:?}"),
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
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.tools.len(), 1);
        assert_eq!(body.tools[0].name, Some("get_weather".to_owned()));
    }

    #[test]
    fn encode_reasoning_effort() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.sampling.reasoning_effort = Some("high".into());
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(body.reasoning.is_some());
        assert_eq!(body.reasoning.unwrap().effort, Some("high".to_owned()));
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
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.stream, Some(true));
        assert!(proxy_req.stream);
    }

    #[test]
    fn decode_text_response() {
        let target = make_target();
        let resp = ResponsesResponse {
            id: "resp_test".into(),
            object: "response".into(),
            created: 12345,
            model: "gpt-4o-2024-08-06".into(),
            output: vec![ResponsesOutput {
                r#type: "message".into(),
                id: Some("msg_1".into()),
                role: Some("assistant".into()),
                content: Some(vec![ResponsesContent {
                    r#type: "output_text".into(),
                    text: Some("hello world".into()),
                }]),
                call_id: None,
                name: None,
                arguments: None,
            }],
            usage: Some(ResponsesUsage {
                input_tokens: 100,
                output_tokens: 50,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            status: None,
            error: None,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = ResponsesAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.id, Some("resp_test".to_owned()));
        assert_eq!(core_resp.stop_reason, StopReason::EndTurn);
        assert_eq!(core_resp.usage.input_tokens, 100);
        assert_eq!(core_resp.content.len(), 1);
    }

    #[test]
    fn decode_response_absent_or_null_usage_is_labelled_synthetic_zero() {
        // "The provider did not report usage" must be distinguishable from
        // "the provider reported zeros": only the latter is bookable as real.
        let target = make_target();
        let adapter = ResponsesAdapter;
        for body in [
            r#"{"id":"resp_1","object":"response","created":1,"model":"gpt-4o","status":"completed","output":[]}"#,
            r#"{"id":"resp_1","object":"response","created":1,"model":"gpt-4o","status":"completed","output":[],"usage":null}"#,
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

        let reported_zeros = r#"{"id":"resp_1","object":"response","created":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":0,"output_tokens":0}}"#;
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
    fn decode_function_call_response() {
        let target = make_target();
        let resp = ResponsesResponse {
            id: "resp_test".into(),
            object: "response".into(),
            created: 0,
            model: "gpt-4o".into(),
            output: vec![ResponsesOutput {
                r#type: "function_call".into(),
                id: Some("fc_1".into()),
                role: None,
                content: None,
                call_id: Some("call_123".into()),
                name: Some("get_weather".into()),
                arguments: Some(r#"{"city":"SF"}"#.into()),
            }],
            usage: Some(ResponsesUsage {
                input_tokens: 200,
                output_tokens: 80,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            status: None,
            error: None,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = ResponsesAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.stop_reason, StopReason::ToolUse);
        match &core_resp.content[0] {
            CoreContent::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn decode_preserves_requested_model() {
        let mut target = make_target();
        target.requested_model = "my-model".into();

        let resp = ResponsesResponse {
            id: "test".into(),
            object: "response".into(),
            created: 0,
            model: "gpt-4o-2024-08-06".into(),
            output: vec![ResponsesOutput {
                r#type: "message".into(),
                id: None,
                role: Some("assistant".into()),
                content: Some(vec![ResponsesContent {
                    r#type: "output_text".into(),
                    text: Some("hi".into()),
                }]),
                call_id: None,
                name: None,
                arguments: None,
            }],
            usage: Some(ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            status: None,
            error: None,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = ResponsesAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();

        assert_eq!(core_resp.model.requested, "my-model");
    }

    #[test]
    fn stream_text_decoding() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(r#"{"type":"response.created","id":"resp_1"}"#),
            make_frame(r#"{"type":"response.output_text.delta","delta":"Hello"}"#),
            make_frame(r#"{"type":"response.output_text.delta","delta":" world"}"#),
            make_frame(
                r#"{"type":"response.completed","usage":{"input_tokens":10,"output_tokens":5}}"#,
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
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("[DONE]");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_finish_emits_lifecycle() {
        let target = make_target();
        let adapter = ResponsesAdapter;
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
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(r#"{"type":"response.created","id":"resp_1"}"#),
            make_frame(
                r#"{"type":"response.output_item.added","output":[{"type":"function_call","call_id":"call_abc","name":"get_weather"}]}"#,
            ),
            make_frame(r#"{"type":"response.function_call_arguments.delta","delta":"{\"city\":"}"#),
            make_frame(r#"{"type":"response.function_call_arguments.delta","delta":"\"SF\"}"}"#),
            make_frame(r#"{"type":"response.function_call_arguments.done"}"#),
            make_frame(
                r#"{"type":"response.completed","usage":{"input_tokens":20,"output_tokens":10}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallStart { .. })),
            "expected ToolCallStart"
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallDelta { .. })),
            "expected ToolCallDelta"
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::ToolCallStop { .. })),
            "expected ToolCallStop"
        );
        assert!(
            all_events
                .iter()
                .any(|e| matches!(e, CoreEvent::MessageStop { .. })),
            "expected MessageStop with ToolUse"
        );
    }

    #[test]
    fn stream_parallel_tool_calls_partitioned_by_output_index() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(r#"{"type":"response.created","id":"resp_1"}"#),
            make_frame(
                r#"{"type":"response.output_item.added","output_index":0,"output":[{"type":"function_call","call_id":"call_a","name":"get_weather"}]}"#,
            ),
            make_frame(
                r#"{"type":"response.output_item.added","output_index":1,"output":[{"type":"function_call","call_id":"call_b","name":"get_time"}]}"#,
            ),
            make_frame(
                r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}"#,
            ),
            make_frame(
                r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"tz\":"}"#,
            ),
            make_frame(
                r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"SF\"}"}"#,
            ),
            make_frame(
                r#"{"type":"response.function_call_arguments.delta","output_index":1,"delta":"\"UTC\"}"}"#,
            ),
            make_frame(r#"{"type":"response.function_call_arguments.done","output_index":1}"#),
            make_frame(r#"{"type":"response.function_call_arguments.done","output_index":0}"#),
            make_frame(
                r#"{"type":"response.completed","usage":{"input_tokens":40,"output_tokens":20}}"#,
            ),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        let starts: Vec<_> = all_events
            .iter()
            .filter_map(|e| match e {
                CoreEvent::ToolCallStart { index, id, name } => {
                    Some((*index, id.clone(), name.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(starts.len(), 2, "expected two ToolCallStart events");
        let indices: Vec<_> = starts.iter().map(|(i, _, _)| *i).collect();
        assert_eq!(indices, vec![0, 1], "expected distinct ascending indices");

        let mut args_by_index: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();
        for e in &all_events {
            if let CoreEvent::ToolCallDelta { index, args_delta } = e {
                *args_by_index.entry(*index).or_default() += args_delta;
            }
        }
        assert_eq!(
            &args_by_index[&0], r#"{"city":"SF"}"#,
            "output_index 0 args must be partitioned"
        );
        assert_eq!(
            &args_by_index[&1], r#"{"tz":"UTC"}"#,
            "output_index 1 args must be partitioned"
        );

        let stops: Vec<_> = all_events
            .iter()
            .filter_map(|e| match e {
                CoreEvent::ToolCallStop { index } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(stops.len(), 2, "expected two ToolCallStop events");
        assert!(stops.contains(&0) && stops.contains(&1));
    }

    #[test]
    fn stream_malformed_frame_skipped() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame("not valid json {{{");
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_response_failed() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame =
            make_frame(r#"{"type":"response.failed","error":{"message":"rate limit exceeded"}}"#);
        let events = decoder.decode_frame(&frame).unwrap();

        assert!(
            events.iter().any(|e| matches!(e, CoreEvent::Error { .. })),
            "response.failed must emit CoreEvent::Error"
        );
    }

    #[test]
    fn encode_tool_choice_forwarded() {
        let mut core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        core.tool_choice = Some(CoreToolChoice::Auto);
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert!(body.tool_choice.is_some(), "tool_choice must be forwarded");
        let tc = body.tool_choice.unwrap();
        assert_eq!(tc, serde_json::json!("auto"));
    }

    #[test]
    fn encode_tool_use_as_function_call() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Assistant,
            content: vec![CoreContent::ToolUse {
                id: "call_1".into(),
                name: "get_weather".into(),
                input: serde_json::json!({"city": "SF"}),
                signature: None,
            }],
        }]);
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.input.len(), 1);
        match &body.input[0] {
            ResponsesInput::FunctionCall {
                name, arguments, ..
            } => {
                assert_eq!(name, "get_weather");
                assert!(arguments.contains("SF"));
            }
            other => panic!("expected FunctionCall input item, got {other:?}"),
        }
    }

    #[test]
    fn encode_tool_result_as_function_call_output() {
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Tool,
            content: vec![CoreContent::ToolResult {
                tool_use_id: "call_1".into(),
                content: vec![CoreContent::Text {
                    text: "72F sunny".into(),
                    cache: None,
                }],
                is_error: false,
                cache: None,
            }],
        }]);
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body.input.len(), 1);
        match &body.input[0] {
            ResponsesInput::FunctionCallOutput {
                call_id, output, ..
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(output, "72F sunny");
            }
            other => panic!("expected FunctionCallOutput input item, got {other:?}"),
        }
    }

    /// Encodes a tool result and returns the `function_call_output` item's
    /// `output` string.
    fn encode_tool_result_output(result_text: &str, is_error: bool) -> String {
        let result_content = if result_text.is_empty() {
            Vec::new()
        } else {
            vec![CoreContent::Text {
                text: result_text.into(),
                cache: None,
            }]
        };
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::Tool,
            content: vec![CoreContent::ToolResult {
                tool_use_id: "call_1".into(),
                content: result_content,
                is_error,
                cache: None,
            }],
        }]);
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        match &body.input[..] {
            [
                ResponsesInput::FunctionCallOutput {
                    call_id, output, ..
                },
            ] => {
                assert_eq!(call_id, "call_1");
                output.clone()
            }
            other => panic!("expected one FunctionCallOutput input item, got {other:?}"),
        }
    }

    #[test]
    fn encode_failed_tool_result_marks_the_function_call_output() {
        let output = encode_tool_result_output("disk full", true);
        assert!(
            output.starts_with("[tool error] "),
            "a failed tool result must reach the model marked as failed: {output:?}"
        );
        assert!(
            output.contains("disk full"),
            "the tool's own output must survive: {output:?}"
        );
    }

    #[test]
    fn encode_successful_tool_result_output_has_no_error_marker() {
        assert_eq!(encode_tool_result_output("disk full", false), "disk full");
    }

    #[test]
    fn encode_failed_tool_result_without_output_still_says_it_failed() {
        assert_eq!(encode_tool_result_output("", true), "[tool error]");
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
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();

        let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
        let params = body.tools[0].parameters.as_ref().unwrap();
        assert_eq!(params["type"], "object");
    }

    #[test]
    fn stream_stop_reason_mapping_tool_use() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(r#"{"type":"response.created","id":"resp_1"}"#),
            make_frame(
                r#"{"type":"response.output_item.added","output":[{"type":"function_call","call_id":"call_1","name":"test"}]}"#,
            ),
            make_frame(r#"{"type":"response.function_call_arguments.done"}"#),
            make_frame(r#"{"type":"response.done","usage":{"input_tokens":10,"output_tokens":5}}"#),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }

        let msg_stop = all_events
            .iter()
            .find(|e| matches!(e, CoreEvent::MessageStop { .. }));
        let Some(CoreEvent::MessageStop { stop_reason, .. }) = msg_stop else {
            panic!("expected a MessageStop event");
        };
        assert_eq!(*stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn stream_unknown_event_skipped() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(r#"{"type":"some_new_event","data":"whatever"}"#);
        let events = decoder.decode_frame(&frame).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn stream_finish_emits_tool_use_if_saw_tool_call() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frames = vec![
            make_frame(r#"{"type":"response.created","id":"resp_1"}"#),
            make_frame(
                r#"{"type":"response.output_item.added","output":[{"type":"function_call","call_id":"c1","name":"fn"}]}"#,
            ),
            make_frame(r#"{"type":"response.function_call_arguments.done"}"#),
        ];

        let mut all_events = Vec::new();
        for frame in &frames {
            all_events.extend(decoder.decode_frame(frame).unwrap());
        }
        all_events.extend(decoder.finish().unwrap());

        let msg_stop = all_events
            .iter()
            .find(|e| matches!(e, CoreEvent::MessageStop { .. }));
        let Some(CoreEvent::MessageStop { stop_reason, .. }) = msg_stop else {
            panic!("expected a MessageStop event");
        };
        assert_eq!(
            *stop_reason,
            StopReason::ToolUse,
            "finish() should infer ToolUse when saw_tool_call is true"
        );
    }

    #[test]
    fn decode_empty_content_gets_default_text() {
        let target = make_target();
        let resp = ResponsesResponse {
            id: "test".into(),
            object: "response".into(),
            created: 0,
            model: "gpt-4o".into(),
            output: vec![],
            usage: Some(ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            status: None,
            error: None,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = ResponsesAdapter;
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
    fn encode_tool_choice_any_and_none() {
        for (tc, expected_value) in [
            (CoreToolChoice::Any, serde_json::json!("required")),
            (CoreToolChoice::None, serde_json::json!("none")),
        ] {
            let mut core = make_core_request(vec![CoreMessage {
                role: CoreRole::User,
                content: vec![CoreContent::Text {
                    text: "hi".into(),
                    cache: None,
                }],
            }]);
            core.tool_choice = Some(tc);
            let adapter = ResponsesAdapter;
            let target = make_target();
            let proxy_req = adapter
                .encode_request(&core, &target, &client_wire(&core))
                .unwrap();

            let body: ResponsesRequest = serde_json::from_slice(&proxy_req.body).unwrap();
            assert_eq!(body.tool_choice.unwrap(), expected_value);
        }
    }

    #[test]
    fn encode_uses_upstream_model() {
        let mut target = make_target();
        target.upstream_model = "gpt-4o-2024-08-06-alias".into();
        let core = make_core_request(vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hi".into(),
                cache: None,
            }],
        }]);
        let adapter = ResponsesAdapter;
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["model"], "gpt-4o-2024-08-06-alias");
    }

    #[test]
    fn encode_empty_messages() {
        let core = make_core_request(vec![]);
        let adapter = ResponsesAdapter;
        let target = make_target();
        let proxy_req = adapter
            .encode_request(&core, &target, &client_wire(&core))
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&proxy_req.body).unwrap();
        assert_eq!(body["input"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn decode_malformed_json_returns_error() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let result = adapter.decode_response(b"not valid json {{{", &target);
        assert!(result.is_err(), "malformed JSON should produce an error");
    }

    #[test]
    fn decode_empty_bytes_returns_error() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let result = adapter.decode_response(b"", &target);
        assert!(result.is_err(), "empty bytes should produce an error");
    }

    #[test]
    fn stream_response_failed_error_mapping() {
        let target = make_target();
        let adapter = ResponsesAdapter;
        let mut decoder = adapter.new_stream_decoder(&target);

        let frame = make_frame(
            r#"{"type":"response.failed","error":{"code":"rate_limit_exceeded","message":"Too many requests"}}"#,
        );
        let events = decoder.decode_frame(&frame).unwrap();
        let error_event = events.iter().find(|e| matches!(e, CoreEvent::Error { .. }));
        assert!(
            error_event.is_some(),
            "response.failed should produce an Error event"
        );
        if let CoreEvent::Error { error } = error_event.unwrap() {
            assert_eq!(
                error.kind,
                llm_proxy_protocol::core::CoreStreamErrorKind::RateLimit
            );
        }
    }

    #[test]
    fn decode_status_failed_gives_error_stop_reason() {
        let target = make_target();
        let resp = ResponsesResponse {
            id: "resp_fail".into(),
            object: "response".into(),
            created: 0,
            model: "gpt-4o".into(),
            output: vec![],
            usage: Some(ResponsesUsage {
                input_tokens: 0,
                output_tokens: 0,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            status: Some("failed".into()),
            error: None,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = ResponsesAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();
        assert_eq!(core_resp.stop_reason, StopReason::Error);
    }

    #[test]
    fn decode_status_incomplete_gives_max_tokens_stop_reason() {
        let target = make_target();
        let resp = ResponsesResponse {
            id: "resp_inc".into(),
            object: "response".into(),
            created: 0,
            model: "gpt-4o".into(),
            output: vec![],
            usage: Some(ResponsesUsage {
                input_tokens: 0,
                output_tokens: 0,
                input_tokens_details: None,
                output_tokens_details: None,
            }),
            status: Some("incomplete".into()),
            error: None,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let adapter = ResponsesAdapter;
        let core_resp = adapter.decode_response(&bytes, &target).unwrap();
        assert_eq!(core_resp.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn adapter_source_no_forbidden_imports() {
        let source = include_str!("responses.rs");
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
