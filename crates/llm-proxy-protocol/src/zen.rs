//! Responses API (OpenAI-style) and Gemini API (Google generate-content)
//! wire-format request/response types, including streaming chunk support.

use serde::{Deserialize, Serialize};

/// Top-level request body for the Responses API; outbound-only (the proxy
/// constructs it via `transform_to_responses()`), so unknown-field
/// rejection is not applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    /// Model identifier (e.g. `"gpt-4o"`).
    pub model: String,
    /// Conversation turns sent to the model.
    pub input: Vec<ResponsesInput>,
    /// When `Some(true)` the server streams chunked responses.
    pub stream: Option<bool>,
    /// Tools available to the model during generation.
    pub tools: Vec<ResponsesTool>,
    /// Optional reasoning / chain-of-thought configuration.
    pub reasoning: Option<ResponsesReasoning>,
    /// Controls which (if any) tool the model must call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Upper bound on tokens generated in the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,
    /// Carried client fields the Responses API accepts verbatim (e.g.
    /// `service_tier`, `parallel_tool_calls`, `store`, `prompt_cache_key`).
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// One entry in the Responses API `input` array: either a conversational
/// message or a typed item (`function_call` / `function_call_output`) per
/// the documented request shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    /// A conversational message (`{"role": ..., "content": ...}`).
    Message {
        /// Speaker role (e.g. `"user"`, `"assistant"`, `"developer"`).
        role: String,
        /// Arbitrary JSON content payload.
        content: Option<serde_json::Value>,
    },
    /// A function call the model previously emitted.
    FunctionCall {
        /// Always `"function_call"`.
        #[serde(rename = "type")]
        r#type: String,
        /// The call identifier echoed back from the response.
        call_id: String,
        /// Function name.
        name: String,
        /// JSON-encoded function arguments.
        arguments: String,
    },
    /// The result of a function call, fed back to the model.
    FunctionCallOutput {
        /// Always `"function_call_output"`.
        #[serde(rename = "type")]
        r#type: String,
        /// The call identifier this output belongs to.
        call_id: String,
        /// The function output payload.
        output: String,
    },
}

/// A tool definition offered to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesTool {
    /// Tool variant discriminator (e.g. `"function"`).
    #[serde(rename = "type")]
    pub r#type: String,
    /// Name of the tool.
    pub name: Option<String>,
    /// Human-readable description of what the tool does.
    pub description: Option<String>,
    /// JSON Schema describing the tool's parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Reasoning effort configuration for supported models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesReasoning {
    /// Effort level (e.g. `"low"`, `"medium"`, `"high"`).
    pub effort: Option<String>,
}

/// Non-streaming response from the Responses API, also embedded in stream
/// terminal events (`response.completed` / `response.done`); unknown
/// provider fields are silently ignored rather than failing decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    /// Unique response identifier.
    pub id: String,
    /// Object type discriminator (e.g. `"response"`).
    #[serde(default)]
    pub object: String,
    /// Unix timestamp (seconds) of creation.
    #[serde(default)]
    pub created: i64,
    /// Model that produced the response.
    #[serde(default)]
    pub model: String,
    /// Ordered list of output items (messages, tool calls, etc.).
    #[serde(default)]
    pub output: Vec<ResponsesOutput>,
    /// Token usage statistics; `None` on non-terminal events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
    /// Response status (e.g. `"completed"`, `"failed"`, `"incomplete"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Error object present when the response failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

/// A single output item in a Responses API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesOutput {
    /// Item type discriminator (e.g. `"message"`, `"function_call"`).
    #[serde(rename = "type")]
    pub r#type: String,
    /// Unique item identifier.
    pub id: Option<String>,
    /// Speaker role (present for message items).
    pub role: Option<String>,
    /// Content blocks (present for message items).
    pub content: Option<Vec<ResponsesContent>>,
    /// Function call identifier (present for function-call items).
    pub call_id: Option<String>,
    /// Function name (present for function-call items).
    pub name: Option<String>,
    /// JSON-encoded function arguments (present for function-call items).
    pub arguments: Option<String>,
}

/// A content block inside a Responses API output message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesContent {
    /// Content type discriminator (e.g. `"output_text"`).
    #[serde(rename = "type")]
    pub r#type: String,
    /// The text payload.
    pub text: Option<String>,
}

/// Token usage reported by the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    /// Number of tokens in the prompt.
    #[serde(default)]
    pub input_tokens: i32,
    /// Number of tokens in the completion.
    #[serde(default)]
    pub output_tokens: i32,
    /// OpenAI-documented prompt token breakdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<crate::openai::PromptTokensDetails>,
    /// OpenAI-documented completion token breakdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<crate::openai::CompletionTokensDetails>,
}

/// A single chunk in a streaming Responses API response; `delta` carries
/// incremental text or tool-call JSON arguments, interleaved across parallel
/// calls and disambiguated by `output_index`. Unknown fields are ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesChunk {
    /// Chunk type discriminator.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Response / item identifier.
    #[serde(default)]
    pub id: Option<String>,
    /// Incremental text delta.
    #[serde(default)]
    pub delta: Option<String>,
    /// Complete output items (emitted at certain events).
    #[serde(default)]
    pub output: Option<Vec<ResponsesOutput>>,
    /// Token usage (usually present on the final chunk).
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    /// Error object present on `response.failed` events.
    #[serde(default)]
    pub error: Option<serde_json::Value>,
    /// Upstream output-item index carried on function-call events; parallel
    /// function-call argument deltas are disambiguated via this index.
    /// Defaults to `None` so legacy/non-indexed chunks still parse.
    #[serde(default)]
    pub output_index: Option<usize>,
    /// The complete added output item on `response.output_item.added`
    /// (`{"type":"function_call","call_id":...,"name":...,"arguments":""}`),
    /// per the documented SSE shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<ResponsesOutput>,
    /// The complete response object on `response.created` /
    /// `response.completed` / `response.done` / `response.failed`, carrying
    /// `usage` and `output` per the documented SSE shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponsesResponse>,
}

/// Top-level request body for the Gemini generate-content endpoint;
/// outbound-only, so unknown-field rejection is not applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiRequest {
    /// Conversation contents to send to the model.
    pub contents: Vec<GeminiContent>,
    /// Optional generation parameters.
    pub generation_config: Option<GeminiGenerationConfig>,
    /// Tools available to the model.
    pub tools: Vec<GeminiTool>,
    /// Tool-calling control, carried from the core `tool_choice` as
    /// `toolConfig.functionCallingConfig`; emitted only when the request
    /// carries `tools`.
    #[serde(
        rename = "toolConfig",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_config: Option<serde_json::Value>,
    /// Streaming control for the adapter only; the Gemini wire format has no
    /// body field for it — streaming is selected by calling
    /// `:streamGenerateContent?alt=sse` instead of `:generateContent`, which
    /// the adapter does from this flag. Never serialized.
    #[serde(skip)]
    pub stream: Option<bool>,
}

/// A single message in a Gemini conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    /// Speaker role (e.g. `"user"`, `"model"`).
    pub role: String,
    /// Ordered content parts within this message; defaults to empty when
    /// omitted (Gemini may return candidates with no `parts`).
    #[serde(default)]
    pub parts: Vec<GeminiPart>,
}

/// A single content part inside a Gemini message; unknown fields are
/// ignored since Gemini may return part types the proxy does not model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GeminiPart {
    /// Text content of this part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Function call (present when the model invokes a tool); camelCase
    /// `functionCall` per the Gemini wire format.
    #[serde(
        rename = "functionCall",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub function_call: Option<GeminiFunctionCall>,
    /// Function response (present when sending a tool result back to the
    /// model); camelCase `functionResponse` per the Gemini wire format.
    #[serde(
        rename = "functionResponse",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub function_response: Option<GeminiFunctionResponse>,
    /// Opaque thought signature attached to a `functionCall` part (and echoed
    /// on the matching `functionResponse` part when replaying tool state);
    /// camelCase `thoughtSignature` per the Gemini wire format.
    #[serde(
        rename = "thoughtSignature",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub thought_signature: Option<String>,
    /// Inline media bytes (`inlineData`), used for base64 image parts.
    #[serde(
        rename = "inlineData",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub inline_data: Option<GeminiInlineData>,
    /// Media referenced by URI (`fileData`), used for URL image parts.
    #[serde(rename = "fileData", default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<GeminiFileData>,
}

/// A function call emitted by the Gemini model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

/// A function response sent back to the model as a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    /// The name of the function that was called.
    pub name: String,
    /// The response payload from the function.
    pub response: serde_json::Value,
}

impl GeminiPart {
    /// Create a text part.
    pub fn text(text: String) -> Self {
        Self {
            text: Some(text),
            function_call: None,
            function_response: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
        }
    }

    /// Create a function call part.
    pub fn function_call(name: String, args: Option<serde_json::Value>) -> Self {
        Self {
            text: None,
            function_call: Some(GeminiFunctionCall { name, args }),
            function_response: None,
            thought_signature: None,
            inline_data: None,
            file_data: None,
        }
    }

    /// Create a function response part.
    pub fn function_response(name: String, response: serde_json::Value) -> Self {
        Self {
            text: None,
            function_call: None,
            function_response: Some(GeminiFunctionResponse { name, response }),
            thought_signature: None,
            inline_data: None,
            file_data: None,
        }
    }

    /// Attach a thought signature to this part (present on `functionCall`
    /// parts the model signed, and on `functionResponse` parts that replay
    /// them).
    pub fn with_thought_signature(mut self, signature: String) -> Self {
        self.thought_signature = Some(signature);
        self
    }

    /// Create an inline-media (base64) part.
    pub fn inline_data(mime_type: String, data: String) -> Self {
        Self {
            text: None,
            function_call: None,
            function_response: None,
            thought_signature: None,
            inline_data: Some(GeminiInlineData { mime_type, data }),
            file_data: None,
        }
    }

    /// Create a URI-referenced media part.
    pub fn file_data(mime_type: String, file_uri: String) -> Self {
        Self {
            text: None,
            function_call: None,
            function_response: None,
            thought_signature: None,
            inline_data: None,
            file_data: Some(GeminiFileData {
                mime_type,
                file_uri,
            }),
        }
    }
}

/// Inline media bytes (`inlineData`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiInlineData {
    /// MIME type of the media (`mimeType` on the wire).
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// Base64-encoded media bytes.
    pub data: String,
}

/// Media referenced by URI (`fileData`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFileData {
    /// MIME type of the media (`mimeType` on the wire).
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    /// URI of the media (`fileUri` on the wire).
    #[serde(rename = "fileUri")]
    pub file_uri: String,
}

/// Generation parameters for a Gemini request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiGenerationConfig {
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f64>,
    /// Top-K sampling (`topK` on the wire).
    #[serde(rename = "topK", default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Maximum number of tokens in the completion.
    pub max_output_tokens: Option<i32>,
    /// Stop sequences that cause generation to stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Thinking configuration (`thinkingConfig` on the wire).
    #[serde(
        rename = "thinkingConfig",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub thinking_config: Option<GeminiThinkingConfig>,
}

/// Thinking configuration for a Gemini request (`generationConfig.thinkingConfig`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiThinkingConfig {
    /// The thinking budget in tokens (`thinkingBudget`); 0 disables thinking.
    #[serde(rename = "thinkingBudget")]
    pub thinking_budget: i64,
}

/// A tool declaration in the Gemini API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiTool {
    /// Function declarations exposed to the model.
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

/// A single function declaration inside a Gemini tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionDeclaration {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    pub description: Option<String>,
    /// JSON Schema describing the function parameters.
    pub parameters: Option<serde_json::Value>,
}

/// Non-streaming response from the Gemini generate-content endpoint;
/// unknown provider fields are silently ignored rather than failing decode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiResponse {
    /// Candidate completions.
    pub candidates: Vec<GeminiCandidate>,
    /// Token usage metadata.
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsage>,
}

/// A single candidate in a Gemini response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiCandidate {
    /// The generated content; optional because Gemini omits it for candidates
    /// blocked by `SAFETY`/`RECITATION`. Missing `content` yields no text
    /// parts; derive the stop reason from `finish_reason`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<GeminiContent>,
    /// Reason the model stopped generating (e.g. `"STOP"`).
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
}

/// Token usage metadata reported by the Gemini API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiUsage {
    /// Number of tokens in the prompt.
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: i32,
    /// Number of tokens across all candidates.
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: i32,
    /// Total tokens (prompt + candidates).
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: i32,
}

/// A single chunk in a streaming Gemini response; unknown fields are
/// silently ignored rather than causing chunk drops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiStreamChunk {
    /// Candidate completions in this chunk.
    pub candidates: Vec<GeminiCandidate>,
    /// Token usage metadata (usually present on the final chunk).
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_request_round_trip() {
        let req = ResponsesRequest {
            model: "gpt-4o".into(),
            input: vec![ResponsesInput::Message {
                role: "user".into(),
                content: Some(serde_json::json!("hello")),
            }],
            stream: Some(true),
            tools: vec![],
            reasoning: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            extra: serde_json::Map::new(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ResponsesRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-4o");
        assert_eq!(back.input.len(), 1);
    }

    #[test]
    fn responses_input_function_call_items_round_trip() {
        let json = r#"[
            {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},
            {"type":"function_call_output","call_id":"call_1","output":"72F and sunny"}
        ]"#;
        let items: Vec<ResponsesInput> = serde_json::from_str(json).unwrap();
        match &items[0] {
            ResponsesInput::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "get_weather");
                assert!(arguments.contains("SF"));
            }
            other => panic!("expected FunctionCall item, got {other:?}"),
        }
        match &items[1] {
            ResponsesInput::FunctionCallOutput {
                call_id, output, ..
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(output, "72F and sunny");
            }
            other => panic!("expected FunctionCallOutput item, got {other:?}"),
        }
        let back = serde_json::to_string(&items).unwrap();
        assert!(back.contains("\"function_call\""), "{back}");
        assert!(back.contains("\"function_call_output\""), "{back}");
    }

    #[test]
    fn responses_chunk_parses_documented_completed_event() {
        let json = r#"{"type":"response.completed","response":{"id":"resp_1","object":"response","created":1,"model":"gpt-4o","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"input_tokens_details":{"cached_tokens":3}}}}"#;
        let chunk: ResponsesChunk = serde_json::from_str(json).unwrap();
        let response = chunk.response.as_ref().expect("response object");
        let usage = response.usage.as_ref().expect("usage");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(
            usage
                .input_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens),
            Some(3)
        );
    }

    #[test]
    fn responses_chunk_parses_documented_output_item_added() {
        let json = r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#;
        let chunk: ResponsesChunk = serde_json::from_str(json).unwrap();
        let item = chunk.item.as_ref().expect("item");
        assert_eq!(item.r#type, "function_call");
        assert_eq!(item.call_id.as_deref(), Some("call_1"));
        assert_eq!(item.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn gemini_part_parses_camel_case_function_call() {
        let json = r#"{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}"#;
        let part: GeminiPart = serde_json::from_str(json).unwrap();
        assert_eq!(part.function_call.as_ref().unwrap().name, "get_weather");
        let back = serde_json::to_string(&part).unwrap();
        assert!(back.contains("functionCall"), "{back}");
        assert!(!back.contains("function_call"), "{back}");
    }

    #[test]
    fn gemini_request_stream_is_not_serialized() {
        let req = GeminiRequest {
            contents: vec![],
            generation_config: None,
            tools: vec![],
            tool_config: None,
            stream: Some(true),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("stream"), "{json}");
    }

    #[test]
    fn gemini_request_round_trip() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".into(),
                parts: vec![GeminiPart::text("hello".into())],
            }],
            generation_config: Some(GeminiGenerationConfig {
                temperature: Some(0.7),
                top_p: None,
                top_k: None,
                max_output_tokens: Some(1024),
                stop_sequences: None,
                thinking_config: None,
            }),
            tools: vec![],
            tool_config: None,
            stream: Some(true),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: GeminiRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.contents.len(), 1);
        assert!(back.generation_config.is_some());
    }

    #[test]
    fn gemini_function_response_serializes_camel_case() {
        let part = GeminiPart::function_response(
            "get_weather".into(),
            serde_json::json!({"result": "72F"}),
        );
        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("functionResponse"), "{json}");
        let back: GeminiPart = serde_json::from_str(&json).unwrap();
        assert_eq!(back.function_response.as_ref().unwrap().name, "get_weather");
    }

    #[test]
    fn gemini_response_round_trip() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::text("hi there".into())],
                }),
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: Some(GeminiUsage {
                prompt_token_count: 10,
                candidates_token_count: 5,
                total_token_count: 15,
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: GeminiResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.candidates.len(), 1);
        let usage = back.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.total_token_count, 15);
    }

    #[test]
    fn gemini_usage_camel_case_serialization() {
        let usage = GeminiUsage {
            prompt_token_count: 100,
            candidates_token_count: 50,
            total_token_count: 150,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(
            json.contains("promptTokenCount"),
            "expected camelCase: {json}"
        );
        assert!(
            json.contains("candidatesTokenCount"),
            "expected camelCase: {json}"
        );
        assert!(
            json.contains("totalTokenCount"),
            "expected camelCase: {json}"
        );
    }

    #[test]
    fn responses_chunk_round_trip() {
        let chunk = ResponsesChunk {
            r#type: "response.output_text.delta".into(),
            id: Some("resp_123".into()),
            delta: Some("hello".into()),
            output: None,
            usage: None,
            error: None,
            output_index: None,
            item: None,
            response: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: ResponsesChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.delta.as_deref(), Some("hello"));
    }

    #[test]
    fn responses_chunk_output_index_defaulted_when_absent() {
        let json = r#"{"type":"response.created","id":"resp_1"}"#;
        let back: ResponsesChunk = serde_json::from_str(json).unwrap();
        assert!(back.output_index.is_none());
    }

    #[test]
    fn responses_chunk_output_index_parsed_when_present() {
        let json = r#"{"type":"response.output_item.added","output_index":3}"#;
        let back: ResponsesChunk = serde_json::from_str(json).unwrap();
        assert_eq!(back.output_index, Some(3));
    }

    #[test]
    fn gemini_stream_chunk_round_trip() {
        let chunk = GeminiStreamChunk {
            candidates: vec![],
            usage_metadata: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: GeminiStreamChunk = serde_json::from_str(&json).unwrap();
        assert!(back.candidates.is_empty());
    }
}
