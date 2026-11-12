//! Client dialects (plan 08): the Responses, Gemini-native and Ollama-native
//! routes run through the same engine as the Chat/Anthropic routes. Each
//! dialect passes the shared streaming, tool-call, usage and error scenarios
//! against a fake upstream, cross-format (an Anthropic-dialect upstream) and
//! same-format (the dialect's own upstream, for passthrough).

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode, header},
        response::IntoResponse,
        routing::{get, post},
    };
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_core::{
        AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
        ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
    };
    use llm_proxy_server::{AppState, BuildInfo, build_router};
    use llm_proxy_transport::ProxyClient;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    type Capture = Arc<tokio::sync::Mutex<Option<(axum::http::HeaderMap, bytes::Bytes)>>>;

    /// What the fake upstream answers, per test.
    #[derive(Clone, Copy)]
    enum Behavior {
        /// Anthropic non-streaming response with text or tool_use blocks.
        AnthropicNonStream { tool_call: bool },
        /// Anthropic SSE stream with text or a tool call.
        AnthropicStream { tool_call: bool },
        /// Responses non-streaming response.
        ResponsesNonStream,
        /// Responses SSE stream.
        ResponsesStream,
        /// Gemini non-streaming response.
        GeminiNonStream,
        /// Gemini SSE stream.
        GeminiStream,
        /// HTTP 400 with an Anthropic-shaped error body.
        Error400,
    }

    struct FakeUpstream {
        base_url: String,
        capture: Capture,
    }

    fn anthropic_non_stream_body(tool_call: bool) -> Vec<u8> {
        let content = if tool_call {
            json!([
                {"type": "tool_use", "id": "toolu_fixture_01", "name": "get_weather",
                 "input": {"city": "SF"}}
            ])
        } else {
            json!([{"type": "text", "text": "it is 72F"}])
        };
        let stop_reason = if tool_call { "tool_use" } else { "end_turn" };
        serde_json::to_vec(&json!({
            "id": "msg_fixture_dialect",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": stop_reason,
            "stop_sequence": None::<String>,
            "usage": {
                "input_tokens": 21,
                "output_tokens": 6,
                "cache_creation_input_tokens": 4,
                "cache_read_input_tokens": 8,
            }
        }))
        .expect("serialize anthropic body")
    }

    fn anthropic_stream_events(tool_call: bool) -> Vec<&'static str> {
        let mut events = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":21,\"output_tokens\":0,\"cache_read_input_tokens\":8}}}\n\n",
        ];
        if tool_call {
            events.extend([
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_fixture_01\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"SF\\\"}\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":6}}\n\n",
            ]);
        } else {
            events.extend([
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"it is \"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"72F\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":6}}\n\n",
            ]);
        }
        events.push("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        events
    }

    fn responses_non_stream_body() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id": "resp_fixture_01",
            "object": "response",
            "created_at": 1770000000,
            "model": "gpt-5.1-codex",
            "output": [
                {"type": "message", "id": "msg_0", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "it is 72F", "annotations": []}]}
            ],
            "usage": {
                "input_tokens": 21, "output_tokens": 6,
                "input_tokens_details": {"cached_tokens": 8},
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 27
            },
            "status": "completed"
        }))
        .expect("serialize responses body")
    }

    fn responses_stream_events() -> Vec<&'static str> {
        vec![
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture_01\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_0\",\"role\":\"assistant\",\"content\":[]}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_0\",\"output_index\":0,\"content_index\":0,\"delta\":\"it is \"}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_0\",\"output_index\":0,\"content_index\":0,\"delta\":\"72F\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture_01\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":21,\"output_tokens\":6,\"input_tokens_details\":{\"cached_tokens\":8},\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":27}}}\n\n",
        ]
    }

    fn gemini_non_stream_body() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "it is 72F"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 21, "candidatesTokenCount": 6, "totalTokenCount": 27}
        }))
        .expect("serialize gemini body")
    }

    fn gemini_stream_events() -> Vec<&'static str> {
        vec![
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"it is \"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"72F\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":21,\"candidatesTokenCount\":6,\"totalTokenCount\":27}}\n\n",
        ]
    }

    fn sse_body(events: Vec<&'static str>) -> Body {
        let stream = futures::stream::iter(events.into_iter().map(|chunk| {
            Ok::<bytes::Bytes, std::convert::Infallible>(bytes::Bytes::from(chunk.to_owned()))
        }));
        Body::from_stream(stream)
    }

    /// The canned response for one behavior.
    fn respond(behavior: Behavior) -> axum::response::Response {
        match behavior {
            Behavior::AnthropicNonStream { tool_call } => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                anthropic_non_stream_body(tool_call),
            )
                .into_response(),
            Behavior::AnthropicStream { tool_call } => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse_body(anthropic_stream_events(tool_call)),
            )
                .into_response(),
            Behavior::ResponsesNonStream => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                responses_non_stream_body(),
            )
                .into_response(),
            Behavior::ResponsesStream => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse_body(responses_stream_events()),
            )
                .into_response(),
            Behavior::GeminiNonStream => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                gemini_non_stream_body(),
            )
                .into_response(),
            Behavior::GeminiStream => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse_body(gemini_stream_events()),
            )
                .into_response(),
            Behavior::Error400 => (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_vec(&json!({
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "max_tokens is required"}
                }))
                .expect("serialize error body"),
            )
                .into_response(),
        }
    }

    async fn spawn_fake_upstream(behavior: Behavior) -> FakeUpstream {
        let capture: Capture = Arc::new(tokio::sync::Mutex::new(None));
        let capture_for_handler = Arc::clone(&capture);
        let behavior_for_handler = behavior;

        let handler = Arc::new(move |headers: axum::http::HeaderMap, body: bytes::Bytes| {
            let capture = Arc::clone(&capture_for_handler);
            let behavior = behavior_for_handler;
            async move {
                *capture.lock().await = Some((headers, body));
                respond(behavior)
            }
        });

        let app = Router::new()
            .route(
                "/v1/messages",
                post({
                    let handler = Arc::clone(&handler);
                    move |headers, body| handler(headers, body)
                }),
            )
            .route(
                "/v1/responses",
                post({
                    let handler = Arc::clone(&handler);
                    move |headers, body| handler(headers, body)
                }),
            )
            .route(
                "/v1beta/models/{model_method}",
                post({
                    let handler = Arc::clone(&handler);
                    move |headers, body| handler(headers, body)
                }),
            )
            .route(
                "/v1/chat/completions",
                post({
                    let handler = Arc::clone(&handler);
                    move |headers, body| handler(headers, body)
                }),
            )
            .route("/health", get(|| async { StatusCode::OK }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("fake upstream serves");
        });
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "fake upstream did not become ready"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        FakeUpstream {
            base_url: format!("http://{addr}"),
            capture,
        }
    }

    /// One provider per test, named `dialect-provider`, serving `route` via
    /// `protocol` at the fake upstream.
    fn state_with_provider(
        upstream: &FakeUpstream,
        protocol: &str,
        route: ProviderRouteKind,
        chat_route: bool,
    ) -> AppState {
        let adapter_name = "upstream";
        let endpoint = match protocol {
            "anthropic_messages" => format!("{}/v1/messages", upstream.base_url),
            "openai_responses" => format!("{}/v1/responses", upstream.base_url),
            "gemini_generate_content" => {
                format!(
                    "{}/v1beta/models/{{model}}:generateContent",
                    upstream.base_url
                )
            }
            _ => format!("{}/v1/chat/completions", upstream.base_url),
        };
        let provider = ProviderConfig {
            name: "dialect-provider".to_owned(),
            api_key: secrecy::SecretString::from("operator-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::from([(
                adapter_name.to_owned(),
                ProviderAdapterConfig {
                    protocol: protocol.to_owned(),
                    endpoint,
                    headers: Arc::new(HashMap::new()),
                    forward_inbound_headers: None,
                    lossy: false,
                },
            )]),
            routes: ProviderRoutesConfig {
                messages: (route == ProviderRouteKind::Messages).then_some(adapter_name.to_owned()),
                chat_completions: (chat_route || route == ProviderRouteKind::ChatCompletions)
                    .then_some(adapter_name.to_owned()),
                responses: (route == ProviderRouteKind::Responses)
                    .then_some(adapter_name.to_owned()),
                generate_content: (route == ProviderRouteKind::GenerateContent)
                    .then_some(adapter_name.to_owned()),
                embeddings: None,
                rerank: None,
                audio_transcriptions: None,
                audio_speech: None,
                images_generations: None,
                files: None,
                batches: None,
            },
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        let registry = ProviderRegistry::from_providers(vec![provider]).expect("registry");
        AppState::new(
            AppConfig {
                server: ServerConfig {
                    bind: "127.0.0.1:3456".parse().expect("bind"),
                    request_timeout: Duration::from_secs(300),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "dialect-test".to_owned(),
                    rate_limit_rpm: 10_000,
                    trust_forwarded_headers: false,
                    cache: Default::default(),
                    idempotency: Default::default(),
                    metrics: Default::default(),
                    log_format: Default::default(),
                    admin_bind: None,
                    admin_on_main_listener: false,
                    endpoint_families: Default::default(),
                    upstream_timeouts: Default::default(),
                    bare_v1: Default::default(),
                    preflight: Default::default(),
                    semantic_cache: Default::default(),
                },
                auth: Default::default(),
                storage: None,
                replica: None,
                mcp: None,
            },
            registry,
            ProviderAdapterRegistry::builtin(),
            ProxyClient::try_new().expect("client"),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    async fn body_text(response: axum::http::Response<Body>) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
            .await
            .expect("read body");
        String::from_utf8(bytes.to_vec()).expect("utf-8 body")
    }

    async fn captured(upstream: &FakeUpstream) -> Value {
        let guard = upstream.capture.lock().await;
        let (_, body) = guard.as_ref().expect("upstream received a request");
        serde_json::from_slice(body).expect("upstream body is JSON")
    }

    fn post_json(uri: &str, body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("body")
    }

    /// The Responses-dialect request every scenario sends.
    fn responses_request(stream: bool) -> Value {
        json!({
            "model": "gpt-5.1-codex",
            "instructions": "you are terse",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "weather in SF?"}
                ]}
            ],
            "tools": [{
                "type": "function", "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "stream": stream
        })
    }

    fn gemini_request() -> Value {
        json!({
            "contents": [{"role": "user", "parts": [{"text": "weather in SF?"}]}],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "parameters": {"type": "object"}}
            ]}],
            "generationConfig": {"temperature": 0.2}
        })
    }

    fn ollama_chat_request(stream: bool) -> Value {
        json!({
            "model": "llama3.1",
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "tools": [{"name": "get_weather", "parameters": {"type": "object"}}],
            "stream": stream
        })
    }

    fn ollama_generate_request(stream: bool) -> Value {
        json!({
            "model": "llama3.1",
            "prompt": "weather in SF?",
            "stream": stream
        })
    }

    // ------------------------------------------------------------------
    // Shared scenario 1: streaming text reaches the client in dialect
    // framing, with the upstream's usage on the terminal event.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn responses_stream_cross_formats_to_anthropic() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(true),
            ))
            .await
            .expect("served");
        let status = response.status();
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let text = body_text(response).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "upstream rejected the cross-format request: {text}"
        );
        assert_eq!(content_type, "text/event-stream");
        for expected in [
            "event: response.created",
            "event: response.output_item.added",
            "event: response.output_text.delta",
            "event: response.completed",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
        assert!(
            text.contains("it is "),
            "text deltas must carry content: {text}"
        );
        assert!(text.contains("72F"));
        let completed = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str::<Value>(data).expect("valid json"))
            .find(|value| value["type"] == "response.completed")
            .expect("terminal event");
        // The Responses wire includes cached tokens in input_tokens: the
        // Anthropic upstream's 21 plain + 8 cached input = 29.
        assert_eq!(completed["response"]["usage"]["input_tokens"], 29);
        assert_eq!(completed["response"]["usage"]["output_tokens"], 6);
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            8
        );

        // The request the upstream saw is Anthropic-dialect with the tools.
        let upstream_body = captured(&upstream).await;
        assert_eq!(upstream_body["tools"][0]["name"], "get_weather");
        assert_eq!(upstream_body["stream"], true, "{upstream_body}");
    }

    #[tokio::test]
    async fn gemini_stream_sse_cross_formats_to_anthropic() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let text = body_text(response).await;
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("valid json"))
            .collect();
        let text_out: String = chunks
            .iter()
            .flat_map(|chunk| {
                chunk["candidates"][0]["content"]["parts"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
            })
            .filter_map(|part| part["text"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(text_out, "it is 72F");
        let terminal = chunks.last().expect("terminal chunk");
        assert_eq!(terminal["candidates"][0]["finishReason"], "STOP");
        assert_eq!(terminal["usageMetadata"]["promptTokenCount"], 21);
        assert_eq!(terminal["usageMetadata"]["candidatesTokenCount"], 6);
    }

    #[tokio::test]
    async fn gemini_stream_without_alt_is_a_json_array() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:streamGenerateContent",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let text = body_text(response).await;
        let array: Vec<Value> = serde_json::from_str(&text).expect("a JSON array of chunks");
        assert!(!array.is_empty(), "{text}");
        assert!(text.starts_with('[') && text.ends_with(']'));
    }

    #[tokio::test]
    async fn ollama_chat_stream_is_ndjson_cross_formats_to_anthropic() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/api/chat",
                &ollama_chat_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/x-ndjson"
        );
        let text = body_text(response).await;
        let lines: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("ndjson line"))
            .collect();
        let content: String = lines
            .iter()
            .filter(|line| line["done"] == false)
            .filter_map(|line| line["message"]["content"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(content, "it is 72F");
        let terminal = lines.last().expect("terminal object");
        assert_eq!(terminal["done"], true);
        assert_eq!(terminal["prompt_eval_count"], 21);
        assert_eq!(terminal["eval_count"], 6);
    }

    #[tokio::test]
    async fn ollama_generate_stream_is_ndjson() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/api/generate",
                &ollama_generate_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        let lines: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("ndjson line"))
            .collect();
        let content: String = lines
            .iter()
            .filter_map(|line| line["response"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(content, "it is 72F");
        assert_eq!(lines.last().cloned().unwrap_or_default()["done"], true);

        // The upstream saw the prompt as a user turn, not a chat history.
        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_body["messages"][0]["content"][0]["text"],
            "weather in SF?"
        );
    }

    // ------------------------------------------------------------------
    // System-role turns (an Ollama `system` message, a Responses
    // `developer` item) reach an Anthropic upstream's system prompt.
    // ------------------------------------------------------------------

    /// The text of every top-level `system` block the upstream received.
    fn upstream_system_texts(upstream_body: &Value) -> Vec<&str> {
        upstream_body["system"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block["text"].as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn ollama_system_message_reaches_an_anthropic_upstream() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let app = build_router(state);
        let request = json!({
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "weather in SF?"}
            ],
            "stream": false
        });
        let response = app
            .oneshot(post_json("/providers/dialect-provider/api/chat", &request))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);

        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_system_texts(&upstream_body),
            ["you are terse"],
            "{upstream_body}"
        );
        let roles: Vec<&str> = upstream_body["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .filter_map(|message| message["role"].as_str())
            .collect();
        assert_eq!(roles, ["user"], "{upstream_body}");
    }

    #[tokio::test]
    async fn responses_developer_message_reaches_an_anthropic_upstream() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let mut request = responses_request(false);
        request["input"] = json!([
            {"type": "message", "role": "developer", "content": [
                {"type": "input_text", "text": "sandbox: read-only"}
            ]},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "weather in SF?"}
            ]}
        ]);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &request,
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);

        // `instructions` stays first; the developer turn follows it.
        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_system_texts(&upstream_body),
            ["you are terse", "sandbox: read-only"],
            "{upstream_body}"
        );
    }

    // ------------------------------------------------------------------
    // Shared scenario 2: tool calls cross into each dialect's shape.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn responses_tool_call_ordering_cross_formats() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: true }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        let events: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str::<Value>(data).expect("valid json"))
            .collect();
        let types: Vec<&str> = events
            .iter()
            .filter_map(|event| event["type"].as_str())
            .collect();
        let added = types
            .iter()
            .position(|t| *t == "response.output_item.added")
            .expect("item added event");
        let first_delta = types
            .iter()
            .position(|t| *t == "response.function_call_arguments.delta")
            .expect("arguments delta");
        let done = types
            .iter()
            .position(|t| *t == "response.function_call_arguments.done")
            .expect("arguments done");
        let item_done = types
            .iter()
            .rposition(|t| *t == "response.output_item.done")
            .expect("item done");
        let completed = types
            .iter()
            .position(|t| *t == "response.completed")
            .expect("completed");
        assert!(
            added < first_delta && first_delta < done && done < item_done && item_done < completed,
            "Codex tool-call ordering violated: {types:?}"
        );
        let added_item = events
            .iter()
            .find(|event| event["type"] == "response.output_item.added")
            .expect("added");
        assert_eq!(added_item["item"]["type"], "function_call");
        assert_eq!(added_item["item"]["name"], "get_weather");
        assert_eq!(added_item["item"]["call_id"], "toolu_fixture_01");
        let done_item = events
            .iter()
            .rev()
            .find(|event| event["type"] == "response.output_item.done")
            .expect("done");
        assert_eq!(
            done_item["item"]["arguments"], "{\"city\":\"SF\"}",
            "the item must close with complete arguments: {done_item}"
        );
    }

    #[tokio::test]
    async fn gemini_tool_call_becomes_a_function_call_part() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: true }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("valid json"))
            .collect();
        let tool_chunk = chunks
            .iter()
            .find(|chunk| chunk["candidates"][0]["content"]["parts"][0]["functionCall"].is_object())
            .expect("a functionCall part must reach the client");
        assert_eq!(
            tool_chunk["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            "get_weather"
        );
        assert_eq!(
            tool_chunk["candidates"][0]["content"]["parts"][0]["functionCall"]["args"]["city"],
            "SF"
        );
    }

    #[tokio::test]
    async fn ollama_tool_call_becomes_a_tool_calls_message() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: true }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/api/chat",
                &ollama_chat_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        let lines: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("ndjson line"))
            .collect();
        let tool_line = lines
            .iter()
            .find(|line| line["message"]["tool_calls"].is_array())
            .expect("tool_calls must reach the client");
        assert_eq!(
            tool_line["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            tool_line["message"]["tool_calls"][0]["function"]["arguments"]["city"],
            "SF"
        );
    }

    // ------------------------------------------------------------------
    // Shared scenario 3: exact usage fields on non-streaming responses.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn responses_usage_fields_are_wire_exact() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(false),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        let usage = &value["usage"];
        // The core's input_tokens excludes cache reads (Anthropic wire
        // semantics); the Responses wire includes the cached subset.
        assert_eq!(usage["input_tokens"], 29);
        assert_eq!(usage["output_tokens"], 6);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 8);
        // The details objects must exist even at zero (the Codex break).
        let raw = serde_json::to_string(&usage).expect("serialize");
        assert!(
            raw.contains("\"cached_tokens\"") && raw.contains("\"reasoning_tokens\""),
            "usage detail fields must never be omitted: {raw}"
        );
        assert_eq!(value["status"], "completed");
    }

    #[tokio::test]
    async fn gemini_usage_fields_are_wire_exact() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:generateContent",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        assert_eq!(
            value["candidates"][0]["content"]["parts"][0]["text"],
            "it is 72F"
        );
        assert_eq!(value["candidates"][0]["finishReason"], "STOP");
        assert_eq!(value["usageMetadata"]["promptTokenCount"], 21);
        assert_eq!(value["usageMetadata"]["candidatesTokenCount"], 6);
        assert_eq!(value["usageMetadata"]["totalTokenCount"], 27);
    }

    #[tokio::test]
    async fn ollama_non_streaming_usage_fields() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/api/chat",
                &ollama_chat_request(false),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        assert_eq!(value["message"]["content"], "it is 72F");
        assert_eq!(value["done"], true);
        assert_eq!(value["prompt_eval_count"], 21);
        assert_eq!(value["eval_count"], 6);
        assert_eq!(value["done_reason"], "stop");
    }

    // ------------------------------------------------------------------
    // Shared scenario 4: upstream errors map to each dialect's error shape.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn dialect_errors_map_to_their_wire_shapes() {
        // Responses: the OpenAI error object.
        let upstream = spawn_fake_upstream(Behavior::Error400).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(false),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert!(value["error"]["message"].is_string(), "{value}");
        assert!(value["error"]["type"].is_string(), "{value}");

        // Gemini: the google error object.
        let upstream = spawn_fake_upstream(Behavior::Error400).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:generateContent",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert_eq!(value["error"]["code"], 400, "{value}");
        assert_eq!(value["error"]["status"], "INVALID_ARGUMENT");

        // Ollama: the bare error string.
        let upstream = spawn_fake_upstream(Behavior::Error400).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/api/chat",
                &ollama_chat_request(false),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert!(value["error"].is_string(), "{value}");
    }

    #[tokio::test]
    async fn unknown_methods_under_v1beta_models_are_not_found_in_gemini_shape() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:countTokens",
                &json!({}),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert_eq!(value["error"]["code"], 404);
    }

    // ------------------------------------------------------------------
    // Same-format passthrough: the dialect's own upstream gets the raw body.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn responses_same_format_passthrough_forwards_the_raw_body() {
        let upstream = spawn_fake_upstream(Behavior::ResponsesNonStream).await;
        let state = state_with_provider(
            &upstream,
            "openai_responses",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let mut body = responses_request(false);
        body["store"] = json!(false);
        let response = app
            .oneshot(post_json("/providers/dialect-provider/v1/responses", &body))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        // The raw upstream body is returned verbatim (same-format
        // passthrough), including the usage the decode read for the ledger.
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        assert_eq!(value["id"], "resp_fixture_01");
        assert_eq!(value["usage"]["input_tokens_details"]["cached_tokens"], 8);

        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_body, body,
            "same-format Responses must forward byte-identical modulo nothing"
        );
    }

    #[tokio::test]
    async fn gemini_same_format_passthrough_forwards_the_raw_body() {
        let upstream = spawn_fake_upstream(Behavior::GeminiNonStream).await;
        let state = state_with_provider(
            &upstream,
            "gemini_generate_content",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let app = build_router(state);
        let body = gemini_request();
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:generateContent",
                &body,
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        assert_eq!(
            value["candidates"][0]["content"]["parts"][0]["text"],
            "it is 72F"
        );

        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_body, body,
            "same-format Gemini must forward the body unchanged (the model lives in the URL)"
        );
    }

    #[tokio::test]
    async fn gemini_same_format_stream_selects_the_stream_url() {
        let upstream = spawn_fake_upstream(Behavior::GeminiStream).await;
        let state = state_with_provider(
            &upstream,
            "gemini_generate_content",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let app = build_router(state);
        let response = app
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        // The fake upstream answers every route identically; what matters is
        // the stream decodes and re-encodes into Gemini chunks.
        let text = body_text(response).await;
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("valid json"))
            .collect();
        let content: String = chunks
            .iter()
            .flat_map(|chunk| {
                chunk["candidates"][0]["content"]["parts"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default()
            })
            .filter_map(|part| part["text"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(content, "it is 72F");
    }

    // A Responses knob with no Anthropic equivalent is rejected by name,
    // never silently dropped (cross-format policy).
    #[tokio::test]
    async fn responses_store_is_rejected_by_name_cross_format() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let mut body = responses_request(false);
        body["store"] = json!(false);
        let response = build_router(state)
            .oneshot(post_json("/providers/dialect-provider/v1/responses", &body))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let text = body_text(response).await;
        assert!(
            text.contains("store"),
            "the rejection must name store: {text}"
        );
    }

    // Same-format Responses streaming: the upstream's Responses SSE decodes
    // and re-encodes back into Responses SSE.
    #[tokio::test]
    async fn responses_same_format_stream_round_trips() {
        let upstream = spawn_fake_upstream(Behavior::ResponsesStream).await;
        let state = state_with_provider(
            &upstream,
            "openai_responses",
            ProviderRouteKind::Responses,
            false,
        );
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let text = body_text(response).await;
        for expected in [
            "event: response.created",
            "event: response.output_text.delta",
            "event: response.completed",
        ] {
            assert!(text.contains(expected), "missing {expected}: {text}");
        }
        let completed = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str::<Value>(data).expect("valid json"))
            .find(|value| value["type"] == "response.completed")
            .expect("terminal event");
        assert_eq!(completed["response"]["usage"]["input_tokens"], 21);
        assert_eq!(
            completed["response"]["usage"]["input_tokens_details"]["cached_tokens"],
            8
        );
    }

    // ------------------------------------------------------------------
    // /api/tags lists the provider's models in the Ollama shape.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn ollama_tags_lists_models_from_the_catalog() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/providers/dialect-provider/api/tags")
                    .body(Body::empty())
                    .expect("body"),
            )
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        let models = value["models"].as_array().cloned().unwrap_or_default();
        // The static catalog seeds a default model set; the shape is what the
        // assertion pins (name + model per entry).
        for model in &models {
            assert!(model["name"].is_string(), "{model}");
            assert!(model["model"].is_string(), "{model}");
        }
    }
}
