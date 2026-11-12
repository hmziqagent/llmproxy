//! Client dialects (plan 08): the Responses, Gemini-native and Ollama-native
//! routes run through the same engine as the Chat/Anthropic routes. Each
//! dialect passes the shared streaming, tool-call, usage and error scenarios
//! against a fake upstream, cross-format (an Anthropic-dialect upstream) and
//! same-format (the dialect's own upstream, for passthrough).

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
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
        AppConfig, AuthStyle, PoolConfig, PoolFile, ProviderAdapterConfig, ProviderCatalogConfig,
        ProviderCatalogMode, ProviderConfig, ProviderRegistry, ProviderRouteKind,
        ProviderRoutesConfig, ServerConfig, StaticModelCatalogEntry,
    };
    use llm_proxy_server::{AppState, BuildInfo, build_router};
    use llm_proxy_storage::{
        EventBus, RequestFilter, RequestRow, RetentionPolicy, SqliteEventBus, UsageReader,
    };
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
        /// Anthropic SSE stream whose `event: error` arrives only after the
        /// committed prefix, so the proxy has already answered 200 when it
        /// decodes: the error must surface in-band, not as an HTTP status.
        AnthropicStreamThenError,
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

    /// Like [`sse_body`], but the last event is withheld for `gap`: the
    /// earlier events are decoded (and the proxy commits its first bytes)
    /// before the trailing event is even written, so a trailing error cannot
    /// fold into the same transport read as the committed prefix and read as
    /// a leading error.
    fn sse_body_with_trailing_gap(events: Vec<&'static str>, gap: Duration) -> Body {
        use futures::StreamExt as _;
        let mut events = events;
        let last = events.pop().expect("at least one event").to_owned();
        let prefix = futures::stream::iter(events.into_iter().map(|chunk| {
            Ok::<bytes::Bytes, std::convert::Infallible>(bytes::Bytes::from(chunk.to_owned()))
        }));
        let tail = futures::stream::once(async move {
            tokio::time::sleep(gap).await;
            Ok::<bytes::Bytes, std::convert::Infallible>(bytes::Bytes::from(last))
        });
        Body::from_stream(prefix.chain(tail))
    }

    /// The committed prefix of the text stream followed by a mid-stream
    /// `event: error`: `message_start`, a text `content_block_start`, one
    /// `text_delta`, then the overloaded error.
    fn anthropic_stream_then_error_events() -> Vec<&'static str> {
        let mut events: Vec<&'static str> =
            anthropic_stream_events(false).into_iter().take(3).collect();
        events.push(
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
        );
        events
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
            Behavior::AnthropicStreamThenError => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse_body_with_trailing_gap(
                    anthropic_stream_then_error_events(),
                    Duration::from_millis(50),
                ),
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
        let provider = provider_config(
            "dialect-provider",
            upstream,
            protocol,
            route,
            chat_route,
            None,
        );
        state_from(vec![provider], Vec::new())
    }

    /// A `ProviderConfig` named `name`, serving `route` (and optionally the
    /// chat-completions route) via `protocol` at the fake upstream, with an
    /// optional model catalog.
    fn provider_config(
        name: &str,
        upstream: &FakeUpstream,
        protocol: &str,
        route: ProviderRouteKind,
        chat_route: bool,
        catalog: Option<ProviderCatalogConfig>,
    ) -> ProviderConfig {
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
        ProviderConfig {
            name: name.to_owned(),
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
            catalog,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        }
    }

    /// App state over `providers` and `pools`, against the shared server
    /// config every scenario in this file uses.
    fn state_from(providers: Vec<ProviderConfig>, pools: Vec<PoolConfig>) -> AppState {
        let registry = ProviderRegistry::from_parts(providers, pools).expect("registry");
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

    /// A static catalog entry with only its id set.
    fn catalog_entry(id: &str) -> StaticModelCatalogEntry {
        StaticModelCatalogEntry {
            id: id.to_owned(),
            display_name: None,
            supports: Vec::new(),
            context_length: None,
        }
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
        // `promptTokenCount` includes the 8 cache-read tokens from the
        // stream fixture's `message_start` (21 + 8).
        assert_eq!(terminal["usageMetadata"]["promptTokenCount"], 29);
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
        // Google documents `promptTokenCount` as the total prompt size
        // including cached content, with `cachedContentTokenCount` as the
        // cached subset; the fake upstream reports 21 evaluated + 8 cache
        // read, so the prompt total is 29.
        assert_eq!(value["usageMetadata"]["promptTokenCount"], 29);
        assert_eq!(value["usageMetadata"]["cachedContentTokenCount"], 8);
        assert_eq!(value["usageMetadata"]["candidatesTokenCount"], 6);
        assert_eq!(value["usageMetadata"]["totalTokenCount"], 35);
    }

    /// The streamed terminal chunk carries the same cache-inclusive prompt
    /// total: the Anthropic stream fixture's `message_start` reports
    /// `cache_read_input_tokens: 8`.
    #[tokio::test]
    async fn gemini_stream_usage_includes_cache_reads() {
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
        let text = body_text(response).await;
        let chunks: Vec<Value> = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("valid json"))
            .collect();
        let terminal = chunks.last().expect("terminal chunk");
        assert_eq!(terminal["usageMetadata"]["promptTokenCount"], 29);
        assert_eq!(terminal["usageMetadata"]["cachedContentTokenCount"], 8);
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

    // ------------------------------------------------------------------
    // Streaming x error: a request that asks for a stream but fails before
    // any stream bytes are committed keeps the dialect's HTTP error shape
    // (never a half-open stream), and an error after bytes were committed
    // surfaces in-band, in the dialect's own stream framing, with no
    // success terminal.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn streaming_http_errors_use_the_dialect_error_shape() {
        // Responses: the OpenAI error object. The failure happens before the
        // stream is committed, so the answer is JSON, never
        // text/event-stream.
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
                &responses_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        assert_eq!(content_type, "application/json");
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
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert_eq!(value["error"]["code"], 400, "{value}");
        assert_eq!(value["error"]["status"], "INVALID_ARGUMENT");

        // Ollama: the bare error string, and not an NDJSON stream of done
        // objects.
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
                &ollama_chat_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        assert_ne!(
            content_type, "application/x-ndjson",
            "an uncommitted failure is an HTTP error, not a stream"
        );
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert!(value["error"].is_string(), "{value}");
    }

    // The upstream errors after the proxy committed its first bytes: the
    // client already holds a 200 and an open stream, so the error surfaces
    // in-band as the dialect's failure terminal and no success terminal is
    // ever emitted.
    #[tokio::test]
    async fn responses_stream_error_renders_response_failed() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStreamThenError).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
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
        assert!(
            text.contains("event: response.failed"),
            "the failure terminal is missing: {text}"
        );
        assert!(
            !text.contains("response.completed"),
            "a failed stream must not carry the success terminal: {text}"
        );
        let failed = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str::<Value>(data).expect("valid json"))
            .find(|value| value["type"] == "response.failed")
            .expect("response.failed event");
        let message = failed["response"]["error"]["message"]
            .as_str()
            .expect("error message string");
        assert!(
            message.contains("Overloaded"),
            "the upstream message must reach the client: {message}"
        );
    }

    #[tokio::test]
    async fn gemini_stream_error_renders_an_error_chunk() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStreamThenError).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let response = build_router(state)
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
        let errored = chunks
            .iter()
            .filter_map(|chunk| chunk["error"]["message"].as_str())
            .any(|message| message.contains("Overloaded"));
        assert!(errored, "no chunk carries the error message: {text}");
        assert!(
            chunks.iter().all(|chunk| chunk
                .get("candidates")
                .and_then(|candidates| candidates.get(0))
                .and_then(|candidate| candidate.get("finishReason"))
                .is_none()),
            "a failed stream must not carry a finishReason: {text}"
        );
    }

    #[tokio::test]
    async fn ollama_stream_error_renders_an_error_line() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStreamThenError).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        );
        let response = build_router(state)
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
        assert!(
            lines
                .iter()
                .all(|line| line["done"].as_bool() != Some(true)),
            "a failed stream must not carry a done object: {text}"
        );
        let last = lines.last().expect("terminal line");
        let message = last["error"]
            .as_str()
            .expect("terminal line carries the error string");
        assert!(
            message.contains("Overloaded"),
            "the upstream message must reach the client: {message}"
        );
    }

    // Without `alt=sse` the stream is a JSON array, and the array framing
    // must still close as valid JSON after an in-band error: its last
    // element carries the error.
    #[tokio::test]
    async fn gemini_stream_without_alt_error_closes_the_json_array() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicStreamThenError).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:streamGenerateContent",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let array: Vec<Value> =
            serde_json::from_str(&body_text(response).await).expect("a JSON array of chunks");
        let last = array.last().expect("terminal element");
        let message = last["error"]["message"]
            .as_str()
            .expect("error message string");
        assert!(
            message.contains("Overloaded"),
            "the upstream message must reach the client: {message}"
        );
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

    // A tool-level `strict: true` on a Responses function tool carries the
    // client's structured-output guarantee; crossing to a protocol with no
    // equivalent must reject it by name, never silently drop it.
    #[tokio::test]
    async fn responses_strict_tool_is_rejected_by_name_cross_format() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let mut body = responses_request(false);
        body["tools"][0]["strict"] = json!(true);
        let response = build_router(state)
            .oneshot(post_json("/providers/dialect-provider/v1/responses", &body))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let text = body_text(response).await;
        assert!(
            text.contains("strict") && text.contains("get_weather"),
            "the rejection must name strict and the tool: {text}"
        );
    }

    // Same-format Responses passthrough forwards the raw body, so the
    // tool-level `strict` flag survives untouched.
    #[tokio::test]
    async fn responses_strict_tool_survives_same_format() {
        let upstream = spawn_fake_upstream(Behavior::ResponsesNonStream).await;
        let state = state_with_provider(
            &upstream,
            "openai_responses",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let mut body = responses_request(false);
        body["tools"][0]["strict"] = json!(true);
        let response = app
            .oneshot(post_json("/providers/dialect-provider/v1/responses", &body))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_body["tools"][0]["strict"],
            json!(true),
            "same-format passthrough must keep tools[0].strict"
        );
    }

    // A Responses server tool (`{"type": "web_search"}`) has no Anthropic
    // equivalent; crossing formats must reject it by name, never silently
    // change what the model can do.
    #[tokio::test]
    async fn responses_web_search_is_rejected_by_name_cross_format() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        );
        let mut body = responses_request(false);
        let function_tool = body["tools"][0].clone();
        body["tools"] = json!([function_tool, {"type": "web_search"}]);
        let response = build_router(state)
            .oneshot(post_json("/providers/dialect-provider/v1/responses", &body))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let text = body_text(response).await;
        assert!(
            text.contains("web_search"),
            "the rejection must name web_search: {text}"
        );
    }

    // The Gemini server tool (`{"googleSearch": {}}`) is rejected the same
    // way, and the Gemini-dialect client gets the google error shape.
    #[tokio::test]
    async fn gemini_google_search_is_rejected_by_name_cross_format() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        );
        let mut body = gemini_request();
        body["tools"] = json!([{"googleSearch": {}}]);
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:generateContent",
                &body,
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json error");
        assert_eq!(value["error"]["code"], 400, "{value}");
        let message = value["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("googleSearch"),
            "the rejection must name googleSearch: {message}"
        );
    }

    // Same-format Responses passthrough forwards the raw body, so a server
    // tool the upstream itself supports survives untouched.
    #[tokio::test]
    async fn responses_web_search_survives_same_format() {
        let upstream = spawn_fake_upstream(Behavior::ResponsesNonStream).await;
        let state = state_with_provider(
            &upstream,
            "openai_responses",
            ProviderRouteKind::Responses,
            false,
        );
        let app = build_router(state);
        let mut body = responses_request(false);
        let function_tool = body["tools"][0].clone();
        body["tools"] = json!([function_tool, {"type": "web_search"}]);
        let response = app
            .oneshot(post_json("/providers/dialect-provider/v1/responses", &body))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let upstream_body = captured(&upstream).await;
        assert_eq!(
            upstream_body["tools"][1],
            json!({"type": "web_search"}),
            "same-format passthrough must keep the server tool verbatim"
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
    // /api/tags lists the provider's static catalog in the Ollama shape,
    // the deduplicated union of a pool's member catalogs, and a 404 in
    // Ollama error shape for an unknown provider.
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn ollama_tags_lists_models_from_the_catalog() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let provider = provider_config(
            "dialect-provider",
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
            Some(ProviderCatalogConfig {
                mode: ProviderCatalogMode::Static,
                models: vec![catalog_entry("alpha-model"), catalog_entry("beta-model")],
                ..Default::default()
            }),
        );
        let state = state_from(vec![provider], Vec::new());
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
        assert_eq!(
            value["models"],
            json!([
                {"name": "alpha-model", "model": "alpha-model"},
                {"name": "beta-model", "model": "beta-model"},
            ]),
            "the catalog must list exactly its two static entries, name == model"
        );
    }

    #[tokio::test]
    async fn ollama_tags_for_a_pool_lists_the_deduplicated_union() {
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let dialect_a = provider_config(
            "dialect-a",
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
            Some(ProviderCatalogConfig {
                mode: ProviderCatalogMode::Static,
                models: vec![catalog_entry("alpha"), catalog_entry("shared")],
                ..Default::default()
            }),
        );
        let dialect_b = provider_config(
            "dialect-b",
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
            Some(ProviderCatalogConfig {
                mode: ProviderCatalogMode::Static,
                models: vec![catalog_entry("shared"), catalog_entry("beta")],
                ..Default::default()
            }),
        );
        let pool = toml::from_str::<PoolFile>(
            "[pool]\nname=\"duo\"\nmembers=[\"dialect-a\",\"dialect-b\"]\n",
        )
        .expect("pool file parses")
        .pool;
        let state = state_from(vec![dialect_a, dialect_b], vec![pool]);
        let response = build_router(state)
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/providers/duo/api/tags")
                    .body(Body::empty())
                    .expect("body"),
            )
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        let names: Vec<&str> = value["models"]
            .as_array()
            .expect("models is an array")
            .iter()
            .map(|model| model["name"].as_str().expect("name is a string"))
            .collect();
        assert_eq!(
            names,
            ["alpha", "shared", "beta"],
            "union of both members' catalogs, shared listed once"
        );
    }

    #[tokio::test]
    async fn ollama_tags_for_an_unknown_provider_is_a_not_found_error_in_ollama_shape() {
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
                    .uri("/providers/nope/api/tags")
                    .body(Body::empty())
                    .expect("body"),
            )
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let value: Value = serde_json::from_str(&body_text(response).await).expect("json response");
        let error = value["error"]
            .as_str()
            .expect("ollama error body: error is a string");
        assert!(
            error.contains("nope"),
            "the message names the provider: {error}"
        );
    }

    // ------------------------------------------------------------------
    // Usage ledger (08-2): each new dialect's request lands exactly one
    // completed row carrying its dialect, route kind, provider, streaming
    // flag and the upstream's usage numbers. The write path was already
    // real when these were written (lifecycle.rs emits RequestReceived
    // with client_dialect_str / route_kind_str and ResponseCompleted with
    // usage), so they pass on current code; they would fail if
    // `client_dialect_str` (crates/llm-proxy-engine/src/lifecycle.rs)
    // returned "openai_chat" for the OpenAiResponses/Gemini/Ollama arms —
    // every `client_protocol` assertion below pins one distinct string.
    // ------------------------------------------------------------------

    fn open_ledger(path: &Path) -> Arc<SqliteEventBus> {
        Arc::new(
            SqliteEventBus::open(
                path,
                RetentionPolicy {
                    max_age: None,
                    max_rows: None,
                    prune_every: Duration::from_secs(3600),
                },
            )
            .expect("ledger opens"),
        )
    }

    async fn close_ledger(bus: Arc<SqliteEventBus>) {
        tokio::task::spawn_blocking(move || bus.close())
            .await
            .expect("close joins");
    }

    async fn ledger_rows(db: &Path) -> Vec<RequestRow> {
        UsageReader::new(db)
            .requests(RequestFilter {
                limit: 50,
                ..Default::default()
            })
            .await
            .expect("rows")
            .rows
    }

    /// The completed row every dialect request must produce, beyond its
    /// dialect-specific strings.
    fn assert_completed_row(row: &RequestRow, dialect: &str, route_kind: &str, streaming: bool) {
        assert_eq!(row.client_protocol.as_deref(), Some(dialect), "{row:?}");
        assert_eq!(row.route_kind.as_deref(), Some(route_kind), "{row:?}");
        assert_eq!(row.provider.as_deref(), Some("dialect-provider"), "{row:?}");
        assert_eq!(row.streaming, Some(streaming), "{row:?}");
        assert_eq!(row.state, "completed", "{row:?}");
    }

    /// The Anthropic fake's usage: input 21 (cache reads excluded, core
    /// semantics), output 6, cache_read 8, cache_creation 4, no reasoning.
    fn assert_row_tokens(
        row: &RequestRow,
        input: i64,
        output: i64,
        cache_read: i64,
        cache_creation: i64,
    ) {
        let tokens = row.tokens.as_ref().expect("usage is on the row");
        assert_eq!(tokens.input, input, "{row:?}");
        assert_eq!(tokens.output, output, "{row:?}");
        assert_eq!(tokens.cache_read, cache_read, "{row:?}");
        assert_eq!(tokens.cache_creation, cache_creation, "{row:?}");
        assert_eq!(tokens.reasoning, 0, "{row:?}");
    }

    #[tokio::test]
    async fn responses_request_is_recorded_with_its_dialect() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("usage.db");
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let bus = open_ledger(&db);
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        )
        .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(false),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        close_ledger(bus).await;
        let rows = ledger_rows(&db).await;
        assert_eq!(rows.len(), 1, "exactly one ledger row: {rows:?}");
        let row = rows.first().expect("the one row");
        assert_completed_row(row, "openai_responses", "responses", false);
        assert_row_tokens(row, 21, 6, 8, 4);
    }

    #[tokio::test]
    async fn gemini_request_is_recorded_with_its_dialect() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("usage.db");
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let bus = open_ledger(&db);
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::GenerateContent,
            false,
        )
        .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1beta/models/gemini-2.5-flash:generateContent",
                &gemini_request(),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        close_ledger(bus).await;
        let rows = ledger_rows(&db).await;
        assert_eq!(rows.len(), 1, "exactly one ledger row: {rows:?}");
        let row = rows.first().expect("the one row");
        assert_completed_row(row, "gemini", "generate_content", false);
        assert_row_tokens(row, 21, 6, 8, 4);
    }

    #[tokio::test]
    async fn ollama_request_is_recorded_with_its_dialect() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("usage.db");
        let upstream = spawn_fake_upstream(Behavior::AnthropicNonStream { tool_call: false }).await;
        let bus = open_ledger(&db);
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::ChatCompletions,
            true,
        )
        .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/api/chat",
                &ollama_chat_request(false),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        close_ledger(bus).await;
        let rows = ledger_rows(&db).await;
        assert_eq!(rows.len(), 1, "exactly one ledger row: {rows:?}");
        let row = rows.first().expect("the one row");
        assert_completed_row(row, "ollama", "chat_completions", false);
        assert_row_tokens(row, 21, 6, 8, 4);
    }

    #[tokio::test]
    async fn streamed_dialect_request_is_recorded_after_the_stream_ends() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("usage.db");
        let upstream = spawn_fake_upstream(Behavior::AnthropicStream { tool_call: false }).await;
        let bus = open_ledger(&db);
        let state = state_with_provider(
            &upstream,
            "anthropic_messages",
            ProviderRouteKind::Responses,
            false,
        )
        .with_event_bus(Arc::clone(&bus) as Arc<dyn EventBus>);
        let response = build_router(state)
            .oneshot(post_json(
                "/providers/dialect-provider/v1/responses",
                &responses_request(true),
            ))
            .await
            .expect("served");
        assert_eq!(response.status(), StatusCode::OK);
        // The completion row is written when the stream ends, so the SSE
        // body must be drained before the ledger closes.
        let text = body_text(response).await;
        assert!(text.contains("event: response.completed"), "{text}");
        close_ledger(bus).await;
        let rows = ledger_rows(&db).await;
        assert_eq!(rows.len(), 1, "exactly one ledger row: {rows:?}");
        let row = rows.first().expect("the one row");
        assert_completed_row(row, "openai_responses", "responses", true);
        assert_eq!(row.tokens.as_ref().map(|tokens| tokens.input), Some(21));
    }
}
