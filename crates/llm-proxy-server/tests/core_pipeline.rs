//! Integration tests for the core pipeline: the full request path through
//! the axum router against a local mock upstream, with canned provider
//! responses translated into Anthropic-shaped responses.

#[cfg(test)]
mod tests {

    use std::collections::HashMap;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::time::Duration;

    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode, header},
        response::IntoResponse,
        response::sse::{Event, KeepAlive, Sse},
        routing::post,
    };
    use futures::FutureExt;
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_core::{
        AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry, ServerConfig,
    };
    use llm_proxy_server::{AppState, BuildInfo, build_router};
    use llm_proxy_transport::ProxyClient;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    async fn wait_for_ready(addr: std::net::SocketAddr) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return;
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("mock server at {addr} did not become ready within 5 s");
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn spawn_mock_serve(listener: tokio::net::TcpListener, app: Router) {
        tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(axum::serve(listener, app).into_future())
                .catch_unwind()
                .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => eprintln!("mock axum::serve error: {e}"),
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(|s| s.as_str())
                        .or_else(|| panic.downcast_ref::<&'static str>().copied())
                        .unwrap_or("<non-string panic>");
                    eprintln!("mock server task panicked: {msg}");
                    std::process::abort();
                }
            }
        });
    }

    fn anthropic_success_response() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id": "msg_mock123",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": "Hello from mock!" }],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        }))
        .unwrap()
    }

    fn openai_chat_success_response() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 12345,
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from OpenAI mock!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .unwrap()
    }

    fn openai_responses_success_response() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id": "resp_test123",
            "object": "response",
            "created": 12345,
            "model": "gpt-4o-2024-08-06",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Hello from Responses mock!"
                }]
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        }))
        .unwrap()
    }

    fn gemini_success_response() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{ "text": "Hello from Gemini mock!" }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        }))
        .unwrap()
    }

    fn state_with_provider(
        mock_endpoint: &str,
        protocol: &str,
        adapter_name: &str,
        model_name: &str,
        pricing: HashMap<llm_proxy_core::ModelId, llm_proxy_core::ModelPricing>,
    ) -> AppState {
        state_with_provider_edited(
            mock_endpoint,
            protocol,
            adapter_name,
            model_name,
            pricing,
            |_| {},
        )
    }

    /// [`state_with_provider`] with an edit applied to the app config before
    /// the state is built (the plan 06 opt-ins, per test).
    fn state_with_provider_edited(
        mock_endpoint: &str,
        protocol: &str,
        adapter_name: &str,
        _model_name: &str,
        pricing: HashMap<llm_proxy_core::ModelId, llm_proxy_core::ModelPricing>,
        edit: impl FnOnce(&mut AppConfig),
    ) -> AppState {
        let routes = match protocol {
            "anthropic_messages" => llm_proxy_core::ProviderRoutesConfig {
                messages: Some(adapter_name.to_owned()),
                chat_completions: None,
                ..Default::default()
            },
            "openai_chat_completions" => llm_proxy_core::ProviderRoutesConfig {
                chat_completions: Some(adapter_name.to_owned()),
                messages: Some(adapter_name.to_owned()),
                ..Default::default()
            },
            _ => llm_proxy_core::ProviderRoutesConfig {
                messages: Some(adapter_name.to_owned()),
                chat_completions: None,
                ..Default::default()
            },
        };

        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    adapter_name.to_owned(),
                    ProviderAdapterConfig {
                        protocol: protocol.to_owned(),
                        endpoint: mock_endpoint.to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes,
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing,
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };

        let registry = ProviderRegistry::from_providers(vec![provider]).expect("registry");

        let mut app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: Duration::from_secs(300),
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "test-proxy".to_owned(),
                rate_limit_rpm: 100,
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
        };
        edit(&mut app_config);

        AppState::new(
            app_config,
            registry,
            ProviderAdapterRegistry::builtin(),
            ProxyClient::new(),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    fn state_with_anthropic_provider(mock_endpoint: &str) -> AppState {
        state_with_provider(
            mock_endpoint,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
            HashMap::new(),
        )
    }

    fn state_with_anthropic_provider_priced(
        mock_endpoint: &str,
        pricing: HashMap<llm_proxy_core::ModelId, llm_proxy_core::ModelPricing>,
    ) -> AppState {
        state_with_provider(
            mock_endpoint,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
            pricing,
        )
    }

    fn state_with_openai_chat_provider(mock_endpoint: &str) -> AppState {
        state_with_provider(
            mock_endpoint,
            "openai_chat_completions",
            "chat",
            "gpt-4o",
            HashMap::new(),
        )
    }

    fn state_with_openai_responses_provider(mock_endpoint: &str) -> AppState {
        state_with_provider(
            mock_endpoint,
            "openai_responses",
            "responses",
            "gpt-4o-responses",
            HashMap::new(),
        )
    }

    fn state_with_gemini_provider(mock_endpoint: &str) -> AppState {
        state_with_provider(
            mock_endpoint,
            "gemini_generate_content",
            "gemini",
            "gemini-2.5-pro",
            HashMap::new(),
        )
    }

    async fn spawn_mock_server(response_body: Vec<u8>, content_type: &str) -> String {
        let ct = content_type.to_owned();
        let app = Router::new().route(
            "/{*path}",
            post(move || async move {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, ct.clone())],
                    response_body.clone(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_anthropic_non_stream() -> String {
        spawn_mock_server(anthropic_success_response(), "application/json").await
    }

    async fn spawn_mock_openai_chat_non_stream() -> String {
        spawn_mock_server(openai_chat_success_response(), "application/json").await
    }

    async fn spawn_mock_openai_responses_non_stream() -> String {
        spawn_mock_server(openai_responses_success_response(), "application/json").await
    }

    async fn spawn_mock_gemini_non_stream() -> String {
        spawn_mock_server(gemini_success_response(), "application/json").await
    }

    async fn spawn_mock_anthropic_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_mock_stream","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_start")
                    .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi!"}}"#),
                Event::default()
                    .event("content_block_stop")
                    .data(r#"{"type":"content_block_stop","index":0}"#),
                Event::default()
                    .event("message_delta")
                    .data(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}"#),
                Event::default()
                    .event("message_stop")
                    .data(r#"{"type":"message_stop"}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_openai_chat_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .data(r#"{"id":"chatcmpl-stream","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#),
                Event::default()
                    .data(r#"{"id":"chatcmpl-stream","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi from OpenAI!"},"finish_reason":null}]}"#),
                Event::default()
                    .data(r#"{"id":"chatcmpl-stream","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
                Event::default()
                    .data("[DONE]"),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_openai_responses_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .data(r#"{"type":"response.created","response":{"id":"resp_stream","object":"response","created":12345,"model":"gpt-4o","status":"in_progress","output":[]}}"#),
                Event::default()
                    .data(r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}"#),
                Event::default()
                    .data(r#"{"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}"#),
                Event::default()
                    .data(r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi from Responses!"}"#),
                Event::default()
                    .data(r#"{"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hi from Responses!"}"#),
                Event::default()
                    .data(r#"{"type":"response.completed","response":{"id":"resp_stream","object":"response","created":12345,"model":"gpt-4o","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hi from Responses!"}]}],"usage":{"input_tokens":10,"output_tokens":5}}}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_gemini_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .data(r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi from Gemini!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_500() -> String {
        let app = Router::new().route(
            "/{*path}",
            post(|| async {
                (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"type":"error","error":{"type":"internal_error","message":"upstream crash"}}"#
                    .as_bytes()
                    .to_vec(),
            )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_malformed_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_mock","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_start")
                    .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"good-delta"}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data("this is not valid json {{{"),
                Event::default()
                    .event("message_stop")
                    .data(r#"{"type":"message_stop"}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_in_band_error_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_err","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("error")
                    .data(r#"{"type":"error","error":{"type":"overloaded_error","message":"Too many requests"}}"#),
                Event::default()
                    .event("message_stop")
                    .data(r#"{"type":"message_stop"}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_disconnect_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_mock","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_start")
                    .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
            ];
            let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_anthropic_terminal_then_reset_stream() -> String {
        use futures::StreamExt;

        // A complete upstream message whose terminal frame is followed by a
        // transport error: the body stream yields Err after message_stop, as
        // a reset or timed-out connection would.
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events: Vec<Result<Event, std::io::Error>> = vec![
                Ok(Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_cut","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#)),
                Ok(Event::default()
                    .event("content_block_start")
                    .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)),
                Ok(Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#)),
                Ok(Event::default()
                    .event("content_block_stop")
                    .data(r#"{"type":"content_block_stop","index":0}"#)),
                Ok(Event::default()
                    .event("message_delta")
                    .data(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#)),
                Ok(Event::default()
                    .event("message_stop")
                    .data(r#"{"type":"message_stop"}"#)),
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "upstream connection reset after terminal",
                )),
            ];
            // Yield before the terminal error so the server flushes the
            // frames above first; without this the whole body sits in the
            // write buffer and the abort loses it all.
            let stream = futures::stream::iter(events).then(|item| async move {
                if item.is_err() {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                item
            });
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{addr}/v1/messages")
    }

    async fn spawn_mock_anthropic_reset_before_terminal_stream() -> String {
        use futures::StreamExt;

        // A stream that starts delivering content, then dies on a transport
        // error before any terminal event: the post-commit failure path with
        // no decoder terminal behind it.
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events: Vec<Result<Event, std::io::Error>> = vec![
                Ok(Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_cut2","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#)),
                Ok(Event::default()
                    .event("content_block_start")
                    .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#)),
                Ok(Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#)),
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "upstream connection reset before terminal",
                )),
            ];
            // Yield before the error so the server flushes the frames above
            // first; without this the whole body sits in the write buffer
            // and the abort loses it all.
            let stream = futures::stream::iter(events).then(|item| async move {
                if item.is_err() {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                item
            });
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{addr}/v1/messages")
    }

    async fn spawn_mock_with_request_tracker(response_body: Vec<u8>) -> (String, Arc<AtomicBool>) {
        let received = Arc::new(AtomicBool::new(false));
        let received_clone = Arc::clone(&received);
        let app = Router::new().route(
            "/{*path}",
            post(move || {
                let received = Arc::clone(&received_clone);
                async move {
                    received.store(true, Ordering::SeqCst);
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        response_body.clone(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        (
            format!("http://{}/providers/mock-provider/v1/messages", addr),
            received,
        )
    }

    async fn spawn_mock_with_body_capture(
        response_body: Vec<u8>,
    ) -> (String, Arc<tokio::sync::Mutex<Option<Vec<u8>>>>) {
        let captured: Arc<tokio::sync::Mutex<Option<Vec<u8>>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let captured_clone = Arc::clone(&captured);
        let app = Router::new().route(
            "/{*path}",
            post(move |body: axum::body::Bytes| {
                let captured = Arc::clone(&captured_clone);
                async move {
                    let mut guard = captured.lock().await;
                    *guard = Some(body.to_vec());
                    drop(guard);
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/json")],
                        response_body.clone(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        (
            format!("http://{}/providers/mock-provider/v1/messages", addr),
            captured,
        )
    }

    async fn spawn_mock_anthropic_documented_stream() -> String {
        // The documented Anthropic stream shape: message_start carries the
        // input-side usage, and the final message_delta carries only the
        // cumulative output_tokens alongside the stop reason.
        let app = Router::new().route(
            "/{*path}",
            post(|| async move {
                let events = vec![
                    Event::default()
                        .event("message_start")
                        .data(r#"{"type":"message_start","message":{"id":"msg_doc","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1,"cache_read_input_tokens":7,"cache_creation_input_tokens":3}}}"#),
                    Event::default()
                        .event("content_block_start")
                        .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#),
                    Event::default()
                        .event("content_block_delta")
                        .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#),
                    Event::default()
                        .event("content_block_stop")
                        .data(r#"{"type":"content_block_stop","index":0}"#),
                    Event::default()
                        .event("message_delta")
                        .data(r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":15}}"#),
                    Event::default()
                        .event("message_stop")
                        .data(r#"{"type":"message_stop"}"#),
                ];
                let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    sse.into_response(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_openai_chat_usage_after_finish_stream() -> String {
        // Documented chunk order: the finish_reason chunk arrives before the
        // usage-only chunk.
        let app = Router::new().route(
            "/{*path}",
            post(|| async move {
                let events = vec![
                    Event::default()
                        .data(r#"{"id":"chatcmpl-uf","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#),
                    Event::default()
                        .data(r#"{"id":"chatcmpl-uf","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi!"},"finish_reason":null}]}"#),
                    Event::default()
                        .data(r#"{"id":"chatcmpl-uf","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
                    Event::default()
                        .data(r#"{"id":"chatcmpl-uf","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105}}"#),
                    Event::default()
                        .data("[DONE]"),
                ];
                let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    sse.into_response(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_openai_chat_error_frame_stream(content_first: bool) -> String {
        let app = Router::new().route(
            "/{*path}",
            post(move || async move {
                let mut events = Vec::new();
                if content_first {
                    events.push(Event::default()
                        .data(r#"{"id":"chatcmpl-err","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#));
                    events.push(Event::default()
                        .data(r#"{"id":"chatcmpl-err","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#));
                }
                events.push(Event::default()
                    .data(r#"{"error":{"message":"upstream overloaded","type":"server_error","code":"overloaded"}}"#));
                let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    sse.into_response(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_openai_chat_truncated_stream() -> String {
        // Sends content and then closes the connection mid-stream.
        let app = Router::new().route(
            "/{*path}",
            post(|| async move {
                let events = vec![
                    Event::default()
                        .data(r#"{"id":"chatcmpl-tr","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#),
                    Event::default()
                        .data(r#"{"id":"chatcmpl-tr","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#),
                ];
                let stream = futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    sse.into_response(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    async fn spawn_mock_silent_stream() -> String {
        // Accepts the request and never sends an event.
        let app = Router::new().route(
            "/{*path}",
            post(|| async move {
                let stream = futures::stream::pending::<Result<Event, std::convert::Infallible>>();
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    sse.into_response(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{}/providers/mock-provider/v1/messages", addr)
    }

    fn make_messages_body(model: &str, stream: bool) -> String {
        json!({
            "model": model,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64,
            "stream": stream
        })
        .to_string()
    }

    fn openai_chat_request(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap()
    }

    fn make_openai_chat_stream_body(model: &str, include_usage: bool) -> String {
        json!({
            "model": model,
            "stream": true,
            "stream_options": if include_usage {
                json!({ "include_usage": true })
            } else {
                Value::Null
            },
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        })
        .to_string()
    }

    async fn spawn_mock_openai_chat_stream_with_body_capture()
    -> (String, Arc<tokio::sync::Mutex<Option<Vec<u8>>>>) {
        let captured: Arc<tokio::sync::Mutex<Option<Vec<u8>>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let captured_clone = Arc::clone(&captured);
        let app = Router::new().route(
            "/{*path}",
            post(move |body: axum::body::Bytes| {
                let captured = Arc::clone(&captured_clone);
                async move {
                    let mut guard = captured.lock().await;
                    *guard = Some(body.to_vec());
                    drop(guard);
                    let events = vec![
                        Event::default()
                            .data(r#"{"id":"chatcmpl-cap","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}"#),
                        Event::default()
                            .data(r#"{"id":"chatcmpl-cap","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
                        Event::default()
                            .data(r#"{"id":"chatcmpl-cap","object":"chat.completion.chunk","created":12345,"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}"#),
                        Event::default()
                            .data("[DONE]"),
                    ];
                    let stream = futures::stream::iter(
                        events.into_iter().map(Ok::<_, std::convert::Infallible>),
                    );
                    let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        sse.into_response(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        (
            format!(
                "http://{}/providers/mock-provider/v1/chat/completions",
                addr
            ),
            captured,
        )
    }

    fn messages_request(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap()
    }

    #[tokio::test]
    async fn documented_anthropic_stream_reports_usage_and_max_tokens() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_documented_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        let completed = bus
            .snapshot()
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseCompleted(c) => Some(c.clone()),
                _ => None,
            })
            .expect("documented stream must complete with real usage");
        assert_eq!(completed.usage.input_tokens, 25);
        assert_eq!(completed.usage.output_tokens, 15);
        assert_eq!(completed.usage.cache_read_input_tokens, Some(7));
        assert_eq!(completed.usage.cache_creation_input_tokens, Some(3));
        assert_eq!(
            completed.stop_reason,
            llm_proxy_protocol::core::StopReason::MaxTokens
        );

        assert!(
            text.contains("\"input_tokens\":25"),
            "client message_delta must carry the merged usage: {text}"
        );
        assert!(
            text.contains("\"stop_reason\":\"max_tokens\""),
            "client must see the provider's stop reason: {text}"
        );
    }

    #[tokio::test]
    async fn openai_client_sees_real_usage_after_finish_chunk() {
        let mock_url = spawn_mock_openai_chat_usage_after_finish_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "stream": true,
            "stream_options": { "include_usage": true },
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        let usage_pos = text.find("\"prompt_tokens\":100");
        assert!(
            usage_pos.is_some(),
            "client that asked for usage must see the real numbers: {text}"
        );
        let finish_pos = text.find("\"finish_reason\":\"stop\"");
        assert!(
            finish_pos.is_some(),
            "client must see the finish chunk: {text}"
        );
        assert!(
            finish_pos < usage_pos,
            "finish chunk must precede the usage chunk: {text}"
        );
        assert!(text.trim_end().ends_with("data: [DONE]"));
    }

    #[tokio::test]
    async fn openai_client_without_include_usage_gets_no_usage_chunk() {
        let mock_url = spawn_mock_openai_chat_usage_after_finish_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            !text.contains("\"prompt_tokens\":100"),
            "client that did not ask for usage must not receive a usage chunk: {text}"
        );
    }

    #[tokio::test]
    async fn chat_streams_always_request_usage_upstream() {
        let (mock_url, captured) = spawn_mock_openai_chat_stream_with_body_capture().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        // A client that did NOT set stream_options still makes the proxy ask
        // the upstream for usage.
        let body = make_openai_chat_stream_body("gpt-4o", false);
        let resp = app.oneshot(openai_chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();

        let guard = captured.lock().await;
        let upstream = guard.as_ref().expect("upstream body captured");
        let json: Value = serde_json::from_slice(upstream).unwrap();
        assert_eq!(
            json["stream_options"]["include_usage"], true,
            "the proxy must always ask chat upstreams for stream usage: {json}"
        );
    }

    #[tokio::test]
    async fn leading_chat_error_frame_is_marked_failed_in_ledger() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_openai_chat_error_frame_stream(false).await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_openai_chat_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("gpt-4o", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        // Plan 04: an error frame before any output event never commits —
        // the caller gets an HTTP error status instead of a 200 stream.
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            !text.contains("upstream overloaded"),
            "the upstream's own error schema must not reach the client: {text}"
        );

        let events = bus.snapshot();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseFailed(_))),
            "a leading error frame must not be recorded as success: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseCompleted(_))),
            "a leading error frame must not record a success row: {events:?}"
        );
    }

    #[tokio::test]
    async fn midstream_chat_error_frame_is_marked_failed_and_keeps_framing() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_openai_chat_error_frame_stream(true).await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_openai_chat_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_openai_chat_stream_body("gpt-4o", false);
        let resp = app.oneshot(openai_chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            text.contains("partial"),
            "content before the error must have been delivered: {text}"
        );
        assert!(
            text.contains("upstream overloaded"),
            "client must see the error payload: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "stream must still terminate with [DONE]: {text}"
        );

        let events = bus.snapshot();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseFailed(_))),
            "a mid-stream error frame must be recorded as a failure: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseCompleted(_))),
            "a mid-stream error frame must not record a success row: {events:?}"
        );
    }

    #[tokio::test]
    async fn truncated_chat_stream_is_marked_failed_and_keeps_framing() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_openai_chat_truncated_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_openai_chat_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_openai_chat_stream_body("gpt-4o", false);
        let resp = app.oneshot(openai_chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            text.contains("upstream stream ended before the response completed"),
            "the client must see an in-band error for the cut stream: {text}"
        );
        assert!(
            text.contains("\"finish_reason\":\"stop\""),
            "synthetic terminal must keep the framing valid: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "stream must still terminate with [DONE]: {text}"
        );

        let events = bus.snapshot();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseFailed(_))),
            "a truncated stream must be recorded as a failure: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseCompleted(_))),
            "a truncated stream must not record a success row: {events:?}"
        );
    }

    #[tokio::test]
    async fn truncated_anthropic_stream_is_marked_failed_and_keeps_framing() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_disconnect_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            text.contains("event: error"),
            "the Anthropic client must see an in-band error event: {text}"
        );
        assert!(
            text.contains("event: message_stop"),
            "the synthetic terminal must keep framing valid: {text}"
        );

        let events = bus.snapshot();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseFailed(_))),
            "a truncated stream must be recorded as a failure: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseCompleted(_))),
            "a truncated stream must not record a success row: {events:?}"
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "one end-to-end stream scenario plus its ledger assertions; the two config \
                  fields plan 06 added pushed it over the mechanical line budget"
    )]
    async fn streaming_alias_rows_keep_model_upstream() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_stream().await;
        let mut provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "messages".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "anthropic_messages".to_owned(),
                        endpoint: mock_url,
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: llm_proxy_core::ProviderRoutesConfig {
                messages: Some("messages".to_owned()),
                chat_completions: None,
                ..Default::default()
            },
            model_aliases: HashMap::from([(
                "claude-alias".to_owned(),
                "claude-sonnet-4-6".to_owned(),
            )]),
            discovery: None,
            catalog: None,
            pricing: HashMap::new(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        provider.name = "mock-provider".to_owned();
        let registry =
            llm_proxy_core::ProviderRegistry::from_providers(vec![provider]).expect("registry");
        let app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: Duration::from_secs(300),
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "test-proxy".to_owned(),
                rate_limit_rpm: 100,
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
        };
        let bus = Arc::new(RecordingBus::new());
        let state = AppState::new(
            app_config,
            registry,
            ProviderAdapterRegistry::builtin(),
            ProxyClient::new(),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
        .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-alias", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();

        let completed = bus
            .snapshot()
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseCompleted(c) => Some(c.clone()),
                _ => None,
            })
            .expect("streaming row emitted");
        assert_eq!(completed.model.requested, "claude-alias");
        assert_eq!(
            completed.model.upstream.as_deref(),
            Some("claude-sonnet-4-6"),
            "streaming rows must keep model_upstream under aliases"
        );
    }

    #[tokio::test]
    async fn client_disconnect_before_first_event_settles_the_row() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_silent_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));

        let app = build_router(state);
        let body = make_messages_body("claude-sonnet-4-6", true);
        let request = messages_request(&body);
        let disconnect = || async {
            let fut = app.clone().oneshot(request);
            tokio::pin!(fut);

            // The upstream never sends an event; give up like a client that
            // disconnects before the first SSE byte. Leaving this block
            // drops the in-flight handler future.
            let result = tokio::time::timeout(Duration::from_millis(300), &mut fut).await;
            match result {
                Ok(Ok(_)) => panic!("response must not complete while the upstream is silent"),
                Ok(Err(_)) => panic!("oneshot failed"),
                Err(_) => {}
            }
        };
        disconnect().await;

        // The stream task must settle the RequestReceived row on its own.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            let events = bus.snapshot();
            let settled = events.iter().any(|e| {
                matches!(e, ProxyEvent::ResponseCompleted(_))
                    || matches!(e, ProxyEvent::ResponseFailed(_))
            });
            if settled {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "a disconnect before the first event must leave a settled row; events: {:?}",
                events
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    #[tokio::test]
    async fn anthropic_provider_returns_anthropic_response() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert!(
            body["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Hello from mock!")
        );
    }

    #[tokio::test]
    async fn request_id_header_present_on_success() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let request_id = resp.headers().get("x-request-id");
        assert!(
            request_id.is_some(),
            "x-request-id header must be present on success"
        );
        let id_str = request_id.unwrap().to_str().unwrap();
        assert!(!id_str.is_empty(), "x-request-id must not be empty");
    }

    #[tokio::test]
    async fn upstream_500_returns_502() {
        let mock_url = spawn_mock_500().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "error");
        assert_eq!(resp_body["error"]["type"], "api_error");
    }

    #[tokio::test]
    async fn unknown_model_is_passed_through() {
        let (mock_url, received) =
            spawn_mock_with_request_tracker(anthropic_success_response()).await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("nonexistent-model", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            received.load(Ordering::SeqCst),
            "upstream should be called -- model name is passed through"
        );
    }

    #[tokio::test]
    async fn openai_chat_provider_returns_anthropic_response() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("gpt-4o", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "message");
        assert_eq!(resp_body["role"], "assistant");
        assert!(
            resp_body["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Hello from OpenAI mock!"),
            "response should contain translated OpenAI text"
        );
    }

    #[tokio::test]
    async fn openai_responses_provider_returns_anthropic_response() {
        let mock_url = spawn_mock_openai_responses_non_stream().await;
        let state = state_with_openai_responses_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("gpt-4o-responses", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "message");
        assert_eq!(resp_body["role"], "assistant");
        assert!(
            resp_body["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Hello from Responses mock!"),
            "response should contain translated Responses text"
        );
    }

    #[tokio::test]
    async fn gemini_provider_returns_anthropic_response() {
        let mock_url = spawn_mock_gemini_non_stream().await;
        let state = state_with_gemini_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("gemini-2.5-pro", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "message");
        assert_eq!(resp_body["role"], "assistant");
        assert!(
            resp_body["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("Hello from Gemini mock!"),
            "response should contain translated Gemini text"
        );
    }

    async fn collect_anthropic_stream_body(mock_url: &str) -> String {
        let state = state_with_anthropic_provider(mock_url);
        let app = build_router(state);
        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn stream_anthropic_provider_returns_sse_content_type() {
        let mock_url = spawn_mock_anthropic_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }

    #[tokio::test]
    async fn stream_anthropic_provider_includes_request_id() {
        let mock_url = spawn_mock_anthropic_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let request_id = resp.headers().get("x-request-id");
        assert!(
            request_id.is_some(),
            "x-request-id must be present on stream response"
        );
    }

    #[tokio::test]
    async fn stream_anthropic_provider_emits_sse_protocol_events() {
        let mock_url = spawn_mock_anthropic_stream().await;
        let text = collect_anthropic_stream_body(&mock_url).await;

        assert!(
            text.contains("event: message_start"),
            "missing message_start event"
        );
        assert!(
            text.contains("event: content_block_delta"),
            "missing content_block_delta"
        );
        assert!(
            text.contains("event: message_stop"),
            "missing message_stop event"
        );
    }

    #[tokio::test]
    async fn stream_anthropic_provider_forwards_text_delta() {
        let mock_url = spawn_mock_anthropic_stream().await;
        let text = collect_anthropic_stream_body(&mock_url).await;

        assert!(
            text.contains("Hi!"),
            "expected text delta 'Hi!' in SSE output"
        );
    }

    #[tokio::test]
    async fn stream_anthropic_provider_surfaces_upstream_message_id() {
        let mock_url = spawn_mock_anthropic_stream().await;
        let text = collect_anthropic_stream_body(&mock_url).await;
        assert!(
            text.contains("msg_mock_stream"),
            "client SSE must surface the upstream message id; got: {text}"
        );
    }

    async fn collect_openai_chat_stream_body(mock_url: &str) -> String {
        let state = state_with_openai_chat_provider(mock_url);
        let app = build_router(state);
        let body = make_messages_body("gpt-4o", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn stream_openai_chat_provider_surfaces_upstream_message_id() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let text = collect_openai_chat_stream_body(&mock_url).await;
        assert!(
            text.contains("chatcmpl-stream"),
            "client SSE must surface the upstream chunk id; got: {text}"
        );
    }

    #[tokio::test]
    async fn event_bus_records_openai_chat_stream_upstream_message_id() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_openai_chat_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_openai_chat_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("gpt-4o", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();

        let events = bus.snapshot();
        let completed = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseCompleted(c) => Some(c),
                _ => None,
            })
            .expect("ResponseCompleted emitted for OpenAI Chat stream");
        assert_eq!(
            completed.upstream_message_id.as_deref(),
            Some("chatcmpl-stream"),
            "OpenAI Chat stream must capture the real upstream chunk id on the event bus"
        );
    }

    #[tokio::test]
    async fn event_bus_records_request_received_and_response_completed() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let events = bus.snapshot();
        assert_eq!(
            events.len(),
            3,
            "expected RequestReceived + AttemptCompleted + ResponseCompleted"
        );
        assert!(
            matches!(events[0], ProxyEvent::RequestReceived(_)),
            "first event must be RequestReceived"
        );
        assert!(
            matches!(events[1], ProxyEvent::AttemptCompleted(_)),
            "second event must be the upstream attempt row"
        );
        match &events[2] {
            ProxyEvent::ResponseCompleted(rc) => {
                assert_eq!(rc.provider, "mock-provider");
                assert_eq!(rc.upstream_message_id.as_deref(), Some("msg_mock123"));
                assert_eq!(rc.usage.input_tokens, 10);
                assert_eq!(rc.usage.output_tokens, 5);
                assert!(
                    rc.cost.is_none(),
                    "cost must be None when no pricing is configured, got {:?}",
                    rc.cost
                );
            }
            other => panic!("expected ResponseCompleted, got {other:?}"),
        }

        let json = serde_json::to_string(&events).expect("serialize events");
        assert!(
            !json.contains("test-key") && !json.contains("sk-"),
            "event JSON must not contain api-key fragments: {json}"
        );
    }

    #[tokio::test]
    async fn priced_provider_attaches_cost_to_response_completed() {
        use std::sync::Arc;

        use llm_proxy_core::{ModelId, ModelPricing};
        use llm_proxy_storage::{ProxyEvent, RecordingBus};
        use rust_decimal::Decimal;

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let pricing = HashMap::from([(
            ModelId::new("claude-sonnet-4-6"),
            ModelPricing {
                input: "0.000003".parse::<Decimal>().unwrap(),
                output: "0.000015".parse::<Decimal>().unwrap(),
                ..ModelPricing::default()
            },
        )]);
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider_priced(&mock_url, pricing).with_event_bus(Arc::<
            RecordingBus,
        >::clone(
            &bus
        ));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let rc = bus
            .snapshot()
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseCompleted(rc) => Some(rc.clone()),
                _ => None,
            })
            .expect("ResponseCompleted emitted");
        let cost = rc
            .cost
            .expect("cost must be Some when pricing is configured");
        assert_eq!(cost.input.normalize().to_string(), "0.00003");
        assert_eq!(cost.output.normalize().to_string(), "0.000075");
        assert_eq!(cost.total.normalize().to_string(), "0.000105");
    }

    #[tokio::test]
    async fn stream_multi_event_first_frame_emits_all_without_loss_or_duplication() {
        let body = anthropic_sse_body("msg_multi", "");
        let mock_url = spawn_mock_server(body, "text/event-stream").await;
        let text = collect_anthropic_stream_body(&mock_url).await;
        assert_eq!(
            text.matches("event: message_start").count(),
            1,
            "message_start must appear exactly once: {text}"
        );
        assert_eq!(
            text.matches("event: content_block_delta").count(),
            1,
            "content_block_delta must appear exactly once (no loss/duplication): {text}"
        );
        assert_eq!(
            text.matches("event: message_stop").count(),
            1,
            "message_stop must appear exactly once: {text}"
        );
        assert!(
            text.contains("msg_multi"),
            "real upstream id must surface: {text}"
        );
    }

    #[tokio::test]
    async fn event_bus_records_stream_request_and_response_completed() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();

        let events = bus.snapshot();
        let received = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::RequestReceived(r) => Some(r),
                _ => None,
            })
            .expect("RequestReceived emitted for stream");
        assert!(
            received.streaming,
            "stream request must be marked streaming"
        );
        let completed = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseCompleted(c) => Some(c),
                _ => None,
            })
            .expect("ResponseCompleted emitted for stream");
        assert_eq!(
            completed.upstream_message_id.as_deref(),
            Some("msg_mock_stream")
        );
    }

    fn client_auth_for(key_id: &str, token: &str) -> llm_proxy_server::ClientAuth {
        use argon2::{Algorithm, Argon2, CustomizedPasswordHasher, Params, Version};
        let params = Params::new(8, 1, 1, None).expect("valid test params");
        let hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params.clone());
        let phc = hasher
            .hash_password_customized(
                token.as_bytes(),
                b"0123456789abcdef",
                Some(&argon2::ARGON2ID_IDENT),
                None,
                params,
            )
            .expect("hash token")
            .to_string();
        let mut clients = HashMap::new();
        clients.insert(
            key_id.to_owned(),
            llm_proxy_core::ClientKeySpec::Verifier(secrecy::SecretString::from(phc)),
        );
        llm_proxy_server::ClientAuth::from_config(&llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
        })
        .expect("valid client auth")
    }

    #[tokio::test]
    async fn enforced_client_keys_reject_missing_bearer_with_sanitized_401() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url).with_client_auth(std::sync::Arc::new(
            client_auth_for("laptop", "sekrit-token"),
        ));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(
            resp.headers().get("x-request-id").is_some(),
            "401 must carry the request id"
        );
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            !text.contains("sekrit-token"),
            "401 body must not echo any secret material, got: {text}"
        );
    }

    #[tokio::test]
    async fn enforced_client_keys_reject_wrong_bearer_without_echoing_it() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url).with_client_auth(std::sync::Arc::new(
            client_auth_for("laptop", "sekrit-token"),
        ));
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "Bearer attacker-guess")
            .body(Body::from(make_messages_body("claude-sonnet-4-6", false)))
            .unwrap();
        let resp = app.oneshot(request).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(
            !text.contains("attacker-guess"),
            "401 body must not echo the attempted token, got: {text}"
        );
    }

    #[tokio::test]
    async fn valid_bearer_passes_and_attributes_caller_on_event() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus))
            .with_client_auth(std::sync::Arc::new(client_auth_for(
                "laptop",
                "sekrit-token",
            )));
        let app = build_router(state);

        let request = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", "bearer sekrit-token")
            .body(Body::from(make_messages_body("claude-sonnet-4-6", false)))
            .unwrap();
        let resp = app.oneshot(request).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let events = bus.snapshot();
        let received = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::RequestReceived(r) => Some(r),
                _ => None,
            })
            .expect("RequestReceived emitted");
        assert_eq!(received.caller, "laptop");
        let event_json =
            serde_json::to_string(&ProxyEvent::RequestReceived(received.clone())).unwrap();
        assert!(
            !event_json.contains("sekrit-token"),
            "client secret must never appear on the event, got: {event_json}"
        );
    }

    #[tokio::test]
    async fn no_auth_mode_attributes_anonymous_caller() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let events = bus.snapshot();
        let received = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::RequestReceived(r) => Some(r),
                _ => None,
            })
            .expect("RequestReceived emitted");
        assert_eq!(received.caller, "anonymous");
    }

    #[tokio::test]
    async fn enforced_client_keys_leave_health_open() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url).with_client_auth(std::sync::Arc::new(
            client_auth_for("laptop", "sekrit-token"),
        ));
        let app = build_router(state);

        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(request).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn event_bus_records_response_failed_for_unknown_provider() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let req = Request::builder()
            .method("POST")
            .uri("/providers/does-not-exist/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let events = bus.snapshot();
        let failed = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseFailed(f) => Some(f),
                _ => None,
            })
            .expect("ResponseFailed emitted for unknown provider");
        assert_eq!(failed.http_status, 404);
        assert_eq!(failed.provider.as_deref(), Some("does-not-exist"));
    }

    #[tokio::test]
    async fn empty_upstream_stream_returns_502_not_200() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_server(Vec::new(), "text/event-stream").await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_GATEWAY,
            "empty stream must be 502, not 200"
        );

        let events = bus.snapshot();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseCompleted(_))),
            "no ResponseCompleted for an empty stream"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseFailed(_))),
            "ResponseFailed emitted for an empty stream"
        );
    }

    fn anthropic_sse_body(message_start_id: &str, prefix: &str) -> Vec<u8> {
        let mut s = String::new();
        match prefix {
            "keepalive" => s.push_str(":keepalive\n\n"),
            "ping" => s.push_str("event: ping\ndata: {\"type\":\"ping\"}\n\n"),
            _ => {}
        }
        s.push_str(&format!(
            "event: message_start\ndata: {}\n\n",
            json!({
                "type": "message_start",
                "message": {
                    "id": message_start_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 10, "output_tokens": 0 }
                }
            })
        ));
        s.push_str(&format!(
        "event: content_block_delta\ndata: {}\n\n",
        json!({ "type": "content_block_delta", "delta": { "type": "text_delta", "text": "Hi!" } })
    ));
        s.push_str(&format!(
            "event: message_stop\ndata: {}\n\n",
            json!({ "type": "message_stop" })
        ));
        s.into_bytes()
    }

    #[tokio::test]
    async fn stream_empty_message_start_id_falls_back_to_synthetic() {
        let body = anthropic_sse_body("", "");
        let mock_url = spawn_mock_server(body, "text/event-stream").await;
        let text = collect_anthropic_stream_body(&mock_url).await;
        assert!(
            text.contains("\"id\":\"msg_"),
            "empty upstream id must produce a synthetic msg_-prefixed id; got: {text}"
        );
    }

    #[tokio::test]
    async fn stream_ping_then_message_start_surfaces_real_id() {
        let body = anthropic_sse_body("msg_ping_real", "ping");
        let mock_url = spawn_mock_server(body, "text/event-stream").await;
        let text = collect_anthropic_stream_body(&mock_url).await;
        assert!(
            text.contains("msg_ping_real"),
            "real upstream id must survive a leading ping; got: {text}"
        );
    }

    #[tokio::test]
    async fn stream_keepalive_then_message_start_surfaces_real_id() {
        let body = anthropic_sse_body("msg_ka_real", "keepalive");
        let mock_url = spawn_mock_server(body, "text/event-stream").await;
        let text = collect_anthropic_stream_body(&mock_url).await;
        assert!(
            text.contains("msg_ka_real"),
            "real upstream id must survive a leading keepalive comment; got: {text}"
        );
    }

    #[tokio::test]
    async fn stream_openai_chat_provider_returns_sse_content_type() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("gpt-4o", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }

    #[tokio::test]
    async fn stream_openai_chat_provider_emits_sse_protocol_events() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let text = collect_openai_chat_stream_body(&mock_url).await;

        assert!(
            text.contains("event: message_start"),
            "missing message_start event"
        );
        assert!(
            text.contains("event: message_stop"),
            "missing message_stop event"
        );
    }

    #[tokio::test]
    async fn stream_openai_chat_provider_forwards_translated_text_delta() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let text = collect_openai_chat_stream_body(&mock_url).await;

        assert!(
            text.contains("Hi from OpenAI!"),
            "expected translated text delta in SSE output, got: {text}"
        );
    }

    async fn collect_openai_responses_stream_body(mock_url: &str) -> String {
        let state = state_with_openai_responses_provider(mock_url);
        let app = build_router(state);
        let body = make_messages_body("gpt-4o-responses", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn stream_openai_responses_provider_returns_sse_content_type() {
        let mock_url = spawn_mock_openai_responses_stream().await;
        let state = state_with_openai_responses_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("gpt-4o-responses", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }

    #[tokio::test]
    async fn stream_openai_responses_provider_emits_sse_protocol_events() {
        let mock_url = spawn_mock_openai_responses_stream().await;
        let text = collect_openai_responses_stream_body(&mock_url).await;

        assert!(
            text.contains("event: message_start"),
            "missing message_start event"
        );
        assert!(
            text.contains("event: message_stop"),
            "missing message_stop event"
        );
    }

    #[tokio::test]
    async fn stream_openai_responses_provider_forwards_translated_text_delta() {
        let mock_url = spawn_mock_openai_responses_stream().await;
        let text = collect_openai_responses_stream_body(&mock_url).await;

        assert!(
            text.contains("Hi from Responses!"),
            "expected translated text delta in SSE output, got: {text}"
        );
    }

    async fn collect_gemini_stream_body(mock_url: &str) -> String {
        let state = state_with_gemini_provider(mock_url);
        let app = build_router(state);
        let body = make_messages_body("gemini-2.5-pro", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn stream_gemini_provider_returns_sse_content_type() {
        let mock_url = spawn_mock_gemini_stream().await;
        let state = state_with_gemini_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("gemini-2.5-pro", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/event-stream"),
            "expected SSE content-type, got: {ct}"
        );
    }

    #[tokio::test]
    async fn stream_gemini_provider_emits_sse_protocol_events() {
        let mock_url = spawn_mock_gemini_stream().await;
        let text = collect_gemini_stream_body(&mock_url).await;

        assert!(
            text.contains("event: message_start"),
            "missing message_start event"
        );
        assert!(
            text.contains("event: message_stop"),
            "missing message_stop event"
        );
    }

    #[tokio::test]
    async fn stream_gemini_provider_forwards_translated_text_delta() {
        let mock_url = spawn_mock_gemini_stream().await;
        let text = collect_gemini_stream_body(&mock_url).await;

        assert!(
            text.contains("Hi from Gemini!"),
            "expected translated text delta in SSE output, got: {text}"
        );
    }

    #[tokio::test]
    async fn stream_error_before_first_byte_returns_http_502() {
        let mock_url = spawn_mock_500().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_GATEWAY,
            "stream with upstream 500 should return 502"
        );
    }

    #[tokio::test]
    async fn malformed_frame_is_skipped_and_valid_events_delivered() {
        let mock_url = spawn_mock_malformed_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("event: message_start"),
            "stream should start with message_start"
        );

        assert!(
            text.contains("good-delta"),
            "valid events before a malformed frame must be delivered, got: {text}"
        );

        assert!(
            text.contains("event: message_stop"),
            "stream should complete with message_stop even after malformed data"
        );
    }

    #[tokio::test]
    async fn upstream_disconnect_completes_with_synthetic_terminal() {
        let mock_url = spawn_mock_disconnect_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("event: message_start"),
            "stream should start with message_start"
        );

        assert!(
            text.contains("event: message_stop"),
            "stream should have a synthetic message_stop after premature upstream end"
        );
    }

    #[tokio::test]
    async fn transport_error_after_terminal_keeps_terminal_and_error_events() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_terminal_then_reset_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("partial"),
            "content before the terminal must have been delivered: {text}"
        );
        let stop_pos = text.find("event: message_stop").unwrap_or_else(|| {
            panic!("the buffered terminal pair must survive the transport error: {text}")
        });
        let err_pos = text
            .find("event: error")
            .unwrap_or_else(|| panic!("the transport error must reach the client in-band: {text}"));
        assert!(
            stop_pos < err_pos,
            "the buffered terminal pair must precede the error event: {text}"
        );

        let events = bus.snapshot();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseFailed(_))),
            "a transport error after the terminal must be recorded as a failure: {events:?}"
        );
    }

    #[tokio::test]
    async fn a_post_commit_failure_emits_exactly_one_error_event() {
        // A transport error after the stream committed already crosses to
        // the client in-band; the missing-terminal backstop must not append
        // a second (truncation) error event on top of it.
        let mock_url = spawn_mock_anthropic_reset_before_terminal_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("partial"),
            "content before the failure must have been delivered: {text}"
        );
        assert_eq!(
            text.matches("event: error").count(),
            1,
            "exactly one terminal error event must cross, got: {text}"
        );
        assert!(
            text.contains("event: message_stop"),
            "the synthetic terminal must still close the stream: {text}"
        );
    }

    #[tokio::test]
    async fn in_band_upstream_error_emits_anthropic_error_event() {
        let mock_url = spawn_mock_in_band_error_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("event: message_start"),
            "stream should start with message_start"
        );

        assert!(
            text.contains("event: error"),
            "in-band upstream error must emit an event: error frame, got: {text}"
        );
        assert!(
            text.contains("Too many requests"),
            "event: error should carry the upstream error message, got: {text}"
        );
    }

    struct SlowDripStream {
        first: Option<bytes::Bytes>,
        canceled: Arc<AtomicBool>,
    }

    impl std::fmt::Debug for SlowDripStream {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SlowDripStream")
                .field("first_pending", &self.first.is_some())
                .field("canceled", &self.canceled)
                .finish()
        }
    }

    impl futures::Stream for SlowDripStream {
        type Item = Result<bytes::Bytes, std::io::Error>;

        fn poll_next(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            if let Some(chunk) = self.first.take() {
                std::task::Poll::Ready(Some(Ok(chunk)))
            } else {
                std::task::Poll::Pending
            }
        }
    }

    impl Drop for SlowDripStream {
        fn drop(&mut self) {
            self.canceled.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn client_disconnect_cancels_upstream_stream() {
        use futures::StreamExt;

        let first_frame = bytes::Bytes::from(
        "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cancel\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n"
            .to_owned(),
    );
        let upstream_canceled = Arc::new(AtomicBool::new(false));
        let request_received = Arc::new(AtomicBool::new(false));

        let canceled_for_route = Arc::clone(&upstream_canceled);
        let received_for_route = Arc::clone(&request_received);
        let app = Router::new().route(
            "/{*path}",
            post(move || {
                let canceled = Arc::clone(&canceled_for_route);
                let received = Arc::clone(&received_for_route);
                async move {
                    received.store(true, Ordering::SeqCst);
                    let body = SlowDripStream {
                        first: Some(first_frame.clone()),
                        canceled,
                    };
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(body).into_response(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        let mock_url = format!("http://{}/providers/mock-provider/v1/messages", addr);

        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut stream = resp.into_body().into_data_stream();
        let first_chunk = stream.next().await;
        assert!(
            first_chunk.is_some(),
            "client should receive at least the first SSE chunk"
        );
        drop(stream);

        let teardown = tokio::time::timeout(Duration::from_secs(2), async {
            while !upstream_canceled.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await;

        assert!(
            teardown.is_ok(),
            "upstream stream must be torn down within 2s of client disconnect \
         (cancellation did not propagate; record_client_cancel path is broken)"
        );
        assert!(
            upstream_canceled.load(Ordering::SeqCst),
            "upstream SlowDripStream must have been dropped on client disconnect"
        );
        assert!(
            request_received.load(Ordering::SeqCst),
            "mock upstream should have received the request"
        );
    }

    #[tokio::test]
    async fn stream_terminal_event_emitted_after_decoder_finish() {
        let mock_url = spawn_mock_disconnect_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("event: message_start"),
            "should have message_start from upstream"
        );
        assert!(
            text.contains("event: content_block_start"),
            "should have content_block_start from upstream"
        );

        let start_pos = text.find("event: message_start").expect("message_start");
        let stop_pos = text.find("event: message_stop").expect("message_stop");
        assert!(
            stop_pos > start_pos,
            "message_stop must appear after message_start"
        );

        let delta_pos = text.find("event: message_delta").expect("message_delta");
        assert!(
            delta_pos < stop_pos,
            "message_delta must appear before message_stop"
        );
        assert!(
            delta_pos > start_pos,
            "message_delta must appear after message_start"
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one end-to-end scenario: the config, provider and request setup read as a unit"
    )]
    #[tokio::test]
    async fn rate_limited_request_returns_429() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "messages".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "anthropic_messages".to_owned(),
                        endpoint: mock_url,
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: llm_proxy_core::ProviderRoutesConfig {
                messages: Some("messages".to_owned()),
                chat_completions: None,
                ..Default::default()
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
        let state = AppState::new(
            AppConfig {
                server: ServerConfig {
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: Duration::from_secs(300),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test-proxy".to_owned(),
                    rate_limit_rpm: 1,
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
            ProxyClient::new(),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        );
        let app = build_router(state);

        let body1 = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello one" }],
            "max_tokens": 64
        });
        let resp1 = app
            .clone()
            .oneshot(messages_request(&body1.to_string()))
            .await
            .unwrap();
        assert_ne!(
            resp1.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "first request should not be rate-limited"
        );

        let body2 = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello two" }],
            "max_tokens": 64
        });
        let resp2 = app
            .clone()
            .oneshot(messages_request(&body2.to_string()))
            .await
            .unwrap();
        assert_eq!(
            resp2.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "second request should be rate-limited with rpm=1"
        );
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp2.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "error");
        assert_eq!(resp_body["error"]["type"], "rate_limit_error");
    }

    #[tokio::test]
    async fn identical_requests_are_never_answered_409_by_default() {
        // The dedup window was retired (plan 06 step 1): identical requests
        // are answered, never 409-rejected, when every opt-in is off. Each
        // one reaches the upstream; per-caller single-flight only merges
        // requests that are in flight at the same instant, and these two are
        // sequential.
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);

        let resp1 = app.clone().oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(
            resp1.status(),
            StatusCode::OK,
            "first request must be answered, not rejected"
        );

        let resp2 = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(
            resp2.status(),
            StatusCode::OK,
            "the retired dedup window must not 409 the second identical request"
        );
    }

    #[tokio::test]
    async fn response_cache_opt_in_serves_the_second_identical_request() {
        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_provider_edited(
            &mock_url,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
            HashMap::new(),
            |config| {
                config.server.cache = llm_proxy_core::ResponseCacheConfig {
                    enabled: true,
                    ttl: Duration::from_secs(60),
                    max_entries: 16,
                };
            },
        )
        .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let first = app.clone().oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first_bytes = axum::body::to_bytes(first.into_body(), 64 * 1024)
            .await
            .unwrap();

        // Where the retired dedup window answered 409, the cache answers
        // the identical request with the stored response.
        let second = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        let second_bytes = axum::body::to_bytes(second.into_body(), 64 * 1024)
            .await
            .unwrap();
        // The first response returns the upstream bytes verbatim
        // (same-format passthrough); the hit re-encodes the stored core
        // response, so the equality that matters is content equality.
        let first_json: Value = serde_json::from_slice(&first_bytes).unwrap();
        let second_json: Value = serde_json::from_slice(&second_bytes).unwrap();
        assert_eq!(
            first_json["content"], second_json["content"],
            "a cache hit returns the stored response content"
        );
        assert_eq!(first_json["id"], second_json["id"]);
        assert_eq!(first_json["usage"], second_json["usage"]);

        let rows: Vec<bool> = bus
            .snapshot()
            .into_iter()
            .filter_map(|event| match event {
                ProxyEvent::ResponseCompleted(row) => Some(row.cache_hit),
                _ => None,
            })
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.iter().filter(|hit| **hit).count(), 1);
    }

    #[cfg(feature = "semantic-cache")]
    #[tokio::test]
    async fn enabling_both_cache_optins_refuses_to_build_an_engine() {
        // The engine holds one cache: both opt-ins together would silently
        // replace the exact-match cache with the semantic one. Programmatic
        // configs are refused at the engine build, mirroring the load-time
        // refusal in llm-proxy-core.
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_provider_edited(
            &mock_url,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
            HashMap::new(),
            |config| {
                config.server.cache = llm_proxy_core::ResponseCacheConfig {
                    enabled: true,
                    ttl: Duration::from_secs(60),
                    max_entries: 16,
                };
                config.server.semantic_cache.enabled = true;
            },
        );
        let error = state.engine().expect_err("both cache opt-ins must refuse");
        assert!(
            error.to_string().contains("mutually exclusive"),
            "the refusal must name the conflict, got: {error}"
        );
    }

    #[tokio::test]
    async fn idempotency_key_opt_in_replays_for_a_matching_retry() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_provider_edited(
            &mock_url,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
            HashMap::new(),
            |config| {
                config.server.idempotency = llm_proxy_core::IdempotencyConfig {
                    enabled: true,
                    ttl: Duration::from_secs(3600),
                    max_entries: 1024,
                };
            },
        );
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let keyed = |body: &str| {
            Request::builder()
                .method("POST")
                .uri("/providers/mock-provider/v1/messages")
                .header("content-type", "application/json")
                .header("Idempotency-Key", "retry-after-timeout-1")
                .body(Body::from(body.to_owned()))
                .unwrap()
        };

        let first = app.clone().oneshot(keyed(&body)).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first_bytes = axum::body::to_bytes(first.into_body(), 64 * 1024)
            .await
            .unwrap();

        // A client retry after a timeout replays the stored response instead
        // of executing and billing again.
        let second = app.oneshot(keyed(&body)).await.unwrap();
        assert_eq!(second.status(), StatusCode::OK);
        let second_bytes = axum::body::to_bytes(second.into_body(), 64 * 1024)
            .await
            .unwrap();
        assert_eq!(first_bytes, second_bytes);
    }

    #[tokio::test]
    async fn token_count_with_tools_returns_positive_count() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64,
            "tools": [{
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    }
                }
            }]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            resp_body["input_tokens"].as_u64().unwrap() > 0,
            "token count should be positive even when tools are present"
        );
    }

    #[tokio::test]
    async fn token_count_with_non_text_content_returns_positive_count() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "describe this" },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "fake" } }
                ]
            }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            resp_body["input_tokens"].as_u64().unwrap() > 0,
            "token count should be positive even with non-text content"
        );
    }

    #[tokio::test]
    async fn token_count_with_tool_result_returns_positive_count() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                { "role": "user", "content": "what is the weather?" },
                { "role": "assistant", "content": [{ "type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": { "location": "SF" } }] },
                { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny, 72F" }] }
            ],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            resp_body["input_tokens"].as_u64().unwrap() > 0,
            "token count should be positive even with tool_result content"
        );
    }

    #[tokio::test]
    async fn token_count_endpoint_works_with_provider_state() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn invalid_json_returns_400_anthropic_shape() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn tool_result_with_undecodable_nested_block_returns_400() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [
                        { "type": "text", "text": "ok" },
                        { "type": "tool_use", "name": "no_id" }
                    ]
                }]
            }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "error");
        assert_eq!(resp_body["error"]["type"], "invalid_request_error");
        assert!(
            resp_body["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("tool_result content")),
            "error body should name tool_result content, got {resp_body}"
        );
    }

    #[tokio::test]
    async fn provider_messages_route_works() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn route_preserves_fields_through_core() {
        let (mock_url, captured_body) =
            spawn_mock_with_body_capture(anthropic_success_response()).await;
        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "system": "You are a helpful assistant.",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": false
        });
        let resp = app
            .oneshot(messages_request(&body.to_string()))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        tokio::time::sleep(Duration::from_millis(100)).await;
        let guard = captured_body.lock().await;
        let upstream_body = guard
            .as_ref()
            .expect("upstream should have received a request body");
        let upstream_json: Value =
            serde_json::from_slice(upstream_body).expect("upstream body should be valid JSON");

        assert_eq!(upstream_json["model"], "claude-sonnet-4-6");
        assert!(
            upstream_json["system"].is_string() || upstream_json["system"].is_array(),
            "system field should be preserved"
        );
        assert_eq!(upstream_json["max_tokens"], 128);
        assert_eq!(upstream_json["temperature"], 0.7);
        assert_eq!(upstream_json["top_p"], 0.9);
        let stream_val = &upstream_json["stream"];
        assert!(
            stream_val.is_null() || stream_val == &json!(false),
            "stream field should be false or absent, got: {stream_val}"
        );
    }

    fn state_for_validation_tests() -> AppState {
        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "messages".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "anthropic_messages".to_owned(),
                        endpoint: "https://127.0.0.1:0/v1/messages".to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: llm_proxy_core::ProviderRoutesConfig {
                messages: Some("messages".to_owned()),
                chat_completions: Some("messages".to_owned()),
                ..Default::default()
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
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: Duration::from_secs(300),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test-proxy".to_owned(),
                    rate_limit_rpm: 100,
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
            ProxyClient::new(),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    #[tokio::test]
    async fn unknown_provider_name_returns_404() {
        let app = build_router(state_for_validation_tests());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/nonexistent-provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resp_body["type"], "error");
        assert_eq!(resp_body["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn provider_name_with_uppercase_returns_400() {
        let app = build_router(state_for_validation_tests());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/MyProvider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn provider_name_with_special_chars_returns_400() {
        let app = build_router(state_for_validation_tests());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/my%20provider/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_on_messages_route_returns_405() {
        let app = build_router(state_for_validation_tests());
        let req = Request::builder()
            .method("GET")
            .uri("/providers/mock-provider/v1/messages")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn missing_content_type_on_messages_still_parses() {
        let app = build_router(state_for_validation_tests());
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 64
        });
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert!(
            resp.status().is_client_error()
                || resp.status().is_server_error()
                || resp.status() == StatusCode::OK
        );
    }

    #[tokio::test]
    async fn stream_emits_request_received_before_response_failed_on_upstream_failure() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_500().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", true);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("draining the error body should succeed");

        let events = bus.snapshot();
        let received_idx = events
            .iter()
            .position(|e| matches!(e, ProxyEvent::RequestReceived(_)));
        let failed_idx = events
            .iter()
            .position(|e| matches!(e, ProxyEvent::ResponseFailed(_)));
        assert!(
            received_idx.is_some(),
            "RequestReceived must be emitted for a failing stream"
        );
        assert!(
            failed_idx.is_some(),
            "ResponseFailed must be emitted for a failing stream"
        );
        assert!(
            received_idx.unwrap() < failed_idx.unwrap(),
            "RequestReceived must precede ResponseFailed (no orphaned failure)"
        );
    }

    #[tokio::test]
    async fn non_stream_upstream_500_emits_response_failed_not_completed() {
        use std::sync::Arc;

        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_500().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_anthropic_provider(&mock_url)
            .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let app = build_router(state);

        let body = make_messages_body("claude-sonnet-4-6", false);
        let resp = app.oneshot(messages_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let events = bus.snapshot();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ProxyEvent::ResponseCompleted(_))),
            "a failed non-stream request must not emit ResponseCompleted"
        );
        let failed = events
            .iter()
            .find_map(|e| match e {
                ProxyEvent::ResponseFailed(f) => Some(f),
                _ => None,
            })
            .expect("ResponseFailed emitted for upstream 500");
        assert_eq!(failed.http_status, 502);
    }

    fn state_with_openai_chat_client_anthropic_upstream(mock_endpoint: &str) -> AppState {
        let provider = ProviderConfig {
            name: "mock-provider".to_owned(),
            api_key: secrecy::SecretString::from("test-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "messages".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "anthropic_messages".to_owned(),
                        endpoint: mock_endpoint.to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: llm_proxy_core::ProviderRoutesConfig {
                messages: None,
                chat_completions: Some("messages".to_owned()),
                ..Default::default()
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
        let app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: Duration::from_secs(300),
                shutdown_timeout: Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "test-proxy".to_owned(),
                rate_limit_rpm: 100,
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
        };
        AppState::new(
            app_config,
            registry,
            ProviderAdapterRegistry::builtin(),
            ProxyClient::new(),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    async fn spawn_mock_anthropic_leading_error_stream() -> String {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![Event::default()
                .event("error")
                .data(r#"{"type":"error","error":{"type":"overloaded_error","message":"Upstream overloaded"}}"#)];
            let stream =
                futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
            let sse = Sse::new(stream).keep_alive(KeepAlive::default());
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                sse.into_response(),
            )
        }),
    );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{addr}/v1/messages")
    }

    fn make_chat_completions_stream_body(model: &str) -> String {
        json!({
            "model": model,
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": true
        })
        .to_string()
    }

    fn chat_completions_stream_request(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap()
    }

    #[tokio::test]
    async fn stream_openai_chat_leading_error_returns_an_http_error_before_commit() {
        let mock_url = spawn_mock_anthropic_leading_error_stream().await;
        let state = state_with_openai_chat_client_anthropic_upstream(&mock_url);
        let app = build_router(state);

        let body = make_chat_completions_stream_body("claude-sonnet-4-6");
        let resp = app
            .oneshot(chat_completions_stream_request(&body))
            .await
            .unwrap();

        // Plan 04 (backlog 2): a 200 whose first frame is an error never
        // commits — the failure happened before any output event, so the
        // caller gets an HTTP error status instead of a 200 stream.
        assert_eq!(
            resp.status(),
            StatusCode::BAD_GATEWAY,
            "a leading error frame is a pre-commit failure"
        );

        let ct = resp
            .headers()
            .get("content-type")
            .expect("content-type header")
            .to_str()
            .unwrap();
        assert!(
            ct.contains("application/json"),
            "expected a JSON error body, got: {ct}"
        );

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("\"error\""),
            "the error body must carry an OpenAI error object; got: {text}"
        );
        assert!(
            !text.contains("Upstream overloaded"),
            "the upstream's own error schema must not reach the client: {text}"
        );
        assert!(
            !text.contains("empty stream"),
            "must not surface the misleading empty-stream 502 body; got: {text}"
        );
    }

    #[tokio::test]
    async fn transport_error_after_terminal_keeps_finish_chunk_openai_client() {
        let mock_url = spawn_mock_anthropic_terminal_then_reset_stream().await;
        let state = state_with_openai_chat_client_anthropic_upstream(&mock_url);
        let app = build_router(state);

        let body = make_chat_completions_stream_body("claude-sonnet-4-6");
        let resp = app
            .oneshot(chat_completions_stream_request(&body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("partial"),
            "content before the terminal must have been delivered: {text}"
        );
        let finish_pos = text.find("\"finish_reason\":\"stop\"").unwrap_or_else(|| {
            panic!("the buffered finish chunk must survive the transport error: {text}")
        });
        let err_pos = text
            .find("\"error\"")
            .unwrap_or_else(|| panic!("the transport error must reach the client in-band: {text}"));
        assert!(
            finish_pos < err_pos,
            "the buffered finish chunk must precede the error payload: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "stream must still terminate with [DONE]: {text}"
        );
    }

    fn authorized_messages_request(token: &str, body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::from(body.to_owned()))
            .unwrap()
    }

    /// Swapping the authenticator on the running state is what makes a
    /// managed key issued or revoked through the admin API take effect:
    /// the old token dies with the old authenticator (its digest cache
    /// included), the new one starts working — no restart, no rebuild.
    #[tokio::test]
    async fn a_swapped_client_auth_takes_effect_on_the_running_router() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url)
            .with_client_auth(Arc::new(client_auth_for("laptop", "first-token")));
        let app = build_router(state.clone());
        let body = make_messages_body("claude-sonnet-4-6", false);

        let first = app
            .clone()
            .oneshot(authorized_messages_request("first-token", &body))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);

        state.swap_client_auth(Arc::new(client_auth_for("laptop", "second-token")));

        let old = app
            .clone()
            .oneshot(authorized_messages_request("first-token", &body))
            .await
            .unwrap();
        assert_eq!(
            old.status(),
            StatusCode::UNAUTHORIZED,
            "the old token must stop authenticating after the swap"
        );
        let new = app
            .oneshot(authorized_messages_request("second-token", &body))
            .await
            .unwrap();
        assert_eq!(new.status(), StatusCode::OK);
    }

    fn admission_for(
        key_id: &str,
        allow_models: Vec<String>,
        rpd: Option<u64>,
        backend: Arc<dyn llm_proxy_engine::AdmissionBackend>,
        registry: &ProviderRegistry,
    ) -> Arc<llm_proxy_engine::PolicyAdmission> {
        let spec = if allow_models.is_empty() && rpd.is_none() {
            llm_proxy_core::ClientKeySpec::Verifier(secrecy::SecretString::from("unused"))
        } else {
            llm_proxy_core::ClientKeySpec::Policy {
                verifier: secrecy::SecretString::from("unused"),
                allow_models,
                allow_providers: Vec::new(),
                limits: Some(llm_proxy_core::KeyWindowLimits {
                    rpm: None,
                    tpm: None,
                    rpd,
                }),
                budget: None,
            }
        };
        let mut clients = HashMap::new();
        clients.insert(key_id.to_owned(), spec);
        let auth = llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
        };
        Arc::new(llm_proxy_engine::PolicyAdmission::over_config_backend(
            &auth, registry, backend,
        ))
    }

    /// Swapping the admission rules on the running engine is what makes a
    /// managed key's policy take effect the moment the daemon loads it:
    /// the engine keeps serving on the same instance (no rebuild — the
    /// caches and cooldowns stand) while the rules change under it.
    #[tokio::test]
    async fn swapped_admission_rules_take_effect_without_rebuilding_the_engine() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let registry = state.providers();
        let pair = Arc::new(llm_proxy_server::SwappingAuthPair::new(
            Arc::new(client_auth_for("laptop", "sekrit-token")),
            admission_for(
                "laptop",
                Vec::new(),
                None,
                Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new()),
                &registry,
            ),
        ));
        let state = state.with_auth_pair(Arc::clone(&pair));
        let app = build_router(state.clone());
        let body = make_messages_body("claude-sonnet-4-6", false);

        let allowed = app
            .clone()
            .oneshot(authorized_messages_request("sekrit-token", &body))
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);

        let engine_before = state.engine().unwrap();
        pair.replace_admission(admission_for(
            "laptop",
            vec!["other-*".to_owned()],
            None,
            Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new()),
            &registry,
        ));
        let engine_after = state.engine().unwrap();
        assert!(
            Arc::ptr_eq(&engine_before, &engine_after),
            "a rules swap must not rebuild the engine"
        );

        let refused = app
            .oneshot(authorized_messages_request("sekrit-token", &body))
            .await
            .unwrap();
        assert_eq!(
            refused.status(),
            StatusCode::FORBIDDEN,
            "the swapped-in allowlist must refuse the model the running engine serves"
        );
    }

    /// A rules swap that tightens a key's allowlist also cuts the key off
    /// from the answers the running engine already stored for it: the
    /// allowlist verdict runs ahead of the response cache, so the request
    /// that was a cache hit a moment ago is refused, not served.
    #[tokio::test]
    async fn a_tightened_allowlist_refuses_a_request_the_cache_was_serving() {
        use llm_proxy_storage::{ProxyEvent, RecordingBus};

        let mock_url = spawn_mock_anthropic_non_stream().await;
        let bus = Arc::new(RecordingBus::new());
        let state = state_with_provider_edited(
            &mock_url,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
            HashMap::new(),
            |config| {
                config.server.cache = llm_proxy_core::ResponseCacheConfig {
                    enabled: true,
                    ttl: Duration::from_secs(60),
                    max_entries: 16,
                };
            },
        )
        .with_event_bus(Arc::<RecordingBus>::clone(&bus));
        let registry = state.providers();
        let backend: Arc<dyn llm_proxy_engine::AdmissionBackend> =
            Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new());
        let pair = Arc::new(llm_proxy_server::SwappingAuthPair::new(
            Arc::new(client_auth_for("laptop", "sekrit-token")),
            admission_for("laptop", Vec::new(), None, Arc::clone(&backend), &registry),
        ));
        let state = state.with_auth_pair(Arc::clone(&pair));
        let app = build_router(state);
        let body = make_messages_body("claude-sonnet-4-6", false);
        let cache_hits = || {
            bus.snapshot()
                .into_iter()
                .filter(
                    |event| matches!(event, ProxyEvent::ResponseCompleted(row) if row.cache_hit),
                )
                .count()
        };

        // Fill the cache, then show the stored answer is what gets served.
        for _ in 0..2 {
            let served = app
                .clone()
                .oneshot(authorized_messages_request("sekrit-token", &body))
                .await
                .unwrap();
            assert_eq!(served.status(), StatusCode::OK);
        }
        assert_eq!(
            cache_hits(),
            1,
            "the second identical request is served from the cache"
        );

        pair.replace_admission(admission_for(
            "laptop",
            vec!["other-*".to_owned()],
            None,
            backend,
            &registry,
        ));

        let refused = app
            .oneshot(authorized_messages_request("sekrit-token", &body))
            .await
            .unwrap();
        assert_eq!(
            refused.status(),
            StatusCode::FORBIDDEN,
            "the tightened allowlist must beat the answer cached before the swap"
        );
        assert_eq!(
            cache_hits(),
            1,
            "the refused request is not recorded as a served copy"
        );
    }

    /// The counters a key already charged survive a rules swap (issuing
    /// another managed key must not hand anyone a fresh window): both the
    /// old and the new rules run over the same backend.
    #[tokio::test]
    async fn an_admission_swap_keeps_the_counters_already_charged() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let registry = state.providers();
        let backend: Arc<dyn llm_proxy_engine::AdmissionBackend> =
            Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new());
        let pair = Arc::new(llm_proxy_server::SwappingAuthPair::new(
            Arc::new(client_auth_for("laptop", "sekrit-token")),
            admission_for(
                "laptop",
                Vec::new(),
                Some(1),
                Arc::clone(&backend),
                &registry,
            ),
        ));
        let state = state.with_auth_pair(Arc::clone(&pair));
        let app = build_router(state);
        let body = make_messages_body("claude-sonnet-4-6", false);

        let first = app
            .clone()
            .oneshot(authorized_messages_request("sekrit-token", &body))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK, "rpd of 1 admits one");

        pair.replace_admission(admission_for(
            "laptop",
            Vec::new(),
            Some(1),
            backend,
            &registry,
        ));

        let second = app
            .oneshot(authorized_messages_request("sekrit-token", &body))
            .await
            .unwrap();
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the day's request count survived the swap; a reset backend would have answered 200"
        );
    }

    /// The single-swap invariant, pinned through the running router: after
    /// every whole-pair swap there is no observable state where the router
    /// authenticates against the old key set while admission enforces the
    /// new rules, nor the mirror. A key from the OLD pair must already 401
    /// (the new authenticator forgot it) and never slip through the new
    /// rules; a key from the NEW pair must hit the NEW rules (403 here),
    /// never still be riding the old permissive ones (200). The two
    /// adjacent swaps this design replaced could answer exactly those two
    /// wrong codes in the window between them.
    #[tokio::test]
    async fn a_pair_swap_never_serves_the_old_authenticator_with_the_new_rules() {
        let mock_url = spawn_mock_anthropic_non_stream().await;
        let state = state_with_anthropic_provider(&mock_url);
        let registry = state.providers();
        let backend: Arc<dyn llm_proxy_engine::AdmissionBackend> =
            Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new());
        let permissive =
            || admission_for("laptop", Vec::new(), None, Arc::clone(&backend), &registry);
        let restrictive = || {
            admission_for(
                "laptop",
                vec!["other-*".to_owned()],
                None,
                Arc::clone(&backend),
                &registry,
            )
        };
        let pair = Arc::new(llm_proxy_server::SwappingAuthPair::new(
            Arc::new(client_auth_for("laptop", "one-token")),
            permissive(),
        ));
        let state = state.with_auth_pair(Arc::clone(&pair));
        let app = build_router(state);
        let body = make_messages_body("claude-sonnet-4-6", false);

        // Sanity: generation one serves one-token under permissive rules.
        let served = app
            .clone()
            .oneshot(authorized_messages_request("one-token", &body))
            .await
            .unwrap();
        assert_eq!(served.status(), StatusCode::OK);

        for swap in 0..4 {
            let (token, rules, expected_new_key) = if swap % 2 == 0 {
                ("two-token", restrictive(), StatusCode::FORBIDDEN)
            } else {
                ("one-token", permissive(), StatusCode::OK)
            };
            pair.replace(Arc::new(client_auth_for("laptop", token)), rules);
            let other_token = if swap % 2 == 0 {
                "one-token"
            } else {
                "two-token"
            };

            // The previous generation's token is already unknown to the
            // new authenticator — 401, never authenticated-under-old and
            // judged-under-new.
            let stale = app
                .clone()
                .oneshot(authorized_messages_request(other_token, &body))
                .await
                .unwrap();
            assert_eq!(
                stale.status(),
                StatusCode::UNAUTHORIZED,
                "the swapped-out token must die in the same instant the rules flip"
            );

            // The installed token meets the INSTALLED rules: restrictive
            // refuses the model (403), permissive serves (200) — a 403
            // under permissive or a 200 under restrictive would mean the
            // halves were one swap apart.
            let fresh = app
                .clone()
                .oneshot(authorized_messages_request(token, &body))
                .await
                .unwrap();
            assert_eq!(
                fresh.status(),
                expected_new_key,
                "the freshly installed token must be judged by the freshly installed rules"
            );
        }
    }
}
