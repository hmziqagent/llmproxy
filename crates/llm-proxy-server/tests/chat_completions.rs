//! Integration tests for the `/v1/chat/completions` route: mock upstreams
//! driven through the router to verify the OpenAI decode -> core pipeline
//! -> OpenAI encode path end-to-end.

#[cfg(test)]
mod tests {

    use std::collections::HashMap;
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

    fn openai_chat_tool_call_response() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "id": "chatcmpl-tool123",
            "object": "chat.completion",
            "created": 12345,
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"SF\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25
            }
        }))
        .unwrap()
    }

    fn state_with_provider(
        mock_endpoint: &str,
        protocol: &str,
        adapter_name: &str,
        _model_name: &str,
    ) -> AppState {
        state_with_provider_timeout(
            mock_endpoint,
            protocol,
            adapter_name,
            _model_name,
            Duration::from_secs(300),
        )
    }

    fn state_with_provider_timeout(
        mock_endpoint: &str,
        protocol: &str,
        adapter_name: &str,
        _model_name: &str,
        request_timeout: Duration,
    ) -> AppState {
        let routes = if protocol == "anthropic_messages" {
            llm_proxy_core::ProviderRoutesConfig {
                messages: Some(adapter_name.to_owned()),
                chat_completions: Some(adapter_name.to_owned()),
                ..Default::default()
            }
        } else {
            llm_proxy_core::ProviderRoutesConfig {
                chat_completions: Some(adapter_name.to_owned()),
                messages: None,
                ..Default::default()
            }
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
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };

        let registry = ProviderRegistry::from_providers(vec![provider]).expect("registry");

        let app_config = AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout,
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
            ProxyClient::try_new().expect("client"),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    fn state_with_openai_chat_provider(mock_endpoint: &str) -> AppState {
        state_with_provider(mock_endpoint, "openai_chat_completions", "chat", "gpt-4o")
    }

    fn state_with_anthropic_provider(mock_endpoint: &str) -> AppState {
        state_with_provider(
            mock_endpoint,
            "anthropic_messages",
            "messages",
            "claude-sonnet-4-6",
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
        format!(
            "http://{}/providers/mock-provider/v1/chat/completions",
            addr
        )
    }

    async fn spawn_mock_openai_chat_non_stream() -> String {
        spawn_mock_server(openai_chat_success_response(), "application/json").await
    }

    async fn spawn_mock_openai_chat_tool_call() -> String {
        spawn_mock_server(openai_chat_tool_call_response(), "application/json").await
    }

    async fn spawn_hanging_mock() -> String {
        let app = Router::new().route(
            "/{*path}",
            post(|| async move {
                tokio::time::sleep(Duration::from_secs(30)).await;
                StatusCode::OK
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!("http://{addr}/providers/mock-provider/v1/chat/completions")
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
        format!(
            "http://{}/providers/mock-provider/v1/chat/completions",
            addr
        )
    }

    async fn spawn_mock_500() -> String {
        let app = Router::new().route(
            "/{*path}",
            post(|| async {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"error":{"message":"internal error","type":"server_error","code":null}}"#
                        .as_bytes()
                        .to_vec(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        spawn_mock_serve(listener, app);
        wait_for_ready(addr).await;
        format!(
            "http://{}/providers/mock-provider/v1/chat/completions",
            addr
        )
    }

    async fn spawn_mock_with_body_capture(
        response_body: Vec<u8>,
    ) -> (String, std::sync::Arc<tokio::sync::Mutex<Option<Vec<u8>>>>) {
        let captured: std::sync::Arc<tokio::sync::Mutex<Option<Vec<u8>>>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let captured_clone = std::sync::Arc::clone(&captured);
        let app = Router::new().route(
            "/{*path}",
            post(move |body: axum::body::Bytes| {
                let captured = std::sync::Arc::clone(&captured_clone);
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
            format!(
                "http://{}/providers/mock-provider/v1/chat/completions",
                addr
            ),
            captured,
        )
    }

    fn empty_state() -> AppState {
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
        let registry = ProviderRegistry::from_providers(vec![]).expect("empty registry");
        AppState::new(
            app_config,
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

    fn state_with_mock_provider_no_upstream() -> AppState {
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
                    "openai-chat".to_owned(),
                    ProviderAdapterConfig {
                        protocol: "openai_chat".to_owned(),
                        endpoint: "http://127.0.0.1:1/unreachable".to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                );
                m
            },
            routes: llm_proxy_core::ProviderRoutesConfig {
                chat_completions: Some("openai-chat".to_owned()),
                messages: None,
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
            ProxyClient::try_new().expect("client"),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        )
    }

    fn chat_request(body: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap()
    }

    fn make_chat_body(model: &str, stream: bool) -> String {
        json!({
            "model": model,
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": stream
        })
        .to_string()
    }

    #[tokio::test]
    async fn route_is_mounted_no_longer_404() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let app = build_router(state_with_openai_chat_provider(&mock_url));
        let body = make_chat_body("gpt-4o", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "route must be mounted (not 404)"
        );
    }

    #[tokio::test]
    async fn non_streaming_text_request_returns_openai_shaped_response() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert_eq!(
            resp_body["object"], "chat.completion",
            "object must be chat.completion"
        );
        assert!(resp_body["id"].is_string(), "id must be a string");
        assert!(resp_body["model"].is_string(), "model must be a string");
        assert!(resp_body["created"].is_number(), "created must be a number");

        let choices = resp_body["choices"]
            .as_array()
            .expect("choices must be array");
        assert!(!choices.is_empty(), "must have at least one choice");
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[0]["finish_reason"], "stop");

        let message = &choices[0]["message"];
        assert_eq!(message["role"], "assistant");
        assert!(
            message["content"]
                .as_str()
                .unwrap()
                .contains("Hello from OpenAI mock!"),
            "response should contain translated text"
        );

        let usage = &resp_body["usage"];
        assert_eq!(usage["prompt_tokens"], 10);
        assert_eq!(usage["completion_tokens"], 5);
        assert_eq!(usage["total_tokens"], 15);
    }

    #[tokio::test]
    async fn non_streaming_response_shape_detailed() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert_eq!(resp_body["object"], "chat.completion");
        assert!(
            resp_body["id"].as_str().unwrap().starts_with("chatcmpl-"),
            "id must start with chatcmpl-"
        );
        // Same-format passthrough: the upstream response body reaches the
        // client verbatim, so it advertises the upstream model name.
        assert_eq!(resp_body["model"], "gpt-4o-2024-08-06");
        assert_eq!(resp_body["choices"][0]["finish_reason"], "stop");
        assert!(resp_body["choices"][0]["message"]["content"].is_string());
        assert!(resp_body["usage"]["prompt_tokens"].is_number());
        assert!(resp_body["usage"]["completion_tokens"].is_number());
        assert!(resp_body["usage"]["total_tokens"].is_number());
    }

    #[tokio::test]
    async fn tool_call_response_returns_tool_calls() {
        let mock_url = spawn_mock_openai_chat_tool_call().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "what is the weather in SF?" }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            }]
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert_eq!(resp_body["choices"][0]["finish_reason"], "tool_calls");
        let tool_calls = resp_body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("tool_calls must be array");
        assert!(!tool_calls.is_empty());
        assert_eq!(tool_calls[0]["id"], "call_abc123");
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
    }

    #[tokio::test]
    async fn route_preserves_fields_through_core() {
        let (mock_url, captured_body) =
            spawn_mock_with_body_capture(openai_chat_success_response()).await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are helpful",
                    "cache_control": { "type": "ephemeral" }
                },
                { "role": "user", "content": "hello" }
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": false,
            "stream_options": { "include_usage": true },
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
                }
            }],
            "tool_choice": "auto",
            "reasoning_effort": "high",
            "thinking": { "type": "enabled", "budget_tokens": 10000 },
            "user": "test-user-123"
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        tokio::time::sleep(Duration::from_millis(100)).await;
        let guard = captured_body.lock().await;
        let upstream_body = guard
            .as_ref()
            .expect("upstream should have received a request body");
        let upstream_json: Value =
            serde_json::from_slice(upstream_body).expect("upstream body should be valid JSON");

        // Same-format passthrough (plan 02): the upstream body is the client's
        // raw body with only the proxy's surgical edits — the field names the
        // client used survive untouched.
        assert_eq!(upstream_json["model"], "gpt-4o");
        assert_eq!(upstream_json["max_tokens"], 256);
        assert!(
            upstream_json.get("max_completion_tokens").is_none(),
            "same-format passthrough must not rename client fields"
        );
        assert_eq!(upstream_json["temperature"], 0.7);
        assert_eq!(upstream_json["top_p"], 0.9);
        assert!(upstream_json["tools"].is_array(), "tools must be preserved");
        assert_eq!(upstream_json["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(upstream_json["tool_choice"], "auto");
        assert_eq!(upstream_json["user"], "test-user-123");
        // BUG-20: an Anthropic-style `thinking` object never reaches a chat
        // API; the explicit reasoning_effort wins.
        assert!(
            upstream_json.get("thinking").is_none(),
            "thinking must be translated, not forwarded: {upstream_json}"
        );
        assert_eq!(upstream_json["reasoning_effort"], "high");
    }

    #[tokio::test]
    async fn unknown_model_returns_openai_shaped_400() {
        let app = build_router(empty_state());
        let body = make_chat_body("nonexistent-model", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert_eq!(resp_body["error"]["type"], "invalid_request_error");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");

        assert!(
            !resp_body.as_object().unwrap().contains_key("type"),
            "must not have Anthropic type field"
        );
    }

    #[tokio::test]
    async fn invalid_json_returns_openai_shaped_400() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from("this is not json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert_eq!(resp_body["error"]["type"], "invalid_request_error");
        assert!(
            resp_body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid JSON"),
            "error message should mention invalid JSON"
        );
        assert!(resp_body["error"]["code"].is_null(), "code must be null");

        assert!(
            !resp_body.as_object().unwrap().contains_key("type"),
            "must not have Anthropic type field"
        );
    }

    #[tokio::test]
    async fn upstream_failure_returns_openai_shaped_502() {
        let mock_url = spawn_mock_500().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert_eq!(resp_body["error"]["type"], "api_error");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");
    }

    #[tokio::test]
    async fn provider_decode_failure_returns_openai_shaped_502() {
        let malformed = serde_json::to_vec(&json!({"not": "a valid response"})).unwrap();
        let mock_url = spawn_mock_server(malformed, "application/json").await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert_eq!(resp_body["error"]["type"], "api_error");
        assert!(resp_body["error"]["code"].is_null(), "code must be null");
    }

    #[tokio::test]
    async fn streaming_response_emits_chat_completion_chunk() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", true);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("chat.completion.chunk"),
            "must emit chat.completion.chunk objects"
        );
    }

    #[tokio::test]
    async fn streaming_response_sets_sse_content_type() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", true);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
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
    async fn stream_true_uses_shared_streaming_pipeline() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", true);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let request_id = resp.headers().get("x-request-id");
        assert!(
            request_id.is_some(),
            "x-request-id must be present on stream response"
        );

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("Hi from OpenAI!"),
            "expected translated text delta in SSE output, got: {text}"
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one end-to-end streaming scenario: the mock feed, the SSE \
                 consumption, and the start/delta/stop assertions read as a \
                 single narrative"
    )]
    #[tokio::test]
    async fn streaming_tool_call_maps_to_delta_tool_calls() {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_mock","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_start")
                    .data(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"get_weather"}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}"#),
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
        let mock_url = format!("http://{}/providers/mock-provider/v1/messages", addr);

        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "weather in SF?" }],
            "stream": true
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        let chunks: Vec<Value> = text
            .lines()
            .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
            .filter_map(|l| serde_json::from_str(l.trim_start_matches("data: ")).ok())
            .collect();

        let start_chunk = chunks
            .iter()
            .find(|c| {
                c.get("choices")
                    .and_then(|ch| ch.get(0))
                    .and_then(|ch| ch.get("delta"))
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|tc| tc.get(0))
                    .and_then(|tc| tc.get("id"))
                    .is_some()
            })
            .expect("should have a tool_call start chunk with an id");

        let tc_start = &start_chunk["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(
            tc_start["id"].as_str(),
            Some("toolu_123"),
            "tool_call start id must be toolu_123"
        );
        assert_eq!(
            tc_start["function"]["name"].as_str(),
            Some("get_weather"),
            "tool_call start function.name must be get_weather"
        );
        assert_eq!(
            tc_start["index"].as_i64(),
            Some(0),
            "tool_call index must be 0"
        );

        let delta_chunk = chunks
            .iter()
            .find(|c| {
                c.get("choices")
                    .and_then(|ch| ch.get(0))
                    .and_then(|ch| ch.get("delta"))
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|tc| tc.get(0))
                    .and_then(|tc| tc.get("function"))
                    .and_then(|f| f.get("arguments"))
                    .is_some()
            })
            .expect("should have a tool_call delta chunk with arguments");

        let tc_delta = &delta_chunk["choices"][0]["delta"]["tool_calls"][0];
        let args = tc_delta["function"]["arguments"].as_str().unwrap_or("");
        assert!(
            !args.is_empty() || tc_delta["function"].get("arguments").is_some(),
            "tool_call delta must have a function.arguments field (may be empty string for start chunk)"
        );
    }

    #[tokio::test]
    async fn streaming_usage_maps_when_requested() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": true,
            "stream_options": { "include_usage": true }
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(text.contains("[DONE]"), "stream must end with [DONE]");

        assert!(
            text.contains("\"usage\""),
            "stream with include_usage=true should contain a usage key, got: {text}"
        );

        let chunks: Vec<Value> = text
            .lines()
            .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
            .filter_map(|l| serde_json::from_str(l.trim_start_matches("data: ")).ok())
            .collect();

        let usage_chunk = chunks.iter().find(|c| c.get("usage").is_some());
        assert!(
            usage_chunk.is_some(),
            "at least one chunk must contain a 'usage' field when include_usage is true"
        );
        let usage = usage_chunk.unwrap()["usage"].as_object().unwrap();
        assert!(
            usage.contains_key("prompt_tokens"),
            "usage must have prompt_tokens"
        );
        assert!(
            usage.contains_key("completion_tokens"),
            "usage must have completion_tokens"
        );
    }

    #[tokio::test]
    async fn streaming_stop_reason_maps_to_finish_reason() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", true);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("stop"),
            "stream must contain finish_reason: stop"
        );
    }

    #[tokio::test]
    async fn stream_error_before_first_byte_returns_http_502() {
        let mock_url = spawn_mock_500().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", true);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_GATEWAY,
            "stream with upstream 500 should return 502"
        );

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["code"].is_null(), "code must be null");
    }

    #[tokio::test]
    async fn streaming_response_ends_with_done() {
        let mock_url = spawn_mock_openai_chat_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = make_chat_body("gpt-4o", true);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("data: [DONE]"),
            "stream must end with data: [DONE], got: {text}"
        );
    }

    #[tokio::test]
    async fn no_anthropic_error_envelope_on_openai_route() {
        let app = build_router(empty_state());
        let body = make_chat_body("nonexistent-model", false);
        let resp = app.oneshot(chat_request(&body)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(
            !resp_body.as_object().unwrap().contains_key("type"),
            "must not have Anthropic 'type' field at top level"
        );
        assert!(
            !resp_body["error"]["type"]
                .as_str()
                .unwrap_or("")
                .contains("not_found_error"),
            "must not use Anthropic error types"
        );
        assert!(resp_body["error"].is_object());
        assert_eq!(resp_body["error"]["type"], "invalid_request_error");
        assert!(resp_body["error"]["code"].is_null());
    }

    #[test]
    fn source_guard_chat_rs_architecture_boundaries() {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let chat_path = std::path::Path::new(&manifest_dir)
            .join("src")
            .join("routes")
            .join("chat.rs");
        let source =
            std::fs::read_to_string(&chat_path).expect("failed to read chat.rs for source guard");
        let prod = source
            .split_once("#[cfg(test)]")
            .map(|(p, _)| p)
            .unwrap_or(&source);

        let forbidden = [
            "llm_proxy_core::router",
            "detect_scenario",
            "route_for_streaming",
            "classify_endpoint",
            "EndpointType",
            "OpenCodeClient",
            "transformer",
            "StreamProxy",
            "spawn_proxy_task",
            "handle_anthropic_streaming",
            "handle_openai_streaming",
            "handle_responses_streaming",
            "handle_gemini_streaming",
            "ApiError",
            "ScenarioConfig",
            "axum_serde",
            "Sonic<",
        ];

        for pattern in &forbidden {
            assert!(
                !prod.contains(pattern),
                "routes/chat.rs production code must not contain '{pattern}'"
            );
        }
    }

    #[tokio::test]
    async fn not_found_openai_path_returns_openai_shaped_error() {
        let app = build_router(empty_state());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions/extra")
            .header("content-type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");

        assert!(
            resp_body.get("type").is_none() || resp_body["type"].is_null(),
            "must not have Anthropic 'type' field"
        );
    }

    #[tokio::test]
    async fn not_found_anthropic_path_returns_anthropic_shaped_error() {
        let app = build_router(empty_state());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/messages/typo")
            .header("content-type", "application/json")
            .body(Body::empty())
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
        assert!(resp_body["error"]["type"].is_string());
    }

    #[tokio::test]
    async fn empty_body_returns_openai_shaped_400() {
        let app = build_router(state_with_mock_provider_no_upstream());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");
    }

    #[tokio::test]
    async fn wrong_field_types_returns_openai_shaped_400() {
        let app = build_router(state_with_mock_provider_no_upstream());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model": 123, "messages": "hello", "stream": "yes"}"#.to_owned(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");
    }

    #[tokio::test]
    async fn missing_model_returns_openai_shaped_400() {
        let app = build_router(state_with_mock_provider_no_upstream());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages": [{"role": "user", "content": "hello"}]}"#.to_owned(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["code"].is_null(), "code must be null");

        let msg = resp_body["error"]["message"].as_str().unwrap();
        assert!(
            msg.to_lowercase().contains("model"),
            "error should mention model, got: {msg}"
        );
    }

    #[tokio::test]
    async fn missing_messages_returns_openai_shaped_400() {
        let app = build_router(state_with_mock_provider_no_upstream());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "gpt-4o"}"#.to_owned()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["code"].is_null(), "code must be null");

        let msg = resp_body["error"]["message"].as_str().unwrap();
        assert!(
            msg.to_lowercase().contains("messages"),
            "error should mention messages, got: {msg}"
        );
    }

    #[tokio::test]
    async fn invalid_utf8_body_returns_openai_shaped_400() {
        let app = build_router(state_with_mock_provider_no_upstream());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(vec![0xff, 0xfe, 0x00, 0x01]))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");
    }

    #[tokio::test]
    async fn max_tokens_zero_is_accepted() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 0,
            "stream": false
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "max_tokens=0 should not cause an internal server error"
        );
    }

    #[tokio::test]
    async fn chat_completions_oversized_body_returns_payload_too_large() {
        let app = build_router(empty_state());
        let oversized_body = "X".repeat(33 * 1024 * 1024);
        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(oversized_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );
        assert!(resp.headers().get("x-request-id").is_some());
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("too large")
        );
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn timeout_returns_normalised_json_408() {
        let mock_url = spawn_hanging_mock().await;
        let state = state_with_provider_timeout(
            &mock_url,
            "openai_chat_completions",
            "chat",
            "gpt-4o",
            Duration::from_millis(50),
        );
        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/providers/mock-provider/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "application/json"
        );
        assert!(
            resp.headers().get("x-request-id").is_some(),
            "408 must carry x-request-id after normalisation"
        );
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "timeout_error");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("timed out")
        );
        assert!(json["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn malformed_frame_is_skipped_and_stream_completes() {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"this is not valid JSON for the provider decoder"#),
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
        let mock_url = format!("http://{}/providers/mock-provider/v1/messages", addr);

        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": true
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("[DONE]"),
            "stream must end with [DONE] even when a frame is skipped, got: {text}"
        );
    }

    #[tokio::test]
    async fn in_band_stream_error_emits_openai_error_chunk() {
        use bytes::Bytes;

        let first_frame = Bytes::from(
        "event: message_start\n\
         data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_err\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n"
            .to_owned(),
    );
        let bad_frame = Bytes::from(vec![b'd', b'a', b't', b'a', b':', b' ', 0xFF, b'\n', b'\n']);

        let app = Router::new().route(
            "/{*path}",
            post(move || {
                let first = first_frame.clone();
                let bad = bad_frame.clone();
                async move {
                    let body_stream = futures::stream::iter(vec![
                        Ok::<Bytes, std::convert::Infallible>(first),
                        Ok::<Bytes, std::convert::Infallible>(bad),
                    ]);
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(body_stream).into_response(),
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

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": true
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("\"type\":\"server_error\""),
            "in-band error must emit an OpenAI error chunk, got: {text}"
        );
        assert!(
            text.contains("[DONE]"),
            "stream must end with [DONE] after the in-band error, got: {text}"
        );
    }

    #[tokio::test]
    async fn upstream_disconnect_completes_with_synthetic_terminal() {
        let app = Router::new().route(
        "/{*path}",
        post(|| async move {
            let events = vec![
                Event::default()
                    .event("message_start")
                    .data(r#"{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#),
                Event::default()
                    .event("content_block_delta")
                    .data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#),
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
        let mock_url = format!("http://{}/providers/mock-provider/v1/messages", addr);

        let state = state_with_anthropic_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": true
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        assert!(
            text.contains("[DONE]"),
            "stream must end with [DONE] after upstream disconnect, got: {text}"
        );
        assert!(
            text.contains("stop"),
            "stream should contain synthetic finish_reason: stop"
        );
    }

    #[tokio::test]
    async fn system_only_messages_does_not_panic() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let body = json!({
            "model": "gpt-4o",
            "messages": [
                { "role": "system", "content": "You are a helpful assistant." }
            ]
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "system-only messages should not cause an internal server error"
        );
    }

    #[tokio::test]
    async fn unknown_model_empty_routing_table_openai_shaped_400() {
        let app = build_router(empty_state());

        let body = json!({
            "model": "nonexistent-model",
            "messages": [{ "role": "user", "content": "hello" }]
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "unknown provider should return 404"
        );

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["message"].is_string());
        assert!(resp_body["error"]["code"].is_null(), "code must be null");

        assert!(
            resp_body.get("type").is_none() || resp_body["type"].is_null(),
            "must not have Anthropic 'type' field"
        );
    }

    #[tokio::test]
    async fn large_messages_array_is_handled() {
        let mock_url = spawn_mock_openai_chat_non_stream().await;
        let state = state_with_openai_chat_provider(&mock_url);
        let app = build_router(state);

        let mut messages = Vec::new();
        for i in 0..1000 {
            messages.push(json!({
                "role": if i % 2 == 0 { "user" } else { "assistant" },
                "content": format!("message {i}")
            }));
        }
        let body = json!({
            "model": "gpt-4o",
            "messages": messages
        });
        let resp = app.oneshot(chat_request(&body.to_string())).await.unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "large messages array should not cause an internal server error"
        );
    }

    #[tokio::test]
    async fn bare_v1_chat_completions_returns_404_openai_shaped() {
        let app = build_router(empty_state());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"model": "gpt-4o", "messages": []}).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(resp_body["error"].is_object(), "must have error object");
        assert!(resp_body["error"]["message"].is_string());
    }

    #[tokio::test]
    async fn bare_v1_messages_returns_404_anthropic_shaped() {
        let app = build_router(empty_state());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"model": "claude-3", "messages": []}).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let resp_body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap(),
        )
        .unwrap();

        assert!(
            resp_body["type"].is_string(),
            "must have Anthropic 'type' field"
        );
    }

    #[tokio::test]
    async fn invalid_provider_name_returns_400() {
        let app = build_router(empty_state());
        let req = Request::builder()
            .method("POST")
            .uri("/providers/INVALID.NAME/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"model": "gpt-4o", "messages": []}).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "invalid provider name must return 400"
        );
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one end-to-end routing scenario: two provider registries, two \
                 dispatches, and their divergent assertions read as a single \
                 narrative"
    )]
    #[tokio::test]
    async fn same_model_routes_to_different_providers() {
        let mut adapters_a = HashMap::new();
        adapters_a.insert(
            "chat".to_owned(),
            ProviderAdapterConfig {
                protocol: "openai_chat_completions".to_owned(),
                endpoint: "https://provider-a.example.com/v1/chat/completions".to_owned(),
                headers: std::sync::Arc::new(HashMap::new()),
                forward_inbound_headers: None,
                lossy: false,
            },
        );
        let provider_a = ProviderConfig {
            name: "provider-a".to_owned(),
            api_key: secrecy::SecretString::from("key-a"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: adapters_a,
            routes: llm_proxy_core::ProviderRoutesConfig {
                chat_completions: Some("chat".to_owned()),
                messages: None,
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

        let mut adapters_b = HashMap::new();
        adapters_b.insert(
            "chat".to_owned(),
            ProviderAdapterConfig {
                protocol: "openai_chat_completions".to_owned(),
                endpoint: "https://provider-b.example.com/v1/chat/completions".to_owned(),
                headers: std::sync::Arc::new(HashMap::new()),
                forward_inbound_headers: None,
                lossy: false,
            },
        );
        let provider_b = ProviderConfig {
            name: "provider-b".to_owned(),
            api_key: secrecy::SecretString::from("key-b"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: adapters_b,
            routes: llm_proxy_core::ProviderRoutesConfig {
                chat_completions: Some("chat".to_owned()),
                messages: None,
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

        let registry = ProviderRegistry::from_providers(vec![provider_a, provider_b]).unwrap();
        let state = AppState::new(
            AppConfig {
                server: ServerConfig {
                    bind: "127.0.0.1:3456".parse().unwrap(),
                    request_timeout: Duration::from_secs(30),
                    shutdown_timeout: Duration::from_secs(30),
                    log_level: "info".to_owned(),
                    hot_reload: false,
                    allowed_origins: None,
                    server_name: "test".to_owned(),
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
            ProxyClient::try_new().expect("client"),
            BuildInfo {
                name: "test",
                version: "0.0.0",
                target: "test",
                git_sha: "test",
            },
        );
        let app = build_router(state);

        let body = serde_json::json!({"model": "gpt-4o", "messages": []}).to_string();

        let req_a = Request::builder()
            .method("POST")
            .uri("/providers/provider-a/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.clone()))
            .unwrap();
        let resp_a = app.clone().oneshot(req_a).await.unwrap();
        assert_ne!(
            resp_a.status(),
            StatusCode::NOT_FOUND,
            "provider-a should resolve, not 404"
        );

        let req_b = Request::builder()
            .method("POST")
            .uri("/providers/provider-b/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp_b = app.oneshot(req_b).await.unwrap();
        assert_ne!(
            resp_b.status(),
            StatusCode::NOT_FOUND,
            "provider-b should resolve, not 404"
        );
    }
}
