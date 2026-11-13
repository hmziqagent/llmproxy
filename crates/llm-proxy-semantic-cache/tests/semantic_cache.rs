//! The semantic cache end to end (plan 09 step 5): near-duplicate prompts
//! hit, unrelated prompts miss, one caller's response never serves
//! another, entries expire, and through a real engine a paraphrase is
//! answered without a second upstream call.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
    ProviderRouteKind, ProviderRoutesConfig, SemanticCacheConfig, ServerConfig,
};
use llm_proxy_engine::{
    Cache, CacheEntry, CacheKey, ClientDialect, Engine, EngineBuilder, EngineError, EngineRequest,
    InboundRequest,
};
use llm_proxy_protocol::core::{
    CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole, CoreTool,
};
use llm_proxy_semantic_cache::{Embedder, Embedding, NgramEmbedder, SemanticCache};
use llm_proxy_storage::{EventBus, NoopBus};
use tokio::sync::Mutex;

fn config(threshold: f32, ttl: Duration, max_entries: usize) -> SemanticCacheConfig {
    SemanticCacheConfig {
        enabled: true,
        ttl,
        max_entries,
        threshold,
    }
}

fn core_request(text: &str) -> CoreRequest {
    CoreRequest {
        model: llm_proxy_protocol::core::ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        system: Vec::new(),
        messages: vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: text.to_owned(),
                cache: None,
            }],
        }],
        tools: Vec::new(),
        tool_choice: None,
        sampling: Default::default(),
        stream: false,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

fn cache_key<'a>(caller: &'a str, request: &'a CoreRequest) -> CacheKey<'a> {
    CacheKey {
        caller,
        provider: "p",
        route_kind: ProviderRouteKind::Messages,
        model: "m",
        body_hash: "hash",
        streaming: false,
        request,
    }
}

fn response(text: &str) -> CoreResponse {
    CoreResponse {
        id: Some("msg_1".to_owned()),
        model: llm_proxy_protocol::core::ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        content: vec![CoreContent::Text {
            text: text.to_owned(),
            cache: None,
        }],
        stop_reason: llm_proxy_protocol::core::StopReason::EndTurn,
        stop_sequence: None,
        usage: Default::default(),
        provider_meta: Default::default(),
        cost: None,
    }
}

async fn store_one(cache: &SemanticCache, caller: &str, prompt: &str) {
    let request = core_request(prompt);
    let key = cache_key(caller, &request);
    cache
        .store(
            &key,
            &CacheEntry {
                response: response("the cached answer"),
                events: Vec::new(),
            },
        )
        .await;
}

fn cached_entry() -> CacheEntry {
    CacheEntry {
        response: response("the cached answer"),
        events: Vec::new(),
    }
}

/// Requests whose `tools`, `tool_choice`, sampling parameters, provider
/// hints, route kind, message roles or non-text content differ must never
/// share a cached entry, no matter how similar their text is.
#[tokio::test]
async fn identical_text_with_different_sampling_does_not_share_an_entry() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_secs(600), 16),
        Box::new(NgramEmbedder::new()),
    );
    let prompt = "Summarize this log line as JSON.";
    let mut stored = core_request(prompt);
    stored.sampling.temperature = Some(0.0);
    cache
        .store(&cache_key("key", &stored), &cached_entry())
        .await;

    let mut same_sampling = core_request(prompt);
    same_sampling.sampling.temperature = Some(0.0);
    let hit = cache
        .lookup(&cache_key("key", &same_sampling))
        .await
        .expect("lookup");
    assert!(hit.is_some(), "the identical shape hits");
    assert_eq!(cache.hits(), 1);

    let mut other_sampling = core_request(prompt);
    other_sampling.sampling.temperature = Some(1.0);
    let miss = cache
        .lookup(&cache_key("key", &other_sampling))
        .await
        .expect("lookup");
    assert!(
        miss.is_none(),
        "the same text at a different temperature must not be served a 0.0-temperature answer"
    );
    assert_eq!(cache.hits(), 1, "only the identical-shape request counted");
}

#[tokio::test]
async fn identical_text_with_different_tools_does_not_share_an_entry() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_secs(600), 16),
        Box::new(NgramEmbedder::new()),
    );
    let prompt = "What files are in this directory?";
    let mut with_tools = core_request(prompt);
    with_tools.tools = vec![CoreTool {
        name: "list_files".to_owned(),
        description: None,
        input_schema: serde_json::json!({"type": "object", "properties": {}}),
    }];
    cache
        .store(&cache_key("key", &with_tools), &cached_entry())
        .await;

    let mut again_with_tools = core_request(prompt);
    again_with_tools.tools = with_tools.tools.clone();
    let hit = cache
        .lookup(&cache_key("key", &again_with_tools))
        .await
        .expect("lookup");
    assert!(hit.is_some(), "the identical tool-bearing shape hits");
    assert_eq!(cache.hits(), 1);

    let bare = core_request(prompt);
    let miss = cache
        .lookup(&cache_key("key", &bare))
        .await
        .expect("lookup");
    assert!(
        miss.is_none(),
        "a tool-less request must not be served the tool call the cached answer may contain"
    );
    assert_eq!(cache.hits(), 1, "only the identical-shape request counted");
}

#[tokio::test]
async fn a_request_with_an_image_is_never_cached_or_served() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_secs(600), 16),
        Box::new(NgramEmbedder::new()),
    );
    let text = "Describe this image.";
    let mut with_image = core_request(text);
    with_image.messages[0].content.push(CoreContent::Image {
        source: serde_json::json!({"type": "base64", "data": "aGVsbG8="}),
    });
    cache
        .store(&cache_key("key", &with_image), &cached_entry())
        .await;

    let same_image = with_image.clone();
    let miss = cache
        .lookup(&cache_key("key", &same_image))
        .await
        .expect("lookup");
    assert!(
        miss.is_none(),
        "an image request is never cached, so it can never be served"
    );
    assert_eq!(cache.hits(), 0);

    let text_only = core_request(text);
    let miss = cache
        .lookup(&cache_key("key", &text_only))
        .await
        .expect("lookup");
    assert!(
        miss.is_none(),
        "a text-only request with the same words is a miss too: nothing was stored"
    );
    assert_eq!(cache.hits(), 0, "an image-bearing request stored nothing");
}

#[tokio::test]
async fn a_streaming_lookup_and_store_do_nothing() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_secs(600), 16),
        Box::new(NgramEmbedder::new()),
    );
    let request = core_request("a prompt that streamed");
    let streaming_key = CacheKey {
        streaming: true,
        ..cache_key("key", &request)
    };
    cache.store(&streaming_key, &cached_entry()).await;
    let miss = cache
        .lookup(&cache_key("key", &request))
        .await
        .expect("lookup");
    assert!(miss.is_none(), "a streaming store left nothing to serve");

    cache
        .store(&cache_key("key", &request), &cached_entry())
        .await;
    let streaming_miss = cache.lookup(&streaming_key).await.expect("lookup");
    assert!(
        streaming_miss.is_none(),
        "a streaming lookup is never served"
    );
}

#[tokio::test]
async fn a_paraphrase_hits_and_an_unrelated_prompt_misses() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_secs(600), 16),
        Box::new(NgramEmbedder::new()),
    );
    store_one(&cache, "key", "What is the capital of France?").await;

    let paraphrase = core_request("What is France's capital city?");
    let hit = cache
        .lookup(&cache_key("key", &paraphrase))
        .await
        .expect("lookup");
    assert!(
        hit.is_some(),
        "a near-duplicate prompt is served from the cache"
    );
    assert_eq!(cache.hits(), 1);

    let other = core_request("Write a Rust function that sorts a list of integers.");
    let miss = cache
        .lookup(&cache_key("key", &other))
        .await
        .expect("lookup");
    assert!(miss.is_none(), "an unrelated prompt is not served");
    assert_eq!(cache.hits(), 1, "only the hit counted");
}

#[tokio::test]
async fn one_callers_cache_never_serves_another() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_secs(600), 16),
        Box::new(NgramEmbedder::new()),
    );
    store_one(&cache, "caller-a", "What is the capital of France?").await;

    let same_prompt = core_request("What is the capital of France?");
    let result = cache
        .lookup(&cache_key("caller-b", &same_prompt))
        .await
        .expect("lookup");
    assert!(
        result.is_none(),
        "an identical prompt from another caller must miss"
    );
}

#[tokio::test]
async fn entries_expire_after_the_ttl() {
    let cache = SemanticCache::new(
        config(0.6, Duration::from_millis(50), 16),
        Box::new(NgramEmbedder::new()),
    );
    store_one(&cache, "key", "What is the capital of France?").await;
    tokio::time::sleep(Duration::from_millis(120)).await;
    let same_prompt = core_request("What is the capital of France?");
    let result = cache
        .lookup(&cache_key("key", &same_prompt))
        .await
        .expect("lookup");
    assert!(result.is_none(), "an expired entry is not served");
}

#[tokio::test]
async fn a_disabled_cache_stores_and_serves_nothing() {
    let disabled = SemanticCacheConfig {
        enabled: false,
        ..config(0.6, Duration::from_secs(600), 16)
    };
    let cache = SemanticCache::new(disabled, Box::new(NgramEmbedder::new()));
    let request = core_request("anything");
    let key = cache_key("key", &request);
    cache
        .store(
            &key,
            &CacheEntry {
                response: response("never stored"),
                events: Vec::new(),
            },
        )
        .await;
    assert!(
        cache.lookup(&key).await.expect("lookup").is_none(),
        "a disabled semantic cache answers nothing"
    );
}

#[tokio::test]
async fn a_broken_embedder_surfaces_as_an_error_not_a_silent_miss() {
    #[derive(Debug)]
    struct Broken;
    #[async_trait::async_trait]
    impl Embedder for Broken {
        async fn embed(
            &self,
            _texts: &[String],
        ) -> Result<Vec<Embedding>, llm_proxy_semantic_cache::EmbedError> {
            Err(llm_proxy_semantic_cache::EmbedError(
                "embeddings endpoint down".to_owned(),
            ))
        }
    }
    let cache = SemanticCache::new(config(0.6, Duration::from_secs(600), 16), Box::new(Broken));
    let request = core_request("anything");
    match cache.lookup(&cache_key("key", &request)).await {
        Err(EngineError::Internal(message)) => {
            assert!(
                message.contains("embeddings endpoint down"),
                "got {message}"
            );
        }
        other => panic!("expected an internal error, got {other:?}"),
    }
}

/// A bucket holds at most max_entries, evicting the oldest.
#[tokio::test]
async fn buckets_are_bounded() {
    let cache = SemanticCache::new(
        config(0.99, Duration::from_secs(600), 2),
        Box::new(NgramEmbedder::new()),
    );
    // Threshold 0.99 ≈ exact: only the identical prompt can hit.
    for prompt in ["first prompt", "second prompt", "third prompt"] {
        let request = core_request(prompt);
        let key = cache_key("key", &request);
        cache
            .store(
                &key,
                &CacheEntry {
                    response: response(prompt),
                    events: Vec::new(),
                },
            )
            .await;
    }
    let first = core_request("first prompt");
    assert!(
        cache
            .lookup(&cache_key("key", &first))
            .await
            .expect("l")
            .is_none(),
        "the oldest entry was evicted by the cap"
    );
    let third = core_request("third prompt");
    assert!(
        cache
            .lookup(&cache_key("key", &third))
            .await
            .expect("l")
            .is_some(),
        "the newest entries survive"
    );
}

/// The bound is per caller, not per bucket: one caller spreading requests
/// over many model strings cannot exceed max_entries in total
/// (docs/data-class-policy.md: "at most max_entries per caller").
#[tokio::test]
async fn one_callers_total_across_models_is_bounded() {
    let cache = SemanticCache::new(
        config(0.99, Duration::from_secs(600), 2),
        Box::new(NgramEmbedder::new()),
    );
    let prompt = "the same prompt under three different models";
    for model in ["model-a", "model-b", "model-c"] {
        let request = core_request(prompt);
        let key = CacheKey {
            caller: "key",
            provider: "p",
            route_kind: ProviderRouteKind::Messages,
            model,
            body_hash: "hash",
            streaming: false,
            request: &request,
        };
        cache
            .store(
                &key,
                &CacheEntry {
                    response: response(prompt),
                    events: Vec::new(),
                },
            )
            .await;
    }
    // The caller holds at most two of the three; the oldest (model-a) is
    // the one evicted, wherever it sits.
    async fn lookup_in(cache: &SemanticCache, prompt: &str, model: &str) -> Option<CacheEntry> {
        let request = core_request(prompt);
        let key = CacheKey {
            caller: "key",
            provider: "p",
            route_kind: ProviderRouteKind::Messages,
            model,
            body_hash: "hash",
            streaming: false,
            request: &request,
        };
        cache.lookup(&key).await.expect("lookup")
    }
    assert!(
        lookup_in(&cache, prompt, "model-a").await.is_none(),
        "the caller's oldest entry was evicted by the per-caller cap"
    );
    assert!(
        lookup_in(&cache, prompt, "model-b").await.is_some()
            && lookup_in(&cache, prompt, "model-c").await.is_some(),
        "the newest entries across the caller's buckets survive"
    );
}

// ---------------------------------------------------------------------------
// Through a real engine against a loopback upstream.

struct CountingUpstream {
    requests: Mutex<u64>,
}

async fn spawn_upstream() -> (String, Arc<CountingUpstream>) {
    let counter = Arc::new(CountingUpstream {
        requests: Mutex::new(0),
    });
    let sink = Arc::clone(&counter);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move |body: String| {
            let sink = Arc::clone(&sink);
            async move {
                {
                    let mut count = sink.requests.lock().await;
                    *count += 1;
                }
                let reply = format!(
                    "{{\"id\":\"msg_h\",\"type\":\"message\",\"role\":\"assistant\",                     \"content\":[{{\"type\":\"text\",\"text\":\"upstream answer {}\"}}],                     \"model\":\"claude-sonnet-4-6\",\"stop_reason\":\"end_turn\",                     \"stop_sequence\":null,\"usage\":{{\"input_tokens\":10,\"output_tokens\":4}}}}",
                    body.len()
                );
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    reply,
                )
                    .into_response()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    (format!("http://{addr}/v1/messages"), counter)
}

fn engine_request(core: CoreRequest) -> EngineRequest {
    let body = serde_json::json!({"model": "claude-sonnet-4-6", "max_tokens": 64});
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core,
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: llm_proxy_adapters::client_wire::ClientWireRequest::new(
                llm_proxy_adapters::client_wire::ClientWireFormat::AnthropicMessages,
                body,
            ),
        },
    )
    .with_request_id("req-s")
    .with_caller("caller-s")
    .with_body_hash("hash-s")
}

fn test_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind"),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "semantic-test".to_owned(),
            rate_limit_rpm: 1000,
            trust_forwarded_headers: false,
            log_format: Default::default(),
            admin_bind: None,
            admin_on_main_listener: false,
            endpoint_families: Default::default(),
            upstream_timeouts: Default::default(),
            bare_v1: Default::default(),
            preflight: Default::default(),
            semantic_cache: Default::default(),
            cache: Default::default(),
            idempotency: Default::default(),
            metrics: Default::default(),
        },
        auth: Default::default(),
        storage: None,
        replica: None,
        mcp: None,
    }
}

#[tokio::test]
async fn a_paraphrase_is_answered_without_a_second_upstream_call() {
    let (endpoint, counter) = spawn_upstream().await;
    let engine = engine_over(endpoint, config(0.6, Duration::from_secs(600), 16)).await;

    let first = engine
        .execute(engine_request(core_request(
            "What is the capital of France?",
        )))
        .await
        .expect("first request served");
    assert_eq!(*counter.requests.lock().await, 1);

    let second = engine
        .execute(engine_request(core_request(
            "What is France's capital city?",
        )))
        .await
        .expect("paraphrase served");
    assert_eq!(
        *counter.requests.lock().await,
        1,
        "the paraphrase was served from the semantic cache"
    );
    assert_eq!(
        first.core.content.len(),
        second.core.content.len(),
        "the cached response carries the same shape"
    );

    let third = engine
        .execute(engine_request(core_request(
            "Now write a bubble sort in Rust.",
        )))
        .await
        .expect("unrelated prompt served");
    assert_eq!(
        *counter.requests.lock().await,
        2,
        "the unrelated prompt went upstream"
    );
    assert!(
        !third.core.content.is_empty(),
        "the upstream answer came back"
    );
}

async fn engine_over(endpoint: String, semantic: SemanticCacheConfig) -> Engine {
    let mut adapters = std::collections::HashMap::new();
    adapters.insert(
        "messages".to_owned(),
        ProviderAdapterConfig {
            protocol: "anthropic_messages".to_owned(),
            endpoint,
            headers: Arc::new(std::collections::HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    );
    let provider = ProviderConfig {
        name: "fake".to_owned(),
        api_key: secrecy::SecretString::from("upstream-key"),
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes: ProviderRoutesConfig {
            messages: Some("messages".to_owned()),
            ..Default::default()
        },
        model_aliases: std::collections::HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    };
    let registry = ProviderRegistry::from_providers(vec![provider]).expect("registry");
    EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .cache(Arc::new(SemanticCache::new(
            semantic,
            Box::new(NgramEmbedder::new()),
        )))
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds")
}
