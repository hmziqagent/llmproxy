//! The Guard response-masking contract against same-format passthrough: a
//! guard that rewrites the completed response (a Buffered masking rule whose
//! pattern appears only in model output) must not be defeated by the raw
//! same-format body. The masked core — not the upstream bytes — is what the
//! client receives, what an idempotent replay serves, while a same-format
//! request that no guard transformed keeps receiving the upstream bytes
//! verbatim.

#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::http::{StatusCode, header};
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_core::{
    AppConfig, AuthStyle, IdempotencyConfig, ProviderAdapterConfig, ProviderConfig,
    ProviderRegistry, ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_engine::{
    Cache, ClientDialect, Engine, EngineBuilder, EngineError, EngineRequest, EngineResponse, Guard,
    GuardDecision, InboundRequest, MemoryCache,
};
use llm_proxy_protocol::core::{
    CoreContent, CoreMessage, CoreRequest, CoreResponse, CoreRole, ModelRef, SamplingOptions,
};
use llm_proxy_storage::{EventBus, RecordingBus};
use sha2::{Digest as _, Sha256};

/// The secret the fake upstream echoes back in its response text.
const SECRET: &str = "sk-upstreamsecret99";
/// The placeholder the masking guard substitutes for [`SECRET`].
const PLACEHOLDER: &str = "[MASKED]";
/// The exact bytes the fake upstream answers with, so the no-guard fast path
/// can be held to byte identity.
const UPSTREAM_BODY: &str = r#"{"id":"msg_mask","type":"message","role":"assistant","content":[{"type":"text","text":"the upstream saw sk-upstreamsecret99"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#;

/// The engine-trait shape of a Buffered masking guard whose rule matches
/// only response text: the request passes untouched (so nothing forces a
/// core re-encode and the same-format passthrough stays eligible), and the
/// completed response has the secret replaced with the placeholder.
#[derive(Debug)]
struct ResponseMaskingGuard;

#[async_trait]
impl Guard for ResponseMaskingGuard {
    async fn transform_response(&self, response: &mut CoreResponse) -> Result<(), EngineError> {
        for content in response.content.iter_mut() {
            if let CoreContent::Text { text, .. } = content {
                *text = text.replace(SECRET, PLACEHOLDER);
            }
        }
        Ok(())
    }
}

/// The request-side secret: a masking guard rewrites it before egress, but
/// the rejecting guard must see it first (slot 7 before slot 8).
const REQUEST_SECRET: &str = "sk-requestsecret42";

/// A Buffered request-masking guard: it replaces [`REQUEST_SECRET`] in the
/// request's words.
#[derive(Debug)]
struct RequestMaskingGuard;

#[async_trait]
impl Guard for RequestMaskingGuard {
    async fn transform_request(&self, request: &mut CoreRequest) -> Result<(), EngineError> {
        for message in &mut request.messages {
            for content in &mut message.content {
                if let CoreContent::Text { text, .. } = content {
                    *text = text.replace(REQUEST_SECRET, PLACEHOLDER);
                }
            }
        }
        Ok(())
    }
}

/// A pre-call guard that rejects any request whose words still carry
/// [`REQUEST_SECRET`]: its rule cannot hold once the text has been
/// rewritten, so it refuses instead of trusting a masker.
#[derive(Debug)]
struct RejectSecretRequestGuard;

#[async_trait]
impl Guard for RejectSecretRequestGuard {
    async fn inspect_request(&self, request: &CoreRequest) -> Result<GuardDecision, EngineError> {
        for message in &request.messages {
            for content in &message.content {
                if let CoreContent::Text { text, .. } = content
                    && text.contains(REQUEST_SECRET)
                {
                    return Ok(GuardDecision::Reject(
                        "request still carries a secret".to_owned(),
                    ));
                }
            }
        }
        Ok(GuardDecision::Allow)
    }
}

/// A request-rewriting guard that only counts how many times the engine
/// asked it to rewrite; it changes nothing itself.
#[derive(Debug)]
struct CountingRequestTransformGuard {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Guard for CountingRequestTransformGuard {
    async fn transform_request(&self, _request: &mut CoreRequest) -> Result<(), EngineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// A loopback fake upstream that counts calls and answers the fixed
/// secret-bearing Anthropic body.
async fn spawn_secret_upstream() -> (String, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move || {
            let calls = Arc::clone(&counted);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    UPSTREAM_BODY,
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("fake upstream addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    (format!("http://{addr}/v1/messages"), calls)
}

fn test_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind addr"),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "masking-passthrough-test".to_owned(),
            rate_limit_rpm: 1000,
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
    }
}

fn provider(endpoint: String) -> ProviderConfig {
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
    let routes = ProviderRoutesConfig {
        messages: Some("messages".to_owned()),
        chat_completions: None,
        ..Default::default()
    };
    ProviderConfig {
        name: "fake".to_owned(),
        api_key: secrecy::SecretString::from("upstream-key"),
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes,
        model_aliases: std::collections::HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn build_engine(endpoint: String, guard: Option<Arc<dyn Guard>>, idempotency: bool) -> Engine {
    let mut config = test_config();
    if idempotency {
        config.server.idempotency = IdempotencyConfig {
            enabled: true,
            ttl: Duration::from_secs(3600),
            max_entries: 1024,
        };
    }
    let registry = ProviderRegistry::from_providers(vec![provider(endpoint)]).expect("registry");
    let mut builder = EngineBuilder::new(config)
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>);
    if let Some(guard) = guard {
        builder = builder.guard(guard);
    }
    builder.build().expect("engine builds")
}

/// Like [`build_engine`], but with several guards in the given order and an
/// optional cache (the engine defaults to [`NoCache`]).
fn build_engine_with_guards_and_cache(
    endpoint: String,
    guards: Vec<Arc<dyn Guard>>,
    cache: Option<Arc<dyn Cache>>,
) -> Engine {
    let config = test_config();
    let registry = ProviderRegistry::from_providers(vec![provider(endpoint)]).expect("registry");
    let mut builder = EngineBuilder::new(config)
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>);
    for guard in guards {
        builder = builder.guard(guard);
    }
    if let Some(cache) = cache {
        builder = builder.cache(cache);
    }
    builder.build().expect("engine builds")
}

fn core_request(prompt: &str) -> CoreRequest {
    CoreRequest {
        model: ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        messages: vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: prompt.to_owned(),
                cache: None,
            }],
        }],
        system: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        sampling: SamplingOptions::default(),
        stream: false,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

/// An Anthropic-dialect same-format request whose prompt is clean: nothing
/// in it matches the guard's rule, so the request-side re-encode signal
/// never fires and the response-side passthrough decision is the only thing
/// standing between the client and the unmasked upstream bytes.
fn engine_request(prompt: &str, idempotency_key: Option<&str>) -> EngineRequest {
    let mut headers = std::collections::HashMap::new();
    if let Some(key) = idempotency_key {
        headers.insert("idempotency-key".to_owned(), key.to_owned());
    }
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let body_hash = format!("{:x}", hasher.finalize());
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core_request(prompt),
        InboundRequest {
            auth: None,
            headers,
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({"model": "claude-sonnet-4-6", "max_tokens": 64}),
            ),
        },
    )
    .with_caller("caller-mask")
    .with_body_hash(body_hash)
}

/// The bytes the server hands the client for an Anthropic-dialect response
/// (`routes/core_pipeline.rs`): the engine's raw body when it provides one,
/// the core re-encoded in the client's dialect otherwise.
fn client_visible_body(response: &EngineResponse) -> String {
    let bytes = match &response.raw_body {
        Some(raw) => raw.clone(),
        None => {
            let encoded =
                llm_proxy_protocol::client::anthropic::encode_response(response.core.clone())
                    .expect("the masked core re-encodes for the client");
            serde_json::to_vec(&encoded).expect("the encoded response serializes")
        }
    };
    String::from_utf8(bytes).expect("the client body is UTF-8")
}

/// A guard whose rule matches only the response must not be defeated by the
/// same-format passthrough: the client-visible bytes are the masked form,
/// not the raw upstream body carrying the secret.
#[tokio::test]
async fn a_response_only_masking_guard_serves_the_masked_form_on_a_same_format_route() {
    let (endpoint, calls) = spawn_secret_upstream().await;
    let engine = build_engine(endpoint, Some(Arc::new(ResponseMaskingGuard)), false);

    let response = engine
        .execute(engine_request("a perfectly clean prompt", None))
        .await
        .expect("request served");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the clean request reaches the upstream"
    );
    assert!(
        response.raw_body.is_none(),
        "a rewritten response must not carry the raw upstream bytes downstream, got {}",
        response
            .raw_body
            .as_deref()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_default()
    );
    let client_body = client_visible_body(&response);
    assert!(
        !client_body.contains(SECRET),
        "the secret must not reach the client, got {client_body}"
    );
    assert!(
        client_body.contains(PLACEHOLDER),
        "the placeholder must reach the client, got {client_body}"
    );
}

/// The idempotency claim stores the engine's outcome: when the guard masked
/// the response, a replay under the same key serves the masked form — the
/// unmasked raw body must never be stored or replayed.
#[tokio::test]
async fn an_idempotent_replay_of_a_masked_same_format_request_serves_masked_bytes() {
    let (endpoint, calls) = spawn_secret_upstream().await;
    let engine = build_engine(endpoint, Some(Arc::new(ResponseMaskingGuard)), true);

    let first = engine
        .execute(engine_request(
            "clean prompt under a key",
            Some("key-mask-1"),
        ))
        .await
        .expect("first request serves");
    let replay = engine
        .execute(engine_request(
            "clean prompt under a key",
            Some("key-mask-1"),
        ))
        .await
        .expect("the keyed retry replays");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the replay is answered from the idempotency store, not a second upstream call"
    );
    for (label, response) in [("first", &first), ("replay", &replay)] {
        assert!(
            response.raw_body.is_none(),
            "{label}: the stored outcome must not carry the raw upstream bytes, got {}",
            response
                .raw_body
                .as_deref()
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                .unwrap_or_default()
        );
        let client_body = client_visible_body(response);
        assert!(
            !client_body.contains(SECRET),
            "{label}: the secret must not reach the client, got {client_body}"
        );
        assert!(
            client_body.contains(PLACEHOLDER),
            "{label}: the placeholder must reach the client, got {client_body}"
        );
    }
}

/// The fast path the masking suppression must not break: with no guard
/// configured, a same-format request returns the upstream bytes verbatim.
#[tokio::test]
async fn without_a_guard_the_same_format_fast_path_stays_byte_identical() {
    let (endpoint, _calls) = spawn_secret_upstream().await;
    let engine = build_engine(endpoint, None, false);

    let response = engine
        .execute(engine_request("any prompt", None))
        .await
        .expect("request served");

    assert_eq!(
        response.raw_body.as_deref(),
        Some(UPSTREAM_BODY.as_bytes()),
        "no response transform ran: the client must receive the upstream bytes verbatim"
    );
}

/// A rejecting guard judges the client's words, not an earlier masker's
/// output: the pre-call guard pass (slot 7) runs before any request rewrite
/// (slot 8), so a secret the rejector blocks is refused — before routing,
/// admission or any upstream attempt — even though the masking guard
/// installed before it would have redacted it.
#[tokio::test]
async fn a_rejecting_guard_sees_the_unmasked_request() {
    let (endpoint, calls) = spawn_secret_upstream().await;
    let engine = build_engine_with_guards_and_cache(
        endpoint,
        vec![
            Arc::new(RequestMaskingGuard) as Arc<dyn Guard>,
            Arc::new(RejectSecretRequestGuard) as Arc<dyn Guard>,
        ],
        None,
    );

    let outcome = engine
        .execute(engine_request(&format!("echo {REQUEST_SECRET} back"), None))
        .await;

    let error = outcome.expect_err("the rejecting guard refuses the request");
    assert!(
        matches!(&error, EngineError::InvalidRequest(message) if message.contains("secret")),
        "expected the pre-call guard rejection, got {error}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the request must be refused before any upstream attempt"
    );
}

/// The guards' request rewrite (slot 8) runs on the ladder path only: an
/// identical retry answered from the cache (slot 5) never re-runs it, and
/// cache identity is what the client sent — not what a guard rewrote.
#[tokio::test]
async fn a_cache_hit_is_served_without_running_request_transforms() {
    let (endpoint, calls) = spawn_secret_upstream().await;
    let transforms = Arc::new(AtomicUsize::new(0));
    let engine = build_engine_with_guards_and_cache(
        endpoint,
        vec![Arc::new(CountingRequestTransformGuard {
            calls: Arc::clone(&transforms),
        }) as Arc<dyn Guard>],
        Some(Arc::new(MemoryCache::new(Duration::from_secs(60), 16)) as Arc<dyn Cache>),
    );

    let first = engine
        .execute(engine_request("cache identity prompt", None))
        .await
        .expect("first request serves");
    let second = engine
        .execute(engine_request("cache identity prompt", None))
        .await
        .expect("second request serves");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the second request must be a cache hit, not a second upstream call"
    );
    assert_eq!(
        transforms.load(Ordering::SeqCst),
        1,
        "only the cache-missing request runs the guards' request rewrite"
    );
    assert_eq!(
        first.core.content, second.core.content,
        "the cache hit serves the first request's stored response"
    );
}
