//! Plan 03, step 4: the provider set is open. A custom protocol implemented
//! against the public traits routes through the full pipeline, and the
//! extension traits (admission, cache, credential source) are driven by the
//! engine in the documented slot order.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_adapters::request::ProxyRequest;
use llm_proxy_adapters::{
    ProtocolName, ProviderAdapter, ProviderAdapterTarget, ProviderStreamDecoder,
};
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
    ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_engine::error::EngineError;
use llm_proxy_engine::traits::{
    Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket, Cache, CacheEntry, CacheKey,
    CredentialSource, Guard, GuardDecision, UsageSink,
};
use llm_proxy_engine::{
    ClientDialect, EngineBuilder, EngineRequest, EngineResponse, InboundRequest,
};
use llm_proxy_protocol::core::{CoreEvent, CoreRequest, CoreResponse};
use llm_proxy_storage::{EventBus, NoopBus, ProxyEvent, RecordingBus};
use secrecy::SecretString;

const ECHO_PROTOCOL: &str = "echo_v1";

/// A user-supplied provider codec: encodes the model into the URL query and
/// decodes a fixed response. Registered like an SDK user would.
#[derive(Debug)]
struct EchoAdapter;

impl ProviderAdapter for EchoAdapter {
    fn protocol(&self) -> ProtocolName {
        ProtocolName::new(ECHO_PROTOCOL)
    }

    fn encode_request(
        &self,
        core: &CoreRequest,
        target: &ProviderAdapterTarget,
        _client: &ClientWireRequest,
    ) -> Result<ProxyRequest, llm_proxy_adapters::ProviderError> {
        let body = format!(
            "{{\"model\":\"{}\",\"text\":\"{}\"}}",
            target.upstream_model,
            core.messages.len()
        );
        Ok(ProxyRequest {
            url: target.endpoint.clone(),
            auth: llm_proxy_adapters::request::AuthHeaders {
                style: target.auth_style,
                api_key: target.api_key.clone(),
            },
            body: body.into_bytes(),
            stream: false,
            extra_headers: HashMap::new(),
            forwarded_headers: client_forwarded(_client),
        })
    }

    fn decode_response(
        &self,
        bytes: &[u8],
        target: &ProviderAdapterTarget,
    ) -> Result<CoreResponse, llm_proxy_adapters::ProviderError> {
        let parsed: serde_json::Value =
            serde_json::from_slice(bytes).map_err(llm_proxy_adapters::ProviderError::Serialize)?;
        let text = parsed
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();
        Ok(CoreResponse {
            id: Some("echo_1".to_owned()),
            model: llm_proxy_protocol::core::ModelRef {
                requested: target.requested_model.clone(),
                upstream: Some(target.upstream_model.clone()),
            },
            content: vec![llm_proxy_protocol::core::CoreContent::Text { text, cache: None }],
            stop_reason: llm_proxy_protocol::core::StopReason::EndTurn,
            stop_sequence: None,
            usage: llm_proxy_protocol::core::Usage::default(),
            provider_meta: serde_json::Map::new(),
            cost: None,
        })
    }

    fn new_stream_decoder(
        &self,
        _target: &ProviderAdapterTarget,
    ) -> Box<dyn ProviderStreamDecoder + Send> {
        // The echo test only exercises the non-streaming path; the trait
        // still requires a decoder, so return one that decodes nothing.
        Box::new(EmptyDecoder)
    }
}

fn client_forwarded(_client: &ClientWireRequest) -> HashMap<String, String> {
    HashMap::new()
}

/// A decoder that yields nothing: the echo adapter never streams.
#[derive(Debug)]
struct EmptyDecoder;

impl ProviderStreamDecoder for EmptyDecoder {
    fn decode_frame(
        &mut self,
        _frame: &llm_proxy_adapters::SseFrame,
    ) -> Result<Vec<CoreEvent>, llm_proxy_adapters::ProviderError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<CoreEvent>, llm_proxy_adapters::ProviderError> {
        Ok(Vec::new())
    }

    fn saw_terminal(&self) -> bool {
        false
    }
}

/// Records the order the engine drives the admission slots.
#[derive(Debug, Default)]
struct RecordingAdmission {
    events: std::sync::Mutex<Vec<&'static str>>,
}

#[async_trait]
impl Admission for RecordingAdmission {
    async fn reserve(&self, claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        assert_eq!(claim.provider, "echo");
        self.events.lock().expect("admission lock").push("reserve");
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, outcome: &AdmissionOutcome<'_>) {
        self.events
            .lock()
            .expect("admission lock")
            .push(if outcome.succeeded {
                "settle-ok"
            } else {
                "settle-failed"
            });
    }
}

/// Rotates credentials per attempt.
#[derive(Debug)]
struct RotatingCredential {
    counter: AtomicUsize,
}

#[async_trait]
impl CredentialSource for RotatingCredential {
    async fn pick(
        &self,
        _query: &llm_proxy_engine::CredentialQuery<'_>,
    ) -> Result<llm_proxy_engine::CredentialPick, EngineError> {
        let n = self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(llm_proxy_engine::CredentialPick {
            secret: SecretString::from(format!("rotated-key-{n}")),
            credential_id: Some(format!("rot-{n}")),
        })
    }
}

/// An in-memory cache.
#[derive(Debug, Default)]
struct HashMapCache {
    hits: AtomicUsize,
    store: std::sync::Mutex<HashMap<String, CacheEntry>>,
}

#[async_trait]
impl Cache for HashMapCache {
    async fn lookup(&self, key: &CacheKey<'_>) -> Result<Option<CacheEntry>, EngineError> {
        let found = self
            .store
            .lock()
            .expect("cache lock")
            .get(key.body_hash)
            .cloned();
        if found.is_some() {
            self.hits.fetch_add(1, Ordering::SeqCst);
        }
        Ok(found)
    }

    async fn store(&self, key: &CacheKey<'_>, entry: &CacheEntry) {
        self.store
            .lock()
            .expect("cache lock")
            .insert(key.body_hash.to_owned(), entry.clone());
    }
}

/// A pre-call guard that rejects a marker string.
#[derive(Debug)]
struct BlockSecretText;

#[async_trait]
impl Guard for BlockSecretText {
    async fn inspect_request(&self, request: &CoreRequest) -> Result<GuardDecision, EngineError> {
        let has_secret = request.messages.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(content, llm_proxy_protocol::core::CoreContent::Text { text, .. } if text.contains("forbidden-marker"))
            })
        });
        if has_secret {
            Ok(GuardDecision::Reject(
                "request rejected by test guard".to_owned(),
            ))
        } else {
            Ok(GuardDecision::Allow)
        }
    }
}

fn upstream_app(seen: std::sync::Arc<std::sync::Mutex<Vec<(String, String)>>>) -> axum::Router {
    use axum::response::IntoResponse;
    axum::Router::new().route(
        "/{*path}",
        axum::routing::post(move |headers: axum::http::HeaderMap, body: bytes::Bytes| {
            let seen = Arc::clone(&seen);
            async move {
                let auth = headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("<none>")
                    .to_owned();
                let body = String::from_utf8_lossy(&body).into_owned();
                seen.lock()
                    .expect("upstream lock")
                    .push((auth, body.clone()));
                let payload = serde_json::json!({ "text": body });
                (axum::http::StatusCode::OK, payload.to_string()).into_response()
            }
        }),
    )
}

async fn spawn(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    format!("http://{addr}/echo")
}

fn echo_provider(endpoint: String) -> ProviderConfig {
    let mut adapters = HashMap::new();
    adapters.insert(
        "main".to_owned(),
        ProviderAdapterConfig {
            protocol: ECHO_PROTOCOL.to_owned(),
            endpoint,
            headers: Arc::new(HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    );
    let routes = ProviderRoutesConfig {
        messages: Some("main".to_owned()),
        chat_completions: None,
        ..Default::default()
    };
    ProviderConfig {
        name: "echo".to_owned(),
        api_key: SecretString::from("configured-key"),
        auth_style: AuthStyle::Bearer,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes,
        model_aliases: HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind"),
            request_timeout: std::time::Duration::from_secs(30),
            shutdown_timeout: std::time::Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "open-set-test".to_owned(),
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
    }
}

fn request(text: &str) -> EngineRequest {
    EngineRequest::new(
        "echo",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        llm_proxy_protocol::core::CoreRequest {
            model: llm_proxy_protocol::core::ModelRef {
                requested: "echo-model".to_owned(),
                upstream: None,
            },
            messages: vec![llm_proxy_protocol::core::CoreMessage {
                role: llm_proxy_protocol::core::CoreRole::User,
                content: vec![llm_proxy_protocol::core::CoreContent::Text {
                    text: text.to_owned(),
                    cache: None,
                }],
            }],
            ..empty_core()
        },
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_body_hash("hash-1")
}

fn streaming_request(text: &str) -> EngineRequest {
    let mut req = request(text);
    req.core.stream = true;
    req
}

fn empty_core() -> CoreRequest {
    CoreRequest {
        model: llm_proxy_protocol::core::ModelRef {
            requested: "echo-model".to_owned(),
            upstream: None,
        },
        system: Vec::new(),
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        sampling: llm_proxy_protocol::core::SamplingOptions::default(),
        stream: false,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

#[tokio::test]
async fn a_registered_custom_protocol_serves_through_the_pipeline() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let endpoint = spawn(upstream_app(Arc::clone(&seen))).await;

    let registry =
        ProviderRegistry::from_providers(vec![echo_provider(endpoint)]).expect("registry");
    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("custom adapter registers")
        .providers(Arc::new(registry))
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");

    let response: EngineResponse = engine.execute(request("hello")).await.expect("execute");
    let text = match response.core.content.first() {
        Some(llm_proxy_protocol::core::CoreContent::Text { text, .. }) => text.clone(),
        _ => String::new(),
    };
    assert!(
        text.contains("echo-model"),
        "custom codec round-trips: {text}"
    );

    let upstream = seen.lock().expect("upstream lock").clone();
    assert_eq!(upstream.len(), 1, "exactly one upstream call");
    assert_eq!(upstream[0].0, "Bearer configured-key");
}

#[tokio::test]
async fn duplicate_protocol_registration_is_rejected() {
    let mut registry = llm_proxy_adapters::ProviderAdapterRegistry::builtin();
    let first = registry.register(Arc::new(EchoAdapter)).is_ok();
    assert!(first, "first registration succeeds");
    let second = registry.register(Arc::new(EchoAdapter));
    assert!(
        second.is_err(),
        "registering the same protocol twice must fail, not silently replace"
    );
}

#[tokio::test]
async fn the_engine_drives_admission_reserve_then_settle() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let endpoint = spawn(upstream_app(Arc::clone(&seen))).await;
    let registry =
        ProviderRegistry::from_providers(vec![echo_provider(endpoint)]).expect("registry");
    let admission = Arc::new(RecordingAdmission::default());

    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(registry))
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");

    engine.execute(request("hello")).await.expect("execute");
    assert_eq!(
        admission.events.lock().expect("admission lock").as_slice(),
        ["reserve", "settle-ok"],
        "a successful request reserves then settles"
    );

    // A guard rejection settles as failed without egress.
    admission.events.lock().expect("admission lock").clear();
    let failing = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(
            ProviderRegistry::from_providers(vec![echo_provider(
                "http://127.0.0.1:9/echo".to_owned(),
            )])
            .expect("registry"),
        ))
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .guard(Arc::new(BlockSecretText) as Arc<dyn Guard>)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");
    let error = failing
        .execute(request("forbidden-marker"))
        .await
        .expect_err("guard rejects");
    assert!(matches!(error, EngineError::InvalidRequest(ref msg) if msg.contains("guard")));
    assert_eq!(
        admission.events.lock().expect("admission lock").as_slice(),
        ["reserve", "settle-failed"],
        "a rejected request refunds its reservation"
    );

    // An upstream failure after a successful reserve must not leak the
    // reservation: the egress error settles it as failed too.
    admission.events.lock().expect("admission lock").clear();
    let unreachable = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(
            ProviderRegistry::from_providers(vec![echo_provider(
                "http://127.0.0.1:9/echo".to_owned(),
            )])
            .expect("registry"),
        ))
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");
    let error = unreachable
        .execute(request("hello"))
        .await
        .expect_err("upstream unreachable");
    assert!(
        !matches!(error, EngineError::InvalidRequest(_)),
        "the failure must come from egress, not a guard: {error:?}"
    );
    assert_eq!(
        admission.events.lock().expect("admission lock").as_slice(),
        ["reserve", "settle-failed"],
        "an upstream failure refunds its reservation"
    );

    // Same contract on the streaming path: egress failing before any byte
    // settles the reservation the stream task would otherwise own.
    admission.events.lock().expect("admission lock").clear();
    let error = unreachable
        .execute_stream(streaming_request("hello"))
        .await
        .expect_err("stream egress fails");
    assert!(
        !matches!(error, EngineError::InvalidRequest(_)),
        "the failure must come from egress, not a guard: {error:?}"
    );
    assert_eq!(
        admission.events.lock().expect("admission lock").as_slice(),
        ["reserve", "settle-failed"],
        "a failed stream setup refunds its reservation"
    );
}

/// Panics inside the stream task (the pick runs there), standing in for
/// any bug that unwinds the task after the reservation was taken.
#[derive(Debug)]
struct PanickingCredential;

#[async_trait]
impl CredentialSource for PanickingCredential {
    async fn pick(
        &self,
        _query: &llm_proxy_engine::CredentialQuery<'_>,
    ) -> Result<llm_proxy_engine::CredentialPick, EngineError> {
        panic!("injected stream-task panic");
    }
}

#[tokio::test]
async fn a_stream_task_that_panics_still_settles_its_reservation() {
    let admission = Arc::new(RecordingAdmission::default());
    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(
            ProviderRegistry::from_providers(vec![echo_provider(
                "http://127.0.0.1:9/echo".to_owned(),
            )])
            .expect("registry"),
        ))
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .credentials(Arc::new(PanickingCredential) as Arc<dyn CredentialSource>)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");

    let error = engine
        .execute_stream(streaming_request("hello"))
        .await
        .expect_err("the stream task died before its first byte");
    assert!(
        matches!(error, EngineError::Internal(_)),
        "a dead task surfaces as an internal error, got {error:?}"
    );

    // The supervisor settles once the task's join handle resolves, which
    // happens after `execute_stream` has already returned.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let events = admission.events.lock().expect("admission lock").clone();
        if events == ["reserve", "settle-failed"] {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the panicked task's reservation must be settled exactly once as failed, got {events:?}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn a_custom_credential_source_rotates_keys() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let endpoint = spawn(upstream_app(Arc::clone(&seen))).await;
    let registry =
        ProviderRegistry::from_providers(vec![echo_provider(endpoint)]).expect("registry");

    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(registry))
        .credentials(Arc::new(RotatingCredential {
            counter: AtomicUsize::new(0),
        }) as Arc<dyn CredentialSource>)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");

    engine.execute(request("one")).await.expect("execute");
    engine.execute(request("two")).await.expect("execute");

    let upstream = seen.lock().expect("upstream lock").clone();
    assert_eq!(upstream.len(), 2);
    assert_eq!(upstream[0].0, "Bearer rotated-key-0");
    assert_eq!(upstream[1].0, "Bearer rotated-key-1");
}

#[tokio::test]
async fn a_custom_cache_serves_the_second_request_without_egress() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let endpoint = spawn(upstream_app(Arc::clone(&seen))).await;
    let registry =
        ProviderRegistry::from_providers(vec![echo_provider(endpoint)]).expect("registry");
    let cache = Arc::new(HashMapCache::default());

    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(registry))
        .cache(Arc::clone(&cache) as Arc<dyn Cache>)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");

    let first = engine.execute(request("same")).await.expect("first");
    let second = engine.execute(request("same")).await.expect("second");

    assert_eq!(
        seen.lock().expect("upstream lock").len(),
        1,
        "the second identical request must be served from the cache"
    );
    assert_eq!(cache.hits.load(Ordering::SeqCst), 1);
    assert_eq!(first.core.id, second.core.id, "the cached response replays");
}

/// A user-supplied [`UsageSink`]: captures every ledger event the engine
/// emits, exactly as an SDK user would.
#[derive(Debug, Default)]
struct CollectingSink {
    events: std::sync::Mutex<Vec<ProxyEvent>>,
}

impl UsageSink for CollectingSink {
    fn record(&self, event: &ProxyEvent) {
        self.events
            .lock()
            .expect("usage sink lock")
            .push(event.clone());
    }
}

fn event_kind(event: &ProxyEvent) -> &'static str {
    match event {
        ProxyEvent::RequestReceived(_) => "request_received",
        ProxyEvent::AttemptCompleted(_) => "attempt_completed",
        ProxyEvent::ResponseCompleted(_) => "response_completed",
        ProxyEvent::ResponseFailed(_) => "response_failed",
    }
}

fn event_request_id(event: &ProxyEvent) -> String {
    match event {
        ProxyEvent::RequestReceived(row) => row.request_id.clone(),
        ProxyEvent::AttemptCompleted(row) => row.request_id.clone(),
        ProxyEvent::ResponseCompleted(row) => row.request_id.clone(),
        ProxyEvent::ResponseFailed(row) => row.request_id.clone(),
    }
}

#[tokio::test]
async fn a_custom_usage_sink_receives_the_ledger_events() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let endpoint = spawn(upstream_app(Arc::clone(&seen))).await;
    let registry =
        ProviderRegistry::from_providers(vec![echo_provider(endpoint.clone())]).expect("registry");
    let sink = Arc::new(CollectingSink::default());

    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(registry))
        .usage_sink(Arc::clone(&sink) as Arc<dyn UsageSink>)
        .build()
        .expect("engine builds");

    let req = request("hello");
    let request_id = req.request_id.clone();
    engine.execute(req).await.expect("execute");

    let events = sink.events.lock().expect("usage sink lock").clone();
    let kinds: Vec<&'static str> = events.iter().map(event_kind).collect();
    assert_eq!(
        kinds,
        [
            "request_received",
            "attempt_completed",
            "response_completed"
        ],
        "one successful non-streaming request writes its ledger rows in order"
    );
    let ids: Vec<String> = events.iter().map(event_request_id).collect();
    assert_eq!(
        ids,
        [request_id.clone(), request_id.clone(), request_id],
        "every row carries the request's id"
    );

    // Last setter wins: a builder that installs `.event_bus(recording)` and
    // then `.usage_sink(custom)` delivers to the custom sink only.
    let later_sink = Arc::new(CollectingSink::default());
    let bus = Arc::new(RecordingBus::new());
    let registry =
        ProviderRegistry::from_providers(vec![echo_provider(endpoint)]).expect("registry");
    let engine = EngineBuilder::new(config())
        .register_adapter(Arc::new(EchoAdapter))
        .expect("adapter registers")
        .providers(Arc::new(registry))
        .event_bus(Arc::clone(&bus) as Arc<dyn EventBus>)
        .usage_sink(Arc::clone(&later_sink) as Arc<dyn UsageSink>)
        .build()
        .expect("engine builds");
    engine.execute(request("hello")).await.expect("execute");
    assert!(
        bus.snapshot().is_empty(),
        "a later usage_sink replaces an earlier event_bus: got {:?}",
        bus.snapshot()
    );
    let later_events = later_sink.events.lock().expect("usage sink lock").clone();
    let later_kinds: Vec<&'static str> = later_events.iter().map(event_kind).collect();
    assert_eq!(
        later_kinds,
        [
            "request_received",
            "attempt_completed",
            "response_completed"
        ],
        "the later-set sink receives the ledger rows"
    );
}
