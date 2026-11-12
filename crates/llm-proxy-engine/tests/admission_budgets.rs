//! Admission and budgets through the real engine, against a loopback fake
//! upstream (plan 05 steps 3-7): the done-when properties — concurrent
//! requests cannot overspend a budget by more than the reserved amount, a
//! refund restores the budget, cache hits never reserve — plus the
//! concurrency cap's 503 and the per-key allowlists end to end.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::http::{StatusCode, header};
use axum::routing::post;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_core::{
    AppConfig, AuthStyle, ClientKeySpec, KeyBudgetConfig, KeyWindowLimits, PoolConfig,
    ProviderAdapterConfig, ProviderConcurrencyConfig, ProviderConfig, ProviderRegistry,
    ProviderRouteKind, ProviderRoutesConfig, RetryConfig, ServerConfig,
};
use llm_proxy_engine::{
    Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket, Cache, CacheEntry, CacheKey,
    ClientDialect, EngineBuilder, EngineError, EngineRequest, Guard, GuardDecision, InboundRequest,
    KeyPolicy, PolicyAdmission,
};
use llm_proxy_protocol::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreResponse, CoreRole, Cost, ModelRef,
    SamplingOptions, Usage,
};
use llm_proxy_storage::{EventBus, ProxyEvent, RecordingBus};

async fn spawn_upstream(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("fake upstream addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "fake upstream stopped");
        }
    });
    format!("http://{addr}/v1/messages")
}

fn upstream_always_ok() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
            )
        }),
    )
}

fn upstream_always_500() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"error":"boom"}"#,
            )
        }),
    )
}

/// An upstream that answers only after `release` is set, holding its
/// in-flight slot open; once set it stays set, so later requests answer
/// immediately.
fn upstream_held(release: Arc<tokio::sync::watch::Receiver<bool>>) -> Router {
    Router::new().route(
        "/{*path}",
        post(move || {
            let mut release = release.as_ref().clone();
            async move {
                while !*release.borrow() {
                    if release.changed().await.is_err() {
                        break;
                    }
                }
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
                )
            }
        }),
    )
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
            server_name: "admission-test".to_owned(),
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

fn provider(
    endpoint: String,
    concurrency: Option<llm_proxy_core::ProviderConcurrencyConfig>,
) -> ProviderConfig {
    let mut adapters = HashMap::new();
    adapters.insert(
        "messages".to_owned(),
        ProviderAdapterConfig {
            protocol: "anthropic_messages".to_owned(),
            endpoint,
            headers: Arc::new(HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    );
    ProviderConfig {
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
            chat_completions: None,
            ..Default::default()
        },
        model_aliases: HashMap::new(),
        discovery: None,
        catalog: None,
        pricing: HashMap::from([(
            llm_proxy_core::ModelId::new("claude-sonnet-4-6"),
            llm_proxy_core::ModelPricing {
                input: "0.0001".parse().expect("decimal"),
                output: "0.0001".parse().expect("decimal"),
                ..llm_proxy_core::ModelPricing::default()
            },
        )]),
        concurrency,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn core_request(model: &str) -> CoreRequest {
    CoreRequest {
        model: ModelRef {
            requested: model.to_owned(),
            upstream: None,
        },
        messages: vec![CoreMessage {
            role: CoreRole::User,
            content: vec![CoreContent::Text {
                text: "hello".to_owned(),
                cache: None,
            }],
        }],
        system: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        sampling: SamplingOptions {
            max_tokens: Some(5),
            ..SamplingOptions::default()
        },
        stream: false,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

fn engine_request(caller: &str, model: &str) -> EngineRequest {
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core_request(model),
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id(format!("req-{model}"))
    .with_caller(caller.to_owned())
    .with_body_hash("hash")
}

/// A request with a distinct body, for tests of concurrent *demand*:
/// identical concurrent requests from one caller share one upstream call
/// since plan 06, so a test that means "another request" must send one.
fn distinct_engine_request(caller: &str, model: &str, nonce: &str) -> EngineRequest {
    let mut core = core_request(model);
    if let Some(message) = core.messages.first_mut()
        && let Some(CoreContent::Text { text, .. }) = message.content.first_mut()
    {
        *text = format!("hello {nonce}");
    }
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core,
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id(format!("req-{model}-{nonce}"))
    .with_caller(caller.to_owned())
    .with_body_hash(format!("hash-{nonce}"))
}

#[tokio::test]
async fn a_failing_request_refunds_the_budget_and_successes_charge_real_usage() {
    let ok = spawn_upstream(upstream_always_ok()).await;
    let failing = spawn_upstream(upstream_always_500()).await;

    // One shared admission across two engines (ok + failing upstream), one
    // budgeted key: each success bills 10 in + 4 out = 0.0014 USD. Pricing
    // comes from the registry the admission is built from.
    let registry = ProviderRegistry::from_providers(vec![provider(ok, None)]).expect("registry");
    let admission = Arc::new(PolicyAdmission::from_config(&fake_auth_config(), &registry));
    let ok_engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("ok engine");
    let failing_registry =
        ProviderRegistry::from_providers(vec![provider(failing, None)]).expect("registry");
    let failing_engine = EngineBuilder::new(test_config())
        .providers(Arc::new(failing_registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("failing engine");

    // A failing request reserves and refunds: afterwards the budget is
    // untouched, so both successes below still fit.
    let error = failing_engine
        .execute(engine_request("laptop", "claude-sonnet-4-6"))
        .await
        .expect_err("upstream 500");
    assert_eq!(error.status_code(), 502, "got {error:?}");

    ok_engine
        .execute(engine_request("laptop", "claude-sonnet-4-6"))
        .await
        .expect("first success");
    ok_engine
        .execute(engine_request("laptop", "claude-sonnet-4-6"))
        .await
        .expect("second success");
    // 0.0014 x 2 = 0.0028 spent; a third would need 0.0028 + estimate
    // (~0.0014) > 0.0032.
    let error = ok_engine
        .execute(engine_request("laptop", "claude-sonnet-4-6"))
        .await
        .expect_err("budget refuses the third");
    match &error {
        EngineError::BudgetExhausted {
            message,
            retry_after,
        } => {
            assert!(message.contains("budget"), "got {message}");
            assert!(retry_after.is_some(), "the daily reset is advised");
        }
        other => panic!("expected BudgetExhausted, got {other:?}"),
    }
    assert_eq!(error.status_code(), 402);
}

fn fake_auth_config() -> llm_proxy_core::AuthConfig {
    llm_proxy_core::AuthConfig {
        mode: llm_proxy_core::AuthMode::ClientKeys,
        clients: HashMap::from([(
            "laptop".to_owned(),
            ClientKeySpec::Policy {
                verifier: secrecy::SecretString::from("$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA"),
                allow_models: Vec::new(),
                allow_providers: Vec::new(),
                limits: Some(KeyWindowLimits {
                    rpm: None,
                    tpm: None,
                    rpd: None,
                }),
                budget: Some(KeyBudgetConfig {
                    usd_per_day: "0.0032".parse().expect("decimal"),
                    unestimatable: llm_proxy_core::UnestimatableCostPolicy::Reject,
                    assumed_max_output_tokens: 5,
                }),
            },
        )]),
        admin: None,
        legacy_bare_tokens: false,
    }
}

#[tokio::test]
async fn concurrent_requests_cannot_overspend_the_budget_by_more_than_reserved() {
    let ok = spawn_upstream(upstream_always_ok()).await;
    let registry = ProviderRegistry::from_providers(vec![provider(ok, None)]).expect("registry");
    let admission = Arc::new(PolicyAdmission::from_config(&fake_auth_config(), &registry));
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("engine");

    // Fire five concurrent requests against a 0.0032 budget whose requests
    // each reserve ~0.0014: at most two are admitted; the rest answer 402
    // before any egress.
    let mut handles = Vec::new();
    for index in 0..5 {
        let engine = engine.clone();
        handles.push(tokio::spawn(async move {
            engine
                .execute(distinct_engine_request(
                    "laptop",
                    "claude-sonnet-4-6",
                    &index.to_string(),
                ))
                .await
                .map(|_| ())
                .map_err(|error| error.status_code())
                .map_err(|code| (index, code))
        }));
    }
    let mut admitted = 0usize;
    let mut refused_402 = 0usize;
    for handle in handles {
        match handle.await.expect("task") {
            Ok(()) => admitted += 1,
            Err((_, 402)) => refused_402 += 1,
            Err((index, code)) => panic!("request {index} failed with {code}, expected 200 or 402"),
        }
    }
    assert_eq!(admitted + refused_402, 5);
    assert!(
        (1..=2).contains(&admitted),
        "admitted {admitted}; the budget bounds committed plus reserved spend"
    );
    assert!(refused_402 >= 3, "refused {refused_402}");
}

#[tokio::test]
async fn a_saturated_concurrency_cap_answers_503_and_frees_on_completion() {
    let release = Arc::new(tokio::sync::watch::Sender::new(false));
    let release_rx = release.subscribe();
    let endpoint = spawn_upstream(upstream_held(Arc::new(release_rx.clone()))).await;
    let bus = Arc::new(RecordingBus::new());
    let registry = ProviderRegistry::from_providers(vec![provider(
        endpoint,
        Some(llm_proxy_core::ProviderConcurrencyConfig {
            total: Some(1),
            models: HashMap::new(),
        }),
    )])
    .expect("registry");
    let admission = Arc::new(PolicyAdmission::from_config(
        &llm_proxy_core::AuthConfig::default(),
        &registry,
    ));
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::clone(&bus) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("engine");

    let first = tokio::spawn({
        let engine = engine.clone();
        async move {
            engine
                .execute(engine_request("anonymous", "claude-sonnet-4-6"))
                .await
                .expect("the held request completes after release")
        }
    });
    // Give the first request a moment to reserve; then the cap is full.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let error = engine
        .execute(distinct_engine_request(
            "anonymous",
            "claude-sonnet-4-6",
            "while-held",
        ))
        .await
        .expect_err("cap of 1 is held");
    match &error {
        EngineError::Overloaded { retry_after, .. } => {
            assert!(
                retry_after.is_some_and(|wait| wait <= Duration::from_secs(5)),
                "Retry-After must be short, got {retry_after:?}"
            );
        }
        other => panic!("expected Overloaded, got {other:?}"),
    }
    assert_eq!(error.status_code(), 503);

    release.send_replace(true);
    first.await.expect("task joins");
    engine
        .execute(distinct_engine_request(
            "anonymous",
            "claude-sonnet-4-6",
            "after-release",
        ))
        .await
        .expect("the slot freed at settlement");
}

#[tokio::test]
async fn a_request_dropped_mid_ladder_releases_its_in_flight_slot() {
    // A client disconnect (or the per-family timeout) drops the non-streaming
    // handler future between reserve and settle. That drop must release the
    // concurrency slot, or a cap of `n` leaked leases answers 503 forever.
    let release = Arc::new(tokio::sync::watch::Sender::new(false));
    let arrivals = Arc::new(AtomicUsize::new(0));
    let endpoint = spawn_upstream(upstream_held_counting(
        Arc::new(release.subscribe()),
        Arc::clone(&arrivals),
    ))
    .await;
    let registry = ProviderRegistry::from_providers(vec![provider(
        endpoint,
        Some(llm_proxy_core::ProviderConcurrencyConfig {
            total: Some(1),
            models: HashMap::new(),
        }),
    )])
    .expect("registry");
    let admission = Arc::new(PolicyAdmission::from_config(
        &llm_proxy_core::AuthConfig::default(),
        &registry,
    ));
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("engine");

    // Park one request upstream — past reserve, holding the only slot.
    let dropped = tokio::spawn({
        let engine = engine.clone();
        async move {
            engine
                .execute(engine_request("anonymous", "claude-sonnet-4-6"))
                .await
        }
    });
    await_arrivals(&arrivals, 1).await;
    dropped.abort();
    dropped.await.expect_err("the dropped task was aborted");
    // Let the orphaned upstream handler finish; the slot it gated is gone
    // with the dropped caller.
    release.send_replace(true);

    engine
        .execute(distinct_engine_request(
            "anonymous",
            "claude-sonnet-4-6",
            "after-drop",
        ))
        .await
        .expect("the dropped request's slot must be released, not leaked");
}

/// A held upstream that counts arrivals before parking, so a test can wait
/// until a request is genuinely in flight there (holding its slot).
fn upstream_held_counting(
    release: Arc<tokio::sync::watch::Receiver<bool>>,
    arrivals: Arc<AtomicUsize>,
) -> Router {
    Router::new().route(
        "/{*path}",
        post(move || {
            let mut release = release.as_ref().clone();
            let arrivals = Arc::clone(&arrivals);
            async move {
                arrivals.fetch_add(1, Ordering::SeqCst);
                while !*release.borrow() {
                    if release.changed().await.is_err() {
                        break;
                    }
                }
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hello"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
                )
            }
        }),
    )
}

fn pool_member(
    name: &str,
    endpoint: String,
    concurrency: Option<ProviderConcurrencyConfig>,
) -> ProviderConfig {
    let mut config = provider(endpoint, concurrency);
    config.name = name.to_owned();
    config
}

fn pool_engine_request(route: &str, request_id: &str, stream: bool) -> EngineRequest {
    // Each request id is a distinct request (plan 06: identical concurrent
    // requests from one caller would share one upstream call instead of
    // exercising the failover ladder).
    let mut core = core_request("claude-sonnet-4-6");
    if let Some(message) = core.messages.first_mut()
        && let Some(CoreContent::Text { text, .. }) = message.content.first_mut()
    {
        *text = format!("hello {request_id}");
    }
    core.stream = stream;
    EngineRequest::new(
        route,
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core,
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id(request_id)
    .with_caller("anonymous")
    .with_body_hash(format!("hash-{request_id}"))
}

/// A pool whose first member always 500s (every request fails over) and
/// whose spare is capped at one in-flight request, parked on a gated
/// upstream.
async fn capped_spare_pool() -> (
    llm_proxy_engine::Engine,
    Arc<tokio::sync::watch::Sender<bool>>,
    Arc<AtomicUsize>,
) {
    let failing = spawn_upstream(upstream_always_500()).await;
    let release = Arc::new(tokio::sync::watch::Sender::new(false));
    let arrivals = Arc::new(AtomicUsize::new(0));
    let held = spawn_upstream(upstream_held_counting(
        Arc::new(release.subscribe()),
        Arc::clone(&arrivals),
    ))
    .await;
    let registry = ProviderRegistry::from_parts(
        vec![
            pool_member("flaky", failing, None),
            pool_member(
                "spare",
                held,
                Some(ProviderConcurrencyConfig {
                    total: Some(1),
                    models: HashMap::new(),
                }),
            ),
        ],
        vec![PoolConfig {
            name: "pooled".to_owned(),
            members: vec!["flaky".to_owned(), "spare".to_owned()],
            retry: RetryConfig {
                max_attempts: 1,
                backoff: Duration::from_millis(5),
                cooldown: Duration::ZERO,
                total_timeout: Duration::ZERO,
                hold: Duration::ZERO,
            },
            affinity: Default::default(),
            health: Default::default(),
        }],
    )
    .expect("registry with pool");
    let admission = Arc::new(PolicyAdmission::from_config(
        &llm_proxy_core::AuthConfig::default(),
        &registry,
    ));
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("engine");
    (engine, release, arrivals)
}

/// Waits until `arrivals` requests have parked on the held upstream.
async fn await_arrivals(arrivals: &Arc<AtomicUsize>, count: usize) {
    for _ in 0..400 {
        if arrivals.load(Ordering::SeqCst) >= count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        arrivals.load(Ordering::SeqCst),
        count,
        "the held upstream never saw the request"
    );
}

#[tokio::test]
async fn a_pool_failover_respects_the_serving_members_concurrency_cap() {
    let (engine, release, arrivals) = capped_spare_pool().await;

    // Request one fails over to the spare and parks there, holding its
    // only in-flight slot.
    let first = tokio::spawn({
        let engine = engine.clone();
        async move {
            engine
                .execute(pool_engine_request("pooled", "one", false))
                .await
                .expect("held request completes after release")
        }
    });
    await_arrivals(&arrivals, 1).await;

    // Request two also fails over; the spare's cap — not the first
    // member's key the reservation started on — must refuse it, instead of
    // letting it park unsynchronized on the capped member.
    let second = tokio::time::timeout(
        Duration::from_secs(5),
        engine.execute(pool_engine_request("pooled", "two", false)),
    )
    .await
    .expect("the capped failover refuses fast instead of parking on the held upstream")
    .expect_err("the spare is at its in-flight cap");
    let EngineError::RetryExhausted { last, .. } = &second else {
        panic!("expected an exhausted ladder, got {second:?}");
    };
    assert!(
        matches!(**last, EngineError::Overloaded { .. }),
        "the ladder's last error is the saturated cap, got {last:?}"
    );
    assert_eq!(second.status_code(), 503);

    release.send_replace(true);
    first.await.expect("held request joins");
    engine
        .execute(pool_engine_request("pooled", "three", false))
        .await
        .expect("the slot freed at settlement");
}

#[tokio::test]
async fn a_streaming_pool_failover_respects_the_serving_members_concurrency_cap() {
    let (engine, release, arrivals) = capped_spare_pool().await;

    // A non-streaming request holds the spare's only slot.
    let first = tokio::spawn({
        let engine = engine.clone();
        async move {
            engine
                .execute(pool_engine_request("pooled", "one", false))
                .await
                .expect("held request completes after release")
        }
    });
    await_arrivals(&arrivals, 1).await;

    // A streaming request that fails over must meet the same cap before
    // any byte is committed.
    let error = engine
        .execute_stream(pool_engine_request("pooled", "two", true))
        .await
        .expect_err("the spare is at its in-flight cap");
    let EngineError::RetryExhausted { last, .. } = &error else {
        panic!("expected an exhausted ladder, got {error:?}");
    };
    assert!(
        matches!(**last, EngineError::Overloaded { .. }),
        "the ladder's last error is the saturated cap, got {last:?}"
    );
    assert_eq!(error.status_code(), 503);

    release.send_replace(true);
    first.await.expect("held request joins");
}

#[tokio::test]
async fn per_key_allowlists_are_enforced_through_the_engine() {
    let ok = spawn_upstream(upstream_always_ok()).await;
    let registry = ProviderRegistry::from_providers(vec![provider(ok, None)]).expect("registry");
    let mut keys = HashMap::new();
    keys.insert(
        "scoped".to_owned(),
        KeyPolicy {
            allow_models: vec!["claude-*".to_owned()],
            allow_providers: vec!["fake".to_owned()],
            ..KeyPolicy::default()
        },
    );
    let admission = Arc::new(PolicyAdmission::new(keys, HashMap::new(), HashMap::new()));
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .build()
        .expect("engine");

    engine
        .execute(engine_request("scoped", "claude-sonnet-4-6"))
        .await
        .expect("matching model and provider");

    let error = engine
        .execute(engine_request("scoped", "gpt-4o"))
        .await
        .expect_err("model outside the allowlist");
    assert!(matches!(error, EngineError::Forbidden(_)));
    assert_eq!(error.status_code(), 403);

    // Unlisted callers stay unrestricted.
    engine
        .execute(engine_request("someone-else", "gpt-4o"))
        .await
        .expect("keys without a policy are unrestricted");
}

/// An admission that counts reservations, for the cache-hit rule.
#[derive(Debug, Default)]
struct CountingAdmission {
    reserves: AtomicUsize,
    settles: AtomicUsize,
}

#[async_trait::async_trait]
impl Admission for CountingAdmission {
    async fn reserve(&self, _claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        self.reserves.fetch_add(1, Ordering::SeqCst);
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, _outcome: &AdmissionOutcome<'_>) {
        self.settles.fetch_add(1, Ordering::SeqCst);
    }
}

/// A cache that answers every lookup with one stored entry.
#[derive(Debug)]
struct AlwaysHitCache {
    entry: CacheEntry,
}

#[async_trait::async_trait]
impl Cache for AlwaysHitCache {
    async fn lookup(&self, _key: &CacheKey<'_>) -> Result<Option<CacheEntry>, EngineError> {
        Ok(Some(self.entry.clone()))
    }
}

#[tokio::test]
async fn cache_hits_never_reserve_and_are_recorded_as_hits() {
    let ok = spawn_upstream(upstream_always_ok()).await;
    let bus = Arc::new(RecordingBus::new());
    let registry = ProviderRegistry::from_providers(vec![provider(ok, None)]).expect("registry");

    // Prime the cache entry from one real request's shape.
    let entry = {
        let engine = EngineBuilder::new(test_config())
            .providers(Arc::new(
                ProviderRegistry::from_providers(vec![provider(
                    spawn_upstream(upstream_always_ok()).await,
                    None,
                )])
                .expect("registry"),
            ))
            .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
            .build()
            .expect("engine");
        let response = engine
            .execute(engine_request("anonymous", "claude-sonnet-4-6"))
            .await
            .expect("prime");
        CacheEntry {
            response: response.core,
            events: Vec::new(),
        }
    };

    let admission = Arc::new(CountingAdmission::default());
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::clone(&bus) as Arc<dyn EventBus>)
        .admission(Arc::clone(&admission) as Arc<dyn Admission>)
        .cache(Arc::new(AlwaysHitCache { entry }))
        .build()
        .expect("engine");

    let response = engine
        .execute(engine_request("anonymous", "claude-sonnet-4-6"))
        .await
        .expect("served from the cache");
    assert_eq!(response.core.usage.input_tokens, 10);
    assert_eq!(
        admission.reserves.load(Ordering::SeqCst),
        0,
        "a cache hit must not reserve: it charges nothing"
    );
    assert_eq!(admission.settles.load(Ordering::SeqCst), 0);
    let hits = bus
        .snapshot()
        .iter()
        .filter(|event| match event {
            ProxyEvent::ResponseCompleted(done) => done.cache_hit,
            _ => false,
        })
        .count();
    assert_eq!(hits, 1, "the hit is recorded as a hit on its ledger row");
}

/// Records every reservation and settlement the engine hands to admission:
/// what a budget would be charged.
#[derive(Debug, Default)]
struct SettlementLog {
    reserves: AtomicUsize,
    settled: std::sync::Mutex<Vec<(bool, Option<Usage>, Option<Cost>)>>,
}

#[async_trait]
impl Admission for SettlementLog {
    async fn reserve(&self, _claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        self.reserves.fetch_add(1, Ordering::SeqCst);
        Ok(AdmissionTicket::default())
    }

    fn settle(&self, _ticket: &AdmissionTicket, outcome: &AdmissionOutcome<'_>) {
        self.settled.lock().expect("settlement log").push((
            outcome.succeeded,
            outcome.usage.cloned(),
            outcome.cost.cloned(),
        ));
    }
}

/// A post-call guard that refuses every response and every streamed event.
#[derive(Debug)]
struct RefuseEverything;

#[async_trait]
impl Guard for RefuseEverything {
    async fn inspect_response(
        &self,
        _response: &CoreResponse,
    ) -> Result<GuardDecision, EngineError> {
        Ok(GuardDecision::Reject(
            "refused by the test guard".to_owned(),
        ))
    }

    async fn inspect_event(&self, _event: &CoreEvent) -> Result<GuardDecision, EngineError> {
        Ok(GuardDecision::Reject(
            "refused by the test guard".to_owned(),
        ))
    }
}

/// A complete Anthropic SSE response (7 in, 1 out) sent as one body, so the
/// whole stream lands in the pre-commit head the guards vet.
fn upstream_stream_in_one_body() -> Router {
    Router::new().route(
        "/{*path}",
        post(|| async {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                concat!(
                    "event: message_start\n",
                    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_g\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n",
                    "event: content_block_delta\n",
                    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
                    "event: message_delta\n",
                    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
                    "event: message_stop\n",
                    "data: {\"type\":\"message_stop\"}\n\n",
                ),
            )
        }),
    )
}

fn guarded_engine(endpoint: String) -> (llm_proxy_engine::Engine, Arc<SettlementLog>) {
    let registry =
        ProviderRegistry::from_providers(vec![provider(endpoint, None)]).expect("registry");
    let log = Arc::new(SettlementLog::default());
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&log) as Arc<dyn Admission>)
        .guard(Arc::new(RefuseEverything))
        .build()
        .expect("engine");
    (engine, log)
}

/// The one settlement the log holds, with the usage and cost it carried.
fn only_settlement(log: &SettlementLog) -> (bool, Option<Usage>, Option<Cost>) {
    let settled = log.settled.lock().expect("settlement log").clone();
    let [only] = settled.as_slice() else {
        panic!("exactly one settlement per request, got {settled:?}");
    };
    only.clone()
}

#[tokio::test]
async fn a_guard_rejected_response_still_charges_what_the_provider_billed() {
    let ok = spawn_upstream(upstream_always_ok()).await;
    let (engine, log) = guarded_engine(ok);

    let error = engine
        .execute(pool_engine_request("fake", "guarded", false))
        .await
        .expect_err("the guard refuses the response");
    assert_eq!(
        error.status_code(),
        400,
        "a guard rejection is the proxy's own 400, got {error:?}"
    );

    // The provider answered and billed 10 in / 4 out before the proxy
    // refused it: that usage, priced at the member's rate, reaches
    // settlement — budgets charge what providers bill, per attempt.
    let (succeeded, usage, cost) = only_settlement(&log);
    assert!(!succeeded, "the request failed from the client's view");
    let usage = usage.expect("the billed usage reaches settlement");
    assert_eq!((usage.input_tokens, usage.output_tokens), (10, 4));
    assert_eq!(
        cost.map(|cost| cost.total.normalize().to_string())
            .as_deref(),
        Some("0.0014"),
        "priced at the member's 0.0001 per token"
    );
}

#[tokio::test]
async fn a_guard_rejected_stream_prefix_still_charges_what_the_provider_billed() {
    let stream = spawn_upstream(upstream_stream_in_one_body()).await;
    let (engine, log) = guarded_engine(stream);

    let error = engine
        .execute_stream(pool_engine_request("fake", "guarded-stream", true))
        .await
        .expect_err("the guard refuses the buffered prefix");
    assert_eq!(
        error.status_code(),
        400,
        "a guard rejection before commit is the proxy's own 400, got {error:?}"
    );

    let (succeeded, usage, cost) = only_settlement(&log);
    assert!(!succeeded);
    let usage = usage.expect("the usage the rejected prefix reported reaches settlement");
    assert_eq!((usage.input_tokens, usage.output_tokens), (7, 1));
    assert_eq!(
        cost.map(|cost| cost.total.normalize().to_string())
            .as_deref(),
        Some("0.0008"),
        "priced at the member's 0.0001 per token"
    );
}

/// A pre-call guard that refuses any request whose text carries the marker
/// (mirrors `BlockSecretText` in `open_provider_set.rs`).
#[derive(Debug)]
struct RejectMarkedRequest;

#[async_trait]
impl Guard for RejectMarkedRequest {
    async fn inspect_request(&self, request: &CoreRequest) -> Result<GuardDecision, EngineError> {
        let marked = request.messages.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(content, CoreContent::Text { text, .. } if text.contains("forbidden-marker"))
            })
        });
        if marked {
            Ok(GuardDecision::Reject(
                "request rejected by test guard".to_owned(),
            ))
        } else {
            Ok(GuardDecision::Allow)
        }
    }
}

fn marked_engine_request(request_id: &str, stream: bool) -> EngineRequest {
    let mut core = core_request("claude-sonnet-4-6");
    if let Some(message) = core.messages.first_mut()
        && let Some(CoreContent::Text { text, .. }) = message.content.first_mut()
    {
        *text = "hello forbidden-marker".to_owned();
    }
    core.stream = stream;
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        core,
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id(request_id)
    .with_caller("anonymous")
    .with_body_hash(format!("hash-{request_id}"))
}

#[tokio::test]
async fn a_guard_refused_stream_reserves_then_refunds() {
    // Behaviour parity with the non-streaming path: the pre-call guards run
    // after the budget reservation, so a guard-refused request counts toward
    // rpm/rpd exactly as a guard-refused non-streaming one does today — and
    // the reservation is refunded with no usage charged.
    let ok = spawn_upstream(upstream_always_ok()).await;
    let registry = ProviderRegistry::from_providers(vec![provider(ok, None)]).expect("registry");
    let log = Arc::new(SettlementLog::default());
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .event_bus(Arc::new(RecordingBus::new()) as Arc<dyn EventBus>)
        .admission(Arc::clone(&log) as Arc<dyn Admission>)
        .guard(Arc::new(RejectMarkedRequest) as Arc<dyn Guard>)
        .build()
        .expect("engine");

    let error = engine
        .execute_stream(marked_engine_request("marked-stream", true))
        .await
        .expect_err("the guard refuses the request");
    assert!(
        matches!(error, EngineError::InvalidRequest(ref message) if message.contains("guard")),
        "got {error:?}"
    );
    assert_eq!(
        log.reserves.load(Ordering::SeqCst),
        1,
        "the stream reserves before the guards refuse it"
    );
    assert_eq!(
        log.settled.lock().expect("settlement log").as_slice(),
        [(false, None, None)],
        "the refusal refunds its reservation with no usage charged"
    );

    // Parity: a non-streaming refusal produces the identical log.
    log.reserves.store(0, Ordering::SeqCst);
    log.settled.lock().expect("settlement log").clear();
    let error = engine
        .execute(marked_engine_request("marked", false))
        .await
        .expect_err("the guard refuses the request");
    assert!(matches!(error, EngineError::InvalidRequest(_)));
    assert_eq!(log.reserves.load(Ordering::SeqCst), 1);
    assert_eq!(
        log.settled.lock().expect("settlement log").as_slice(),
        [(false, None, None)]
    );
}
