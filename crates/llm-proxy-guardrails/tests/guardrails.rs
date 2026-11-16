//! Guardrails end to end (plan 09 step 4): the wrapper's mode and fail
//! policy are observable through the engine's [`Guard`] trait and through
//! a real engine against a loopback fake upstream — the masked request
//! reaches the upstream and the masked response reaches the client, while
//! a streaming request is refused by a masking guard rather than masked
//! unsafely across chunks.

#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
    ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_engine::{
    ClientDialect, Engine, EngineBuilder, EngineError, EngineRequest, Guard, GuardDecision,
    InboundRequest,
};
use llm_proxy_guardrails::{
    FailPolicy, GuardMode, GuardRule, GuardVerdict, Guardrail, PatternRule, RuleError,
    request_texts, response_texts,
};
use llm_proxy_protocol::core::{CoreContent, CoreRequest, CoreResponse};
use llm_proxy_storage::{EventBus, NoopBus};
use tokio::sync::Mutex;

fn core_request(stream: bool, text: &str) -> CoreRequest {
    CoreRequest {
        model: llm_proxy_protocol::core::ModelRef {
            requested: "claude-sonnet-4-6".to_owned(),
            upstream: None,
        },
        system: vec![CoreContent::Text {
            text: "You are a coding agent.".to_owned(),
            cache: None,
        }],
        messages: vec![llm_proxy_protocol::core::CoreMessage {
            role: llm_proxy_protocol::core::CoreRole::User,
            content: vec![CoreContent::Text {
                text: text.to_owned(),
                cache: None,
            }],
        }],
        tools: Vec::new(),
        tool_choice: None,
        sampling: Default::default(),
        stream,
        metadata: Default::default(),
        provider_hints: Default::default(),
    }
}

fn core_response(text: &str) -> CoreResponse {
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

/// A non-matching rule must leave every text block exactly as it was: the
/// mask walk hands each text to the rule and a "no match" answer means
/// "leave this block alone", never "empty it".
#[tokio::test]
async fn a_non_matching_rule_leaves_request_and_response_texts_intact() {
    // Response-scoped mask: the request is out of scope entirely, and the
    // response's clean block must survive alongside the masked one.
    let rule = PatternRule::new(
        r"sk-[a-z0-9]{8,}",
        llm_proxy_guardrails::RuleAction::Mask {
            placeholder: "[REDACTED]".to_owned(),
        },
        llm_proxy_guardrails::RuleScope::Responses,
        "response-only masking",
    )
    .expect("rule");
    let guard = Arc::new(Guardrail::masking(Arc::new(rule), FailPolicy::Closed));

    let mut request = core_request(false, "rotate sk-clientsecret77 now");
    guard
        .transform_request(&mut request)
        .await
        .expect("transform");
    assert_eq!(
        request.messages[0].content[0],
        CoreContent::Text {
            text: "rotate sk-clientsecret77 now".to_owned(),
            cache: None,
        },
        "a response-scoped rule must not touch the request at all"
    );

    let mut response = core_response("clean text next to sk-upstreamsecret99");
    response.content.insert(
        0,
        CoreContent::Text {
            text: "entirely clean".to_owned(),
            cache: None,
        },
    );
    guard
        .transform_response(&mut response)
        .await
        .expect("transform");
    match (response.content.first(), response.content.get(1)) {
        (
            Some(CoreContent::Text {
                text: untouched, ..
            }),
            Some(CoreContent::Text { text: masked, .. }),
        ) => {
            assert_eq!(untouched, "entirely clean", "no match means no change");
            assert!(!masked.contains("sk-upstreamsecret99"), "got {masked}");
            assert!(masked.contains("[REDACTED]"), "got {masked}");
        }
        other => panic!("two text blocks expected, got {other:?}"),
    }
}

/// A rule whose checker always fails, to observe the fail policy.
#[derive(Debug)]
struct FailingRule;

#[async_trait]
impl GuardRule for FailingRule {
    async fn check_request_text(&self, _text: &str) -> Result<GuardVerdict, RuleError> {
        Err(RuleError("checker unreachable".to_owned()))
    }

    async fn check_response_text(&self, _text: &str) -> Result<GuardVerdict, RuleError> {
        Err(RuleError("checker unreachable".to_owned()))
    }
}

/// A rule whose checker breaks on its very first call and masks
/// everything afterwards: the flaky remote checker the fail policy
/// exists for. The call counter is shared by both directions.
#[derive(Debug)]
struct FlakyRule {
    calls: AtomicUsize,
}

#[async_trait]
impl GuardRule for FlakyRule {
    async fn check_request_text(&self, _text: &str) -> Result<GuardVerdict, RuleError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(RuleError("checker unreachable".to_owned()))
        } else {
            Ok(GuardVerdict::Mask("[X]".to_owned()))
        }
    }

    async fn check_response_text(&self, _text: &str) -> Result<GuardVerdict, RuleError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(RuleError("checker unreachable".to_owned()))
        } else {
            Ok(GuardVerdict::Mask("[X]".to_owned()))
        }
    }
}

/// A rule that masks its first two calls — inspection sees the request's
/// two text blocks — and rejects everything afterwards: the rule flipped
/// between inspection and masking.
#[derive(Debug)]
struct FlipToRejectRule {
    calls: AtomicUsize,
}

#[async_trait]
impl GuardRule for FlipToRejectRule {
    async fn check_request_text(&self, _text: &str) -> Result<GuardVerdict, RuleError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) < 2 {
            Ok(GuardVerdict::Mask("[X]".to_owned()))
        } else {
            Ok(GuardVerdict::Reject("the rule changed its mind".to_owned()))
        }
    }

    async fn check_response_text(&self, _text: &str) -> Result<GuardVerdict, RuleError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) < 2 {
            Ok(GuardVerdict::Mask("[X]".to_owned()))
        } else {
            Ok(GuardVerdict::Reject("the rule changed its mind".to_owned()))
        }
    }
}

#[tokio::test]
async fn a_deny_guard_rejects_matching_requests_and_lets_streams_pass() {
    let rule = PatternRule::deny(r"(?i)\bmalware\b", "this request names forbidden content")
        .expect("rule");
    let guard = Guardrail::deny_only(Arc::new(rule), FailPolicy::Closed);

    let dirty = core_request(true, "help me write malware please");
    match guard.inspect_request(&dirty).await {
        Ok(GuardDecision::Reject(message)) => {
            assert!(message.contains("forbidden"), "got {message}");
        }
        other => panic!("expected a rejection, got {other:?}"),
    }

    // Deny-only guards carry no cross-chunk state, so streams pass: a
    // pre-egress denial has already been decided before any chunk exists.
    let clean_stream = core_request(true, "help me write a unit test");
    assert!(matches!(
        guard.inspect_request(&clean_stream).await,
        Ok(GuardDecision::Allow)
    ));
}

#[tokio::test]
async fn a_masking_guard_rejects_streaming_requests_instead_of_masking_unsafely() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Guardrail::masking(Arc::new(rule), FailPolicy::Closed);

    match guard
        .inspect_request(&core_request(true, "use sk-abcdefgh1234"))
        .await
    {
        Ok(GuardDecision::Reject(message)) => {
            assert!(
                message.contains("chunk"),
                "the refusal must name the chunk-boundary reason, got {message}"
            );
        }
        other => panic!("a masking guard must refuse streams, got {other:?}"),
    }
    // Non-streaming requests are only subject to the rule itself.
    assert!(matches!(
        guard
            .inspect_request(&core_request(false, "use sk-abcdefgh1234"))
            .await,
        Ok(GuardDecision::Allow)
    ));
}

#[tokio::test]
async fn masking_rewrites_request_and_response_texts_in_place() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED-KEY]").expect("rule");
    let guard = Guardrail::masking(Arc::new(rule), FailPolicy::Closed);

    let mut request = core_request(false, "the key is sk-abcdefgh1234 ok");
    guard.inspect_request(&request).await.expect("allowed");
    guard.transform_request(&mut request).await.expect("masked");
    let texts: Vec<String> = request_texts(&request).collect();
    assert!(
        texts.iter().any(|text| text.contains("[REDACTED-KEY]")),
        "the placeholder replaced the secret, got {texts:?}"
    );
    assert!(
        texts.iter().all(|text| !text.contains("sk-abcdefgh1234")),
        "the secret is gone from every text block"
    );

    let mut response = core_response("echoing sk-abcdefgh1234 back");
    guard.inspect_response(&response).await.expect("allowed");
    guard
        .transform_response(&mut response)
        .await
        .expect("masked");
    match response.content.first() {
        Some(CoreContent::Text { text, .. }) => {
            assert!(text.contains("[REDACTED-KEY]"));
            assert!(!text.contains("sk-abcdefgh1234"));
        }
        other => panic!("text block expected, got {other:?}"),
    }
}

#[tokio::test]
async fn fail_open_serves_and_fail_closed_refuses_when_the_rule_breaks() {
    let open = Guardrail::masking(Arc::new(FailingRule), FailPolicy::Open);
    assert!(matches!(
        open.inspect_request(&core_request(false, "anything")).await,
        Ok(GuardDecision::Allow)
    ));

    let closed = Guardrail::masking(Arc::new(FailingRule), FailPolicy::Closed);
    assert!(matches!(
        closed
            .inspect_request(&core_request(false, "anything"))
            .await,
        Err(EngineError::Internal(_))
    ));
}

/// The fail policy must govern the masking pass too, not just
/// inspection: a broken rule under `Closed` refuses the transform and
/// leaves the text exactly as it arrived, never half-redacted.
#[tokio::test]
async fn fail_closed_refuses_when_the_rule_breaks_during_masking() {
    let guard = Guardrail::masking(Arc::new(FailingRule), FailPolicy::Closed);

    let mut request = core_request(false, "the key is sk-abcdefgh1234 ok");
    let request_before = request_texts(&request).collect::<Vec<_>>();
    let error = guard
        .transform_request(&mut request)
        .await
        .expect_err("closed masking must refuse when the rule breaks");
    assert!(
        matches!(&error, EngineError::Internal(message) if message.contains("failed closed")),
        "got {error:?}"
    );
    assert_eq!(
        request_texts(&request).collect::<Vec<_>>(),
        request_before,
        "the refused transform must leave the request text untouched"
    );

    let mut response = core_response("echoing sk-abcdefgh1234 back");
    let response_before = response_texts(&response).collect::<Vec<_>>();
    guard
        .transform_response(&mut response)
        .await
        .expect_err("closed masking must refuse when the rule breaks");
    assert_eq!(
        response_texts(&response).collect::<Vec<_>>(),
        response_before,
        "the refused transform must leave the response text untouched"
    );
}

/// Under `Open` a masking-pass failure keeps serving: the texts pass
/// unchanged (the rule answered nothing usable) and the transform still
/// reports success.
#[tokio::test]
async fn fail_open_leaves_the_text_when_masking_breaks() {
    let guard = Guardrail::masking(Arc::new(FailingRule), FailPolicy::Open);

    let mut request = core_request(false, "the key is sk-abcdefgh1234 ok");
    let request_before = request_texts(&request).collect::<Vec<_>>();
    guard
        .transform_request(&mut request)
        .await
        .expect("open masking serves");
    assert_eq!(
        request_texts(&request).collect::<Vec<_>>(),
        request_before,
        "open masking must not invent a redaction it cannot compute"
    );

    let mut response = core_response("echoing sk-abcdefgh1234 back");
    let response_before = response_texts(&response).collect::<Vec<_>>();
    guard
        .transform_response(&mut response)
        .await
        .expect("open masking serves");
    assert_eq!(
        response_texts(&response).collect::<Vec<_>>(),
        response_before
    );
}

/// A rule that fails once under `Closed` must never send unredacted text
/// anywhere: inspection spends the failure and refuses, and the same
/// flow through the engine ends with the guard error and zero upstream
/// requests.
#[tokio::test]
async fn a_rule_that_fails_once_under_fail_closed_never_leaks_unmasked_text() {
    let guard = Guardrail::masking(
        Arc::new(FlakyRule {
            calls: AtomicUsize::new(0),
        }),
        FailPolicy::Closed,
    );
    assert!(
        matches!(
            guard
                .inspect_request(&core_request(false, "rotate sk-clientsecret77 now"))
                .await,
            Err(EngineError::Internal(_))
        ),
        "the flaky rule's single failure must refuse the request under fail closed"
    );

    let guard = Arc::new(Guardrail::masking(
        Arc::new(FlakyRule {
            calls: AtomicUsize::new(0),
        }),
        FailPolicy::Closed,
    ));
    let (engine, _endpoint, captured) = engine_with(guard).await;
    let error = engine
        .execute(engine_request(core_request(
            false,
            "rotate sk-clientsecret77 now",
        )))
        .await
        .expect_err("fail closed must refuse the whole flow");
    assert!(
        matches!(&error, EngineError::Internal(message) if message.contains("failed closed")),
        "got {error:?}"
    );
    assert!(
        captured.lock().await.is_empty(),
        "the refused request never reached the upstream"
    );
}

/// A `Reject` verdict that surfaces only on the masking pass — the rule
/// flipped since inspection — must not quietly pass the text: `Closed`
/// refuses, `Open` keeps serving with the text unchanged.
#[tokio::test]
async fn a_verdict_flip_to_reject_during_masking_is_decided_by_the_fail_policy() {
    let closed = Guardrail::masking(
        Arc::new(FlipToRejectRule {
            calls: AtomicUsize::new(0),
        }),
        FailPolicy::Closed,
    );
    let mut request = core_request(false, "anything");
    closed
        .inspect_request(&request)
        .await
        .expect("inspection masked the texts");
    let error = closed
        .transform_request(&mut request)
        .await
        .expect_err("a flipped rejection must refuse under fail closed");
    assert!(
        matches!(&error, EngineError::Internal(message) if message.contains("verdict changed")),
        "got {error:?}"
    );

    let open = Guardrail::masking(
        Arc::new(FlipToRejectRule {
            calls: AtomicUsize::new(0),
        }),
        FailPolicy::Open,
    );
    let mut request = core_request(false, "anything");
    open.inspect_request(&request)
        .await
        .expect("inspection masked the texts");
    open.transform_request(&mut request)
        .await
        .expect("open keeps serving");
    assert_eq!(
        request_texts(&request).collect::<Vec<_>>(),
        request_texts(&core_request(false, "anything")).collect::<Vec<_>>(),
        "open leaves the text unchanged when the verdict flips"
    );
}

#[tokio::test]
async fn modes_declare_what_a_guard_sees() {
    let post_hoc = Guardrail::post_hoc(
        Arc::new(PatternRule::deny("never", "never").expect("rule")),
        FailPolicy::Closed,
    );
    // A PostHoc guard never judges requests, even matching ones.
    assert!(matches!(
        post_hoc
            .inspect_request(&core_request(false, "never"))
            .await,
        Ok(GuardDecision::Allow)
    ));
    assert!(matches!(post_hoc.mode(), GuardMode::PostHoc));
}

/// A tracing subscriber that collects warn-level event messages, so the
/// fail-open path's warning is observable in-process. It installs as the
/// process-global default on purpose: warn events are gated by a
/// process-global per-callsite interest cache, and a subscriber that is
/// only thread-local loses to sibling tests firing the same callsite
/// without one (their first registration can cache `Interest::never`,
/// which silences the event for every later observer). Stray sibling
/// events may land in the capture; assertions read the events emitted
/// after the test installed the subscriber.
#[derive(Debug, Default, Clone)]
struct WarnCapture {
    messages: Arc<std::sync::Mutex<Vec<String>>>,
}

struct WarnMessage {
    message: Option<String>,
}

impl tracing::field::Visit for WarnMessage {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{value:?}"));
        }
    }
}

impl tracing::Subscriber for WarnCapture {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _attrs: &tracing::span::Attributes<'_>) -> tracing::Id {
        tracing::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::Id, _follows: &tracing::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        let mut visitor = WarnMessage { message: None };
        event.record(&mut visitor);
        if let Some(message) = visitor.message {
            self.messages.lock().expect("capture lock").push(message);
        }
    }

    fn enter(&self, _span: &tracing::Id) {}

    fn exit(&self, _span: &tracing::Id) {}

    fn clone_span(&self, id: &tracing::Id) -> tracing::Id {
        id.clone()
    }
}

/// A mask rule wired into a deny-only guard is a misconfiguration: the
/// guard can never redact, so under `FailPolicy::Closed` the matching
/// request is refused instead of passing unredacted with no signal.
#[tokio::test]
async fn a_mask_rule_in_a_deny_only_guard_is_refused_under_fail_closed() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Guardrail::deny_only(Arc::new(rule), FailPolicy::Closed);

    match guard
        .inspect_request(&core_request(false, "rotate sk-clientsecret77 now"))
        .await
    {
        Err(EngineError::Internal(message)) => {
            assert!(
                message.contains("non-masking guard"),
                "the refusal must name the misconfiguration, got {message}"
            );
        }
        other => {
            panic!("a mask rule in a deny-only guard must refuse under fail closed, got {other:?}")
        }
    }
}

/// The same misconfiguration under `FailPolicy::Open` keeps serving — but
/// logs a warning, never a silent pass-through of text the rule wanted
/// redacted.
#[tokio::test]
async fn a_mask_rule_in_a_deny_only_guard_is_allowed_with_a_warning_under_fail_open() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Guardrail::deny_only(Arc::new(rule), FailPolicy::Open);
    let capture = WarnCapture::default();
    let reader = capture.clone();
    tracing::subscriber::set_global_default(capture).expect("global warn capture");
    let observed = reader.messages.lock().expect("capture lock").len();

    let decision = guard
        .inspect_request(&core_request(false, "rotate sk-clientsecret77 now"))
        .await;
    assert!(
        matches!(decision, Ok(GuardDecision::Allow)),
        "fail open must serve the request, got {decision:?}"
    );
    let messages = reader.messages.lock().expect("capture lock").clone()[observed..].to_vec();
    assert!(
        messages
            .iter()
            .any(|message| message.contains("failing open")),
        "the misconfigured mask verdict must be logged, got {messages:?}"
    );
}

/// A post-hoc guard with a mask rule masks the completed response — the
/// response is finished, so masking is safe — and never touches requests.
#[tokio::test]
async fn a_post_hoc_masking_guard_masks_the_response() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Guardrail::post_hoc(Arc::new(rule), FailPolicy::Closed);

    let mut response = core_response("the upstream saw sk-upstreamsecret99");
    guard.inspect_response(&response).await.expect("allowed");
    guard
        .transform_response(&mut response)
        .await
        .expect("masked");
    match response.content.first() {
        Some(CoreContent::Text { text, .. }) => {
            assert!(!text.contains("sk-upstreamsecret99"), "got {text}");
            assert!(text.contains("[REDACTED]"), "got {text}");
        }
        other => panic!("text block expected, got {other:?}"),
    }

    let mut request = core_request(false, "rotate sk-clientsecret77 now");
    guard
        .transform_request(&mut request)
        .await
        .expect("untouched");
    assert_eq!(
        request_texts(&request).collect::<Vec<_>>(),
        request_texts(&core_request(false, "rotate sk-clientsecret77 now")).collect::<Vec<_>>(),
        "a post-hoc guard never touches the request"
    );
}

// ---------------------------------------------------------------------------
// Through a real engine against a loopback upstream.

fn test_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind"),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "guardrails-test".to_owned(),
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

/// A fake upstream that records the request body and echoes a response
/// containing a secret, so both masking directions are observable.
async fn spawn_upstream() -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
    let captured: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    let app = Router::new().route(
        "/{*path}",
        axum::routing::post(move |body: String| {
            let sink = Arc::clone(&sink);
            async move {
                let value: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                sink.lock().await.push(value);
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"id":"msg_h","type":"message","role":"assistant","content":[{"type":"text","text":"the upstream saw sk-upstreamsecret99"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
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
    (format!("http://{addr}/v1/messages"), captured)
}

async fn engine_with(
    guard: Arc<dyn Guard>,
) -> (Engine, String, Arc<Mutex<Vec<serde_json::Value>>>) {
    let (endpoint, captured) = spawn_upstream().await;
    let mut adapters = std::collections::HashMap::new();
    adapters.insert(
        "messages".to_owned(),
        ProviderAdapterConfig {
            protocol: "anthropic_messages".to_owned(),
            endpoint: endpoint.clone(),
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
    let engine = EngineBuilder::new(test_config())
        .providers(Arc::new(registry))
        .guard(guard)
        .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
        .build()
        .expect("engine builds");
    (engine, endpoint, captured)
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
    .with_request_id("req-g")
    .with_caller("caller-g")
    .with_body_hash("hash-g")
}

/// The done-when of the masking path: the secret never reaches the
/// upstream, and the upstream's secret never reaches the client.
#[tokio::test]
async fn the_engine_masks_both_directions_against_a_live_upstream() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Arc::new(Guardrail::masking(Arc::new(rule), FailPolicy::Closed));
    let (engine, _endpoint, captured) = engine_with(guard).await;

    let response = engine
        .execute(engine_request(core_request(
            false,
            "rotate sk-clientsecret77 now",
        )))
        .await
        .expect("request served");
    let upstream_body = captured.lock().await.pop().expect("upstream saw a request");
    let upstream_text = upstream_body.to_string();
    assert!(
        !upstream_text.contains("sk-clientsecret77"),
        "the client's secret never reaches the upstream, got {upstream_text}"
    );
    assert!(
        upstream_text.contains("[REDACTED]"),
        "the placeholder reached the upstream"
    );
    match response.core.content.first() {
        Some(CoreContent::Text { text, .. }) => {
            assert!(!text.contains("sk-upstreamsecret99"), "got {text}");
            assert!(text.contains("[REDACTED]"), "got {text}");
        }
        other => panic!("text block expected, got {other:?}"),
    }
}

/// The pattern appears only in the model's output: the request is clean
/// (so nothing forces a core re-encode) and the same-format passthrough is
/// eligible — it must still be suppressed, or the client would receive the
/// raw upstream bytes with the secret while the cache and ledger hold the
/// masked core.
#[tokio::test]
async fn a_response_only_masking_reaches_the_same_format_client_masked() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Arc::new(Guardrail::masking(Arc::new(rule), FailPolicy::Closed));
    let (engine, _endpoint, _captured) = engine_with(guard).await;

    let response = engine
        .execute(engine_request(core_request(
            false,
            "a perfectly clean prompt",
        )))
        .await
        .expect("request served");
    assert!(
        response.raw_body.is_none(),
        "the same-format raw passthrough must not carry the unmasked upstream bytes, got {:?}",
        response
            .raw_body
            .as_deref()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
    );
    match response.core.content.first() {
        Some(CoreContent::Text { text, .. }) => {
            assert!(!text.contains("sk-upstreamsecret99"), "got {text}");
            assert!(text.contains("[REDACTED]"), "got {text}");
        }
        other => panic!("text block expected, got {other:?}"),
    }
}

/// A denied request never reaches the upstream at all.
#[tokio::test]
async fn the_engine_refuses_a_denied_request_before_egress() {
    let rule = PatternRule::deny(r"\bforbidden-topic\b", "topic is not allowed").expect("rule");
    let guard = Arc::new(Guardrail::deny_only(Arc::new(rule), FailPolicy::Closed));
    let (engine, _endpoint, captured) = engine_with(guard).await;

    let error = engine
        .execute(engine_request(core_request(
            false,
            "tell me about forbidden-topic",
        )))
        .await
        .expect_err("denied");
    assert!(
        matches!(error, EngineError::InvalidRequest(ref message) if message.contains("not allowed")),
        "got {error:?}"
    );
    assert!(
        captured.lock().await.is_empty(),
        "the denied request never reached the upstream"
    );
}

/// The plan's stream rule through the engine: a masking guard refuses a
/// streaming request before egress, naming the chunk reason.
#[tokio::test]
async fn the_engine_refuses_a_streaming_request_under_a_masking_guard() {
    let rule = PatternRule::mask(r"sk-[a-z0-9]{8,}", "[REDACTED]").expect("rule");
    let guard = Arc::new(Guardrail::masking(Arc::new(rule), FailPolicy::Closed));
    let (engine, _endpoint, captured) = engine_with(guard).await;

    let error = engine
        .execute_stream(engine_request(core_request(
            true,
            "stream sk-clientsecret77",
        )))
        .await
        .expect_err("streams are refused under masking guards");
    assert!(
        matches!(&error, EngineError::InvalidRequest(message) if message.contains("chunk")),
        "got {error:?}"
    );
    assert!(
        captured.lock().await.is_empty(),
        "the refused stream never reached the upstream"
    );
}
