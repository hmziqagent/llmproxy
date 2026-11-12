//! The request lifecycle (plan 03's pipeline slots, plan 04's ladder),
//! transport- and HTTP-free: admission, routing, credentials, guards, cache,
//! the attempt loop against one upstream with retries and failover, and
//! settlement. This module was extracted from the axum layer so the server,
//! an SDK embedder and the bindings all run the same pipeline.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use time::OffsetDateTime;

use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use llm_proxy_adapters::client_wire::ClientWireRequest;
use llm_proxy_adapters::request::ProxyRequest;
use llm_proxy_adapters::sse::SseFramer;
use llm_proxy_adapters::{ProtocolName, ProviderStreamDecoder};
use llm_proxy_core::{AffinityConflictPolicy, ModelPricing, ProviderRouteKind};
use llm_proxy_protocol::core::{
    CoreContent, CoreEvent, CoreMessage, CoreRequest, CoreRole, CoreStreamErrorKind, Cost,
    ModelRef, StopReason, Usage, UsageProvenance,
};
use llm_proxy_storage::{
    AttemptOutcome, ProxyEvent, RequestReceived, ResponseCompleted, ResponseFailed,
};
use rust_decimal::Decimal;
use secrecy::SecretString;
use tokio_util::sync::CancellationToken;
use tracing::Instrument as _;

use crate::cache::{
    FlightAcquire, IdemMatch, IdempotencyAcquire, IdempotencyClaim, STREAM_LEADER_LOST,
    StoredOutcome, StreamAcquire, StreamLead, StreamShare, is_flight_leader_lost,
    is_stream_leader_lost,
};
use crate::classification::{ErrorClass, classify, next_wait};
use crate::error::{AuthOwner, EngineError, map_provider_error, sanitize_upstream_error_body};
use crate::headers::select_forwarded_headers;
use crate::traits::{
    AdmissionClaim, AdmissionOutcome, AdmissionTicket, CacheEntry, CacheKey, CredentialFeedback,
    CredentialPick, CredentialQuery, GuardDecision, ResolvedTarget, RoutePlan, RouteQuery,
    UnsettledLeaseGuard, UsageSink,
};
use crate::types::{ClientDialect, EngineRequest, EngineResponse, EngineStream};
use crate::{EngineInner, ProviderAdapterTarget};

/// Client-facing message when the upstream ends a stream without a terminal
/// event; deliberately generic (no upstream body) since it is sent in-band.
const TRUNCATED_STREAM_MESSAGE: &str = "upstream stream ended before the response completed";

/// Client-facing message for a stream that fails to decode after the first
/// byte was committed; generic because it is sent in-band.
const PROVIDER_DECODE_CLIENT_MESSAGE: &str = "provider response decode error";

/// The inbound header the audit's session-attribution evidence uses
/// (backlog 17): the session selector Claude Code sends.
const SESSION_HEADER: &str = "x-claude-code-session-id";

/// Provider hints that survive a failover strip: dialect-neutral options any
/// adapter may consume, unlike account-bound state (signed thinking, file
/// ids) which the strip removes.
const HINTS_KEPT_ACROSS_FAILOVER: &[&str] = &["stream_options"];

fn map_encode_error(e: llm_proxy_adapters::ProviderError) -> EngineError {
    match e {
        llm_proxy_adapters::ProviderError::InvalidRequest(message) => {
            EngineError::InvalidRequest(message)
        }
        other => EngineError::Internal(format!("encode error: {other}")),
    }
}

pub(crate) fn route_kind_str(kind: ProviderRouteKind) -> &'static str {
    match kind {
        ProviderRouteKind::ChatCompletions => "chat_completions",
        ProviderRouteKind::Messages => "messages",
        ProviderRouteKind::Responses => "responses",
        ProviderRouteKind::GenerateContent => "generate_content",
        ProviderRouteKind::Embeddings => "embeddings",
        ProviderRouteKind::Rerank => "rerank",
        ProviderRouteKind::AudioTranscriptions => "audio_transcriptions",
        ProviderRouteKind::AudioSpeech => "audio_speech",
        ProviderRouteKind::ImagesGenerations => "images_generations",
        ProviderRouteKind::Files => "files",
        ProviderRouteKind::Batches => "batches",
    }
}

pub(crate) fn client_dialect_str(dialect: ClientDialect) -> &'static str {
    match dialect {
        ClientDialect::OpenAiChat => "openai_chat",
        ClientDialect::Anthropic => "anthropic",
        ClientDialect::OpenAiResponses => "openai_responses",
        ClientDialect::Gemini => "gemini",
        ClientDialect::Ollama => "ollama",
    }
}

/// The session id attributed to a request (plan 04 step 10): the inbound
/// session header, else the request metadata's `user_id`.
pub(crate) fn session_id_of(req: &EngineRequest) -> Option<String> {
    req.inbound
        .headers
        .get(SESSION_HEADER)
        .cloned()
        .or_else(|| req.core.metadata.user_id.clone())
}

fn compute_cost(usage: &Usage, pricing: &ModelPricing) -> Cost {
    fn priced(count: i32, price: Decimal) -> Decimal {
        Decimal::from(count)
            .checked_mul(price)
            .unwrap_or(Decimal::MAX)
            .round_dp(6)
    }
    fn sum(a: Decimal, b: Decimal) -> Decimal {
        a.checked_add(b).unwrap_or(Decimal::MAX)
    }

    let input = priced(usage.input_tokens, pricing.input);
    let output = priced(usage.output_tokens, pricing.output);
    let cache_creation = priced(
        usage.cache_creation_input_tokens.unwrap_or(0),
        pricing.cache_creation,
    );
    let cache_read = priced(
        usage.cache_read_input_tokens.unwrap_or(0),
        pricing.cache_read,
    );
    let reasoning = priced(usage.reasoning_tokens.unwrap_or(0), pricing.reasoning);
    let total = [input, output, cache_creation, cache_read, reasoning]
        .into_iter()
        .reduce(sum)
        .unwrap_or(Decimal::ZERO);
    Cost {
        input,
        output,
        cache_creation,
        cache_read,
        reasoning,
        total,
    }
}

pub(crate) fn emit_request_received(inner: &EngineInner, req: &EngineRequest, streaming: bool) {
    inner
        .usage
        .record(&ProxyEvent::RequestReceived(RequestReceived {
            request_id: req.request_id.clone(),
            timestamp: time::OffsetDateTime::now_utc(),
            caller: req.caller.clone(),
            provider: req.provider_name.clone(),
            route_kind: route_kind_str(req.route_kind).to_owned(),
            client_protocol: client_dialect_str(req.client_dialect).to_owned(),
            model: req.core.model.clone(),
            streaming,
            body_hash: req.body_hash.clone(),
            session_id: session_id_of(req),
        }));
}

/// Emits a [`ProxyEvent::ResponseFailed`] row for `error`.
pub(crate) fn emit_response_failed(
    inner: &EngineInner,
    request_id: &str,
    provider: Option<&str>,
    model: Option<&ModelRef>,
    error: &EngineError,
    start: Instant,
) {
    let provider = provider.or(match error {
        EngineError::UnknownProvider(name) => Some(name.as_str()),
        _ => None,
    });
    if let EngineError::Upstream {
        status,
        body,
        auth_owner,
        retry_after: _,
    } = error
    {
        tracing::warn!(
            upstream_status = status,
            auth_owner = ?auth_owner,
            upstream_body = %body,
            "upstream error mapped for the client; real body logged here only"
        );
    }
    inner
        .usage
        .record(&ProxyEvent::ResponseFailed(ResponseFailed {
            request_id: request_id.to_owned(),
            timestamp: time::OffsetDateTime::now_utc(),
            provider: provider.map(str::to_owned),
            model: model.cloned(),
            error_kind: error.kind_name().to_owned(),
            message: error.client_message(),
            http_status: error.status_code(),
            latency_ms: start.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        }));
}

/// Emits one [`ProxyEvent::AttemptCompleted`] row (plan 04 step 1): the
/// ledger's per-attempt record, linked to the request.
/// One attempt row's payload: who sent it, what it encoded for, and how it
/// ended.
struct AttemptRecord<'a> {
    request_id: &'a str,
    seq: u32,
    route: &'a str,
    provider: &'a str,
    model: &'a str,
    credential_id: Option<&'a str>,
    outcome: AttemptOutcome,
    error: Option<&'a EngineError>,
    latency: Duration,
}

fn emit_attempt(inner: &EngineInner, record: &AttemptRecord<'_>) {
    inner.usage.record(&ProxyEvent::AttemptCompleted(
        llm_proxy_storage::AttemptCompleted {
            request_id: record.request_id.to_owned(),
            timestamp: time::OffsetDateTime::now_utc(),
            attempt: record.seq,
            route: record.route.to_owned(),
            provider: record.provider.to_owned(),
            model: record.model.to_owned(),
            credential_id: record.credential_id.map(str::to_owned),
            outcome: record.outcome,
            error_kind: record.error.map(|e| e.kind_name().to_owned()),
            http_status: record.error.map(|e| match e {
                EngineError::Upstream { status, .. } => *status,
                other => other.status_code(),
            }),
            message: record.error.map(|e| e.client_message()),
            latency_ms: record.latency.as_millis().try_into().unwrap_or(u64::MAX),
        },
    ));
}

/// One resolved attempt: the target plus the codec and credential that
/// serve it.
struct Attempt {
    target: ProviderAdapterTarget,
    adapter: Arc<dyn llm_proxy_adapters::ProviderAdapter>,
    /// Which pool credential was picked (`key-<i>`), when the member has
    /// more than one key.
    credential_id: Option<String>,
    /// Whether the member forwards the client's own token.
    passthrough: bool,
}

/// The member's credential pool: the base key plus any `api_keys` extras
/// (plan 04 step 5).
pub(crate) fn probe_credential_pool(inner: &EngineInner, provider: &str) -> Vec<SecretString> {
    credential_pool(inner, provider)
}

fn credential_pool(inner: &EngineInner, provider: &str) -> Vec<SecretString> {
    let Some(config) = inner.providers.get(provider) else {
        return Vec::new();
    };
    let mut keys = vec![config.api_key.clone()];
    if let Some(extra) = &config.api_keys {
        keys.extend(extra.iter().cloned());
    }
    keys
}

/// Resolves one plan candidate into a sendable attempt: adapter, endpoint,
/// and a credential picked from the member's pool.
async fn resolve_attempt(
    inner: &Arc<EngineInner>,
    candidate: &ResolvedTarget,
    avoid_credential: Option<&str>,
    inbound: Option<&SecretString>,
) -> Result<Attempt, EngineError> {
    let configured = &candidate.target;
    let adapter = inner.adapters.get(&configured.protocol).ok_or_else(|| {
        EngineError::Internal(format!(
            "no adapter registered for protocol: {}",
            configured.protocol
        ))
    })?;
    let passthrough = inner
        .providers
        .get(&configured.provider_name)
        .is_some_and(|p| p.passthrough_auth);
    let keys = credential_pool(inner, &configured.provider_name);
    let cooling = inner
        .health
        .cooling_credentials(&configured.provider_name, OffsetDateTime::now_utc())
        .await;
    let pick: CredentialPick = inner
        .credentials
        .pick(&CredentialQuery {
            provider: &configured.provider_name,
            keys: &keys,
            passthrough,
            inbound,
            avoid: avoid_credential,
            cooling: &cooling,
        })
        .await?;

    let target = ProviderAdapterTarget {
        provider_name: configured.provider_name.clone(),
        adapter_name: configured.adapter_name.clone(),
        protocol: ProtocolName::new(&configured.protocol),
        endpoint: configured.endpoint.clone(),
        auth_style: configured.auth_style,
        api_key: pick.secret,
        requested_model: configured.requested_model.clone(),
        upstream_model: configured.upstream_model.clone(),
        headers: Arc::clone(&configured.headers),
        lossy: configured.lossy,
        forward_inbound_headers: Arc::clone(&configured.forward_inbound_headers),
        prompt_cache: configured.prompt_cache.clone(),
    };
    Ok(Attempt {
        target,
        adapter,
        credential_id: pick.credential_id,
        passthrough,
    })
}

fn ledger_model(core_model: &ModelRef, upstream_model: &str) -> ModelRef {
    let mut model = core_model.clone();
    if model.upstream.is_none() && upstream_model != model.requested {
        model.upstream = Some(upstream_model.to_owned());
    }
    model
}

fn settle(
    inner: &EngineInner,
    ticket: &AdmissionTicket,
    succeeded: bool,
    usage: Option<&Usage>,
    cost: Option<&Cost>,
) {
    inner.admission.settle(
        ticket,
        &AdmissionOutcome {
            succeeded,
            usage,
            cost,
        },
    );
}

/// The cost of provider-reported usage at the member's pricing, when both
/// exist: what the provider bills, which is what budgets charge (plan 05
/// step 6). Synthetic or unknown usage is never priced.
pub fn billed_cost(usage: &Usage, pricing: Option<&ModelPricing>) -> Option<Cost> {
    if usage.provenance != UsageProvenance::ProviderReported {
        return None;
    }
    pricing.map(|p| compute_cost(usage, p))
}

/// The member and upstream model the ladder will try first — cooldown
/// reorder included (plan 05 step 4). Slot 2's allowlist verdict and slot
/// 6's reservation both key on this pair, so they judge the same target.
async fn first_member_and_model<'a>(
    inner: &EngineInner,
    plan: &'a RoutePlan,
) -> (&'a str, &'a str) {
    let first = ordered_candidates(inner, plan, OffsetDateTime::now_utc())
        .await
        .first()
        .and_then(|index| plan.candidates.get(*index))
        .or_else(|| plan.candidates.first());
    (
        first.map_or("", |candidate| candidate.target.provider_name.as_str()),
        first.map_or("", |candidate| candidate.target.upstream_model.as_str()),
    )
}

/// The admission claim for one request against the member the ladder starts
/// on: what [`authorize_step`] and [`admit`] both hand to the admission
/// step.
fn admission_claim<'a>(
    req: &'a EngineRequest,
    member: &'a str,
    model: &'a str,
    estimated_input_tokens: u64,
) -> AdmissionClaim<'a> {
    AdmissionClaim {
        caller: &req.caller,
        provider: &req.provider_name,
        member,
        model,
        requested_model: &req.core.model.requested,
        body_hash: &req.body_hash,
        estimated_input_tokens,
        max_output_tokens: req
            .core
            .sampling
            .max_tokens
            .and_then(|n| u64::try_from(n).ok()),
    }
}

/// Slot 2 for one request: the per-key allowlist verdict
/// ([`Admission::authorize`](crate::traits::Admission::authorize)). It runs
/// once the route plan (slot 3) names the member and model the allowlists
/// match, and before the idempotency claim (slot 4), the cache lookup and
/// single-flight (slot 5): a replay, a cached answer or a joined flight must
/// never answer a key that may no longer run the request. Nothing is
/// estimated or reserved here.
async fn authorize_step(
    inner: &EngineInner,
    req: &EngineRequest,
    plan: &RoutePlan,
) -> Result<(), EngineError> {
    let (member, model) = first_member_and_model(inner, plan).await;
    inner
        .admission
        .authorize(&admission_claim(req, member, model, 0))
        .await
}

/// Slots 6 and 8 for one request: the token estimate, the context-window
/// preflight (plan 05 step 8) and the admission reservation against the
/// member the ladder will try first — cooldown reorder included, so the
/// concurrency slot starts on the member that will actually serve (plan 05
/// step 4). Cache hits never reach here, so they are neither preflighted
/// nor charged.
async fn admit(
    inner: &Arc<EngineInner>,
    req: &EngineRequest,
    plan: &RoutePlan,
) -> Result<AdmissionTicket, EngineError> {
    let (member, model) = first_member_and_model(inner, plan).await;
    let estimated_input_tokens = crate::preflight::estimate_input_tokens(&inner.counter, &req.core);
    let claim = admission_claim(req, member, model, estimated_input_tokens);
    let preflight_config = inner
        .config
        .load()
        .map(|config| config.server.preflight)
        .unwrap_or_default();
    let context_length = match inner.providers.get(member) {
        Some(provider) => inner.catalogs.context_length(provider, model).await,
        None => None,
    };
    crate::preflight::preflight(
        preflight_config,
        crate::preflight::PreflightInput {
            model,
            estimated_input_tokens,
            max_output_tokens: claim.max_output_tokens,
            context_length,
        },
    )?;
    inner.admission.reserve(&claim).await
}

/// The result of [`strip_provider_bound_state`]: the core request to encode
/// for the failover member, and whether any provider-bound state was removed
/// (which forces the adapter to re-encode from the stripped core instead of
/// passing the client's raw body through).
struct StripOutcome {
    core: CoreRequest,
    changed: bool,
}

/// Strips provider-bound request state for a failover target (plan 04 step
/// 10, conflict 6, fix brief 04-1): everything in `provider_hints` except the
/// dialect-neutral keys the ladder keeps, tool-call signatures, and thinking
/// blocks from earlier assistant turns. Pool members are always distinct
/// providers, so a provider-issued signature or file id is foreign to the
/// next member. Refuses the switch when the request carries state no other
/// provider can honour: a media block referencing a provider file, or signed
/// thinking in an open tool loop (the provider requires that block back
/// unmodified while thinking is on, so it can be neither stripped nor sent
/// elsewhere). The client's own words never change.
fn strip_provider_bound_state(core: &CoreRequest) -> Result<StripOutcome, EngineError> {
    for content in std::iter::once(core.system.as_slice())
        .chain(core.messages.iter().map(|m| m.content.as_slice()))
    {
        refuse_provider_file_references(content)?;
    }
    let loop_turn = open_tool_loop_turn(&core.messages);
    if let Some(index) = loop_turn {
        let open_loop_thinks = core.messages.get(index).is_some_and(|message| {
            message.content.iter().any(|block| {
                matches!(
                    block,
                    CoreContent::Thinking { .. } | CoreContent::RedactedThinking { .. }
                )
            })
        });
        if open_loop_thinks {
            return Err(EngineError::FailoverRefused {
                reason: "signed thinking in an open tool loop cannot move between providers"
                    .to_owned(),
            });
        }
    }

    let mut stripped = core.clone();
    let mut changed = false;
    let kept_hints = stripped
        .provider_hints
        .raw
        .iter()
        .filter(|(key, _)| {
            HINTS_KEPT_ACROSS_FAILOVER
                .iter()
                .any(|kept| key.eq_ignore_ascii_case(kept))
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    changed |= kept_hints.len() != stripped.provider_hints.raw.len();
    stripped.provider_hints.raw = kept_hints;

    for (index, message) in stripped.messages.iter_mut().enumerate() {
        if message.role == CoreRole::Assistant
            && Some(index) != loop_turn
            && remove_thinking_blocks(&mut message.content)
        {
            changed = true;
        }
        changed |= clear_tool_signatures(&mut message.content);
    }

    Ok(StripOutcome {
        core: stripped,
        changed,
    })
}

/// Refuses media blocks whose `source` references a provider file, recursing
/// into `ToolResult` content (a tool result can embed images).
fn refuse_provider_file_references(content: &[CoreContent]) -> Result<(), EngineError> {
    for block in content {
        match block {
            CoreContent::Image { source }
            | CoreContent::Document { source }
            | CoreContent::Audio { source }
            | CoreContent::Video { source } => {
                if source_references_provider_file(source) {
                    return Err(EngineError::FailoverRefused {
                        reason: "file reference cannot move between providers".to_owned(),
                    });
                }
            }
            CoreContent::ToolResult {
                content: nested, ..
            } => refuse_provider_file_references(nested)?,
            _ => {}
        }
    }
    Ok(())
}

/// Whether a JSON `source` value references a provider file at any depth: a
/// `file_id` key, a `file_uri` key, or a `"type": "file"` marker (the Files
/// API source shape). A provider-issued id cannot be resolved by another
/// provider.
fn source_references_provider_file(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map.iter().any(|(key, inner)| {
            if key == "file_id" || key == "file_uri" {
                return true;
            }
            if key == "type" && inner.as_str() == Some("file") {
                return true;
            }
            source_references_provider_file(inner)
        }),
        serde_json::Value::Array(items) => items.iter().any(source_references_provider_file),
        _ => false,
    }
}

/// The assistant turn an open tool loop continues: the last assistant message
/// whose `ToolUse` is answered by a `ToolResult` in the final user message.
/// `None` when no tool loop is open.
fn open_tool_loop_turn(messages: &[CoreMessage]) -> Option<usize> {
    let final_user = messages.iter().rposition(|m| m.role == CoreRole::User)?;
    let answered_ids: Vec<&str> = messages
        .get(final_user)?
        .content
        .iter()
        .filter_map(|block| match block {
            CoreContent::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
            _ => None,
        })
        .collect();
    if answered_ids.is_empty() {
        return None;
    }
    messages.iter().rposition(|message| {
        message.role == CoreRole::Assistant
            && message.content.iter().any(|block| {
                matches!(block, CoreContent::ToolUse { id, .. } if answered_ids.contains(&id.as_str()))
            })
    })
}

/// Removes `Thinking` and `RedactedThinking` blocks; true when it removed any.
fn remove_thinking_blocks(content: &mut Vec<CoreContent>) -> bool {
    let before = content.len();
    content.retain(|block| {
        !matches!(
            block,
            CoreContent::Thinking { .. } | CoreContent::RedactedThinking { .. }
        )
    });
    content.len() != before
}

/// Clears tool-call signatures (a provider-issued token replayed for another
/// provider), recursing into tool results; true when it cleared any.
fn clear_tool_signatures(content: &mut [CoreContent]) -> bool {
    let mut changed = false;
    for block in content {
        match block {
            CoreContent::ToolUse { signature, .. } if signature.is_some() => {
                *signature = None;
                changed = true;
            }
            CoreContent::ToolResult {
                content: nested, ..
            } => changed |= clear_tool_signatures(nested),
            _ => {}
        }
    }
    changed
}

/// A failed non-streaming ladder: the error, plus the usage and cost the
/// serving attempt reported when the provider billed it before the proxy
/// refused the response (a post-call guard rejection). Budgets charge what
/// providers bill, per attempt (plan 05 step 6), so settlement charges it
/// even though the client gets an error.
struct LadderFailure {
    error: EngineError,
    usage: Option<Usage>,
    cost: Option<Cost>,
}

impl From<EngineError> for LadderFailure {
    fn from(error: EngineError) -> Self {
        Self {
            error,
            usage: None,
            cost: None,
        }
    }
}

/// The per-request ladder state shared by both execution modes.
struct Ladder {
    plan: RoutePlan,
    session_id: Option<String>,
    deadline: Option<Instant>,
    /// 0-based attempt sequence across the whole request (attempt rows).
    seq: u32,
    /// The member the first attempt targeted; failover away from it strips
    /// provider-bound state.
    first_member: Option<String>,
    /// Candidate indexes in the order the ladder walks them.
    order: Vec<usize>,
    request_id: String,
    /// Credential id the current attempt used.
    current_credential: Option<String>,
}

impl Ladder {
    fn remaining(&self) -> Option<Duration> {
        self.deadline
            .and_then(|deadline| deadline.checked_duration_since(Instant::now()))
    }

    fn deadline_passed(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn attempts_per_member(&self) -> u32 {
        self.plan.retry.max_attempts.max(1)
    }

    /// The upstream model a member encodes for.
    fn member_model(&self, member: &str) -> String {
        self.plan
            .candidates
            .iter()
            .find(|candidate| candidate.target.provider_name == member)
            .map(|candidate| candidate.target.upstream_model.clone())
            .unwrap_or_default()
    }

    fn is_last_member(&self, member: &str) -> bool {
        self.order
            .last()
            .and_then(|index| self.plan.candidates.get(*index))
            .is_some_and(|candidate| candidate.target.provider_name == member)
    }

    /// The strip outcome for this member: the first member is the provider
    /// that issued the request's provider-bound state, so it is never
    /// stripped or refused; every failover target is a distinct provider
    /// (pool members reject duplicates), so it gets the stripped core — or a
    /// refusal naming the state that cannot move (plan 04 step 10).
    fn strip_for(&self, req: &EngineRequest, member: &str) -> Result<StripOutcome, EngineError> {
        match &self.first_member {
            Some(first) if first != member => strip_provider_bound_state(&req.core),
            _ => Ok(StripOutcome {
                core: req.core.clone(),
                changed: false,
            }),
        }
    }
}

/// The candidate order the ladder walks (plan 04 step 9): members past their
/// cooldown first in configured order; when every member is cooling, the
/// configured order stands — cooldowns reorder, they never refuse service.
async fn ordered_candidates(
    inner: &EngineInner,
    plan: &RoutePlan,
    now: OffsetDateTime,
) -> Vec<usize> {
    let mut available = Vec::new();
    for (index, candidate) in plan.candidates.iter().enumerate() {
        if inner
            .health
            .provider_available(&candidate.target.provider_name, now)
            .await
        {
            available.push(index);
        }
    }
    if available.is_empty() {
        (0..plan.candidates.len()).collect()
    } else {
        available
    }
}

/// The one deadline across all attempts (plan 04 step 3): the tighter of the
/// route's ladder deadline and the server-wide upstream total, when set.
fn ladder_deadline(inner: &EngineInner, plan: &RoutePlan, start: Instant) -> Option<Instant> {
    let route_deadline =
        (!plan.retry.total_timeout.is_zero()).then(|| start + plan.retry.total_timeout);
    let config_deadline = inner
        .config
        .load()
        .ok()
        .and_then(|config| config.server.upstream_timeouts.total)
        .map(|total| start + total);
    match (route_deadline, config_deadline) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (some, None) | (None, some) => some,
    }
}

/// Bounds one in-flight attempt by the ladder's total deadline (plan 04
/// step 3, fix brief 04-2): the transport's per-attempt `first_byte`/`idle`
/// budgets do not know the ladder, so a slow upstream could otherwise run
/// one attempt past `total_timeout`. The elapsed deadline is a transient
/// [`EngineError::UpstreamTimeout`] that flows through the ladder's failure
/// path; without a deadline the future runs unbounded.
async fn within_deadline<T>(
    deadline: Option<Instant>,
    fut: impl Future<Output = T>,
) -> Result<T, EngineError> {
    // Boxed so the attempt's transport future stays off the ladder's own
    // future, which the `large_futures` lint and the spawned task stack both
    // hold to a modest size.
    let fut = Box::pin(fut);
    match deadline {
        Some(deadline) => {
            let outcome =
                tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), fut).await;
            match outcome {
                Ok(output) => Ok(output),
                // `Elapsed` carries no cause: the deadline itself is the fact
                // the client needs.
                Err(_) => Err(EngineError::UpstreamTimeout(
                    "retry ladder deadline exceeded during the attempt".to_owned(),
                )),
            }
        }
        None => Ok(fut.await),
    }
}

/// Resolves when the ladder's total deadline passes; never resolves without
/// one (plan 04 step 3).
async fn ladder_deadline_elapsed(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await,
        None => std::future::pending::<()>().await,
    }
}

/// Whether the failure feedback marks the credential as rejected (its key
/// should cool down separately from the member).
fn credential_feedback(class: &ErrorClass, error: &EngineError) -> CredentialFeedback {
    match class {
        ErrorClass::Terminal => match error {
            EngineError::Upstream {
                status: 401 | 403, ..
            } => CredentialFeedback::Rejected,
            _ => CredentialFeedback::TransientFailure,
        },
        ErrorClass::Transient {
            retry_after: Some(_),
        } => CredentialFeedback::Rejected,
        ErrorClass::Transient { retry_after: None } => CredentialFeedback::TransientFailure,
    }
}

/// The ladder's view of one failure: classification, except that an auth
/// rejection (401/403) stays retryable while the member still has untried
/// keys in its credential pool (plan 04 step 5) — the next key may simply be
/// the working one.
fn class_for_ladder(inner: &EngineInner, member: &str, error: &EngineError) -> ErrorClass {
    match classify(error) {
        ErrorClass::Terminal => {
            let auth_rejected = matches!(
                error,
                EngineError::Upstream {
                    status: 401 | 403,
                    ..
                }
            );
            if auth_rejected && credential_pool(inner, member).len() > 1 {
                ErrorClass::Transient { retry_after: None }
            } else {
                ErrorClass::Terminal
            }
        }
        other => other,
    }
}

/// Updates passive health (plan 04 step 9): a throttle with real reset
/// advice cools until the reported reset; any other transient failure cools
/// for the ladder's configured cooldown. A failure that belongs to one
/// operator key cools that key, and the member only while it has no other
/// key to serve with; a failure deterministic for this request (a
/// client-side 4xx, the client's own passthrough token being rejected)
/// cools nothing — see [`member_cooldown_applies`].
async fn apply_cooldowns(
    inner: &EngineInner,
    ladder: &Ladder,
    provider: &str,
    error: &EngineError,
    class: &ErrorClass,
    feedback: CredentialFeedback,
) {
    let now = OffsetDateTime::now_utc();
    let advised = match error {
        EngineError::Upstream { retry_after, .. } => *retry_after,
        _ => None,
    };
    let until = now + advised.unwrap_or(ladder.plan.retry.cooldown);
    let credential_id = ladder.current_credential.as_deref();
    if feedback == CredentialFeedback::Rejected
        && let Some(id) = credential_id
    {
        inner.health.cooldown_credential(provider, id, until).await;
    }
    let another_key = another_key_available(inner, provider, credential_id, now).await;
    if member_cooldown_applies(class, error, feedback, another_key) {
        inner.health.cooldown_provider(provider, until).await;
    }
}

/// Whether a failed attempt should cool the member itself. An upstream
/// fault (5xx, timeout, connect, throttle without key attribution) says the
/// member is unhealthy; a rejected operator credential says only that key
/// is; a client fault says nothing about the member — one client's
/// malformed request must not demote the member for everyone else.
fn member_cooldown_applies(
    class: &ErrorClass,
    error: &EngineError,
    feedback: CredentialFeedback,
    another_key_available: bool,
) -> bool {
    if operator_credential_rejected(error, feedback) {
        return !another_key_available;
    }
    matches!(class, ErrorClass::Transient { .. })
}

/// A failure that belongs to the operator's credential rather than to the
/// member or the client: an auth rejection or an advised throttle on a key
/// the operator configured. Never a passthrough client token — rejecting
/// that is between the upstream and the client.
fn operator_credential_rejected(error: &EngineError, feedback: CredentialFeedback) -> bool {
    feedback == CredentialFeedback::Rejected
        && matches!(
            error,
            EngineError::Upstream {
                status: 401 | 403 | 429,
                auth_owner: AuthOwner::Operator,
                ..
            }
        )
}

/// Whether the member has a key other than the one this failure rejected
/// that is not itself cooling (so the member can still serve). A pick that
/// recorded no credential id had no pool to rotate — nothing else to try.
/// Reads the store after the rejected key's cooldown was written, so that
/// key counts as unavailable.
async fn another_key_available(
    inner: &EngineInner,
    provider: &str,
    rejected_credential: Option<&str>,
    now: OffsetDateTime,
) -> bool {
    if rejected_credential.is_none() {
        return false;
    }
    let cooling = inner.health.cooling_credentials(provider, now).await;
    credential_pool(inner, provider).len() > cooling.len()
}

/// Whether the RetryExhausted wrap applies: only when the route had a real
/// ladder (failover members or more than one attempt per member). A bare
/// single-provider route surfaces the raw error exactly as before plan 04.
fn ladder_had_capacity(plan: &RoutePlan) -> bool {
    plan.candidates.len() > 1 || plan.retry.max_attempts > 1
}

/// Wraps the ladder's last error for the client (plan 04 step 12): the
/// proxy already did the retries, so the client must not stack its own.
async fn exhausted(
    inner: &EngineInner,
    plan: &RoutePlan,
    attempts: u32,
    last: EngineError,
) -> EngineError {
    let members: Vec<&str> = plan
        .candidates
        .iter()
        .map(|c| c.target.provider_name.as_str())
        .collect();
    let mut retry_after = last.retry_after();
    if retry_after.is_none() {
        retry_after = inner
            .health
            .soonest_reset(&members, OffsetDateTime::now_utc())
            .await;
    }
    let retry_after =
        retry_after.or_else(|| (!plan.retry.cooldown.is_zero()).then_some(plan.retry.cooldown));
    EngineError::RetryExhausted {
        attempts,
        last: Box::new(last),
        retry_after,
    }
}

/// Records one failed attempt: the row, passive health, credential
/// feedback. `Ok(())` when the ladder may run another attempt; the final
/// error when the failure is terminal.
async fn record_attempt_failure(
    inner: &Arc<EngineInner>,
    ladder: &Ladder,
    member: &str,
    error: &EngineError,
    attempt_start: Instant,
) {
    tracing::debug!(
        route = %ladder.plan.route,
        member = %member,
        error = %error,
        "upstream attempt failed"
    );
    let class = classify(error);
    let feedback = credential_feedback(&class, error);
    emit_attempt(
        inner,
        &AttemptRecord {
            request_id: &ladder.request_id,
            seq: ladder.seq - 1,
            route: &ladder.plan.route,
            provider: member,
            model: &ladder.member_model(member),
            credential_id: ladder.current_credential.as_deref(),
            outcome: AttemptOutcome::Failed,
            error: Some(error),
            latency: attempt_start.elapsed(),
        },
    );
    apply_cooldowns(inner, ladder, member, error, &class, feedback).await;
    inner
        .credentials
        .feedback(member, ladder.current_credential.as_deref(), feedback);
}

/// Refuses the failover when the route's affinity policy says so (conflict
/// 6): a session pinned to one member may not land on another with state it
/// cannot honour — not after a failed first attempt, and not because the
/// pinned member was cooling down when the ladder ordered its candidates.
fn refuse_switch(inner: &EngineInner, ladder: &Ladder, member: &str) -> Result<(), EngineError> {
    if ladder.plan.affinity.on_conflict != AffinityConflictPolicy::Refuse {
        return Ok(());
    }
    let Some(session) = &ladder.session_id else {
        return Ok(());
    };
    let Some(pinned) = inner.affinity.pinned_provider(session) else {
        return Ok(());
    };
    if pinned != member {
        return Err(EngineError::AffinityRefused {
            pinned,
            session: session.clone(),
        });
    }
    Ok(())
}

/// Non-streaming execution.
pub(crate) async fn execute(
    inner: Arc<EngineInner>,
    req: EngineRequest,
) -> Result<EngineResponse, EngineError> {
    let start = req.start;
    crate::validate_provider_name(&req.provider_name)?;
    inner.metrics.record_request(false);

    emit_request_received(&inner, &req, false);

    let mut req = req;
    apply_request_transforms(&inner, &mut req).await?;
    let outcome = run_once(&inner, &req).await;
    match outcome {
        Ok(response) => Ok(response),
        Err(error) => {
            inner.metrics.record_failure();
            emit_response_failed(
                &inner,
                &req.request_id,
                Some(&req.provider_name),
                Some(&req.core.model),
                &error,
                start,
            );
            Err(error)
        }
    }
}

/// Slot 7.5 for non-streaming and streaming alike: the guards' request
/// rewrites (masking, plan 09 step 4), applied once before the ladder so
/// every attempt encodes the same redacted core. With no guards
/// configured this is a no-op.
async fn apply_request_transforms(
    inner: &Arc<EngineInner>,
    req: &mut EngineRequest,
) -> Result<(), EngineError> {
    if inner.guards.is_empty() {
        return Ok(());
    }
    let before: Vec<String> = crate::guard_texts(&req.core).collect();
    for guard in &inner.guards {
        guard.transform_request(&mut req.core).await?;
    }
    let after: Vec<String> = crate::guard_texts(&req.core).collect();
    if before != after {
        // A guard rewrote the request's words. The client's raw bytes are
        // no longer the truth: the same-format passthrough shortcut must
        // not carry the un-redacted text to the upstream.
        req.inbound.wire.force_core_encode = true;
    }
    Ok(())
}

async fn resolve_plan(
    inner: &Arc<EngineInner>,
    req: &EngineRequest,
) -> Result<RoutePlan, EngineError> {
    inner
        .router
        .resolve(&RouteQuery {
            provider: &req.provider_name,
            route_kind: req.route_kind,
            requested_model: &req.core.model.requested,
            session_id: session_id_of(req).as_deref(),
        })
        .await
}

/// The inbound header that carries the client's idempotency key (plan 06
/// step 3, backlog 24). Lowercased by the server's header conversion.
const IDEMPOTENCY_HEADER: &str = "idempotency-key";

/// Longest idempotency key accepted; longer values are a client error, not
/// silently truncated map keys.
const MAX_IDEMPOTENCY_KEY_LEN: usize = 255;

/// Slot 4's outcome for one request.
enum IdempotencyStep {
    /// The opt-in is off or the request carried no key: nothing to do.
    Ignored,
    /// This request claimed the key and must complete or release it.
    Claimed(IdempotencyClaim),
    /// A stored outcome exists: replay it.
    Replay(Box<StoredOutcome>),
}

/// Claims the idempotency key (slot 4) before cache lookup, single-flight
/// and budget checks (plan 06 step 3). In-flight conflicts are 409; a key
/// reused with a different body or under a different provider route is a
/// client error. An empty `body_hash` means the HTTP layer did not digest
/// the body, so the key cannot be matched safely and is ignored.
fn idempotency_step(
    inner: &Arc<EngineInner>,
    req: &EngineRequest,
) -> Result<IdempotencyStep, EngineError> {
    let Some(store) = &inner.idempotency else {
        return Ok(IdempotencyStep::Ignored);
    };
    let Some(key) = req.inbound.headers.get(IDEMPOTENCY_HEADER) else {
        return Ok(IdempotencyStep::Ignored);
    };
    if key.is_empty() || req.body_hash.is_empty() {
        return Ok(IdempotencyStep::Ignored);
    }
    if key.len() > MAX_IDEMPOTENCY_KEY_LEN {
        return Err(EngineError::InvalidRequest(format!(
            "Idempotency-Key is longer than {MAX_IDEMPOTENCY_KEY_LEN} characters"
        )));
    }
    // The match carries the provider route: body bytes alone do not say
    // which provider served them, so a key reused across routes must not
    // replay the first route's response under the second's name.
    let matched = IdemMatch::new(&req.provider_name, req.route_kind, &req.body_hash);
    match store.claim(&req.caller, key, &matched) {
        IdempotencyAcquire::Claim(claim) => Ok(IdempotencyStep::Claimed(claim)),
        IdempotencyAcquire::InFlight => Err(EngineError::IdempotencyInFlight),
        IdempotencyAcquire::Replay(outcome) => Ok(IdempotencyStep::Replay(outcome)),
        IdempotencyAcquire::BodyMismatch => Err(EngineError::InvalidRequest(
            "Idempotency-Key was already used with a different request".to_owned(),
        )),
    }
}

/// Records a row for a request answered without a fresh upstream attempt —
/// a cache hit, a single-flight join, or an idempotent replay (plan 05 step
/// 6's rule): the stored usage is kept for the record, nothing is charged.
/// A served copy records no cost: the one upstream call's cost lives on the
/// row of the request that made it.
fn record_served_copy(
    inner: &EngineInner,
    req: &EngineRequest,
    upstream_model: &str,
    served: &llm_proxy_protocol::core::CoreResponse,
    latency_ms: u64,
) {
    inner.metrics.record_cache_hit();
    inner
        .usage
        .record(&ProxyEvent::ResponseCompleted(ResponseCompleted {
            request_id: req.request_id.clone(),
            timestamp: time::OffsetDateTime::now_utc(),
            provider: req.provider_name.clone(),
            upstream_message_id: served.id.clone(),
            model: ledger_model(&req.core.model, upstream_model),
            usage: served.usage.clone(),
            cost: None,
            stop_reason: served.stop_reason.clone(),
            latency_ms,
            credential_id: None,
            cache_hit: true,
        }));
}

/// One non-streaming request pass (slots 2-13). A single-flight follower
/// whose leader vanished without publishing (its handler future dropped on a
/// client disconnect, or a panic) re-executes the request itself once: the
/// leader's disappearance must not answer the follower with an internal
/// error for a request it never ran.
async fn run_once(
    inner: &Arc<EngineInner>,
    req: &EngineRequest,
) -> Result<EngineResponse, EngineError> {
    let mut leader_losses_left = 1;
    loop {
        match run_once_attempt(inner, req).await {
            Err(error) if leader_losses_left > 0 && is_flight_leader_lost(&error) => {
                leader_losses_left -= 1;
                tracing::debug!(
                    request_id = %req.request_id,
                    route = %req.provider_name,
                    "single-flight leader vanished; re-executing the request"
                );
            }
            outcome => return outcome,
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "slots 4-6 in one place: the idempotency claim, cache lookup and single-flight \
              join read as one ordered sequence; splitting them would hide the order the \
              plan fixes"
)]
async fn run_once_attempt(
    inner: &Arc<EngineInner>,
    req: &EngineRequest,
) -> Result<EngineResponse, EngineError> {
    // Slot 3: resolve the route to a plan (one candidate, or a pool's).
    let plan = resolve_plan(inner, req).await?;
    // Slot 2: the per-key allowlist verdict. It needs the plan's first
    // member, so it follows slot 3 — and precedes slots 4 and 5, so a key
    // restricted since an answer was stored for it is refused, not served
    // that answer. `execute` records the failure row for this error.
    authorize_step(inner, req, &plan).await?;
    let first_model = plan
        .candidates
        .first()
        .map(|candidate| candidate.target.upstream_model.clone())
        .unwrap_or_default();

    // Slot 4: claim the idempotency key — before the cache, single-flight
    // and budget checks (plan 06 step 3).
    let idempotency = idempotency_step(inner, req)?;
    if let IdempotencyStep::Replay(outcome) = &idempotency {
        // A stored outcome answers immediately — unless the first request
        // ran in the other mode, whose outcome this mode cannot render.
        let Some(response) = outcome.response.clone() else {
            return Err(EngineError::InvalidRequest(
                "Idempotency-Key was first used by a streaming request; its outcome \
                 can only be replayed by one"
                    .to_owned(),
            ));
        };
        tracing::debug!(
            request_id = %req.request_id,
            route = %req.provider_name,
            "idempotent replay"
        );
        record_served_copy(inner, req, &first_model, &response.core, 0);
        return Ok(response);
    }

    // Slot 5a: cache lookup, before single-flight and upstream calls (plan
    // 06 step 4). Only complete non-streaming responses are served from the
    // cache.
    let cache_key = CacheKey {
        caller: &req.caller,
        provider: &req.provider_name,
        route_kind: req.route_kind,
        model: &first_model,
        request: &req.core,
        body_hash: &req.body_hash,
        streaming: false,
    };
    if let Some(entry) = inner
        .cache
        .lookup(&cache_key)
        .await
        .map_err(|error| EngineError::Internal(format!("cache lookup: {error}")))?
        && entry.events.is_empty()
    {
        tracing::debug!(
            request_id = %req.request_id,
            route = %req.provider_name,
            "cache hit"
        );
        let mut cached = entry.response;
        // A hit is recorded as a hit (plan 05 step 6): the row keeps the
        // cached usage for the record, but the request never reserved and
        // is never charged — it did not reach admission.
        record_served_copy(inner, req, &first_model, &cached, 0);
        cached.model = ledger_model(&req.core.model, &first_model);
        let response = EngineResponse {
            core: cached,
            raw_body: None,
        };
        if let IdempotencyStep::Claimed(claim) = idempotency {
            claim.complete(StoredOutcome {
                response: Some(response.clone()),
                events: Vec::new(),
            });
        }
        return Ok(response);
    }

    // Slot 5b: per-caller single-flight (plan 06 step 2). Without a body
    // digest there is no safe identity, so identical requests cannot be
    // merged.
    let mut lead = None;
    if !req.body_hash.is_empty() {
        match inner.singleflight.acquire(&cache_key) {
            FlightAcquire::Join(follower) => {
                // Identical request from this caller already in flight:
                // share its upstream call. The joiner never reserves and is
                // never charged; the leader's row carries the charge.
                let shared = follower.wait().await;
                tracing::debug!(
                    request_id = %req.request_id,
                    route = %req.provider_name,
                    "single-flight join"
                );
                return match shared {
                    Ok(response) => {
                        record_served_copy(
                            inner,
                            req,
                            &first_model,
                            &response.core,
                            req.start
                                .elapsed()
                                .as_millis()
                                .try_into()
                                .unwrap_or(u64::MAX),
                        );
                        if let IdempotencyStep::Claimed(claim) = idempotency {
                            claim.complete(StoredOutcome {
                                response: Some(response.clone()),
                                events: Vec::new(),
                            });
                        }
                        Ok(response)
                    }
                    // The claim drops, releasing the key: a retry executes.
                    Err(error) => Err(error),
                };
            }
            FlightAcquire::Lead(flight) => lead = Some(flight),
        }
    }

    // Slots 6 and 8: preflight, then one reservation for the whole ladder;
    // settlement happens exactly once at the end. A failure that the
    // provider still billed (a guard-rejected response) charges its usage.
    // An admission refusal reaches the followers as itself (402/429/…), not
    // through the lead drop backstop's internal error.
    let ticket = match admit(inner, req, &plan).await {
        Ok(ticket) => ticket,
        Err(error) => {
            if let Some(lead) = lead {
                lead.publish(Err(error.clone()));
            }
            return Err(error);
        }
    };
    // The reservation lives inside this future: a drop before settlement
    // (client disconnect, per-family timeout mid-ladder) must release the
    // lease, not hold its slot and budget forever. Disarmed on both paths
    // below, which settle explicitly.
    let mut settle_guard = UnsettledLeaseGuard::new(Arc::clone(&inner.admission), &ticket);

    match run_ladder_once(inner, req, plan, &ticket, &cache_key).await {
        Ok(response) => {
            // run_ladder_once settled on success (slot 13) before
            // returning; dropping here must not settle twice.
            settle_guard.disarm();
            if let Some(lead) = lead {
                lead.publish(Ok(response.clone()));
            }
            if let IdempotencyStep::Claimed(claim) = idempotency {
                claim.complete(StoredOutcome {
                    response: Some(response.clone()),
                    events: Vec::new(),
                });
            }
            Ok(response)
        }
        Err(failure) => {
            settle_guard.disarm();
            settle(
                inner,
                &ticket,
                false,
                failure.usage.as_ref(),
                failure.cost.as_ref(),
            );
            if let Some(lead) = lead {
                lead.publish(Err(failure.error.clone()));
            }
            // The idempotency claim drops and releases the key: failed
            // requests are not stored, so a retry executes again.
            Err(failure.error)
        }
    }
}

/// Slots 7-13 of the non-streaming lifecycle against a reserved ticket: the
/// attempt loop (plan 04 steps 1-5). The success path settles the ticket;
/// every failure propagates to [`run_once`], which settles it as failed
/// (with the billed usage when the failure carries one).
#[expect(
    clippy::too_many_lines,
    reason = "one attempt loop: candidate iteration, credential rotation, \
              classification, cooldowns and the deadline read as a unit; \
              splitting the arms would scatter the ladder state they mutate"
)]
async fn run_ladder_once(
    inner: &Arc<EngineInner>,
    req: &EngineRequest,
    plan: RoutePlan,
    ticket: &AdmissionTicket,
    cache_key: &CacheKey<'_>,
) -> Result<EngineResponse, LadderFailure> {
    // Slot 7: pre-call guards, once per request — the request does not
    // change between attempts (only provider-bound hints do, which are not
    // the client's words).
    for guard in &inner.guards {
        match guard.inspect_request(&req.core).await? {
            GuardDecision::Allow => {}
            GuardDecision::Reject(message) => {
                return Err(EngineError::InvalidRequest(message).into());
            }
        }
    }

    let mut ladder = Ladder {
        session_id: session_id_of(req),
        deadline: ladder_deadline(inner, &plan, req.start),
        seq: 0,
        first_member: None,
        order: ordered_candidates(inner, &plan, OffsetDateTime::now_utc()).await,
        request_id: req.request_id.clone(),
        current_credential: None,
        plan,
    };
    let mut last_error: Option<EngineError> = None;
    let order = ladder.order.clone();

    'ladder: for &member_index in &order {
        let Some(candidate) = ladder.plan.candidates.get(member_index).cloned() else {
            return Err(EngineError::Internal("candidate index out of range".to_owned()).into());
        };
        let member = candidate.target.provider_name.clone();
        refuse_switch(inner, &ladder, &member)?;
        // The cap follows the ladder (plan 05 step 7): move the in-flight
        // slot to the member about to serve — failover and the first
        // candidate both land here — and skip the member when its cap is
        // saturated, exactly as a cooldown would.
        if let Err(error) = inner
            .admission
            .rekey(ticket, &member, &candidate.target.upstream_model)
            .await
        {
            last_error = Some(error);
            continue 'ladder;
        }
        if ladder.first_member.is_none() {
            ladder.first_member = Some(member.clone());
        }
        // Provider-bound state this member cannot honour: skip the candidate
        // and keep the ladder going; the refusal becomes the ladder's last
        // error when no member remains (plan 04 step 10, fix brief 04-1).
        let strip = match ladder.strip_for(req, &member) {
            Ok(strip) => strip,
            Err(refusal) => {
                last_error = Some(refusal);
                continue 'ladder;
            }
        };
        // Pool credential ids are positional (`key-<i>`): an id that failed
        // against the previous member names a different secret here, so the
        // avoid hint lives and dies with one member's attempts.
        let mut avoid_credential: Option<String> = None;

        for attempt_in_member in 0..ladder.attempts_per_member() {
            if ladder.deadline_passed() {
                last_error = Some(EngineError::UpstreamTimeout(
                    "retry ladder deadline exceeded before the next attempt".to_owned(),
                ));
                break 'ladder;
            }
            ladder.seq += 1;
            let attempt_start = Instant::now();

            let attempt = resolve_attempt(
                inner,
                &candidate,
                avoid_credential.as_deref(),
                req.inbound.auth.as_ref(),
            )
            .await?;
            ladder.current_credential = attempt.credential_id.clone();
            avoid_credential = attempt.credential_id.clone();

            let mut wire: ClientWireRequest = req.inbound.wire.clone();
            wire.forwarded_headers = select_forwarded_headers(
                &req.inbound.headers,
                &attempt.target.forward_inbound_headers,
            );
            // The strip removed provider-bound state: the client's raw bytes
            // are no longer the truth for this member, so the adapter must
            // re-encode from the stripped core instead of passing the raw
            // body through. When nothing was stripped, keep the passthrough
            // (byte-for-byte fidelity on the happy path).
            if strip.changed {
                wire.force_core_encode = true;
            }
            let proxy_req: ProxyRequest = attempt
                .adapter
                .encode_request(&strip.core, &attempt.target, &wire)
                .map_err(map_encode_error)?;

            let sent = within_deadline(
                ladder.deadline,
                inner.transport.client_for(&member).send(proxy_req),
            )
            .await;

            let response_bytes = match sent {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(error)) => {
                    let engine_error = map_provider_error(&error, attempt.passthrough);
                    last_error = Some(engine_error.clone());
                    record_attempt_failure(inner, &ladder, &member, &engine_error, attempt_start)
                        .await;
                    if matches!(
                        class_for_ladder(inner, &member, &engine_error),
                        ErrorClass::Terminal
                    ) {
                        return Err(engine_error.into());
                    }
                    if attempt_in_member + 1 < ladder.attempts_per_member() {
                        let wait = next_wait(
                            engine_error.retry_after(),
                            ladder.plan.retry.backoff,
                            ladder.remaining(),
                        );
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                    continue 'ladder;
                }
                Err(engine_error) => {
                    // The ladder deadline consumed the attempt mid-flight:
                    // the row is written through the same failure path, then
                    // the loop's deadline check ends the ladder — no further
                    // attempt can fit (fix brief 04-2).
                    last_error = Some(engine_error.clone());
                    record_attempt_failure(inner, &ladder, &member, &engine_error, attempt_start)
                        .await;
                    continue 'ladder;
                }
            };

            let decoded = attempt
                .adapter
                .decode_response(&response_bytes, &attempt.target);
            let mut core_resp = match decoded {
                Ok(response) => response,
                Err(error) => {
                    let engine_error =
                        EngineError::ProviderDecode(format!("decode response: {error}"));
                    last_error = Some(engine_error.clone());
                    record_attempt_failure(inner, &ladder, &member, &engine_error, attempt_start)
                        .await;
                    if attempt_in_member + 1 < ladder.attempts_per_member() {
                        tokio::time::sleep(ladder.plan.retry.backoff).await;
                        continue;
                    }
                    continue 'ladder;
                }
            };

            let pricing = inner
                .providers
                .pricing_for(
                    &attempt.target.provider_name,
                    &attempt.target.upstream_model,
                )
                .cloned();
            let cost = billed_cost(&core_resp.usage, pricing.as_ref());

            // Slot 11: post-call guards. A guard rejection is proxy policy,
            // not an upstream fault: no retry, no failover — but the
            // provider billed this attempt, so its usage still reaches
            // settlement (plan 05 step 6).
            for guard in &inner.guards {
                let error = match guard.inspect_response(&core_resp).await {
                    Ok(GuardDecision::Allow) => continue,
                    Ok(GuardDecision::Reject(message)) => EngineError::InvalidRequest(message),
                    Err(error) => error,
                };
                return Err(LadderFailure {
                    error,
                    usage: Some(core_resp.usage.clone()),
                    cost,
                });
            }
            // Slot 11.5 (plan 09 step 4): the guards' response rewrites —
            // the masked form is what the client receives, what the cache
            // stores and what usage rows describe.
            let content_before_guards =
                (!inner.guards.is_empty()).then(|| core_resp.content.clone());
            for guard in &inner.guards {
                guard.transform_response(&mut core_resp).await?;
            }
            let guards_rewrote_response =
                content_before_guards.is_some_and(|before| before != core_resp.content);

            let latency = req.start.elapsed();
            core_resp.cost = cost.clone();

            emit_attempt(
                inner,
                &AttemptRecord {
                    request_id: &req.request_id,
                    seq: ladder.seq - 1,
                    route: &ladder.plan.route,
                    provider: &member,
                    model: &attempt.target.upstream_model,
                    credential_id: attempt.credential_id.as_deref(),
                    outcome: AttemptOutcome::Succeeded,
                    error: None,
                    latency: attempt_start.elapsed(),
                },
            );

            // Slot 13: settle (charging what the provider billed), store,
            // record.
            settle(inner, ticket, true, Some(&core_resp.usage), cost.as_ref());
            let entry = CacheEntry {
                response: core_resp.clone(),
                events: Vec::new(),
            };
            inner.cache.store(cache_key, &entry).await;

            // Pinning is the pool's opt-in (`affinity.sticky`); a direct
            // route or a non-sticky pool leaves the session's pin alone.
            if ladder.plan.affinity.sticky
                && let Some(session) = &ladder.session_id
            {
                inner.affinity.pin(session, &member);
            }
            inner.credentials.feedback(
                &member,
                attempt.credential_id.as_deref(),
                CredentialFeedback::Succeeded,
            );

            let resp_usage = core_resp.usage.clone();
            let resp_model = core_resp.model.clone();
            let resp_upstream_id = core_resp.id.clone();
            let resp_stop_reason = core_resp.stop_reason.clone();
            let credential_id = attempt.credential_id.clone();
            inner.metrics.record_success(
                &attempt.target.provider_name,
                &attempt.target.upstream_model,
                latency,
            );
            inner
                .usage
                .record(&ProxyEvent::ResponseCompleted(ResponseCompleted {
                    request_id: req.request_id.clone(),
                    timestamp: time::OffsetDateTime::now_utc(),
                    provider: attempt.target.provider_name.clone(),
                    upstream_message_id: resp_upstream_id,
                    model: ledger_model(&resp_model, &attempt.target.upstream_model),
                    usage: resp_usage,
                    cost,
                    stop_reason: resp_stop_reason,
                    latency_ms: latency.as_millis().try_into().unwrap_or(u64::MAX),
                    credential_id,
                    cache_hit: false,
                }));

            // Same-format passthrough: the upstream response body reaches the
            // client verbatim; the decode above only read usage and the stop
            // reason for the ledger, it did not rewrite the body. Unless a
            // guard rewrote the response: the raw upstream bytes would then
            // carry the un-redacted text past the masked core — to the
            // client and, through the idempotency claim, to retries — so the
            // client is served the re-encoded masked form instead.
            let raw_body = if guards_rewrote_response {
                None
            } else if wire.same_format_as(&attempt.target.protocol) {
                Some(response_bytes)
            } else {
                None
            };

            return Ok(EngineResponse {
                core: core_resp,
                raw_body,
            });
        }
    }

    // Every candidate exhausted: wrap for the client when the route had a
    // ladder (plan 04 step 12).
    let last = last_error.unwrap_or_else(|| {
        EngineError::Internal("retry ladder ended without attempting".to_owned())
    });
    if ladder_had_capacity(&ladder.plan) {
        Err(exhausted(inner, &ladder.plan, ladder.seq, last)
            .await
            .into())
    } else {
        Err(last.into())
    }
}

/// What the pre-commit window reports: the buffered prefix (possibly empty,
/// when the hold budget expired in silence) that crosses the commit
/// boundary, or the error that aborts before any byte is committed.
enum FirstByteResult {
    Head(Vec<CoreEvent>),
    PreStreamError(EngineError),
}

type ByteStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<Bytes, llm_proxy_adapters::ProviderError>> + Send>,
>;

/// Streaming execution. Returns once the stream is committed to the client —
/// the first core events decoded, or the hold budget expiring in silence
/// (plan 04 step 6) — and hands back the buffered prefix plus the
/// continuation stream. Settlement happens inside the stream task as it
/// completes or is cancelled.
///
/// A stream single-flight joiner whose leader went away before its first
/// byte (its caller's connection dropped, or the task died) re-runs the
/// request itself once, mirroring [`run_once`]'s non-streaming follower: the
/// leader's disappearance must not answer a connected joiner with an
/// internal error for a request it never ran.
pub(crate) async fn execute_stream(
    inner: Arc<EngineInner>,
    mut req: EngineRequest,
) -> Result<EngineStream, EngineError> {
    let start = req.start;
    crate::validate_provider_name(&req.provider_name)?;
    inner.metrics.record_request(true);

    emit_request_received(&inner, &req, true);
    apply_request_transforms(&inner, &mut req).await?;

    let mut leader_losses_left = 1;
    loop {
        match execute_stream_attempt(Arc::clone(&inner), req.clone(), start).await {
            Err(error) if leader_losses_left > 0 && is_stream_leader_lost(&error) => {
                leader_losses_left -= 1;
                tracing::debug!(
                    request_id = %req.request_id,
                    route = %req.provider_name,
                    "stream single-flight leader vanished; re-running the request"
                );
            }
            Err(error) if is_stream_leader_lost(&error) => {
                // No leader stayed for this joiner even after re-running:
                // record the failure it now owns.
                emit_response_failed(
                    &inner,
                    &req.request_id,
                    Some(&req.provider_name),
                    Some(&req.core.model),
                    &error,
                    start,
                );
                return Err(error);
            }
            outcome => return outcome,
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the streaming entry mirrors run_once's slot order plus the first-byte window; \
              the sequence is the specification"
)]
async fn execute_stream_attempt(
    inner: Arc<EngineInner>,
    req: EngineRequest,
    start: Instant,
) -> Result<EngineStream, EngineError> {
    // Slot 3: the route plan.
    let plan = resolve_plan(&inner, &req).await?;
    // Slot 2: the per-key allowlist verdict, ahead of the idempotency
    // replay and the stream shares — the same rule the non-streaming path
    // follows. No lead exists yet, so there is nothing to publish.
    if let Err(error) = authorize_step(&inner, &req, &plan).await {
        emit_response_failed(
            &inner,
            &req.request_id,
            Some(&req.provider_name),
            Some(&req.core.model),
            &error,
            start,
        );
        return Err(error);
    }
    let first_model = plan
        .candidates
        .first()
        .map(|candidate| candidate.target.upstream_model.clone())
        .unwrap_or_default();

    // Slot 4: claim the idempotency key (plan 06 step 3), before
    // single-flight and budget checks.
    let idempotency = match idempotency_step(&inner, &req) {
        Ok(step) => step,
        Err(error) => {
            emit_response_failed(
                &inner,
                &req.request_id,
                Some(&req.provider_name),
                Some(&req.core.model),
                &error,
                start,
            );
            return Err(error);
        }
    };
    if let IdempotencyStep::Replay(outcome) = &idempotency {
        // Replay the stored events as a fresh stream; a non-streaming
        // outcome cannot be rendered as one.
        if outcome.events.is_empty() {
            let error = EngineError::InvalidRequest(
                "Idempotency-Key was first used by a non-streaming request; its outcome \
                 can only be replayed by one"
                    .to_owned(),
            );
            emit_response_failed(
                &inner,
                &req.request_id,
                Some(&req.provider_name),
                Some(&req.core.model),
                &error,
                start,
            );
            return Err(error);
        }
        let events = outcome.events.clone();
        return Ok(replay_events_stream(Arc::clone(&inner), &req, events));
    }

    // Slot 5: join or lead per-caller stream single-flight (plan 06 step 2).
    let mut stream_lead = None;
    if !req.body_hash.is_empty() {
        let stream_key = CacheKey {
            caller: &req.caller,
            provider: &req.provider_name,
            route_kind: req.route_kind,
            model: &first_model,
            request: &req.core,
            body_hash: &req.body_hash,
            streaming: true,
        };
        match inner.stream_shares.acquire(&stream_key) {
            StreamAcquire::Join(share) => {
                tracing::debug!(
                    request_id = %req.request_id,
                    route = %req.provider_name,
                    "stream single-flight join"
                );
                // The joiner waits for the leader's first-byte outcome before
                // it answers: a leader that fails before committing (guards,
                // admission, a pre-stream upstream error) fails the joiner
                // with that same error, not with a 200 whose stream then ends
                // in a generic in-band failure.
                if let Err(error) = share.committed().await {
                    if !is_stream_leader_lost(&error) {
                        emit_response_failed(
                            &inner,
                            &req.request_id,
                            Some(&req.provider_name),
                            Some(&req.core.model),
                            &error,
                            start,
                        );
                    }
                    // A vanished leader is not this request's failure: the
                    // caller of execute_stream re-runs the request once, and
                    // records the row only when no leader stays.
                    return Err(error);
                }
                // The joiner's own idempotency claim completes with the
                // shared outcome once the leader's stream ends cleanly —
                // mirroring the non-streaming follower — so a matching retry
                // replays instead of re-executing (a failed leader releases
                // it: failures are not stored).
                let claim = match idempotency {
                    IdempotencyStep::Claimed(claim) => Some(claim),
                    IdempotencyStep::Ignored | IdempotencyStep::Replay(_) => None,
                };
                return Ok(joined_stream(Arc::clone(&inner), &req, share, claim));
            }
            StreamAcquire::Lead(lead) => stream_lead = Some(lead),
        }
    }

    // Slot 7: pre-call guards, once per request. A refusal reaches the
    // joiners through the lead as itself, not through the drop backstop's
    // internal error.
    for guard in &inner.guards {
        let refusal = match guard.inspect_request(&req.core).await {
            Ok(GuardDecision::Allow) => continue,
            Ok(GuardDecision::Reject(message)) => EngineError::InvalidRequest(message),
            Err(error) => error,
        };
        emit_response_failed(
            &inner,
            &req.request_id,
            Some(&req.provider_name),
            Some(&req.core.model),
            &refusal,
            start,
        );
        if let Some(lead) = stream_lead.take() {
            lead.publish(Err(refusal.clone()));
        }
        return Err(refusal);
    }

    // Slots 6 and 8: preflight, then one reservation for the whole ladder.
    // An admission refusal reaches the joiners as itself (402/429/…), not
    // through the lead drop backstop's internal error — same rule as the
    // non-streaming path.
    let ticket = match admit(&inner, &req, &plan).await {
        Ok(ticket) => ticket,
        Err(error) => {
            emit_response_failed(
                &inner,
                &req.request_id,
                Some(&req.provider_name),
                Some(&req.core.model),
                &error,
                start,
            );
            if let Some(lead) = stream_lead.take() {
                lead.publish(Err(error.clone()));
            }
            return Err(error);
        }
    };

    let request_id = req.request_id.clone();
    let route_name = req.provider_name.clone();
    let requested_model = req.core.model.clone();

    let (first_byte_tx, first_byte_rx) = tokio::sync::oneshot::channel::<FirstByteResult>();
    let failure_recorded = Arc::new(AtomicBool::new(false));
    let idempotency_claim = match idempotency {
        IdempotencyStep::Claimed(claim) => Some(claim),
        IdempotencyStep::Ignored | IdempotencyStep::Replay(_) => None,
    };
    let core_stream = build_core_event_stream(
        Arc::clone(&inner),
        req,
        plan,
        ticket,
        first_byte_tx,
        Arc::clone(&failure_recorded),
        stream_lead,
        idempotency_claim,
    );

    let first_head = first_byte_rx.await.map_err(|error| {
        let err = EngineError::Internal(format!(
            "stream task exited before first event ({error}: possible panic, cancellation, or empty stream)"
        ));
        if !failure_recorded.swap(true, Ordering::SeqCst) {
            inner.metrics.record_failure();
            emit_response_failed(
                &inner,
                &request_id,
                Some(route_name.as_str()),
                Some(&requested_model),
                &err,
                start,
            );
        }
        err
    })?;

    match first_head {
        FirstByteResult::PreStreamError(error) => {
            emit_response_failed(
                &inner,
                &request_id,
                Some(route_name.as_str()),
                Some(&requested_model),
                &error,
                start,
            );
            Err(error)
        }
        FirstByteResult::Head(head) => Ok(EngineStream {
            head,
            tail: core_stream,
        }),
    }
}

struct StreamContext {
    decoder: Box<dyn ProviderStreamDecoder + Send>,
    sse_framer: SseFramer,
    core: CoreRequest,
    request_id: String,
    provider_name: String,
    upstream_model: String,
    start: Instant,
    pricing: Option<ModelPricing>,
    passthrough_auth: bool,
    pending_usage: Option<Usage>,
    stop_reason: Option<StopReason>,
    upstream_message_id: Option<String>,
    ticket: AdmissionTicket,
    credential_id: Option<String>,
    /// The single-flight share this stream taps its events into, when
    /// identical concurrent requests from the same caller may join it
    /// (plan 06 step 2).
    share: Option<Arc<StreamShare>>,
}

impl StreamContext {
    #[expect(
        clippy::too_many_arguments,
        reason = "the context is the one place the committed stream's state lives; a builder would just re-list these"
    )]
    fn new(
        decoder: Box<dyn ProviderStreamDecoder + Send>,
        core: CoreRequest,
        request_id: String,
        provider_name: String,
        upstream_model: String,
        start: Instant,
        pricing: Option<ModelPricing>,
        passthrough_auth: bool,
        ticket: AdmissionTicket,
        credential_id: Option<String>,
    ) -> Self {
        Self {
            decoder,
            sse_framer: SseFramer::new(),
            core,
            request_id,
            provider_name,
            upstream_model,
            start,
            pricing,
            passthrough_auth,
            pending_usage: None,
            stop_reason: None,
            upstream_message_id: None,
            ticket,
            credential_id,
            share: None,
        }
    }

    /// The model ref recorded on ledger rows: the requested model plus the
    /// upstream model actually serving it, so alias rows stay attributable.
    fn ledger_model(&self) -> ModelRef {
        ledger_model(&self.core.model, &self.upstream_model)
    }

    /// Settles this stream's reservation with whatever usage the provider
    /// reported so far and its billed cost (plan 05 step 6): a stream that
    /// ends early still charges the tokens it consumed; one that reported
    /// nothing charges nothing.
    fn settle(&self, inner: &EngineInner, succeeded: bool) {
        let cost = self
            .pending_usage
            .as_ref()
            .and_then(|usage| billed_cost(usage, self.pricing.as_ref()));
        settle(
            inner,
            &self.ticket,
            succeeded,
            self.pending_usage.as_ref(),
            cost.as_ref(),
        );
    }
}

/// Records one failed attempt inline (encode failures never reach
/// `record_attempt_failure`, which needs the full ladder).
#[expect(
    clippy::too_many_arguments,
    reason = "one flat record call; grouping these into a struct would duplicate AttemptRecord"
)]
fn record_failed_attempt(
    inner: &EngineInner,
    request_id: &str,
    seq: u32,
    route: &str,
    provider: &str,
    model: &str,
    credential_id: Option<&str>,
    error: &EngineError,
    latency: Duration,
) {
    emit_attempt(
        inner,
        &AttemptRecord {
            request_id,
            seq,
            route,
            provider,
            model,
            credential_id,
            outcome: AttemptOutcome::Failed,
            error: Some(error),
            latency,
        },
    );
}

fn capture_lifecycle(ctx: &mut StreamContext, event: &CoreEvent) {
    if let Some(share) = &ctx.share {
        share.record(event);
    }
    match event {
        CoreEvent::MessageStart { id: Some(id), .. } => {
            ctx.upstream_message_id = Some(id.clone());
        }
        CoreEvent::UsageDelta { usage } => {
            ctx.pending_usage = Some(usage.clone());
        }
        CoreEvent::MessageStop { stop_reason, .. } => {
            ctx.stop_reason = Some(stop_reason.clone());
        }
        _ => {}
    }
}

/// Yields the in-band `CoreEvent::Error` for a post-commit failure; the
/// client codec renders it (an OpenAI client gets an error chunk, an
/// Anthropic client an `error` event) and closes cleanly. The event is
/// tapped into the single-flight share, so joiners see it too.
async fn emit_in_band_error(
    ctx: &mut StreamContext,
    tx: &tokio::sync::mpsc::Sender<CoreEvent>,
    message: &str,
) -> bool {
    let error = llm_proxy_protocol::core::CoreStreamError::new(
        CoreStreamErrorKind::Upstream,
        message.to_owned(),
    );
    let event = CoreEvent::Error { error };
    capture_lifecycle(ctx, &event);
    tx.send(event).await.is_ok()
}

fn record_stream_success(inner: &EngineInner, ctx: &StreamContext) {
    inner
        .metrics
        .record_success(&ctx.provider_name, &ctx.upstream_model, ctx.start.elapsed());
    let usage = ctx
        .pending_usage
        .clone()
        .unwrap_or_else(Usage::synthetic_zero);
    let cost = billed_cost(&usage, ctx.pricing.as_ref());
    inner
        .usage
        .record(&ProxyEvent::ResponseCompleted(ResponseCompleted {
            request_id: ctx.request_id.clone(),
            timestamp: time::OffsetDateTime::now_utc(),
            provider: ctx.provider_name.clone(),
            upstream_message_id: ctx.upstream_message_id.clone(),
            model: ctx.ledger_model(),
            usage,
            cost,
            stop_reason: ctx.stop_reason.clone().unwrap_or(StopReason::EndTurn),
            latency_ms: ctx
                .start
                .elapsed()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            credential_id: ctx.credential_id.clone(),
            cache_hit: false,
        }));
}

fn record_client_cancelled(inner: &EngineInner, ctx: &StreamContext) {
    inner
        .usage
        .record(&ProxyEvent::ResponseFailed(ResponseFailed {
            request_id: ctx.request_id.clone(),
            timestamp: time::OffsetDateTime::now_utc(),
            provider: Some(ctx.provider_name.clone()),
            model: Some(ctx.ledger_model()),
            error_kind: "ClientCancelled".to_owned(),
            message: "client disconnected before the stream completed".to_owned(),
            http_status: 200,
            latency_ms: ctx
                .start
                .elapsed()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
        }));
}

fn record_client_cancelled_no_ctx(inner: &EngineInner, request_id: &str) {
    inner.metrics.record_client_cancel();
    inner
        .usage
        .record(&ProxyEvent::ResponseFailed(ResponseFailed {
            request_id: request_id.to_owned(),
            timestamp: time::OffsetDateTime::now_utc(),
            provider: None,
            model: None,
            error_kind: "ClientCancelled".to_owned(),
            message: "client disconnected before the stream completed".to_owned(),
            http_status: 200,
            latency_ms: 0,
        }));
}

/// The sanitized message of a `CoreEvent::Error`, for ledger rows.
fn core_event_message(event: &CoreEvent) -> &str {
    match event {
        CoreEvent::Error { error } => error.message(),
        _ => "",
    }
}

/// Captures lifecycle state for and forwards the final events; `false` when
/// the client went away.
///
/// Holds `&mut StreamContext` across the awaits (the send backs up on a slow
/// client); `&mut` of a `Send` struct is `Send`, unlike `&`.
async fn emit_final_events(
    inner: &Arc<EngineInner>,
    ctx: &mut StreamContext,
    tx: &tokio::sync::mpsc::Sender<CoreEvent>,
    events: Vec<CoreEvent>,
) -> bool {
    for core_event in events {
        capture_lifecycle(ctx, &core_event);
        if tx.send(core_event).await.is_err() {
            record_client_cancelled(inner, ctx);
            ctx.settle(inner, false);
            return false;
        }
    }
    true
}

/// Spawns the supervised streaming task and returns its event stream: the
/// task runs the ladder (attempts, the hold-budget buffer, failover) and the
/// committed stream to its end.
#[expect(
    clippy::too_many_arguments,
    reason = "the stream task owns the request, its plan, its reservation, the first-byte \
              channel and the sharing handles together; a struct would re-list them"
)]
fn build_core_event_stream(
    inner: Arc<EngineInner>,
    req: EngineRequest,
    plan: RoutePlan,
    ticket: AdmissionTicket,
    first_byte_tx: tokio::sync::oneshot::Sender<FirstByteResult>,
    failure_recorded: Arc<AtomicBool>,
    lead: Option<StreamLead>,
    idempotency: Option<IdempotencyClaim>,
) -> BoxStream<'static, CoreEvent> {
    let (tx, rx) = tokio::sync::mpsc::channel::<CoreEvent>(256);
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let supervise_request_id = req.request_id.clone();
    let supervise_route = req.provider_name.clone();
    let supervise_model = req.core.model.clone();
    let supervise_start = req.start;
    // The task settles its reservation on every path it completes; the
    // supervisor keeps its own handle for the paths it does not.
    let supervise_ticket = ticket.clone();
    let stream_span = tracing::info_span!("stream_task", request_id = %supervise_request_id);

    let task_inner = Arc::clone(&inner);
    let stream_task = tokio::spawn(
        async move {
            let inner = task_inner;
            let mut stream = StreamLadder {
                ladder: Ladder {
                    session_id: session_id_of(&req),
                    deadline: ladder_deadline(&inner, &plan, req.start),
                    seq: 0,
                    first_member: None,
                    order: ordered_candidates(&inner, &plan, OffsetDateTime::now_utc()).await,
                    request_id: req.request_id.clone(),
                    current_credential: None,
                    plan,
                },
                ticket,
                first_byte_tx: Some(first_byte_tx),
                last_error: None,
                lead,
                idempotency,
                last_billed: None,
            };
            let order = stream.ladder.order.clone();
            let io = StreamIo {
                tx: &tx,
                cancel: &cancel_clone,
            };

            for &member_index in &order {
                if !stream_attempt(&inner, &mut stream, &req, member_index, &io).await {
                    return;
                }
            }

            // Every candidate exhausted before commit.
            let last = stream.last_error.clone().unwrap_or_else(|| {
                EngineError::Internal("retry ladder ended without attempting".to_owned())
            });
            stop_with(&inner, &mut stream, last, true).await;
        }
        .instrument(stream_span),
    );

    tokio::spawn(async move {
        match stream_task.await {
            Ok(()) => {}
            Err(join_error) => {
                let panicked = join_error.is_panic();
                let error = EngineError::Internal(format!(
                    "stream task {}",
                    if panicked {
                        "panicked"
                    } else {
                        "was cancelled"
                    }
                ));
                tracing::warn!(
                    request_id = %supervise_request_id,
                    error = %join_error,
                    panicked,
                    "detached stream task did not complete cleanly"
                );
                if !failure_recorded.swap(true, Ordering::SeqCst) {
                    inner.metrics.record_failure();
                    emit_response_failed(
                        &inner,
                        &supervise_request_id,
                        Some(supervise_route.as_str()),
                        Some(&supervise_model),
                        &error,
                        supervise_start,
                    );
                }
                // The row above is deduplicated with the caller's first-byte
                // error path; the reservation is not — only the task settles
                // it, and it never reached that point.
                settle(&inner, &supervise_ticket, false, None, None);
            }
        }
    });
    let rx_stream = futures::stream::unfold(rx, |mut rx| async move {
        let received = rx.recv().await;
        received.map(|event| (event, rx))
    });
    let cancel_guard = cancel.drop_guard();
    rx_stream
        .map(move |item| {
            let _guard = &cancel_guard;
            item
        })
        .boxed()
}

/// Commits the buffered prefix to the caller and streams the response to
/// its end: streaming guards first (a rejection still maps to an HTTP error
/// and never fails over — proxy policy, not an upstream fault), then the
/// first-byte handoff, then the post-commit stream where errors are in-band
/// and a terminal row and settlement are guaranteed (plan 04 steps 6-7).
#[expect(
    clippy::too_many_arguments,
    reason = "the commit boundary carries the ladder, the committed stream and the IO handles together"
)]
async fn commit_and_stream(
    inner: &Arc<EngineInner>,
    stream: &mut StreamLadder,
    req: &EngineRequest,
    mut committed: CommittedStream,
    attempt_start: Instant,
    tx: &tokio::sync::mpsc::Sender<CoreEvent>,
    cancel: &CancellationToken,
) {
    let ladder = &mut stream.ladder;
    let member = committed.ctx.provider_name.clone();
    let head = std::mem::take(&mut committed.head);
    // The head's usage is captured before the guards see it: a rejected
    // prefix was still billed by the provider, so its settlement charges
    // what was reported (plan 05 step 6).
    for core_event in head.iter() {
        capture_lifecycle(&mut committed.ctx, core_event);
    }
    if let Some(error) = vet_prefix(inner, &head).await {
        inner.metrics.record_failure();
        signal_pre_stream_error(inner, stream, error.clone());
        committed.ctx.settle(inner, false);
        stream.finish_stream(StreamEnd::Failed(error));
        return;
    }
    // Same opt-in as the non-streaming path: only sticky pools pin.
    if ladder.plan.affinity.sticky
        && let Some(session) = &ladder.session_id
    {
        inner.affinity.pin(session, &member);
    }
    emit_attempt(
        inner,
        &AttemptRecord {
            request_id: &req.request_id,
            seq: ladder.seq - 1,
            route: &ladder.plan.route,
            provider: &member,
            model: &committed.ctx.upstream_model,
            credential_id: ladder.current_credential.as_deref(),
            outcome: AttemptOutcome::Streaming,
            error: None,
            latency: attempt_start.elapsed(),
        },
    );
    let signalled = stream
        .first_byte_tx
        .take()
        .map(|tx| tx.send(FirstByteResult::Head(head)))
        .is_some_and(|result| result.is_ok());
    if !signalled {
        // The caller is gone, so nobody else will settle the reservation;
        // write the row here.
        inner.metrics.record_client_cancel();
        record_client_cancelled(inner, &committed.ctx);
        committed.ctx.settle(inner, false);
        // The leader's own caller is gone; joiners re-run the request rather
        // than fail for a stream they never saw.
        stream.finish_stream(StreamEnd::Failed(EngineError::Internal(
            STREAM_LEADER_LOST.to_owned(),
        )));
        return;
    }
    // The leader committed: joiners may answer 200 and replay the collected
    // events (plan 06 step 2).
    if let Some(share) = &committed.ctx.share {
        share.signal_committed();
    }
    let end = stream_committed(inner, &mut committed.ctx, tx, committed.stream, cancel).await;
    // Publish the terminal outcome to single-flight joiners and store the
    // collected events for idempotent replay on a clean end only — a failed
    // stream releases the key so a retry executes (plan 06 steps 2 and 3).
    stream.finish_stream(end);
}

/// The task's IO handles: the client event channel and the cancellation
/// token.
struct StreamIo<'a> {
    tx: &'a tokio::sync::mpsc::Sender<CoreEvent>,
    cancel: &'a CancellationToken,
}

/// Runs the attempts against one plan candidate: encode, send, buffer to
/// the pre-commit window, and either hand the committed stream to
/// [`commit_and_stream`] or move the ladder on. Returns `false` when the
/// task must return (the failure was signalled and the ticket settled).
#[expect(
    clippy::too_many_lines,
    reason = "one candidate's attempt ladder: deadline, credential, encode, send and the buffer handoff read as a unit"
)]
async fn stream_attempt(
    inner: &Arc<EngineInner>,
    stream: &mut StreamLadder,
    req: &EngineRequest,
    member_index: usize,
    io: &StreamIo<'_>,
) -> bool {
    let hold = stream.ladder.plan.retry.hold;
    let Some(candidate) = stream.ladder.plan.candidates.get(member_index).cloned() else {
        stop_with(
            inner,
            stream,
            EngineError::Internal("candidate index out of range".to_owned()),
            false,
        )
        .await;
        return false;
    };
    let member = candidate.target.provider_name.clone();
    if let Err(error) = refuse_switch(inner, &stream.ladder, &member) {
        stop_with(inner, stream, error, false).await;
        return false;
    }
    // The cap follows the ladder (plan 05 step 7), same rule as the
    // non-streaming path: a saturated member is skipped like a cooling one,
    // and the last error surfaces when every member is capped.
    if let Err(error) = inner
        .admission
        .rekey(&stream.ticket, &member, &candidate.target.upstream_model)
        .await
    {
        stream.last_error = Some(error);
        return true;
    }
    if stream.ladder.first_member.is_none() {
        stream.ladder.first_member = Some(member.clone());
    }
    // Provider-bound state this member cannot honour: keep the ladder going;
    // when no member remains the refusal surfaces through `stop_with` (plan
    // 04 step 10, fix brief 04-1).
    let strip = match stream.ladder.strip_for(req, &member) {
        Ok(strip) => strip,
        Err(refusal) => {
            stream.last_error = Some(refusal);
            return true;
        }
    };
    // Same boundary rule as the non-streaming ladder: `current_credential`
    // doubles as the avoid hint, and a positional id from another member
    // must not skip this member's key at the same index. It is re-set right
    // after each pick, so record rows for this member's attempts are
    // unaffected.
    stream.ladder.current_credential = None;

    for attempt_in_member in 0..stream.ladder.attempts_per_member() {
        if stream.ladder.deadline_passed() {
            let timeout = EngineError::UpstreamTimeout(
                "retry ladder deadline exceeded before the next attempt".to_owned(),
            );
            stream.last_error = Some(timeout.clone());
            stop_with(inner, stream, timeout, true).await;
            return false;
        }
        stream.ladder.seq += 1;
        let attempt_start = Instant::now();

        let attempt = match resolve_attempt(
            inner,
            &candidate,
            stream.ladder.current_credential.as_deref(),
            req.inbound.auth.as_ref(),
        )
        .await
        {
            Ok(attempt) => attempt,
            Err(error) => {
                stop_with(inner, stream, error, false).await;
                return false;
            }
        };
        stream.ladder.current_credential = attempt.credential_id.clone();

        let mut wire: ClientWireRequest = req.inbound.wire.clone();
        wire.forwarded_headers = select_forwarded_headers(
            &req.inbound.headers,
            &attempt.target.forward_inbound_headers,
        );
        // Same rule as the non-streaming ladder: a strip that removed
        // provider-bound state forces the adapter to re-encode from the
        // stripped core; an unchanged strip keeps the raw-body passthrough.
        if strip.changed {
            wire.force_core_encode = true;
        }
        let proxy_req = match attempt
            .adapter
            .encode_request(&strip.core, &attempt.target, &wire)
        {
            Ok(proxy_req) => proxy_req,
            Err(error) => {
                let engine_error = map_encode_error(error);
                record_failed_attempt(
                    inner,
                    &stream.ladder.request_id,
                    stream.ladder.seq - 1,
                    &stream.ladder.plan.route,
                    &member,
                    &stream.ladder.member_model(&member),
                    stream.ladder.current_credential.as_deref(),
                    &engine_error,
                    attempt_start.elapsed(),
                );
                stop_with(inner, stream, engine_error, false).await;
                return false;
            }
        };

        let sent = within_deadline(
            stream.ladder.deadline,
            inner.transport.client_for(&member).send_stream(proxy_req),
        )
        .await;
        let byte_stream = match sent {
            Ok(Ok(byte_stream)) => byte_stream,
            Ok(Err(error)) => {
                let engine_error = map_provider_error(&error, attempt.passthrough);
                if !pre_commit_failure(
                    inner,
                    stream,
                    &member,
                    &engine_error,
                    attempt_start,
                    attempt_in_member,
                )
                .await
                {
                    return false;
                }
                continue;
            }
            Err(engine_error) => {
                // Same rule as the non-streaming ladder (fix brief 04-2): the
                // deadline that ran out mid-attempt still writes its row
                // through the failure path, then the loop's deadline check
                // ends the ladder.
                if !pre_commit_failure(
                    inner,
                    stream,
                    &member,
                    &engine_error,
                    attempt_start,
                    attempt_in_member,
                )
                .await
                {
                    return false;
                }
                continue;
            }
        };

        match buffer_pre_commit(
            inner,
            stream,
            req,
            &member,
            &attempt,
            byte_stream,
            io.cancel,
            hold,
            attempt_start,
            attempt_in_member,
        )
        .await
        {
            PreCommit::Retry => continue,
            PreCommit::Stop => return false,
            PreCommit::Committed(committed) => {
                commit_and_stream(
                    inner,
                    stream,
                    req,
                    *committed,
                    attempt_start,
                    io.tx,
                    io.cancel,
                )
                .await;
                return false;
            }
        }
    }
    true
}

/// The streaming ladder's task-local state: the shared ladder plus the
/// channel handles only the stream task owns.
struct StreamLadder {
    ladder: Ladder,
    ticket: AdmissionTicket,
    first_byte_tx: Option<tokio::sync::oneshot::Sender<FirstByteResult>>,
    last_error: Option<EngineError>,
    /// The single-flight lead (plan 06 step 2): publishes the terminal
    /// outcome to joiners. Present whenever the request carries a body
    /// digest.
    lead: Option<StreamLead>,
    /// The idempotency claim (plan 06 step 3): completed with the collected
    /// events on a clean end.
    idempotency: Option<IdempotencyClaim>,
    /// The last provider-reported usage a failed pre-commit prefix carried,
    /// with the cost it was billed at the reporting attempt's pricing: what
    /// settlement charges when every attempt fails and the task stops (the
    /// provider billed those tokens; settlement must not drop them). The
    /// latest report wins and a report-less attempt never erases one that
    /// had it; an earlier failed attempt's billed prefix is superseded and
    /// not charged — the serving attempt's usage is charged at success.
    last_billed: Option<(Usage, Option<Cost>)>,
}

/// How a committed or abandoned stream attempt ended, for the single-flight
/// joiners and the idempotency claim: only a genuinely clean end completes
/// the claim (failures release the key so a retry executes — plan 06 step 3,
/// docs/data-class-policy.md "Failed requests are not stored"); the carried
/// error reaches pre-commit joiners as their own first-byte outcome.
enum StreamEnd {
    /// The stream ran to a clean terminal event.
    Clean,
    /// The stream did not complete cleanly: it failed before committing
    /// (joiners fail with this same error), failed in-band after committing,
    /// or its client went away mid-stream.
    Failed(EngineError),
}

impl StreamLadder {
    /// Publish the terminal outcome to stream joiners and, on a clean end,
    /// store the collected events for idempotent replay. A claim can only
    /// exist alongside a lead (both need the body digest), so the collected
    /// events are always available where a claim is.
    fn finish_stream(&mut self, end: StreamEnd) {
        if let Some(lead) = self.lead.take() {
            let share = lead.share();
            if matches!(end, StreamEnd::Clean)
                && let Some(claim) = self.idempotency.take()
            {
                claim.complete(StoredOutcome {
                    response: None,
                    events: share.collected(),
                });
            }
            let outcome = match end {
                StreamEnd::Clean => Ok(()),
                StreamEnd::Failed(error) => Err(error),
            };
            lead.publish(outcome);
        }
    }
}

/// A committed stream handed from the buffer phase to the commit boundary.
struct CommittedStream {
    /// The per-attempt stream state.
    ctx: StreamContext,
    /// The upstream byte stream, positioned after the buffered prefix.
    stream: ByteStream,
    /// The buffered prefix (empty when the hold budget expired first).
    head: Vec<CoreEvent>,
}

/// The buffered pre-commit phase's outcome.
enum PreCommit {
    /// The stream is committed: the (possibly empty) head crosses the
    /// first-byte window and the caller owns the post-commit streaming.
    Committed(Box<CommittedStream>),
    /// The attempt failed pre-commit and the ladder continues.
    Retry,
    /// The task must return: cancelled, terminal or exhausted — the caller
    /// was signalled and the ticket settled inside.
    Stop,
}

/// Sleeps until the hold deadline, or forever when there is none.
async fn hold_expired(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending::<()>().await,
    }
}

/// Frames and decodes one upstream chunk into `pending`; `Err` carries the
/// framing/decode failure for the ladder to classify.
fn decode_chunk(
    ctx: &mut StreamContext,
    bytes: &Bytes,
    pending: &mut Vec<CoreEvent>,
) -> Result<(), EngineError> {
    let frames = ctx
        .sse_framer
        .push_chunk(bytes)
        .map_err(|e| EngineError::ProviderDecode(format!("stream framing error: {e}")))?;
    for frame in &frames {
        let events = ctx
            .decoder
            .decode_frame(frame)
            .map_err(|e| EngineError::ProviderDecode(format!("provider decode error: {e}")))?;
        pending.extend(events);
    }
    Ok(())
}

/// A leading in-band error event (backlog 2): the upstream answered 200 and
/// rejected the stream in its first frame. It never crossed as output, so
/// the ladder may still recover.
fn leading_stream_error(pending: &[CoreEvent]) -> Option<EngineError> {
    pending
        .iter()
        .find(|event| matches!(event, CoreEvent::Error { .. }))
        .map(|event| {
            EngineError::ProviderDecode(sanitize_upstream_error_body(core_event_message(event)))
        })
}

/// The last provider-reported usage in a buffered pre-commit prefix: what a
/// pre-commit failure must still charge at settlement — the provider billed
/// those tokens even though nothing was committed to the client (the same
/// capture the committed head gets at the commit boundary).
fn prefix_usage(pending: &[CoreEvent]) -> Option<Usage> {
    pending.iter().rev().find_map(|event| match event {
        CoreEvent::UsageDelta { usage } => Some(usage.clone()),
        _ => None,
    })
}

/// Stashes a failing prefix's reported usage and its billed cost on the
/// ladder, so the failure funnel's settles (and `stop_with`, when every
/// attempt fails) charge what the provider billed. The cost is priced at the
/// reporting attempt's own pricing (`ctx.pricing`), not whichever member the
/// ladder is on when it settles. A later attempt's report supersedes an
/// earlier one's; a report-less attempt never erases one that had it.
fn stash_prefix_usage(
    stream_ladder: &mut StreamLadder,
    pending: &[CoreEvent],
    pricing: Option<&ModelPricing>,
) {
    if let Some(usage) = prefix_usage(pending) {
        let cost = billed_cost(&usage, pricing);
        stream_ladder.last_billed = Some((usage, cost));
    }
}

/// Signals a terminal pre-commit failure to the caller (which writes the
/// request's failure row); when the caller is already gone, records the
/// cancellation here.
fn signal_pre_stream_error(inner: &EngineInner, stream: &mut StreamLadder, error: EngineError) {
    match stream.first_byte_tx.take() {
        Some(tx) => {
            if tx.send(FirstByteResult::PreStreamError(error)).is_err() {
                record_client_cancelled_no_ctx(inner, &stream.ladder.request_id);
            }
        }
        None => record_client_cancelled_no_ctx(inner, &stream.ladder.request_id),
    }
}

/// Stops the task with `error`: signal the caller, settle the reservation.
/// `wrap` decides the RetryExhausted wrap (plan 04 step 12).
async fn stop_with(
    inner: &Arc<EngineInner>,
    stream: &mut StreamLadder,
    error: EngineError,
    wrap: bool,
) {
    let final_error = if wrap && ladder_had_capacity(&stream.ladder.plan) {
        exhausted(inner, &stream.ladder.plan, stream.ladder.seq, error).await
    } else {
        error
    };
    inner.metrics.record_failure();
    signal_pre_stream_error(inner, stream, final_error.clone());
    let last_billed = stream.last_billed.take();
    settle(
        inner,
        &stream.ticket,
        false,
        last_billed.as_ref().map(|(usage, _)| usage),
        last_billed.as_ref().and_then(|(_, cost)| cost.as_ref()),
    );
    // Joiners receive the same failure — a pre-commit failure reaches them
    // as their own first-byte error; the idempotency claim drops with the
    // task, releasing the key (failures are not stored).
    stream.finish_stream(StreamEnd::Failed(final_error));
}

/// Slot 11 for streams: the guards vet the buffered prefix; the first
/// rejection wins.
async fn vet_prefix(inner: &Arc<EngineInner>, pending: &[CoreEvent]) -> Option<EngineError> {
    for core_event in pending {
        for guard in &inner.guards {
            match guard.inspect_event(core_event).await {
                Ok(GuardDecision::Allow) => {}
                Ok(GuardDecision::Reject(message)) => {
                    return Some(EngineError::InvalidRequest(message));
                }
                Err(error) => return Some(error),
            }
        }
    }
    None
}

/// The pre-commit buffer phase (plan 04 step 6): frame and decode the
/// upstream stream until the first events, an upstream failure, or the hold
/// deadline in silence — then commit; or hand a pre-commit failure to the
/// ladder.
#[expect(
    clippy::too_many_arguments,
    reason = "the buffer phase is one unit: attempt, ladder, ticket, request and timing state all read together"
)]
async fn buffer_pre_commit(
    inner: &Arc<EngineInner>,
    stream_ladder: &mut StreamLadder,
    req: &EngineRequest,
    member: &str,
    attempt: &Attempt,
    mut stream: ByteStream,
    cancel: &CancellationToken,
    hold: Duration,
    attempt_start: Instant,
    attempt_in_member: u32,
) -> PreCommit {
    let ladder = &mut stream_ladder.ladder;
    let mut ctx = StreamContext::new(
        attempt.adapter.new_stream_decoder(&attempt.target),
        req.core.clone(),
        req.request_id.clone(),
        member.to_owned(),
        ladder.member_model(member),
        req.start,
        inner
            .providers
            .pricing_for(member, &ladder.member_model(member))
            .cloned(),
        attempt.passthrough,
        stream_ladder.ticket.clone(),
        ladder.current_credential.clone(),
    );
    // The share joiners replay from (plan 06 step 2): the leader's events
    // are tapped as they are forwarded.
    ctx.share = stream_ladder.lead.as_ref().map(StreamLead::share);
    let ladder_deadline = ladder.deadline;
    let hold_deadline =
        (!hold.is_zero()).then(|| tokio::time::Instant::from_std(Instant::now() + hold));
    let mut pending: Vec<CoreEvent> = Vec::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                inner.metrics.record_client_cancel();
                let cancelled = EngineError::Internal(
                    "client disconnected before first byte".to_owned(),
                );
                let signalled = stream_ladder
                    .first_byte_tx
                    .take()
                    .map(|tx| tx.send(FirstByteResult::PreStreamError(cancelled.clone())))
                    .is_some_and(|result| result.is_ok());
                if !signalled {
                    record_client_cancelled(inner, &ctx);
                }
                ctx.settle(inner, false);
                // Joiners re-run the request: this failure belongs to the
                // leader's caller, who is no longer connected.
                stream_ladder.finish_stream(StreamEnd::Failed(EngineError::Internal(
                    STREAM_LEADER_LOST.to_owned(),
                )));
                return PreCommit::Stop;
            }
            _ = hold_expired(hold_deadline), if pending.is_empty() => {
                // Hold budget expired in silence: commit with an empty
                // head. The client gets HTTP 200 plus heartbeats while the
                // upstream stays silent; errors from here on are in-band.
                return PreCommit::Committed(Box::new(CommittedStream { ctx, stream, head: pending }));
            }
            _ = ladder_deadline_elapsed(ladder_deadline), if pending.is_empty() => {
                // The ladder's total deadline ran out while the attempt sat
                // waiting for its first event (fix brief 04-2): the same
                // pre-commit failure path records the row, and the attempt
                // loop's deadline check ends the ladder. After commit the
                // stream is bounded by `idle`, not `total`.
                let error = EngineError::UpstreamTimeout(
                    "retry ladder deadline exceeded during the attempt".to_owned(),
                );
                return failure_or_retry(
                    inner,
                    stream_ladder,
                    member,
                    &error,
                    attempt_start,
                    attempt_in_member,
                )
                .await;
            }
            chunk = stream.next() => match chunk {
                Some(Ok(bytes)) => {
                    if decode_chunk(&mut ctx, &bytes, &mut pending).is_err() {
                        // Framing/decode garbage: the ladder classifies it.
                        // The frames decoded before the garbage still billed.
                        stash_prefix_usage(stream_ladder, &pending, ctx.pricing.as_ref());
                        let error = EngineError::ProviderDecode(
                            "provider decode error before the first event".to_owned(),
                        );
                        return failure_or_retry(inner, stream_ladder, member, &error, attempt_start, attempt_in_member).await;
                    }
                    if pending.is_empty() {
                        continue;
                    }
                    if let Some(error) = leading_stream_error(&pending) {
                        stash_prefix_usage(stream_ladder, &pending, ctx.pricing.as_ref());
                        return failure_or_retry(inner, stream_ladder, member, &error, attempt_start, attempt_in_member).await;
                    }
                    return PreCommit::Committed(Box::new(CommittedStream { ctx, stream, head: pending }));
                }
                Some(Err(e)) => {
                    let error = map_provider_error(&e, ctx.passthrough_auth);
                    stash_prefix_usage(stream_ladder, &pending, ctx.pricing.as_ref());
                    return failure_or_retry(inner, stream_ladder, member, &error, attempt_start, attempt_in_member).await;
                }
                None => {
                    // The upstream ended with no events at all: an empty
                    // 200 is not a response; treat it as a transient
                    // failure.
                    let error = EngineError::ProviderDecode(
                        "upstream returned an empty stream with no events".to_owned(),
                    );
                    return failure_or_retry(inner, stream_ladder, member, &error, attempt_start, attempt_in_member).await;
                }
            }
        }
    }
}

/// One pre-commit failure funnel: record it, decide the next move, and map
/// the decision onto [`PreCommit`].
async fn failure_or_retry(
    inner: &Arc<EngineInner>,
    stream_ladder: &mut StreamLadder,
    member: &str,
    error: &EngineError,
    attempt_start: Instant,
    attempt_in_member: u32,
) -> PreCommit {
    if pre_commit_failure(
        inner,
        stream_ladder,
        member,
        error,
        attempt_start,
        attempt_in_member,
    )
    .await
    {
        PreCommit::Retry
    } else {
        PreCommit::Stop
    }
}

/// Handles one pre-commit streaming failure inside the buffer phase:
/// record it (row, cooldowns, feedback) and decide the next move. Returns
/// `true` when the ladder continues with another attempt (the caller breaks
/// the buffer loop and retries), `false` when the task must return — the
/// terminal and exhausted cases signal the caller and settle here. The
/// settles charge the prefix usage the caller stashed on the ladder, at the
/// pricing of the attempt that reported it (a retried attempt may report
/// its own; the last reported one is what settlement charges).
async fn pre_commit_failure(
    inner: &Arc<EngineInner>,
    stream: &mut StreamLadder,
    member: &str,
    error: &EngineError,
    attempt_start: Instant,
    attempt_in_member: u32,
) -> bool {
    stream.last_error = Some(error.clone());
    let ladder = &mut stream.ladder;
    record_attempt_failure(inner, ladder, member, error, attempt_start).await;
    let class = class_for_ladder(inner, member, error);
    let is_last_member_attempt = attempt_in_member + 1 >= ladder.attempts_per_member();
    match class {
        ErrorClass::Terminal => {
            let final_error = error.clone();
            inner.metrics.record_failure();
            signal_pre_stream_error(inner, stream, final_error.clone());
            settle(
                inner,
                &stream.ticket,
                false,
                stream.last_billed.as_ref().map(|(usage, _)| usage),
                stream
                    .last_billed
                    .as_ref()
                    .and_then(|(_, cost)| cost.as_ref()),
            );
            stream.finish_stream(StreamEnd::Failed(final_error));
            false
        }
        ErrorClass::Transient { retry_after } => {
            if is_last_member_attempt && ladder.is_last_member(member) {
                // Ladder exhausted (plan 04 step 12): wrap for the client.
                let final_error = exhausted(inner, &ladder.plan, ladder.seq, error.clone()).await;
                inner.metrics.record_failure();
                signal_pre_stream_error(inner, stream, final_error.clone());
                settle(
                    inner,
                    &stream.ticket,
                    false,
                    stream.last_billed.as_ref().map(|(usage, _)| usage),
                    stream
                        .last_billed
                        .as_ref()
                        .and_then(|(_, cost)| cost.as_ref()),
                );
                stream.finish_stream(StreamEnd::Failed(final_error));
                false
            } else {
                if !is_last_member_attempt {
                    // Another attempt against the same member: honour the
                    // advised wait, clamped to the deadline.
                    let wait =
                        next_wait(retry_after, ladder.plan.retry.backoff, ladder.remaining());
                    tokio::time::sleep(wait).await;
                }
                true
            }
        }
    }
}

/// Drives the committed stream to its end: decode and forward events, mark
/// in-band failures, finalize with the decoder's terminal, settle, and
/// record the terminal row. The returned [`StreamEnd`] carries the failure
/// for joiners and the idempotency claim: only a clean end is stored.
#[expect(
    clippy::too_many_lines,
    reason = "one committed-stream state machine: decode, forward, in-band failure, terminal flush and settlement read as a unit"
)]
async fn stream_committed(
    inner: &Arc<EngineInner>,
    ctx: &mut StreamContext,
    tx: &tokio::sync::mpsc::Sender<CoreEvent>,
    mut stream: ByteStream,
    cancel_clone: &CancellationToken,
) -> StreamEnd {
    // The first failure that marked this stream failed, if one did (in-band
    // error, decode/framing fault, transport error, truncation). A client
    // that went away mid-stream returns early instead.
    let mut failure: Option<EngineError> = None;
    'outer: loop {
        tokio::select! {
            _ = cancel_clone.cancelled() => {
                inner.metrics.record_client_cancel();
                record_client_cancelled(inner, ctx);
                ctx.settle(inner, false);
                return StreamEnd::Failed(EngineError::Internal(
                    "client disconnected before the stream completed".to_owned(),
                ));
            }
            chunk = stream.next() => {
                match chunk {
                    Some(Ok(bytes)) => {
                        let frames = match ctx.sse_framer.push_chunk(&bytes) {
                            Ok(frames) => frames,
                            Err(e) => {
                                let error = EngineError::ProviderDecode(format!(
                                    "stream framing error: {e}"
                                ));
                                inner.metrics.record_failure();
                                if !emit_in_band_error(ctx, tx, PROVIDER_DECODE_CLIENT_MESSAGE)
                                    .await
                                {
                                    record_client_cancelled(inner, ctx);
                                }
                                emit_response_failed(
                                    inner,
                                    &ctx.request_id,
                                    Some(ctx.provider_name.as_str()),
                                    Some(&ctx.ledger_model()),
                                    &error,
                                    ctx.start,
                                );
                                failure = Some(error);
                                break 'outer;
                            }
                        };
                        for frame in &frames {
                            let core_events = match ctx.decoder.decode_frame(frame) {
                                Ok(events) => events,
                                Err(e) => {
                                    let error = EngineError::ProviderDecode(format!(
                                        "provider decode error: {e}"
                                    ));
                                    inner.metrics.record_failure();
                                    if !emit_in_band_error(ctx, tx, PROVIDER_DECODE_CLIENT_MESSAGE)
                                        .await
                                    {
                                        record_client_cancelled(inner, ctx);
                                    }
                                    emit_response_failed(
                                        inner,
                                        &ctx.request_id,
                                        Some(ctx.provider_name.as_str()),
                                        Some(&ctx.ledger_model()),
                                        &error,
                                        ctx.start,
                                    );
                                    failure = Some(error);
                                    break 'outer;
                                }
                            };
                            for core_event in core_events {
                                capture_lifecycle(ctx, &core_event);
                                if matches!(core_event, CoreEvent::Error { .. }) {
                                    // In-band upstream failure after visible
                                    // output: mark it for the client and the
                                    // ledger, then keep the framing valid.
                                    let error = EngineError::ProviderDecode(
                                        sanitize_upstream_error_body(core_event_message(
                                            &core_event,
                                        )),
                                    );
                                    failure = Some(error.clone());
                                    inner.metrics.record_failure();
                                    emit_response_failed(
                                        inner,
                                        &ctx.request_id,
                                        Some(ctx.provider_name.as_str()),
                                        Some(&ctx.ledger_model()),
                                        &error,
                                        ctx.start,
                                    );
                                    if tx.send(core_event).await.is_err() {
                                        record_client_cancelled(inner, ctx);
                                        ctx.settle(inner, false);
                                        return StreamEnd::Failed(EngineError::Internal(
                                            "client disconnected before the stream completed"
                                                .to_owned(),
                                        ));
                                    }
                                    continue;
                                }
                                if tx.send(core_event).await.is_err() {
                                    record_client_cancelled(inner, ctx);
                                    ctx.settle(inner, false);
                                    return StreamEnd::Failed(EngineError::Internal(
                                        "client disconnected before the stream completed"
                                            .to_owned(),
                                    ));
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        let error = map_provider_error(&e, ctx.passthrough_auth);
                        inner.metrics.record_failure();
                        let sanitized =
                            crate::error::fully_sanitize_upstream_error(&e.to_string());
                        if !emit_in_band_error(ctx, tx, &sanitized).await {
                            record_client_cancelled(inner, ctx);
                        }
                        emit_response_failed(
                            inner,
                            &ctx.request_id,
                            Some(ctx.provider_name.as_str()),
                            Some(&ctx.ledger_model()),
                            &error,
                            ctx.start,
                        );
                        failure = Some(error);
                        break 'outer;
                    }
                    None => break 'outer,
                }
            }
        }
    }

    // Flush any trailing partial SSE frame.
    let trailing_frames = match ctx.sse_framer.finish() {
        Ok(frames) => frames,
        Err(e) => {
            tracing::warn!(
                request_id = %ctx.request_id,
                error = %e,
                "SSE framer finish error"
            );
            Vec::new()
        }
    };
    for frame in &trailing_frames {
        let events = match ctx.decoder.decode_frame(frame) {
            Ok(events) => events,
            Err(e) => {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    error = %e,
                    "provider decode error on trailing frame"
                );
                break;
            }
        };
        if !emit_final_events(inner, ctx, tx, events).await {
            return StreamEnd::Failed(EngineError::Internal(
                "client disconnected before the stream completed".to_owned(),
            ));
        }
    }
    if !ctx.decoder.saw_terminal() && failure.is_none() {
        // The upstream ended without a terminal event: keep valid client
        // framing (synthetic terminal below) but mark the failure in-band
        // and in the ledger. Only when no in-band failure was already
        // emitted — a framing, decode or transport error already crossed to
        // the client, and a second terminal error event would duplicate it
        // (the synthetic MessageStop below still keeps the framing valid).
        let error = EngineError::ProviderDecode(TRUNCATED_STREAM_MESSAGE.to_owned());
        failure = Some(error.clone());
        inner.metrics.record_failure();
        emit_response_failed(
            inner,
            &ctx.request_id,
            Some(ctx.provider_name.as_str()),
            Some(&ctx.ledger_model()),
            &error,
            ctx.start,
        );
        let error = llm_proxy_protocol::core::CoreStreamError::new(
            CoreStreamErrorKind::Upstream,
            TRUNCATED_STREAM_MESSAGE.to_owned(),
        );
        let event = CoreEvent::Error { error };
        capture_lifecycle(ctx, &event);
        if tx.send(event).await.is_err() {
            record_client_cancelled(inner, ctx);
            ctx.settle(inner, false);
            return StreamEnd::Failed(EngineError::Internal(
                "client disconnected before the stream completed".to_owned(),
            ));
        }
    }
    match ctx.decoder.finish() {
        Ok(final_events) => {
            if !emit_final_events(inner, ctx, tx, final_events).await {
                return StreamEnd::Failed(EngineError::Internal(
                    "client disconnected before the stream completed".to_owned(),
                ));
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %ctx.request_id,
                error = %e,
                "provider decoder finish error"
            );
        }
    }
    match failure {
        Some(error) => {
            ctx.settle(inner, false);
            StreamEnd::Failed(error)
        }
        None => {
            record_stream_success(inner, ctx);
            ctx.settle(inner, true);
            StreamEnd::Clean
        }
    }
}

/// The lifecycle facts a replayed stream carries onto its ledger row: the
/// last usage delta and stop reason in the collected events.
fn replay_lifecycle_of(events: &[CoreEvent]) -> (Option<Usage>, Option<StopReason>) {
    let mut usage = None;
    let mut stop_reason = None;
    for event in events {
        match event {
            CoreEvent::UsageDelta { usage: delta } => usage = Some(delta.clone()),
            CoreEvent::MessageStop {
                stop_reason: reason,
                ..
            } => {
                stop_reason = Some(reason.clone());
            }
            _ => {}
        }
    }
    (usage, stop_reason)
}

/// The joiner's replay stream (plan 06 step 2), and the idempotent replay of
/// stored events (plan 06 step 3): yield the shared or stored events in
/// order. The row is written when the stream ends — a completed served copy
/// on a clean end, a failure row when the replayed events end in an error;
/// a caller that drops the tail mid-replay is recorded as a client
/// cancellation, mirroring the leader's task.
struct StreamReplay {
    inner: Arc<EngineInner>,
    request_id: String,
    provider: String,
    model: ModelRef,
    start: Instant,
    phase: ReplayPhase,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
    /// The in-band error the replayed events end with, when they do: the
    /// row records the failure, not a completed hit.
    failure: Option<(&'static str, String)>,
    recorded: bool,
    /// A joining request's idempotency claim: completed with the leader's
    /// collected events when the shared stream ends cleanly, released by
    /// drop otherwise (failures are not stored). An idempotent replay of
    /// stored events never carries one — its key already holds the outcome.
    idempotency: Option<IdempotencyClaim>,
}

enum ReplayPhase {
    /// A share is still being produced: await its terminal outcome.
    Waiting(Arc<StreamShare>),
    /// Yielding the events, in order.
    Replaying(std::vec::IntoIter<CoreEvent>),
    Done,
}

/// The `error_kind` string a replayed in-band error records on its row,
/// matching the ledger's CamelCase kinds. The protocol enum is
/// non-exhaustive; a future kind records under the generic `StreamError`.
fn core_error_kind_name(kind: CoreStreamErrorKind) -> &'static str {
    match kind {
        CoreStreamErrorKind::InvalidRequest => "InvalidRequest",
        CoreStreamErrorKind::Authentication => "Authentication",
        CoreStreamErrorKind::Permission => "Permission",
        CoreStreamErrorKind::RateLimit => "RateLimit",
        CoreStreamErrorKind::Upstream => "Upstream",
        CoreStreamErrorKind::Internal => "Internal",
        _ => "StreamError",
    }
}

impl StreamReplay {
    /// Records the row the replay's end earns: a completed served copy on a
    /// clean end, a failure row when the replayed events end in an error.
    fn record_terminal_row(&mut self) {
        match self.failure.clone() {
            Some((kind, message)) => self.record_replay_failed(kind, &message),
            None => self.record_completed(),
        }
    }

    fn record_completed(&mut self) {
        self.recorded = true;
        self.inner.metrics.record_cache_hit();
        let usage = self.usage.clone().unwrap_or_else(Usage::synthetic_zero);
        self.inner
            .usage
            .record(&ProxyEvent::ResponseCompleted(ResponseCompleted {
                request_id: self.request_id.clone(),
                timestamp: time::OffsetDateTime::now_utc(),
                provider: self.provider.clone(),
                upstream_message_id: None,
                model: self.model.clone(),
                usage,
                // The serving member's pricing is the leader's fact; a
                // served copy records no cost and charges nothing (plan 05
                // step 6).
                cost: None,
                stop_reason: self.stop_reason.clone().unwrap_or(StopReason::EndTurn),
                latency_ms: self
                    .start
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
                credential_id: None,
                cache_hit: true,
            }));
    }

    /// The replay ended in an in-band error: record the failure the leader's
    /// stream also recorded, not a completed served copy.
    fn record_replay_failed(&mut self, kind: &'static str, message: &str) {
        self.recorded = true;
        self.inner.metrics.record_failure();
        self.inner
            .usage
            .record(&ProxyEvent::ResponseFailed(ResponseFailed {
                request_id: self.request_id.clone(),
                timestamp: time::OffsetDateTime::now_utc(),
                provider: Some(self.provider.clone()),
                model: Some(self.model.clone()),
                error_kind: kind.to_owned(),
                message: message.to_owned(),
                http_status: 200,
                latency_ms: self
                    .start
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
            }));
    }

    fn record_cancelled(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        self.inner.metrics.record_client_cancel();
        self.inner
            .usage
            .record(&ProxyEvent::ResponseFailed(ResponseFailed {
                request_id: self.request_id.clone(),
                timestamp: time::OffsetDateTime::now_utc(),
                provider: Some(self.provider.clone()),
                model: Some(self.model.clone()),
                error_kind: "ClientCancelled".to_owned(),
                message: "client disconnected before the stream completed".to_owned(),
                http_status: 200,
                latency_ms: self
                    .start
                    .elapsed()
                    .as_millis()
                    .try_into()
                    .unwrap_or(u64::MAX),
            }));
    }
}

impl Drop for StreamReplay {
    fn drop(&mut self) {
        // A caller that dropped the tail before the replay finished.
        self.record_cancelled();
    }
}

/// Drives one step of a replay: wait for the leader (once), then hand over
/// each event; `None` once the replay is exhausted.
async fn replay_next(mut state: StreamReplay) -> Option<(CoreEvent, StreamReplay)> {
    if let ReplayPhase::Waiting(share) = &state.phase {
        let share = Arc::clone(share);
        let outcome = share.outcome().await;
        let mut events = share.collected();
        if outcome.is_err()
            && !events.iter().any(|event| {
                matches!(
                    event,
                    CoreEvent::Error { .. } | CoreEvent::MessageStop { .. }
                )
            })
        {
            // The shared stream was cut short without a visible terminal:
            // the joiner's replay ends with the same in-band error its
            // leader's client saw.
            events.push(CoreEvent::Error {
                error: llm_proxy_protocol::core::CoreStreamError::new(
                    CoreStreamErrorKind::Upstream,
                    "the shared upstream stream ended early".to_owned(),
                ),
            });
        }
        // The leader ran to a clean terminal: the joiner's idempotency claim
        // stores the very events this replay hands its caller, so a matching
        // retry replays them (a failed leader leaves the claim to drop,
        // releasing the key — failures are not stored). The leader has
        // published its outcome by now, so its own claim already completed.
        if outcome.is_ok()
            && let Some(claim) = state.idempotency.take()
        {
            claim.complete(StoredOutcome {
                response: None,
                events: events.clone(),
            });
        }
        let (usage, stop_reason) = replay_lifecycle_of(&events);
        state.usage = usage;
        state.stop_reason = stop_reason;
        state.failure = replay_failure_of(&events);
        state.phase = ReplayPhase::Replaying(events.into_iter());
    }
    let ReplayPhase::Replaying(events) = &mut state.phase else {
        // Waiting was handled above; Done never reaches the driver again.
        state.record_terminal_row();
        state.phase = ReplayPhase::Done;
        return None;
    };
    match events.next() {
        Some(event) => Some((event, state)),
        None => {
            state.record_terminal_row();
            state.phase = ReplayPhase::Done;
            None
        }
    }
}

/// The in-band error a replay ends with, when it ends in one: the last
/// `Error` event in the collected list, which the leader's stream also
/// recorded as its failure.
fn replay_failure_of(events: &[CoreEvent]) -> Option<(&'static str, String)> {
    events.iter().rev().find_map(|event| match event {
        CoreEvent::Error { error } => {
            Some((core_error_kind_name(error.kind), error.message().to_owned()))
        }
        _ => None,
    })
}

/// A joiner's view of a shared streaming request (plan 06 step 2): the
/// leader's collected events, replayed in order once it finishes. The
/// joiner's idempotency claim, when it holds one, completes with the
/// leader's collected events on a clean end.
fn joined_stream(
    inner: Arc<EngineInner>,
    req: &EngineRequest,
    share: Arc<StreamShare>,
    idempotency: Option<IdempotencyClaim>,
) -> EngineStream {
    EngineStream {
        head: Vec::new(),
        tail: futures::stream::unfold(
            StreamReplay {
                inner,
                request_id: req.request_id.clone(),
                provider: req.provider_name.clone(),
                model: req.core.model.clone(),
                start: req.start,
                phase: ReplayPhase::Waiting(share),
                usage: None,
                stop_reason: None,
                failure: None,
                recorded: false,
                idempotency,
            },
            replay_next,
        )
        .boxed(),
    }
}

/// The idempotent replay of a stored streaming outcome (plan 06 step 3).
fn replay_events_stream(
    inner: Arc<EngineInner>,
    req: &EngineRequest,
    events: Vec<CoreEvent>,
) -> EngineStream {
    let (usage, stop_reason) = replay_lifecycle_of(&events);
    EngineStream {
        head: Vec::new(),
        tail: futures::stream::unfold(
            StreamReplay {
                inner,
                request_id: req.request_id.clone(),
                provider: req.provider_name.clone(),
                model: req.core.model.clone(),
                start: req.start,
                phase: ReplayPhase::Replaying(events.into_iter()),
                usage,
                stop_reason,
                failure: None,
                recorded: false,
                idempotency: None,
            },
            replay_next,
        )
        .boxed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upstream(status: u16, retry_after: Option<Duration>, auth_owner: AuthOwner) -> EngineError {
        EngineError::Upstream {
            status,
            body: String::new(),
            auth_owner,
            retry_after,
        }
    }

    /// The decision as the attempt loop makes it: classification and
    /// credential feedback derived from the error itself.
    fn cools_member(error: &EngineError, another_key_available: bool) -> bool {
        let class = classify(error);
        let feedback = credential_feedback(&class, error);
        member_cooldown_applies(&class, error, feedback, another_key_available)
    }

    #[test]
    fn client_faults_never_cool_the_member() {
        for status in [400, 404, 413, 422] {
            assert!(
                !cools_member(&upstream(status, None, AuthOwner::Operator), false),
                "a {status} is deterministic for this request, not a verdict on the member"
            );
        }
        assert!(
            !cools_member(&upstream(401, None, AuthOwner::Client), false),
            "a rejected passthrough token is the client's to fix"
        );
    }

    #[test]
    fn upstream_faults_cool_the_member() {
        assert!(cools_member(
            &upstream(500, None, AuthOwner::Operator),
            true
        ));
        assert!(cools_member(
            &EngineError::UpstreamTimeout("first byte".to_owned()),
            true
        ));
        assert!(
            cools_member(&upstream(429, None, AuthOwner::Operator), true),
            "a throttle without reset advice is not attributed to one key"
        );
        assert!(
            cools_member(
                &upstream(503, Some(Duration::from_secs(5)), AuthOwner::Operator),
                true
            ),
            "a 5xx with Retry-After is still the member's fault"
        );
        assert!(
            cools_member(
                &upstream(429, Some(Duration::from_secs(5)), AuthOwner::Client),
                true
            ),
            "a throttled passthrough token cannot be pinned on one operator key"
        );
    }

    #[test]
    fn a_rejected_operator_key_cools_the_member_only_when_no_other_key_can_serve() {
        let rejected = upstream(401, None, AuthOwner::Operator);
        let throttled = upstream(429, Some(Duration::from_secs(30)), AuthOwner::Operator);
        for error in [&rejected, &throttled] {
            assert!(
                !cools_member(error, true),
                "with a spare key the member keeps serving: {error:?}"
            );
            assert!(
                cools_member(error, false),
                "with no spare key the member cannot serve: {error:?}"
            );
        }
    }

    #[test]
    fn strip_removes_provider_hints_but_keeps_stream_options() {
        use llm_proxy_protocol::core::{ProviderHints, SamplingOptions};
        let mut raw = serde_json::Map::new();
        raw.insert(
            "stream_options".to_owned(),
            serde_json::json!({"include_usage": true}),
        );
        raw.insert(
            "anthropic_beta".to_owned(),
            serde_json::json!("prompt-caching-2024"),
        );
        let core = CoreRequest {
            model: ModelRef {
                requested: "m".to_owned(),
                upstream: None,
            },
            system: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: Default::default(),
            provider_hints: ProviderHints { raw },
        };

        let outcome = strip_provider_bound_state(&core).expect("a hints-only strip never refuses");
        assert!(outcome.changed, "dropping provider-bound hints is a change");
        assert_eq!(
            outcome.core.provider_hints.raw.get("stream_options"),
            Some(&serde_json::json!({"include_usage": true})),
            "the dialect-neutral hint survives the failover"
        );
        assert!(
            !outcome
                .core
                .provider_hints
                .raw
                .contains_key("anthropic_beta"),
            "provider-bound hints do not cross the failover"
        );
    }
}
