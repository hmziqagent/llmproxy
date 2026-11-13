//! The engine's extension points (audit conflict 13): every feature that
//! needs process-shared or backend-specific state implements one of these
//! traits instead of living in the axum layer. Each trait ships a local,
//! in-process default so a single node runs with no external services; the
//! Redis/Postgres-backed implementations arrive as optional crates and slot
//! in behind the same trait.
//!
//! [`EventBus`](llm_proxy_storage::EventBus) (usage events) and
//! [`Tokenizer`](llm_proxy_core::Counter) (token estimation) already existed
//! and are not duplicated here.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use llm_proxy_core::{AffinityConfig, AppConfig, ProviderRegistry, ProviderRouteKind, RetryConfig};
use llm_proxy_protocol::core::{CoreEvent, CoreRequest, CoreResponse};
use secrecy::SecretString;

use crate::catalog::ModelCatalogService;

/// The ladder for a route that did not opt into retries: one attempt, no
/// backoff, no hold.
fn single_attempt() -> RetryConfig {
    RetryConfig {
        max_attempts: 1,
        backoff: std::time::Duration::ZERO,
        cooldown: std::time::Duration::from_secs(30),
        total_timeout: std::time::Duration::ZERO,
        hold: std::time::Duration::ZERO,
    }
}
use crate::error::EngineError;
use crate::health::AffinityTable;

/// Resolves a route to a plan of candidate targets (pipeline slot 3). The
/// default implementation is the TOML/config-loaded [`ProviderRegistry`]
/// plus catalog enforcement, and it understands pools: a pool route resolves
/// to its members in failover order with the pool's ladder policy.
#[async_trait]
pub trait Router: Debug + Send + Sync {
    /// Resolve `provider` + route kind + requested model to the candidate
    /// targets the attempt loop tries in order, plus the ladder policy that
    /// governs them.
    async fn resolve(&self, query: &RouteQuery<'_>) -> Result<RoutePlan, EngineError>;
}

/// Everything the router is asked per request.
#[derive(Debug, Clone, Copy)]
pub struct RouteQuery<'a> {
    /// The provider or pool named in the URL.
    pub provider: &'a str,
    /// The endpoint family the client called.
    pub route_kind: ProviderRouteKind,
    /// The model name the client asked for.
    pub requested_model: &'a str,
    /// The session id attributed to this request (plan 04 step 10), when
    /// one is known: the sticky-affinity input. `None` disables pinning for
    /// this request.
    pub session_id: Option<&'a str>,
}

/// The routing decision: the ordered candidate targets plus the ladder
/// policy that applies to them.
#[derive(Clone)]
pub struct RoutePlan {
    /// The route the client addressed: a provider name or a pool name.
    pub route: String,
    /// Candidate targets in failover order. A direct provider route has
    /// exactly one; a pool has one per member.
    pub candidates: Vec<ResolvedTarget>,
    /// The retry/failover ladder for this route: attempts per member, one
    /// deadline across all attempts, and the streaming hold budget.
    pub retry: RetryConfig,
    /// Session affinity policy for this route (pools only).
    pub affinity: AffinityConfig,
}

impl Debug for RoutePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutePlan")
            .field("route", &self.route)
            .field("candidates", &self.candidates.len())
            .field("retry", &self.retry)
            .field("affinity", &self.affinity)
            .finish()
    }
}

/// The routing decision for one candidate: which adapter config serves this
/// member.
#[derive(Clone)]
pub struct ResolvedTarget {
    /// The provider config's resolved adapter target (endpoint, protocol,
    /// auth style, static headers, model aliasing, `lossy`, and the inbound
    /// header forwarding allowlist).
    pub target: llm_proxy_core::ProviderAdapterTargetConfig,
}

impl Debug for ResolvedTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The target carries the upstream credential; never include it.
        f.debug_struct("ResolvedTarget")
            .field("provider_name", &self.target.provider_name)
            .field("protocol", &self.target.protocol)
            .field("adapter_name", &self.target.adapter_name)
            .field("upstream_model", &self.target.upstream_model)
            .finish()
    }
}

/// The default [`Router`]: the loaded [`ProviderRegistry`], with catalog
/// enforcement for providers that opt in, pool expansion for pool routes,
/// and (when constructed with one) sticky-affinity reordering of pool
/// candidates.
#[derive(Debug)]
pub struct RegistryRouter {
    registry: Arc<ProviderRegistry>,
    catalogs: Arc<ModelCatalogService>,
    affinity: Option<Arc<AffinityTable>>,
}

impl RegistryRouter {
    /// Resolves one provider through the registry with catalog enforcement.
    async fn resolve_one(
        &self,
        query: &RouteQuery<'_>,
        provider: &str,
    ) -> Result<ResolvedTarget, EngineError> {
        let target = self
            .registry
            .resolve_provider_route(provider, query.route_kind, query.requested_model)
            .map_err(EngineError::from_route_error)?;
        if let Some(provider) = self.registry.get(provider)
            && provider
                .catalog
                .as_ref()
                .is_some_and(|catalog| catalog.enforce)
        {
            let allowed = self
                .catalogs
                .contains_model(provider, &target.upstream_model, &target.protocol)
                .await
                .map_err(|error| EngineError::Internal(error.to_string()))?;
            if !allowed {
                return Err(EngineError::ModelNotAllowed(target.upstream_model.clone()));
            }
        }
        Ok(ResolvedTarget { target })
    }

    /// Wrap a registry and the catalog service.
    #[must_use]
    pub fn new(registry: Arc<ProviderRegistry>, catalogs: Arc<ModelCatalogService>) -> Self {
        Self {
            registry,
            catalogs,
            affinity: None,
        }
    }

    /// Give the router the shared affinity table, so a session pinned to a
    /// pool member is tried first while that member is healthy.
    #[must_use]
    pub fn with_affinity(mut self, affinity: Arc<AffinityTable>) -> Self {
        self.affinity = Some(affinity);
        self
    }
}

#[async_trait]
impl Router for RegistryRouter {
    async fn resolve(&self, query: &RouteQuery<'_>) -> Result<RoutePlan, EngineError> {
        // Pool routes: resolve every member, apply the pool's ladder, and
        // reorder for sticky affinity.
        if let Some(pool) = self.registry.pool(query.provider) {
            // Resolve each member (a member that cannot serve the route kind
            // is a config-time fact, so it fails the request rather than
            // being silently skipped — the operator asked for it as fallback
            // capacity).
            let mut candidates = Vec::with_capacity(pool.members.len());
            for member in &pool.members {
                candidates.push(self.resolve_one(query, member).await?);
            }
            let mut plan = RoutePlan {
                route: pool.name.clone(),
                candidates,
                retry: pool.retry.clone(),
                affinity: pool.affinity.clone(),
            };
            if pool.affinity.sticky
                && let (Some(table), Some(session)) = (&self.affinity, query.session_id)
                && let Some(pinned) = table.pinned_provider(session)
            {
                plan.candidates
                    .sort_by_key(|candidate| u8::from(candidate.target.provider_name != pinned));
            }
            return Ok(plan);
        }

        // Direct provider route: exactly one candidate, the provider's own
        // opt-in ladder, and no failover beyond it. Without `[provider.retry]`
        // the ladder is exactly one attempt — the pre-plan behaviour.
        let resolved = self.resolve_one(query, query.provider).await?;
        let retry = self
            .registry
            .get(query.provider)
            .and_then(|p| p.retry.clone())
            .unwrap_or_else(single_attempt);
        Ok(RoutePlan {
            route: query.provider.to_owned(),
            candidates: vec![resolved],
            retry,
            affinity: AffinityConfig::default(),
        })
    }
}

/// One admission step: authorize the caller, reserve budget up front, then
/// settle or refund (audit conflict 2's resolution). The default admits
/// everything; [`PolicyAdmission`](crate::admission::PolicyAdmission)
/// enforces per-key allowlists, rate windows, spend budgets and concurrency
/// caps.
///
/// `authorize` (slot 2) is the verdict that needs no accounting; `reserve`
/// (slot 6) is async (it may consult a backend budget store); `settle` is
/// synchronous fire-and-forget accounting, mirroring the contract of
/// [`EventBus::emit`](llm_proxy_storage::EventBus::emit): implementations
/// must not block the calling task.
#[async_trait]
pub trait Admission: Debug + Send + Sync {
    /// Slot 2: whether this caller may run this request at all — the
    /// per-key allowlists. Called before the idempotency claim, the cache
    /// lookup and single-flight (slots 4 and 5), so a key restricted after
    /// an answer was stored for it is refused instead of being served that
    /// answer. No accounting: nothing is reserved here and nothing is
    /// settled, so implementations must be cheap and side-effect free. The
    /// claim carries no input estimate yet (`estimated_input_tokens` is 0).
    ///
    /// The verdict covers the member the ladder starts on. A failover to a
    /// later member with a different upstream model is not authorized
    /// again.
    async fn authorize(&self, claim: &AdmissionClaim<'_>) -> Result<(), EngineError> {
        let _ = claim;
        Ok(())
    }

    /// Reserve capacity (budget, tokens, concurrency) before egress. Returns
    /// an opaque ticket handed back at [`Admission::settle`].
    async fn reserve(&self, claim: &AdmissionClaim<'_>) -> Result<AdmissionTicket, EngineError> {
        let _ = claim;
        Ok(AdmissionTicket::default())
    }

    /// Settle or refund the reservation once the attempt outcome is known.
    /// Called exactly once per successful [`Admission::reserve`].
    fn settle(&self, ticket: &AdmissionTicket, outcome: &AdmissionOutcome<'_>) {
        let _ = (ticket, outcome);
    }

    /// Move the reservation's in-flight slot to the member the ladder is
    /// about to try (failover or a cooldown reorder, plan 05 step 7): the
    /// serving member's concurrency cap is checked and its slot held, and
    /// the previously reserved member's slot is released. The reservation's
    /// budget estimate is re-priced at the serving member's rates, so a
    /// failover to a pricier member grows the day reservation (and a
    /// pricier estimate that no longer fits refuses like a saturated cap).
    /// Fails when the new member's cap is saturated or the re-priced
    /// reservation would exceed the budget, in which case the ladder skips
    /// that member. The default keeps whatever `reserve` keyed on.
    async fn rekey(
        &self,
        ticket: &AdmissionTicket,
        member: &str,
        model: &str,
    ) -> Result<(), EngineError> {
        let _ = (ticket, member, model);
        Ok(())
    }
}

/// What is being admitted.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionClaim<'a> {
    /// The caller identity (key id or `"anonymous"`).
    pub caller: &'a str,
    /// The route the client addressed: a provider name or a pool name (what
    /// per-key provider allowlists match).
    pub provider: &'a str,
    /// The member provider the reservation starts on — the first member
    /// the ladder will try, cooldown reorder included: the initial pricing
    /// and concurrency-cap key. [`Admission::rekey`] moves the in-flight
    /// slot when the ladder fails over. Equals `provider` on a direct
    /// route.
    pub member: &'a str,
    /// The upstream model the first attempt encodes for (the billing key).
    pub model: &'a str,
    /// The model name the client asked for (what per-key model allowlists
    /// match, alongside `model`).
    pub requested_model: &'a str,
    /// SHA-256 of the raw client body (never the body itself).
    pub body_hash: &'a str,
    /// The engine's input token estimate (plan 05 step 8's counter): BPE for
    /// known OpenAI models, chars/4 otherwise; tools, tool results and
    /// images are not counted. Zero in the claim [`Admission::authorize`]
    /// sees: nothing is estimated before the cache lookup.
    pub estimated_input_tokens: u64,
    /// The request's `max_tokens`, when it set one.
    pub max_output_tokens: Option<u64>,
}

/// An admission reservation; implementations carry what they need to settle.
#[derive(Debug, Default, Clone)]
pub struct AdmissionTicket {
    /// Reserved units, if the implementation accounts in tokens/credits.
    pub reserved: Option<u64>,
    /// An implementation-defined handle to the reservation record, so
    /// settlement can find what was reserved. `None` when nothing was.
    pub lease: Option<crate::admission_backend::LeaseId>,
    /// The caller the reservation was made for, so settlement diagnostics
    /// can name it without the backend knowing the policy tables.
    pub caller: Option<String>,
    /// The input estimate the reservation was priced from, so
    /// [`Admission::rekey`] can re-price it at the serving member's rates
    /// on failover.
    pub estimated_input_tokens: u64,
    /// The request's `max_tokens`, when it set one — the other input the
    /// failover re-pricing needs.
    pub max_output_tokens: Option<u64>,
}

/// Releases a reservation whose holder was dropped before settling: the
/// drop backstop of [`Admission::reserve`]. A non-streaming request holds
/// its ticket inside the handler future, so a client disconnect or a
/// per-family timeout that drops that future mid-ladder would otherwise
/// leave the lease unsettled forever — its in-flight slot held, its budget
/// reservation kept. Hold this next to every reserve, and [`Self::disarm`]
/// on each path that settles explicitly; the drop then settles as failed
/// (no usage, no charge), exactly like the streaming supervisor does for a
/// task that never completed.
#[derive(Debug)]
pub struct UnsettledLeaseGuard {
    admission: Option<Arc<dyn Admission>>,
    ticket: AdmissionTicket,
}

impl UnsettledLeaseGuard {
    /// Arms the backstop for `ticket`. Keep the guard alive for as long as
    /// the future that owns the reservation.
    pub fn new(admission: Arc<dyn Admission>, ticket: &AdmissionTicket) -> Self {
        Self {
            admission: Some(admission),
            ticket: ticket.clone(),
        }
    }

    /// Cancels the backstop: an explicit settlement has run (or is about
    /// to, with no await in between), so dropping the guard must not settle
    /// a second time.
    pub fn disarm(&mut self) {
        self.admission = None;
    }
}

impl Drop for UnsettledLeaseGuard {
    fn drop(&mut self) {
        if let Some(admission) = self.admission.take() {
            admission.settle(
                &self.ticket,
                &AdmissionOutcome {
                    succeeded: false,
                    usage: None,
                    cost: None,
                },
            );
        }
    }
}

/// The outcome handed back at settlement.
#[derive(Debug)]
pub struct AdmissionOutcome<'a> {
    /// Whether the request completed.
    pub succeeded: bool,
    /// Final usage, when the attempt produced one.
    pub usage: Option<&'a llm_proxy_protocol::core::Usage>,
    /// The cost the engine computed from provider-reported usage and the
    /// member's pricing, when both exist: what the provider bills, which is
    /// what a budget charges (plan 05 step 6). `None` with usage present
    /// means the model is unpriced.
    pub cost: Option<&'a llm_proxy_protocol::core::Cost>,
}

/// The default [`Admission`]: everything is admitted, nothing is accounted.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllowAllAdmission;

#[async_trait]
impl Admission for AllowAllAdmission {}

/// Response cache lookup (slot 5) and store (slot 13). The default caches
/// nothing. Keys are tenant-safe by construction: caller, provider, route,
/// model and body digest.
#[async_trait]
pub trait Cache: Debug + Send + Sync {
    /// Look up a stored response for this key.
    async fn lookup(&self, key: &CacheKey<'_>) -> Result<Option<CacheEntry>, EngineError> {
        let _ = key;
        Ok(None)
    }

    /// Store a completed response under this key.
    async fn store(&self, key: &CacheKey<'_>, entry: &CacheEntry) {
        let _ = (key, entry);
    }
}

/// A cache key: identifies one normalized request, caller-scoped. The
/// exact-match cache keys on `body_hash`; a semantic cache (plan 09 step
/// 5) instead embeds the request's text from `request`, with everything
/// else — caller, provider, route, model — still scoping the key.
#[derive(Debug)]
pub struct CacheKey<'a> {
    /// The caller identity (key id or `"anonymous"`).
    pub caller: &'a str,
    /// Provider the request routed to.
    pub provider: &'a str,
    /// The endpoint family.
    pub route_kind: ProviderRouteKind,
    /// The model name sent upstream (after aliasing).
    pub model: &'a str,
    /// SHA-256 of the raw client body.
    pub body_hash: &'a str,
    /// Whether the client asked for a stream.
    pub streaming: bool,
    /// The normalized request itself (plan 09 step 5): what a semantic
    /// cache embeds instead of the exact body hash.
    pub request: &'a CoreRequest,
}

/// A cached response.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The stored core response (a stream is replayed from its collected
    /// events).
    pub response: CoreResponse,
    /// The collected core events of a streamed response, in order.
    pub events: Vec<CoreEvent>,
}

/// The default [`Cache`]: nothing is stored or answered.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoCache;

#[async_trait]
impl Cache for NoCache {}

/// A guardrail hook (audit conflict 9). A guard inspects the normalized
/// request before egress, the completed non-streaming response, and each
/// streamed event — and may rewrite request texts and response content in
/// place (redaction/masking, plan 09 step 4). A guard that cannot keep its
/// placeholders consistent across stream chunks must reject streaming
/// requests instead of masking them (see [`Guard::inspect_request`]).
#[async_trait]
pub trait Guard: Debug + Send + Sync {
    /// Inspect the normalized request before egress. A masking guard
    /// whose rule cannot hold across stream chunks rejects a request with
    /// `stream: true` here rather than masking it unsafely.
    async fn inspect_request(&self, request: &CoreRequest) -> Result<GuardDecision, EngineError> {
        let _ = request;
        Ok(GuardDecision::Allow)
    }

    /// Rewrite the request in place before egress (after
    /// [`Guard::inspect_request`] allowed it). The default changes
    /// nothing.
    async fn transform_request(&self, request: &mut CoreRequest) -> Result<(), EngineError> {
        let _ = request;
        Ok(())
    }

    /// Inspect a completed non-streaming response.
    async fn inspect_response(
        &self,
        response: &CoreResponse,
    ) -> Result<GuardDecision, EngineError> {
        let _ = response;
        Ok(GuardDecision::Allow)
    }

    /// Rewrite a completed non-streaming response in place (after
    /// [`Guard::inspect_response`] allowed it): the redacted form is what
    /// the client receives and what the cache stores.
    async fn transform_response(&self, response: &mut CoreResponse) -> Result<(), EngineError> {
        let _ = response;
        Ok(())
    }

    /// Inspect each streamed core event as it is decoded.
    async fn inspect_event(&self, event: &CoreEvent) -> Result<GuardDecision, EngineError> {
        let _ = event;
        Ok(GuardDecision::Allow)
    }
}

/// What a guard decides.
#[derive(Debug, Clone)]
pub enum GuardDecision {
    /// Let the request/response/event through.
    Allow,
    /// Reject: the message is surfaced to the client as the rejection reason.
    Reject(String),
}

/// Supplies the upstream credential for an attempt (slot 9's credential
/// pick). The default is the static key from provider config, with
/// client-token passthrough when the provider opts in; providers configured
/// with more than one key rotate across the pool with failure feedback
/// (plan 04 step 5, backlog 31).
#[async_trait]
pub trait CredentialSource: Debug + Send + Sync {
    /// Pick the credential for one attempt.
    ///
    /// `query.keys` is the member's resolved key pool (`api_keys` plus the
    /// base `api_key`, or just the base key); `query.avoid` names the
    /// credential that just failed, so a rotating implementation can move
    /// on even statelessly. Returns the secret plus the id recorded on
    /// ledger rows (never the secret itself).
    async fn pick(&self, query: &CredentialQuery<'_>) -> Result<CredentialPick, EngineError> {
        if query.passthrough {
            return Ok(CredentialPick {
                secret: query.inbound.cloned().ok_or(EngineError::Unauthorized)?,
                credential_id: None,
            });
        }
        let secret = query.keys.first().cloned().ok_or(EngineError::Internal(
            "no upstream credential configured".to_owned(),
        ))?;
        Ok(CredentialPick {
            secret,
            credential_id: None,
        })
    }

    /// Feedback after the attempt ran (plan 04 step 5): which credential was
    /// used and how it ended. Default: ignore.
    fn feedback(
        &self,
        _provider: &str,
        _credential_id: Option<&str>,
        _outcome: CredentialFeedback,
    ) {
    }
}

/// What a credential pick hands back.
#[derive(Debug, Clone)]
pub struct CredentialPick {
    /// The secret to send upstream.
    pub secret: SecretString,
    /// The id recorded on usage rows identifying which pool key was used
    /// (for example `key-2`); `None` when the member has a single key or
    /// passthrough auth supplied it.
    pub credential_id: Option<String>,
}

/// What the attempt loop knows when picking a credential.
#[derive(Debug)]
pub struct CredentialQuery<'a> {
    /// The member provider the attempt targets.
    pub provider: &'a str,
    /// The member's resolved credential pool.
    pub keys: &'a [SecretString],
    /// Whether the member forwards the client's own token.
    pub passthrough: bool,
    /// The client's token when one arrived.
    pub inbound: Option<&'a SecretString>,
    /// The id of the credential that failed the previous attempt against
    /// this member, if that attempt used one.
    pub avoid: Option<&'a str>,
    /// Ids of this member's credentials still cooling down from an earlier
    /// rejection (the engine's [`CooldownStore`](crate::health::CooldownStore)
    /// view). A pick skips them while another key exists; they are the ids
    /// the source itself returned, so any naming scheme round-trips.
    pub cooling: &'a [String],
}

/// How an attempt that used a credential ended, for
/// [`CredentialSource::feedback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialFeedback {
    /// The attempt succeeded.
    Succeeded,
    /// The credential was rejected or throttled (401/403/429): a pool should
    /// let this key cool down.
    Rejected,
    /// The member failed transiently (5xx/timeout): the key is probably
    /// fine, but the member cools down.
    TransientFailure,
}

/// The default [`CredentialSource`]: the configured static key, or the
/// client's token on `passthrough_auth` providers (which requires one).
/// With a multi-key pool it rotates round-robin, avoiding the key that just
/// failed and the keys the engine's
/// [`CooldownStore`](crate::health::CooldownStore) reports as cooling
/// (`query.cooling`, fed by the attempt loop's feedback) while another key
/// exists.
#[derive(Debug)]
pub struct ConfigCredentialSource {
    last_pick: Mutex<HashMap<String, usize>>,
}

use std::sync::Mutex;

impl Default for ConfigCredentialSource {
    fn default() -> Self {
        Self {
            last_pick: Mutex::new(HashMap::new()),
        }
    }
}

impl ConfigCredentialSource {
    /// A fresh source with no rotation memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CredentialSource for ConfigCredentialSource {
    async fn pick(&self, query: &CredentialQuery<'_>) -> Result<CredentialPick, EngineError> {
        if query.passthrough {
            return Ok(CredentialPick {
                secret: query.inbound.cloned().ok_or(EngineError::Unauthorized)?,
                credential_id: None,
            });
        }
        let Some(first) = query.keys.first() else {
            return Err(EngineError::Internal(
                "no upstream credential configured".to_owned(),
            ));
        };
        if query.keys.len() == 1 {
            // Single-key member: the base key, no id to record.
            return Ok(CredentialPick {
                secret: first.clone(),
                credential_id: None,
            });
        }
        // Multi-key pool (the base key plus extras): walk the rotation from
        // the key after the last pick. The key that just failed is skipped
        // first, cooling keys next; when every key is excluded the plain
        // rotation stands — cooldowns reorder, they never refuse service.
        let total = query.keys.len();
        let mut guard = self.last_pick.lock().unwrap_or_else(|e| e.into_inner());
        let last = guard.get(query.provider).copied().unwrap_or(total - 1);
        let start = (last + 1) % total;
        let avoid = query.avoid.and_then(parse_key_index);
        let cooling: Vec<usize> = query
            .cooling
            .iter()
            .filter_map(|id| parse_key_index(id))
            .collect();
        let rotation = (0..total).map(|step| (start + step) % total);
        let next = rotation
            .clone()
            .find(|index| avoid != Some(*index) && !cooling.contains(index))
            .or_else(|| rotation.clone().find(|index| avoid != Some(*index)))
            .unwrap_or(start);
        guard.insert(query.provider.to_owned(), next);
        let secret = query.keys.get(next).cloned().ok_or(EngineError::Internal(
            "credential pool index out of range".to_owned(),
        ))?;
        Ok(CredentialPick {
            secret,
            credential_id: Some(format!("key-{next}")),
        })
    }
}

/// Pool credential ids are `key-<index>`; parse the index back out.
fn parse_key_index(id: &str) -> Option<usize> {
    id.strip_prefix("key-")?.parse().ok()
}

/// Where usage rows go (slot 13). The default forwards to the
/// [`EventBus`](llm_proxy_storage::EventBus) the app configured (SQLite
/// ledger or [`NoopBus`](llm_proxy_storage::NoopBus)).
pub trait UsageSink: Debug + Send + Sync {
    /// Record one lifecycle event.
    fn record(&self, event: &llm_proxy_storage::ProxyEvent);
}

/// The default [`UsageSink`]: delegate to an event bus.
#[derive(Debug)]
pub struct EventBusSink {
    bus: Arc<dyn llm_proxy_storage::EventBus>,
}

impl EventBusSink {
    /// Wrap an event bus.
    #[must_use]
    pub fn new(bus: Arc<dyn llm_proxy_storage::EventBus>) -> Self {
        Self { bus }
    }

    /// The wrapped bus.
    #[must_use]
    pub fn bus(&self) -> &Arc<dyn llm_proxy_storage::EventBus> {
        &self.bus
    }
}

impl UsageSink for EventBusSink {
    fn record(&self, event: &llm_proxy_storage::ProxyEvent) {
        self.bus.emit(event);
    }
}

/// Supplies the configuration snapshot (audit conflict 11). TOML files are
/// one source; programmatic config is another, and a record-store backend
/// (key API, hot reload) is a later plan implementing this same trait.
///
/// One stated precedence governs what any snapshot contains —
/// **CLI > env > config file > default** — resolved before the snapshot is
/// built. A swap through [`SwappingConfigSource::swap`] replaces the whole
/// snapshot; settings are never merged into a live one, and holders of an
/// old snapshot's `Arc` keep it alive.
pub trait ConfigSource: Debug + Send + Sync {
    /// The current configuration snapshot.
    fn load(&self) -> Result<Arc<AppConfig>, EngineError>;
}

/// A [`ConfigSource`] over one immutable [`AppConfig`] snapshot — what a
/// TOML file or a builder produces.
#[derive(Debug, Clone)]
pub struct StaticConfigSource {
    config: Arc<AppConfig>,
}

impl StaticConfigSource {
    /// Wrap a config snapshot.
    #[must_use]
    pub fn new(config: Arc<AppConfig>) -> Self {
        Self { config }
    }
}

impl ConfigSource for StaticConfigSource {
    fn load(&self) -> Result<Arc<AppConfig>, EngineError> {
        Ok(Arc::clone(&self.config))
    }
}

/// A [`ConfigSource`] whose immutable snapshot can be swapped atomically
/// (plan 07 step 10, audit conflict 11). The engine reads the current
/// snapshot per pipeline stage, so a swap takes effect for every request
/// that has not yet read the affected setting; in-flight requests that
/// already read the old snapshot finish on it. One writer swaps; any number
/// of readers load.
#[derive(Debug)]
pub struct SwappingConfigSource {
    current: std::sync::RwLock<Arc<AppConfig>>,
}

impl SwappingConfigSource {
    /// Start from an initial snapshot.
    #[must_use]
    pub fn new(initial: Arc<AppConfig>) -> Self {
        Self {
            current: std::sync::RwLock::new(initial),
        }
    }

    /// Replace the snapshot wholesale, returning the previous one so the
    /// caller can diff or roll back. This never merges: the replacement must
    /// be a complete, validated [`AppConfig`].
    pub fn swap(&self, next: Arc<AppConfig>) -> Arc<AppConfig> {
        let mut guard = self
            .current
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::mem::replace(&mut *guard, next)
    }
}

impl ConfigSource for SwappingConfigSource {
    fn load(&self) -> Result<Arc<AppConfig>, EngineError> {
        let guard = self
            .current
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(Arc::clone(&guard))
    }
}

/// Inbound headers as the engine sees them: names lowercased, UTF-8 values
/// only. The server performs the conversion from its HTTP layer.
pub type InboundHeaders = HashMap<String, String>;
