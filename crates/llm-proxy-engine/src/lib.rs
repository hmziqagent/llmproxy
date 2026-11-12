//! llm-proxy-engine: the one request engine the server, the embedded router,
//! the SDK and the bindings all share. It runs the full pipeline — admission,
//! routing, credential selection, guards, cache, the upstream attempt, and
//! settlement — with no HTTP layer: it takes
//! [`core`](llm_proxy_protocol::core) types in and returns a
//! [`CoreResponse`](llm_proxy_protocol::core::CoreResponse) or a stream of
//! [`CoreEvent`](llm_proxy_protocol::core::CoreEvent)s out.
//!
//! Pipeline slots, in order (plan 03):
//!
//! 1. Authenticate the caller — the HTTP layer's job; the engine records the
//!    identity it is given.
//! 2. Apply caller limits and key allowlists — the per-key allowlists are
//!    [`Admission::authorize`], asked as soon as slot 3 names the first
//!    member and before slots 4 and 5; the per-IP rate limit is server
//!    middleware, and per-key windows and budgets are reserved at slot 6.
//! 3. Resolve the route — [`Router`].
//! 4. Claim the idempotency key — [`IdempotencyStore`] (opt-in; 409 only
//!    while the first request runs, replay after).
//! 5. Look up the cache, then join or lead per-caller single-flight —
//!    [`Cache`], [`SingleFlight`], [`StreamShares`].
//! 6. Reserve budget — [`Admission`].
//! 7. Pre-call guards — [`Guard`].
//! 8. Apply request mutations in their fixed phases — the guards' request
//!    rewrites (after slot 7 judged the client's words), then adapter encode
//!    (same-format passthrough or the cross-format field policy).
//! 9. The attempt loop — pick credential ([`CredentialSource`]), encode, send
//!    via the transport, retry or fail over (later plan).
//! 10. Hold the stream until it can be committed — the first-byte window of
//!     [`Engine::execute_stream`].
//! 11. Streaming and post-call guards — [`Guard`].
//! 12. Encode for the client — the server's codec layer.
//! 13. Settle budget, write usage rows, store cache results — [`Admission`]
//!     settlement, [`UsageSink`], [`Cache`].

#![deny(missing_docs)]

pub mod admission;
pub mod admission_backend;
pub mod cache;
pub mod catalog;
pub mod classification;
pub mod error;
pub mod error_policy;
pub mod headers;
pub mod health;
pub mod lifecycle;
pub mod preflight;
pub mod probe;
pub mod traits;
pub mod types;

use std::collections::HashMap;
use std::sync::Arc;

use llm_proxy_adapters::{ProviderAdapterRegistry, ProviderAdapterTarget};
use llm_proxy_core::{AppConfig, Counter, Metrics, ProviderRegistry};
use llm_proxy_storage::EventBus;
use llm_proxy_transport::ProxyClient;

pub use admission::{KeyPolicy, PolicyAdmission};
pub use admission_backend::{
    AdmissionBackend, LeaseId, MemoryAdmissionBackend, MicroUsd, ReservationDemand, Settlement,
    from_micro_usd, to_micro_usd,
};
pub use cache::{
    IdempotencyClaim, IdempotencyStore, MemoryCache, SingleFlight, StoredOutcome, StreamShares,
};
pub use catalog::{ModelCatalogService, write_catalog_atomic};
pub use classification::{ErrorClass, classify, parse_duration_suffix, parse_retry_after};
pub use error::{
    AuthOwner, EngineError, map_provider_error, map_upstream_status, sanitize_upstream_error_body,
};
pub use error_policy::{STATUS_POLICY, StatusCodePolicy, policy_for};
pub use headers::select_forwarded_headers;
pub use health::{AffinityTable, CooldownStore, Cooldowns};
pub use traits::{
    Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket, AllowAllAdmission, Cache,
    CacheEntry, CacheKey, ConfigCredentialSource, ConfigSource, CredentialFeedback, CredentialPick,
    CredentialQuery, CredentialSource, EventBusSink, Guard, GuardDecision, InboundHeaders, NoCache,
    RegistryRouter, ResolvedTarget, RoutePlan, RouteQuery, Router, StaticConfigSource,
    SwappingConfigSource, UnsettledLeaseGuard, UsageSink,
};
pub use types::{ClientDialect, EngineRequest, EngineResponse, EngineStream, InboundRequest};

const MAX_PROVIDER_NAME_LEN: usize = 128;
#[expect(
    clippy::expect_used,
    reason = "the pattern is a compile-time-constant regex literal; a failure is a programming \
              error caught the moment this static is first used, not a runtime input"
)]
static PROVIDER_NAME_REGEX: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^[a-z0-9_-]+$").expect("valid provider name regex")
});

/// Every text block of a core request, in order — the guard-rewrite
/// comparison (plan 09 step 4).
pub(crate) fn guard_texts(
    request: &llm_proxy_protocol::core::CoreRequest,
) -> impl Iterator<Item = String> + '_ {
    use llm_proxy_protocol::core::CoreContent;
    request
        .system
        .iter()
        .chain(
            request
                .messages
                .iter()
                .flat_map(|message| message.content.iter()),
        )
        .filter_map(|content| match content {
            CoreContent::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
}

/// Validates a provider name from the URL path.
///
/// # Errors
///
/// [`EngineError::InvalidProviderName`] when the name is empty, too long, or
/// contains characters outside `[a-z0-9_-]`.
pub fn validate_provider_name(name: &str) -> Result<(), EngineError> {
    if name.is_empty() {
        return Err(EngineError::InvalidProviderName(
            "provider name is empty".to_owned(),
        ));
    }
    if name.len() > MAX_PROVIDER_NAME_LEN {
        return Err(EngineError::InvalidProviderName(format!(
            "provider name exceeds maximum length of {MAX_PROVIDER_NAME_LEN} characters"
        )));
    }
    if !PROVIDER_NAME_REGEX.is_match(name) {
        return Err(EngineError::InvalidProviderName(format!(
            "provider name \"{name}\" contains invalid characters; \
             only lowercase ASCII letters, digits, hyphens, and underscores are allowed"
        )));
    }
    Ok(())
}

/// Shared, immutable engine state.
pub(crate) struct EngineInner {
    pub(crate) config: Arc<dyn ConfigSource>,
    pub(crate) providers: Arc<ProviderRegistry>,
    pub(crate) adapters: Arc<ProviderAdapterRegistry>,
    pub(crate) transport: TransportTable,
    pub(crate) router: Arc<dyn Router>,
    pub(crate) admission: Arc<dyn Admission>,
    pub(crate) cache: Arc<dyn Cache>,
    pub(crate) guards: Vec<Arc<dyn Guard>>,
    pub(crate) credentials: Arc<dyn CredentialSource>,
    pub(crate) usage: EventBusSink,
    pub(crate) catalogs: Arc<ModelCatalogService>,
    pub(crate) metrics: Arc<Metrics>,
    /// Passive upstream health (plan 04 steps 3, 9): provider and
    /// credential cooldowns keyed to real reset times. Behind the
    /// [`Cooldowns`] trait since plan 09, so a replica deployment can hold
    /// one cooldown view across instances.
    pub(crate) health: Arc<dyn Cooldowns>,
    /// Sticky-session pins (plan 04 step 10).
    pub(crate) affinity: Arc<AffinityTable>,
    /// The token counter behind the input estimate admission and preflight
    /// share (plan 05 steps 4 and 8).
    pub(crate) counter: Counter,
    /// Per-caller single-flight for non-streaming requests (plan 06 step 2).
    pub(crate) singleflight: Arc<SingleFlight>,
    /// Per-caller single-flight for streaming requests (plan 06 step 2).
    pub(crate) stream_shares: Arc<StreamShares>,
    /// The opt-in `Idempotency-Key` store (plan 06 step 3); `None` when the
    /// feature is off, in which case the header is ignored.
    pub(crate) idempotency: Option<Arc<IdempotencyStore>>,
}

impl std::fmt::Debug for EngineInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineInner")
            .field("config", &self.config)
            .field("providers", &self.providers)
            .field("adapters", &self.adapters)
            .field("transport", &self.transport)
            .field("router", &self.router)
            .field("admission", &self.admission)
            .field("cache", &self.cache)
            .field("guards", &self.guards.len())
            .field("credentials", &self.credentials)
            .field("usage", &self.usage)
            .field("metrics", &self.metrics)
            .field("health", &self.health)
            .field("affinity", &self.affinity)
            .field("counter", &self.counter)
            .field("singleflight", &self.singleflight)
            .field("stream_shares", &self.stream_shares)
            .field("idempotency", &self.idempotency)
            .finish()
    }
}

/// The transport clients the engine sends with: one shared client plus the
/// per-provider egress-proxy overrides wired from config.
#[derive(Debug, Clone)]
pub struct TransportTable {
    shared: Arc<ProxyClient>,
    egress: Arc<HashMap<String, Arc<ProxyClient>>>,
}

impl TransportTable {
    /// Everyone on one client.
    #[must_use]
    pub fn new(shared: Arc<ProxyClient>) -> Self {
        Self {
            shared,
            egress: Arc::new(HashMap::new()),
        }
    }

    /// Per-provider egress overrides.
    #[must_use]
    pub fn with_egress(mut self, egress: HashMap<String, Arc<ProxyClient>>) -> Self {
        self.egress = Arc::new(egress);
        self
    }

    /// The client for an upstream request to `provider`: its wired
    /// `[provider.egress]` client, or the shared client otherwise.
    pub fn client_for(&self, provider: &str) -> &ProxyClient {
        if let Some(client) = self.egress.get(provider) {
            return client;
        }
        &self.shared
    }
}

/// The request engine. Clone is cheap (`Arc`-shared); build one with
/// [`EngineBuilder`] and share it across handlers, tasks or embedders.
#[derive(Debug, Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

impl Engine {
    /// Run one non-streaming request end to end.
    ///
    /// # Errors
    ///
    /// [`EngineError`] when any stage fails; usage rows are written for every
    /// failure.
    pub async fn execute(&self, request: EngineRequest) -> Result<EngineResponse, EngineError> {
        lifecycle::execute(Arc::clone(&self.inner), request).await
    }

    /// Run one streaming request. Resolves when the first events are decoded
    /// (the pre-commit window) so the caller can still return an HTTP error;
    /// afterwards the returned [`EngineStream`] carries the events.
    ///
    /// # Errors
    ///
    /// [`EngineError`] for any failure before the first byte is committed.
    pub async fn execute_stream(
        &self,
        request: EngineRequest,
    ) -> Result<EngineStream, EngineError> {
        lifecycle::execute_stream(Arc::clone(&self.inner), request).await
    }

    /// The provider registry backing the default router.
    #[must_use]
    pub fn providers(&self) -> &ProviderRegistry {
        &self.inner.providers
    }

    /// The provider registry as an owned `Arc` (plan 07 step 11): handlers
    /// that read the live provider set per request follow the current engine
    /// after a hot reload swaps it.
    #[must_use]
    pub fn provider_registry(&self) -> Arc<ProviderRegistry> {
        Arc::clone(&self.inner.providers)
    }

    /// The provider adapter registry (the open provider set).
    #[must_use]
    pub fn adapters(&self) -> &ProviderAdapterRegistry {
        &self.inner.adapters
    }

    /// The transport clients this engine sends with: handlers that egress
    /// outside the chat lifecycle (the non-chat endpoint families, plan 09
    /// step 7) pick a provider's client here, so `[provider.egress]`
    /// governs every surface and follows hot reloads with the engine.
    #[must_use]
    pub fn transport(&self) -> &TransportTable {
        &self.inner.transport
    }

    /// The model catalog service (discovery cache and enforcement).
    #[must_use]
    pub fn catalogs(&self) -> &ModelCatalogService {
        &self.inner.catalogs
    }

    /// Runtime metrics.
    #[must_use]
    pub fn metrics(&self) -> &Metrics {
        &self.inner.metrics
    }

    /// The usage sink (wrap of the configured event bus).
    #[must_use]
    pub fn usage_sink(&self) -> &EventBusSink {
        &self.inner.usage
    }

    /// The admission step this engine reserves through, shared with
    /// sidecar surfaces (the MCP gateway, plan 09 step 6) so one budget
    /// bounds every surface.
    #[must_use]
    pub fn shared_admission(&self) -> Arc<dyn Admission> {
        Arc::clone(&self.inner.admission)
    }

    /// The configuration snapshot source.
    #[must_use]
    pub fn config_source(&self) -> &Arc<dyn ConfigSource> {
        &self.inner.config
    }
}

/// Builds an [`Engine`] from code: providers, adapters (built-in plus any
/// registered customs), transport, and the extension traits. The TOML
/// [`AppConfig`] is one [`ConfigSource`] (see [`StaticConfigSource`]); an
/// embedder may supply its own.
#[derive(Debug)]
pub struct EngineBuilder {
    config: Arc<dyn ConfigSource>,
    providers: Arc<ProviderRegistry>,
    adapters: Arc<ProviderAdapterRegistry>,
    shared_client: Option<Arc<ProxyClient>>,
    egress: HashMap<String, Arc<ProxyClient>>,
    router: Option<Arc<dyn Router>>,
    admission: Option<Arc<dyn Admission>>,
    cache: Option<Arc<dyn Cache>>,
    guards: Vec<Arc<dyn Guard>>,
    credentials: Option<Arc<dyn CredentialSource>>,
    usage: Option<EventBusSink>,
    catalogs: Option<Arc<ModelCatalogService>>,
    metrics: Option<Arc<Metrics>>,
    counter: Option<Counter>,
    cooldowns: Option<Arc<dyn Cooldowns>>,
}

impl EngineBuilder {
    /// Start from a configuration snapshot, an empty provider registry, the
    /// built-in adapters and a shared transport client built from the
    /// snapshot's `[server.upstream_timeouts]` in [`Self::build`].
    ///
    /// # Panics
    ///
    /// Never in practice: an empty provider registry is always valid.
    #[expect(
        clippy::expect_used,
        reason = "an empty provider registry cannot fail validation; the builder surfaces real config errors from build()"
    )]
    #[must_use]
    pub fn new(config: AppConfig) -> Self {
        let providers = ProviderRegistry::from_providers(vec![])
            .expect("an empty provider registry is always valid");
        Self {
            config: Arc::new(StaticConfigSource::new(Arc::new(config))),
            providers: Arc::new(providers),
            adapters: Arc::new(ProviderAdapterRegistry::builtin()),
            shared_client: None,
            egress: HashMap::new(),
            router: None,
            admission: None,
            cache: None,
            guards: Vec::new(),
            credentials: None,
            usage: None,
            catalogs: None,
            metrics: None,
            counter: None,
            cooldowns: None,
        }
    }

    /// Start from explicit parts.
    #[must_use]
    pub fn from_parts(
        config: Arc<dyn ConfigSource>,
        providers: Arc<ProviderRegistry>,
        adapters: Arc<ProviderAdapterRegistry>,
        shared_client: Arc<ProxyClient>,
    ) -> Self {
        Self {
            config,
            providers,
            adapters,
            shared_client: Some(shared_client),
            egress: HashMap::new(),
            router: None,
            admission: None,
            cache: None,
            guards: Vec::new(),
            credentials: None,
            usage: None,
            catalogs: None,
            metrics: None,
            counter: None,
            cooldowns: None,
        }
    }

    /// Replace the provider registry.
    #[must_use]
    pub fn providers(mut self, providers: Arc<ProviderRegistry>) -> Self {
        self.providers = providers;
        self
    }

    /// Replace the adapter registry (e.g. one with custom protocols
    /// registered).
    #[must_use]
    pub fn adapters(mut self, adapters: Arc<ProviderAdapterRegistry>) -> Self {
        self.adapters = adapters;
        self
    }

    /// Register a custom provider adapter; the provider set is open.
    ///
    /// # Errors
    ///
    /// [`llm_proxy_adapters::RegistrationError`] when the protocol name is
    /// already taken.
    pub fn register_adapter(
        mut self,
        adapter: Arc<dyn llm_proxy_adapters::ProviderAdapter>,
    ) -> Result<Self, llm_proxy_adapters::RegistrationError> {
        let mut registry = (*self.adapters).clone();
        registry.register(adapter)?;
        self.adapters = Arc::new(registry);
        Ok(self)
    }

    /// Wire the per-provider egress clients.
    #[must_use]
    pub fn egress_clients(mut self, egress: HashMap<String, Arc<ProxyClient>>) -> Self {
        self.egress = egress;
        self
    }

    /// Replace the router (default: [`RegistryRouter`]).
    #[must_use]
    pub fn router(mut self, router: Arc<dyn Router>) -> Self {
        self.router = Some(router);
        self
    }

    /// Replace the admission step (default: [`AllowAllAdmission`]).
    #[must_use]
    pub fn admission(mut self, admission: Arc<dyn Admission>) -> Self {
        self.admission = Some(admission);
        self
    }

    /// Replace the cache (default: [`NoCache`]).
    #[must_use]
    pub fn cache(mut self, cache: Arc<dyn Cache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Add a guard (default: none).
    #[must_use]
    pub fn guard(mut self, guard: Arc<dyn Guard>) -> Self {
        self.guards.push(guard);
        self
    }

    /// Replace the credential source (default:
    /// [`ConfigCredentialSource`]).
    #[must_use]
    pub fn credentials(mut self, credentials: Arc<dyn CredentialSource>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Send usage rows to this event bus (default: [`NoopBus`]).
    #[must_use]
    pub fn event_bus(mut self, bus: Arc<dyn EventBus>) -> Self {
        self.usage = Some(EventBusSink::new(bus));
        self
    }

    /// Provide the model catalog service (default: no cache dir).
    #[must_use]
    pub fn catalogs(mut self, catalogs: Arc<ModelCatalogService>) -> Self {
        self.catalogs = Some(catalogs);
        self
    }

    /// Provide the metrics registry (default: fresh).
    #[must_use]
    pub fn metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Provide the token counter the input estimate uses (default: a fresh
    /// [`Counter`]; sharing the server's keeps one BPE cache).
    #[must_use]
    pub fn counter(mut self, counter: Counter) -> Self {
        self.counter = Some(counter);
        self
    }

    /// Provide the cooldown view (plan 09 step 1): the local
    /// [`CooldownStore`] by default, or a shared-store backend from an
    /// optional replica crate, so every replica sees the same member
    /// cooldowns.
    #[must_use]
    pub fn cooldowns(mut self, cooldowns: Arc<dyn Cooldowns>) -> Self {
        self.cooldowns = Some(cooldowns);
        self
    }

    /// Assemble the engine.
    ///
    /// # Errors
    ///
    /// [`EngineError::Internal`] when a configured provider names a protocol
    /// no registered adapter serves, when the configuration snapshot cannot
    /// be loaded, or when the transport client cannot be built.
    pub fn build(self) -> Result<Engine, EngineError> {
        let shared_client = match self.shared_client {
            Some(client) => client,
            None => {
                let config = self.config.load()?;
                Arc::new(
                    ProxyClient::try_with_timeouts(&config.server.upstream_timeouts)
                        .map_err(|error| EngineError::Internal(error.to_string()))?,
                )
            }
        };
        let catalogs = self
            .catalogs
            .unwrap_or_else(|| Arc::new(ModelCatalogService::new(None)));
        let known_protocols = self.adapters.protocol_names();
        self.providers
            .validate_protocols(known_protocols.iter().map(String::as_str))
            .map_err(|error| EngineError::Internal(error.to_string()))?;
        let usage = self.usage.unwrap_or_else(|| {
            EventBusSink::new(Arc::new(llm_proxy_storage::NoopBus) as Arc<dyn EventBus>)
        });
        let health = self
            .cooldowns
            .unwrap_or_else(|| Arc::new(CooldownStore::new()));
        let affinity = Arc::new(AffinityTable::default());
        let idempotency = self
            .config
            .load()
            .ok()
            .filter(|config| config.server.idempotency.enabled)
            .map(|config| {
                Arc::new(IdempotencyStore::new(
                    config.server.idempotency.ttl,
                    config.server.idempotency.max_entries,
                ))
            });
        let router: Arc<dyn Router> = match self.router {
            Some(router) => router,
            None => Arc::new(
                RegistryRouter::new(Arc::clone(&self.providers), Arc::clone(&catalogs))
                    .with_affinity(Arc::clone(&affinity)),
            ),
        };
        let inner = EngineInner {
            config: self.config,
            providers: self.providers,
            adapters: self.adapters,
            transport: TransportTable::new(shared_client).with_egress(self.egress),
            router,
            admission: self
                .admission
                .unwrap_or_else(|| Arc::new(AllowAllAdmission)),
            cache: self.cache.unwrap_or_else(|| Arc::new(NoCache)),
            guards: self.guards,
            credentials: self
                .credentials
                .unwrap_or_else(|| Arc::new(ConfigCredentialSource::new())),
            usage,
            catalogs,
            metrics: self.metrics.unwrap_or_else(|| Arc::new(Metrics::new())),
            health,
            affinity,
            counter: self.counter.unwrap_or_default(),
            singleflight: Arc::new(SingleFlight::new()),
            stream_shares: Arc::new(StreamShares::new()),
            idempotency,
        };
        let inner = Arc::new(inner);
        crate::probe::spawn_pool_probes(&inner);
        Ok(Engine { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_table_picks_the_wired_egress_client_only_for_that_provider() {
        let shared = Arc::new(ProxyClient::try_new().expect("client"));
        let egress = Arc::new(
            ProxyClient::try_with_egress_proxy("socks5h://127.0.0.1:1080")
                .expect("egress proxy client builds"),
        );
        let table = TransportTable::new(Arc::clone(&shared)).with_egress(
            std::collections::HashMap::from([("opencode-go".to_owned(), Arc::clone(&egress))]),
        );

        assert!(
            std::ptr::eq(table.client_for("opencode-go"), egress.as_ref()),
            "the provider with egress must use its wired client"
        );
        assert!(
            std::ptr::eq(table.client_for("opencode-zen"), shared.as_ref()),
            "providers without egress must keep the shared client"
        );
    }

    #[test]
    fn provider_names_are_validated_before_routing() {
        validate_provider_name("openai-go").expect("valid name accepted");
        assert!(validate_provider_name("").is_err());
        assert!(validate_provider_name("UPPER").is_err());
        assert!(validate_provider_name("has space").is_err());
        assert!(validate_provider_name(&"x".repeat(129)).is_err());
    }
}
