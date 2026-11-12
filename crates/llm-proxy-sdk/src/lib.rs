//! The `llm-proxy` facade crate. Published as `llm-proxy-sdk` (the
//! `llm-proxy` name on crates.io is held by an unrelated 2023 placeholder),
//! it is the one dependency an embedder adds: it re-exports the layer crates
//! as modules and adds a [`ProxyBuilder`] that assembles the stack from code
//! instead of TOML.
//!
//! ```no_run
//! use std::time::Duration;
//! use llm_proxy_sdk::prelude::*;
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! # let app_config = core::AppConfig {
//! #     server: core::ServerConfig {
//! #         bind: "127.0.0.1:0".parse()?,
//! #         request_timeout: Duration::from_secs(30),
//! #         shutdown_timeout: Duration::from_secs(5),
//! #         log_level: "info".to_owned(),
//! #         hot_reload: false,
//! #         allowed_origins: None,
//! #         server_name: "embed".to_owned(),
//! #         rate_limit_rpm: 1000,
//! #         trust_forwarded_headers: false,
//! #         log_format: Default::default(),
//! #         admin_bind: None,
//! #         admin_on_main_listener: false,
//! #         endpoint_families: Default::default(),
//! #         upstream_timeouts: Default::default(),
//! #         bare_v1: Default::default(),
//! #         preflight: Default::default(),
//! #         semantic_cache: Default::default(),
//! #         cache: Default::default(),
//! #         idempotency: Default::default(),
//! #         metrics: Default::default(),
//! #     },
//! #     auth: Default::default(),
//! #     storage: None,
//! #     replica: None,
//! #     mcp: None,
//! # };
//! let engine = ProxyBuilder::new()
//!     .app_config(app_config)
//!     .engine()?;
//! # let _ = &engine;
//! # Ok(())
//! # }
//! ```
//!
//! Cargo features: `server` (the axum binding, default), `sqlite` (the
//! persistent ledger, default), `admin` (the read-only usage API),
//! `semantic-cache` (the optional semantic cache, plan 09 step 5),
//! `endpoint-families` (the non-chat routes, plan 09 step 7). With all
//! defaults off you get the engine, adapters and transport only.

#![deny(missing_docs)]

pub use llm_proxy_adapters as adapters;
pub use llm_proxy_core as core;
pub use llm_proxy_engine as engine;
pub use llm_proxy_protocol as protocol;
#[cfg(feature = "server")]
pub use llm_proxy_server as server;
pub use llm_proxy_storage as storage;
pub use llm_proxy_transport as transport;

pub use llm_proxy_engine::{
    Admission, AdmissionClaim, AdmissionOutcome, AdmissionTicket, AllowAllAdmission, Cache,
    CacheEntry, CacheKey, ClientDialect, ConfigCredentialSource, ConfigSource, Engine,
    EngineBuilder, EngineError, EngineRequest, EngineResponse, EngineStream, EventBusSink, Guard,
    GuardDecision, InboundRequest, ModelCatalogService, NoCache, RegistryRouter, ResolvedTarget,
    RouteQuery, Router, StaticConfigSource, TransportTable, UsageSink, write_catalog_atomic,
};

#[cfg(feature = "server")]
pub use llm_proxy_server::{AppState, BuildInfo, build_router};

use std::sync::Arc;

/// Assembles the proxy stack from code (plan 03, step 5): providers,
/// adapters, transport, ledger, then an [`engine::Engine`] for in-process
/// use, or (with the `server` feature) an axum `Router` to mount.
///
/// TOML configuration remains the `llm-proxy` daemon's path; a builder-made
/// config is just another [`ConfigSource`].
#[derive(Debug, Default)]
pub struct ProxyBuilder {
    inner: EngineBuilderSpec,
}

#[derive(Debug, Default)]
struct EngineBuilderSpec {
    providers: Vec<llm_proxy_core::ProviderConfig>,
    adapters: Option<Arc<llm_proxy_adapters::ProviderAdapterRegistry>>,
    event_bus: Option<Arc<dyn storage::EventBus>>,
    metrics: Option<Arc<llm_proxy_core::Metrics>>,
}

impl ProxyBuilder {
    /// An empty builder; call [`Self::app_config`] next.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the configuration snapshot (server settings, auth, storage).
    #[must_use]
    pub fn app_config(self, config: core::AppConfig) -> ConfiguredProxy {
        ConfiguredProxy {
            config,
            inner: self.inner,
        }
    }

    /// Register a custom provider adapter (the open provider set).
    pub fn adapter(
        mut self,
        adapter: Arc<dyn llm_proxy_adapters::ProviderAdapter>,
    ) -> Result<Self, llm_proxy_adapters::RegistrationError> {
        let mut registry =
            self.inner.adapters.clone().unwrap_or_else(|| {
                Arc::new(llm_proxy_adapters::ProviderAdapterRegistry::builtin())
            });
        let mutable = Arc::make_mut(&mut registry);
        mutable.register(adapter)?;
        self.inner.adapters = Some(registry);
        Ok(self)
    }

    /// Send usage rows to this event bus.
    #[must_use]
    pub fn event_bus(mut self, bus: Arc<dyn storage::EventBus>) -> Self {
        self.inner.event_bus = Some(bus);
        self
    }

    /// Share a metrics registry.
    #[must_use]
    pub fn metrics(mut self, metrics: Arc<core::Metrics>) -> Self {
        self.inner.metrics = Some(metrics);
        self
    }

    /// Add one provider.
    #[must_use]
    pub fn provider(mut self, provider: core::ProviderConfig) -> Self {
        self.inner.providers.push(provider);
        self
    }
}

/// A [`ProxyBuilder`] once its config snapshot is set.
#[derive(Debug)]
pub struct ConfiguredProxy {
    config: core::AppConfig,
    inner: EngineBuilderSpec,
}

impl ConfiguredProxy {
    /// Add one provider.
    #[must_use]
    pub fn provider(mut self, provider: core::ProviderConfig) -> Self {
        self.inner.providers.push(provider);
        self
    }

    /// Build the engine.
    ///
    /// # Errors
    ///
    /// [`EngineError`] when the provider set names protocols no registered
    /// adapter serves, or the transport client cannot be built from the
    /// config's `[server.upstream_timeouts]`.
    pub fn engine(self) -> Result<Engine, EngineError> {
        let registry = core::ProviderRegistry::from_providers(self.inner.providers)
            .map_err(|error| EngineError::Internal(error.to_string()))?;
        let client =
            transport::ProxyClient::try_with_timeouts(&self.config.server.upstream_timeouts)
                .map_err(|error| EngineError::Internal(error.to_string()))?;
        let mut builder = EngineBuilder::from_parts(
            Arc::new(StaticConfigSource::new(Arc::new(self.config))),
            Arc::new(registry),
            self.inner.adapters.unwrap_or_else(|| {
                Arc::new(llm_proxy_adapters::ProviderAdapterRegistry::builtin())
            }),
            Arc::new(client),
        );
        if let Some(bus) = self.inner.event_bus {
            builder = builder.event_bus(bus);
        }
        if let Some(metrics) = self.inner.metrics {
            builder = builder.metrics(metrics);
        }
        builder.build()
    }
}

/// The names most embedders need, in one glob.
pub mod prelude {
    #[cfg(feature = "server")]
    pub use crate::{AppState, BuildInfo, build_router, server};
    pub use crate::{
        ClientDialect, ConfiguredProxy, Engine, EngineBuilder, EngineError, EngineRequest,
        InboundRequest, ProxyBuilder, adapters, core, engine, protocol, storage, transport,
    };
    pub use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
    pub use llm_proxy_core::{
        AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
        ProviderRouteKind, ProviderRoutesConfig,
    };
    pub use llm_proxy_storage::{EventBus, NoopBus, ProxyEvent};
}
