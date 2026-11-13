//! The optional admin surface for llm-proxy (plan 09 steps 2 and 3):
//! key management, usage and config views on the dedicated admin
//! listener, a small embedded web UI over the same JSON API, and the same
//! API exposed as MCP tools behind the same admin token (backlog 35).
//!
//! Everything answers to `Authorization: Bearer <admin-token>` — the MCP
//! endpoint reuses the admin API's auth, it does not have its own — and
//! every response is `Cache-Control: no-store` (backlog 37).
//!
//! The crate is opt-in: `llm-proxy-server` and the daemon build without
//! it, and it never becomes a dependency of the engine.

#![deny(missing_docs)]

mod auth;
mod config_view;
mod keys;
mod mcp;
mod routes;
mod ui;

pub use config_view::sanitize_config;
pub use keys::{IssuedKey, KeyPolicyInput, KeyRecord, ManagedKeyError, ManagedKeys};
pub use mcp::MCP_PROTOCOL_VERSION;

use std::sync::Arc;

use axum::Router;
use llm_proxy_storage::UsageReader;

use crate::auth::AdminAuth;

/// The admin surface: what it serves depends on which pieces the builder
/// was given — keys always need a [`ManagedKeys`], the usage views need
/// the ledger reader, the config view needs the running [`AppConfig`],
/// and `/metrics` needs the runtime [`llm_proxy_core::Metrics`].
#[derive(Clone)]
pub struct AdminSurface {
    pub(crate) auth: Arc<AdminAuth>,
    pub(crate) keys: Option<Arc<ManagedKeys>>,
    pub(crate) config: Option<Arc<llm_proxy_core::AppConfig>>,
    pub(crate) usage: Option<Arc<UsageReader>>,
    pub(crate) metrics: Option<Arc<llm_proxy_core::Metrics>>,
}

impl std::fmt::Debug for AdminSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdminSurface")
            .field("has_keys", &self.keys.is_some())
            .field("has_config", &self.config.is_some())
            .field("has_usage", &self.usage.is_some())
            .field("has_metrics", &self.metrics.is_some())
            .finish_non_exhaustive()
    }
}

impl AdminSurface {
    /// A surface guarded by the `[auth.admin]` token.
    #[must_use]
    pub fn new(admin_token: &str) -> Self {
        Self {
            auth: Arc::new(AdminAuth::new(admin_token)),
            keys: None,
            config: None,
            usage: None,
            metrics: None,
        }
    }

    /// Serve key management from this store.
    #[must_use]
    pub fn keys(mut self, keys: Arc<ManagedKeys>) -> Self {
        self.keys = Some(keys);
        self
    }

    /// Serve the sanitized running config.
    #[must_use]
    pub fn config(mut self, config: Arc<llm_proxy_core::AppConfig>) -> Self {
        self.config = Some(config);
        self
    }

    /// Serve the read-only usage views over the SQLite ledger.
    #[must_use]
    pub fn usage(mut self, reader: UsageReader) -> Self {
        self.usage = Some(Arc::new(reader));
        self
    }

    /// Serve the token-gated `/metrics` route over the runtime counters
    /// (plan 07 step 7): the Prometheus exposition of the scalar
    /// counters, behind the same admin token. The route does not exist
    /// unless the metrics registry is attached.
    #[must_use]
    pub fn metrics(mut self, metrics: Arc<llm_proxy_core::Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// The admin router: the web UI at `/`, key management under `/keys`,
    /// `/config`, the usage views under `/usage/*`, the MCP tools
    /// endpoint at `/mcp`, and — when the metrics registry is attached —
    /// the token-gated Prometheus exposition at `/metrics`. Admin auth,
    /// `no-store` and panic-catching are applied here; every route is
    /// read-only except key management.
    #[must_use = "the Router must be used with a server or merged into one"]
    pub fn router(self) -> Router {
        routes::router(self)
    }

    pub(crate) fn keys_ref(&self) -> Option<&Arc<ManagedKeys>> {
        self.keys.as_ref()
    }

    pub(crate) fn config_ref(&self) -> Option<&Arc<llm_proxy_core::AppConfig>> {
        self.config.as_ref()
    }

    pub(crate) fn usage_ref(&self) -> Option<&Arc<UsageReader>> {
        self.usage.as_ref()
    }

    pub(crate) fn metrics_ref(&self) -> Option<&Arc<llm_proxy_core::Metrics>> {
        self.metrics.as_ref()
    }
}
