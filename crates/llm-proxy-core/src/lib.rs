//! Core types shared across all `llm-proxy` crates: configuration, routing,
//! the provider registry, metrics, token counting, PID management, and
//! errors.

#![deny(missing_docs)]

/// Provider model catalog merge, filter, and cache-file types.
pub mod catalog;
/// Shared `${ENV_VAR}` interpolation for config files.
pub mod env_interpolate;
/// Crate-level error type.
pub mod error;
/// Runtime metrics (counters, latency ring-buffer, per-provider-model counts).
pub mod metrics;
/// TOML provider configuration types, parsing, and validation.
pub mod provider_config;
/// Provider registry: load provider TOML files and resolve route targets.
pub mod provider_registry;
/// Self-loop prevention (egress hop header, config-time upstream checks).
pub mod self_loop;
/// Token counting utilities.
pub mod token;

/// Shared test utilities (env-var guards, crate-level test mutex).
#[cfg(test)]
pub mod test_support;

pub use catalog::{
    CatalogFile, CatalogFileMetadata, glob_match, merge_catalog, model_allowed, parse_catalog_file,
};
pub use error::CoreError;
pub use metrics::{Metrics, Snapshot};
pub use provider_config::{
    ANONYMOUS_CALLER, AdminAuthConfig, AdminMountPosture, AffinityConfig, AffinityConflictPolicy,
    AppConfig, AuthConfig, AuthMode, AuthStyle, BareV1Config, ClientKeySpec, ConfigValidationError,
    DEFAULT_MAX_BODY_BYTES, EndpointFamilyConfig, IdempotencyConfig, KeyBudgetConfig,
    KeyWindowLimits, LogFormat, MAX_PROVIDER_NAME_LEN, McpGatewayConfig, McpUpstreamConfig,
    MetricsConfig, ModelId, ModelPricing, OtlpMetricsConfig, PoolConfig, PoolFile, PreflightConfig,
    PreflightMode, PricingSyncConfig, PromptCacheConfig, ProviderAdapterConfig,
    ProviderCatalogConfig, ProviderCatalogMode, ProviderConcurrencyConfig, ProviderConfig,
    ProviderDiscoveryConfig, ProviderDiscoveryKind, ProviderEgressConfig, ProviderFile,
    ProviderRouteKind, ProviderRoutesConfig, ReplicaBackend, ReplicaConfig, ResponseCacheConfig,
    RetryConfig, SemanticCacheConfig, ServerConfig, StaticModelCatalogEntry, StorageConfig,
    UnestimatableCostPolicy, UpstreamTimeoutsConfig, bare_v1_target_problems,
    family_max_body_bytes, family_request_timeout, is_valid_provider_name, load_app_config,
    load_provider_config, restart_required_changes, validate_metrics_config,
    validate_passthrough_auth_combination, validate_pool_config, validate_provider_config,
    validate_upstream_timeouts,
};
pub use provider_registry::{
    ProviderAdapterTargetConfig, ProviderRegistry, ProviderRouteResolutionError,
};
pub use self_loop::{HOP_HEADER, HOP_HEADER_VALUE, self_loop_problems};
pub use token::{Counter, MessageContent};
