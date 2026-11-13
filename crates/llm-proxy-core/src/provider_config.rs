//! TOML provider configuration types, parsing, env-var interpolation, and
//! validation. The main `config.toml` holds server settings; provider files
//! hold provider-scoped routing and catalog configuration.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rust_decimal::Decimal;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::env_interpolate::{find_env_var_refs, find_unresolved_env_var, interpolate_env_vars};
use crate::error::CoreError;

/// Top-level TOML application configuration.
///
/// Loaded from the main `config.toml`, which contains server and auth
/// settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    /// Server bind and operational settings.
    pub server: ServerConfig,
    /// Client authentication settings (`[auth]` section).
    #[serde(default)]
    pub auth: AuthConfig,
    /// Usage-ledger persistence settings (`[storage]` section). Absent keeps
    /// the in-memory no-op event bus (nothing is persisted).
    #[serde(default)]
    pub storage: Option<StorageConfig>,
    /// Replica-shared state (plan 09 step 1, audit conflict 13): when set,
    /// limits, cooldowns, budgets and usage live in the shared store, so
    /// replicas enforce one policy between them. Requires the matching
    /// optional store crate to be compiled in (`store-redis` /
    /// `store-postgres` features of the daemon); with the feature off the
    /// section is refused at startup.
    #[serde(default)]
    pub replica: Option<ReplicaConfig>,
    /// MCP gateway (plan 09 step 6, audit conflict 17): its own listener,
    /// auth and session model, sharing the proxy's admission and
    /// accounting. Requires the `mcp-gateway` feature of the daemon.
    #[serde(default)]
    pub mcp: Option<McpGatewayConfig>,
}

/// Which shared store holds the replica state (plan 09 step 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaBackend {
    /// Redis (`[replica]` with `backend = "redis"`): the
    /// `llm-proxy-store-redis` crate.
    Redis,
    /// Postgres (`[replica]` with `backend = "postgres"`): the
    /// `llm-proxy-store-postgres` crate.
    Postgres,
}

/// Replica-shared state settings (`[replica]` section, plan 09 step 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaConfig {
    /// Which shared store holds limits, cooldowns, budgets and usage.
    pub backend: ReplicaBackend,
    /// Connection string (`redis://host:port` or
    /// `postgres://user:pass@host/db`). Never serialized back out.
    #[serde(skip_serializing)]
    pub url: secrecy::SecretString,
    /// Key prefix (Redis) all replicas must agree on; distinct prefixes
    /// are distinct deployments.
    #[serde(default = "default_replica_prefix")]
    pub prefix: String,
}

fn default_replica_prefix() -> String {
    "llm-proxy".to_owned()
}

/// One upstream MCP server the gateway exposes (plan 09 step 6).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpUpstreamConfig {
    /// The name tools are namespaced under (`name__tool`).
    pub name: String,
    /// The upstream's MCP endpoint URL (streamable HTTP).
    pub url: String,
    /// The upstream's bearer token. Never serialized back out.
    #[serde(skip_serializing)]
    pub token: Option<secrecy::SecretString>,
}

/// MCP gateway settings (`[mcp]` section, plan 09 step 6): the gateway has
/// its own auth (a separate token list) and session model, and shares the
/// proxy's admission and accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpGatewayConfig {
    /// The dedicated listener the gateway binds (its auth model is not the
    /// proxy's, so it never shares a listener with it).
    pub bind: std::net::SocketAddr,
    /// Bearer tokens MCP clients may present. Empty refuses everyone.
    #[serde(default, skip_serializing)]
    pub tokens: Vec<secrecy::SecretString>,
    /// The upstream MCP servers whose tools the gateway exposes.
    #[serde(default)]
    pub upstreams: Vec<McpUpstreamConfig>,
    /// How long an idle gateway session survives.
    #[serde(default = "default_mcp_session_ttl", with = "humantime_serde")]
    pub session_ttl: Duration,
}

fn default_mcp_session_ttl() -> Duration {
    Duration::from_secs(1800)
}

/// Named security postures for where the admin `/usage` API is reachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdminMountPosture {
    /// Default: `/usage` is served only on the dedicated
    /// [`ServerConfig::admin_bind`] listener (loopback unless configured
    /// otherwise); the main listener has no admin surface at all.
    DedicatedListener,
    /// Opt-in (`server.admin_on_main_listener = true`): `/usage` is also
    /// served on the main `bind` listener behind the same admin token, so
    /// the admin surface reaches every network the main listener does.
    MainListener,
}

impl AdminMountPosture {
    /// Stable name for this posture, used in startup logs.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DedicatedListener => "dedicated-admin-listener",
            Self::MainListener => "admin-on-main-listener",
        }
    }
}

impl AppConfig {
    /// The admin `/usage` mount posture, or `None` when no
    /// `[auth.admin]` token is configured (no admin surface anywhere).
    #[must_use]
    pub fn admin_posture(&self) -> Option<AdminMountPosture> {
        let posture = if self.server.admin_on_main_listener {
            AdminMountPosture::MainListener
        } else {
            AdminMountPosture::DedicatedListener
        };
        self.auth.admin.as_ref().map(|_| posture)
    }
}

/// Log output format for the tracing subscriber.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    /// Human-readable plain-text logs (default).
    #[default]
    Plain,
    /// Structured JSON logs (one JSON object per line).
    Json,
}

impl LogFormat {
    /// Read the format from the `RUST_LOG_FORMAT` env var (case-insensitive).
    ///
    /// Defaults to [`LogFormat::Plain`] when unset or unrecognized. Read at
    /// subscriber init, before config load.
    pub fn from_env() -> Self {
        match std::env::var("RUST_LOG_FORMAT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => Self::Json,
            _ => Self::Plain,
        }
    }

    /// The format subscriber init should use (BUG-25): a set, non-empty
    /// `RUST_LOG_FORMAT` wins over the configured `server.log_format`, as
    /// documented; unset or empty falls back to the configured value. A set
    /// but unrecognized value wins as [`LogFormat::Plain`], matching
    /// [`LogFormat::from_env`].
    ///
    /// The env value is a parameter (not read here) so precedence is
    /// testable without touching the process environment.
    #[must_use]
    pub fn effective(env: Option<&str>, configured: Self) -> Self {
        let Some(value) = env.map(str::trim).filter(|value| !value.is_empty()) else {
            return configured;
        };
        match value.to_ascii_lowercase().as_str() {
            "json" => Self::Json,
            _ => Self::Plain,
        }
    }
}

/// Server operational settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address the HTTP server binds to.
    pub bind: SocketAddr,
    /// Maximum request duration before timeout.
    #[serde(with = "humantime_serde")]
    pub request_timeout: Duration,
    /// Hard deadline for graceful shutdown.
    ///
    /// After receiving SIGINT/SIGTERM the server stops accepting new
    /// connections and waits this long for in-flight requests to drain, then
    /// force-exits so orchestrators can recycle the process. `0s` disables
    /// the hard deadline (drain forever) but is strongly discouraged.
    #[serde(default = "default_shutdown_timeout", with = "humantime_serde")]
    pub shutdown_timeout: Duration,
    /// Log level: `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`.
    pub log_level: String,
    /// Log output format for the tracing subscriber: `"plain"` (default) or
    /// `"json"`. The startup path honours this value
    /// (see [`LogFormat::effective`]); a set `RUST_LOG_FORMAT` env var wins
    /// over it.
    #[serde(default)]
    pub log_format: LogFormat,
    /// Enable hot reload of the provider and pool files (plan 07 step 11):
    /// SIGHUP (or `llm-proxy reload`) re-reads them and swaps providers
    /// without dropping in-flight requests. Changes anywhere else —
    /// `[server]`, `[auth]`, `[storage]` — are refused with a
    /// restart-required error instead of being half-applied. Default `false`:
    /// without it, reload signals are ignored.
    #[serde(default)]
    pub hot_reload: bool,
    /// Per-client requests allowed per minute. `0` disables rate limiting.
    #[serde(default = "default_rate_limit_rpm")]
    pub rate_limit_rpm: u32,
    /// Whether client identity may use `X-Forwarded-For` and `X-Real-IP`.
    ///
    /// # Trust model
    ///
    /// When true, the **leftmost** value of `X-Forwarded-For` (and the value of
    /// `X-Real-IP`) is trusted as the client address. This is only correct
    /// behind a **single** reverse proxy that strips or overwrites any inbound
    /// `X-Forwarded-For` before appending its own. With multiple untrusted hops
    /// the leftmost entry is attacker-controllable (a client can prepend
    /// arbitrary values), so the extracted address must not be trusted for
    /// authentication, rate-limit bypass, or audit logging.
    #[serde(default)]
    pub trust_forwarded_headers: bool,
    /// Public server name reported in version/health endpoints.
    pub server_name: String,
    /// Cross-origin origins permitted to call the proxy from a browser.
    ///
    /// `None` (the default) installs **no** CORS layer, so browsers enforce a
    /// same-origin policy and this server-to-server proxy is not unexpectedly
    /// reachable from arbitrary web origins. Setting a list installs a
    /// restrictive `CorsLayer` that echoes `Access-Control-Allow-Origin` only
    /// for these exact origins.
    ///
    /// Each entry should be an absolute origin (`scheme://host[:port]`),
    /// e.g. `"https://app.example.com"`. Entries that fail to parse as a
    /// header value are skipped (with a warning) when the router is built.
    #[serde(default)]
    pub allowed_origins: Option<Vec<String>>,
    /// Dedicated listener bind for the admin `/usage` API. Only read when
    /// `[auth.admin]` is configured; the default is `127.0.0.1` on the port
    /// after `bind`'s port (see [`ServerConfig::effective_admin_bind`]).
    #[serde(default)]
    pub admin_bind: Option<SocketAddr>,
    /// Also serve `/usage` on the main listener behind the same admin token.
    ///
    /// Opt-in; selects the [`AdminMountPosture::MainListener`] posture, which
    /// the serve path announces with a loud startup warning: the admin
    /// surface then reaches every network the main listener does. With no
    /// explicit `admin_bind` this posture serves `/usage` on the main
    /// listener ONLY — no dedicated socket is attempted.
    #[serde(default)]
    pub admin_on_main_listener: bool,
    /// Per-endpoint-family overrides (audit conflict 16): a chat family and
    /// an audio/file family do not share one body limit or one timeout. Keys
    /// are family names (`"messages"`, `"chat_completions"`, `"models"`,
    /// `"count_tokens"`); absent families use the server-wide defaults.
    #[serde(default)]
    pub endpoint_families: std::collections::HashMap<String, EndpointFamilyConfig>,
    /// Phase-shaped upstream timeouts (backlog 8, BUG-14): connect,
    /// first-byte, in-stream idle and an optional one-deadline-per-request
    /// total, validated at config load.
    #[serde(default)]
    pub upstream_timeouts: UpstreamTimeoutsConfig,
    /// Bare `/v1/*` routing (plan 04 step 11): off by default; when enabled
    /// a model name resolves to a provider or pool.
    #[serde(default)]
    pub bare_v1: BareV1Config,
    /// Context-window preflight (plan 05 step 8): warn by default.
    #[serde(default)]
    pub preflight: PreflightConfig,
    /// Exact-match response cache (plan 06): off by default; stores complete
    /// non-streaming responses per caller for the configured TTL.
    #[serde(default)]
    pub cache: ResponseCacheConfig,
    /// `Idempotency-Key` replay (plan 06): off by default.
    #[serde(default)]
    pub idempotency: IdempotencyConfig,
    /// Metrics export (plan 07 step 7, backlog 21): the in-process counters
    /// are always kept; this adds export paths, off by default.
    #[serde(default)]
    pub metrics: MetricsConfig,
    /// Semantic cache (plan 09 step 5, conflict 7): off by default. When
    /// enabled, prompts are embedded and near-duplicate hits (per caller)
    /// reuse the stored response, under the data-class policy in
    /// `docs/data-class-policy.md`.
    #[serde(default)]
    pub semantic_cache: SemanticCacheConfig,
}

/// Semantic cache settings (`[server.semantic_cache]`, plan 09 step 5):
/// opt-in, caller-scoped, threshold-gated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticCacheConfig {
    /// Whether the semantic cache is on. Off by default.
    #[serde(default)]
    pub enabled: bool,
    /// How long a cached response may be served.
    #[serde(default = "default_semantic_ttl", with = "humantime_serde")]
    pub ttl: Duration,
    /// Bound on stored entries (per caller).
    #[serde(default = "default_semantic_max_entries")]
    pub max_entries: usize,
    /// Cosine-similarity threshold in (0.0, 1.0]: a stored prompt must be
    /// at least this similar to be served. 1.0 behaves like an exact
    /// cache.
    #[serde(default = "default_semantic_threshold")]
    pub threshold: f32,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl: default_semantic_ttl(),
            max_entries: default_semantic_max_entries(),
            threshold: default_semantic_threshold(),
        }
    }
}

fn default_semantic_ttl() -> Duration {
    Duration::from_secs(600)
}

fn default_semantic_max_entries() -> usize {
    256
}

fn default_semantic_threshold() -> f32 {
    0.95
}

/// Per-endpoint-family settings (audit conflict 16): the body limit and
/// request timeout one family applies, independent of other families.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointFamilyConfig {
    /// Maximum request body this family accepts, in bytes.
    pub max_body_bytes: usize,
    /// Maximum request duration this family applies.
    #[serde(with = "humantime_serde")]
    pub request_timeout: Duration,
}

/// Phase-shaped upstream timeouts (backlog 8, BUG-14): one set of phases for
/// every upstream request, replacing the fixed 60-second headers deadline.
/// Cross-validated at config load ([`validate_upstream_timeouts`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamTimeoutsConfig {
    /// TCP/TLS connect deadline per upstream attempt.
    #[serde(default = "default_connect_timeout", with = "humantime_serde")]
    pub connect: Duration,
    /// Time-to-first-byte deadline: waiting for response headers on a
    /// non-streaming request, or for the first stream event on a streaming
    /// request. Does not cap the body.
    #[serde(default = "default_first_byte_timeout", with = "humantime_serde")]
    pub first_byte: Duration,
    /// In-stream idle deadline: a stream silent for longer than this between
    /// chunks is a dead upstream.
    #[serde(default = "default_idle_timeout", with = "humantime_serde")]
    pub idle: Duration,
    /// One deadline across every attempt of one request (retry ladder
    /// included). `None` (the default) leaves the ladder unbounded here; the
    /// per-route `[provider.retry]` / `[pool.retry]` `total_timeout` may still
    /// bound it.
    #[serde(default, with = "humantime_serde::option")]
    pub total: Option<Duration>,
}

const fn default_connect_timeout() -> Duration {
    Duration::from_secs(10)
}

const fn default_first_byte_timeout() -> Duration {
    Duration::from_secs(60)
}

const fn default_idle_timeout() -> Duration {
    Duration::from_secs(60)
}

impl Default for UpstreamTimeoutsConfig {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            first_byte: Duration::from_secs(60),
            idle: Duration::from_secs(60),
            total: None,
        }
    }
}

/// Validates one [`UpstreamTimeoutsConfig`]: every phase is positive, connect
/// fits inside first-byte, and the total deadline (when set) covers
/// first-byte. Called at config load and by builders.
///
/// # Errors
///
/// A [`ConfigValidationError::InvalidTimeouts`] naming the first broken rule.
pub fn validate_upstream_timeouts(
    timeouts: &UpstreamTimeoutsConfig,
) -> Result<(), ConfigValidationError> {
    let phases = [
        ("connect", timeouts.connect),
        ("first_byte", timeouts.first_byte),
        ("idle", timeouts.idle),
    ];
    for &(name, value) in &phases {
        if value.is_zero() {
            return Err(ConfigValidationError::InvalidTimeouts {
                message: format!("upstream timeout phase \"{name}\" must be positive"),
            });
        }
    }
    if timeouts.connect > timeouts.first_byte {
        return Err(ConfigValidationError::InvalidTimeouts {
            message: format!(
                "upstream timeout \"connect\" ({:?}) must not exceed \"first_byte\" ({:?})",
                timeouts.connect, timeouts.first_byte
            ),
        });
    }
    if let Some(total) = timeouts.total
        && total < timeouts.first_byte
    {
        return Err(ConfigValidationError::InvalidTimeouts {
            message: format!(
                "upstream timeout \"total\" ({total:?}) must cover \"first_byte\" ({:?})",
                timeouts.first_byte
            ),
        });
    }
    Ok(())
}

/// Bare `/v1/*` routing (plan 04 step 11): off by default. When enabled, a
/// model name resolves to a provider or pool through `models`, falling back
/// to `default_target`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BareV1Config {
    /// Serve the bare `/v1/messages` and `/v1/chat/completions` routes.
    /// Default `false`: those routes 404.
    #[serde(default)]
    pub enabled: bool,
    /// Model name to provider or pool name.
    #[serde(default)]
    pub models: HashMap<String, String>,
    /// Target used when no `models` entry matches. `None` makes an unmapped
    /// model a 404 naming the model.
    #[serde(default)]
    pub default_target: Option<String>,
}

/// The retry/failover ladder for one route (plan 04 steps 2-3, 6). Present
/// only by opt-in on a provider (`[provider.retry]`) or a pool
/// (`[pool.retry]`); absent means exactly one attempt and no failover, the
/// pre-plan behaviour.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryConfig {
    /// Attempts against one member before moving to the next (pools) or
    /// giving up (single provider). `1` disables same-member retries.
    pub max_attempts: u32,
    /// Base delay between attempts against the same member. An upstream
    /// `Retry-After` longer than this wins, clamped to the remaining
    /// deadline.
    #[serde(with = "humantime_serde")]
    pub backoff: Duration,
    /// How long a member that failed transiently is skipped by later
    /// requests. A 429 with real reset information overrides this until the
    /// reported reset time.
    #[serde(with = "humantime_serde")]
    pub cooldown: Duration,
    /// One deadline across every attempt of one request. `0s` disables it.
    #[serde(with = "humantime_serde")]
    pub total_timeout: Duration,
    /// Streaming pre-commit hold budget (plan 04 step 6): how long the route
    /// buffers waiting for the first output event before committing HTTP 200
    /// (after which heartbeats cover silence and errors are in-band).
    /// `0s` (the default) never commits without an event, keeping HTTP error
    /// semantics for as long as nothing was sent.
    #[serde(with = "humantime_serde")]
    pub hold: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 2,
            backoff: Duration::from_millis(250),
            cooldown: Duration::from_secs(30),
            total_timeout: Duration::ZERO,
            hold: Duration::ZERO,
        }
    }
}

/// What a pool does when a sticky session's pinned member must be switched
/// away from (plan 04 step 10, conflict 6).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AffinityConflictPolicy {
    /// Strip provider-bound request state (`provider_hints`) before
    /// switching. The default: availability over fidelity.
    #[default]
    Strip,
    /// Refuse the switch: the request fails rather than reaching a different
    /// account with state it cannot honour.
    Refuse,
}

/// Session affinity for a pool (plan 04 step 10): record session ids on usage
/// rows always; pinning a session to its last successful member is opt-in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AffinityConfig {
    /// Pin a session id to the member that last served it, so conversation
    /// cache warmth survives across requests. Default `false`.
    #[serde(default)]
    pub sticky: bool,
    /// What happens when the pinned member has to be switched away from.
    #[serde(default)]
    pub on_conflict: AffinityConflictPolicy,
}

impl Default for AffinityConfig {
    fn default() -> Self {
        Self {
            sticky: false,
            on_conflict: AffinityConflictPolicy::Strip,
        }
    }
}

/// Pool health settings (plan 04 step 9): passive cooldowns from 429 and
/// rate-limit headers are always on; the synthetic probe is opt-in.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolHealthConfig {
    /// How often to probe cooled-down members with a synthetic one-token
    /// request; a healthy answer clears the cooldown early. `None` (the
    /// default) disables probing — cooldowns then simply expire.
    #[serde(default, with = "humantime_serde::option")]
    pub probe_interval: Option<Duration>,
}

/// A named pool of providers served at `/providers/{pool}/v1/...`
/// (plan 04 step 4, conflict 1): fallback inside the pool, same URL shape as
/// a single provider. A pool name may not collide with a provider name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    /// Pool name; occupies the same URL namespace as provider names.
    pub name: String,
    /// Member provider names, in failover order. Duplicates and unknown
    /// names are rejected at config validation.
    pub members: Vec<String>,
    /// Retry ladder applied to the pool and inherited as the per-member
    /// attempt policy.
    #[serde(default)]
    pub retry: RetryConfig,
    /// Session affinity policy for this pool.
    #[serde(default)]
    pub affinity: AffinityConfig,
    /// Health tracking for this pool's members.
    #[serde(default)]
    pub health: PoolHealthConfig,
}

/// Wrapper for a single pool TOML file (a `[pool]` table instead of a
/// `[provider]` one).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolFile {
    /// The pool definition.
    pub pool: PoolConfig,
}

/// Default body limit for one request: 32 MiB (as before the family split).
pub const DEFAULT_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// The body limit a family applies, given the server defaults.
#[must_use]
pub fn family_max_body_bytes(server: &ServerConfig, family: &str) -> usize {
    server
        .endpoint_families
        .get(family)
        .map_or(DEFAULT_MAX_BODY_BYTES, |family_cfg| {
            family_cfg.max_body_bytes
        })
}

/// The request timeout a family applies, given the server default.
#[must_use]
pub fn family_request_timeout(server: &ServerConfig, family: &str) -> Duration {
    server
        .endpoint_families
        .get(family)
        .map_or(server.request_timeout, |cfg| cfg.request_timeout)
}

impl ServerConfig {
    /// Whether the dedicated admin listener binds at all.
    ///
    /// An explicit `admin_bind` always binds one. With none set, the
    /// default loopback listener binds unless `admin_on_main_listener`
    /// took over.
    #[must_use]
    pub fn dedicated_admin_listener(&self) -> bool {
        self.admin_bind.is_some() || !self.admin_on_main_listener
    }

    /// Bind address for the dedicated admin listener.
    ///
    /// `admin_bind` when configured; otherwise the default: `127.0.0.1` on
    /// the port after `bind`'s port. `None` when that default port cannot be
    /// derived (`bind` on port 65535) — `load_app_config` rejects that
    /// combination whenever the dedicated listener would bind the default.
    #[must_use]
    pub fn effective_admin_bind(&self) -> Option<SocketAddr> {
        match self.admin_bind {
            Some(addr) => Some(addr),
            None => self
                .bind
                .port()
                .checked_add(1)
                .map(|port| SocketAddr::new(std::net::IpAddr::from([127, 0, 0, 1]), port)),
        }
    }
}

const fn default_rate_limit_rpm() -> u32 {
    100
}

/// Client authentication mode for proxy routes.
///
/// Serialized in TOML as `"none"` / `"client-keys"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthMode {
    /// No client authentication: every request is attributed to the
    /// `"anonymous"` caller. Only permitted while the server binds loopback;
    /// `load_app_config` rejects a non-loopback bind in this mode.
    #[default]
    None,
    /// Requests to proxy routes must present
    /// `Authorization: Bearer <client-key>` matching a secret hashed under
    /// `[auth.clients]`.
    ClientKeys,
}

/// Client authentication configuration (`[auth]` section of `config.toml`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    /// Authentication mode. Defaults to [`AuthMode::None`].
    #[serde(default)]
    pub mode: AuthMode,
    /// Key id to key spec, under `[auth.clients]`.
    ///
    /// A spec is either a bare PHC-format password-hashing string (argon2id)
    /// or a table carrying the verifier plus this key's admission policy
    /// (allowlists, rate windows, spend budget — plan 05 steps 3-5). The
    /// verifier is never a plaintext secret and is wrapped in
    /// [`SecretString`] so it is redacted in `Debug` output.
    /// `skip_serializing` keeps the hashes out of every `Serialize` path, so
    /// serializing then deserializing a config fails on this field — by
    /// design.
    #[serde(default, skip_serializing)]
    pub clients: HashMap<String, ClientKeySpec>,
    /// Admin API authentication (`[auth.admin]` section). When present, the
    /// read-only `/usage` query surface is enabled and every `/usage` route
    /// on every mount requires its bearer token.
    #[serde(default)]
    pub admin: Option<AdminAuthConfig>,
    /// Accept tokens without a `<key-id>.` prefix (pre-prefix behaviour).
    ///
    /// Default `false`: a bare token is refused without running any
    /// verifier, so a bad token's lookup cost stays flat whatever the key
    /// count. When `true`, a bare token is checked against every
    /// configured key in turn — its cost grows with the key count, up to
    /// the cap the server applies (`ClientAuth`). Existing deployments
    /// that hand out bare secrets set this to keep them working.
    #[serde(default)]
    pub legacy_bare_tokens: bool,
}

/// One client key: its verifier, optionally plus its admission policy
/// (`[auth.clients.<id>]` as a table instead of a bare string).
///
/// ```toml
/// # bare: no policy, the key reaches everything
/// laptop = "$argon2id$v=19$m=19456,t=2,p=1$..."
///
/// # with policy
/// [auth.clients.laptop]
/// verifier = "$argon2id$v=19$m=19456,t=2,p=1$..."
/// allow_models = ["claude-*", "gpt-4o"]
/// allow_providers = ["anthropic-pool", "openai"]
/// [auth.clients.laptop.limits]
/// rpm = 60
/// tpm = 200_000
/// rpd = 5_000
/// [auth.clients.laptop.budget]
/// usd_per_day = "10.00"
/// ```
/// `Serialize` is deliberately not implemented: verifiers must not be
/// serializable, and the `[auth.clients]` map is `skip_serializing` on
/// [`AuthConfig`] anyway.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ClientKeySpec {
    /// A bare PHC verifier string: no policy fields.
    Verifier(
        /// The argon2id PHC string.
        SecretString,
    ),
    /// A verifier table with policy.
    Policy {
        /// The argon2id PHC string.
        verifier: SecretString,
        /// Model globs this key may request (`["*"]` or empty allows all).
        /// An empty or missing model in the request fails closed (backlog
        /// 18): a restricted key may not run unattributable work.
        #[serde(default)]
        allow_models: Vec<String>,
        /// Provider (or pool) name globs this key may address. Empty allows
        /// all.
        #[serde(default)]
        allow_providers: Vec<String>,
        /// Rate windows for this key (plan 05 step 5).
        #[serde(default)]
        limits: Option<KeyWindowLimits>,
        /// Spend budget for this key (plan 05 step 4).
        #[serde(default)]
        budget: Option<KeyBudgetConfig>,
    },
}

impl ClientKeySpec {
    /// The verifier this key authenticates with.
    #[must_use]
    pub fn verifier(&self) -> &SecretString {
        match self {
            Self::Verifier(phc) => phc,
            Self::Policy { verifier, .. } => verifier,
        }
    }
}

/// Per-key rate windows (plan 05 step 5, backlog 6). All optional; an absent
/// window is unlimited.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyWindowLimits {
    /// Requests per calendar minute (the window resets on the minute
    /// boundary, it does not slide).
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Tokens (input estimate at reserve, actual usage at settle) per
    /// calendar minute, resetting on the minute boundary.
    #[serde(default)]
    pub tpm: Option<u64>,
    /// Requests per UTC day.
    #[serde(default)]
    pub rpd: Option<u64>,
}

/// How a key's budget treats a request whose cost cannot be estimated (no
/// pricing configured for the model). Failing closed is the default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum UnestimatableCostPolicy {
    /// Refuse the request (fail closed): the key cannot spend what cannot be
    /// bounded.
    #[default]
    Reject,
    /// Reserve this fixed USD amount in place of an estimate.
    ReserveUsd {
        /// The fixed per-request reservation.
        usd: rust_decimal::Decimal,
    },
}

/// Per-key spend budget (plan 05 step 4): a USD cap per UTC day, enforced
/// reserve-then-settle so concurrent requests cannot overspend it by more
/// than the reserved amount.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyBudgetConfig {
    /// Maximum USD spend per UTC day, as a quoted decimal string.
    pub usd_per_day: rust_decimal::Decimal,
    /// What to do when a request's cost cannot be estimated (the model has
    /// no pricing entry): refuse it (default) or reserve a fixed amount.
    #[serde(default)]
    pub unestimatable: UnestimatableCostPolicy,
    /// Output tokens assumed when the request sets no `max_tokens`, for
    /// estimating the reservation. Default 4096.
    #[serde(default = "default_assumed_max_output_tokens")]
    pub assumed_max_output_tokens: u64,
}

const fn default_assumed_max_output_tokens() -> u64 {
    4096
}

/// Opt-in pricing sync from models.dev (plan 05 step 9, backlog 22).
/// Declares which models.dev provider this proxy provider's pricing comes
/// from; `llm-proxy pricing sync` writes `[provider.pricing]` from it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricingSyncConfig {
    /// The provider id in the models.dev catalogue (for example
    /// `"anthropic"`). The proxy provider's own name need not match it.
    pub models_dev_id: String,
}

/// Per-provider in-flight concurrency caps (plan 05 step 7, backlog 30).
/// Saturated caps answer 503 with a short `Retry-After`, never 429.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConcurrencyConfig {
    /// Maximum in-flight requests against this provider. The cap is per
    /// member: on a pool route it applies while that member is the one
    /// serving (the reservation moves to each member the ladder tries), so
    /// a capped member is capped however the request reaches it. There is
    /// no whole-pool cap. `None` = uncapped.
    #[serde(default)]
    pub total: Option<u32>,
    /// Tighter per-model caps (upstream model ids), overriding `total`
    /// when lower.
    #[serde(default)]
    pub models: HashMap<ModelId, u32>,
}

/// Context-window preflight (plan 05 step 8, backlog 4, conflict 15).
/// Warn by default: non-OpenAI estimates use chars/4 and counting ignores
/// tools, tool results and images, so a hard reject needs a margin or a
/// provider token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreflightConfig {
    /// Whether preflight warns, rejects, or stays off. Default
    /// [`PreflightMode::Warn`].
    #[serde(default)]
    pub mode: PreflightMode,
    /// Safety margin in tokens: enforce mode rejects only when the
    /// estimated input exceeds the context window by at least this much, so
    /// estimate error cannot produce a false rejection. Default 512.
    #[serde(default = "default_preflight_margin_tokens")]
    pub margin_tokens: u32,
}

const fn default_preflight_margin_tokens() -> u32 {
    512
}

impl Default for PreflightConfig {
    fn default() -> Self {
        Self {
            mode: PreflightMode::default(),
            margin_tokens: default_preflight_margin_tokens(),
        }
    }
}

/// Preflight mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreflightMode {
    /// No preflight at all.
    Off,
    /// Log a warning when the estimate does not fit; admit the request.
    #[default]
    Warn,
    /// Warn as above, and reject (400) only when the estimated input
    /// exceeds the window by at least `margin_tokens`.
    Enforce,
}

/// Response cache (`[server.cache]`, plan 06): opt-in exact-match storage of
/// complete non-streaming responses, keyed per caller (caller, provider,
/// route, upstream model, body digest). Off by default: with it off the
/// proxy stores no prompt or response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseCacheConfig {
    /// Whether completed responses are stored and answered from the cache.
    #[serde(default)]
    pub enabled: bool,
    /// How long a stored response stays fresh. A lookup after the TTL is a
    /// miss. Must be non-zero when enabled.
    #[serde(default = "default_response_cache_ttl", with = "humantime_serde")]
    pub ttl: Duration,
    /// Bound on stored responses (FIFO eviction at the bound).
    #[serde(default = "default_response_cache_max_entries")]
    pub max_entries: usize,
}

const fn default_response_cache_ttl() -> Duration {
    Duration::from_secs(600)
}

const fn default_response_cache_max_entries() -> usize {
    1000
}

impl Default for ResponseCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl: default_response_cache_ttl(),
            max_entries: default_response_cache_max_entries(),
        }
    }
}

/// `Idempotency-Key` replay (`[server.idempotency]`, plan 06): opt-in. A
/// request that carries an `Idempotency-Key` header claims the key before
/// budget checks; a matching retry after completion replays the stored
/// response, and a request arriving while the first is still in flight gets
/// 409. Off by default: with it off the header is ignored and nothing is
/// retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdempotencyConfig {
    /// Whether `Idempotency-Key` headers are honoured.
    #[serde(default)]
    pub enabled: bool,
    /// How long a completed outcome is replayed for a matching retry. Must
    /// be non-zero when enabled.
    #[serde(default = "default_idempotency_ttl", with = "humantime_serde")]
    pub ttl: Duration,
    /// Bound on stored completed outcomes (FIFO eviction at the bound;
    /// expired outcomes are swept). In-flight claims are never counted.
    #[serde(default = "default_idempotency_max_entries")]
    pub max_entries: usize,
}

const fn default_idempotency_ttl() -> Duration {
    Duration::from_secs(24 * 60 * 60)
}

const fn default_idempotency_max_entries() -> usize {
    1000
}

impl Default for IdempotencyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ttl: default_idempotency_ttl(),
            max_entries: default_idempotency_max_entries(),
        }
    }
}

/// Metrics export settings (`[server.metrics]`, plan 07 step 7, backlog 21).
/// The existing in-process counters are always kept; this section only adds
/// export paths, both off by default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// OTLP export over HTTP JSON to a collector (for example an OpenTelemetry
    /// Collector at `http://127.0.0.1:4318/v1/metrics`). Absent = off.
    #[serde(default)]
    pub otlp: Option<OtlpMetricsConfig>,
}

/// OTLP metric export: one endpoint, one interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OtlpMetricsConfig {
    /// The collector's full metrics endpoint URL (`.../v1/metrics`).
    pub endpoint: String,
    /// How often counters are pushed. Default 60s; must be positive.
    #[serde(default = "default_otlp_interval", with = "humantime_serde")]
    pub interval: Duration,
}

const fn default_otlp_interval() -> Duration {
    Duration::from_secs(60)
}

/// Validates `[server.metrics]`: the OTLP endpoint must be an http(s) URL
/// with a host, and the interval positive.
pub fn validate_metrics_config(metrics: &MetricsConfig) -> Result<(), CoreError> {
    let Some(otlp) = &metrics.otlp else {
        return Ok(());
    };
    let endpoint = otlp.endpoint.trim();
    if endpoint.is_empty() {
        return Err(CoreError::ConfigValidation {
            message: "server.metrics.otlp.endpoint must not be empty".to_owned(),
            source: None,
        });
    }
    let scheme = endpoint.split("://").next().unwrap_or_default();
    if !matches!(scheme, "http" | "https") {
        return Err(CoreError::ConfigValidation {
            message: format!(
                "server.metrics.otlp.endpoint must be an http:// or https:// URL, got {:?}",
                endpoint
            ),
            source: None,
        });
    }
    if otlp.interval.is_zero() {
        return Err(CoreError::ConfigValidation {
            message: "server.metrics.otlp.interval must be positive".to_owned(),
            source: None,
        });
    }
    Ok(())
}

/// Admin API configuration (`[auth.admin]` section of `config.toml`).
///
/// A single bearer token guarding every `/usage` route.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdminAuthConfig {
    /// The admin bearer token, presented as `Authorization: Bearer <token>`.
    ///
    /// Compared in constant time against a SHA-256 digest of the configured
    /// token, so the plaintext never needs to be retained in memory. Wrapped
    /// in [`SecretString`] and `skip_serializing`, keeping it out of `Debug`
    /// output and every `Serialize` path.
    #[serde(skip_serializing)]
    pub token: SecretString,
    /// Where the admin surface persists managed client keys (plan 09 step
    /// 2): a SQLite file. Absent means keys are managed only in
    /// `[auth.clients]`. Requires the daemon's `admin-ui` feature.
    #[serde(default)]
    pub key_store: Option<std::path::PathBuf>,
}

const fn default_shutdown_timeout() -> Duration {
    Duration::from_secs(30)
}

const fn default_prune_interval() -> Duration {
    Duration::from_secs(3600)
}

/// Usage-ledger persistence configuration (`[storage]` section of
/// `config.toml`).
///
/// When present, every proxied request is persisted to a SQLite ledger by the
/// storage crate's event bus. When absent, the in-memory no-op bus is used and
/// nothing is written to disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    /// Path to the SQLite database file. Parent directories are created on
    /// startup.
    pub path: PathBuf,
    /// Delete ledger rows older than this age. Absent means rows are kept
    /// forever. Must be positive when present.
    #[serde(default, with = "humantime_serde")]
    pub max_age: Option<Duration>,
    /// Keep at most this many ledger rows; the oldest rows are deleted when
    /// the cap is exceeded. Absent means unlimited. Must be at least 1 when
    /// present.
    #[serde(default)]
    pub max_rows: Option<u64>,
    /// How often the retention pruning pass runs. Defaults to 1 hour. Must be
    /// at least 1 second (the pruning timer rejects a zero period).
    #[serde(default = "default_prune_interval", with = "humantime_serde")]
    pub prune_interval: Duration,
}

fn validate_storage_config(storage: &StorageConfig) -> Result<(), CoreError> {
    if storage.path.as_os_str().is_empty() {
        return Err(CoreError::ConfigValidation {
            message: "storage.path must not be empty".to_owned(),
            source: None,
        });
    }
    if storage.max_age == Some(Duration::ZERO) {
        return Err(CoreError::ConfigValidation {
            message: "storage.max_age must be a positive duration; a zero age would \
                      delete every ledger row on every pruning pass"
                .to_owned(),
            source: None,
        });
    }
    if storage.max_rows == Some(0) {
        return Err(CoreError::ConfigValidation {
            message: "storage.max_rows must be at least 1; a cap of zero would keep \
                      no rows at all"
                .to_owned(),
            source: None,
        });
    }
    const MIN_PRUNE_INTERVAL: Duration = Duration::from_secs(1);
    if storage.prune_interval < MIN_PRUNE_INTERVAL {
        return Err(CoreError::ConfigValidation {
            message: format!(
                "storage.prune_interval must be at least 1 second, but is {:?}",
                storage.prune_interval
            ),
            source: None,
        });
    }
    Ok(())
}

/// Wrapper for a single provider TOML file.
///
/// Each provider is defined in its own file with a top-level `[provider]`
/// section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderFile {
    /// The provider definition.
    pub provider: ProviderConfig,
}

/// Provider configuration loaded from a dedicated TOML file.
///
/// The `api_key` field is redacted in [`Debug`] output and excluded from
/// [`Serialize`] output to prevent credential leakage in logs, error messages,
/// and API responses.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// Provider name (e.g. `"opencode-go"`, `"opencode-zen"`).
    pub name: String,
    /// API key for authenticating with the upstream provider.
    /// Supports `${ENV_VAR}` interpolation (applied to the raw TOML *before*
    /// this field is constructed, so the resolved value is what gets wrapped).
    ///
    /// Wrapped in [`SecretString`]: redacted in [`Debug`] and [`Display`] output
    /// and zeroized on drop. Not serialized (`skip_serializing`): credentials
    /// never leak through serialization paths, so round-tripping a
    /// `ProviderFile` through serialize/deserialize FAILS because `api_key`
    /// is a required field. This is intentional.
    #[serde(skip_serializing)]
    pub api_key: SecretString,
    /// Additional upstream credentials (plan 04 step 5, backlog 31): when
    /// present and non-empty, these plus `api_key` form the provider's
    /// credential pool; the attempt loop rotates over them and cools a key
    /// down on auth and rate-limit failures. Never serialized.
    #[serde(default, skip_serializing)]
    pub api_keys: Option<Vec<SecretString>>,
    /// Authentication header style.
    pub auth_style: AuthStyle,
    /// When true, ignore this provider's configured `api_key` and instead use
    /// the inbound client's auth token (the `Authorization: Bearer ...` or
    /// `x-api-key` request header) as the upstream credential
    /// (bring-your-own-key / passthrough auth). The request fails with 401 if
    /// the client sends no recognizable token. When false (default), the
    /// provider's static `api_key` is used.
    #[serde(default)]
    pub passthrough_auth: bool,
    /// Opt-in retry ladder for this provider (plan 04 steps 2-3): absent
    /// means exactly one attempt, the pre-plan behaviour, and no failover on
    /// a direct `/providers/{provider}/...` route.
    #[serde(default)]
    pub retry: Option<RetryConfig>,
    /// Named adapter configurations (protocol + endpoint pairs).
    #[serde(default)]
    pub adapters: HashMap<String, ProviderAdapterConfig>,
    /// Route kind to adapter name mapping (e.g. chat_completions -> openai_adapter).
    #[serde(default)]
    pub routes: ProviderRoutesConfig,
    /// Model alias mapping (requested model -> upstream model name).
    #[serde(default)]
    pub model_aliases: HashMap<String, String>,
    /// Live model discovery configuration (optional).
    #[serde(default)]
    pub discovery: Option<ProviderDiscoveryConfig>,
    /// Model catalog configuration (optional, defaults to advisory hybrid).
    #[serde(default)]
    pub catalog: Option<ProviderCatalogConfig>,
    /// Per-model pricing, keyed by upstream model id. Used for cost estimation
    /// only; not enforced for routing. Aliases are resolved before lookup, so
    /// the key is the upstream model id. Defaults to empty (no cost accounting).
    #[serde(default)]
    pub pricing: HashMap<ModelId, ModelPricing>,
    /// In-flight concurrency caps (plan 05 step 7). Absent = uncapped.
    #[serde(default)]
    pub concurrency: Option<ProviderConcurrencyConfig>,
    /// Opt-in pricing sync from models.dev (plan 05 step 9). Absent keeps
    /// operator-typed pricing untouched.
    #[serde(default)]
    pub pricing_sync: Option<PricingSyncConfig>,
    /// Upstream egress (network exit) configuration. When present, this
    /// provider's upstream requests are sent through the configured forward
    /// proxy instead of a direct connection; absent means direct egress.
    #[serde(default)]
    pub egress: Option<ProviderEgressConfig>,
    /// Prompt-cache hygiene (plan 06 steps 6-7, backlog 15): opt-in
    /// request edits at the provider edge that protect the provider's own
    /// prompt cache. Absent — the default — means no hygiene edits at all.
    #[serde(default)]
    pub prompt_cache: Option<PromptCacheConfig>,
}

/// Prompt-cache hygiene (`[provider.prompt_cache]`): the operator's opt-in
/// to volatile-byte removal and cache-marker injection at the provider edge.
/// Nothing here fingerprints or impersonates clients.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCacheConfig {
    /// RFC 6901 JSON pointers stripped from the encoded upstream request
    /// before egress: volatile per-request bytes — client correlation
    /// tokens such as `metadata.user_id` or `user` — that would bust the
    /// provider's prompt cache. A pointer that does not resolve is a no-op.
    #[serde(default)]
    pub strip_volatile: Vec<String>,
    /// Inject cache markers on the stable prefix so the provider caches it.
    /// Anthropic Messages only: `cache_control` breakpoints within the
    /// provider's four-breakpoint budget (the last tool, the last system
    /// block, then the last content block of earlier messages — the newest
    /// message is never marked). Markers the client already set are kept.
    #[serde(default)]
    pub inject_markers: bool,
}

/// Per-provider forward-proxy egress configuration (`[provider.egress]`).
///
/// The proxy URL scheme must be one of `socks5h://`, `socks5://`, `http://`,
/// or `https://`; anything else is rejected at config load, never silently
/// ignored. `socks5h://` resolves upstream hostnames at the proxy (the exit
/// point's DNS view); `socks5://` resolves them locally.
///
/// The URL may carry userinfo (`http://user:pass@host:port`), which is passed
/// to the transport as proxy authentication. [`Debug`] redacts the userinfo;
/// only `scheme://host:port` is ever printed.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderEgressConfig {
    /// Forward proxy URL (`socks5h://host:port`, `socks5://`, `http://`, or
    /// `https://`). Supports `${ENV_VAR}` interpolation like other string
    /// fields. The port is optional and defaults per scheme (1080 for socks).
    pub proxy: String,
}

impl std::fmt::Debug for ProviderEgressConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderEgressConfig")
            .field("proxy", &redacted_proxy_url(&self.proxy))
            .finish()
    }
}

fn redacted_proxy_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return "<unparsed>".to_owned();
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let after_authority = rest.strip_prefix(authority).unwrap_or_default();
    format!("{scheme}://{host}{after_authority}")
}

/// Provider-local model identifier used as a pricing key.
///
/// Transparent newtype over `String` so it serializes as a plain string in
/// TOML/JSON.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(
    /// The model id string (the upstream model id, after alias resolution).
    pub String,
);

impl ModelId {
    /// Create a model id from anything string-like.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    /// The model id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Per-token price for one model, in USD.
///
/// All fields are USD per token as [`Decimal`] to avoid floating-point money. A
/// price of `0` means "unmetered". `cache_creation`, `cache_read`, and
/// `reasoning` default to `0` and may be omitted.
///
/// **Operators must always quote decimal values in TOML** (e.g.
/// `input = "0.0000014"`). An unquoted `0.0000014` is parsed by TOML as an
/// `f64` (losing precision), and `rust_decimal::Decimal` rejects floats, so
/// deserialization fails with a confusing error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPricing {
    /// USD per input token.
    pub input: Decimal,
    /// USD per output token.
    pub output: Decimal,
    /// USD per cache-creation input token.
    #[serde(default)]
    pub cache_creation: Decimal,
    /// USD per cache-read input token.
    #[serde(default)]
    pub cache_read: Decimal,
    /// USD per reasoning token.
    #[serde(default)]
    pub reasoning: Decimal,
}

fn max_per_token_price() -> Decimal {
    Decimal::new(1_000, 0)
}

/// Protocol default allowlists for inbound client headers forwarded upstream
/// (row 12, backlog 10). Anthropic forwards its beta selector and session
/// headers by default so per-request beta features work through the proxy.
#[must_use]
pub fn default_forwarded_inbound_headers(protocol: &str) -> &'static [&'static str] {
    match protocol {
        "anthropic_messages" => &["anthropic-beta", "x-claude-code-session-id"],
        _ => &[],
    }
}

/// Inbound headers that are never forwarded upstream, even when configured:
/// client credentials, hop-by-hop and content headers the transport owns, and
/// browser headers that leak the caller's environment or break upstreams.
/// The proxy's own egress hop marker is owned by the transport and refused on
/// ingress (backlog 32).
pub const NEVER_FORWARDED_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "host",
    "connection",
    "content-length",
    "content-type",
    "accept",
    "accept-encoding",
    "accept-language",
    "origin",
    "referer",
    "user-agent",
    "x-request-id",
    "x-llm-proxy-hop",
];

/// Header-name prefixes that are never forwarded upstream: browser headers
/// that leak the caller's environment, and CORS/security headers the browser
/// owns. One definition shared by config validation and the runtime
/// forwarder.
pub const NEVER_FORWARDED_HEADER_PREFIXES: &[&str] = &["sec-", "access-control-", "browser-"];

/// Whether a lowercased header name is on the never-forwarded set, by exact
/// name or by prefix. Config validation rejects configured allowlist entries
/// with this; the runtime forwarder drops them even when configured.
pub fn is_never_forwarded_header(lower_name: &str) -> bool {
    NEVER_FORWARDED_HEADERS.contains(&lower_name)
        || NEVER_FORWARDED_HEADER_PREFIXES
            .iter()
            .any(|p| lower_name.starts_with(p))
}

/// How the API key is sent in HTTP headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>`
    Bearer,
    /// `x-api-key: <key>`
    XApiKey,
    /// `x-goog-api-key: <key>` (Google/Gemini providers).
    XGoogleApiKey,
    /// Both `Authorization: Bearer <key>` and `x-api-key: <key>`.
    Both,
}

fn serialize_arc_headers<S>(
    headers: &Arc<HashMap<String, String>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    headers.as_ref().serialize(serializer)
}

fn deserialize_arc_headers<'de, D>(
    deserializer: D,
) -> Result<Arc<HashMap<String, String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    HashMap::<String, String>::deserialize(deserializer).map(Arc::new)
}

/// A single adapter entry within a provider.
///
/// Maps a protocol name to an upstream endpoint URL.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderAdapterConfig {
    /// Protocol family (e.g. `"openai_chat_completions"`, `"anthropic_messages"`).
    pub protocol: String,
    /// Upstream endpoint URL.
    ///
    /// **Warning:** This field may contain sensitive query parameters (e.g. API
    /// keys passed as URL query strings). Avoid logging or displaying the raw
    /// value. The [`Debug`](std::fmt::Debug) impl strips query parameters for
    /// safe display.
    pub endpoint: String,
    /// Static headers to include in upstream requests (e.g. anthropic-version).
    ///
    /// Wrapped in [`Arc`]: adapter headers are fixed at config-load time and
    /// shared across every request that resolves to this adapter, so downstream
    /// code that needs to mutate must materialize an owned copy.
    #[serde(
        default,
        serialize_with = "serialize_arc_headers",
        deserialize_with = "deserialize_arc_headers"
    )]
    pub headers: Arc<HashMap<String, String>>,
    /// Inbound client header names (case-insensitive) forwarded verbatim to
    /// this upstream. Unset uses the protocol default allowlist
    /// ([`default_forwarded_inbound_headers`]); headers on the
    /// never-forwarded list (client auth, `Origin`, browser and transport
    /// headers) are rejected at validation even when listed here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_inbound_headers: Option<Vec<String>>,
    /// Per-route `lossy` opt-in: cross-format request fields with no provider
    /// equivalent are dropped with a logged warning instead of rejecting the
    /// request with a 400 that names them.
    #[serde(default)]
    pub lossy: bool,
}

impl ProviderAdapterConfig {
    fn forward_forwarded_for_debug(&self) -> &str {
        match &self.forward_inbound_headers {
            Some(_) => "<configured>",
            None => "<protocol default>",
        }
    }
}

impl std::fmt::Debug for ProviderAdapterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderAdapterConfig")
            .field("protocol", &self.protocol)
            .field("endpoint", &endpoint_without_query(&self.endpoint))
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field(
                "forward_inbound_headers",
                &self.forward_forwarded_for_debug(),
            )
            .field("lossy", &self.lossy)
            .finish()
    }
}

/// The endpoint URL with any query string removed, for safe display (query
/// strings can carry credentials).
#[must_use]
pub fn endpoint_without_query(endpoint: &str) -> &str {
    endpoint
        .split_once('?')
        .map_or(endpoint, |(base, _query)| base)
}

/// An endpoint URL fit for human-facing output (errors, doctor rows): the
/// userinfo (`user:pass@`) is stripped from the authority and the query
/// string and fragment are dropped — all three are places credentials hide
/// (`http://user:pass@host`, a Gemini-style `?key=` parameter). The scheme,
/// host and path stay so the operator can still find the adapter the error
/// is about.
#[must_use]
pub fn endpoint_for_display(endpoint: &str) -> String {
    let Some((scheme, rest)) = endpoint.split_once("://") else {
        // No scheme means no authority, so no userinfo to strip; the query
        // and fragment can still carry secrets.
        return endpoint
            .split(['?', '#'])
            .next()
            .unwrap_or_default()
            .to_owned();
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let path = rest
        .strip_prefix(authority)
        .unwrap_or_default()
        .split(['?', '#'])
        .next()
        .unwrap_or_default();
    format!("{scheme}://{host}{path}")
}

/// The kind of client-facing route being requested.
///
/// Each variant corresponds to an inbound endpoint that the proxy exposes.
/// Providers map these route kinds to adapter names via [`ProviderRoutesConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRouteKind {
    /// OpenAI-compatible `/chat/completions` endpoint.
    ChatCompletions,
    /// Anthropic-compatible `/messages` endpoint.
    Messages,
    /// OpenAI-compatible `/v1/responses` endpoint (plan 08 step 1).
    Responses,
    /// Gemini-native
    /// `/v1beta/models/{model}:generateContent|:streamGenerateContent`
    /// endpoint (plan 08 step 2).
    GenerateContent,
    /// `/v1/embeddings` (plan 09 step 7): a family-specific handler, not
    /// the chat core.
    Embeddings,
    /// `/v1/rerank` (plan 09 step 7).
    Rerank,
    /// `/v1/audio/transcriptions` (plan 09 step 7); multipart body.
    AudioTranscriptions,
    /// `/v1/audio/speech` (plan 09 step 7); binary response.
    AudioSpeech,
    /// `/v1/images/generations` (plan 09 step 7).
    ImagesGenerations,
    /// `/v1/files` and `/v1/files/{id}[/content]` (plan 09 step 7).
    Files,
    /// `/v1/batches` and `/v1/batches/{id}` (plan 09 step 7).
    Batches,
}

impl ProviderRouteKind {
    /// Every route kind, in `[provider.routes]` declaration order.
    pub const ALL: [Self; 11] = [
        Self::ChatCompletions,
        Self::Messages,
        Self::Responses,
        Self::GenerateContent,
        Self::Embeddings,
        Self::Rerank,
        Self::AudioTranscriptions,
        Self::AudioSpeech,
        Self::ImagesGenerations,
        Self::Files,
        Self::Batches,
    ];

    /// This kind's key in `[provider.routes]`.
    #[must_use]
    pub const fn config_key(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat_completions",
            Self::Messages => "messages",
            Self::Responses => "responses",
            Self::GenerateContent => "generate_content",
            Self::Embeddings => "embeddings",
            Self::Rerank => "rerank",
            Self::AudioTranscriptions => "audio_transcriptions",
            Self::AudioSpeech => "audio_speech",
            Self::ImagesGenerations => "images_generations",
            Self::Files => "files",
            Self::Batches => "batches",
        }
    }
}

/// Maps [`ProviderRouteKind`] to adapter names within a provider.
///
/// Defined in TOML as:
///
/// ```toml
/// [provider.routes]
/// chat_completions = "openai_adapter"
/// messages = "anthropic_adapter"
/// responses = "responses_adapter"
/// generate_content = "gemini_adapter"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRoutesConfig {
    /// Route kind: adapter name for the `chat_completions` endpoint. The
    /// Ollama-native `/api/chat` and `/api/generate` routes (plan 08 step 3)
    /// also resolve through this kind: they are chat endpoints reached in a
    /// different client dialect.
    pub chat_completions: Option<String>,
    /// Route kind: adapter name for the `messages` endpoint.
    pub messages: Option<String>,
    /// Route kind: adapter name for the `responses` endpoint.
    #[serde(default)]
    pub responses: Option<String>,
    /// Route kind: adapter name for the `generate_content` endpoint.
    #[serde(default)]
    pub generate_content: Option<String>,
    /// Route kind: adapter name for the `embeddings` family (plan 09
    /// step 7; requires the server's `endpoint-families` feature).
    #[serde(default)]
    pub embeddings: Option<String>,
    /// Route kind: adapter name for the `rerank` family.
    #[serde(default)]
    pub rerank: Option<String>,
    /// Route kind: adapter name for the `audio_transcriptions` family.
    #[serde(default)]
    pub audio_transcriptions: Option<String>,
    /// Route kind: adapter name for the `audio_speech` family.
    #[serde(default)]
    pub audio_speech: Option<String>,
    /// Route kind: adapter name for the `images_generations` family.
    #[serde(default)]
    pub images_generations: Option<String>,
    /// Route kind: adapter name for the `files` family.
    #[serde(default)]
    pub files: Option<String>,
    /// Route kind: adapter name for the `batches` family.
    #[serde(default)]
    pub batches: Option<String>,
}

impl ProviderRoutesConfig {
    /// Look up the adapter name for the given route kind.
    pub fn get(&self, kind: ProviderRouteKind) -> Option<&str> {
        match kind {
            ProviderRouteKind::ChatCompletions => self.chat_completions.as_deref(),
            ProviderRouteKind::Messages => self.messages.as_deref(),
            ProviderRouteKind::Responses => self.responses.as_deref(),
            ProviderRouteKind::GenerateContent => self.generate_content.as_deref(),
            ProviderRouteKind::Embeddings => self.embeddings.as_deref(),
            ProviderRouteKind::Rerank => self.rerank.as_deref(),
            ProviderRouteKind::AudioTranscriptions => self.audio_transcriptions.as_deref(),
            ProviderRouteKind::AudioSpeech => self.audio_speech.as_deref(),
            ProviderRouteKind::ImagesGenerations => self.images_generations.as_deref(),
            ProviderRouteKind::Files => self.files.as_deref(),
            ProviderRouteKind::Batches => self.batches.as_deref(),
        }
    }
}

/// Supported discovery backends for fetching upstream model lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderDiscoveryKind {
    /// Generic OpenAI-style `GET /v1/models` returning `{ "data": [{ "id": ... }] }`.
    #[serde(rename = "openai_compatible_models")]
    OpenAiCompatibleModels,
    /// OpenAI's own model-list endpoint (uses org-level auth).
    #[serde(rename = "openai_models")]
    OpenAiModels,
    /// Anthropic model-list endpoint (`GET /v1/models` with `x-api-key`).
    AnthropicModels,
    /// Google Gemini model-list endpoint.
    GeminiModels,
    /// Fireworks account-level model-list endpoint.
    FireworksAccountModels,
}

/// Configuration for live model discovery from an upstream provider.
///
/// Discovery does **not** run automatically during server startup. Live refresh
/// is triggered explicitly via CLI or a `refresh=live` query parameter on the
/// models endpoint.
///
/// ```toml
/// [provider.discovery]
/// kind = "openai_compatible_models"
/// endpoint = "https://api.example.com/v1/models"
/// max_pages = 100
/// max_models = 20000
/// max_response_bytes = 4194304
/// ```
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDiscoveryConfig {
    /// Which discovery backend protocol to use.
    pub kind: ProviderDiscoveryKind,
    /// Upstream URL to fetch the model list from.
    ///
    /// **Warning:** This field may contain sensitive query parameters. Avoid
    /// logging or displaying the raw value. The [`Debug`](std::fmt::Debug) impl
    /// strips query parameters for safe display.
    pub endpoint: String,
    /// Static headers used only for discovery requests.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Maximum number of pages fetched during one discovery refresh.
    #[serde(default = "default_discovery_max_pages")]
    pub max_pages: usize,
    /// Maximum number of model records accepted during one discovery refresh.
    #[serde(default = "default_discovery_max_models")]
    pub max_models: usize,
    /// Maximum response body size accepted for each discovery page.
    #[serde(default = "default_discovery_max_response_bytes")]
    pub max_response_bytes: usize,
}

const fn default_discovery_max_pages() -> usize {
    100
}

const fn default_discovery_max_models() -> usize {
    20_000
}

const fn default_discovery_max_response_bytes() -> usize {
    4 * 1024 * 1024
}

impl std::fmt::Debug for ProviderDiscoveryConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderDiscoveryConfig")
            .field("kind", &self.kind)
            .field("endpoint", &endpoint_without_query(&self.endpoint))
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("max_pages", &self.max_pages)
            .field("max_models", &self.max_models)
            .field("max_response_bytes", &self.max_response_bytes)
            .finish()
    }
}

/// How the model catalog is assembled for a provider.
///
/// - `Static`: only use operator-defined `[[provider.catalog.models]]` entries.
/// - `Discovered`: only use live/cached discovery results.
/// - `Hybrid`: merge both sources; static metadata wins on ID conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCatalogMode {
    /// Only use `[[provider.catalog.models]]` entries.
    Static,
    /// Only use live/cached discovery results.
    Discovered,
    /// Merge static and discovered models; static metadata wins on ID conflicts.
    #[serde(rename = "hybrid")]
    #[default]
    Hybrid,
}

/// A single static model entry in the provider catalog.
///
/// Defined in TOML as:
///
/// ```toml
/// [[provider.catalog.models]]
/// id = "accounts/fireworks/models/deepseek-v3p1"
/// display_name = "DeepSeek V3.1"
/// supports = ["chat_completions"]
/// context_length = 131_072
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticModelCatalogEntry {
    /// Model ID (must be non-empty, matches the upstream model name).
    pub id: String,
    /// Human-readable display name.
    #[serde(default)]
    pub display_name: Option<String>,
    /// Which route kinds this model supports.
    #[serde(default)]
    pub supports: Vec<ProviderRouteKind>,
    /// Maximum context length in tokens, if known.
    #[serde(default)]
    pub context_length: Option<u32>,
}

/// Provider catalog configuration.
///
/// Controls how the `/providers/{provider}/v1/models` endpoint assembles its
/// response and whether catalog enforcement blocks unknown model IDs.
///
/// ```toml
/// [provider.catalog]
/// mode = "hybrid"
/// enforce = false
/// cache_ttl = "24h"
/// allow = ["*"]
/// deny = ["*-preview"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCatalogConfig {
    /// How to assemble the catalog: static, discovered, or hybrid merge.
    #[serde(default)]
    pub mode: ProviderCatalogMode,
    /// When `true`, requests using a model ID not in the filtered catalog are
    /// rejected. When `false` (default), the catalog is advisory only.
    #[serde(default)]
    pub enforce: bool,
    /// How long to cache discovered models before allowing a refresh.
    #[serde(default = "default_cache_ttl")]
    #[serde(with = "humantime_serde")]
    pub cache_ttl: Duration,
    /// Glob patterns for model IDs that are allowed. `["*"]` allows all.
    #[serde(default = "default_allow_all")]
    pub allow: Vec<String>,
    /// Glob patterns for model IDs that are denied unless a non-wildcard allow
    /// pattern explicitly includes the model.
    #[serde(default)]
    pub deny: Vec<String>,
    /// Operator-defined static model entries.
    #[serde(default)]
    pub models: Vec<StaticModelCatalogEntry>,
}

fn default_cache_ttl() -> Duration {
    Duration::from_secs(24 * 60 * 60)
}

fn default_allow_all() -> Vec<String> {
    vec!["*".to_owned()]
}

impl Default for ProviderCatalogConfig {
    fn default() -> Self {
        Self {
            mode: ProviderCatalogMode::Hybrid,
            enforce: false,
            cache_ttl: default_cache_ttl(),
            allow: default_allow_all(),
            deny: Vec::new(),
            models: Vec::new(),
        }
    }
}

/// Errors produced during config validation.
///
/// This enum is `#[non_exhaustive]` so validation can evolve without breaking
/// downstream `match` expressions.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigValidationError {
    /// A prompt-cache strip entry is not an RFC 6901 JSON pointer
    /// (plan 06 step 7): a malformed pointer would be a silent no-op at the
    /// provider edge, so it is refused at load.
    #[error(
        "provider \"{provider}\": prompt_cache.strip_volatile entries must be RFC 6901 JSON \
         pointers like \"/metadata/user_id\", got {pointer:?}"
    )]
    InvalidStripPointer {
        /// Provider name.
        provider: String,
        /// The offending entry.
        pointer: String,
    },
    /// An adapter `endpoint` URL is empty.
    #[error("provider \"{provider}\": adapter \"{adapter}\" has an empty endpoint")]
    EmptyEndpoint {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
    },
    /// An environment variable referenced in `api_key` could not be resolved.
    #[error(
        "provider \"{provider}\": api_key contains unresolvable environment variable \"{var}\""
    )]
    UnresolvedEnvVar {
        /// Provider name.
        provider: String,
        /// The `${VAR}` pattern that could not be resolved.
        var: String,
    },
    /// An environment variable resolved to an empty value.
    ///
    /// This is emitted by [`load_provider_config`], which scans the entire raw
    /// TOML for `${VAR}` references (any field: `endpoint`, `protocol`, `name`,
    /// `headers`, `api_key`, ...), not just `api_key`. The message is therefore
    /// field-agnostic; it does not name a specific field.
    #[error(
        "provider \"{provider}\": environment variable \"{var}\" referenced in the provider config resolved to an empty value"
    )]
    EmptyEnvVar {
        /// Provider name.
        provider: String,
        /// The environment variable name.
        var: String,
    },
    /// The literal `api_key` value (after interpolation) is empty.
    #[error("provider \"{provider}\": api_key is empty")]
    EmptyApiKey {
        /// Provider name.
        provider: String,
    },
    /// A concurrency cap is invalid (zero would refuse every request).
    #[error("provider \"{provider}\": concurrency cap is invalid: {message}")]
    InvalidConcurrency {
        /// Provider whose cap is invalid.
        provider: String,
        /// What is wrong.
        message: String,
    },
    /// A pricing entry is invalid (empty model id or negative price).
    #[error("provider \"{provider}\": pricing for model \"{model}\" is invalid: {message}")]
    InvalidPricing {
        /// Provider name.
        provider: String,
        /// Model id whose pricing is invalid.
        model: String,
        /// What is wrong (e.g. "negative input price").
        message: String,
    },
    /// A provider name is empty.
    #[error("provider name is empty")]
    EmptyProviderName,
    /// A catalog model entry has an empty ID.
    #[error("provider \"{provider}\": catalog entry has an empty id")]
    EmptyCatalogEntryId {
        /// Provider name.
        provider: String,
    },
    /// An adapter name is empty.
    #[error("provider \"{provider}\": adapter name is empty")]
    EmptyAdapterName {
        /// Provider name.
        provider: String,
    },
    /// An adapter has an empty protocol.
    #[error("provider \"{provider}\": adapter \"{adapter}\" has an empty protocol")]
    EmptyProtocol {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
    },
    /// A protocol is not in the known set supplied by the caller.
    #[error("provider \"{provider}\": adapter \"{adapter}\" uses unknown protocol \"{protocol}\"")]
    UnknownProtocol {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
        /// Protocol name.
        protocol: String,
    },
    /// An adapter endpoint has a non-HTTP(S) URL scheme.
    #[error(
        "provider \"{provider}\": adapter \"{adapter}\" has invalid endpoint scheme (expected http:// or https://): \"{endpoint}\""
    )]
    InvalidEndpointScheme {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
        /// The endpoint URL that has an invalid scheme.
        endpoint: String,
    },
    /// An endpoint would carry this provider's credentials in cleartext to an
    /// off-machine host (backlog 25): an `http://` adapter or discovery URL
    /// whose host is neither a loopback address nor `localhost`. Loopback is
    /// exempt; a hostname that merely resolves to loopback is not, because
    /// validation runs offline and a name is exactly what a mistake or an
    /// attacker can repoint.
    #[error(
        "provider \"{provider}\": adapter \"{adapter}\" endpoint \"{endpoint}\" would send the upstream credential in cleartext to the off-machine host \"{host}\"; use https:// (or a loopback host such as 127.0.0.1 behind a local tunnel)"
    )]
    CleartextCredentials {
        /// Provider name.
        provider: String,
        /// Adapter name (`"discovery"` for the discovery endpoint).
        adapter: String,
        /// The cleartext endpoint URL.
        endpoint: String,
        /// The off-machine host the endpoint names.
        host: String,
    },
    /// The `[provider.egress]` proxy URL is not usable as a forward proxy.
    #[error(
        "provider \"{provider}\": [provider.egress] proxy URL is invalid ({reason}): \"{proxy}\""
    )]
    InvalidEgressProxy {
        /// Provider name.
        provider: String,
        /// The rejected proxy URL.
        proxy: String,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// An adapter header name or value contains CR or LF characters.
    #[error("provider \"{provider}\": adapter \"{adapter}\" header contains CR/LF characters")]
    HeaderContainsCrlf {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
    },
    /// An adapter header uses a forbidden transport header name.
    #[error("provider \"{provider}\": adapter \"{adapter}\" sets forbidden header \"{header}\"")]
    ForbiddenHeader {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
        /// The forbidden header name.
        header: String,
    },
    /// A configured inbound-header forwarding name is on the never-forwarded
    /// list (client credentials, hop-by-hop, browser headers).
    #[error(
        "provider \"{provider}\": adapter \"{adapter}\" may not forward inbound header \"{header}\" (client credentials, hop-by-hop, and browser headers are never forwarded)"
    )]
    ForbiddenForwardedHeader {
        /// Name of the offending provider.
        provider: String,
        /// Name of the offending adapter.
        adapter: String,
        /// The header name that may not be forwarded.
        header: String,
    },
    /// An adapter header name is empty.
    #[error("provider \"{provider}\": adapter \"{adapter}\" has an empty header name")]
    EmptyHeaderName {
        /// Provider name.
        provider: String,
        /// Adapter name.
        adapter: String,
    },
    /// A provider name contains invalid characters (must be URL-safe slug).
    #[error(
        "provider name \"{provider}\" contains invalid characters; \
         only lowercase ASCII letters, digits, hyphens, and underscores are allowed"
    )]
    InvalidProviderNameFormat {
        /// The invalid provider name.
        provider: String,
    },
    /// A route adapter name is empty.
    #[error("provider \"{provider}\": route \"{route_kind}\" has an empty adapter name")]
    EmptyRouteAdapterName {
        /// Provider name.
        provider: String,
        /// Route kind (e.g. "chat_completions" or "messages").
        route_kind: String,
    },
    /// A route references an adapter that does not exist.
    #[error(
        "provider \"{provider}\": route \"{route_kind}\" references unknown adapter \"{adapter}\""
    )]
    UnknownRouteAdapter {
        /// Provider name.
        provider: String,
        /// Route kind.
        route_kind: String,
        /// Adapter name.
        adapter: String,
    },
    /// A model alias key is empty.
    #[error("provider \"{provider}\": model alias key is empty")]
    EmptyModelAliasKey {
        /// Provider name.
        provider: String,
    },
    /// A model alias target is empty.
    #[error("provider \"{provider}\": model alias \"{alias}\" has an empty target")]
    EmptyModelAliasTarget {
        /// Provider name.
        provider: String,
        /// The alias key.
        alias: String,
    },
    /// A configured model ID cannot be safely interpolated into a Gemini URL template.
    #[error(
        "provider \"{provider}\": {location} model ID \"{model}\" contains characters that are unsafe for Gemini URL templates"
    )]
    UnsafeGeminiModelId {
        /// Provider name.
        provider: String,
        /// Configuration source, such as a catalog entry or alias target.
        location: &'static str,
        /// Unsafe model ID.
        model: String,
    },
    /// A discovery resource limit is zero.
    #[error("provider \"{provider}\": discovery limit \"{limit}\" must be greater than zero")]
    InvalidDiscoveryLimit {
        /// Provider name.
        provider: String,
        /// Invalid limit name.
        limit: &'static str,
    },
    /// A phase-shaped timeout rule is broken (backlog 8, BUG-14).
    #[error("invalid upstream timeouts: {message}")]
    InvalidTimeouts {
        /// What is wrong.
        message: String,
    },
    /// A pool definition is invalid (plan 04 step 4).
    #[error("pool \"{pool}\": {message}")]
    InvalidPool {
        /// Pool name.
        pool: String,
        /// What is wrong.
        message: String,
    },
    /// A retry ladder on a provider or pool is invalid.
    #[error("retry config for \"{route}\": {message}")]
    InvalidRetry {
        /// Provider or pool name.
        route: String,
        /// What is wrong.
        message: String,
    },
}

/// Validate a single [`ProviderConfig`].
///
/// Checks:
/// - provider name is non-empty (whitespace-only names are rejected)
/// - `api_key` is non-empty (after interpolation) and no unresolved `${VAR}` patterns remain
/// - all adapter names, protocol names, and endpoints are non-empty (whitespace-only values are rejected)
/// - adapter static headers reject CRLF characters and forbidden transport headers
/// - provider name is a URL-safe slug (lowercase ASCII letters, digits, hyphens, underscores)
/// - route adapter references point to existing adapters
/// - model alias keys and targets are non-empty
/// - static catalog IDs and alias targets used by Gemini URL templates are path-safe
///
/// `known_protocols` is an optional list of protocol names that are compiled
/// into the running binary. When `None`, protocol-name validation is skipped
/// (the core crate does not own the adapter registry). The server layer passes
/// `Some(ProviderAdapterRegistry::builtin().protocol_names())` for full validation.
///
/// The caller should ensure env-var interpolation has already been performed on
/// the `api_key` field before calling this function. If the raw `${VAR}` pattern
/// remains in `api_key`, it will be caught as an unresolved env var.
///
/// # Errors
///
/// Returns a [`ConfigValidationError`] variant describing the first validation
/// failure encountered. Checks run in a defined order: provider name, unresolved
/// env vars in `api_key`, empty `api_key`, adapter names/protocols/endpoints/headers,
/// protocol-name membership, route adapter references,
/// and model alias validation.
///
/// # Examples
///
/// A minimal, well-formed provider validates cleanly:
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use llm_proxy_core::provider_config::{
///     AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRoutesConfig,
///     validate_provider_config,
/// };
/// use secrecy::SecretString;
///
/// let provider = ProviderConfig {
///     name: "example".to_owned(),
///     api_key: SecretString::from("secret"),
///     auth_style: AuthStyle::Bearer,
///     passthrough_auth: false,
///     api_keys: None,
///     retry: None,
///     egress: None,
///     adapters: HashMap::from([(
///         "chat".to_owned(),
///         ProviderAdapterConfig {
///             protocol: "openai_chat_completions".to_owned(),
///             endpoint: "https://example.com/v1/chat/completions".to_owned(),
///             headers: Arc::new(HashMap::new()),
///             forward_inbound_headers: None,
///             lossy: false,
///         },
///     )]),
///     routes: ProviderRoutesConfig::default(),
///     model_aliases: HashMap::new(),
///     discovery: None,
///     catalog: None,
///     pricing: Default::default(),
///     concurrency: None,
///     pricing_sync: None,
///     prompt_cache: None,
/// };
/// assert!(validate_provider_config(&provider, None).is_ok());
/// ```
///
/// An empty provider name is rejected with [`ConfigValidationError::EmptyProviderName`]:
///
/// ```
/// use std::collections::HashMap;
/// use llm_proxy_core::provider_config::{
///     AuthStyle, ConfigValidationError, ProviderConfig, ProviderRoutesConfig,
///     validate_provider_config,
/// };
/// use secrecy::SecretString;
///
/// let provider = ProviderConfig {
///     name: "   ".to_owned(), // whitespace-only is treated as empty
///     api_key: SecretString::from("secret"),
///     auth_style: AuthStyle::Bearer,
///     passthrough_auth: false,
///     api_keys: None,
///     retry: None,
///     egress: None,
///     adapters: HashMap::new(),
///     routes: ProviderRoutesConfig::default(),
///     model_aliases: HashMap::new(),
///     discovery: None,
///     catalog: None,
///     pricing: Default::default(),
///     concurrency: None,
///     pricing_sync: None,
///     prompt_cache: None,
/// };
/// assert_eq!(
///     validate_provider_config(&provider, None),
///     Err(ConfigValidationError::EmptyProviderName),
/// );
/// ```
pub fn validate_provider_config(
    provider: &ProviderConfig,
    known_protocols: Option<&[&str]>,
) -> Result<(), ConfigValidationError> {
    let name = &provider.name;

    validate_name(name)?;
    validate_api_key(provider, name)?;
    validate_pricing(provider, name)?;
    validate_pricing_sync(provider, name)?;
    validate_concurrency(provider, name)?;
    validate_adapters(provider, name, known_protocols)?;
    validate_discovery(provider, name)?;
    validate_egress(provider, name)?;
    validate_prompt_cache(provider, name)?;

    let uses_gemini_model_template = provider.adapters.values().any(|adapter| {
        adapter.protocol == "gemini_generate_content" && adapter.endpoint.contains("{model}")
    });
    validate_catalog(provider, name, uses_gemini_model_template)?;
    validate_routes(provider, name)?;
    validate_model_aliases(provider, name, uses_gemini_model_template)?;

    Ok(())
}

fn validate_name(name: &str) -> Result<(), ConfigValidationError> {
    if name.trim().is_empty() {
        return Err(ConfigValidationError::EmptyProviderName);
    }

    #[expect(
        clippy::expect_used,
        reason = "the pattern is a compile-time-constant regex literal; a failure is a \
                  programming error caught the moment this static is first used"
    )]
    static PROVIDER_NAME_SLUG: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^[a-z0-9_-]{1,64}$").expect("valid provider name slug regex")
    });
    if !PROVIDER_NAME_SLUG.is_match(name) {
        return Err(ConfigValidationError::InvalidProviderNameFormat {
            provider: name.to_owned(),
        });
    }
    Ok(())
}

fn validate_api_key(provider: &ProviderConfig, name: &str) -> Result<(), ConfigValidationError> {
    if provider.passthrough_auth {
        return Ok(());
    }
    // Every key in the pool reaches the upstream verbatim, so each gets the
    // same checks as the base key.
    let extras = provider.api_keys.iter().flatten();
    for key in std::iter::once(&provider.api_key).chain(extras) {
        if let Some(var) = find_unresolved_env_var(key.expose_secret()) {
            return Err(ConfigValidationError::UnresolvedEnvVar {
                provider: name.to_owned(),
                var,
            });
        }
        if key.expose_secret().trim().is_empty() {
            return Err(ConfigValidationError::EmptyApiKey {
                provider: name.to_owned(),
            });
        }
    }
    if let Some(retry) = &provider.retry
        && retry.max_attempts == 0
    {
        return Err(ConfigValidationError::InvalidRetry {
            route: name.to_owned(),
            message: "retry.max_attempts must be at least 1".to_owned(),
        });
    }
    Ok(())
}

fn validate_pricing_sync(
    provider: &ProviderConfig,
    name: &str,
) -> Result<(), ConfigValidationError> {
    if let Some(sync) = &provider.pricing_sync
        && sync.models_dev_id.trim().is_empty()
    {
        return Err(ConfigValidationError::InvalidConcurrency {
            provider: name.to_owned(),
            message: "pricing_sync.models_dev_id must not be empty".to_owned(),
        });
    }
    Ok(())
}

fn validate_concurrency(
    provider: &ProviderConfig,
    name: &str,
) -> Result<(), ConfigValidationError> {
    let Some(concurrency) = &provider.concurrency else {
        return Ok(());
    };
    if concurrency.total == Some(0) {
        return Err(ConfigValidationError::InvalidConcurrency {
            provider: name.to_owned(),
            message: "total must be at least 1, not 0 (0 would refuse every request)".to_owned(),
        });
    }
    for (model_id, cap) in &concurrency.models {
        if *cap == 0 {
            return Err(ConfigValidationError::InvalidConcurrency {
                provider: name.to_owned(),
                message: format!(
                    "model cap for \"{}\" must be at least 1, not 0 (0 would refuse every request)",
                    model_id.as_str()
                ),
            });
        }
    }
    Ok(())
}

fn validate_pricing(provider: &ProviderConfig, name: &str) -> Result<(), ConfigValidationError> {
    for (model_id, pricing) in &provider.pricing {
        if model_id.as_str().is_empty() {
            return Err(ConfigValidationError::InvalidPricing {
                provider: name.to_owned(),
                model: model_id.as_str().to_owned(),
                message: "pricing key is empty".to_owned(),
            });
        }
        for (field, value) in [
            ("input", pricing.input),
            ("output", pricing.output),
            ("cache_creation", pricing.cache_creation),
            ("cache_read", pricing.cache_read),
            ("reasoning", pricing.reasoning),
        ] {
            if value < Decimal::ZERO {
                return Err(ConfigValidationError::InvalidPricing {
                    provider: name.to_owned(),
                    model: model_id.as_str().to_owned(),
                    message: format!("negative {field} price"),
                });
            }
            let max_price = max_per_token_price();
            if value > max_price {
                return Err(ConfigValidationError::InvalidPricing {
                    provider: name.to_owned(),
                    model: model_id.as_str().to_owned(),
                    message: format!(
                        "{field} price {value} exceeds the ${max_price}/token upper \
                         bound; per-token USD prices are expected to be far below $1 -- did you \
                         enter a per-1M-token price?"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_adapters(
    provider: &ProviderConfig,
    name: &str,
    known_protocols: Option<&[&str]>,
) -> Result<(), ConfigValidationError> {
    for (adapter_name, adapter_cfg) in &provider.adapters {
        if adapter_name.trim().is_empty() {
            return Err(ConfigValidationError::EmptyAdapterName {
                provider: name.to_owned(),
            });
        }
        if adapter_cfg.protocol.trim().is_empty() {
            return Err(ConfigValidationError::EmptyProtocol {
                provider: name.to_owned(),
                adapter: adapter_name.clone(),
            });
        }
        if adapter_cfg.endpoint.trim().is_empty() {
            return Err(ConfigValidationError::EmptyEndpoint {
                provider: name.to_owned(),
                adapter: adapter_name.clone(),
            });
        }
        if !has_valid_http_scheme(adapter_cfg.endpoint.trim()) {
            return Err(ConfigValidationError::InvalidEndpointScheme {
                provider: name.to_owned(),
                adapter: adapter_name.clone(),
                endpoint: endpoint_for_display(&adapter_cfg.endpoint),
            });
        }
        validate_cleartext_endpoint(name, adapter_name, &adapter_cfg.endpoint)?;
        if let Some(known) = known_protocols
            && !known.contains(&adapter_cfg.protocol.as_str())
        {
            return Err(ConfigValidationError::UnknownProtocol {
                provider: name.to_owned(),
                adapter: adapter_name.clone(),
                protocol: adapter_cfg.protocol.clone(),
            });
        }

        validate_headers(&adapter_cfg.headers, name, adapter_name)?;
        validate_forwarded_inbound_headers(adapter_cfg, name, adapter_name)?;
    }
    Ok(())
}

/// Validates a configured inbound-header forwarding allowlist: names must be
/// non-empty, CRLF-free, and not on the never-forwarded list.
fn validate_forwarded_inbound_headers(
    adapter_cfg: &ProviderAdapterConfig,
    provider: &str,
    adapter: &str,
) -> Result<(), ConfigValidationError> {
    let Some(names) = &adapter_cfg.forward_inbound_headers else {
        return Ok(());
    };
    for header in names {
        if header.trim().is_empty() {
            return Err(ConfigValidationError::EmptyHeaderName {
                provider: provider.to_owned(),
                adapter: adapter.to_owned(),
            });
        }
        if header.contains('\r') || header.contains('\n') {
            return Err(ConfigValidationError::HeaderContainsCrlf {
                provider: provider.to_owned(),
                adapter: adapter.to_owned(),
            });
        }
        let lower = header.to_ascii_lowercase();
        if is_never_forwarded_header(&lower) {
            return Err(ConfigValidationError::ForbiddenForwardedHeader {
                provider: provider.to_owned(),
                adapter: adapter.to_owned(),
                header: header.clone(),
            });
        }
    }
    Ok(())
}

fn validate_discovery(provider: &ProviderConfig, name: &str) -> Result<(), ConfigValidationError> {
    let Some(discovery) = &provider.discovery else {
        return Ok(());
    };
    let endpoint = discovery.endpoint.trim();
    if endpoint.is_empty() {
        return Err(ConfigValidationError::EmptyEndpoint {
            provider: name.to_owned(),
            adapter: "discovery".to_owned(),
        });
    }
    if !has_valid_http_scheme(endpoint) {
        return Err(ConfigValidationError::InvalidEndpointScheme {
            provider: name.to_owned(),
            adapter: "discovery".to_owned(),
            endpoint: endpoint_for_display(&discovery.endpoint),
        });
    }
    validate_cleartext_endpoint(name, "discovery", &discovery.endpoint)?;
    validate_headers(&discovery.headers, name, "discovery")?;
    for (limit, value) in [
        ("max_pages", discovery.max_pages),
        ("max_models", discovery.max_models),
        ("max_response_bytes", discovery.max_response_bytes),
    ] {
        if value == 0 {
            return Err(ConfigValidationError::InvalidDiscoveryLimit {
                provider: name.to_owned(),
                limit,
            });
        }
    }
    Ok(())
}

fn validate_catalog(
    provider: &ProviderConfig,
    name: &str,
    uses_gemini_model_template: bool,
) -> Result<(), ConfigValidationError> {
    let Some(catalog) = &provider.catalog else {
        return Ok(());
    };
    for entry in &catalog.models {
        if entry.id.trim().is_empty() {
            return Err(ConfigValidationError::EmptyCatalogEntryId {
                provider: name.to_owned(),
            });
        }
        if uses_gemini_model_template && !is_safe_gemini_model_id(&entry.id) {
            return Err(ConfigValidationError::UnsafeGeminiModelId {
                provider: name.to_owned(),
                location: "catalog entry",
                model: entry.id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_routes(provider: &ProviderConfig, name: &str) -> Result<(), ConfigValidationError> {
    for kind in ProviderRouteKind::ALL {
        let Some(adapter_name) = provider.routes.get(kind) else {
            continue;
        };
        if adapter_name.trim().is_empty() {
            return Err(ConfigValidationError::EmptyRouteAdapterName {
                provider: name.to_owned(),
                route_kind: kind.config_key().to_owned(),
            });
        }
        // The registry resolves `provider.routes.get(kind)` to
        // `provider.adapters.get(adapter_name)` with the value verbatim, so
        // the existence check must compare the same exact string. A
        // whitespace-padded name is therefore simply unknown here.
        if !provider.adapters.contains_key(adapter_name) {
            return Err(ConfigValidationError::UnknownRouteAdapter {
                provider: name.to_owned(),
                route_kind: kind.config_key().to_owned(),
                adapter: adapter_name.to_owned(),
            });
        }
    }
    Ok(())
}

fn validate_model_aliases(
    provider: &ProviderConfig,
    name: &str,
    uses_gemini_model_template: bool,
) -> Result<(), ConfigValidationError> {
    for (alias_key, alias_target) in &provider.model_aliases {
        if alias_key.trim().is_empty() {
            return Err(ConfigValidationError::EmptyModelAliasKey {
                provider: name.to_owned(),
            });
        }
        if alias_target.trim().is_empty() {
            return Err(ConfigValidationError::EmptyModelAliasTarget {
                provider: name.to_owned(),
                alias: alias_key.clone(),
            });
        }
        if uses_gemini_model_template && !is_safe_gemini_model_id(alias_target) {
            return Err(ConfigValidationError::UnsafeGeminiModelId {
                provider: name.to_owned(),
                location: "model alias target",
                model: alias_target.clone(),
            });
        }
    }
    Ok(())
}

const FORBIDDEN_HEADERS: &[&str] = &[
    "authorization",
    "cookie",
    "set-cookie",
    "upgrade",
    "proxy-authorization",
    "host",
    "connection",
    "content-length",
    "transfer-encoding",
];

fn has_valid_http_scheme(endpoint: &str) -> bool {
    let scheme_region = scheme_region(endpoint);
    scheme_region.eq_ignore_ascii_case("http") || scheme_region.eq_ignore_ascii_case("https")
}

/// The host of an endpoint URL's authority, brackets stripped from IPv6
/// literals. userinfo (`user:pass@`) is dropped. `None` when there is no host.
fn endpoint_host(endpoint: &str) -> Option<&str> {
    let after_scheme = endpoint
        .split_once("://")
        .map_or(endpoint, |(_, rest)| rest);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    let authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = if let Some(inner) = authority.strip_prefix('[') {
        inner.split_once(']')?.0
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    (!host.is_empty()).then_some(host)
}

/// Whether an endpoint URL's host is machine-local by spelling alone: a
/// loopback IP literal (any `127.0.0.0/8` or `::1` form) or `localhost`.
/// A hostname that merely resolves to loopback does not count — validation
/// runs without DNS, and a resolvable name is re-pointable by whoever
/// controls it.
fn host_is_machine_local(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Refuses an `http://` endpoint to an off-machine host (backlog 25): the
/// upstream credential — the provider key, or the client's own token on a
/// `passthrough_auth` provider — would cross the network in cleartext.
fn validate_cleartext_endpoint(
    provider: &str,
    adapter: &str,
    endpoint: &str,
) -> Result<(), ConfigValidationError> {
    if !scheme_region(endpoint.trim()).eq_ignore_ascii_case("http") {
        return Ok(());
    }
    if let Some(host) = endpoint_host(endpoint.trim())
        && !host_is_machine_local(host)
    {
        return Err(ConfigValidationError::CleartextCredentials {
            provider: provider.to_owned(),
            adapter: adapter.to_owned(),
            endpoint: endpoint_for_display(endpoint),
            host: host.to_owned(),
        });
    }
    Ok(())
}

fn scheme_region(endpoint: &str) -> &str {
    match endpoint.find("://") {
        Some(idx) => &endpoint[..idx],
        None => "",
    }
}

const EGRESS_PROXY_SCHEMES: [&str; 4] = ["socks5h", "socks5", "http", "https"];

fn validate_egress(provider: &ProviderConfig, name: &str) -> Result<(), ConfigValidationError> {
    let Some(egress) = &provider.egress else {
        return Ok(());
    };
    let url = egress.proxy.trim();
    if url.is_empty() {
        return Err(ConfigValidationError::InvalidEgressProxy {
            provider: name.to_owned(),
            proxy: egress.proxy.clone(),
            reason: "empty proxy URL",
        });
    }
    let scheme = scheme_region(url);
    if scheme.is_empty() {
        return Err(ConfigValidationError::InvalidEgressProxy {
            provider: name.to_owned(),
            proxy: egress.proxy.clone(),
            reason: "no URL scheme; expected socks5h://, socks5://, http://, or https://",
        });
    }
    if !EGRESS_PROXY_SCHEMES
        .iter()
        .any(|accepted| scheme.eq_ignore_ascii_case(accepted))
    {
        return Err(ConfigValidationError::InvalidEgressProxy {
            provider: name.to_owned(),
            proxy: egress.proxy.clone(),
            reason: "unsupported scheme; expected socks5h://, socks5://, http://, or https://",
        });
    }
    let authority = url
        .strip_prefix(scheme)
        .and_then(|after_scheme| after_scheme.strip_prefix("://"))
        .unwrap_or_default();
    let authority = authority.split(['/', '?', '#']).next().unwrap_or_default();
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    if host.trim().is_empty() {
        return Err(ConfigValidationError::InvalidEgressProxy {
            provider: name.to_owned(),
            proxy: egress.proxy.clone(),
            reason: "no proxy host",
        });
    }
    Ok(())
}

/// Validates `[provider.prompt_cache]` (plan 06 step 7): every strip
/// pointer must be an RFC 6901 pointer (`/metadata/user_id`), so a typo is
/// a config error instead of a silent no-op at the provider edge.
fn validate_prompt_cache(
    provider: &ProviderConfig,
    name: &str,
) -> Result<(), ConfigValidationError> {
    let Some(config) = &provider.prompt_cache else {
        return Ok(());
    };
    for pointer in &config.strip_volatile {
        let invalid = pointer.is_empty()
            || !pointer.starts_with('/')
            || pointer.ends_with('/')
            || pointer.contains("//")
            || has_invalid_escape(pointer);
        if invalid {
            return Err(ConfigValidationError::InvalidStripPointer {
                provider: name.to_owned(),
                pointer: pointer.clone(),
            });
        }
    }
    Ok(())
}

/// RFC 6901: within a reference token, `~` must be followed by `0` (a
/// literal `~`) or `1` (a literal `/`); any other follower, or a dangling
/// `~` at the end of a token, is an invalid escape.
fn has_invalid_escape(pointer: &str) -> bool {
    let mut chars = pointer.chars().peekable();
    while let Some(current) = chars.next() {
        if current == '~' && !matches!(chars.peek(), Some(&'0') | Some(&'1')) {
            return true;
        }
    }
    false
}

fn validate_headers(
    headers: &HashMap<String, String>,
    provider: &str,
    adapter: &str,
) -> Result<(), ConfigValidationError> {
    for (hname, hvalue) in headers {
        if hname.trim().is_empty() {
            return Err(ConfigValidationError::EmptyHeaderName {
                provider: provider.to_owned(),
                adapter: adapter.to_owned(),
            });
        }
        if hname.contains('\r')
            || hname.contains('\n')
            || hvalue.contains('\r')
            || hvalue.contains('\n')
        {
            return Err(ConfigValidationError::HeaderContainsCrlf {
                provider: provider.to_owned(),
                adapter: adapter.to_owned(),
            });
        }
        let lower = hname.to_ascii_lowercase();
        if FORBIDDEN_HEADERS.contains(&lower.as_str()) {
            return Err(ConfigValidationError::ForbiddenHeader {
                provider: provider.to_owned(),
                adapter: adapter.to_owned(),
                header: hname.clone(),
            });
        }
    }
    Ok(())
}

fn is_safe_gemini_model_id(model: &str) -> bool {
    let component = model.strip_prefix("models/").unwrap_or(model);
    !component.is_empty()
        && component != "."
        && component != ".."
        && component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
}

/// Load and parse a main [`AppConfig`] TOML file.
///
/// The raw file content is first processed for `${ENV_VAR}` interpolation,
/// then parsed as TOML, then validated.
///
/// # Blocking I/O
///
/// This function performs synchronous filesystem I/O and is intended for use
/// at startup. If hot-reload is implemented in future, an async variant will
/// be needed.
///
/// # Trust boundary
///
/// The `path` parameter accepts arbitrary filesystem paths without
/// canonicalisation or path-traversal checks. This is standard for
/// env-var/CLI-driven config loading -- the process owner controls the path.
/// The config file content is treated as trusted input.
///
/// # Errors
///
/// Returns a [`CoreError`] if the file cannot be read, the TOML is malformed,
/// env-var interpolation encounters issues, or route validation fails.
pub fn load_app_config(path: impl AsRef<Path>) -> Result<AppConfig, CoreError> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|source| CoreError::ConfigLoad {
        path: path.to_path_buf(),
        source,
    })?;

    for (var_name, resolved) in find_env_var_refs(&raw) {
        if let Some(value) = resolved {
            if value.is_empty() {
                return Err(CoreError::ConfigValidation {
                    message: format!(
                        "app config environment variable \"{var_name}\" resolved to an empty value"
                    ),
                    source: None,
                });
            }
        }
    }

    let interpolated = interpolate_env_vars(&raw);

    if let Some(var) = find_unresolved_env_var(&interpolated) {
        return Err(CoreError::ConfigValidation {
            message: format!("app config contains unresolvable environment variable \"{var}\""),
            source: None,
        });
    }

    let cfg: AppConfig = toml::from_str(&interpolated).map_err(CoreError::ConfigParse)?;
    const MIN_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);
    if cfg.server.request_timeout < MIN_REQUEST_TIMEOUT {
        return Err(CoreError::ConfigValidation {
            message: format!(
                "server.request_timeout must be at least 1 second, but is {:?}; \
                 a zero or sub-second timeout makes every API request return 408 \
                 before any handler runs",
                cfg.server.request_timeout
            ),
            source: None,
        });
    }
    validate_auth_config(&cfg.auth, cfg.server.bind)?;
    if cfg.server.cache.enabled {
        if cfg.server.cache.ttl.is_zero() {
            return Err(CoreError::ConfigValidation {
                message: "server.cache.ttl must be greater than 0s when the \
                          response cache is enabled"
                    .to_owned(),
                source: None,
            });
        }
        if cfg.server.cache.max_entries == 0 {
            return Err(CoreError::ConfigValidation {
                message: "server.cache.max_entries must be at least 1 when the \
                          response cache is enabled"
                    .to_owned(),
                source: None,
            });
        }
    }
    if cfg.server.cache.enabled && cfg.server.semantic_cache.enabled {
        return Err(CoreError::ConfigValidation {
            message: "server.cache and server.semantic_cache are mutually exclusive \
                      opt-ins; enabling both would silently replace the exact-match \
                      cache with the semantic one — enable one"
                .to_owned(),
            source: None,
        });
    }
    if cfg.server.idempotency.enabled && cfg.server.idempotency.ttl.is_zero() {
        return Err(CoreError::ConfigValidation {
            message: "server.idempotency.ttl must be greater than 0s when \
                      Idempotency-Key handling is enabled"
                .to_owned(),
            source: None,
        });
    }
    if cfg.server.idempotency.enabled && cfg.server.idempotency.max_entries == 0 {
        return Err(CoreError::ConfigValidation {
            message: "server.idempotency.max_entries must be at least 1 when \
                      Idempotency-Key handling is enabled"
                .to_owned(),
            source: None,
        });
    }
    if let Some(storage) = cfg.storage.as_ref() {
        validate_storage_config(storage)?;
    }
    validate_metrics_config(&cfg.server.metrics)?;
    validate_admin_mount(&cfg)?;
    validate_upstream_timeouts(&cfg.server.upstream_timeouts).map_err(|error| {
        CoreError::ConfigValidation {
            message: error.to_string(),
            source: None,
        }
    })?;
    validate_bare_v1(&cfg)?;
    Ok(cfg)
}

/// Bare `/v1/*` validation within the app config alone: the model map must
/// not be empty while enabled without a default target (it could never
/// resolve anything). Targets are cross-checked against the loaded registry
/// where it is available — startup and `llm-proxy validate` — via
/// [`bare_v1_target_problems`], because provider files load after this runs;
/// an unresolved target still 404s at request time naming the target.
fn validate_bare_v1(cfg: &AppConfig) -> Result<(), CoreError> {
    let bare = &cfg.server.bare_v1;
    if !bare.enabled {
        return Ok(());
    }
    if bare.models.is_empty() && bare.default_target.is_none() {
        return Err(CoreError::ConfigValidation {
            message: "server.bare_v1 is enabled but has no model mappings and no \
                      default_target; no request could resolve"
                .to_owned(),
            source: None,
        });
    }
    Ok(())
}

/// Cross-checks every configured bare `/v1/*` target against `route_exists`
/// (the registry's provider and pool names). Returns one message per target
/// that names nothing loadable; a disabled [`BareV1Config`] has no targets to
/// check. Called by the startup path and the validate command, which see both
/// the app config and the registry.
#[must_use]
pub fn bare_v1_target_problems(
    bare: &BareV1Config,
    route_exists: impl Fn(&str) -> bool,
) -> Vec<String> {
    if !bare.enabled {
        return Vec::new();
    }
    let mut problems = Vec::new();
    for (model, target) in &bare.models {
        if !route_exists(target) {
            problems.push(format!(
                "server.bare_v1 model \"{model}\" targets unknown provider or pool \"{target}\""
            ));
        }
    }
    if let Some(target) = &bare.default_target
        && !route_exists(target)
    {
        problems.push(format!(
            "server.bare_v1 default_target \"{target}\" names no configured provider or pool"
        ));
    }
    problems
}

/// Names every change between two config snapshots that hot reload cannot
/// apply and therefore refuses (plan 07 step 11, backlog 23): any `[server]`
/// field (compared as rendered values, so the message can name the field),
/// the auth mode, the whole `[auth.clients]` map (key ids as a set, and —
/// under an unchanged id — the verifier and policy too, named
/// `auth.clients.<id>`), the admin token, the admin key store
/// (`auth.admin.key_store`), the storage ledger, the `[replica]` section
/// and the `[mcp]` section. Provider and pool files are reloadable and are
/// not compared here. Empty result: the reload may proceed.
///
/// Secrets are compared by equality and never rendered. A verifier rotated
/// under an existing key id or an edited policy is restart-required as
/// well: the auth middleware and the engine's admission are not rebuilt by
/// a reload, so applying the rest of the file while they keep the old key
/// material would silently keep authenticating a revoked token.
#[must_use]
pub fn restart_required_changes(current: &AppConfig, next: &AppConfig) -> Vec<String> {
    let mut changes = Vec::new();

    let rendered = |config: &ServerConfig| serde_json::to_value(config).unwrap_or_default();
    let (current_server, next_server) = (rendered(&current.server), rendered(&next.server));
    if let (Some(current_map), Some(next_map)) =
        (current_server.as_object(), next_server.as_object())
    {
        let mut keys: Vec<&String> = current_map.keys().chain(next_map.keys()).collect();
        keys.sort();
        keys.dedup();
        for key in keys {
            if current_map.get(key) != next_map.get(key) {
                changes.push(format!("server.{key}"));
            }
        }
    } else if current_server != next_server {
        changes.push("server".to_owned());
    }

    if current.auth.mode != next.auth.mode {
        changes.push("auth.mode".to_owned());
    }
    fn key_set(config: &AppConfig) -> std::collections::BTreeSet<&str> {
        config.auth.clients.keys().map(String::as_str).collect()
    }
    if key_set(current) != key_set(next) {
        changes.push("auth.clients".to_owned());
    } else {
        for id in key_set(current) {
            let (Some(current_spec), Some(next_spec)) =
                (current.auth.clients.get(id), next.auth.clients.get(id))
            else {
                continue;
            };
            if !client_spec_matches(current_spec, next_spec) {
                changes.push(format!("auth.clients.{id}"));
            }
        }
    }
    let token_of = |config: &AppConfig| {
        config
            .auth
            .admin
            .as_ref()
            .map(|admin| admin.token.expose_secret().to_owned())
    };
    if token_of(current) != token_of(next) {
        changes.push("auth.admin".to_owned());
    }
    let key_store_of = |config: &AppConfig| -> Option<PathBuf> {
        config
            .auth
            .admin
            .as_ref()
            .and_then(|admin| admin.key_store.clone())
    };
    if key_store_of(current) != key_store_of(next) {
        changes.push("auth.admin.key_store".to_owned());
    }
    match (&current.storage, &next.storage) {
        (Some(a), Some(b)) if a == b => {}
        (None, None) => {}
        _ => changes.push("storage".to_owned()),
    }
    if !replica_matches(&current.replica, &next.replica) {
        changes.push("replica".to_owned());
    }
    if !mcp_matches(&current.mcp, &next.mcp) {
        changes.push("mcp".to_owned());
    }
    changes
}

/// Whether two `[replica]` sections are interchangeable without a restart:
/// both absent, or both present with the same backend, key prefix and
/// resolved URL. `ReplicaConfig` carries no `PartialEq` and its URL is
/// `skip_serializing`, so the comparison is field-wise and the connection
/// string is compared by equality without ever being rendered.
fn replica_matches(current: &Option<ReplicaConfig>, next: &Option<ReplicaConfig>) -> bool {
    match (current, next) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.backend == b.backend
                && a.prefix == b.prefix
                && a.url.expose_secret() == b.url.expose_secret()
        }
        _ => false,
    }
}

/// Whether two `[mcp]` sections are interchangeable without a restart:
/// both absent, or both present with the same bind, session TTL, tokens as
/// an ordered list (rotation is a change), and upstreams pairwise on name,
/// URL and bearer token. Every token is `skip_serializing`, so a
/// rendered-JSON comparison would miss a token rotation; the comparison is
/// field-wise and never renders a secret.
fn mcp_matches(current: &Option<McpGatewayConfig>, next: &Option<McpGatewayConfig>) -> bool {
    fn exposed(token: &Option<SecretString>) -> Option<&str> {
        token.as_ref().map(SecretString::expose_secret)
    }
    let upstreams_match = |a: &[McpUpstreamConfig], b: &[McpUpstreamConfig]| -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(a, b)| {
                a.name == b.name && a.url == b.url && exposed(&a.token) == exposed(&b.token)
            })
    };
    let tokens_match = |a: &[SecretString], b: &[SecretString]| -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| a.expose_secret() == b.expose_secret())
    };
    match (current, next) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.bind == b.bind
                && a.session_ttl == b.session_ttl
                && tokens_match(&a.tokens, &b.tokens)
                && upstreams_match(&a.upstreams, &b.upstreams)
        }
        _ => false,
    }
}

/// Whether two specs for the same key id are the same key material and the
/// same policy. Verifiers compare by secret equality (never rendered); a
/// bare string and a table carrying only defaults are the same key, since
/// the defaults are what the bare spelling means.
fn client_spec_matches(current: &ClientKeySpec, next: &ClientKeySpec) -> bool {
    if current.verifier().expose_secret() != next.verifier().expose_secret() {
        return false;
    }
    let policy_is_empty = |models: &[String],
                           providers: &[String],
                           limits: &Option<KeyWindowLimits>,
                           budget: &Option<KeyBudgetConfig>| {
        models.is_empty() && providers.is_empty() && limits.is_none() && budget.is_none()
    };
    match (current, next) {
        (ClientKeySpec::Verifier(_), ClientKeySpec::Verifier(_)) => true,
        (
            ClientKeySpec::Policy {
                allow_models,
                allow_providers,
                limits,
                budget,
                ..
            },
            ClientKeySpec::Policy {
                allow_models: next_models,
                allow_providers: next_providers,
                limits: next_limits,
                budget: next_budget,
                ..
            },
        ) => {
            allow_models == next_models
                && allow_providers == next_providers
                && limits == next_limits
                && budget == next_budget
        }
        (
            ClientKeySpec::Verifier(_),
            ClientKeySpec::Policy {
                allow_models,
                allow_providers,
                limits,
                budget,
                ..
            },
        )
        | (
            ClientKeySpec::Policy {
                allow_models,
                allow_providers,
                limits,
                budget,
                ..
            },
            ClientKeySpec::Verifier(_),
        ) => policy_is_empty(allow_models, allow_providers, limits, budget),
    }
}

/// Validates one pool definition in isolation: non-empty name and members,
/// no duplicate members, a valid retry ladder. Membership and name collisions
/// with providers are checked by [`validate_registry_pools`], which sees both
/// sets.
///
/// # Errors
///
/// [`ConfigValidationError::InvalidPool`] naming the first broken rule.
pub fn validate_pool_config(pool: &PoolConfig) -> Result<(), ConfigValidationError> {
    validate_pool_config_against(pool, &[])
}

/// Validates a pool against the set of known provider names (which may be
/// empty when only structural checks are wanted).
pub(crate) fn validate_pool_config_against(
    pool: &PoolConfig,
    known_providers: &[String],
) -> Result<(), ConfigValidationError> {
    let invalid = |message: String| ConfigValidationError::InvalidPool {
        pool: pool.name.clone(),
        message,
    };
    if pool.name.trim().is_empty() {
        return Err(ConfigValidationError::EmptyProviderName);
    }
    if !pool_name_is_slug(&pool.name) {
        return Err(ConfigValidationError::InvalidProviderNameFormat {
            provider: pool.name.clone(),
        });
    }
    if pool.members.is_empty() {
        return Err(invalid(
            "pool has no members; add at least one provider name".to_owned(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for member in &pool.members {
        if member.trim().is_empty() {
            return Err(invalid(
                "member name is empty; members must name configured providers".to_owned(),
            ));
        }
        if !seen.insert(member.as_str()) {
            return Err(invalid(format!(
                "member \"{member}\" is listed more than once"
            )));
        }
        if !known_providers.is_empty() && !known_providers.iter().any(|p| p == member) {
            return Err(invalid(format!(
                "member \"{member}\" is not a configured provider"
            )));
        }
    }
    let retry = &pool.retry;
    if retry.max_attempts == 0 {
        return Err(invalid("retry.max_attempts must be at least 1".to_owned()));
    }
    Ok(())
}

/// A pool name is valid in the provider URL namespace under the same slug
/// rule as provider names.
fn pool_name_is_slug(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_'))
}

fn validate_admin_mount(cfg: &AppConfig) -> Result<(), CoreError> {
    let Some(admin) = cfg.auth.admin.as_ref() else {
        if cfg.server.admin_bind.is_some() {
            return Err(CoreError::ConfigValidation {
                message: "server.admin_bind is configured but [auth.admin] is not; \
                          set [auth.admin] token or remove server.admin_bind"
                    .to_owned(),
                source: None,
            });
        }
        if cfg.server.admin_on_main_listener {
            return Err(CoreError::ConfigValidation {
                message: "server.admin_on_main_listener = true but [auth.admin] is not \
                          configured; there would be no admin token to guard /usage"
                    .to_owned(),
                source: None,
            });
        }
        return Ok(());
    };
    if admin.token.expose_secret().is_empty() {
        return Err(CoreError::ConfigValidation {
            message: "[auth.admin] token must not be empty".to_owned(),
            source: None,
        });
    }
    if cfg.storage.is_none() {
        return Err(CoreError::ConfigValidation {
            message: "[auth.admin] requires the [storage] usage ledger; configure \
                      [storage] or remove [auth.admin]"
                .to_owned(),
            source: None,
        });
    }
    if let Some(admin_bind) = cfg.server.admin_bind {
        if admin_bind == cfg.server.bind {
            return Err(CoreError::ConfigValidation {
                message: format!(
                    "server.admin_bind {admin_bind} must differ from server.bind; \
                     the admin listener needs its own socket"
                ),
                source: None,
            });
        }
    } else if !cfg.server.admin_on_main_listener && cfg.server.bind.port() == u16::MAX {
        return Err(CoreError::ConfigValidation {
            message: "server.bind port 65535 leaves no distinct port for the default \
                      admin listener; set server.admin_bind explicitly or enable \
                      server.admin_on_main_listener"
                .to_owned(),
            source: None,
        });
    }
    Ok(())
}

/// Reserved caller name used for unauthenticated requests under
/// [`AuthMode::None`]; a client key id must not collide with it or event
/// attribution becomes ambiguous.
pub const ANONYMOUS_CALLER: &str = "anonymous";

/// Refuses `passthrough_auth` providers combined with client-key auth.
///
/// With client keys on, the inbound token that `passthrough_auth` forwards
/// upstream is the proxy's own client key — the very secret that
/// authenticated the caller — so the provider would receive it. A token that
/// authenticated the caller must never be sent upstream; until a separate
/// upstream-key header exists, this combination is a configuration error.
pub fn validate_passthrough_auth_combination(
    auth: &AuthConfig,
    providers: &[ProviderConfig],
) -> Result<(), CoreError> {
    if auth.mode != AuthMode::ClientKeys {
        return Ok(());
    }
    let passthrough: Vec<&str> = providers
        .iter()
        .filter(|p| p.passthrough_auth)
        .map(|p| p.name.as_str())
        .collect();
    if passthrough.is_empty() {
        return Ok(());
    }
    Err(CoreError::ConfigValidation {
        message: format!(
            "auth.mode = \"client-keys\" cannot be combined with \
             passthrough_auth providers ({}): the forwarded inbound token would \
             be the proxy's own client key, leaking it to the provider. Remove \
             [auth.clients] / set auth.mode = \"none\", or configure an api_key \
             for these providers",
            passthrough.join(", ")
        ),
        source: None,
    })
}

fn validate_auth_config(auth: &AuthConfig, bind: SocketAddr) -> Result<(), CoreError> {
    let clients_empty = auth.clients.is_empty();
    // A managed key store ([auth.admin].key_store, plan 09 step 2) can be
    // a deployment's only source of client keys: the daemon merges its
    // live keys into the verifiers at startup, so client-keys mode with
    // an empty [auth.clients] section is valid when one is configured.
    let has_managed_key_store = |auth: &AuthConfig| {
        auth.admin
            .as_ref()
            .is_some_and(|admin| admin.key_store.is_some())
    };
    match auth.mode {
        AuthMode::None if !clients_empty => {
            return Err(CoreError::ConfigValidation {
                message: "auth.mode = \"none\" but [auth.clients] is non-empty; \
                          set auth.mode = \"client-keys\" to enforce the configured keys"
                    .to_owned(),
                source: None,
            });
        }
        AuthMode::ClientKeys if clients_empty && !has_managed_key_store(auth) => {
            return Err(CoreError::ConfigValidation {
                message: "auth.mode = \"client-keys\" requires at least one key under \
                          [auth.clients] or a managed key store \
                          ([auth.admin].key_store)"
                    .to_owned(),
                source: None,
            });
        }
        AuthMode::None if !bind.ip().is_loopback() => {
            return Err(CoreError::ConfigValidation {
                message: format!(
                    "server.bind {bind} is not loopback and no client keys are \
                     configured; anonymous auth is only permitted on loopback binds. \
                     Configure [auth.clients] and set auth.mode = \"client-keys\""
                ),
                source: None,
            });
        }
        AuthMode::None | AuthMode::ClientKeys => {}
    }
    for key_id in auth.clients.keys() {
        if key_id.is_empty() {
            return Err(CoreError::ConfigValidation {
                message: "[auth.clients] contains an empty key id".to_owned(),
                source: None,
            });
        }
        if key_id == ANONYMOUS_CALLER {
            return Err(CoreError::ConfigValidation {
                message: format!(
                    "[auth.clients] key id \"{ANONYMOUS_CALLER}\" is reserved for \
                     unattributed requests; choose a different key id"
                ),
                source: None,
            });
        }
    }
    Ok(())
}

/// Load and parse a provider [`ProviderFile`] TOML file.
///
/// The raw file content is first processed for `${ENV_VAR}` interpolation,
/// then parsed as TOML, then validated.
///
/// `known_protocols` is forwarded to [`validate_provider_config`].
///
/// # Blocking I/O
///
/// This function performs synchronous filesystem I/O and is intended for use
/// at startup. If hot-reload is implemented in future, an async variant will
/// be needed.
///
/// # Trust boundary
///
/// The `path` parameter accepts arbitrary filesystem paths without
/// canonicalisation or path-traversal checks. This is standard for
/// env-var/CLI-driven config loading -- the process owner controls the path.
/// The config file content is treated as trusted input.
///
/// # Errors
///
/// Returns a [`CoreError`] if the file cannot be read, the TOML is malformed,
/// an env var referenced in the file resolves to an empty value, or validation
/// of the parsed [`ProviderConfig`] fails.
pub fn load_provider_config(
    path: impl AsRef<Path>,
    known_protocols: Option<&[&str]>,
) -> Result<ProviderConfig, CoreError> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path).map_err(|source| CoreError::ConfigLoad {
        path: path.to_path_buf(),
        source,
    })?;

    for (var_name, resolved) in find_env_var_refs(&raw) {
        match resolved {
            None => {}
            Some(value) if value.is_empty() => {
                return Err(ConfigValidationError::EmptyEnvVar {
                    provider: format!("(file: {})", path.display()),
                    var: var_name,
                }
                .into());
            }
            Some(_) => {}
        }
    }

    let interpolated = interpolate_env_vars(&raw);
    if let Some(var) = find_unresolved_env_var(&interpolated) {
        return Err(CoreError::ConfigValidation {
            message: format!(
                "provider config contains unresolvable environment variable \"{var}\""
            ),
            source: None,
        });
    }
    let file: ProviderFile = toml::from_str(&interpolated).map_err(CoreError::ConfigParse)?;
    validate_provider_config(&file.provider, known_protocols).map_err(CoreError::from)?;
    Ok(file.provider)
}

#[cfg(test)]
mod tests {

    #[test]
    fn each_route_kind_is_listed_once_under_its_toml_key() {
        let mut seen = std::collections::HashSet::new();
        for kind in ProviderRouteKind::ALL {
            assert!(seen.insert(kind), "{kind:?} listed twice");
            // The key is the one [provider.routes] actually parses
            // (unknown keys are refused), and it reaches this kind.
            let routes: ProviderRoutesConfig =
                toml::from_str(&format!("{} = \"adapter\"", kind.config_key()))
                    .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            assert_eq!(routes.get(kind), Some("adapter"), "{kind:?}");
            let others = ProviderRouteKind::ALL
                .iter()
                .filter(|other| routes.get(**other).is_some())
                .count();
            assert_eq!(others, 1, "{kind:?}'s key sets only {kind:?}");
        }
    }

    #[test]
    fn validate_rejects_route_pointing_at_missing_adapter() {
        for kind in ProviderRouteKind::ALL {
            let provider: ProviderFile = toml::from_str(&format!(
                r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
{} = "ghost"
"#,
                kind.config_key()
            ))
            .expect("provider config");
            let err = validate_provider_config(&provider.provider, None)
                .expect_err("a route naming an unknown adapter must fail at config load");
            assert_eq!(
                err,
                ConfigValidationError::UnknownRouteAdapter {
                    provider: "example".to_owned(),
                    route_kind: kind.config_key().to_owned(),
                    adapter: "ghost".to_owned(),
                },
                "{kind:?} must reject a dangling adapter reference"
            );
        }
    }

    #[test]
    fn validate_rejects_empty_route_adapter_name() {
        for kind in ProviderRouteKind::ALL {
            for value in ["", "  "] {
                let provider: ProviderFile = toml::from_str(&format!(
                    r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
{} = "{}"
"#,
                    kind.config_key(),
                    value
                ))
                .expect("provider config");
                let err = validate_provider_config(&provider.provider, None)
                    .expect_err("an empty route adapter name must fail at config load");
                assert_eq!(
                    err,
                    ConfigValidationError::EmptyRouteAdapterName {
                        provider: "example".to_owned(),
                        route_kind: kind.config_key().to_owned(),
                    },
                    "{kind:?} must reject the empty route adapter name {value:?}"
                );
            }
        }
    }

    #[test]
    fn validate_rejects_route_adapter_name_with_surrounding_whitespace() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
chat_completions = " chat "
"#,
        )
        .expect("provider config");
        let err = validate_provider_config(&provider.provider, None).expect_err(
            "route lookup uses the untrimmed adapter name, so a padded name must fail at load",
        );
        assert_eq!(
            err,
            ConfigValidationError::UnknownRouteAdapter {
                provider: "example".to_owned(),
                route_kind: "chat_completions".to_owned(),
                adapter: " chat ".to_owned(),
            }
        );
    }

    #[test]
    fn passthrough_providers_with_client_keys_are_refused() {
        let auth = AuthConfig {
            mode: AuthMode::ClientKeys,
            clients: HashMap::from([(
                "laptop".to_owned(),
                ClientKeySpec::Verifier(secrecy::SecretString::from("argon2id$hash")),
            )]),
            admin: None,
            legacy_bare_tokens: false,
        };
        let mut provider = make_minimal_provider("passthrough-provider");
        provider.passthrough_auth = true;

        let err = validate_passthrough_auth_combination(&auth, &[provider])
            .expect_err("passthrough + client keys must be refused");
        let message = err.to_string();
        assert!(
            message.contains("passthrough-provider"),
            "the error must name the provider: {message}"
        );
        assert!(
            message.contains("client key"),
            "the error must say why: {message}"
        );
    }

    #[test]
    fn passthrough_providers_with_anonymous_auth_stay_allowed() {
        let auth = AuthConfig {
            mode: AuthMode::None,
            clients: HashMap::new(),
            admin: None,
            legacy_bare_tokens: false,
        };
        let mut provider = make_minimal_provider("passthrough-provider");
        provider.passthrough_auth = true;

        validate_passthrough_auth_combination(&auth, &[provider])
            .expect("passthrough with anonymous auth is the intended use");
    }

    #[test]
    fn keyed_providers_with_client_keys_stay_allowed() {
        let auth = AuthConfig {
            mode: AuthMode::ClientKeys,
            clients: HashMap::from([(
                "laptop".to_owned(),
                ClientKeySpec::Verifier(secrecy::SecretString::from("argon2id$hash")),
            )]),
            admin: None,
            legacy_bare_tokens: false,
        };
        let provider = make_minimal_provider("keyed-provider");

        validate_passthrough_auth_combination(&auth, &[provider])
            .expect("provider api keys with client keys are fine");
    }

    #[test]
    fn client_key_specs_parse_as_bare_verifier_or_policy_table() {
        let raw = r#"
mode = "client-keys"

[clients]
bare = "$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA"

[clients.scoped]
verifier = "$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA"
allow_models = ["claude-*"]
allow_providers = ["anthropic"]

[clients.scoped.limits]
rpm = 60
tpm = 200000

[clients.scoped.budget]
usd_per_day = "10.50"
unestimatable = { reserve_usd = { usd = "0.25" } }
"#;
        let auth: AuthConfig = toml::from_str(raw).expect("auth parses");
        let bare = auth.clients.get("bare").expect("bare key");
        assert!(matches!(bare, ClientKeySpec::Verifier(_)));
        assert!(bare.verifier().expose_secret().starts_with("$argon2id$"));

        let ClientKeySpec::Policy {
            verifier,
            allow_models,
            allow_providers,
            limits,
            budget,
        } = auth.clients.get("scoped").expect("scoped key")
        else {
            panic!("scoped key must parse as a policy table");
        };
        assert!(verifier.expose_secret().starts_with("$argon2id$"));
        assert_eq!(allow_models, &["claude-*".to_owned()]);
        assert_eq!(allow_providers, &["anthropic".to_owned()]);
        let limits = limits.as_ref().expect("limits");
        assert_eq!(
            (limits.rpm, limits.tpm, limits.rpd),
            (Some(60), Some(200_000), None)
        );
        let budget = budget.as_ref().expect("budget");
        assert_eq!(budget.usd_per_day.to_string(), "10.50");
        assert_eq!(
            budget.unestimatable,
            UnestimatableCostPolicy::ReserveUsd {
                usd: "0.25".parse().expect("decimal")
            }
        );
        assert_eq!(budget.assumed_max_output_tokens, 4096);
    }

    #[test]
    fn client_key_budget_defaults_to_rejecting_unestimatable_cost() {
        let raw = r#"
[clients.scoped]
verifier = "$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA"
[clients.scoped.budget]
usd_per_day = "1"
"#;
        let auth: AuthConfig = toml::from_str(raw).expect("auth parses");
        let ClientKeySpec::Policy { budget, .. } = auth.clients.get("scoped").expect("key") else {
            panic!("policy table expected");
        };
        assert_eq!(
            budget.as_ref().expect("budget").unestimatable,
            UnestimatableCostPolicy::Reject,
            "a key with a budget must fail closed on unpriced models unless told otherwise"
        );
    }

    #[test]
    fn concurrency_caps_of_zero_are_rejected_at_validation() {
        let mut provider = make_minimal_provider("capped");
        provider.concurrency = Some(ProviderConcurrencyConfig {
            total: Some(0),
            models: HashMap::new(),
        });
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::InvalidConcurrency { .. })
        ));

        provider.concurrency = Some(ProviderConcurrencyConfig {
            total: Some(4),
            models: HashMap::from([(ModelId::new("gpt-4o"), 0)]),
        });
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::InvalidConcurrency { .. })
        ));

        provider.concurrency = Some(ProviderConcurrencyConfig {
            total: Some(4),
            models: HashMap::from([(ModelId::new("gpt-4o"), 2)]),
        });
        validate_provider_config(&provider, None).expect("positive caps are valid");
    }

    #[test]
    fn preflight_defaults_to_warn_with_a_margin() {
        let raw = r#"
bind = "127.0.0.1:3456"
request_timeout = "30s"
log_level = "info"
hot_reload = false
server_name = "t"
"#;
        let server: ServerConfig = toml::from_str(raw).expect("server parses");
        assert_eq!(server.preflight.mode, PreflightMode::Warn);
        assert!(server.preflight.margin_tokens > 0);

        let raw = r#"
bind = "127.0.0.1:3456"
request_timeout = "30s"
log_level = "info"
hot_reload = false
server_name = "t"
[preflight]
mode = "enforce"
margin_tokens = 1024
"#;
        let server: ServerConfig = toml::from_str(raw).expect("server parses");
        assert_eq!(server.preflight.mode, PreflightMode::Enforce);
        assert_eq!(server.preflight.margin_tokens, 1024);
    }

    fn make_minimal_provider(name: &str) -> ProviderConfig {
        let mut provider = toml::from_str::<ProviderFile>(
            r#"[provider]
name = "placeholder"
api_key = "k"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
chat_completions = "chat"
"#,
        )
        .expect("minimal provider parses")
        .provider;
        provider.name = name.to_owned();
        provider
    }

    #[test]
    fn cleartext_endpoints_to_off_machine_hosts_are_refused() {
        let mut provider = make_minimal_provider("leaky");
        let Some(adapter) = provider.adapters.get_mut("chat") else {
            panic!("minimal provider has a chat adapter");
        };
        adapter.endpoint = "http://api.example.com/v1/chat/completions".to_owned();
        let error = validate_provider_config(&provider, None)
            .expect_err("a cleartext endpoint to a remote host must be refused");
        assert!(
            matches!(error, ConfigValidationError::CleartextCredentials { .. }),
            "got: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("leaky") && message.contains("api.example.com"),
            "the error must name the provider and the host: {message}"
        );
        assert!(
            message.contains("https"),
            "the error must name the fix: {message}"
        );
    }

    /// A cleartext or invalid-scheme error is exactly when an endpoint is
    /// most likely to carry credentials — userinfo in the authority or a
    /// Gemini-style `?key=` in the query — so the Display string must keep
    /// the host but never echo the secrets (fix-brief 07-14).
    #[test]
    fn cleartext_error_does_not_echo_userinfo_or_query() {
        let echo_adapter_endpoint = |endpoint: &str| {
            let mut provider = make_minimal_provider("echo");
            let Some(adapter) = provider.adapters.get_mut("chat") else {
                panic!("minimal provider has a chat adapter");
            };
            adapter.endpoint = endpoint.to_owned();
            validate_provider_config(&provider, None)
                .expect_err("a credential-bearing endpoint must be refused")
        };
        let assert_no_secrets = |message: &str, secrets: &[&str]| {
            for secret in secrets {
                assert!(
                    !message.contains(secret),
                    "the error must not echo {secret:?}: {message}"
                );
            }
        };

        let error =
            echo_adapter_endpoint("http://alice:s3cret@api.example.com/v1/chat?key=TOPSECRET");
        assert!(
            matches!(error, ConfigValidationError::CleartextCredentials { .. }),
            "got: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("api.example.com"),
            "the host must stay in the error so the operator can find the adapter: {message}"
        );
        assert_no_secrets(&message, &["alice", "s3cret", "TOPSECRET", "?key="]);

        let error = echo_adapter_endpoint("ftp://bob:pw@h/x?k=v");
        assert!(
            matches!(error, ConfigValidationError::InvalidEndpointScheme { .. }),
            "got: {error:?}"
        );
        let message = error.to_string();
        assert!(
            message.contains("invalid endpoint scheme"),
            "the invalid-scheme error must fire: {message}"
        );
        assert_no_secrets(&message, &["bob", "pw", "k=v", "?k="]);
    }

    #[test]
    fn cleartext_endpoints_to_machine_local_hosts_are_exempt() {
        for endpoint in [
            "http://127.0.0.1:8080/v1/chat/completions",
            "http://127.8.8.8:8080/v1/chat/completions",
            "http://localhost:8080/v1/chat/completions",
            "http://[::1]:8080/v1/chat/completions",
            // A LAN address is off-machine even though it is "internal".
            // Covered by the refusal test; https to the same host is fine.
            "https://192.168.1.4/v1/chat/completions",
        ] {
            let mut provider = make_minimal_provider("local");
            let Some(adapter) = provider.adapters.get_mut("chat") else {
                panic!("minimal provider has a chat adapter");
            };
            adapter.endpoint = endpoint.to_owned();
            validate_provider_config(&provider, None)
                .unwrap_or_else(|error| panic!("{endpoint} must validate: {error}"));
        }
    }

    #[test]
    fn cleartext_discovery_endpoints_to_remote_hosts_are_refused() {
        let mut provider = make_minimal_provider("leaky-discovery");
        provider.discovery = Some(
            toml::from_str(
                r#"kind = "openai_compatible_models"
            endpoint = "http://models.example.com/v1/models"
            "#,
            )
            .expect("discovery parses"),
        );
        let error = validate_provider_config(&provider, None)
            .expect_err("a cleartext discovery endpoint to a remote host must be refused");
        assert!(
            error.to_string().contains("discovery"),
            "the error must name where the cleartext endpoint lives: {error}"
        );
    }

    #[test]
    fn a_local_tunnel_hostname_is_not_machine_local_by_spelling() {
        // A name that resolves to loopback (SSH tunnels, split-horizon DNS)
        // is still refused: validation runs without DNS, and a resolvable
        // name is re-pointable. Operators tunneling must spell the address.
        let mut provider = make_minimal_provider("tunneled");
        let Some(adapter) = provider.adapters.get_mut("chat") else {
            panic!("minimal provider has a chat adapter");
        };
        adapter.endpoint = "http://box.internal:8080/v1/chat/completions".to_owned();
        assert!(
            validate_provider_config(&provider, None).is_err(),
            "a hostname is off-machine until it is localhost or an IP literal"
        );
    }

    #[test]
    fn prompt_cache_strip_pointers_are_validated_at_load() {
        let mut provider = make_minimal_provider("hygienic");
        provider.prompt_cache = Some(PromptCacheConfig {
            strip_volatile: vec!["/metadata/user_id".to_owned(), "/user".to_owned()],
            inject_markers: true,
        });
        validate_provider_config(&provider, None).expect("well-formed pointers validate");

        for malformed in [
            "metadata/user_id",
            "/metadata//user_id",
            "/metadata/",
            "",
            // RFC 6901: a `~` in a reference token must be followed by `0`
            // or `1`; anything else would survive the split as a literal.
            "/a~2",
            "/a~",
            "/~",
            "/a~x/b",
            "/a/b~",
        ] {
            let mut provider = make_minimal_provider("hygienic");
            provider.prompt_cache = Some(PromptCacheConfig {
                strip_volatile: vec![malformed.to_owned()],
                inject_markers: false,
            });
            let error = validate_provider_config(&provider, None)
                .expect_err("a malformed strip pointer must be refused at load");
            assert!(
                error.to_string().contains("RFC 6901"),
                "the error must say what a pointer is: {error}"
            );
        }

        let mut provider = make_minimal_provider("hygienic");
        provider.prompt_cache = Some(PromptCacheConfig {
            strip_volatile: vec!["/a~0b".to_owned(), "/a~1b/c".to_owned()],
            inject_markers: false,
        });
        assert!(
            validate_provider_config(&provider, None).is_ok(),
            "escaped `~0` and `~1` are well-formed RFC 6901 pointers"
        );
    }

    #[test]
    fn metrics_config_validates_the_otlp_endpoint_and_interval() {
        let good = toml::from_str::<MetricsConfig>(
            "[otlp]\nendpoint = \"http://127.0.0.1:4318/v1/metrics\"\ninterval = \"30s\"\n",
        )
        .expect("metrics config parses");
        validate_metrics_config(&good).expect("a full http endpoint and positive interval pass");

        let default_interval = toml::from_str::<MetricsConfig>(
            "[otlp]\nendpoint = \"https://collector.internal/v1/metrics\"\n",
        )
        .expect("metrics config parses");
        validate_metrics_config(&default_interval)
            .expect("the interval defaults without being spelled out");

        for bad in [
            "[otlp]\nendpoint = \"\"\n",
            "[otlp]\nendpoint = \"grpc://collector:4317\"\n",
            "[otlp]\nendpoint = \"http://127.0.0.1:4318/v1/metrics\"\ninterval = \"0s\"\n",
        ] {
            let config = toml::from_str::<MetricsConfig>(bad).expect("metrics config parses");
            assert!(
                validate_metrics_config(&config).is_err(),
                "this metrics config must be refused: {bad}"
            );
        }

        // Off by default: an absent section is valid.
        validate_metrics_config(&MetricsConfig::default()).expect("default is off and fine");
    }

    #[test]
    fn log_format_precedence_env_wins_then_config_then_default() {
        use LogFormat::{Json, Plain};
        // BUG-25: the config value is honoured when the env var is unset.
        assert_eq!(LogFormat::effective(None, Json), Json);
        assert_eq!(LogFormat::effective(Some(""), Json), Json, "empty is unset");
        assert_eq!(LogFormat::effective(Some("   "), Plain), Plain);
        // A set env var wins over whatever the config says.
        assert_eq!(LogFormat::effective(Some("json"), Plain), Json);
        assert_eq!(LogFormat::effective(Some("JSON"), Plain), Json);
        assert_eq!(LogFormat::effective(Some("plain"), Json), Plain);
        // Set but unrecognized: the env still decides, as Plain — matching
        // from_env, so a typo never silently resurrects the config value.
        assert_eq!(LogFormat::effective(Some("yaml"), Json), Plain);
    }

    #[test]
    fn upstream_timeouts_are_cross_validated() {
        validate_upstream_timeouts(&UpstreamTimeoutsConfig::default())
            .expect("defaults are internally consistent");

        let connect_gt_first_byte = UpstreamTimeoutsConfig {
            connect: Duration::from_secs(30),
            first_byte: Duration::from_secs(5),
            ..UpstreamTimeoutsConfig::default()
        };
        let err = validate_upstream_timeouts(&connect_gt_first_byte).expect_err("connect > first");
        assert!(
            err.to_string().contains("connect"),
            "the error names the offending phase: {err}"
        );

        let total_below_first_byte = UpstreamTimeoutsConfig {
            total: Some(Duration::from_secs(10)),
            first_byte: Duration::from_secs(60),
            ..UpstreamTimeoutsConfig::default()
        };
        let err = validate_upstream_timeouts(&total_below_first_byte).expect_err("total < first");
        assert!(
            err.to_string().contains("total"),
            "the error names the total: {err}"
        );

        let zero_idle = UpstreamTimeoutsConfig {
            idle: Duration::ZERO,
            ..UpstreamTimeoutsConfig::default()
        };
        assert!(validate_upstream_timeouts(&zero_idle).is_err());
    }

    #[test]
    fn upstream_timeouts_parse_from_toml_and_validate_at_load() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg_path = dir.path().join("config.toml");
        std::fs::write(
            &cfg_path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"

[server.upstream_timeouts]
connect = "5s"
first_byte = "15s"
idle = "30s"
total = "120s"
"#,
        )
        .expect("write config");

        let cfg = load_app_config(&cfg_path).expect("config loads");
        assert_eq!(cfg.server.upstream_timeouts.connect, Duration::from_secs(5));
        assert_eq!(
            cfg.server.upstream_timeouts.total,
            Some(Duration::from_secs(120))
        );
    }

    #[test]
    fn invalid_upstream_timeouts_are_rejected_at_config_load() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg_path = dir.path().join("config.toml");
        std::fs::write(
            &cfg_path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"

[server.upstream_timeouts]
connect = "50s"
first_byte = "5s"
"#,
        )
        .expect("write config");

        let err = load_app_config(&cfg_path).expect_err("cross-validation fires at load");
        assert!(
            err.to_string().contains("first_byte"),
            "the load error names the broken phase rule: {err}"
        );
    }

    #[test]
    fn bare_v1_defaults_off_and_empty_enabled_config_is_rejected() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg_path = dir.path().join("config.toml");
        std::fs::write(
            &cfg_path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"

[server.bare_v1]
enabled = true
"#,
        )
        .expect("write config");

        let err = load_app_config(&cfg_path).expect_err("enabled bare_v1 needs a target");
        assert!(
            err.to_string().contains("bare_v1"),
            "the load error names bare_v1: {err}"
        );
    }

    #[test]
    fn bare_v1_targets_naming_nothing_loadable_are_reported() {
        let bare = BareV1Config {
            enabled: true,
            models: std::collections::HashMap::from([
                ("claude-sonnet-4-6".to_owned(), "real".to_owned()),
                ("gpt-4o".to_owned(), "ghost".to_owned()),
            ]),
            default_target: Some("also-ghost".to_owned()),
        };
        let problems = bare_v1_target_problems(&bare, |name| name == "real");
        assert_eq!(problems.len(), 2, "one problem per unknown target");
        assert!(
            problems[0].contains("gpt-4o") && problems[0].contains("ghost"),
            "the problem names the model and its target: {}",
            problems[0]
        );
        assert!(
            problems[1].contains("also-ghost"),
            "the default target is checked too: {}",
            problems[1]
        );
    }

    #[test]
    fn bare_v1_targets_that_name_loadable_routes_have_no_problems() {
        let bare = BareV1Config {
            enabled: true,
            models: std::collections::HashMap::from([("m".to_owned(), "pool-a".to_owned())]),
            default_target: Some("provider-b".to_owned()),
        };
        assert!(
            bare_v1_target_problems(&bare, |name| { name == "pool-a" || name == "provider-b" })
                .is_empty()
        );
    }

    #[test]
    fn disabled_bare_v1_targets_are_not_checked() {
        // Off by default: even targets that name nothing are not problems,
        // because the routes they would serve are not mounted.
        let bare = BareV1Config {
            enabled: false,
            models: std::collections::HashMap::from([("m".to_owned(), "ghost".to_owned())]),
            default_target: None,
        };
        assert!(bare_v1_target_problems(&bare, |_| false).is_empty());
    }

    #[test]
    fn provider_retry_and_api_keys_parse_from_toml() {
        let dir = tempfile::tempdir().expect("temp dir");
        let provider_path = dir.path().join("p.toml");
        std::fs::write(
            &provider_path,
            r#"[provider]
name = "retried"
api_key = "k1"
api_keys = ["k2", "k3"]
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
chat_completions = "chat"

[provider.retry]
max_attempts = 3
backoff = "100ms"
cooldown = "45s"
total_timeout = "30s"
hold = "1s"
"#,
        )
        .expect("write provider");

        let provider = load_provider_config(&provider_path, None).expect("provider parses");
        let retry = provider.retry.expect("retry present");
        assert_eq!(retry.max_attempts, 3);
        assert_eq!(retry.backoff, Duration::from_millis(100));
        assert_eq!(retry.total_timeout, Duration::from_secs(30));
        assert_eq!(retry.hold, Duration::from_secs(1));
        assert_eq!(provider.api_keys.expect("key pool present").len(), 2);
    }

    use super::*;

    #[test]
    fn forwarded_inbound_allowlist_rejects_client_auth_headers() {
        let mut provider = make_minimal_provider("fwd-provider");
        let adapter = provider.adapters.get_mut("chat").expect("chat adapter");
        adapter.forward_inbound_headers = Some(vec![
            "anthropic-beta".to_owned(),
            "authorization".to_owned(),
        ]);
        let err = validate_provider_config(&provider, None)
            .expect_err("authorization must never be forwarded");
        assert!(
            matches!(&err, ConfigValidationError::ForbiddenForwardedHeader { header, .. } if header == "authorization"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn forwarded_inbound_allowlist_rejects_browser_and_origin_headers() {
        for header in [
            "origin",
            "user-agent",
            "cookie",
            "sec-fetch-mode",
            "x-request-id",
            "browser-use",
            "Sec-CH-UA",
        ] {
            let mut provider = make_minimal_provider("fwd-provider");
            let adapter = provider.adapters.get_mut("chat").expect("chat adapter");
            adapter.forward_inbound_headers = Some(vec![header.to_owned()]);
            let err = validate_provider_config(&provider, None)
                .expect_err("browser/origin headers must never be forwarded");
            assert!(
                matches!(err, ConfigValidationError::ForbiddenForwardedHeader { .. }),
                "header {header} must be rejected, got {err:?}"
            );
        }
    }

    #[test]
    fn forwarded_inbound_allowlist_accepts_beta_and_session_headers() {
        let mut provider = make_minimal_provider("fwd-provider");
        let adapter = provider.adapters.get_mut("chat").expect("chat adapter");
        adapter.forward_inbound_headers = Some(vec![
            "anthropic-beta".to_owned(),
            "x-claude-code-session-id".to_owned(),
        ]);
        validate_provider_config(&provider, None)
            .expect("beta and session headers are forwardable");
    }

    #[test]
    fn anthropic_default_allowlist_carries_beta_and_session_headers() {
        assert_eq!(
            default_forwarded_inbound_headers("anthropic_messages"),
            &["anthropic-beta", "x-claude-code-session-id"]
        );
        assert!(default_forwarded_inbound_headers("openai_chat_completions").is_empty());
        assert!(default_forwarded_inbound_headers("gemini_generate_content").is_empty());
        assert!(default_forwarded_inbound_headers("openai_responses").is_empty());
    }

    #[test]
    fn lossy_flag_parses_from_toml() {
        let file = toml::from_str::<ProviderFile>(
            r#"[provider]
name = "lossy-provider"
api_key = "k"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"
lossy = true
forward_inbound_headers = ["x-app"]

[provider.routes]
chat_completions = "chat"
"#,
        )
        .expect("lossy provider parses");
        let adapter = file.provider.adapters.get("chat").expect("chat adapter");
        assert!(adapter.lossy);
        assert_eq!(
            adapter.forward_inbound_headers.as_deref(),
            Some(&["x-app".to_owned()][..])
        );
    }

    #[test]
    fn app_config_rejects_removed_models_table() {
        let result = toml::from_str::<AppConfig>(
            "[server]\nbind = '127.0.0.1:3456'\n[models]\nfoo = { provider = 'bar' }\n",
        );
        result.unwrap_err();
    }

    #[test]
    fn app_config_defaults_operational_controls() {
        let config: AppConfig = toml::from_str(
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .expect("app config");

        assert_eq!(config.server.rate_limit_rpm, 100);
        assert!(!config.server.trust_forwarded_headers);
        assert_eq!(config.server.shutdown_timeout, Duration::from_secs(30));
    }

    #[test]
    fn app_config_accepts_explicit_shutdown_timeout() {
        let config: AppConfig = toml::from_str(
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
shutdown_timeout = "10s"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .expect("app config");
        assert_eq!(config.server.shutdown_timeout, Duration::from_secs(10));
    }

    fn base_config() -> AppConfig {
        toml::from_str::<AppConfig>(
            r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "t"

[auth]
mode = "client-keys"
[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA"
"#,
        )
        .expect("base config parses")
    }

    #[test]
    fn restart_required_changes_names_the_exact_server_field() {
        let current = base_config();
        let mut next = current.clone();
        next.server.request_timeout = Duration::from_secs(120);
        let changes = restart_required_changes(&current, &next);
        assert_eq!(changes, vec!["server.request_timeout".to_owned()]);
    }

    #[test]
    fn restart_required_changes_covers_auth_and_storage() {
        let current = base_config();
        // Same key ids, different mode.
        let mut next = current.clone();
        next.auth.mode = AuthMode::None;
        assert!(restart_required_changes(&current, &next).contains(&"auth.mode".to_owned()));
        // A new key id.
        let mut next = current.clone();
        next.auth.clients.insert(
            "phone".to_owned(),
            ClientKeySpec::Verifier(secrecy::SecretString::from("x")),
        );
        assert!(
            restart_required_changes(&current, &next).contains(&"auth.clients".to_owned()),
            "key ids are restart-required: verifiers feed the auth middleware"
        );
        // Admin token.
        let mut next = current.clone();
        next.auth.admin = Some(AdminAuthConfig {
            token: secrecy::SecretString::from("token"),
            key_store: None,
        });
        assert!(restart_required_changes(&current, &next).contains(&"auth.admin".to_owned()));
        // Storage ledger.
        let mut next = current.clone();
        next.storage = Some(StorageConfig {
            path: "usage.db".into(),
            max_age: None,
            max_rows: None,
            prune_interval: Duration::from_secs(3600),
        });
        assert!(restart_required_changes(&current, &next).contains(&"storage".to_owned()));
    }

    /// The managed key store, the `[replica]` store and the `[mcp]` gateway
    /// are all wired once at startup, so any edit of them is
    /// restart-required — including edits a rendered-JSON comparison cannot
    /// see, because `url` and every token are `skip_serializing`. A refusal
    /// names fields and never renders a secret.
    #[test]
    fn restart_required_changes_covers_key_store_replica_and_mcp() {
        // A key_store added under an unchanged admin token is its own field,
        // not blamed on the token.
        let mut current = base_config();
        current.auth.admin = Some(AdminAuthConfig {
            token: SecretString::from("admin-tok"),
            key_store: None,
        });
        let mut next = current.clone();
        next.auth.admin = Some(AdminAuthConfig {
            token: SecretString::from("admin-tok"),
            key_store: Some("keys.db".into()),
        });
        let key_store_change = restart_required_changes(&current, &next);
        assert!(
            key_store_change.contains(&"auth.admin.key_store".to_owned()),
            "an added key_store is restart-required: {key_store_change:?}"
        );
        assert!(
            !key_store_change.contains(&"auth.admin".to_owned()),
            "an unchanged token must not be blamed: {key_store_change:?}"
        );

        // [replica]: added, then URL-rotated between two Some sections, then
        // merely re-stated.
        let mut current = base_config();
        current.replica = Some(ReplicaConfig {
            backend: ReplicaBackend::Redis,
            url: SecretString::from("redis://a"),
            prefix: "p".to_owned(),
        });
        let mut next = current.clone();
        next.replica = None;
        assert_eq!(
            restart_required_changes(&current, &next),
            vec!["replica".to_owned()]
        );
        let mut next = current.clone();
        next.replica = Some(ReplicaConfig {
            backend: ReplicaBackend::Redis,
            url: SecretString::from("redis://b"),
            prefix: "p".to_owned(),
        });
        let replica_change = restart_required_changes(&current, &next);
        assert_eq!(
            replica_change,
            vec!["replica".to_owned()],
            "a rotated replica URL is restart-required even though it never renders"
        );
        assert!(
            restart_required_changes(&current, &current).is_empty(),
            "a merely re-stated [replica] section still reloads"
        );

        // [mcp]: added, then token-rotated between two Some sections (the
        // case a JSON compare would miss), then merely re-stated.
        let mcp = |token: &str| McpGatewayConfig {
            bind: "127.0.0.1:9100".parse().expect("bind parses"),
            tokens: vec![SecretString::from(token)],
            upstreams: Vec::new(),
            session_ttl: Duration::from_secs(1800),
        };
        let mut current = base_config();
        current.mcp = Some(mcp("t1"));
        let mut next = current.clone();
        next.mcp = None;
        assert_eq!(
            restart_required_changes(&current, &next),
            vec!["mcp".to_owned()]
        );
        let mut next = current.clone();
        next.mcp = Some(mcp("t2"));
        let mcp_change = restart_required_changes(&current, &next);
        assert_eq!(
            mcp_change,
            vec!["mcp".to_owned()],
            "a rotated gateway token is restart-required even though it never renders"
        );
        assert!(
            restart_required_changes(&current, &current).is_empty(),
            "a merely re-stated [mcp] section still reloads"
        );

        // No refusal anywhere names a secret.
        for change in key_store_change
            .iter()
            .chain(&replica_change)
            .chain(&mcp_change)
        {
            assert!(!change.contains("t1"), "tokens never render: {change}");
            assert!(
                !change.contains("redis://"),
                "connection strings never render: {change}"
            );
        }
    }

    #[test]
    fn restart_required_changes_allows_provider_only_edits_and_hot_reload_toggle_free_changes() {
        let current = base_config();
        let next = current.clone();
        assert!(
            restart_required_changes(&current, &next).is_empty(),
            "an identical snapshot needs no restart"
        );
        // hot_reload itself flipping is a server change: the flag gates the
        // SIGHUP wiring, so it lands on restart.
        let mut next = current.clone();
        next.server.hot_reload = false;
        assert_eq!(
            restart_required_changes(&current, &next),
            vec!["server.hot_reload".to_owned()]
        );
    }

    /// Audit finding 1: the reload used to exempt same-id client changes,
    /// so a rotated verifier kept authenticating the old token until
    /// restart while the operator believed the reload applied.
    #[test]
    fn same_id_verifier_rotation_is_restart_required() {
        let current = base_config();
        let mut next = current.clone();
        let Some(ClientKeySpec::Verifier(phc)) = next.auth.clients.get("laptop") else {
            panic!("base config keys laptop with a bare verifier");
        };
        let rotated = phc.expose_secret().to_owned() + "other-hash";
        next.auth.clients.insert(
            "laptop".to_owned(),
            ClientKeySpec::Verifier(SecretString::from(rotated)),
        );
        assert_eq!(
            restart_required_changes(&current, &next),
            vec!["auth.clients.laptop".to_owned()],
            "a rotated verifier under an existing key id must be refused, naming the key"
        );
    }

    #[test]
    fn same_id_policy_change_is_restart_required() {
        let current = base_config();
        let verifier = current
            .auth
            .clients
            .get("laptop")
            .map(|spec| match spec {
                ClientKeySpec::Verifier(phc) => SecretString::from(phc.expose_secret()),
                ClientKeySpec::Policy { verifier, .. } => SecretString::clone(verifier),
            })
            .expect("laptop is keyed");
        let mut next = current.clone();
        next.auth.clients.insert(
            "laptop".to_owned(),
            ClientKeySpec::Policy {
                verifier,
                allow_models: vec!["claude-*".to_owned()],
                allow_providers: Vec::new(),
                limits: None,
                budget: None,
            },
        );
        assert_eq!(
            restart_required_changes(&current, &next),
            vec!["auth.clients.laptop".to_owned()],
            "a tightened allowlist under an existing key id must be refused, naming the key"
        );
    }

    /// The bare-string and table spellings of the same key (verifier equal,
    /// policy all defaults) are one unchanged key, so restating the config
    /// in the other spelling does not demand a restart.
    #[test]
    fn same_verifier_with_only_default_policy_fields_is_not_a_change() {
        let current = base_config();
        let Some(ClientKeySpec::Verifier(phc)) = current.auth.clients.get("laptop") else {
            panic!("base config keys laptop with a bare verifier");
        };
        let mut next = current.clone();
        next.auth.clients.insert(
            "laptop".to_owned(),
            ClientKeySpec::Policy {
                verifier: SecretString::from(phc.expose_secret()),
                allow_models: Vec::new(),
                allow_providers: Vec::new(),
                limits: None,
                budget: None,
            },
        );
        assert!(
            restart_required_changes(&current, &next).is_empty(),
            "the table spelling with only defaults is the same key as the bare string"
        );
    }

    #[test]
    fn load_app_config_defaults_hot_reload_to_false_when_omitted() {
        // The doc promises "Default `false`"; the field must parse as
        // defaulted, not fail, when the key is absent.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
server_name = "test"
"#,
        )
        .unwrap();

        let config = load_app_config(&path).expect("a config omitting hot_reload parses");
        assert!(!config.server.hot_reload);
    }

    #[test]
    fn load_app_config_accepts_enabled_hot_reload() {
        // Plan 07 step 11: hot_reload now means something (SIGHUP reloads the
        // provider and pool files), so it loads instead of being rejected.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "test"
"#,
        )
        .unwrap();

        let config = load_app_config(&path).expect("hot_reload=true loads");
        assert!(config.server.hot_reload);
    }

    #[test]
    fn load_app_config_rejects_zero_request_timeout() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "0s"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .unwrap();

        let error = load_app_config(&path).expect_err("zero timeout must be rejected");
        assert!(
            error
                .to_string()
                .contains("server.request_timeout must be at least 1 second"),
            "got: {error}"
        );
    }

    #[test]
    fn load_app_config_rejects_subsecond_request_timeout() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "500ms"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .unwrap();

        let error = load_app_config(&path).expect_err("sub-second timeout must be rejected");
        assert!(
            error
                .to_string()
                .contains("server.request_timeout must be at least 1 second"),
            "got: {error}"
        );
    }

    #[test]
    fn load_app_config_rejects_a_zero_idempotency_max_entries_when_enabled() {
        // max_entries = 0 would silently retain nothing — every completed
        // outcome evicted the moment it lands, so no keyed retry ever
        // replayed. An enabled store must hold at least one outcome, the
        // same rule the response cache applies.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[server.idempotency]
enabled = true
ttl = "24h"
max_entries = 0
"#,
        )
        .unwrap();

        let error = load_app_config(&path).expect_err("zero max_entries must be rejected");
        assert!(
            error
                .to_string()
                .contains("server.idempotency.max_entries must be at least 1"),
            "got: {error}"
        );
    }

    #[test]
    fn load_app_config_rejects_the_retired_dedup_window_field() {
        // The dedup window was retired (plan 06 step 1): a config that still
        // names it must fail loudly, not silently change dedup behaviour.
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
dedup_window = "500ms"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .unwrap();

        let error = load_app_config(&path).expect_err("retired dedup_window must be rejected");
        assert!(
            error.to_string().contains("unknown field")
                && error.to_string().contains("dedup_window"),
            "got: {error}"
        );
    }

    fn config_dir_with_auth(bind: &str, auth_body: &str) -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "{bind}"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

{auth_body}
"#
            ),
        )
        .expect("write config file");
        directory
    }

    #[test]
    fn auth_absent_section_defaults_to_none_on_loopback() {
        let dir = config_dir_with_auth("127.0.0.1:3456", "");
        let config =
            load_app_config(dir.path().join("config.toml")).expect("default auth accepted");
        assert_eq!(config.auth.mode, AuthMode::None);
        assert!(config.auth.clients.is_empty());
    }

    #[test]
    fn auth_none_on_non_loopback_bind_is_rejected() {
        for bind in ["0.0.0.0:3456", "192.168.1.4:3456", "[::]:3456"] {
            let dir = config_dir_with_auth(bind, "");
            let error = load_app_config(dir.path().join("config.toml"))
                .expect_err("non-loopback none must be rejected");
            assert!(
                error.to_string().contains("not loopback"),
                "bind {bind}: got: {error}"
            );
        }
    }

    #[test]
    fn auth_none_on_ipv6_loopback_bind_is_accepted() {
        let dir = config_dir_with_auth("[::1]:3456", "");
        load_app_config(dir.path().join("config.toml")).expect("ipv6 loopback accepted");
    }

    #[test]
    fn auth_none_with_configured_clients_is_rejected() {
        let dir = config_dir_with_auth(
            "127.0.0.1:3456",
            r#"
[auth]
mode = "none"

[auth.clients]
laptop = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHQ$hash"
"#,
        );
        let error = load_app_config(dir.path().join("config.toml"))
            .expect_err("none mode with clients is contradictory");
        assert!(
            error.to_string().contains("[auth.clients] is non-empty"),
            "got: {error}"
        );
    }

    #[test]
    fn auth_client_keys_without_clients_is_rejected() {
        let dir = config_dir_with_auth(
            "0.0.0.0:3456",
            r#"
[auth]
mode = "client-keys"
"#,
        );
        let error = load_app_config(dir.path().join("config.toml"))
            .expect_err("client-keys mode needs at least one key");
        assert!(
            error.to_string().contains("requires at least one key"),
            "got: {error}"
        );
    }

    #[test]
    fn auth_client_keys_with_only_a_managed_key_store_is_accepted() {
        // A deployment may keep every client key in the admin key store;
        // the daemon merges the store's live keys into the verifiers at
        // startup, so an empty [auth.clients] must not fail validation.
        let dir = config_dir_with_auth(
            "0.0.0.0:3456",
            r#"
[auth]
mode = "client-keys"

[auth.admin]
token = "admin-tok"
key_store = "managed-keys.db"

[storage]
path = "usage.db"
"#,
        );
        let config = load_app_config(dir.path().join("config.toml")).expect(
            "managed keys satisfy the \
                                                                    at-least-one-key rule",
        );
        assert!(config.auth.clients.is_empty());
        assert!(
            config
                .auth
                .admin
                .and_then(|admin| admin.key_store)
                .is_some()
        );
    }

    #[test]
    fn auth_client_keys_with_a_client_accepts_any_bind() {
        let dir = config_dir_with_auth(
            "0.0.0.0:3456",
            r#"
[auth]
mode = "client-keys"

[auth.clients]
laptop = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHQ$hash"
"#,
        );
        let config = load_app_config(dir.path().join("config.toml"))
            .expect("client-keys on non-loopback accepted");
        assert_eq!(config.auth.mode, AuthMode::ClientKeys);
        assert_eq!(config.auth.clients.len(), 1);
    }

    #[test]
    fn auth_rejects_reserved_and_empty_key_ids() {
        let reserved_dir = config_dir_with_auth(
            "127.0.0.1:3456",
            r#"
[auth]
mode = "client-keys"

[auth.clients]
anonymous = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHQ$hash"
"#,
        );
        let reserved = load_app_config(reserved_dir.path().join("config.toml"))
            .expect_err("reserved key id must be rejected");
        assert!(reserved.to_string().contains("reserved"), "got: {reserved}");

        let empty_dir = config_dir_with_auth(
            "127.0.0.1:3456",
            r#"
[auth]
mode = "client-keys"

[auth.clients]
"" = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHQ$hash"
"#,
        );
        let empty = load_app_config(empty_dir.path().join("config.toml"))
            .expect_err("empty key id must be rejected");
        assert!(empty.to_string().contains("empty key id"), "got: {empty}");
    }

    #[test]
    fn auth_clients_do_not_leak_through_debug() {
        let config: AppConfig = toml::from_str(
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[auth]
mode = "client-keys"

[auth.clients]
laptop = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHQ$hashvalue"
"#,
        )
        .expect("app config");
        let debug = format!("{config:?}");
        assert!(
            !debug.contains("hashvalue"),
            "client hash must be redacted in Debug, got: {debug}"
        );
        assert!(debug.contains("[REDACTED]"), "got: {debug}");
    }

    #[test]
    fn has_valid_http_scheme_is_case_insensitive() {
        assert!(has_valid_http_scheme("http://host"));
        assert!(has_valid_http_scheme("https://host"));
        assert!(has_valid_http_scheme("HTTP://host"));
        assert!(has_valid_http_scheme("Https://host"));
        assert!(has_valid_http_scheme("HtTp://host/path"));
        assert!(!has_valid_http_scheme("file://host"));
        assert!(!has_valid_http_scheme("ftp://host"));
        assert!(!has_valid_http_scheme("host"));
    }

    #[test]
    fn provider_routes_rejects_unknown_fields() {
        let result = toml::from_str::<ProviderFile>(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.routes]
chat_completion = "chat"
"#,
        );

        result.unwrap_err();
    }

    #[test]
    fn egress_section_parses_and_absent_section_means_none() {
        let with_egress: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.egress]
proxy = "socks5h://127.0.0.1:1080"
"#,
        )
        .expect("provider with egress parses");
        assert_eq!(
            with_egress.provider.egress,
            Some(ProviderEgressConfig {
                proxy: "socks5h://127.0.0.1:1080".to_owned(),
            })
        );

        let without_egress: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"
"#,
        )
        .expect("provider without egress parses");
        assert_eq!(without_egress.provider.egress, None);
    }

    #[test]
    fn egress_section_rejects_unknown_fields() {
        let result = toml::from_str::<ProviderFile>(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.egress]
proxy = "socks5h://127.0.0.1:1080"
bind_out = "198.51.100.10"
"#,
        );
        result.unwrap_err();
    }

    #[test]
    fn egress_debug_redacts_proxy_userinfo_but_keeps_the_host() {
        let egress = ProviderEgressConfig {
            proxy: "http://proxy-user:proxy-pass@proxy.internal:3128".to_owned(),
        };
        let debug = format!("{egress:?}");
        assert!(
            !debug.contains("proxy-user") && !debug.contains("proxy-pass"),
            "proxy credentials must never print: {debug}"
        );
        assert!(
            debug.contains("proxy.internal:3128"),
            "the credential-free host must stay visible for diagnostics: {debug}"
        );
        let raw = ProviderEgressConfig {
            proxy: "user:pass@host".to_owned(),
        };
        assert!(
            !format!("{raw:?}").contains("user:pass"),
            "unvalidated URLs must not be echoed: {:?}",
            raw
        );
    }

    #[test]
    fn validate_accepts_each_supported_egress_proxy_scheme() {
        for url in [
            "socks5h://tunnel.example.com:1080",
            "socks5://127.0.0.1:1080",
            "http://proxy.internal:3128",
            "https://proxy.internal:8443",
            "SOCKS5H://tunnel.example.com:1080",
            "http://user:pass@proxy.internal:3128",
            "socks5h://tunnel.example.com",
        ] {
            let file: ProviderFile = toml::from_str(&format!(
                r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.egress]
proxy = "{url}"
"#
            ))
            .expect("parses");
            assert_eq!(
                validate_provider_config(&file.provider, None),
                Ok(()),
                "egress proxy {url} must validate"
            );
        }
    }

    #[test]
    fn validate_rejects_unusable_egress_proxy_urls() {
        for (url, reason_fragment) in [
            ("ftp://proxy:21", "unsupported scheme"),
            ("socks4://127.0.0.1:1080", "unsupported scheme"),
            ("127.0.0.1:1080", "no URL scheme"),
            ("proxy.internal:3128", "no URL scheme"),
            ("socks5h://", "no proxy host"),
            ("http:///path", "no proxy host"),
            ("", "empty proxy URL"),
            ("   ", "empty proxy URL"),
        ] {
            let file: ProviderFile = toml::from_str(&format!(
                r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.egress]
proxy = "{url}"
"#
            ))
            .expect("parses");
            let err = validate_provider_config(&file.provider, None)
                .expect_err("egress proxy must be rejected");
            match &err {
                ConfigValidationError::InvalidEgressProxy {
                    provider, reason, ..
                } => {
                    assert_eq!(provider, "example");
                    assert!(
                        reason.contains(reason_fragment),
                        "reason {reason:?} must name {reason_fragment:?} for {url:?}"
                    );
                }
                other => panic!("expected InvalidEgressProxy for {url:?}, got {other:?}"),
            }
            let message = err.to_string();
            assert!(message.contains("example"));
            assert!(message.contains("[provider.egress]"));
        }
    }

    #[test]
    fn load_provider_config_rejects_invalid_egress_at_load_time() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("bad-egress.toml");
        std::fs::write(
            &path,
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.egress]
proxy = "ftp://proxy.example.com:21"
"#,
        )
        .expect("write provider toml");

        let err = load_provider_config(&path, None).expect_err("load must reject bad egress");
        assert!(
            err.to_string().contains("egress"),
            "error must point at the egress section: {err}"
        );
    }

    #[test]
    fn provider_debug_redacts_header_values_and_endpoint_query() {
        let config: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "secret-api-key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions?key=adapter-query-secret"

[provider.adapters.chat.headers]
x-secret = "adapter-secret"

[provider.routes]
chat_completions = "chat"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "https://example.com/v1/models?key=query-secret"

[provider.discovery.headers]
x-discovery-secret = "super-sensitive-value"
"#,
        )
        .expect("provider config");
        let debug = format!("{:?}", config.provider);
        assert!(!debug.contains("secret-api-key"));
        assert!(!debug.contains("adapter-secret"));
        assert!(!debug.contains("adapter-query-secret"));
        assert!(!debug.contains("super-sensitive-value"));
        assert!(!debug.contains("query-secret"));
    }

    #[test]
    fn env_interpolated_api_key_loads_into_secret_string() {
        let _lock = crate::test_support::TestEnvLock::acquire();
        let _guard =
            crate::test_support::EnvVarGuard::set("_LLM_PROXY_TEST_SECRET_KEY", "sk-resolved-007");
        let raw = r#"
[provider]
name = "test"
api_key = "${_LLM_PROXY_TEST_SECRET_KEY}"
auth_style = "bearer"
"#;
        let interpolated = interpolate_env_vars(raw);
        let file: ProviderFile = toml::from_str(&interpolated).expect("provider parses");
        let key: &str = file.provider.api_key.expose_secret();
        assert_eq!(
            key, "sk-resolved-007",
            "env-interpolated value must land in the SecretString",
        );
    }

    #[test]
    fn discovery_rejects_transport_headers() {
        let mut provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "https://example.com/v1/models"
"#,
        )
        .expect("provider config");
        provider
            .provider
            .discovery
            .as_mut()
            .expect("discovery")
            .headers
            .insert("host".to_owned(), "evil.example".to_owned());
        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::ForbiddenHeader { .. })
        ));
    }

    #[test]
    fn discovery_limits_use_secure_defaults() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "https://example.com/v1/models"
"#,
        )
        .expect("provider config");
        let discovery = provider.provider.discovery.expect("discovery");

        assert_eq!(
            (
                discovery.max_pages,
                discovery.max_models,
                discovery.max_response_bytes,
            ),
            (100, 20_000, 4 * 1024 * 1024)
        );
    }

    #[test]
    fn discovery_rejects_zero_limits() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "https://example.com/v1/models"
max_pages = 0
"#,
        )
        .expect("provider config");

        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::InvalidDiscoveryLimit {
                limit: "max_pages",
                ..
            })
        ));
    }

    #[test]
    fn gemini_alias_target_rejects_unsafe_url_template_characters() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "gemini"
api_key = "key"
auth_style = "x-google-api-key"

[provider.adapters.generate]
protocol = "gemini_generate_content"
endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

[provider.routes]
messages = "generate"

[provider.model_aliases]
unsafe = "foo/../bar"
"#,
        )
        .expect("provider config");

        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::UnsafeGeminiModelId {
                location: "model alias target",
                ..
            })
        ));
    }

    #[test]
    fn gemini_static_catalog_entry_rejects_unsafe_url_template_characters() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "gemini"
api_key = "key"
auth_style = "x-google-api-key"

[provider.adapters.generate]
protocol = "gemini_generate_content"
endpoint = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"

[provider.routes]
chat_completions = "generate"

[[provider.catalog.models]]
id = "models/group/gemini-pro"
"#,
        )
        .expect("provider config");

        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::UnsafeGeminiModelId {
                location: "catalog entry",
                ..
            })
        ));
    }

    #[test]
    fn gemini_resource_model_id_is_valid_at_config_load() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "gemini"
api_key = "key"
auth_style = "x-google-api-key"

[provider.adapters.generate]
protocol = "gemini_generate_content"
endpoint = "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"

[provider.routes]
chat_completions = "generate"

[provider.model_aliases]
pro = "models/gemini-2.5-pro"

[[provider.catalog.models]]
id = "models/gemini-2.5-pro"
"#,
        )
        .expect("provider config");

        validate_provider_config(&provider.provider, None).unwrap();
    }

    #[test]
    fn adapter_rejects_header_with_crlf() {
        let mut provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
chat_completions = "chat"
"#,
        )
        .expect("provider config");
        let headers = &mut provider.provider.adapters.get_mut("chat").unwrap().headers;
        Arc::make_mut(headers).insert("x-evil".to_owned(), "value\r\nInjected: true".to_owned());
        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::HeaderContainsCrlf { .. })
        ));
    }

    #[test]
    fn discovery_rejects_header_with_crlf() {
        let mut provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "https://example.com/v1/models"
"#,
        )
        .expect("provider config");
        provider
            .provider
            .discovery
            .as_mut()
            .unwrap()
            .headers
            .insert("x-evil".to_owned(), "val\nInjected: true".to_owned());
        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::HeaderContainsCrlf { .. })
        ));
    }

    #[test]
    fn validate_rejects_empty_adapter_name() {
        let provider: ProviderFile = toml::from_str(
            r#"
[provider]
name = "example"
api_key = "key"
auth_style = "bearer"

[provider.adapters."" ]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1"
"#,
        )
        .expect("provider config");
        assert!(matches!(
            validate_provider_config(&provider.provider, None),
            Err(ConfigValidationError::EmptyAdapterName { .. })
        ));
    }

    #[test]
    fn validate_rejects_empty_protocol() {
        let provider = ProviderConfig {
            name: "example".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::from([(
                "chat".to_owned(),
                ProviderAdapterConfig {
                    protocol: String::new(),
                    endpoint: "https://example.com/v1".to_owned(),
                    headers: Arc::new(HashMap::new()),
                    forward_inbound_headers: None,
                    lossy: false,
                },
            )]),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::EmptyProtocol { .. })
        ));
    }

    #[test]
    fn validate_rejects_empty_model_alias_key() {
        let provider = ProviderConfig {
            name: "example".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::from([(
                "chat".to_owned(),
                ProviderAdapterConfig {
                    protocol: "openai_chat_completions".to_owned(),
                    endpoint: "https://example.com/v1".to_owned(),
                    headers: Arc::new(HashMap::new()),
                    forward_inbound_headers: None,
                    lossy: false,
                },
            )]),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::from([("".to_owned(), "target".to_owned())]),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::EmptyModelAliasKey { .. })
        ));
    }

    #[test]
    fn validate_rejects_empty_model_alias_target() {
        let provider = ProviderConfig {
            name: "example".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::from([(
                "chat".to_owned(),
                ProviderAdapterConfig {
                    protocol: "openai_chat_completions".to_owned(),
                    endpoint: "https://example.com/v1".to_owned(),
                    headers: Arc::new(HashMap::new()),
                    forward_inbound_headers: None,
                    lossy: false,
                },
            )]),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::from([("alias".to_owned(), "".to_owned())]),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::EmptyModelAliasTarget { .. })
        ));
    }

    #[test]
    fn validate_rejects_provider_name_exceeding_length_limit() {
        let provider = ProviderConfig {
            name: "a".repeat(65),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::InvalidProviderNameFormat { .. })
        ));
    }

    #[test]
    fn validate_accepts_provider_name_at_max_length() {
        let provider = ProviderConfig {
            name: "a".repeat(64),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        validate_provider_config(&provider, None).unwrap();
    }

    #[test]
    fn model_pricing_parses_from_toml_with_quoted_decimals() {
        let raw = r#"
[provider]
name = "test"
api_key = "key"
auth_style = "bearer"

[provider.pricing."gpt-4o"]
input = "0.0000025"
output = "0.000010"
cache_creation = "0.000003"
cache_read = "0.00000125"
reasoning = "0.000010"
"#;
        let file: ProviderFile = toml::from_str(raw).expect("parses");
        let pricing = file
            .provider
            .pricing
            .get(&ModelId::new("gpt-4o"))
            .expect("pricing for gpt-4o");
        assert_eq!(pricing.input.to_string(), "0.0000025");
        assert_eq!(pricing.output.to_string(), "0.000010");
        assert_eq!(pricing.cache_creation.to_string(), "0.000003");
        assert_eq!(pricing.cache_read.to_string(), "0.00000125");
        assert_eq!(pricing.reasoning.to_string(), "0.000010");
    }

    #[test]
    fn model_pricing_unquoted_float_is_rejected() {
        let raw = r#"
[provider]
name = "test"
api_key = "key"
auth_style = "bearer"
[provider.pricing."gpt-4o"]
input = 0.0000025
"#;
        toml::from_str::<ProviderFile>(raw).unwrap_err();
    }

    #[test]
    fn validate_rejects_negative_pricing() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: HashMap::from([(
                ModelId::new("gpt-4o"),
                ModelPricing {
                    input: "-0.0001".parse().unwrap(),
                    ..ModelPricing::default()
                },
            )]),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::InvalidPricing { .. })
        ));
    }

    #[test]
    fn validate_rejects_empty_pricing_model_id() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: HashMap::from([(ModelId::new(""), ModelPricing::default())]),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::InvalidPricing { .. })
        ));
    }

    #[test]
    fn validate_rejects_implausibly_large_pricing() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: HashMap::from([(
                ModelId::new("gpt-4o"),
                ModelPricing {
                    input: "1000000".parse().unwrap(),
                    ..ModelPricing::default()
                },
            )]),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        match validate_provider_config(&provider, None) {
            Err(ConfigValidationError::InvalidPricing { message, .. }) => {
                assert!(
                    message.contains("upper") || message.contains("bound"),
                    "expected an upper-bound message, got: {message}"
                );
            }
            other => panic!("expected InvalidPricing for a huge price, got: {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_realistic_pricing_below_bound() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from("key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: HashMap::from([(
                ModelId::new("gpt-4o"),
                ModelPricing {
                    input: "0.0000025".parse().unwrap(),
                    output: "0.000010".parse().unwrap(),
                    ..ModelPricing::default()
                },
            )]),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        validate_provider_config(&provider, None).unwrap();
    }

    #[test]
    fn log_format_from_env_reads_rust_log_format() {
        let _lock = crate::test_support::TestEnvLock::acquire();
        {
            let _g = crate::test_support::EnvVarGuard::set("RUST_LOG_FORMAT", "JSON");
            assert_eq!(LogFormat::from_env(), LogFormat::Json);
        }
        {
            let _g = crate::test_support::EnvVarGuard::set("RUST_LOG_FORMAT", "xml");
            assert_eq!(LogFormat::from_env(), LogFormat::Plain);
        }
        {
            let _g = crate::test_support::EnvVarGuard::set("RUST_LOG_FORMAT", "");
            assert_eq!(LogFormat::from_env(), LogFormat::Plain);
        }
    }

    #[test]
    fn validate_accepts_empty_api_key_when_passthrough_auth() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from(""),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: true,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(
            validate_provider_config(&provider, None).is_ok(),
            "passthrough provider with empty api_key must pass validation"
        );
    }

    #[test]
    fn validate_rejects_empty_api_key_when_not_passthrough() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from(""),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: None,
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        assert!(matches!(
            validate_provider_config(&provider, None),
            Err(ConfigValidationError::EmptyApiKey { .. })
        ));
    }

    #[test]
    fn validate_rejects_an_unresolved_env_var_in_the_extra_keys() {
        let provider = ProviderConfig {
            name: "test".to_owned(),
            api_key: SecretString::from("base-key"),
            auth_style: AuthStyle::Bearer,
            passthrough_auth: false,
            api_keys: Some(vec![
                SecretString::from("second-key"),
                SecretString::from("${LLM_PROXY_TEST_MISSING_KEY}"),
            ]),
            retry: None,
            egress: None,
            adapters: HashMap::new(),
            routes: ProviderRoutesConfig::default(),
            model_aliases: HashMap::new(),
            discovery: None,
            catalog: None,
            pricing: Default::default(),
            concurrency: None,
            pricing_sync: None,
            prompt_cache: None,
        };
        match validate_provider_config(&provider, None) {
            Err(ConfigValidationError::UnresolvedEnvVar { provider, var }) => {
                assert_eq!(provider, "test");
                assert_eq!(var, "LLM_PROXY_TEST_MISSING_KEY");
            }
            other => panic!("an unresolved extra key must fail validation, got {other:?}"),
        }
    }

    fn write_app_config(extra_storage: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"
{extra_storage}"#
            ),
        )
        .unwrap();
        (directory, path)
    }

    #[test]
    fn replica_section_parses_and_defaults_to_none() {
        let (_dir, path) = write_app_config(
            r#"
[replica]
backend = "redis"
url = "redis://127.0.0.1:6379"
prefix = "fleet-a""#,
        );
        let config = load_app_config(&path).expect("valid config with [replica]");
        let replica = config.replica.as_ref().expect("section parsed");
        assert!(matches!(replica.backend, super::ReplicaBackend::Redis));
        assert_eq!(replica.prefix, "fleet-a");
        assert_eq!(
            replica.url.expose_secret(),
            "redis://127.0.0.1:6379",
            "the URL round-trips and never serializes back out"
        );
        assert!(
            serde_json::to_value(replica).is_ok(),
            "redacted serialization stays total"
        );

        let (_dir, path) = write_app_config("");
        let config = load_app_config(&path).expect("valid config");
        assert!(
            config.replica.is_none(),
            "absent [replica] keeps single-node state"
        );
    }

    #[test]
    fn replica_prefix_defaults_and_postgres_backend_parses() {
        let (_dir, path) = write_app_config(
            r#"
[replica]
backend = "postgres"
url = "postgres://user:pass@db.example.com/llm""#,
        );
        let config = load_app_config(&path).expect("valid config with [replica]");
        let replica = config.replica.as_ref().expect("section parsed");
        assert!(matches!(replica.backend, super::ReplicaBackend::Postgres));
        assert_eq!(
            replica.prefix, "llm-proxy",
            "an unset prefix defaults instead of rejecting"
        );
    }

    #[test]
    fn mcp_section_parses_with_defaults() {
        let (_dir, path) = write_app_config(
            r#"
[mcp]
bind = "127.0.0.1:3913"
[[mcp.upstreams]]
name = "docs"
url = "https://mcp.example.com/mcp""#,
        );
        let config = load_app_config(&path).expect("valid config with [mcp]");
        let mcp = config.mcp.as_ref().expect("section parsed");
        assert_eq!(
            mcp.bind,
            "127.0.0.1:3913".parse::<std::net::SocketAddr>().unwrap()
        );
        assert_eq!(mcp.upstreams.len(), 1);
        assert_eq!(mcp.upstreams[0].name, "docs");
        assert!(
            mcp.tokens.is_empty(),
            "no tokens configured refuses everyone"
        );
        assert_eq!(mcp.session_ttl, Duration::from_secs(1800));
    }

    #[test]
    fn semantic_cache_is_off_by_default_and_parses_opt_in() {
        let (_dir, path) = write_app_config("");
        let config = load_app_config(&path).expect("valid config");
        assert!(
            !config.server.semantic_cache.enabled,
            "the semantic cache is off unless opted in"
        );

        let (_dir, path) = write_app_config(
            r#"
[server.semantic_cache]
enabled = true
ttl = "5m"
max_entries = 64
threshold = 0.9"#,
        );
        let config = load_app_config(&path).expect("valid config with [server.semantic_cache]");
        assert!(config.server.semantic_cache.enabled);
        assert_eq!(config.server.semantic_cache.ttl, Duration::from_secs(300));
        assert_eq!(config.server.semantic_cache.max_entries, 64);
    }

    #[test]
    fn enabling_both_cache_optins_is_refused() {
        // The engine holds one cache: enabling both opt-ins would silently
        // replace the exact-match cache with the semantic one. The pair is
        // a config error, not a precedence rule.
        let (_dir, path) = write_app_config(
            r#"
[server.cache]
enabled = true
ttl = "30s"
max_entries = 8

[server.semantic_cache]
enabled = true"#,
        );
        let error =
            load_app_config(&path).expect_err("both cache opt-ins must be refused at load time");
        assert!(
            error.to_string().contains("mutually exclusive"),
            "the refusal must name the conflict, got: {error}"
        );
    }

    #[test]
    fn storage_section_defaults_to_none_when_absent() {
        let (_dir, path) = write_app_config("");
        let config = load_app_config(&path).expect("valid config without [storage]");
        assert!(
            config.storage.is_none(),
            "absent [storage] must keep the no-op event bus"
        );
    }

    #[test]
    fn storage_section_parses_and_applies_defaults() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
"#,
        );
        let config = load_app_config(&path).expect("valid [storage]");
        let storage = config.storage.expect("storage present");
        assert_eq!(storage.path, std::path::PathBuf::from("/tmp/usage.db"));
        assert_eq!(storage.max_age, None, "absent max_age keeps rows forever");
        assert_eq!(storage.max_rows, None, "absent max_rows is unlimited");
        assert_eq!(
            storage.prune_interval,
            Duration::from_secs(3600),
            "absent prune_interval defaults to 1h"
        );
    }

    #[test]
    fn storage_section_parses_full_retention() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
max_age = "30d"
max_rows = 500_000
prune_interval = "15m"
"#,
        );
        let config = load_app_config(&path).expect("valid [storage]");
        let storage = config.storage.expect("storage present");
        assert_eq!(storage.max_age, Some(Duration::from_secs(30 * 24 * 3600)));
        assert_eq!(storage.max_rows, Some(500_000));
        assert_eq!(storage.prune_interval, Duration::from_secs(15 * 60));
    }

    #[test]
    fn storage_zero_prune_interval_is_rejected() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
prune_interval = "0s"
"#,
        );
        let error = load_app_config(&path).expect_err("zero prune interval rejected");
        assert!(
            error.to_string().contains("storage.prune_interval"),
            "got: {error}"
        );
    }

    #[test]
    fn storage_subsecond_prune_interval_is_rejected() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
max_age = "1d"
prune_interval = "500ms"
"#,
        );
        assert!(
            load_app_config(&path).is_err(),
            "sub-second prune interval must be rejected (timer panics on zero/near-zero)"
        );
    }

    #[test]
    fn storage_zero_max_age_is_rejected() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
max_age = "0s"
"#,
        );
        let error = load_app_config(&path).expect_err("zero max_age rejected");
        assert!(
            error.to_string().contains("storage.max_age"),
            "got: {error}"
        );
    }

    #[test]
    fn storage_zero_max_rows_is_rejected() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
max_rows = 0
"#,
        );
        let error = load_app_config(&path).expect_err("zero max_rows rejected");
        assert!(
            error.to_string().contains("storage.max_rows"),
            "got: {error}"
        );
    }

    #[test]
    fn storage_empty_path_is_rejected() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = ""
"#,
        );
        let error = load_app_config(&path).expect_err("empty path rejected");
        assert!(error.to_string().contains("storage.path"), "got: {error}");
    }

    #[test]
    fn storage_unknown_field_is_rejected() {
        let (_dir, path) = write_app_config(
            r#"
[storage]
path = "/tmp/usage.db"
db_host = "postgres.example"
"#,
        );
        assert!(
            load_app_config(&path).is_err(),
            "deny_unknown_fields must reject unrecognized [storage] keys"
        );
    }

    const STORAGE_SECTION: &str = "\n[storage]\npath = \"/tmp/usage.db\"\n";

    #[test]
    fn admin_section_parses_token_and_defaults_off() {
        let (_dir, path) = write_app_config(&format!(
            "{STORAGE_SECTION}\n[auth.admin]\ntoken = \"admin-secret\"\n"
        ));
        let config = load_app_config(&path).expect("valid [auth.admin]");
        assert!(
            config
                .auth
                .admin
                .as_ref()
                .is_some_and(|a| a.token.expose_secret() == "admin-secret"),
            "admin token must parse into the SecretString"
        );
        assert!(
            !config.server.admin_on_main_listener,
            "main-listener mount must be opt-in"
        );
        assert_eq!(
            config.server.effective_admin_bind(),
            Some("127.0.0.1:3457".parse().unwrap()),
            "default admin bind is loopback on the port after the main bind"
        );
        assert_eq!(
            config.admin_posture(),
            Some(AdminMountPosture::DedicatedListener),
            "the default posture is the dedicated listener"
        );
    }

    #[test]
    fn admin_explicit_bind_and_main_listener_opt_in_parse() {
        let (_dir, path) = write_app_config(&format!(
            "\nadmin_bind = \"127.0.0.1:9900\"\nadmin_on_main_listener = true\n{STORAGE_SECTION}\n\
             [auth.admin]\ntoken = \"admin-secret\"\n"
        ));
        let config = load_app_config(&path).expect("valid admin config");
        assert_eq!(
            config.server.effective_admin_bind(),
            Some("127.0.0.1:9900".parse().unwrap()),
            "explicit admin_bind wins over the derived default"
        );
        assert_eq!(
            config.admin_posture(),
            Some(AdminMountPosture::MainListener),
            "admin_on_main_listener selects the main-listener posture"
        );
        assert_eq!(
            AdminMountPosture::MainListener.name(),
            "admin-on-main-listener",
            "posture names are stable for log grepping"
        );
    }

    #[test]
    fn admin_without_storage_is_rejected() {
        let (_dir, path) = write_app_config("\n[auth.admin]\ntoken = \"admin-secret\"\n");
        let error = load_app_config(&path).expect_err("admin requires [storage]");
        assert!(
            error.to_string().contains("[auth.admin] requires"),
            "got: {error}"
        );
    }

    #[test]
    fn admin_empty_token_is_rejected() {
        let (_dir, path) =
            write_app_config(&format!("{STORAGE_SECTION}\n[auth.admin]\ntoken = \"\"\n"));
        let error = load_app_config(&path).expect_err("empty admin token rejected");
        assert!(
            error.to_string().contains("token must not be empty"),
            "got: {error}"
        );
    }

    #[test]
    fn admin_bind_equal_to_main_bind_is_rejected() {
        let (_dir, path) = write_app_config(&format!(
            "\nadmin_bind = \"127.0.0.1:3456\"\n{STORAGE_SECTION}\n[auth.admin]\n\
             token = \"admin-secret\"\n"
        ));
        let error = load_app_config(&path).expect_err("same-socket bind rejected");
        assert!(
            error.to_string().contains("must differ from server.bind"),
            "got: {error}"
        );
    }

    #[test]
    fn admin_default_port_underviable_on_65535_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                "[server]\nbind = \"127.0.0.1:65535\"\nrequest_timeout = \"300s\"\n\
                 log_level = \"info\"\nhot_reload = false\nserver_name = \"t\"\n{STORAGE_SECTION}\n\
                 \n[auth.admin]\ntoken = \"admin-secret\"\n"
            ),
        )
        .unwrap();
        let error = load_app_config(&path).expect_err("no derivable admin port");
        assert!(error.to_string().contains("65535"), "got: {error}");
    }

    #[test]
    fn admin_main_listener_optin_needs_no_derivable_port() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                "[server]\nbind = \"127.0.0.1:65535\"\nrequest_timeout = \"300s\"\n\
                 log_level = \"info\"\nhot_reload = false\nserver_name = \"t\"\n\
                 admin_on_main_listener = true\n{STORAGE_SECTION}\n\
                 \n[auth.admin]\ntoken = \"admin-secret\"\n"
            ),
        )
        .unwrap();
        let config = load_app_config(&path).expect("main-listener posture needs no port");
        assert!(
            !config.server.dedicated_admin_listener(),
            "the opt-in without an explicit admin_bind must not bind a dedicated socket"
        );
    }

    #[test]
    fn explicit_admin_bind_keeps_the_dedicated_listener_under_the_optin() {
        let (_dir, path) = write_app_config(&format!(
            "\nadmin_bind = \"127.0.0.1:9900\"\nadmin_on_main_listener = true\n{STORAGE_SECTION}\n\
             [auth.admin]\ntoken = \"admin-secret\"\n"
        ));
        let config = load_app_config(&path).expect("both mounts are legitimate");
        assert!(
            config.server.dedicated_admin_listener(),
            "an explicit admin_bind always binds the dedicated listener"
        );
        assert_eq!(
            config.admin_posture(),
            Some(AdminMountPosture::MainListener),
            "and the main-listener mount is on top of it"
        );
    }

    #[test]
    fn admin_bind_without_admin_section_is_rejected() {
        let (_dir, path) = write_app_config(&format!(
            "\nadmin_bind = \"127.0.0.1:9900\"\n{STORAGE_SECTION}"
        ));
        let error = load_app_config(&path).expect_err("dead admin_bind rejected");
        assert!(
            error
                .to_string()
                .contains("server.admin_bind is configured"),
            "got: {error}"
        );
    }

    #[test]
    fn admin_main_listener_flag_without_admin_section_is_rejected() {
        let (_dir, path) = write_app_config(&format!(
            "\nadmin_on_main_listener = true\n{STORAGE_SECTION}"
        ));
        let error = load_app_config(&path).expect_err("dead flag rejected");
        assert!(
            error.to_string().contains("admin_on_main_listener"),
            "got: {error}"
        );
    }

    #[test]
    fn admin_unknown_field_is_rejected() {
        let (_dir, path) = write_app_config(&format!(
            "{STORAGE_SECTION}\n[auth.admin]\ntoken = \"admin-secret\"\nrotates = \"weekly\"\n"
        ));
        assert!(
            load_app_config(&path).is_err(),
            "deny_unknown_fields must reject unrecognized [auth.admin] keys"
        );
    }

    #[test]
    fn admin_posture_is_none_without_admin_section() {
        let (_dir, path) = write_app_config(STORAGE_SECTION);
        let config = load_app_config(&path).expect("valid config");
        assert_eq!(
            config.admin_posture(),
            None,
            "no [auth.admin] means no admin surface anywhere"
        );
    }

    #[test]
    fn admin_auth_config_debug_redacts_token() {
        let admin = AdminAuthConfig {
            token: SecretString::from("super-secret-admin-token"),
            key_store: None,
        };
        let debug = format!("{admin:?}");
        assert!(
            !debug.contains("super-secret-admin-token"),
            "Debug must redact the admin token, got: {debug}"
        );
    }
}
