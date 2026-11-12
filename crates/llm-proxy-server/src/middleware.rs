//! Middleware components for the LLM proxy server.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use axum::extract::{ConnectInfo, FromRequestParts};
use axum::http::request::Parts;
use llm_proxy_core::{ANONYMOUS_CALLER, AuthConfig, AuthMode};
use secrecy::{ExposeSecret, SecretString};
use sha2::{Digest, Sha256};

use argon2::PasswordVerifier;

const MAX_RATE_LIMIT_BUCKETS: usize = 100_000;

#[derive(Debug)]
struct ClientTokenBucket {
    tokens: f64,
    max_tokens: f64,
    last_refill: Instant,
}

impl ClientTokenBucket {
    fn new(max_tokens: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self, refill_rate: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * refill_rate).min(self.max_tokens);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Per-IP rate limiter using token buckets.
#[derive(Debug)]
pub struct RateLimiter {
    buckets: Mutex<HashMap<String, ClientTokenBucket>>,
    max_requests_per_minute: f64,
    prune_counter: AtomicU64,
}

const PRUNE_INTERVAL: u64 = 128;

const RATE_LIMITER_IDLE_EVICT_SECS: u64 = 300;

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// `max_requests_per_minute` is the per-client-IP request budget per
    /// minute; `0` disables rate limiting.
    pub fn new(max_requests_per_minute: u32) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            max_requests_per_minute: max_requests_per_minute as f64,
            prune_counter: AtomicU64::new(0),
        }
    }

    /// Check whether a request from `client_ip` is allowed.
    ///
    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub fn is_allowed(&self, client_ip: &str) -> bool {
        if self.max_requests_per_minute == 0.0 {
            return true;
        }
        let mut buckets = self.buckets.lock().unwrap_or_else(|e| {
            tracing::warn!("rate limiter mutex poisoned; recovering from panic");
            e.into_inner()
        });

        if self.prune_counter.fetch_add(1, Ordering::Relaxed) % PRUNE_INTERVAL == 0 {
            let now = std::time::Instant::now();
            buckets.retain(|_, bucket| {
                now.duration_since(bucket.last_refill)
                    < std::time::Duration::from_secs(RATE_LIMITER_IDLE_EVICT_SECS)
            });
        }

        let refill_rate = self.max_requests_per_minute / 60.0;

        if let Some(bucket) = buckets.get_mut(client_ip) {
            return bucket.try_consume(refill_rate);
        }

        if buckets.len() >= MAX_RATE_LIMIT_BUCKETS {
            return false;
        }

        let bucket = buckets
            .entry(client_ip.to_owned())
            .or_insert_with(|| ClientTokenBucket::new(self.max_requests_per_minute));

        bucket.try_consume(refill_rate)
    }

    /// Whether a request from `client_ip` would currently be allowed,
    /// **without consuming a token**.
    #[must_use]
    pub fn would_allow(&self, client_ip: &str) -> bool {
        if self.max_requests_per_minute == 0.0 {
            return true;
        }
        let buckets = self.buckets.lock().unwrap_or_else(|e| {
            tracing::warn!("rate limiter mutex poisoned; recovering from panic");
            e.into_inner()
        });
        let Some(bucket) = buckets.get(client_ip) else {
            return true;
        };
        let elapsed = Instant::now()
            .duration_since(bucket.last_refill)
            .as_secs_f64();
        let refill_rate = self.max_requests_per_minute / 60.0;
        let tokens = (bucket.tokens + elapsed * refill_rate).min(bucket.max_tokens);
        tokens >= 1.0
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Generates unique request IDs in the format `req-{unix}-{counter}`.
#[derive(Debug)]
pub struct RequestIdGenerator {
    counter: AtomicU64,
}

impl RequestIdGenerator {
    /// Create a new request ID generator.
    pub fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Generate the next request ID, `req-{unix_seconds}-{monotonic_counter}`.
    ///
    /// A pre-epoch clock falls back to timestamp `0`; the counter still
    /// guarantees uniqueness within the process.
    pub fn next_id(&self) -> String {
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_else(|e| {
                tracing::error!(error = %e, "system clock is before UNIX epoch; using 0 as request ID timestamp");
                0
            });
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("req-{unix}-{counter}")
    }
}

impl Default for RequestIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Typed request identifier carried in request extensions.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Optional peer socket metadata for handlers that also run in router-only tests.
#[derive(Clone, Copy, Debug)]
pub struct OptionalConnectInfo(pub Option<SocketAddr>);

impl<S> FromRequestParts<S> for OptionalConnectInfo
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> + Send {
        let address = parts
            .extensions
            .get::<ConnectInfo<SocketAddr>>()
            .map(|connect_info| connect_info.0);
        async move { Ok(Self(address)) }
    }
}

/// Client IP: the leftmost `X-Forwarded-For`/`X-Real-Ip` value when
/// `trust_forwarded_headers` is set and it parses as an IP, else the
/// connection info, else `"unknown"`. Safe only behind a single proxy.
pub fn get_client_ip(
    headers: &axum::http::HeaderMap,
    connect_info: Option<&SocketAddr>,
    trust_forwarded_headers: bool,
) -> String {
    if trust_forwarded_headers {
        if let Some(xff) = headers.get("x-forwarded-for")
            && let Ok(val) = xff.to_str()
            && let Some(ip) = val.split(',').next()
        {
            let trimmed = ip.trim();
            if trimmed.parse::<std::net::IpAddr>().is_ok() {
                return trimmed.to_owned();
            }
        }

        if let Some(xri) = headers.get("x-real-ip")
            && let Ok(val) = xri.to_str()
        {
            let trimmed = val.trim();
            if trimmed.parse::<std::net::IpAddr>().is_ok() {
                return trimmed.to_owned();
            }
        }
    }

    if let Some(ci) = connect_info {
        return ci.ip().to_string();
    }

    "unknown".to_owned()
}

struct ClientVerifier {
    key_id: String,
    phc: SecretString,
}

struct CachedDigest {
    digest: [u8; 32],
    key_id: String,
}

/// Splits a prefixed client token `<key-id>.<secret>` (audit conflict 12,
/// plan 05 step 2). The id is everything before the first `.`; both parts
/// must be non-empty. A token without a `.` is a legacy bare secret and
/// returns `None` (accepted only under `legacy_bare_tokens`, and then
/// scanned with a cap).
fn split_prefixed_token(token: &str) -> Option<(&str, &str)> {
    let (key_id, secret) = token.split_once('.')?;
    (!key_id.is_empty() && !secret.is_empty()).then_some((key_id, secret))
}

/// Upper bound on the per-request verifier scan a `legacy_bare_tokens`
/// deployment pays for one bare token; more keys than this and the scan
/// is skipped (logged once) rather than paid.
const LEGACY_SCAN_MAX: usize = 16;

/// How many recent failed-token digests the negative cache retains.
const NEGATIVE_CACHE_MAX: usize = 256;

/// How long a failed-token digest stays in the negative cache; a repeat
/// of the same bad token inside this window costs no argon2 run.
const NEGATIVE_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// Whether a managed key store is configured (`[auth.admin].key_store`):
/// it can be a deployment's only key source, so client-keys mode may
/// start with no verifiers loaded and fail closed per request until the
/// admin listener issues one.
fn has_key_store(config: &AuthConfig) -> bool {
    config
        .admin
        .as_ref()
        .is_some_and(|admin| admin.key_store.is_some())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Errors raised while building [`ClientAuth`] from configuration.
#[derive(Debug, thiserror::Error)]
pub enum ClientAuthError {
    /// `auth.mode = "none"` was combined with configured client keys.
    #[error("auth.mode is \"none\" but [auth.clients] is non-empty")]
    ModeNoneWithClients,
    /// `auth.mode = "client-keys"` had no client keys configured.
    #[error("auth.mode is \"client-keys\" but [auth.clients] is empty")]
    ClientKeysWithoutClients,
    /// A key id was empty or the reserved `anonymous` name.
    #[error("client key id \"{0}\" is empty or reserved")]
    InvalidKeyId(String),
    /// A verifier string is not a parseable PHC argon2 hash.
    ///
    /// The error carries the key id, never the hash string itself.
    #[error("client key \"{key_id}\" has an invalid PHC verifier: {source}")]
    InvalidVerifier {
        /// The key id whose verifier failed to parse.
        key_id: String,
        /// The underlying parse error.
        source: argon2::password_hash::Error,
    },
}

/// Client-key authentication state: the configured verifiers plus a digest
/// cache for verified keys. Holds hashed verifiers only, never plaintext.
///
/// Tokens come in two shapes. The documented one is `<key-id>.<secret>`:
/// the id selects exactly one verifier, so a bad token costs at most one
/// argon2 run and an unknown id costs none, whatever the number of keys
/// (audit conflict 12). A bare secret with no `.` is the legacy shape and
/// is refused unless `legacy_bare_tokens` is configured; with the opt-in,
/// it is scanned against every verifier but only up to
/// [`LEGACY_SCAN_MAX`] keys — a deployment with more keys than the cap
/// gets a refusal instead of an unbounded scan, so a bad token's lookup
/// cost never grows unboundedly with the key count.
///
/// The digest of a definitively failed token is remembered briefly in a
/// bounded negative cache, so a client retrying the same bad token costs
/// one argon2 run, not one per attempt. The cache lives inside this
/// instance: a hot-reload swap (new keys, e.g. a freshly issued key)
/// starts with an empty negative cache and cannot be blocked by a stale
/// entry.
///
/// Manual `Debug` prints the mode and key ids, never the verifier strings.
pub struct ClientAuth {
    enforced: bool,
    /// Whether tokens without a `<key-id>.` prefix are accepted by
    /// scanning verifiers (`[auth] legacy_bare_tokens = true`).
    legacy_bare_tokens: bool,
    /// Sorted by `key_id` so a prefix lookup is a binary search.
    verifiers: Vec<ClientVerifier>,
    cache: Mutex<Vec<CachedDigest>>,
    /// Digests of recently failed tokens: no token bytes, digest plus
    /// insertion time, bounded and time-limited.
    failed_recently: Mutex<VecDeque<([u8; 32], Instant)>>,
    /// Set the first time a legacy scan is skipped for exceeding
    /// [`LEGACY_SCAN_MAX`], so the warning fires once per process.
    legacy_cap_warned: AtomicBool,
    /// argon2 password verifications run so far: the observable lookup
    /// cost, exposed so the scaling property is testable.
    verifications: AtomicU64,
}

impl std::fmt::Debug for ClientAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientAuth")
            .field("enforced", &self.enforced)
            .field(
                "client_key_ids",
                &self
                    .verifiers
                    .iter()
                    .map(|v| v.key_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ClientAuth {
    /// Build authentication state from the `[auth]` config section.
    ///
    /// Returns [`ClientAuthError`] on contradictory mode/client combinations,
    /// reserved or empty key ids, or an unparseable verifier string.
    /// Client-keys mode may start with no verifiers at all when a managed
    /// key store is configured: with none loaded yet every proxy request
    /// fails closed until one is issued, which the admin listener (behind
    /// its own token) can always do.
    pub fn from_config(config: &AuthConfig) -> Result<Self, ClientAuthError> {
        match config.mode {
            AuthMode::None if !config.clients.is_empty() => {
                return Err(ClientAuthError::ModeNoneWithClients);
            }
            AuthMode::ClientKeys if config.clients.is_empty() && !has_key_store(config) => {
                return Err(ClientAuthError::ClientKeysWithoutClients);
            }
            AuthMode::None | AuthMode::ClientKeys => {}
        }
        let mut verifiers = Vec::with_capacity(config.clients.len());
        for (key_id, spec) in &config.clients {
            if key_id.is_empty() || key_id == ANONYMOUS_CALLER {
                return Err(ClientAuthError::InvalidKeyId(key_id.clone()));
            }
            let phc = spec.verifier();
            if let Err(source) = argon2::password_hash::phc::PasswordHash::new(phc.expose_secret())
            {
                return Err(ClientAuthError::InvalidVerifier {
                    key_id: key_id.clone(),
                    source: source.into(),
                });
            }
            verifiers.push(ClientVerifier {
                key_id: key_id.clone(),
                phc: SecretString::from(phc.expose_secret().to_owned()),
            });
        }
        verifiers.sort_by(|a, b| a.key_id.cmp(&b.key_id));
        Ok(Self {
            enforced: matches!(config.mode, AuthMode::ClientKeys),
            legacy_bare_tokens: config.legacy_bare_tokens,
            verifiers,
            cache: Mutex::new(Vec::new()),
            failed_recently: Mutex::new(VecDeque::new()),
            legacy_cap_warned: AtomicBool::new(false),
            verifications: AtomicU64::new(0),
        })
    }

    /// Fail-closed authenticator: enforcement on, nothing can authenticate.
    ///
    /// Used when config is invalid at construction time so every proxied
    /// request is rejected rather than silently downgraded to no-auth.
    #[must_use]
    pub fn fail_closed() -> Self {
        Self {
            enforced: true,
            legacy_bare_tokens: false,
            verifiers: Vec::new(),
            cache: Mutex::new(Vec::new()),
            failed_recently: Mutex::new(VecDeque::new()),
            legacy_cap_warned: AtomicBool::new(false),
            verifications: AtomicU64::new(0),
        }
    }

    /// Whether proxy routes require a valid client key.
    #[must_use]
    pub fn is_enforced(&self) -> bool {
        self.enforced
    }

    /// Number of configured client keys (0 when not enforced).
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.verifiers.len()
    }

    /// How many argon2 password verifications this authenticator has run:
    /// the lookup cost a bad token incurs, cumulative since construction.
    #[must_use]
    pub fn argon2_verifications(&self) -> u64 {
        self.verifications.load(Ordering::Relaxed)
    }

    fn cached_key_id(&self, digest: &[u8; 32]) -> Option<String> {
        let cache = self.cache.lock().unwrap_or_else(|e| {
            tracing::warn!("client auth digest cache poisoned; recovering from panic");
            e.into_inner()
        });
        cache
            .iter()
            .find(|entry| constant_time_eq(&entry.digest, digest))
            .map(|entry| entry.key_id.clone())
    }

    /// Runs argon2 against `candidates` (key id, PHC) in order, off the
    /// async runtime, and caches the digest of a matching token.
    ///
    /// Returns the matched key id plus whether the outcome is definitive:
    /// `false` means the verification task itself failed (spawn/join), so
    /// the token must not be remembered as failed.
    async fn verify_against(
        &self,
        token: &str,
        digest: [u8; 32],
        candidates: Vec<(String, String)>,
    ) -> (Option<String>, bool) {
        if candidates.is_empty() {
            return (None, true);
        }
        let token = token.to_owned();
        let outcome = tokio::task::spawn_blocking(move || {
            let argon2 = argon2::Argon2::default();
            let mut runs = 0u64;
            let mut matched = None;
            for (key_id, phc) in &candidates {
                let Ok(hash) = argon2::password_hash::phc::PasswordHash::new(phc) else {
                    continue;
                };
                runs += 1;
                if argon2.verify_password(token.as_bytes(), &hash).is_ok() {
                    matched = Some(key_id.clone());
                    break;
                }
            }
            (matched, runs)
        })
        .await
        .map_err(|error| {
            tracing::warn!(%error, "client auth verification task failed");
            error
        })
        .ok();
        let Some((matched, runs)) = outcome else {
            return (None, false);
        };
        self.verifications.fetch_add(runs, Ordering::Relaxed);
        if let Some(key_id) = &matched {
            let mut cache = self.cache.lock().unwrap_or_else(|e| {
                tracing::warn!("client auth digest cache poisoned; recovering from panic");
                e.into_inner()
            });
            if !cache
                .iter()
                .any(|entry| constant_time_eq(&entry.digest, &digest))
            {
                cache.push(CachedDigest {
                    digest,
                    key_id: key_id.clone(),
                });
            }
        }
        (matched, true)
    }

    /// The one verifier a prefixed token names, if that key id exists.
    fn verifier_for(&self, key_id: &str) -> Option<(String, String)> {
        self.verifiers
            .binary_search_by(|v| v.key_id.as_str().cmp(key_id))
            .ok()
            .and_then(|index| self.verifiers.get(index))
            .map(|v| (v.key_id.clone(), v.phc.expose_secret().to_owned()))
    }

    /// Whether this exact token digest definitively failed recently; also
    /// drops expired entries. Digests compare in constant time.
    fn recently_failed(&self, digest: &[u8; 32]) -> bool {
        let mut failed = self.failed_recently.lock().unwrap_or_else(|e| {
            tracing::warn!("client auth negative cache poisoned; recovering from panic");
            e.into_inner()
        });
        let now = Instant::now();
        failed.retain(|(_, at)| now.duration_since(*at) < NEGATIVE_CACHE_TTL);
        failed
            .iter()
            .any(|(failed_digest, _)| constant_time_eq(failed_digest, digest))
    }

    /// Records a definitively failed token digest (never the token bytes).
    fn remember_failure(&self, digest: [u8; 32]) {
        let mut failed = self.failed_recently.lock().unwrap_or_else(|e| {
            tracing::warn!("client auth negative cache poisoned; recovering from panic");
            e.into_inner()
        });
        failed.push_back((digest, Instant::now()));
        while failed.len() > NEGATIVE_CACHE_MAX {
            failed.pop_front();
        }
    }

    /// Warns once per process that the legacy bare-token scan was skipped
    /// because the key count exceeds the cap.
    fn warn_legacy_scan_capped(&self) {
        if !self.legacy_cap_warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                cap = LEGACY_SCAN_MAX,
                keys = self.verifiers.len(),
                "legacy_bare_tokens scan skipped: more configured keys than the cap, \
                 bare tokens are refused"
            );
        }
    }

    /// Resolves a presented token to its key id, or `None`.
    ///
    /// Order: the digest cache (no argon2), the negative cache of recent
    /// failures (no argon2), then — for a `<key-id>.<secret>` token — the
    /// one verifier that id names (one argon2 run, none when the id is
    /// unknown). A bare token is refused with zero argon2 runs unless
    /// `legacy_bare_tokens` is set, and even then the scan is capped.
    ///
    /// This is the check the managed-key flow must satisfy: a store's
    /// `client_specs()` verifiers have to accept the tokens that store
    /// issues (the secret after the first `.`), which is what
    /// `llm-proxy-admin` tests against.
    pub async fn authenticate(&self, token: &str) -> Option<String> {
        if !self.enforced || self.verifiers.is_empty() {
            return None;
        }
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        if let Some(key_id) = self.cached_key_id(&digest) {
            return Some(key_id);
        }
        if self.recently_failed(&digest) {
            return None;
        }
        match split_prefixed_token(token) {
            Some((key_id, secret)) => {
                let candidates = self.verifier_for(key_id).into_iter().collect();
                let (matched, definitive) = self.verify_against(secret, digest, candidates).await;
                if definitive && matched.is_none() {
                    self.remember_failure(digest);
                }
                matched
            }
            None => {
                if !self.legacy_bare_tokens {
                    return None;
                }
                if self.verifiers.len() > LEGACY_SCAN_MAX {
                    self.warn_legacy_scan_capped();
                    return None;
                }
                let candidates = self
                    .verifiers
                    .iter()
                    .map(|v| (v.key_id.clone(), v.phc.expose_secret().to_owned()))
                    .collect();
                let (matched, definitive) = self.verify_against(token, digest, candidates).await;
                if definitive && matched.is_none() {
                    self.remember_failure(digest);
                }
                matched
            }
        }
    }
}

pub(crate) fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<SecretString> {
    let value = headers.get(axum::http::header::AUTHORIZATION)?;
    let header = value.to_str().ok()?;
    let mut parts = header.splitn(2, ' ');
    match (parts.next(), parts.next()) {
        (Some(scheme), Some(token)) if scheme.eq_ignore_ascii_case("bearer") => {
            let token = token.trim();
            if token.is_empty() {
                None
            } else {
                Some(SecretString::from(token.to_owned()))
            }
        }
        _ => None,
    }
}

/// Typed caller identity carried in request extensions.
///
/// The authenticated key id when client keys are enforced, or
/// [`CallerId::anonymous`] under the loopback no-auth exemption.
#[derive(Debug, Clone)]
pub struct CallerId {
    key_id: Option<String>,
}

impl CallerId {
    /// The unauthenticated caller used under the loopback no-auth exemption.
    #[must_use]
    pub fn anonymous() -> Self {
        Self { key_id: None }
    }

    /// An authenticated caller identified by `key_id`.
    #[must_use]
    pub fn authenticated(key_id: impl Into<String>) -> Self {
        Self {
            key_id: Some(key_id.into()),
        }
    }

    /// Whether a client key authenticated this request.
    #[must_use]
    pub fn is_authenticated(&self) -> bool {
        self.key_id.is_some()
    }

    /// The attribution string for events and logs: the key id, or
    /// `"anonymous"` when unauthenticated.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.key_id.as_deref().unwrap_or(ANONYMOUS_CALLER)
    }
}

impl std::fmt::Display for CallerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RequestMeta {
    pub(crate) request_id: RequestId,
    pub(crate) caller: CallerId,
    pub(crate) connect_info: Option<SocketAddr>,
}

impl<S> FromRequestParts<S> for RequestMeta
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> + Send {
        let request_id = parts
            .extensions
            .get::<RequestId>()
            .cloned()
            .unwrap_or_else(|| RequestId("unknown".to_owned()));
        let caller = parts
            .extensions
            .get::<CallerId>()
            .cloned()
            .unwrap_or_else(CallerId::anonymous);
        let connect_info = parts
            .extensions
            .get::<ConnectInfo<SocketAddr>>()
            .map(|connect_info| connect_info.0);
        async move {
            Ok(Self {
                request_id,
                caller,
                connect_info,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crlf_header_values_cannot_exist_at_the_http_layer() {
        // The selector re-checks for CR/LF as hardening, but the HTTP layer
        // itself refuses to build such a value, so header injection cannot
        // reach the selector's input in the first place.
        axum::http::HeaderValue::from_str("ok\r\ninjected: 1").unwrap_err();
    }

    fn cheap_phc(token: &str, salt: &[u8]) -> String {
        let params = argon2::Params::new(8, 1, 1, None).expect("valid test params");
        let hasher = argon2::Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            params.clone(),
        );
        use argon2::CustomizedPasswordHasher;
        hasher
            .hash_password_customized(
                token.as_bytes(),
                salt,
                Some(&argon2::ARGON2ID_IDENT),
                None,
                params,
            )
            .expect("hash token")
            .to_string()
    }

    fn auth_config_with(phc: &str) -> llm_proxy_core::AuthConfig {
        let mut clients = HashMap::new();
        clients.insert(
            "laptop".to_owned(),
            llm_proxy_core::ClientKeySpec::Verifier(SecretString::from(phc.to_owned())),
        );
        llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
            legacy_bare_tokens: false,
        }
    }

    #[test]
    fn client_auth_from_config_rejects_mode_contradictions() {
        let none_with_clients = {
            let mut clients = HashMap::new();
            clients.insert(
                "laptop".to_owned(),
                llm_proxy_core::ClientKeySpec::Verifier(SecretString::from(cheap_phc(
                    "secret",
                    b"0123456789abcdef",
                ))),
            );
            llm_proxy_core::AuthConfig {
                mode: llm_proxy_core::AuthMode::None,
                clients,
                admin: None,
                legacy_bare_tokens: false,
            }
        };
        assert!(matches!(
            ClientAuth::from_config(&none_with_clients),
            Err(ClientAuthError::ModeNoneWithClients)
        ));

        let keys_without_clients = llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients: HashMap::new(),
            admin: None,
            legacy_bare_tokens: false,
        };
        assert!(matches!(
            ClientAuth::from_config(&keys_without_clients),
            Err(ClientAuthError::ClientKeysWithoutClients)
        ));
    }

    /// A managed-key store can be a deployment's only key source: with
    /// one configured, client-keys mode must start even with no keys
    /// loaded yet (a fresh store, or the last key revoked) and fail
    /// closed per request — otherwise the daemon could never bootstrap
    /// its first key through the admin listener.
    #[tokio::test]
    async fn client_keys_with_a_key_store_starts_empty_and_fails_closed() {
        let config = llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients: HashMap::new(),
            admin: Some(llm_proxy_core::AdminAuthConfig {
                token: SecretString::from("admin-tok".to_owned()),
                key_store: Some(std::path::PathBuf::from("managed-keys.db")),
            }),
            legacy_bare_tokens: false,
        };
        let auth = ClientAuth::from_config(&config).expect("boots with an empty store");
        assert!(auth.is_enforced(), "the mode is still enforced");
        assert!(
            auth.authenticate("anything.secret").await.is_none(),
            "every request fails closed until a key is issued"
        );
    }

    #[test]
    fn client_auth_from_config_rejects_bad_key_ids_and_hashes() {
        for bad_id in ["", "anonymous"] {
            let mut config = auth_config_with(&cheap_phc("secret", b"0123456789abcdef"));
            config.clients.remove("laptop");
            config.clients.insert(
                bad_id.to_owned(),
                llm_proxy_core::ClientKeySpec::Verifier(SecretString::from("x".to_owned())),
            );
            assert!(
                matches!(
                    ClientAuth::from_config(&config),
                    Err(ClientAuthError::InvalidKeyId(id)) if id == bad_id
                ),
                "key id {bad_id:?} must be rejected"
            );
        }

        let config = auth_config_with("not-a-phc-string");
        let error = ClientAuth::from_config(&config).expect_err("malformed verifier rejected");
        assert!(
            error.to_string().contains("laptop"),
            "error names the key id, got: {error}"
        );
        assert!(
            !error.to_string().contains("not-a-phc-string"),
            "error must not echo the verifier, got: {error}"
        );
    }

    #[tokio::test]
    async fn client_auth_authenticates_a_configured_key_and_rejects_others() {
        let secret = cheap_phc("the-real-secret", b"0123456789abcdef");
        let auth = ClientAuth::from_config(&auth_config_with(&secret)).expect("valid config");

        let matched = auth.authenticate("laptop.the-real-secret").await;
        assert_eq!(matched.as_deref(), Some("laptop"));
        assert_eq!(
            auth.authenticate("laptop.the-real-secret").await.as_deref(),
            Some("laptop")
        );
        assert_eq!(auth.authenticate("laptop.wrong-secret").await, None);
        assert_eq!(auth.authenticate("").await, None);
    }

    #[tokio::test]
    async fn client_auth_picks_the_matching_key_among_many() {
        let mut clients = HashMap::new();
        for (id, token) in [("alpha", "one"), ("beta", "two"), ("gamma", "three")] {
            clients.insert(
                id.to_owned(),
                llm_proxy_core::ClientKeySpec::Verifier(SecretString::from(cheap_phc(
                    token,
                    b"0123456789abcdef",
                ))),
            );
        }
        let config = llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
            legacy_bare_tokens: false,
        };
        let auth = ClientAuth::from_config(&config).expect("valid config");
        assert_eq!(auth.authenticate("beta.two").await.as_deref(), Some("beta"));
        assert_eq!(
            auth.authenticate("gamma.three").await.as_deref(),
            Some("gamma")
        );
        assert_eq!(auth.authenticate("ghost.four").await, None);
    }

    /// An authenticator over `count` keys, each with its own cheap PHC.
    async fn auth_with_keys(count: usize) -> ClientAuth {
        auth_with_keys_opt_in(count, false).await
    }

    /// The same key set with `[auth] legacy_bare_tokens` set to `opt_in`.
    async fn auth_with_keys_opt_in(count: usize, opt_in: bool) -> ClientAuth {
        let mut clients = HashMap::new();
        clients.insert(
            "laptop".to_owned(),
            llm_proxy_core::ClientKeySpec::Verifier(SecretString::from(cheap_phc(
                "the-real-secret",
                b"0123456789abcdef",
            ))),
        );
        for index in 0..count.saturating_sub(1) {
            clients.insert(
                format!("key-{index:03}"),
                llm_proxy_core::ClientKeySpec::Verifier(SecretString::from(cheap_phc(
                    format!("secret-{index}").as_str(),
                    b"0123456789abcdef",
                ))),
            );
        }
        let config = llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
            legacy_bare_tokens: opt_in,
        };
        ClientAuth::from_config(&config).expect("valid config")
    }

    /// What one failed authentication cost in argon2 runs.
    async fn bad_token_cost(auth: &ClientAuth, token: &str) -> u64 {
        let before = auth.argon2_verifications();
        assert_eq!(auth.authenticate(token).await, None, "token must not pass");
        auth.argon2_verifications() - before
    }

    #[tokio::test]
    async fn a_bad_prefixed_token_costs_the_same_no_matter_the_key_count() {
        for count in [3, 60] {
            let auth = auth_with_keys(count).await;
            assert_eq!(
                bad_token_cost(&auth, "ghost.wrong-secret").await,
                0,
                "an unknown key id must not run argon2 at all ({count} keys)"
            );
            assert_eq!(
                bad_token_cost(&auth, "laptop.wrong-secret").await,
                1,
                "a known id names exactly one verifier ({count} keys)"
            );
        }
    }

    /// A bare token (no `<key-id>.` prefix) must be refused without
    /// running any verifier: its lookup cost must not grow with the
    /// number of configured keys.
    #[tokio::test]
    async fn a_bare_token_costs_no_argon2_by_default() {
        for count in [3, 60] {
            let auth = auth_with_keys(count).await;
            assert_eq!(
                bad_token_cost(&auth, "no-dot-wrong").await,
                0,
                "a bare token must cost zero argon2 runs, not one per key ({count} keys)"
            );
        }
    }

    /// One bad token verified twice (e.g. a client retrying) must cost one
    /// argon2 run in total, not one per attempt.
    #[tokio::test]
    async fn a_repeated_bad_token_is_not_verified_twice() {
        let auth = auth_with_keys(3).await;
        let before = auth.argon2_verifications();
        assert_eq!(auth.authenticate("laptop.wrong").await, None);
        assert_eq!(auth.authenticate("laptop.wrong").await, None);
        assert_eq!(
            auth.argon2_verifications() - before,
            1,
            "the repeat must be answered without another argon2 run"
        );
    }

    /// With `legacy_bare_tokens = true`, the bare secret shape keeps
    /// working: the real bare secret authenticates and a wrong bare token
    /// costs at most one run per key (bounded by the configured count).
    #[tokio::test]
    async fn a_bare_token_with_the_opt_in_authenticates() {
        let auth = auth_with_keys_opt_in(3, true).await;
        assert_eq!(
            auth.authenticate("the-real-secret").await.as_deref(),
            Some("laptop"),
            "the opt-in keeps the legacy bare-secret shape working"
        );
        assert!(
            bad_token_cost(&auth, "no-such-secret").await <= 3,
            "the opt-in scan checks every key but never more"
        );
    }

    /// The opt-in scan is capped: with more keys than the cap, a bare
    /// token is refused without running any verifier.
    #[tokio::test]
    async fn the_legacy_scan_is_capped() {
        let auth = auth_with_keys_opt_in(60, true).await;
        assert_eq!(
            bad_token_cost(&auth, "no-dot-wrong").await,
            0,
            "over the cap the legacy scan must be skipped entirely"
        );
    }

    /// A rebuilt `ClientAuth` — what a hot-reload key swap constructs —
    /// owns a fresh negative cache, so a token that failed against the
    /// old instance authenticates once its key exists.
    #[tokio::test]
    async fn a_rebuilt_client_auth_is_not_blocked_by_a_stale_failure() {
        let old_auth = auth_with_keys_opt_in(3, true).await;
        assert_eq!(
            old_auth.authenticate("laptop.not-yet-issued").await,
            None,
            "control: the token fails before the key exists"
        );

        let mut clients = HashMap::new();
        clients.insert(
            "laptop".to_owned(),
            llm_proxy_core::ClientKeySpec::Verifier(SecretString::from(cheap_phc(
                "not-yet-issued",
                b"0123456789abcdef",
            ))),
        );
        let new_auth = ClientAuth::from_config(&llm_proxy_core::AuthConfig {
            mode: llm_proxy_core::AuthMode::ClientKeys,
            clients,
            admin: None,
            legacy_bare_tokens: false,
        })
        .expect("valid config");
        assert_eq!(
            new_auth
                .authenticate("laptop.not-yet-issued")
                .await
                .as_deref(),
            Some("laptop"),
            "a stale negative entry must not block a freshly built authenticator"
        );
    }

    #[tokio::test]
    async fn a_prefixed_token_with_the_right_secret_authenticates_and_caches() {
        let auth = auth_with_keys(30).await;
        let before = auth.argon2_verifications();
        assert_eq!(
            auth.authenticate("laptop.the-real-secret").await.as_deref(),
            Some("laptop")
        );
        assert_eq!(auth.argon2_verifications() - before, 1);
        // The digest cache answers the repeat without argon2.
        assert_eq!(
            auth.authenticate("laptop.the-real-secret").await.as_deref(),
            Some("laptop")
        );
        assert_eq!(auth.argon2_verifications() - before, 1);
    }

    #[test]
    fn prefixed_token_splitting_rules() {
        assert_eq!(
            split_prefixed_token("laptop.secret"),
            Some(("laptop", "secret"))
        );
        assert_eq!(
            split_prefixed_token("laptop.dotted.secret"),
            Some(("laptop", "dotted.secret"))
        );
        assert_eq!(split_prefixed_token("no-dot"), None);
        assert_eq!(split_prefixed_token(".leading"), None);
        assert_eq!(split_prefixed_token("trailing."), None);
    }

    #[test]
    fn client_auth_none_mode_is_not_enforced() {
        let auth =
            ClientAuth::from_config(&llm_proxy_core::AuthConfig::default()).expect("valid config");
        assert!(!auth.is_enforced());
    }

    #[test]
    fn client_auth_debug_shows_key_ids_but_never_verifiers() {
        let secret = cheap_phc("the-real-secret", b"0123456789abcdef");
        let auth = ClientAuth::from_config(&auth_config_with(&secret)).expect("valid config");
        let debug = format!("{auth:?}");
        assert!(debug.contains("laptop"), "got: {debug}");
        assert!(
            !debug.contains("argon2id") && !debug.contains(&secret),
            "verifier string must not appear in Debug, got: {debug}"
        );
    }

    #[test]
    fn constant_time_eq_behaves_as_equality() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }

    #[test]
    fn bearer_token_extracts_case_insensitive_scheme_and_trims() {
        use axum::http::HeaderMap;
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer abc123".parse().unwrap());
        assert_eq!(
            extract_bearer_token(&headers).map(|t| t.expose_secret().to_owned()),
            Some("abc123".to_owned())
        );

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "bearer abc123 ".parse().unwrap());
        assert_eq!(
            extract_bearer_token(&headers).map(|t| t.expose_secret().to_owned()),
            Some("abc123".to_owned())
        );
    }

    #[test]
    fn bearer_token_rejects_missing_other_scheme_and_empty() {
        use axum::http::HeaderMap;
        assert!(extract_bearer_token(&HeaderMap::new()).is_none());

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Basic dXNlcjpwYXNz".parse().unwrap());
        assert!(extract_bearer_token(&headers).is_none());

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer   ".parse().unwrap());
        assert!(extract_bearer_token(&headers).is_none());
    }

    #[test]
    fn caller_id_attribution_strings() {
        assert_eq!(CallerId::anonymous().as_str(), "anonymous");
        assert_eq!(CallerId::authenticated("laptop").as_str(), "laptop");
        assert!(!CallerId::anonymous().is_authenticated());
        assert!(CallerId::authenticated("laptop").is_authenticated());
        assert_eq!(CallerId::authenticated("laptop").to_string(), "laptop");
    }

    #[test]
    fn request_id_format() {
        let generator = RequestIdGenerator::new();
        let id = generator.next_id();
        assert!(id.starts_with("req-"));
        assert!(id.contains('-'));
    }

    #[test]
    fn request_id_monotonically_increasing() {
        let generator = RequestIdGenerator::new();
        let id1 = generator.next_id();
        let id2 = generator.next_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn rate_limiter_allows_under_limit() {
        let limiter = RateLimiter::new(100);
        for _ in 0..10 {
            assert!(limiter.is_allowed("127.0.0.1"));
        }
    }

    #[test]
    fn rate_limiter_different_ips_independent() {
        let limiter = RateLimiter::new(5);
        for _ in 0..5 {
            assert!(limiter.is_allowed("10.0.0.1"));
        }
        assert!(limiter.is_allowed("10.0.0.2"));
    }

    #[test]
    fn client_ip_trust_forwarded_headers_false_ignores_headers() {
        use axum::http::HeaderMap;
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        let connect_info = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let ip = super::get_client_ip(&headers, Some(&connect_info), false);
        assert_eq!(
            ip, "127.0.0.1",
            "should ignore X-Forwarded-For when trust=false"
        );
    }

    #[test]
    fn client_ip_trust_forwarded_headers_true_uses_headers() {
        use axum::http::HeaderMap;
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4".parse().unwrap());
        let connect_info = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let ip = super::get_client_ip(&headers, Some(&connect_info), true);
        assert_eq!(ip, "1.2.3.4", "should use X-Forwarded-For when trust=true");
    }

    #[test]
    fn rate_limiter_zero_limit_is_disabled() {
        let limiter = RateLimiter::new(0);
        for _ in 0..1_000 {
            assert!(limiter.is_allowed("127.0.0.1"));
        }
    }

    #[test]
    fn would_allow_is_false_once_the_peer_bucket_is_drained() {
        let limiter = RateLimiter::new(1);
        assert!(
            limiter.is_allowed("10.0.0.9"),
            "first request consumes the only token"
        );
        assert!(
            !limiter.would_allow("10.0.0.9"),
            "drained peer must be denied by the peek"
        );
        assert!(limiter.would_allow("10.0.0.8"));
    }

    #[test]
    fn would_allow_reports_true_for_unknown_peer_and_disabled_limiter() {
        let limiter = RateLimiter::new(100);
        assert!(
            limiter.would_allow("10.0.0.7"),
            "unknown peer starts with a full bucket"
        );

        let disabled = RateLimiter::new(0);
        for _ in 0..50 {
            assert!(disabled.is_allowed("10.0.0.7"));
        }
        assert!(
            disabled.would_allow("10.0.0.7"),
            "a disabled limiter never denies"
        );
    }

    #[test]
    fn would_allow_does_not_consume_tokens() {
        let limiter = RateLimiter::new(1);
        assert!(limiter.is_allowed("10.0.0.6"));
        assert!(!limiter.would_allow("10.0.0.6"));
        assert!(!limiter.would_allow("10.0.0.6"));
        assert!(!limiter.is_allowed("10.0.0.6"));
    }

    #[test]
    fn rate_limiter_rejects_requests_at_the_limit() {
        let limiter = RateLimiter::new(1);
        assert!(
            limiter.is_allowed("10.0.0.1"),
            "first request should be allowed"
        );
        assert!(
            !limiter.is_allowed("10.0.0.1"),
            "second request should be rejected at 1 RPM limit"
        );
    }

    #[test]
    fn client_ip_multi_value_xff_picks_leftmost() {
        use axum::http::HeaderMap;
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "1.2.3.4, 5.6.7.8".parse().unwrap());
        let connect_info = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let ip = super::get_client_ip(&headers, Some(&connect_info), true);
        assert_eq!(
            ip, "1.2.3.4",
            "should pick the leftmost IP from multi-value XFF"
        );
    }

    #[test]
    fn client_ip_returns_unknown_when_no_connect_info_and_no_headers() {
        use axum::http::HeaderMap;

        let headers = HeaderMap::new();
        let ip = super::get_client_ip(&headers, None, false);
        assert_eq!(
            ip, "unknown",
            "should return 'unknown' when no IP source is available"
        );
    }

    #[test]
    fn client_ip_trusted_xff_with_invalid_ip_falls_through() {
        use axum::http::HeaderMap;
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let mut headers = HeaderMap::new();
        headers.insert("x-forwarded-for", "not-an-ip".parse().unwrap());
        let connect_info = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 8080);

        let ip = super::get_client_ip(&headers, Some(&connect_info), true);
        assert_eq!(
            ip, "10.0.0.1",
            "invalid XFF value should fall through to connect_info"
        );
    }

    #[test]
    fn client_ip_x_real_ip_is_used_when_xff_absent() {
        use axum::http::HeaderMap;
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let mut headers = HeaderMap::new();
        headers.insert("x-real-ip", "9.8.7.6".parse().unwrap());
        let connect_info = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080);

        let ip = super::get_client_ip(&headers, Some(&connect_info), true);
        assert_eq!(ip, "9.8.7.6", "should use X-Real-Ip when XFF is absent");
    }
}
