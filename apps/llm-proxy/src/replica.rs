//! Replica-store wiring (plan 09 step 1, audit conflict 13): when
//! `[replica]` is configured, the daemon's admission step, cooldown view
//! and usage feed move to the shared store, so every replica pointing at
//! it enforces one policy. The optional crates are behind cargo features;
//! a configured `[replica]` in a build without the matching feature is a
//! startup error, never a silent single-node fallback.

use anyhow::Result;
use llm_proxy_core::{AppConfig, ProviderRegistry, ReplicaBackend};
use llm_proxy_server::AppState;
use secrecy::ExposeSecret;
use std::sync::Arc;

/// Applies `[replica]` to `state`, returning the replacement state and,
/// when a replica store took over admission, the installer that rebuilds
/// its rules from a merged `[auth]` (the managed-key refresh swaps those
/// in over the shared counters).
///
/// # Errors
///
/// `anyhow` when `[replica]` names a backend this build cannot serve (the
/// feature is off), or connecting to the shared store fails.
pub async fn apply_replica(
    config: &AppConfig,
    registry: &ProviderRegistry,
    state: AppState,
    pair: Option<Arc<llm_proxy_server::SwappingAuthPair>>,
) -> Result<(AppState, Option<crate::state::AdmissionInstaller>)> {
    let Some(replica) = config.replica.as_ref() else {
        return Ok((state, None));
    };
    let url = replica.url.expose_secret();
    match replica.backend {
        ReplicaBackend::Redis => redis_replica(url, &replica.prefix, registry, state, pair).await,
        ReplicaBackend::Postgres => postgres_replica(url, registry, state, pair).await,
    }
}

/// The Redis arm of [`apply_replica`]: shared admission (over the store's
/// own counters), cooldowns and usage feed, plus the rules-rebuild
/// installer the managed-key refresh swaps in.
#[cfg(feature = "store-redis")]
async fn redis_replica(
    url: &str,
    prefix: &str,
    registry: &ProviderRegistry,
    state: AppState,
    pair: Option<Arc<llm_proxy_server::SwappingAuthPair>>,
) -> Result<(AppState, Option<crate::state::AdmissionInstaller>)> {
    let store = llm_proxy_store_redis::RedisStore::connect(url, prefix)
        .await
        .map_err(|error| {
            anyhow::anyhow!("connecting to the Redis replica store from [replica]: {error}")
        })?;
    // The lease crash bound: a replica that dies before settling must not
    // hold its reservation forever, so this timer reaps every lease past
    // its deadline (mirroring the Postgres purge timer below).
    llm_proxy_store_redis::spawn_reaper(store.clone());
    let admission = llm_proxy_store_redis::replica_admission_over_config(
        &state.app_config().auth,
        registry,
        store.clone(),
    );
    let cooldowns = Arc::new(llm_proxy_store_redis::RedisCooldowns::new(store.clone()));
    let bus = Arc::new(llm_proxy_store_redis::RedisEventBus::new(
        store.clone(),
        "usage",
    ));
    tracing::info!(
        prefix = %prefix,
        "replica mode: admission, cooldowns and usage shared through Redis"
    );
    let state = install_replica_admission(state, pair, admission)
        .with_cooldowns(cooldowns)
        .with_event_bus(bus);
    let installer = replica_admission_installer(state.clone(), move |auth, providers| {
        llm_proxy_store_redis::replica_admission_over_config(auth, providers, store.clone())
    });
    Ok((state, Some(installer)))
}

/// The Redis arm in a build without the `store-redis` feature: a startup
/// refusal naming the feature and the rebuild fix, never a silent
/// single-node fallback.
#[cfg(not(feature = "store-redis"))]
async fn redis_replica(
    _url: &str,
    _prefix: &str,
    _registry: &ProviderRegistry,
    _state: AppState,
    _pair: Option<Arc<llm_proxy_server::SwappingAuthPair>>,
) -> Result<(AppState, Option<crate::state::AdmissionInstaller>)> {
    anyhow::bail!(
        "[replica] backend \"redis\" is configured but this build lacks the \
         store-redis feature; rebuild with --features store-redis"
    );
}

/// The Postgres arm of [`apply_replica`]: the same wiring over the
/// Postgres store, plus the stale-row purge timer.
#[cfg(feature = "store-postgres")]
async fn postgres_replica(
    url: &str,
    registry: &ProviderRegistry,
    state: AppState,
    pair: Option<Arc<llm_proxy_server::SwappingAuthPair>>,
) -> Result<(AppState, Option<crate::state::AdmissionInstaller>)> {
    let store = llm_proxy_store_postgres::PostgresStore::connect(url)
        .await
        .map_err(|error| {
            anyhow::anyhow!("connecting to the Postgres replica store from [replica]: {error}")
        })?;
    spawn_purge_timer(store.clone());
    let admission = llm_proxy_store_postgres::replica_admission_over_config(
        &state.app_config().auth,
        registry,
        store.clone(),
    );
    let cooldowns = Arc::new(llm_proxy_store_postgres::PostgresCooldowns::new(
        store.clone(),
    ));
    let bus = Arc::new(llm_proxy_store_postgres::PostgresEventBus::new(
        store.clone(),
    ));
    tracing::info!("replica mode: admission, cooldowns and usage shared through Postgres");
    let state = install_replica_admission(state, pair, admission)
        .with_cooldowns(cooldowns)
        .with_event_bus(bus);
    let installer = replica_admission_installer(state.clone(), move |auth, providers| {
        llm_proxy_store_postgres::replica_admission_over_config(auth, providers, store.clone())
    });
    Ok((state, Some(installer)))
}

/// The Postgres arm in a build without the `store-postgres` feature: the
/// same startup refusal posture as the Redis arm.
#[cfg(not(feature = "store-postgres"))]
async fn postgres_replica(
    _url: &str,
    _registry: &ProviderRegistry,
    _state: AppState,
    _pair: Option<Arc<llm_proxy_server::SwappingAuthPair>>,
) -> Result<(AppState, Option<crate::state::AdmissionInstaller>)> {
    anyhow::bail!(
        "[replica] backend \"postgres\" is configured but this build lacks the \
         store-postgres feature; rebuild with --features store-postgres"
    );
}

/// Installs the replica-shared admission step: into the daemon's swap
/// pair when one was passed (the pair is already wired into both the
/// router's auth check and the engine, so the managed-key refresh can
/// replace the rules later), or directly as the engine override otherwise.
#[cfg(any(feature = "store-redis", feature = "store-postgres"))]
fn install_replica_admission(
    state: AppState,
    pair: Option<Arc<llm_proxy_server::SwappingAuthPair>>,
    admission: llm_proxy_engine::PolicyAdmission,
) -> AppState {
    match pair {
        Some(pair) => {
            pair.replace_admission(Arc::new(admission) as Arc<dyn llm_proxy_engine::Admission>);
            state
        }
        None => state.with_admission(Arc::new(admission) as Arc<dyn llm_proxy_engine::Admission>),
    }
}

/// The rules-rebuild closure over the shared store: the counters live in
/// the store, so a rebuilt step keeps enforcing the same windows and
/// budgets between replicas.
#[cfg(any(feature = "store-redis", feature = "store-postgres"))]
fn replica_admission_installer<
    F: Fn(&llm_proxy_core::AuthConfig, &ProviderRegistry) -> llm_proxy_engine::PolicyAdmission
        + Send
        + Sync
        + 'static,
>(
    state: AppState,
    rebuild: F,
) -> crate::state::AdmissionInstaller {
    Box::new(move |auth| {
        let providers = state.providers();
        Arc::new(rebuild(auth, &providers)) as Arc<dyn llm_proxy_engine::Admission>
    })
}

/// The Postgres store keeps stale window rows until purged; a timer bounds
/// them (Redis keys expire on their own TTLs).
#[cfg(feature = "store-postgres")]
fn spawn_purge_timer(store: llm_proxy_store_postgres::PostgresStore) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(600));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            match store.purge_expired().await {
                Ok(purged) if purged > 0 => {
                    tracing::debug!(purged, "replica store purge removed stale rows");
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(%error, "replica store purge failed");
                }
            }
        }
    });
}

#[cfg(not(all(feature = "store-redis", feature = "store-postgres")))]
#[cfg(test)]
mod tests {
    use super::*;
    use llm_proxy_adapters::ProviderAdapterRegistry;
    use llm_proxy_transport::ProxyClient;

    /// A `[replica]` config shaped like the one
    /// `tests/redis_replica_managed_keys.rs` boots against, pointed at a
    /// URL with no store behind it: the feature-off refusal must fire
    /// before any connection is attempted.
    fn replica_config(backend: ReplicaBackend, url: &str) -> AppConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let backend = match backend {
            ReplicaBackend::Redis => "redis",
            ReplicaBackend::Postgres => "postgres",
        };
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3463"
request_timeout = "300s"
log_level = "info"
server_name = "replica-refusal"

[replica]
backend = "{backend}"
url = "{url}"
prefix = "llm-proxy"
"#
            ),
        )
        .unwrap();
        llm_proxy_core::load_app_config(&path).unwrap()
    }

    /// The state exactly as `load_toml_state_from_config` builds it before
    /// it calls [`apply_replica`]: the config's phase-shaped client, an
    /// empty registry, the builtin adapters.
    fn replica_state(config: &AppConfig, registry: &ProviderRegistry) -> AppState {
        let proxy_client =
            ProxyClient::try_with_timeouts(&config.server.upstream_timeouts).unwrap();
        AppState::new_with_catalog_dir(
            config.clone(),
            registry.clone(),
            ProviderAdapterRegistry::builtin(),
            proxy_client,
            crate::state::build_info(),
            None,
        )
        .unwrap()
    }

    /// A `[replica]` naming Redis in a build without store-redis is a
    /// startup refusal that names the feature and the rebuild fix — never
    /// a silent single-node fallback.
    #[cfg(not(feature = "store-redis"))]
    #[tokio::test]
    async fn a_redis_replica_without_the_feature_refuses_to_start() {
        let config = replica_config(ReplicaBackend::Redis, "redis://127.0.0.1:1");
        let registry = ProviderRegistry::from_providers([]).unwrap();
        let state = replica_state(&config, &registry);

        let result = apply_replica(&config, &registry, state, None).await;
        match result {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("store-redis")
                        && message.contains("rebuild with --features store-redis"),
                    "the refusal names the feature and the rebuild fix, got: {message}"
                );
            }
            Ok(_) => panic!("a redis [replica] must never start single-node silently"),
        }
    }

    /// The same posture for Postgres: without store-postgres the daemon
    /// refuses to start rather than serving single-node.
    #[cfg(not(feature = "store-postgres"))]
    #[tokio::test]
    async fn a_postgres_replica_without_the_feature_refuses_to_start() {
        let config = replica_config(ReplicaBackend::Postgres, "postgres://127.0.0.1:1/llmproxy");
        let registry = ProviderRegistry::from_providers([]).unwrap();
        let state = replica_state(&config, &registry);

        let result = apply_replica(&config, &registry, state, None).await;
        match result {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("store-postgres")
                        && message.contains("rebuild with --features store-postgres"),
                    "the refusal names the feature and the rebuild fix, got: {message}"
                );
            }
            Ok(_) => panic!("a postgres [replica] must never start single-node silently"),
        }
    }
}
