//! Application state construction and tracing initialization.
//!
//! Provides helpers to load TOML config into `AppState`, build `BuildInfo`,
//! check catalog enforcement, and initialize the tracing subscriber.

use std::path::Path;

use anyhow::{Context, Result};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{
    AppConfig, LogFormat, ProviderConfig, ProviderRegistry, merge_catalog, parse_catalog_file,
};
use llm_proxy_server::{AppState, BuildInfo};
use llm_proxy_transport::ProxyClient;
use tracing::info;

use crate::paths::providers_dir_for;

/// Test helper: load config from `path` then build the state. The serve path
/// loads the config itself (before tracing init, BUG-25) and calls
/// [`load_toml_state_from_config`].
#[cfg(test)]
pub(crate) async fn load_toml_state(
    path: &Path,
    adapter_registry: ProviderAdapterRegistry,
    port_override: Option<u16>,
) -> Result<AppState> {
    let mut app_config: AppConfig = llm_proxy_core::load_app_config(path)
        .with_context(|| format!("loading TOML config from {}", path.display()))?;

    if let Some(p) = port_override {
        let mut bind = app_config.server.bind;
        bind.set_port(p);
        app_config.server.bind = bind;
    }

    load_toml_state_from_config(app_config, path, adapter_registry).await
}

/// Opens the managed-key store when `[auth.admin.key_store]` is
/// configured (plan 09 step 2). Without the `admin-ui` feature a
/// configured store is a startup error, never a silent skip.
#[cfg(feature = "admin-ui")]
pub(crate) fn managed_key_store_for_serve(
    config: &AppConfig,
) -> Option<std::sync::Arc<llm_proxy_admin::ManagedKeys>> {
    match managed_key_store(config) {
        Ok(store) => store,
        Err(error) => {
            tracing::error!(
                %error,
                "opening the managed-key store for the admin surface failed; key views are off"
            );
            None
        }
    }
}

#[cfg(feature = "admin-ui")]
fn managed_key_store(
    config: &AppConfig,
) -> Result<Option<std::sync::Arc<llm_proxy_admin::ManagedKeys>>> {
    match config
        .auth
        .admin
        .as_ref()
        .and_then(|admin| admin.key_store.clone())
    {
        Some(path) => {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("creating the managed-key store parent {}", parent.display())
                })?;
            }
            let store = llm_proxy_admin::ManagedKeys::open(&path)
                .with_context(|| format!("opening the managed-key store at {}", path.display()))?;
            tracing::info!(store = %path.display(), "managed client keys enabled");
            Ok(Some(std::sync::Arc::new(store)))
        }
        None => Ok(None),
    }
}

/// Build the serving state from an already-loaded config: the path the serve
/// command takes, so it can initialize the tracing subscriber from the
/// config's `server.log_format` before anything logs through it (BUG-25).
pub async fn load_toml_state_from_config(
    app_config: AppConfig,
    path: &Path,
    adapter_registry: ProviderAdapterRegistry,
) -> Result<AppState> {
    // Phase-shaped upstream timeouts (backlog 8, BUG-14): the configured
    // phases replace the fixed connect/headers/idle constants.
    let proxy_client = ProxyClient::try_with_timeouts(&app_config.server.upstream_timeouts)
        .with_context(|| "building proxy HTTP client from [server.upstream_timeouts]")?;

    let registry = load_registry_with_checks(&app_config, path, &adapter_registry)?;

    let providers_dir = providers_dir_for(path);
    let cache_dir = providers_dir.join(".catalog");
    for provider in registry.iter() {
        if let Some(warning) = catalog_enforcement_warning(provider, &cache_dir) {
            tracing::warn!(provider = %provider.name, "{warning}");
        }
    }

    // Managed keys (plan 09 step 2): when [auth.admin.key_store] is set
    // (and the admin-ui feature is compiled in), the store's live keys
    // join the file's [auth.clients] (`merged_auth_config` states the merge
    // rules).
    #[cfg(feature = "admin-ui")]
    let managed = managed_key_source(&app_config)?;
    #[cfg(feature = "admin-ui")]
    let app_config = match &managed {
        Some(source) => merged_auth_config(app_config, &source.specs),
        None => app_config,
    };
    // A corrupt verifier must still refuse startup (AppState's own build
    // fails closed with a log instead, which is right for the library and
    // wrong for a daemon).
    llm_proxy_server::ClientAuth::from_config(&app_config.auth)
        .with_context(|| "building client auth from [auth] config")?;

    let egress_clients = build_egress_clients(&app_config, &registry)?;

    let event_bus = build_event_bus(&app_config)?;

    info!(
        config = %path.display(),
        providers = registry.len(),
        "loaded TOML config"
    );

    warn_admin_posture(&app_config);

    // [server.semantic_cache] with the feature off is a startup error,
    // never a silent skip (the same posture as [replica]).
    #[cfg(not(feature = "semantic-cache"))]
    if app_config.server.semantic_cache.enabled {
        anyhow::bail!(
            "[server.semantic_cache] is enabled but this build lacks the semantic-cache \
             feature; rebuild with --features semantic-cache"
        );
    }

    // [auth.admin] key_store with the feature off is a startup error,
    // never a silent skip (the same posture as [replica] and
    // semantic-cache).
    #[cfg(not(feature = "admin-ui"))]
    refuse_key_store_without_admin_ui(&app_config)?;

    let base = AppState::new_with_catalog_dir(
        app_config.clone(),
        registry.clone(),
        adapter_registry,
        proxy_client,
        build_info(),
        Some(cache_dir),
    )
    .with_egress_clients(egress_clients)
    .with_event_bus(event_bus);

    // A daemon on a managed key store enforces through ONE swappable
    // (authenticator, admission rules) pair: the refresh loop (below)
    // replaces both halves under a single write guard when the store's
    // live key set changes, so a key issued or revoked at runtime is
    // enforced by the running server without a restart — same rules shape
    // as an equivalent file key, and the counters survive the swap.
    #[cfg(feature = "admin-ui")]
    let state = match managed {
        Some(source) => managed_key_state(&app_config, &registry, base, source).await?,
        None => {
            let base = seeded_single_node_admission(&app_config, &registry, base).await;
            crate::replica::apply_replica(&app_config, &registry, base, None)
                .await?
                .0
        }
    };
    #[cfg(not(feature = "admin-ui"))]
    let state = {
        let base = seeded_single_node_admission(&app_config, &registry, base).await;
        crate::replica::apply_replica(&app_config, &registry, base, None)
            .await?
            .0
    };

    Ok(state)
}

/// The managed-key arm of [`load_toml_state_from_config`]: one swappable
/// (authenticator, admission rules) pair over an app-owned backend, seeded
/// from the usage ledger when `[storage]` is set on a single node, plus the
/// refresh loop that rebuilds both halves when the store's live key set
/// changes. The seeded counters survive those swaps: every rebuild runs
/// over this same backend.
#[cfg(feature = "admin-ui")]
async fn managed_key_state(
    app_config: &AppConfig,
    registry: &ProviderRegistry,
    base: AppState,
    source: ManagedKeySource,
) -> Result<AppState> {
    let memory_backend = std::sync::Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new());
    if let (Some(storage), None) = (&app_config.storage, &app_config.replica) {
        seed_day_budgets(&memory_backend, &storage.path).await;
    }
    let backend = memory_backend as std::sync::Arc<dyn llm_proxy_engine::AdmissionBackend>;
    // The same merged `[auth]` the state itself validated above; re-parsed
    // so the pair owns its authenticator half from the start (the parse is
    // PHC-string work, no hashing).
    let initial_auth = llm_proxy_server::ClientAuth::from_config(&app_config.auth)
        .context("rebuilding client auth for the swap pair")?;
    let pair = std::sync::Arc::new(llm_proxy_server::SwappingAuthPair::new(
        std::sync::Arc::new(initial_auth),
        std::sync::Arc::new(llm_proxy_engine::PolicyAdmission::over_config_backend(
            &app_config.auth,
            registry,
            std::sync::Arc::clone(&backend),
        )),
    ));
    // The startup merge added the store's ids to the running config;
    // record them so a hot reload can attribute the ids the file never had
    // instead of blaming the file's own clients.
    let base = base
        .with_runtime_issued_keys(source.specs.iter().map(|(id, _)| id.clone()).collect())
        .with_auth_pair(std::sync::Arc::clone(&pair));
    let (state, replica_installer) = crate::replica::apply_replica(
        app_config,
        registry,
        base,
        Some(std::sync::Arc::clone(&pair)),
    )
    .await?;
    let installer: AdmissionInstaller = match replica_installer {
        Some(replica) => replica,
        None => single_node_admission_installer(state.clone(), backend),
    };
    spawn_managed_key_refresh(
        source.store,
        source.file_auth,
        source.specs,
        pair,
        installer,
    );
    Ok(state)
}

/// The startup merge (plan 09 step 2): the store's live keys join the
/// file's `[auth.clients]` — in the AuthConfig every consumer reads, not
/// just the authenticator. ClientAuth verifies them, the engine's
/// PolicyAdmission enforces the policies pinned on them (windows, budgets,
/// allowlists) and the replica admissions carry them too. File keys win on
/// id collisions, so the store can never shadow an operator key. The
/// merge only runs in client-keys mode: `auth.mode = "none"` with a
/// configured store is refused earlier (see [`managed_key_source`]), so
/// the merge can never trip ClientAuth's mode-none-with-clients refusal
/// with store keys.
#[cfg(feature = "admin-ui")]
fn merged_auth_config(
    app_config: AppConfig,
    specs: &[(String, llm_proxy_core::ClientKeySpec)],
) -> AppConfig {
    let mut config = app_config;
    for (id, spec) in specs.iter().cloned() {
        config.auth.clients.entry(id).or_insert(spec);
    }
    config
}

/// `[auth.admin] key_store` with the feature off is a startup error, never
/// a silent skip: the store's keys could never load and no surface exists
/// to manage them (the same posture as `[replica]` and semantic-cache).
#[cfg(not(feature = "admin-ui"))]
fn refuse_key_store_without_admin_ui(app_config: &AppConfig) -> Result<()> {
    if app_config
        .auth
        .admin
        .as_ref()
        .is_some_and(|admin| admin.key_store.is_some())
    {
        anyhow::bail!(
            "[auth.admin] key_store is configured but this build lacks the admin-ui \
             feature; rebuild with --features admin-ui or remove key_store"
        );
    }
    Ok(())
}

/// Rebuilds the single-node admission rules from a merged `[auth]` over
/// the same in-process backend, so a refresh changes the rules without
/// resetting anyone's windows or budgets.
#[cfg(feature = "admin-ui")]
fn single_node_admission_installer(
    state: AppState,
    backend: std::sync::Arc<dyn llm_proxy_engine::AdmissionBackend>,
) -> AdmissionInstaller {
    Box::new(move |auth| {
        let providers = state.providers();
        std::sync::Arc::new(llm_proxy_engine::PolicyAdmission::over_config_backend(
            auth,
            &providers,
            std::sync::Arc::clone(&backend),
        ))
    })
}

/// Rebuilds the live admission rules from a merged `[auth]` (the file's
/// clients plus the store's live keys): same counters, fresh rules — what
/// the managed-key refresh swaps in.
pub(crate) type AdmissionInstaller = Box<
    dyn Fn(&llm_proxy_core::AuthConfig) -> std::sync::Arc<dyn llm_proxy_engine::Admission>
        + Send
        + Sync,
>;

/// The managed-key store a daemon runs on, opened once: the store handle,
/// the file's `[auth]` before any merge (the base every refresh re-merges
/// into, so a revoked key leaves the merged set instead of lingering
/// forever), and the live specs as read at startup.
#[cfg(feature = "admin-ui")]
struct ManagedKeySource {
    store: std::sync::Arc<llm_proxy_admin::ManagedKeys>,
    file_auth: llm_proxy_core::AuthConfig,
    specs: Vec<(String, llm_proxy_core::ClientKeySpec)>,
}

/// Opens the managed-key source for the serving state: `None` when no
/// store is configured. `auth.mode = "none"` with a store configured is a
/// startup error, not a warning — no client auth is enforced in that
/// mode, so the store's keys could never take effect and an operator
/// could believe they have auth on an open proxy (the same
/// never-a-silent-skip posture as a key store without the admin-ui
/// feature).
///
/// # Errors
///
/// `anyhow` when `auth.mode = "none"` carries a configured `key_store`,
/// or when the store cannot be opened or read.
#[cfg(feature = "admin-ui")]
fn managed_key_source(config: &AppConfig) -> Result<Option<ManagedKeySource>> {
    if config.auth.mode == llm_proxy_core::AuthMode::None {
        if config
            .auth
            .admin
            .as_ref()
            .is_some_and(|admin| admin.key_store.is_some())
        {
            anyhow::bail!(
                "[auth.admin] key_store is configured but auth.mode is \"none\", where no \
                 client auth is enforced and the store's keys can never take effect; set \
                 auth.mode = \"client-keys\" to use them, or remove key_store"
            );
        }
        return Ok(None);
    }
    match managed_key_store(config)? {
        Some(store) => {
            let specs = store
                .client_specs()
                .with_context(|| "reading managed keys from [auth.admin.key_store]")?;
            Ok(Some(ManagedKeySource {
                store,
                file_auth: config.auth.clone(),
                specs,
            }))
        }
        None => Ok(None),
    }
}

/// How often the daemon re-reads the managed-key store: seconds, so a key
/// issued through the admin API is enforced roughly immediately — the
/// requirement is "without a restart", not "in the same request".
#[cfg(feature = "admin-ui")]
const MANAGED_KEY_REFRESH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// A comparable projection of the store's live key set: ids, verifiers
/// and policies. Verifier secrets appear here on purpose (change
/// detection must see a rotated PHC string) and never leave the process —
/// the fingerprint is compared, never logged.
#[cfg(feature = "admin-ui")]
fn managed_key_fingerprint(
    specs: &[(String, llm_proxy_core::ClientKeySpec)],
) -> Vec<(String, String)> {
    use llm_proxy_core::ClientKeySpec;
    use secrecy::ExposeSecret;
    let mut fingerprint: Vec<(String, String)> = specs
        .iter()
        .map(|(id, spec)| {
            let policy = match spec {
                ClientKeySpec::Verifier(_) => "verifier".to_owned(),
                ClientKeySpec::Policy {
                    allow_models,
                    allow_providers,
                    limits,
                    budget,
                    ..
                } => format!("{allow_models:?}|{allow_providers:?}|{limits:?}|{budget:?}"),
            };
            (
                id.clone(),
                format!("{}|{policy}", spec.verifier().expose_secret()),
            )
        })
        .collect();
    fingerprint.sort();
    fingerprint
}

/// A persistently invalid store would otherwise log the same ERROR once
/// per refresh tick; the first failure and then one per minute is loud
/// enough to act on.
#[cfg(feature = "admin-ui")]
fn should_log_invalid_refresh(consecutive: u32) -> bool {
    consecutive <= 1 || consecutive % 60 == 0
}

/// Watches the managed-key store and, when the live key set changed,
/// re-merges it over the file's `[auth]` and swaps the client
/// authenticator and the admission rules into the running state — both
/// halves of one pair under a single write guard, so no served instant
/// mixes the old key set with the new rules. A store read that fails
/// keeps the installed set; an invalid merged set keeps the installed set
/// too, loudly — both fail safe, never open.
///
/// The running config snapshot keeps the file's `[auth]` as merged at
/// startup; request-time authentication and admission go through the
/// swapped pair, never the snapshot. The hot-reload diff that still names
/// `auth.clients` because of the startup merge is attributed to the
/// runtime-issued ids through [`AppState::runtime_issued_keys`].
#[cfg(feature = "admin-ui")]
fn spawn_managed_key_refresh(
    store: std::sync::Arc<llm_proxy_admin::ManagedKeys>,
    file_auth: llm_proxy_core::AuthConfig,
    initial: Vec<(String, llm_proxy_core::ClientKeySpec)>,
    pair: std::sync::Arc<llm_proxy_server::SwappingAuthPair>,
    install: AdmissionInstaller,
) {
    tokio::spawn(async move {
        let mut installed = managed_key_fingerprint(&initial);
        let mut invalid_refreshes: u32 = 0;
        loop {
            tokio::time::sleep(MANAGED_KEY_REFRESH_INTERVAL).await;
            let specs = match store.client_specs() {
                Ok(specs) => specs,
                Err(error) => {
                    tracing::warn!(
                        %error,
                        "reading the managed-key store for the live refresh failed; \
                         the installed key set stands"
                    );
                    continue;
                }
            };
            let seen = managed_key_fingerprint(&specs);
            if seen == installed {
                continue;
            }
            let mut auth = file_auth.clone();
            for (id, spec) in &specs {
                auth.clients.entry(id.clone()).or_insert(spec.clone());
            }
            match llm_proxy_server::ClientAuth::from_config(&auth) {
                Ok(client_auth) => {
                    invalid_refreshes = 0;
                    // ONE write-guard swap installs the new authenticator
                    // and the new rules together: no instant exists where
                    // the router authenticates against the old key set
                    // while the engine already enforces the new rules (a
                    // revoked token judging as unknown-and-unrestricted),
                    // nor the mirror (a just-issued key riding the old
                    // permissive rules). The per-ClientAuth digest cache
                    // dies with the old half by design — preserving it
                    // would let a revoked token's cached digest survive —
                    // so one extra argon2 per live client per refresh is
                    // the price of correct revocation.
                    pair.replace(std::sync::Arc::new(client_auth), install(&auth));
                    installed = seen;
                    tracing::info!(
                        live_keys = specs.len(),
                        "the managed key set changed; authentication and policies \
                         refreshed without a restart"
                    );
                }
                Err(error) => {
                    invalid_refreshes += 1;
                    if should_log_invalid_refresh(invalid_refreshes) {
                        tracing::error!(
                            %error,
                            consecutive = invalid_refreshes,
                            "the refreshed managed key set is invalid; the previously \
                             installed key set stands"
                        );
                    }
                }
            }
        }
    });
}

/// Loads the provider and pool registry from the directory beside
/// `config_path` and runs every cross-check that needs both the app config
/// and the registry: known protocol names, self-loop refusal (backlog 32),
/// bare `/v1/*` target resolution, and the auth/passthrough combination.
/// Shared by startup and `llm-proxy validate`, so both refuse the same set.
pub fn load_registry_with_checks(
    app_config: &AppConfig,
    config_path: &Path,
    adapter_registry: &ProviderAdapterRegistry,
) -> Result<ProviderRegistry> {
    let providers_dir = providers_dir_for(config_path);
    let registry = if providers_dir.exists() {
        ProviderRegistry::load_from_dir(&providers_dir)
            .with_context(|| format!("loading providers from {}", providers_dir.display()))?
    } else {
        ProviderRegistry::from_providers(vec![])
            .with_context(|| "building empty provider registry")?
    };

    let known_protocols = adapter_registry.protocol_names();
    registry
        .validate_protocols(known_protocols)
        .with_context(|| "validating provider protocols")?;

    // Self-loop refusal (backlog 32): an upstream that resolves to one of
    // this instance's own listeners would forward to itself forever. The
    // listeners are the main bind plus the admin bind when one is served.
    let mut own_listeners = vec![app_config.server.bind];
    if app_config.admin_posture().is_some() && app_config.server.dedicated_admin_listener() {
        if let Some(admin_bind) = app_config.server.effective_admin_bind() {
            own_listeners.push(admin_bind);
        }
    }
    let loop_problems = llm_proxy_core::self_loop_problems(registry.iter(), &own_listeners);
    if !loop_problems.is_empty() {
        anyhow::bail!(
            "refusing to run with a self-referential upstream: {}",
            loop_problems.join("; ")
        );
    }

    // Bare `/v1/*` targets (plan 04 step 11) must name a loaded provider or
    // pool; the app config loads before provider files, so the cross-check
    // runs here, where both are known.
    let bare_problems =
        llm_proxy_core::bare_v1_target_problems(&app_config.server.bare_v1, |name| {
            registry.contains_route(name)
        });
    if !bare_problems.is_empty() {
        anyhow::bail!(
            "validating [server.bare_v1] against the loaded providers and pools: {}",
            bare_problems.join("; ")
        );
    }

    let providers: Vec<llm_proxy_core::ProviderConfig> = registry.iter().cloned().collect();
    llm_proxy_core::validate_passthrough_auth_combination(&app_config.auth, &providers)
        .with_context(|| "validating auth against provider passthrough configuration")?;

    // A family route (plan 09 step 7) in a build without the families
    // would answer 404 forever: a startup error, never a silent skip.
    #[cfg(not(feature = "endpoint-families"))]
    {
        let configured = providers_with_family_routes(&registry);
        if !configured.is_empty() {
            anyhow::bail!(
                "providers {} route a non-chat endpoint family (embeddings, rerank, audio, \
                 images, files or batches) but this build lacks the endpoint-families \
                 feature; rebuild with --features endpoint-families",
                configured.join(", ")
            );
        }
    }

    Ok(registry)
}

/// Providers whose `[provider.routes]` name a non-chat endpoint family.
#[cfg(not(feature = "endpoint-families"))]
fn providers_with_family_routes(registry: &ProviderRegistry) -> Vec<String> {
    use llm_proxy_core::ProviderRouteKind as Kind;
    const FAMILIES: [Kind; 7] = [
        Kind::Embeddings,
        Kind::Rerank,
        Kind::AudioTranscriptions,
        Kind::AudioSpeech,
        Kind::ImagesGenerations,
        Kind::Files,
        Kind::Batches,
    ];
    registry
        .iter()
        .filter(|provider| {
            FAMILIES
                .iter()
                .any(|kind| provider.routes.get(*kind).is_some())
        })
        .map(|provider| provider.name.clone())
        .collect()
}

/// Per-provider egress-proxy clients from `[provider.egress]` — also the
/// rebuild path for a hot reload (plan 07 step 11).
pub(crate) fn build_egress_clients(
    app_config: &AppConfig,
    registry: &ProviderRegistry,
) -> Result<std::collections::HashMap<String, std::sync::Arc<ProxyClient>>> {
    let mut egress_clients = std::collections::HashMap::new();
    for provider in registry.iter() {
        if let Some(egress) = &provider.egress {
            let client = ProxyClient::try_with_egress_proxy_and_timeouts(
                &egress.proxy,
                &app_config.server.upstream_timeouts,
            )
            .with_context(|| {
                format!(
                    "building egress proxy client for provider \"{}\" from [provider.egress]",
                    provider.name
                )
            })?;
            info!(
                provider = %provider.name,
                scheme = egress.proxy.split("://").next().unwrap_or_default(),
                "routing upstream traffic through the configured egress proxy"
            );
            egress_clients.insert(provider.name.clone(), std::sync::Arc::new(client));
        }
    }
    Ok(egress_clients)
}

/// The usage-ledger bus: SQLite when `[storage]` is configured, the no-op
/// bus otherwise.
fn build_event_bus(
    app_config: &AppConfig,
) -> Result<std::sync::Arc<dyn llm_proxy_storage::EventBus>> {
    match &app_config.storage {
        Some(storage) => {
            let bus = llm_proxy_storage::SqliteEventBus::open(
                &storage.path,
                llm_proxy_storage::RetentionPolicy {
                    max_age: storage.max_age,
                    max_rows: storage.max_rows,
                    prune_every: storage.prune_interval,
                },
            )
            .with_context(|| format!("opening usage ledger at {}", storage.path.display()))?;
            info!(
                path = %storage.path.display(),
                "persisting usage events to the SQLite ledger"
            );
            Ok(std::sync::Arc::new(bus))
        }
        None => Ok(std::sync::Arc::new(llm_proxy_storage::NoopBus)),
    }
}

/// Seeds the in-process admission backend's day windows from the usage
/// ledger (plan 05 step 5): today's spend and rpd counts per caller, read
/// from `requests` rows since 00:00 UTC, so a restart cannot reset a
/// `usd_per_day` budget or an rpd counter. Only the day window is seeded —
/// minute windows cover 60 seconds, shorter than any restart gap, so
/// rpm/tpm restart empty. A failed read only warns: the ledger is also what
/// `/usage` reads, so a broken store must not keep the proxy from serving.
/// Replica stores are durable already and skip this (their admission is
/// swapped in over a shared backend, which discards these counters).
async fn seed_day_budgets(backend: &llm_proxy_engine::MemoryAdmissionBackend, ledger: &Path) {
    let since_ms = time::OffsetDateTime::now_utc()
        .replace_time(time::Time::MIDNIGHT)
        .unix_timestamp()
        .saturating_mul(1000);
    let path = ledger.to_owned();
    let rows =
        match tokio::task::spawn_blocking(move || llm_proxy_storage::spend_since(&path, since_ms))
            .await
        {
            Ok(Ok(rows)) => rows,
            Ok(Err(error)) => {
                tracing::warn!(
                    %error,
                    "seeding daily budgets from the usage ledger failed; day budgets start at zero"
                );
                return;
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    "seeding daily budgets task failed; day budgets start at zero"
                );
                return;
            }
        };
    let callers = rows.len();
    let now = time::OffsetDateTime::now_utc();
    for row in &rows {
        let Some(caller) = row.caller.as_deref() else {
            continue;
        };
        backend.seed_day(
            now,
            caller,
            u64::try_from(row.requests).unwrap_or(u64::MAX),
            llm_proxy_engine::to_micro_usd(row.spend),
        );
    }
    info!(callers, "seeded daily budgets from the usage ledger");
}

/// Single-node admission over an app-owned backend so the day windows can
/// be seeded from the usage ledger before serving. Only the `[storage]`
/// single-node case deviates from the engine's own default: without
/// `[storage]` everything resets on restart anyway, and a `[replica]` store
/// is the durable counter already. Rules stay identical to the default
/// (`PolicyAdmission::from_config`).
async fn seeded_single_node_admission(
    app_config: &AppConfig,
    registry: &ProviderRegistry,
    base: AppState,
) -> AppState {
    let (Some(storage), None) = (&app_config.storage, &app_config.replica) else {
        return base;
    };
    let backend = std::sync::Arc::new(llm_proxy_engine::MemoryAdmissionBackend::new());
    let admission = llm_proxy_engine::PolicyAdmission::over_config_backend(
        &app_config.auth,
        registry,
        std::sync::Arc::clone(&backend) as std::sync::Arc<dyn llm_proxy_engine::AdmissionBackend>,
    );
    if admission.is_noop() {
        return base;
    }
    seed_day_budgets(&backend, &storage.path).await;
    base.with_admission(std::sync::Arc::new(admission))
}

/// Emit the startup warnings naming the admin `/usage` mount posture.
///
/// Warns when the admin surface is mounted on the main listener or when the
/// dedicated admin listener binds a non-loopback address.
pub fn warn_admin_posture(config: &AppConfig) {
    use llm_proxy_core::AdminMountPosture;
    match config.admin_posture() {
        Some(AdminMountPosture::MainListener) => {
            tracing::warn!(
                posture = AdminMountPosture::MainListener.name(),
                main_bind = %config.server.bind,
                "ADMIN SECURITY POSTURE: /usage is also served on the main listener, \
                 protected only by the admin token — every network that reaches the \
                 main listener reaches the admin API. Set server.admin_on_main_listener \
                 = false to return to the loopback-only dedicated admin listener."
            );
        }
        Some(AdminMountPosture::DedicatedListener) => {
            if let Some(bind) = config.server.effective_admin_bind()
                && !bind.ip().is_loopback()
            {
                tracing::warn!(
                    posture = AdminMountPosture::DedicatedListener.name(),
                    admin_bind = %bind,
                    "ADMIN SECURITY POSTURE: the dedicated admin listener binds a \
                     non-loopback address; any peer that reaches it can attempt the \
                     admin token. Point server.admin_bind back at loopback unless a \
                     tunnel requires otherwise."
                );
            }
        }
        None => {}
    }
}

/// Check if a provider with catalog enforcement enabled has no usable models.
///
/// Returns a warning string if the provider enforces its catalog but has no
/// static models and no cached live discovery results.
pub fn catalog_enforcement_warning(provider: &ProviderConfig, cache_dir: &Path) -> Option<String> {
    let catalog = provider.catalog.as_ref()?;
    if !catalog.enforce || !merge_catalog(catalog, &[]).is_empty() {
        return None;
    }

    let cache_path = cache_dir.join(format!("{}.toml", provider.name));
    let cached_models = std::fs::symlink_metadata(&cache_path)
        .ok()
        .filter(|metadata| metadata.file_type().is_file() && !metadata.file_type().is_symlink())
        .and_then(|_| std::fs::read_to_string(&cache_path).ok())
        .and_then(|raw| parse_catalog_file(&raw).ok())
        .filter(|cache| cache.catalog.provider == provider.name)
        .map(|cache| cache.catalog.models)
        .unwrap_or_default();

    if !merge_catalog(catalog, &cached_models).is_empty() {
        return None;
    }

    Some(format!(
        "provider \"{}\" has catalog enforcement enabled but no allowed models; requests will be rejected until static models are configured or live discovery writes a cache",
        provider.name
    ))
}

/// Build version/target metadata embedded at compile time.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        name: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        target: env!("VERGEN_CARGO_TARGET_TRIPLE"),
        git_sha: env!("VERGEN_GIT_SHA"),
    }
}

/// The log format subscriber init should use (BUG-25): a set
/// `RUST_LOG_FORMAT` wins over the configured `server.log_format`, as
/// documented; unset falls back to the configured value.
pub fn startup_log_format(config: &AppConfig) -> LogFormat {
    LogFormat::effective(
        std::env::var("RUST_LOG_FORMAT").ok().as_deref(),
        config.server.log_format,
    )
}

/// Initialize the tracing subscriber.
///
/// Reads `RUST_LOG` (falling back to `"info"`); the format comes from the
/// caller, which resolves `server.log_format` against `RUST_LOG_FORMAT`
/// (see [`startup_log_format`]).
pub fn init_tracing(log_format: &LogFormat) {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let filter = build_env_filter();
    let layer = build_log_layer(log_format, std::io::stdout).with_filter(filter);
    tracing_subscriber::registry().with(layer).init();
}

fn build_env_filter() -> tracing_subscriber::EnvFilter {
    use tracing_subscriber::EnvFilter;
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::builder().parse_lossy("info"))
}

fn build_log_layer<W>(
    log_format: &LogFormat,
    make_writer: W,
) -> Box<dyn tracing_subscriber::Layer<tracing_subscriber::Registry> + Send + Sync>
where
    W: for<'writer> tracing_subscriber::fmt::MakeWriter<'writer> + Send + Sync + 'static,
{
    use tracing_subscriber::{fmt, layer::Layer};
    match log_format {
        LogFormat::Plain => Box::new(
            fmt::layer()
                .with_writer(make_writer)
                .with_ansi(false)
                .boxed(),
        ),
        LogFormat::Json => Box::new(fmt::layer().json().with_writer(make_writer).boxed()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "admin-ui")]
    #[test]
    fn invalid_refresh_logging_is_loud_first_then_once_a_minute() {
        // The first failure shouts; the next minute of failures stays
        // quiet, then one line per minute carries the count.
        assert!(should_log_invalid_refresh(1));
        assert!(!should_log_invalid_refresh(2));
        assert!(!should_log_invalid_refresh(59));
        assert!(should_log_invalid_refresh(60));
        assert!(!should_log_invalid_refresh(61));
        assert!(should_log_invalid_refresh(120));
    }

    fn provider_config(raw: &str) -> ProviderConfig {
        toml::from_str::<llm_proxy_core::ProviderFile>(raw)
            .unwrap()
            .provider
    }

    #[test]
    fn enforcement_warning_respects_discovered_mode() {
        let provider = provider_config(
            r#"[provider]
name = "test-provider"
api_key = "test-key"
auth_style = "bearer"

[provider.catalog]
mode = "discovered"
enforce = true

[[provider.catalog.models]]
id = "static-model"
"#,
        );
        let directory = tempfile::tempdir().unwrap();

        let warning = catalog_enforcement_warning(&provider, directory.path());

        assert!(warning.is_some());
    }

    #[test]
    fn enforcement_warning_accepts_usable_cache() {
        let provider = provider_config(
            r#"[provider]
name = "test-provider"
api_key = "test-key"
auth_style = "bearer"

[provider.catalog]
mode = "discovered"
enforce = true
"#,
        );
        let directory = tempfile::tempdir().unwrap();
        let cache = llm_proxy_core::CatalogFile {
            catalog: llm_proxy_core::CatalogFileMetadata {
                provider: "test-provider".to_owned(),
                source: "live".to_owned(),
                generated_at: "2026-06-10T00:00:00Z".to_owned(),
                models: vec![llm_proxy_core::StaticModelCatalogEntry {
                    id: "cached-model".to_owned(),
                    display_name: None,
                    supports: Vec::new(),
                    context_length: None,
                }],
            },
        };
        std::fs::write(
            directory.path().join("test-provider.toml"),
            toml::to_string(&cache).unwrap(),
        )
        .unwrap();

        let warning = catalog_enforcement_warning(&provider, directory.path());

        assert!(warning.is_none());
    }

    fn capture_event_for(log_format: &LogFormat) -> Vec<u8> {
        use std::sync::{Arc, Mutex};
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        let layer = build_log_layer(log_format, BufferWriter(Arc::clone(&buf)));
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::registry().with(layer);
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            tracing::event!(
                target: "log_format_test",
                tracing::Level::INFO,
                message = "log-format-probe",
                probe = "plain-or-json",
            );
        }
        buf.lock().expect("buffer lock not poisoned").clone()
    }

    #[derive(Clone)]
    pub(crate) struct BufferWriter(pub(crate) std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BufferWriter {
        type Writer = BufferWriterGuard<'a>;

        fn make_writer(&'a self) -> Self::Writer {
            BufferWriterGuard(self.0.lock().expect("buffer lock not poisoned"))
        }
    }

    pub(crate) struct BufferWriterGuard<'a>(std::sync::MutexGuard<'a, Vec<u8>>);

    impl<'a> std::io::Write for BufferWriterGuard<'a> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn looks_like_json_object(line: &str) -> bool {
        let trimmed = line.trim();
        trimmed.starts_with('{') && trimmed.ends_with('}')
    }

    #[test]
    fn json_log_format_yields_a_json_object_line() {
        let captured = String::from_utf8(capture_event_for(&LogFormat::Json))
            .expect("captured bytes are UTF-8");
        assert!(
            !captured.is_empty(),
            "a tracing::event! must produce output under LogFormat::Json"
        );
        for line in captured.lines() {
            if line.trim().is_empty() {
                continue;
            }
            assert!(
                looks_like_json_object(line),
                "LogFormat::Json line must be a JSON object, got: {line}"
            );
        }
        assert!(
            captured.contains("\"level\"") && captured.contains("INFO"),
            "JSON log must carry the structured `level` field, got: {captured}"
        );
        assert!(
            captured.contains("log-format-probe"),
            "JSON log must carry the emitted message value, got: {captured}"
        );
    }

    #[test]
    fn plain_log_format_does_not_emit_a_json_object() {
        let captured = String::from_utf8(capture_event_for(&LogFormat::Plain))
            .expect("captured bytes are UTF-8");
        assert!(
            !captured.is_empty(),
            "a tracing::event! must produce output under LogFormat::Plain"
        );
        let first = captured
            .lines()
            .find(|line| !line.trim().is_empty())
            .expect("at least one non-empty log line");
        assert!(
            !looks_like_json_object(first),
            "Plain log line must NOT be a JSON object, got: {first}"
        );
        assert!(
            captured.contains("log-format-probe"),
            "Plain log must still carry the emitted message, got: {captured}"
        );
    }

    fn write_main_config(
        dir: &std::path::Path,
        storage_section: Option<&str>,
    ) -> std::path::PathBuf {
        let path = dir.join("config.toml");
        let storage = storage_section.unwrap_or_default();
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
{storage}"#
            ),
        )
        .unwrap();
        path
    }

    #[tokio::test]
    async fn load_toml_state_wires_sqlite_ledger_when_storage_configured() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("usage.db");
        let storage = format!(
            "
[storage]
path = \"{}\"
",
            db_path.display()
        );
        let path = write_main_config(dir.path(), Some(&storage));

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("state with [storage]");

        let debug = format!("{state:?}");
        assert!(
            debug.contains("SqliteEventBus"),
            "[storage] must wire the SQLite ledger, got: {debug}"
        );

        let event =
            llm_proxy_storage::ProxyEvent::RequestReceived(llm_proxy_storage::RequestReceived {
                request_id: "wiring-1".to_owned(),
                timestamp: time::OffsetDateTime::now_utc(),
                caller: "laptop".to_owned(),
                provider: "prov-a".to_owned(),
                route_kind: "messages".to_owned(),
                client_protocol: "anthropic".to_owned(),
                model: llm_proxy_protocol::core::ModelRef {
                    requested: "claude-3".to_owned(),
                    upstream: None,
                },
                streaming: false,
                body_hash: "digest".to_owned(),
                session_id: None,
            });
        let bus = std::sync::Arc::clone(state.event_bus());
        bus.emit(&event);
        let closer = std::sync::Arc::clone(&bus);
        tokio::task::spawn_blocking(move || llm_proxy_storage::EventBus::close(&*closer))
            .await
            .expect("close");

        let db = std::fs::read(&db_path).expect("database file exists after close");
        assert!(
            !db.is_empty(),
            "database file must hold the committed schema and rows"
        );
    }

    #[tokio::test]
    async fn load_toml_state_keeps_noop_bus_without_storage() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_main_config(dir.path(), None);

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("state without [storage]");

        let debug = format!("{state:?}");
        assert!(
            debug.contains("NoopBus"),
            "absent [storage] must keep the no-op bus (today's behavior), got: {debug}"
        );
        llm_proxy_storage::EventBus::close(state.event_bus().as_ref());
    }

    /// Writes a real ledger with one caller's day: a paid 0.25 USD request
    /// and a cache hit priced at 7.00 USD that the seed must ignore. The
    /// flush runs on the blocking pool: `close` waits on the writer task,
    /// which only progresses while the runtime thread stays free.
    async fn seed_day_ledger(db_path: &std::path::Path) {
        let bus = llm_proxy_storage::SqliteEventBus::open(
            db_path,
            llm_proxy_storage::RetentionPolicy {
                max_age: None,
                max_rows: None,
                prune_every: std::time::Duration::from_secs(3600),
            },
        )
        .expect("seed ledger");
        let now = time::OffsetDateTime::now_utc();
        for (id, cost_total, cache_hit) in [
            ("seed-1", Some(rust_decimal::Decimal::new(25, 2)), false),
            ("seed-2", Some(rust_decimal::Decimal::new(700, 2)), true),
        ] {
            let received = llm_proxy_storage::ProxyEvent::RequestReceived(
                llm_proxy_storage::RequestReceived {
                    request_id: id.to_owned(),
                    timestamp: now,
                    caller: "laptop".to_owned(),
                    provider: "prov-a".to_owned(),
                    route_kind: "messages".to_owned(),
                    client_protocol: "anthropic".to_owned(),
                    model: llm_proxy_protocol::core::ModelRef {
                        requested: "claude-3".to_owned(),
                        upstream: None,
                    },
                    streaming: false,
                    body_hash: "digest".to_owned(),
                    session_id: None,
                },
            );
            let completed = llm_proxy_storage::ProxyEvent::ResponseCompleted(
                llm_proxy_storage::ResponseCompleted {
                    request_id: id.to_owned(),
                    timestamp: now,
                    provider: "prov-a".to_owned(),
                    upstream_message_id: None,
                    model: llm_proxy_protocol::core::ModelRef {
                        requested: "claude-3".to_owned(),
                        upstream: None,
                    },
                    usage: llm_proxy_protocol::core::Usage {
                        input_tokens: 100,
                        output_tokens: 10,
                        provenance: llm_proxy_protocol::core::UsageProvenance::ProviderReported,
                        ..llm_proxy_protocol::core::Usage::default()
                    },
                    cost: cost_total.map(|total| llm_proxy_protocol::core::Cost {
                        input: rust_decimal::Decimal::ZERO,
                        output: rust_decimal::Decimal::ZERO,
                        cache_creation: rust_decimal::Decimal::ZERO,
                        cache_read: rust_decimal::Decimal::ZERO,
                        reasoning: rust_decimal::Decimal::ZERO,
                        total,
                    }),
                    stop_reason: llm_proxy_protocol::core::StopReason::EndTurn,
                    latency_ms: 12,
                    credential_id: None,
                    cache_hit,
                },
            );
            llm_proxy_storage::EventBus::emit(&bus, &received);
            llm_proxy_storage::EventBus::emit(&bus, &completed);
        }
        tokio::task::spawn_blocking(move || llm_proxy_storage::EventBus::close(&bus))
            .await
            .expect("flush the seed ledger");
    }

    #[tokio::test]
    async fn load_toml_state_seeds_day_budgets_from_the_ledger_when_storage_is_set() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("usage.db");
        seed_day_ledger(&db_path).await;

        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            format!(
                r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[storage]
path = "{}"

[auth]
mode = "client-keys"

[auth.clients.laptop]
verifier = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$eEnjdfBL8hfvU2QuGOpow8Yq2Jyrr8GXKYgtAAsscSs"

[auth.clients.laptop.budget]
usd_per_day = "1.00"
"#,
                db_path.display()
            ),
        )
        .unwrap();

        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let layer = build_log_layer(&LogFormat::Json, BufferWriter(std::sync::Arc::clone(&buf)));
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = tracing::subscriber::set_default(subscriber);
        let state = load_toml_state(
            &config,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("state with [storage] and a budgeted key");
        drop(_guard);

        let captured = String::from_utf8(buf.lock().expect("buffer lock not poisoned").clone())
            .expect("captured bytes are UTF-8");
        assert!(
            captured.contains("seeded daily budgets from the usage ledger"),
            "startup must seed day budgets from the ledger when [storage] is set: {captured}"
        );
        assert!(
            captured.contains("\"callers\":1"),
            "exactly one caller group must come out of the seed query (the cache hit is not a \
             paid request): {captured}"
        );
        let closer = std::sync::Arc::clone(state.event_bus());
        tokio::task::spawn_blocking(move || llm_proxy_storage::EventBus::close(&*closer))
            .await
            .expect("close the state's ledger");
    }

    fn write_provider_file(dir: &std::path::Path, egress: Option<&str>) {
        let egress_section = egress.unwrap_or_default();
        std::fs::create_dir_all(dir.join("providers")).unwrap();
        std::fs::write(
            dir.join("providers/egress-test.toml"),
            format!(
                r#"
[provider]
name = "egress-test"
api_key = "test-key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
chat_completions = "chat"
{egress_section}"#
            ),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn load_toml_state_wires_egress_client_only_for_configured_providers() {
        let dir = tempfile::tempdir().unwrap();
        write_provider_file(
            dir.path(),
            Some(
                "[provider.egress]\nproxy = \"socks5h://egress-user:egress-pass@127.0.0.1:1080\"\n",
            ),
        );
        std::fs::write(
            dir.path().join("providers/direct.toml"),
            r#"
[provider]
name = "direct"
api_key = "test-key"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"

[provider.routes]
chat_completions = "chat"
"#,
        )
        .unwrap();
        let path = write_main_config(dir.path(), None);

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("state builds with an egress provider");

        let debug = format!("{state:?}");
        let egress_map = debug
            .split("egress_clients: ")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .unwrap_or_default();
        assert!(
            egress_map.contains("egress-test"),
            "the egress provider must have a wired client: {debug}"
        );
        assert!(
            !egress_map.contains("direct"),
            "providers without egress must not appear in the egress map: {debug}"
        );
        assert!(
            !debug.contains("egress-user") && !debug.contains("egress-pass"),
            "proxy credentials must never print in the state Debug: {debug}"
        );
    }

    #[tokio::test]
    async fn load_toml_state_rejects_egress_url_the_transport_cannot_use() {
        let dir = tempfile::tempdir().unwrap();
        write_provider_file(
            dir.path(),
            Some("[provider.egress]\nproxy = \"http://[::1:8080\"\n"),
        );
        let path = write_main_config(dir.path(), None);

        let err = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect_err("unusable egress proxy must be a startup error");

        let message = format!("{err:#}");
        assert!(
            message.contains("egress"),
            "the error must point at the egress wiring: {message}"
        );
        assert!(
            message.contains("egress-test"),
            "the error must name the provider: {message}"
        );
    }

    fn write_bare_v1_config(dir: &std::path::Path, model_target: &str) -> std::path::PathBuf {
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "bare-v1-test"

[server.bare_v1]
enabled = true
[server.bare_v1.models]
"claude-sonnet-4-6" = "{model_target}"
"#
            ),
        )
        .unwrap();
        path
    }

    #[tokio::test]
    async fn load_toml_state_rejects_a_bare_v1_target_naming_nothing_loadable() {
        let dir = tempfile::tempdir().unwrap();
        write_provider_file(dir.path(), None);
        let path = write_bare_v1_config(dir.path(), "ghost-pool");

        let err = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect_err("a bare /v1 model mapping to an unknown target must fail startup");

        let message = format!("{err:#}");
        assert!(
            message.contains("bare_v1") && message.contains("ghost-pool"),
            "the error names the section and the unknown target: {message}"
        );
    }

    #[tokio::test]
    async fn load_toml_state_builds_with_a_bare_v1_target_that_is_a_real_provider() {
        let dir = tempfile::tempdir().unwrap();
        write_provider_file(dir.path(), None);
        let path = write_bare_v1_config(dir.path(), "egress-test");

        load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("a bare /v1 target naming a configured provider builds the state");
    }

    /// A configured `[auth.admin] key_store` in a build without the
    /// admin-ui feature is a startup error, never a silent skip: the
    /// store's keys could never load here.
    #[cfg(not(feature = "admin-ui"))]
    #[tokio::test]
    async fn load_toml_state_rejects_a_key_store_without_the_admin_ui_feature() {
        let dir = tempfile::tempdir().unwrap();
        write_provider_file(dir.path(), None);
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "keystore-test"

[auth]
mode = "client-keys"

[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$bi0KNLfLnc1qQxSai7uQDVMEMeSRpD/+rFoYCc/kku8"

[auth.admin]
token = "admin-secret"
key_store = "managed-keys.db"

[storage]
path = "usage.db"
"#,
        )
        .unwrap();

        let error = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect_err("a key store without the admin-ui feature must refuse startup");
        let message = format!("{error:#}");
        assert!(
            message.contains("key_store") && message.contains("admin-ui"),
            "the error names the section and the missing feature: {message}"
        );
    }

    fn admin_app_config(
        on_main: bool,
        admin_bind: Option<std::net::SocketAddr>,
    ) -> llm_proxy_core::AppConfig {
        use llm_proxy_core::{AdminAuthConfig, ServerConfig, StorageConfig};
        llm_proxy_core::AppConfig {
            server: ServerConfig {
                bind: "127.0.0.1:3456".parse().unwrap(),
                request_timeout: std::time::Duration::from_secs(300),
                shutdown_timeout: std::time::Duration::from_secs(30),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "posture-test".to_owned(),
                rate_limit_rpm: 100,
                trust_forwarded_headers: false,
                cache: Default::default(),
                idempotency: Default::default(),
                metrics: Default::default(),
                log_format: Default::default(),
                admin_bind,
                admin_on_main_listener: on_main,
                endpoint_families: Default::default(),
                upstream_timeouts: Default::default(),
                bare_v1: Default::default(),
                preflight: Default::default(),
                semantic_cache: Default::default(),
            },
            auth: llm_proxy_core::AuthConfig {
                mode: llm_proxy_core::AuthMode::None,
                clients: std::collections::HashMap::new(),
                admin: Some(AdminAuthConfig {
                    token: secrecy::SecretString::from("admin-secret"),
                    key_store: None,
                }),
                legacy_bare_tokens: false,
            },
            storage: Some(StorageConfig {
                path: "usage.db".into(),
                max_age: None,
                max_rows: None,
                prune_interval: std::time::Duration::from_secs(3600),
            }),
            replica: None,
            mcp: None,
        }
    }

    fn captured_posture_warnings(config: &llm_proxy_core::AppConfig) -> Vec<u8> {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let layer = build_log_layer(&LogFormat::Json, BufferWriter(std::sync::Arc::clone(&buf)));
        use tracing_subscriber::layer::SubscriberExt;
        let subscriber = tracing_subscriber::registry().with(layer);
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            warn_admin_posture(config);
        }
        buf.lock().expect("buffer lock not poisoned").clone()
    }

    #[test]
    fn main_listener_optin_is_warned_about_loudly_by_name() {
        let config = admin_app_config(true, None);
        let captured = String::from_utf8(captured_posture_warnings(&config))
            .expect("captured bytes are UTF-8");
        assert!(
            captured.contains("admin-on-main-listener"),
            "the posture must be named for grepping: {captured}"
        );
        assert!(
            captured.contains("ADMIN SECURITY POSTURE"),
            "the warning must be loud and identifiable: {captured}"
        );
    }

    #[test]
    fn dedicated_loopback_posture_stays_quiet() {
        let config = admin_app_config(false, None);
        let captured = String::from_utf8(captured_posture_warnings(&config))
            .expect("captured bytes are UTF-8");
        assert!(
            !captured.contains("ADMIN SECURITY POSTURE"),
            "the default loopback posture needs no warning: {captured}"
        );
    }

    #[test]
    fn dedicated_non_loopback_bind_is_warned_about() {
        let config = admin_app_config(false, Some("10.0.0.5:9900".parse().unwrap()));
        let captured = String::from_utf8(captured_posture_warnings(&config))
            .expect("captured bytes are UTF-8");
        assert!(
            captured.contains("non-loopback"),
            "a non-loopback admin listener must be called out: {captured}"
        );
    }
}

#[cfg(test)]
mod self_loop_startup_tests {
    use super::*;

    fn write_config(dir: &std::path::Path, endpoint: &str) -> std::path::PathBuf {
        std::fs::create_dir_all(dir.join("providers")).unwrap();
        std::fs::write(
            dir.join("providers/loopy.toml"),
            format!(
                r#"[provider]
name = "loopy"
api_key = "k"
auth_style = "bearer"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "{endpoint}"
[provider.routes]
chat_completions = "chat"
"#
            ),
        )
        .unwrap();
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"
"#,
        )
        .unwrap();
        path
    }

    #[tokio::test]
    async fn startup_refuses_an_upstream_on_the_own_listener() {
        let dir = tempfile::tempdir().unwrap();
        // The main listener binds 127.0.0.1:3456; an upstream on the same
        // loopback port (spelled by name, so DNS resolution is exercised)
        // would forward to itself.
        let path = write_config(dir.path(), "http://localhost:3456/v1/chat/completions");
        let err = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect_err("startup must refuse a self-referential upstream");
        let message = format!("{err:#}");
        assert!(
            message.contains("loopy") && message.contains("self-referential"),
            "the refusal names the provider and the reason: {message}"
        );
    }

    #[tokio::test]
    async fn startup_accepts_an_upstream_on_another_port() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(dir.path(), "http://127.0.0.1:4567/v1/chat/completions");
        load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("an upstream on another port is not a loop");
    }

    /// A `passthrough_auth` provider under client-keys auth forwards the
    /// caller's client key to the upstream, so startup refuses, exactly
    /// like `llm-proxy validate` (same `load_registry_with_checks` path).
    #[tokio::test]
    async fn startup_refuses_passthrough_provider_with_client_keys() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/pt.toml"),
            r#"[provider]
name = "pt"
api_key = "unused"
passthrough_auth = true
auth_style = "bearer"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "http://127.0.0.1:9/v1/chat/completions"
[provider.routes]
chat_completions = "chat"
"#,
        )
        .unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "0.0.0.0:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[auth]
mode = "client-keys"

[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$eEnjdfBL8hfvU2QuGOpow8Yq2Jyrr8GXKYgtAAsscSs"
"#,
        )
        .unwrap();
        let err = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect_err("startup must refuse client-keys with a passthrough_auth provider");
        let message = format!("{err:#}");
        assert!(
            message.contains("pt") && message.contains("client key"),
            "the refusal names the provider and the client key it would leak: {message}"
        );
    }

    #[tokio::test]
    async fn family_routes_need_the_endpoint_families_feature() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/embedder.toml"),
            r#"[provider]
name = "embedder"
api_key = "k"
auth_style = "bearer"
[provider.adapters.embed]
protocol = "openai_chat_completions"
endpoint = "http://127.0.0.1:9/v1/embeddings"
[provider.routes]
embeddings = "embed"
"#,
        )
        .unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3459"
request_timeout = "300s"
log_level = "info"
server_name = "family-refusal"
"#,
        )
        .unwrap();
        let result = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await;
        // With the feature the routes serve; without it a configured
        // family would answer 404 forever, so startup refuses instead.
        #[cfg(feature = "endpoint-families")]
        assert!(result.is_ok(), "with the feature the family routes serve");
        #[cfg(not(feature = "endpoint-families"))]
        match result {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("embedder") && message.contains("endpoint-families"),
                    "the refusal names the provider and the feature, got: {message}"
                );
            }
            Ok(_) => panic!("family routes must never be silently unserved"),
        }
    }

    #[tokio::test]
    async fn semantic_cache_without_the_feature_refuses_to_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[server]
bind = "127.0.0.1:3458"
request_timeout = "300s"
log_level = "info"
server_name = "semantic-refusal"

[server.semantic_cache]
enabled = true
"#,
        )
        .unwrap();
        let result = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await;
        // With the semantic-cache feature the daemon serves it; without
        // the feature the opt-in is a startup error, never a silent skip.
        #[cfg(feature = "semantic-cache")]
        assert!(result.is_ok(), "with the feature the opt-in serves");
        #[cfg(not(feature = "semantic-cache"))]
        match result {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("semantic-cache"),
                    "the refusal names the feature, got: {message}"
                );
            }
            Ok(_) => panic!("[server.semantic_cache] must never be silently skipped"),
        }
    }

    /// A `[replica]` section must never start single-node silently. The
    /// exact outcome depends on the build's features, so each cfg branch
    /// asserts its own.
    #[tokio::test]
    async fn replica_config_refuses_or_connects_by_feature() {
        // Point at a port that is guaranteed to have no Redis: bind an
        // ephemeral listener and free it. A fixed port (6379) would make
        // this test depend on the developer's machine running a local
        // Redis or not.
        let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3457"
request_timeout = "300s"
log_level = "info"
server_name = "replica-refusal"

[replica]
backend = "redis"
url = "redis://127.0.0.1:{port}"
"#
            ),
        )
        .unwrap();
        let result = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await;
        #[cfg(feature = "store-redis")]
        match result {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("replica") && message.contains("connecting"),
                    "the connect failure must name the replica wiring, got: {message}"
                );
            }
            Ok(_) => panic!("[replica] must never start single-node silently"),
        }
        #[cfg(not(feature = "store-redis"))]
        match result {
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    message.contains("replica") && message.contains("store-redis"),
                    "the refusal must name the missing feature, got: {message}"
                );
            }
            Ok(_) => panic!("[replica] must never start single-node silently"),
        }
    }

    /// A config with `[auth.admin].key_store` for the managed-key
    /// startup tests. `policy` seeds one key with that policy; `None`
    /// leaves the store empty.
    #[cfg(feature = "admin-ui")]
    fn config_with_key_store(
        mode: &str,
        policy: Option<&llm_proxy_admin::KeyPolicyInput>,
    ) -> (
        tempfile::TempDir,
        std::path::PathBuf,
        Option<secrecy::SecretString>,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let store_path = dir.path().join("managed-keys.db");
        let token = policy.map(|policy| {
            llm_proxy_admin::ManagedKeys::open(&store_path)
                .expect("key store")
                .create(policy)
                .expect("one live key")
                .token
        });
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3460"
request_timeout = "300s"
log_level = "info"
server_name = "managed-keys-startup"

[auth]
mode = "{mode}"

[auth.admin]
token = "admin-tok"
key_store = "{}"

[storage]
path = "{}"
"#,
                store_path.display(),
                dir.path().join("usage.db").display()
            ),
        )
        .unwrap();
        (dir, path, token)
    }

    #[cfg(feature = "admin-ui")]
    fn config_with_seeded_key_store(
        mode: &str,
    ) -> (tempfile::TempDir, std::path::PathBuf, secrecy::SecretString) {
        let (dir, path, token) = config_with_key_store(mode, Some(&Default::default()));
        (dir, path, token.expect("seeded key"))
    }

    /// auth.mode = "none" plus a configured key store is a startup
    /// refusal, not a warning: no client auth is enforced in that mode,
    /// so the store's keys could never protect anything and an operator
    /// reading a WARN line could believe they have auth on an open proxy.
    /// The refusal names both fixes — flip the mode or drop the store.
    /// (Regression: this used to boot with only a warning; the test
    /// fails against that old behaviour by expecting the error.)
    #[tokio::test]
    #[cfg(feature = "admin-ui")]
    async fn auth_mode_none_with_a_key_store_refuses_to_start() {
        let (_dir, path, _token) = config_with_seeded_key_store("none");
        let error = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect_err("mode none can never enforce managed keys; startup must refuse");
        let message = format!("{error:#}");
        assert!(
            message.contains("client-keys") && message.contains("key_store"),
            "the refusal must name the fix (set the mode or drop key_store): {message}"
        );
    }

    /// client-keys mode with an empty [auth.clients] and a seeded key
    /// store must start and actually enforce the managed key: anonymous
    /// requests get 401, the issued token does not.
    #[tokio::test]
    #[cfg(feature = "admin-ui")]
    async fn client_keys_mode_runs_on_managed_keys_alone() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode, header};
        use secrecy::ExposeSecret;
        use tower::ServiceExt as _;

        let (_dir, path, token) = config_with_seeded_key_store("client-keys");
        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("the merged managed key satisfies the at-least-one-key rule");
        let app = llm_proxy_server::build_router(state);

        let anonymous = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/providers/mock/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("served");
        assert_eq!(
            anonymous.status(),
            StatusCode::UNAUTHORIZED,
            "client auth is enforced by the merged managed key"
        );

        let authenticated = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/providers/mock/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(
                        header::AUTHORIZATION,
                        format!("Bearer {}", token.expose_secret()),
                    )
                    .body(Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("served");
        assert_ne!(
            authenticated.status(),
            StatusCode::UNAUTHORIZED,
            "the issued managed token authenticates after the restart-shaped startup"
        );
    }

    /// Client-keys mode over an empty key store (a fresh deployment
    /// before its first key, or the last key revoked) must still boot —
    /// the admin listener can then issue the first key — and fail closed
    /// on every proxy request until one exists. Startup used to die at
    /// ClientAuth's empty-clients refusal, so such a deployment could
    /// never be bootstrapped.
    #[tokio::test]
    #[cfg(feature = "admin-ui")]
    async fn client_keys_mode_boots_on_an_empty_key_store_and_fails_closed() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode, header};
        use tower::ServiceExt as _;

        let (_dir, path, token) = config_with_key_store("client-keys", None);
        assert!(token.is_none(), "no key was seeded");
        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("an empty key store boots in client-keys mode");
        let app = llm_proxy_server::build_router(state);
        let refused = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/providers/mock/v1/messages")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("served");
        assert_eq!(
            refused.status(),
            StatusCode::UNAUTHORIZED,
            "with no keys loaded every request fails closed"
        );
    }

    /// The policy pinned on a managed key must be enforced, not just
    /// stored and listed back: a key issued with a TPM cap is throttled
    /// by the engine's admission, while an unpolicied managed key on the
    /// same daemon is served. The daemon merges the store's keys into
    /// the AuthConfig every consumer reads — ClientAuth authenticates
    /// them and PolicyAdmission enforces their policies.
    #[tokio::test]
    #[cfg(feature = "admin-ui")]
    async fn a_managed_keys_tpm_policy_is_enforced() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode, header};
        use secrecy::ExposeSecret;
        use tower::ServiceExt as _;

        // A loopback upstream answering an Anthropic messages body.
        let upstream = axum::Router::new().route(
            "/{*path}",
            axum::routing::post(|| async {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"id":"msg_h","type":"message","role":"assistant",
                        "content":[{"type":"text","text":"upstream answer"}],
                        "model":"claude-sonnet-4-6","stop_reason":"end_turn",
                        "stop_sequence":null,
                        "usage":{"input_tokens":10,"output_tokens":4}}"#,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind upstream");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, upstream).await {
                tracing::trace!(%error, "fake upstream stopped");
            }
        });

        let (dir, path, capped) = config_with_key_store(
            "client-keys",
            Some(&llm_proxy_admin::KeyPolicyInput {
                tpm: Some(8),
                ..llm_proxy_admin::KeyPolicyInput::default()
            }),
        );
        // A second, unpolicied key on the same store: the control.
        let plain = llm_proxy_admin::ManagedKeys::open(dir.path().join("managed-keys.db"))
            .expect("key store")
            .create(&llm_proxy_admin::KeyPolicyInput::default())
            .expect("plain key")
            .token;
        std::fs::create_dir_all(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/loopy.toml"),
            format!(
                r#"[provider]
name = "loopy"
api_key = "k"
auth_style = "bearer"
[provider.adapters.msgs]
protocol = "anthropic_messages"
endpoint = "http://{addr}/v1/messages"
[provider.routes]
messages = "msgs"
"#
            ),
        )
        .unwrap();
        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("daemon starts");
        let app = llm_proxy_server::build_router(state);

        let post = |token: &secrecy::SecretString| {
            let request = Request::builder()
                .method("POST")
                .uri("/providers/loopy/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .header(
                    header::AUTHORIZATION,
                    format!("Bearer {}", token.expose_secret()),
                )
                .body(Body::from(
                    r#"{"model":"claude-sonnet-4-6","max_tokens":64,"messages":[{"role":"user","content":"please explain rust ownership and borrowing in a few sentences"}]}"#,
                ))
                .expect("request");
            app.clone().oneshot(request)
        };

        let throttled = post(&capped.expect("capped key")).await.expect("served");
        assert_eq!(
            throttled.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "the capped key's input estimate alone exceeds its tpm of 8"
        );

        let served = post(&plain).await.expect("served");
        assert_eq!(
            served.status(),
            StatusCode::OK,
            "the unpolicied managed key on the same daemon is served"
        );
        let bytes = axum::body::to_bytes(served.into_body(), 1 << 20)
            .await
            .expect("body");
        assert!(
            String::from_utf8_lossy(&bytes).contains("upstream answer"),
            "the control call reached the upstream"
        );
    }
}

/// Managed keys issued or revoked while the daemon is already running (the
/// admin surface's own store connection) must reach enforcement —
/// authentication and admission — without a restart. The daemon watches
/// the store and swaps the live client-auth and admission rules in place.
#[cfg(test)]
#[cfg(feature = "admin-ui")]
mod managed_key_runtime_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};
    use axum::response::Response;
    use secrecy::ExposeSecret;
    use tower::ServiceExt as _;

    use super::load_toml_state;

    /// How long a runtime change may take to reach enforcement: the daemon
    /// refreshes on a short interval, so seconds — never a restart.
    const EFFECT_DEADLINE: std::time::Duration = std::time::Duration::from_secs(20);

    fn write_config(dir: &std::path::Path) -> std::path::PathBuf {
        let store_path = dir.join("managed-keys.db");
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"
[server]
bind = "127.0.0.1:3461"
request_timeout = "300s"
log_level = "info"
server_name = "managed-runtime"

[auth]
mode = "client-keys"

[auth.admin]
token = "admin-tok"
key_store = "{store}"

[storage]
path = "{usage}"
"#,
                store = store_path.display(),
                usage = dir.join("usage.db").display()
            ),
        )
        .unwrap();
        path
    }

    fn write_provider(
        dir: &std::path::Path,
        name: &str,
        adapter_key: &str,
        protocol: &str,
        route_key: &str,
        endpoint: &str,
    ) {
        std::fs::create_dir_all(dir.join("providers")).unwrap();
        std::fs::write(
            dir.join(format!("providers/{name}.toml")),
            format!(
                r#"[provider]
name = "{name}"
api_key = "k"
auth_style = "bearer"
[provider.adapters.{adapter_key}]
protocol = "{protocol}"
endpoint = "{endpoint}"
[provider.routes]
{route_key} = "{adapter_key}"
"#
            ),
        )
        .unwrap();
    }

    /// The admin surface's view of the store: a second connection to the
    /// same file the daemon loaded, exactly what POST /keys uses.
    fn admin_view(dir: &std::path::Path) -> llm_proxy_admin::ManagedKeys {
        llm_proxy_admin::ManagedKeys::open(dir.join("managed-keys.db")).expect("key store")
    }

    async fn spawn_json_upstream(body: &'static str) -> std::net::SocketAddr {
        let app = axum::Router::new().route(
            "/{*path}",
            axum::routing::post(move || async move {
                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    body,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind upstream");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::trace!(%error, "fake upstream stopped");
            }
        });
        addr
    }

    async fn post(
        app: axum::Router,
        uri: &str,
        token: &secrecy::SecretString,
        body: &str,
    ) -> Response {
        let request = Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .header(
                header::AUTHORIZATION,
                format!("Bearer {}", token.expose_secret()),
            )
            .body(Body::from(body.to_owned()))
            .expect("request");
        app.oneshot(request).await.expect("oneshot is infallible")
    }

    /// Retry until the daemon has picked the key up. The 401 answers a
    /// not-yet-installed key (authentication never saw it) and consume
    /// nothing; a 429 proves the key installed but the call was refused,
    /// so it also doesn't count — the first other answer is the call that
    /// counts.
    async fn first_admitted(
        app: axum::Router,
        uri: &str,
        token: &secrecy::SecretString,
        body: &str,
    ) -> Response {
        let deadline = std::time::Instant::now() + EFFECT_DEADLINE;
        loop {
            let response = post(app.clone(), uri, token, body).await;
            let status = response.status();
            if status != StatusCode::UNAUTHORIZED && status != StatusCode::TOO_MANY_REQUESTS {
                return response;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the runtime-issued key never authenticated (still 401 after {EFFECT_DEADLINE:?})"
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    async fn eventually_unauthorized(
        app: axum::Router,
        uri: &str,
        token: &secrecy::SecretString,
        body: &str,
    ) {
        let deadline = std::time::Instant::now() + EFFECT_DEADLINE;
        loop {
            let response = post(app.clone(), uri, token, body).await;
            if response.status() == StatusCode::UNAUTHORIZED {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the revoked key kept authenticating (last answer {}) after {EFFECT_DEADLINE:?}",
                response.status()
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    /// The audit's live repro, issued while the daemon is running: a key
    /// created through the admin surface's store connection with tpm=110,
    /// behind an upstream that reports prompt_tokens=50 per embeddings
    /// call — three tiny calls must answer 200, 200, 429 exactly like the
    /// equivalent file key. Each 48-byte JSON body reserves 12 estimated
    /// input tokens at admission (chars/4), so the second call projects
    /// 50 settled + 12 reserved (admitted) and the third 100 + 12
    /// (refused) — reported usage is what pushes it over.
    #[tokio::test]
    #[cfg(feature = "endpoint-families")]
    async fn a_runtime_issued_managed_keys_tpm_is_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let upstream = spawn_json_upstream(
            r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],
                "model":"text-embedding-3-small","usage":{"prompt_tokens":50,"total_tokens":50}}"#,
        )
        .await;
        write_provider(
            dir.path(),
            "embedder",
            "embed",
            "openai_chat_completions",
            "embeddings",
            &format!("http://{}/v1/embeddings", upstream),
        );
        let path = write_config(dir.path());

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("daemon starts on the fresh store");
        let app = llm_proxy_server::build_router(state);

        let issued = admin_view(dir.path())
            .create(&llm_proxy_admin::KeyPolicyInput {
                tpm: Some(110),
                ..llm_proxy_admin::KeyPolicyInput::default()
            })
            .expect("runtime issuance");

        let call = |app: axum::Router| {
            post(
                app,
                "/providers/embedder/v1/embeddings",
                &issued.token,
                r#"{"model":"text-embedding-3-small","input":"hi"}"#,
            )
        };

        let first = first_admitted(
            app.clone(),
            "/providers/embedder/v1/embeddings",
            &issued.token,
            r#"{"model":"text-embedding-3-small","input":"hi"}"#,
        )
        .await;
        assert_eq!(
            first.status(),
            StatusCode::OK,
            "the runtime-issued key serves once its policy is loaded"
        );
        let second = call(app.clone()).await;
        assert_eq!(second.status(), StatusCode::OK);
        let third = call(app.clone()).await;
        assert_eq!(
            third.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "third call crosses the runtime-issued tpm of 110 (50 charged per call, 12 \
             reserved at admission)"
        );
    }

    /// A key issued at runtime with an allow_models glob must have the
    /// glob enforced: the model that does not match is a 403 (the key
    /// authenticated; admission refused it), not a 401.
    #[tokio::test]
    async fn a_runtime_issued_managed_keys_allow_models_glob_is_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let upstream = spawn_json_upstream(
            r#"{"id":"msg_h","type":"message","role":"assistant",
                "content":[{"type":"text","text":"upstream answer"}],
                "model":"claude-sonnet-4-6","stop_reason":"end_turn",
                "stop_sequence":null,
                "usage":{"input_tokens":10,"output_tokens":4}}"#,
        )
        .await;
        write_provider(
            dir.path(),
            "loopy",
            "msgs",
            "anthropic_messages",
            "messages",
            &format!("http://{}/v1/messages", upstream),
        );
        let path = write_config(dir.path());

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("daemon starts");
        let app = llm_proxy_server::build_router(state);

        let restricted = admin_view(dir.path())
            .create(&llm_proxy_admin::KeyPolicyInput {
                allow_models: vec!["nothing-matches-*".to_owned()],
                ..llm_proxy_admin::KeyPolicyInput::default()
            })
            .expect("runtime issuance");
        let plain = admin_view(dir.path())
            .create(&llm_proxy_admin::KeyPolicyInput::default())
            .expect("control issuance");

        let refused = first_admitted(
            app.clone(),
            "/providers/loopy/v1/messages",
            &restricted.token,
            r#"{"model":"claude-sonnet-4-6","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(
            refused.status(),
            StatusCode::FORBIDDEN,
            "the glob refuses the non-matching model after the key authenticated"
        );

        let served = first_admitted(
            app.clone(),
            "/providers/loopy/v1/messages",
            &plain.token,
            r#"{"model":"claude-sonnet-4-6","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(served.status(), StatusCode::OK);
    }

    /// Revoking a key through the admin surface's store connection must
    /// stop its token authenticating on the running daemon — no restart.
    #[tokio::test]
    async fn revoking_a_managed_key_stops_it_authenticating_without_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let upstream = spawn_json_upstream(
            r#"{"id":"msg_h","type":"message","role":"assistant",
                "content":[{"type":"text","text":"upstream answer"}],
                "model":"claude-sonnet-4-6","stop_reason":"end_turn",
                "stop_sequence":null,
                "usage":{"input_tokens":10,"output_tokens":4}}"#,
        )
        .await;
        write_provider(
            dir.path(),
            "loopy",
            "msgs",
            "anthropic_messages",
            "messages",
            &format!("http://{}/v1/messages", upstream),
        );
        let path = write_config(dir.path());
        let seeded = admin_view(dir.path())
            .create(&llm_proxy_admin::KeyPolicyInput::default())
            .expect("seeded key");

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("daemon starts with the seeded key");
        let app = llm_proxy_server::build_router(state);
        let body = r#"{"model":"claude-sonnet-4-6","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#;
        let served = post(
            app.clone(),
            "/providers/loopy/v1/messages",
            &seeded.token,
            body,
        )
        .await;
        assert_eq!(
            served.status(),
            StatusCode::OK,
            "the seeded key authenticates before the revocation"
        );

        admin_view(dir.path())
            .revoke(&seeded.key_id)
            .expect("runtime revocation");

        eventually_unauthorized(app, "/providers/loopy/v1/messages", &seeded.token, body).await;
    }

    /// A store whose keys were all revoked must still boot (the audit's
    /// brick-on-restart repro): the rows exist, none is live, and the
    /// daemon comes up failing closed until a key is issued again.
    #[tokio::test]
    async fn client_keys_mode_boots_after_every_managed_key_was_revoked() {
        let dir = tempfile::tempdir().unwrap();
        let store = admin_view(dir.path());
        let first = store
            .create(&llm_proxy_admin::KeyPolicyInput::default())
            .expect("first key");
        let second = store
            .create(&llm_proxy_admin::KeyPolicyInput::default())
            .expect("second key");
        store.revoke(&first.key_id).expect("revoke first");
        store.revoke(&second.key_id).expect("revoke second");
        write_provider(
            dir.path(),
            "loopy",
            "msgs",
            "anthropic_messages",
            "messages",
            "http://127.0.0.1:9/v1/messages",
        );
        let path = write_config(dir.path());

        let state = load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("a store with only revoked keys still boots");
        let app = llm_proxy_server::build_router(state);
        let refused = post(
            app,
            "/providers/loopy/v1/messages",
            &second.token,
            r#"{"model":"claude-sonnet-4-6","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .await;
        assert_eq!(
            refused.status(),
            StatusCode::UNAUTHORIZED,
            "the revoked token does not authenticate after the restart-shaped boot"
        );
    }

    /// A store-managed deployment must not log a false invalid-[auth]
    /// ERROR on boot (BUG 3): with no file clients and live managed keys,
    /// the effective auth is valid, so the boot log stays clean while the
    /// tokens are accepted.
    #[tokio::test]
    async fn a_store_managed_boot_logs_no_false_auth_error() {
        use tracing_subscriber::layer::SubscriberExt;

        let dir = tempfile::tempdir().unwrap();
        admin_view(dir.path())
            .create(&llm_proxy_admin::KeyPolicyInput::default())
            .expect("one live key");
        write_provider(
            dir.path(),
            "loopy",
            "msgs",
            "anthropic_messages",
            "messages",
            "http://127.0.0.1:9/v1/messages",
        );
        let path = write_config(dir.path());

        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let layer = super::build_log_layer(
            &llm_proxy_core::LogFormat::Plain,
            super::tests::BufferWriter(std::sync::Arc::clone(&buf)),
        );
        let subscriber = tracing_subscriber::registry().with(layer);
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            let state = load_toml_state(
                &path,
                llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
                None,
            )
            .await
            .expect("the store-backed boot succeeds");
            let app = llm_proxy_server::build_router(state);
            let refused = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/providers/loopy/v1/messages")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from("{}"))
                        .expect("request"),
                )
                .await
                .expect("served");
            assert_eq!(
                refused.status(),
                StatusCode::UNAUTHORIZED,
                "client auth is enforced by the merged managed key"
            );
        }

        let captured = String::from_utf8(buf.lock().expect("buffer lock not poisoned").clone())
            .expect("captured bytes are UTF-8");
        assert!(
            !captured.contains("invalid [auth] config"),
            "a store-managed deployment must not log a false auth ERROR, got: {captured}"
        );
    }
}
