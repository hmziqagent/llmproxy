//! Hot reload (plan 07 step 11, backlog 23): re-read the provider and pool
//! files and swap them in, one provider at a time, without dropping
//! in-flight requests — and refuse, whole and up front, any change that
//! needs a restart.
//!
//! What reload may change: the provider and pool files under `providers/`.
//! What it must refuse: anything under `[server]`, `[auth]` (including the
//! `[auth.admin] key_store`), `[storage]`, `[replica]` or `[mcp]` — those
//! are wired into listeners, middleware, stores and the ledger, and
//! applying half of them would be worse than refusing. The refusal names
//! the exact fields and changes nothing.
//!
//! Mechanics: the new registry is loaded through the same gates startup
//! runs, a fresh engine is built over it (validated before anything live is
//! touched), and only then is the engine slot swapped. Requests already
//! running hold their own `Arc<Engine>` clone and finish on it — on any
//! provider, changed or not.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_server::AppState;
use secrecy::ExposeSecret;

use crate::state::load_registry_with_checks;

/// What one reload did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReloadReport {
    /// Providers that appeared in the new registry.
    pub added: Vec<String>,
    /// Providers that disappeared.
    pub removed: Vec<String>,
    /// Providers present in both but changed (by rendered comparison).
    pub changed: Vec<String>,
    /// Pools added / removed / changed, reported the same way.
    pub pools_changed: Vec<String>,
}

impl ReloadReport {
    /// Whether anything changed at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.changed.is_empty()
            && self.pools_changed.is_empty()
    }
}

impl std::fmt::Display for ReloadReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reload applied: {} added, {} removed, {} changed, {} pool changes",
            self.added.len(),
            self.removed.len(),
            self.changed.len(),
            self.pools_changed.len()
        )?;
        for name in self.added.iter().chain(&self.removed).chain(&self.changed) {
            write!(f, "\n  provider {name}")?;
        }
        for name in &self.pools_changed {
            write!(f, "\n  pool {name}")?;
        }
        Ok(())
    }
}

/// Reloads the provider and pool files beside `config_path` into `state`.
///
/// Refuses — changing nothing — when the new config differs anywhere
/// outside the provider files (stable `restart-required:` prefix naming the
/// fields), when a provider file fails any startup gate, or when the
/// registry cannot be loaded. On success, later requests route through the
/// new registry; in-flight requests finish on the engine they already hold.
///
/// `port_override` is the `--port` the server was started with, if any: the
/// CLI layer outranks the file, so the same override is applied to the
/// re-read bind. Nothing else about `server.bind` is masked — a file edit of
/// the bind reaches the diff and is refused as the restart-required change
/// it is.
///
/// # Errors
///
/// `anyhow` with a `restart-required:` prefix for refused changes, or the
/// underlying loader error for broken provider files.
pub async fn reload_registry(
    state: &AppState,
    config_path: &Path,
    port_override: Option<u16>,
) -> Result<ReloadReport> {
    let mut next = llm_proxy_core::load_app_config(config_path)
        .with_context(|| format!("re-reading {}", config_path.display()))?;
    // The CLI layer outranks the file (the stated precedence): a --port the
    // server was started with stays in force. Applied to the re-read bind it
    // also means the file's own port cannot mask a host change.
    if let Some(port) = port_override {
        next.server.bind.set_port(port);
    }

    let restart = llm_proxy_core::restart_required_changes(state.app_config(), &next);
    if !restart.is_empty() {
        bail!(
            "restart-required: {} changed ({}); the reload was refused and nothing was \
             applied. Restart the server to apply these fields.{}",
            restart.len(),
            restart.join(", "),
            managed_keys_note(
                state.app_config(),
                &next,
                state.runtime_issued_keys(),
                &restart
            )
        );
    }

    // The same gates startup runs; a broken provider file refuses the whole
    // reload rather than half-applying. The gates read the provider files
    // and resolve DNS (the self-loop check), which blocks — so they run off
    // the async runtime, the way doctor already runs its checks.
    let adapter_registry = ProviderAdapterRegistry::builtin();
    let check_config = next.clone();
    let check_path = config_path.to_path_buf();
    let registry = tokio::task::spawn_blocking(move || {
        load_registry_with_checks(&check_config, &check_path, &adapter_registry)
            .context("re-loading providers (reload refused; nothing was applied)")
    })
    .await
    .map_err(|error| anyhow::anyhow!("the reload's config gates task failed: {error}"))??;

    let current = state.providers();
    let report = diff_registries(&current, &registry);

    // Egress clients are per-provider; rebuild them from the new registry.
    let egress = crate::state::build_egress_clients(&next, &registry)?;

    // Build and validate the replacement engine from the proposed config
    // before touching anything live — including the config snapshot the
    // serving engine reads, so a refused build really applies nothing. The
    // snapshot and engine slot then advance together.
    let next_config = std::sync::Arc::new(next);
    let engine = state
        .build_engine_from_parts_with_config(&std::sync::Arc::new(registry), egress, &next_config)
        .map_err(|error| {
            anyhow::anyhow!(
                "building the reloaded engine (reload refused; nothing was applied): {error}"
            )
        })?;
    let _previous_config = state.config_source().swap(next_config);
    let _previous = state.swap_engine(engine);
    Ok(report)
}

/// When a managed key store backs the server, the running config carries the
/// ids the store issued at startup (the merge) while the re-read file never
/// had them, so the restart-required diff names `auth.clients` on every
/// reload. This attributes that diff to the runtime-issued keys — but only
/// when they fully explain it: the ids the running config has and the file
/// lacks must all be store-issued, the file must add no id of its own, and
/// no shared id's spec may differ (those are genuine file edits, which keep
/// the plain refusal).
fn managed_keys_note(
    running: &llm_proxy_core::AppConfig,
    next: &llm_proxy_core::AppConfig,
    runtime_ids: &[String],
    restart: &[String],
) -> String {
    if runtime_ids.is_empty()
        || !restart.iter().any(|change| change == "auth.clients")
        || restart
            .iter()
            .any(|change| change.starts_with("auth.clients."))
    {
        return String::new();
    }
    let running_only: Vec<&String> = running
        .auth
        .clients
        .keys()
        .filter(|id| !next.auth.clients.contains_key(*id))
        .collect();
    let file_added = next
        .auth
        .clients
        .keys()
        .any(|id| !running.auth.clients.contains_key(id));
    if file_added
        || running_only.is_empty()
        || !running_only.iter().all(|id| runtime_ids.contains(id))
    {
        return String::new();
    }
    let ids = running_only
        .iter()
        .map(|id| id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        " The auth.clients diff is the runtime-issued managed keys ({ids}) merged \
         from [auth.admin.key_store], not an edit of the file's own [auth.clients]; \
         editing the file cannot clear it."
    )
}

/// Names what differs between two registries, provider by provider and pool
/// by pool. Provider comparison is by rendered config (secrets redacted by
/// their `Debug`-independent serde skips, so a rotated key counts as a
/// change without being printed).
fn diff_registries(
    current: &llm_proxy_core::ProviderRegistry,
    next: &llm_proxy_core::ProviderRegistry,
) -> ReloadReport {
    let mut report = ReloadReport::default();
    let names = |registry: &llm_proxy_core::ProviderRegistry| -> BTreeSet<String> {
        registry.iter().map(|p| p.name.clone()).collect()
    };
    let (current_names, next_names) = (names(current), names(next));
    for name in next_names.difference(&current_names) {
        report.added.push(name.clone());
    }
    for name in current_names.difference(&next_names) {
        report.removed.push(name.clone());
    }
    for name in current_names.intersection(&next_names) {
        let rendered_changed = match (
            current.get(name).and_then(|p| serde_json::to_value(p).ok()),
            next.get(name).and_then(|p| serde_json::to_value(p).ok()),
        ) {
            (Some(a), Some(b)) => a != b,
            _ => true,
        };
        // Secrets are skip_serializing, so a pure key rotation needs its own
        // comparison — counted as a change, never printed.
        let secret_changed = match (current.get(name), next.get(name)) {
            (Some(a), Some(b)) => {
                a.api_key.expose_secret() != b.api_key.expose_secret()
                    || a.api_keys.as_ref().map(|keys| {
                        keys.iter()
                            .map(|k| k.expose_secret().to_owned())
                            .collect::<Vec<_>>()
                    }) != b.api_keys.as_ref().map(|keys| {
                        keys.iter()
                            .map(|k| k.expose_secret().to_owned())
                            .collect::<Vec<_>>()
                    })
            }
            _ => true,
        };
        if rendered_changed || secret_changed {
            report.changed.push(name.clone());
        }
    }
    let pool_names = |registry: &llm_proxy_core::ProviderRegistry| -> BTreeSet<String> {
        registry.pools().map(|p| p.name.clone()).collect()
    };
    let (current_pools, next_pools) = (pool_names(current), pool_names(next));
    for name in current_pools.symmetric_difference(&next_pools) {
        report.pools_changed.push(name.clone());
    }
    for name in current_pools.intersection(&next_pools) {
        let changed = match (
            current
                .pool(name)
                .and_then(|p| serde_json::to_value(p).ok()),
            next.pool(name).and_then(|p| serde_json::to_value(p).ok()),
        ) {
            (Some(a), Some(b)) => a != b,
            _ => true,
        };
        if changed {
            report.pools_changed.push(name.clone());
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::Router;
    use axum::http::StatusCode;
    use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
    use llm_proxy_engine::{ClientDialect, ConfigSource, EngineRequest, InboundRequest};
    use llm_proxy_protocol::core::{
        CoreContent, CoreMessage, CoreRequest, CoreRole, ModelRef, SamplingOptions,
    };

    fn write_provider(dir: &Path, name: &str, endpoint: &str, marker_port: u16) {
        std::fs::create_dir_all(dir.join("providers")).expect("providers dir");
        std::fs::write(
            dir.join(format!("providers/{name}.toml")),
            format!(
                r#"[provider]
name = "{name}"
api_key = "key-{name}"
auth_style = "x-api-key"
[provider.adapters.messages]
protocol = "anthropic_messages"
endpoint = "{endpoint}"
[provider.routes]
messages = "messages"

[provider.model_aliases]
"m" = "{marker_port}"
"#
            ),
        )
        .expect("provider file");
    }

    fn write_config(dir: &Path, extra: &str) -> PathBuf {
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            format!(
                r#"[server]
bind = "127.0.0.1:3477"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "reload-test"
{extra}"#
            ),
        )
        .expect("config file");
        path
    }

    async fn spawn_echo_upstream(delay: Duration, tokens: u32) -> (String, std::net::SocketAddr) {
        let app = Router::new().route(
            "/{*path}",
            axum::routing::post(move || async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                let body = format!(
                    r#"{{"id":"msg_1","type":"message","role":"assistant","content":[{{"type":"text","text":"upstream says hi"}}],"model":"m","stop_reason":"end_turn","stop_sequence":null,"usage":{{"input_tokens":{tokens},"output_tokens":2}}}}"#,
                );
                (
                    StatusCode::OK,
                    [("content-type", "application/json")],
                    body,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                tracing::trace!(%error, "reload-test upstream stopped");
            }
        });
        (format!("http://{addr}/v1/messages"), addr)
    }

    fn engine_request(provider: &str) -> EngineRequest {
        EngineRequest::new(
            provider,
            llm_proxy_core::ProviderRouteKind::Messages,
            ClientDialect::Anthropic,
            CoreRequest {
                model: ModelRef {
                    requested: "m".to_owned(),
                    upstream: None,
                },
                messages: vec![CoreMessage {
                    role: CoreRole::User,
                    content: vec![CoreContent::Text {
                        text: "hi".to_owned(),
                        cache: None,
                    }],
                }],
                system: Vec::new(),
                tools: Vec::new(),
                tool_choice: None,
                sampling: SamplingOptions::default(),
                stream: false,
                metadata: Default::default(),
                provider_hints: Default::default(),
            },
            InboundRequest {
                auth: None,
                headers: HashMap::new(),
                wire: ClientWireRequest::new(
                    ClientWireFormat::AnthropicMessages,
                    serde_json::json!({}),
                ),
            },
        )
        .with_request_id("req-reload-test")
        .with_body_hash("hash")
    }

    async fn state_for(dir: &Path) -> llm_proxy_server::AppState {
        crate::state::load_toml_state(
            &dir.join("config.toml"),
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            None,
        )
        .await
        .expect("state builds")
    }

    /// The plan's Done-when: a reload that changes one provider leaves
    /// in-flight requests on other providers running — and the changed
    /// provider serves its new upstream afterwards.
    #[tokio::test]
    async fn a_provider_swap_leaves_in_flight_requests_on_other_providers_running() {
        let dir = tempfile::tempdir().expect("tempdir");
        // slow upstream for provider "slow", fast for "quick". The usage
        // values differ (3 / 5 / 7) so the assertions can tell which
        // upstream actually served.
        let (slow_endpoint, _slow_addr) = spawn_echo_upstream(Duration::from_millis(800), 3).await;
        let (quick_endpoint, _quick_addr) = spawn_echo_upstream(Duration::ZERO, 5).await;
        write_provider(dir.path(), "slow", &slow_endpoint, 1);
        write_provider(dir.path(), "quick", &quick_endpoint, 2);
        let _config = write_config(dir.path(), "");

        let state = state_for(dir.path()).await;
        let engine = state.engine().expect("engine builds");

        // Fire a request at the slow provider; it will still be in flight
        // when the reload lands.
        let in_flight = {
            let engine = std::sync::Arc::clone(&engine);
            tokio::spawn(async move {
                engine
                    .execute(engine_request("slow"))
                    .await
                    .expect("the in-flight request on the untouched provider completes")
            })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Sanity: the old quick upstream is distinguishable — it serves 5.
        let before = engine
            .execute(engine_request("quick"))
            .await
            .expect("the pre-reload request reaches the old quick upstream");
        assert_eq!(before.core.usage.input_tokens, 5);

        // Change ONLY the quick provider: point it at a new upstream.
        let (new_quick_endpoint, _new_addr) = spawn_echo_upstream(Duration::ZERO, 7).await;
        write_provider(dir.path(), "quick", &new_quick_endpoint, 3);
        let report = reload_registry(&state, &dir.path().join("config.toml"), None)
            .await
            .expect("reload applies");
        assert_eq!(
            report.changed,
            vec!["quick".to_owned()],
            "exactly the changed provider is reported: {report}"
        );

        // The in-flight request on the untouched provider finishes fine.
        let done = tokio::time::timeout(Duration::from_secs(10), in_flight)
            .await
            .expect("in-flight join")
            .expect("in-flight task");
        assert_eq!(done.core.usage.input_tokens, 3);

        // And a fresh request reaches the reloaded provider's new upstream.
        let fresh = state.engine().expect("post-reload engine");
        let response = fresh
            .execute(engine_request("quick"))
            .await
            .expect("the reloaded provider serves");
        assert_eq!(response.core.usage.input_tokens, 7);
    }

    /// Restart-required refusal: a `[server]` change refuses the reload and
    /// applies nothing — the changed provider file is NOT picked up.
    #[tokio::test]
    async fn a_server_change_refuses_the_reload_and_applies_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        let _initial = write_config(dir.path(), "");

        let state = state_for(dir.path()).await;
        let engine_before = state.engine().expect("engine builds");
        let registry_before = state.providers();

        // Touch the provider AND the server section: the refusal must win,
        // and the provider change must not half-apply.
        write_provider(dir.path(), "prov", &endpoint, 9);
        write_config(dir.path(), "\nrequest_timeout = \"120s\"\n");
        // (rewrite with the same base but a new timeout)
        let path = {
            let p = dir.path().join("config.toml");
            std::fs::write(
                &p,
                r#"[server]
bind = "127.0.0.1:3477"
request_timeout = "120s"
log_level = "info"
hot_reload = true
server_name = "reload-test"
"#,
            )
            .expect("config rewrite");
            p
        };

        let error = reload_registry(&state, &path, None)
            .await
            .expect_err("a server change must refuse the reload");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("restart-required:"),
            "the refusal carries the stable prefix operators grep for: {message}"
        );
        assert!(
            message.contains("server.request_timeout"),
            "the refusal names the exact field: {message}"
        );
        // Nothing was applied: the same engine serves, the same registry.
        let engine_after = state.engine().expect("engine");
        assert!(
            Arc::ptr_eq(&engine_before, &engine_after),
            "the engine was not swapped by a refused reload"
        );
        let registry_after = state.providers();
        assert!(
            Arc::ptr_eq(&registry_before, &registry_after),
            "the provider set was not changed by a refused reload"
        );
        // Nothing was applied to the config snapshot the serving engine
        // reads, either.
        let snapshot_before = state
            .config_source()
            .load()
            .expect("the serving snapshot loads");
        assert_eq!(
            snapshot_before.server.request_timeout,
            Duration::from_secs(300),
            "the snapshot still serves the startup config, not the refused file"
        );
    }

    /// A broken provider file refuses the whole reload rather than
    /// half-applying.
    #[tokio::test]
    async fn a_broken_provider_file_refuses_the_whole_reload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "good", &endpoint, 1);
        write_provider(dir.path(), "also-good", &endpoint, 2);
        let path = write_config(dir.path(), "");
        let state = state_for(dir.path()).await;
        let engine_before = state.engine().expect("engine builds");

        // Corrupt one provider file.
        std::fs::write(dir.path().join("providers/good.toml"), "not toml {{{").expect("corrupt");
        let error = reload_registry(&state, &path, None)
            .await
            .expect_err("a broken file refuses the reload");
        assert!(
            format!("{error:#}").contains("nothing was applied"),
            "the error says nothing was applied: {error:#}"
        );
        let engine_after = state.engine().expect("engine");
        assert!(
            Arc::ptr_eq(&engine_before, &engine_after),
            "the engine was not swapped by a refused reload"
        );
    }

    /// A replacement engine that cannot be built must leave the serving
    /// config snapshot untouched too: the snapshot is shared with the live
    /// engine, so advancing it before the build would make a "refused"
    /// reload change what the serving engine reads.
    #[cfg(feature = "semantic-cache")]
    #[tokio::test]
    async fn a_refused_engine_build_leaves_the_serving_config_snapshot_untouched() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        write_config(dir.path(), "");
        let state = state_for(dir.path()).await;
        state.engine().expect("engine builds");
        let snapshot_before = state
            .config_source()
            .load()
            .expect("the serving snapshot loads");

        // A proposed config the engine build refuses (both cache opt-ins):
        // the reload path builds from exactly this kind of proposal before
        // anything live advances.
        let mut proposal = (*snapshot_before).clone();
        proposal.server.cache.enabled = true;
        proposal.server.semantic_cache.enabled = true;
        let error = state
            .build_engine_from_parts_with_config(&state.providers(), HashMap::new(), &proposal)
            .expect_err("the conflicting proposal must refuse the build");
        assert!(
            error.to_string().contains("mutually exclusive"),
            "got {error}"
        );

        let snapshot_after = state
            .config_source()
            .load()
            .expect("the serving snapshot loads");
        assert!(
            Arc::ptr_eq(&snapshot_before, &snapshot_after),
            "a refused build must not advance the serving config snapshot"
        );
    }

    /// Adding and removing providers round-trips through the report.
    #[tokio::test]
    async fn adding_and_removing_providers_is_reported() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "keep", &endpoint, 1);
        write_provider(dir.path(), "drop-me", &endpoint, 2);
        let path = write_config(dir.path(), "");
        let state = state_for(dir.path()).await;
        state.engine().expect("engine builds");

        std::fs::remove_file(dir.path().join("providers/drop-me.toml")).expect("remove");
        write_provider(dir.path(), "fresh", &endpoint, 3);
        let report = reload_registry(&state, &path, None)
            .await
            .expect("reload applies");
        assert_eq!(report.removed, vec!["drop-me".to_owned()]);
        assert_eq!(report.added, vec!["fresh".to_owned()]);
        assert!(
            report.changed.is_empty(),
            "untouched providers are untouched: {report}"
        );
        // The live set reflects it.
        let live = state.providers();
        assert!(live.get("fresh").is_some());
        assert!(live.get("drop-me").is_none());
    }

    /// Audit finding 2: a file edit of `server.bind` used to be silently
    /// masked before the restart-required diff, so the operator got a
    /// successful reload while the server kept the old bind.
    #[tokio::test]
    async fn a_bind_edit_is_refused_as_restart_required_not_masked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        let _startup_config = write_config(dir.path(), "");

        let state = state_for(dir.path()).await;
        let engine_before = state.engine().expect("engine builds");

        // Move the port in the file; nothing else changes.
        let moved = {
            let p = dir.path().join("config.toml");
            std::fs::write(
                &p,
                r#"[server]
bind = "127.0.0.1:3478"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "reload-test"
"#,
            )
            .expect("config rewrite");
            p
        };

        let error = reload_registry(&state, &moved, None)
            .await
            .expect_err("a bind edit cannot be applied by a reload");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("restart-required:"),
            "the refusal carries the stable prefix operators grep for: {message}"
        );
        assert!(
            message.contains("server.bind"),
            "the refusal names the bind, which was previously masked: {message}"
        );
        let engine_after = state.engine().expect("engine");
        assert!(
            Arc::ptr_eq(&engine_before, &engine_after),
            "the engine was not swapped by a refused reload"
        );
    }

    /// The CLI port override outranks the file on reload too: the server
    /// keeps its `--port`, and the file's own port value does not turn every
    /// reload into a spurious `server.bind` refusal.
    #[tokio::test]
    async fn a_cli_port_override_masks_only_the_port_across_reloads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        let path = write_config(dir.path(), "");

        let state = crate::state::load_toml_state(
            &path,
            llm_proxy_adapters::ProviderAdapterRegistry::builtin(),
            Some(3999),
        )
        .await
        .expect("state builds with the CLI port");
        state.engine().expect("engine builds");

        // Change the provider only; the file still says port 3477 while the
        // server runs on the overridden 3999.
        write_provider(dir.path(), "prov", &endpoint, 7);
        let report = reload_registry(&state, &path, Some(3999))
            .await
            .expect("the CLI port stays in force; the reload applies");
        assert_eq!(
            report.changed,
            vec!["prov".to_owned()],
            "exactly the provider change is applied: {report}"
        );
    }

    /// Audit finding 1: a verifier rotated under an existing key id used to
    /// pass the diff unapplied — the old (possibly leaked) token kept
    /// working until restart. It must be refused, naming the key.
    #[tokio::test]
    async fn a_same_id_verifier_rotation_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        let path = {
            let p = dir.path().join("config.toml");
            std::fs::write(
                &p,
                r#"[server]
bind = "127.0.0.1:3477"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "reload-test"

[auth]
mode = "client-keys"
[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$bi0KNLfLnc1qQxSai7uQDVMEMeSRpD/+rFoYCc/kku8"
"#,
            )
            .expect("config file");
            p
        };

        let state = state_for(dir.path()).await;
        let engine_before = state.engine().expect("engine builds");

        // Rotate the verifier under the same key id, touching nothing else.
        std::fs::write(
            &path,
            r#"[server]
bind = "127.0.0.1:3477"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "reload-test"

[auth]
mode = "client-keys"
[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$KXr5NGmScBYNmF2Vd4mjyfLxyGLpTMhPn5VmRQMb0fw"
"#,
        )
        .expect("config rewrite");

        let error = reload_registry(&state, &path, None)
            .await
            .expect_err("a same-id verifier change cannot be applied by a reload");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("restart-required:"),
            "the refusal carries the stable prefix operators grep for: {message}"
        );
        assert!(
            message.contains("auth.clients.laptop"),
            "the refusal names the rotated key: {message}"
        );
        let engine_after = state.engine().expect("engine");
        assert!(
            Arc::ptr_eq(&engine_before, &engine_after),
            "the engine was not swapped by a refused reload"
        );
    }

    /// A successful reload advances the config snapshot the engine reads, so
    /// the engine is never left serving startup values while the registry
    /// swaps around it.
    #[tokio::test]
    async fn a_successful_reload_advances_the_config_snapshot() {
        use llm_proxy_engine::ConfigSource;

        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        let path = write_config(dir.path(), "");
        let state = state_for(dir.path()).await;
        state.engine().expect("engine builds");
        let snapshot_before = state.config_source().load().expect("startup snapshot");

        write_provider(dir.path(), "prov", &endpoint, 4);
        reload_registry(&state, &path, None)
            .await
            .expect("reload applies");
        let snapshot_after = state.config_source().load().expect("post-reload snapshot");

        assert!(
            !Arc::ptr_eq(&snapshot_before, &snapshot_after),
            "the engine's config snapshot advanced with the reload"
        );
        assert_eq!(
            serde_json::to_value(&*snapshot_before).ok(),
            serde_json::to_value(&*snapshot_after).ok(),
            "the app-level sections are unchanged; only the snapshot identity moved"
        );
    }

    const PHC_A: &str = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$bi0KNLfLnc1qQxSai7uQDVMEMeSRpD/+rFoYCc/kku8";
    const PHC_B: &str = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$KXr5NGmScBYNmF2Vd4mjyfLxyGLpTMhPn5VmRQMb0fw";

    fn write_auth_config(dir: &Path, clients: &[(&str, &str)]) -> PathBuf {
        let path = dir.join("config.toml");
        let mut body = String::from(
            "[server]\nbind = \"127.0.0.1:3477\"\nrequest_timeout = \"300s\"\nlog_level = \
             \"info\"\nhot_reload = true\nserver_name = \"reload-test\"\n\n[auth]\nmode = \
             \"client-keys\"\n\n[auth.clients]\n",
        );
        for (id, phc) in clients {
            body.push_str(&format!("{id} = \"{phc}\"\n"));
        }
        std::fs::write(&path, body).expect("config file");
        path
    }

    /// With a managed key store live, the running config carries ids the
    /// re-read file never had (the startup merge), so every reload would be
    /// refused by an `auth.clients` diff that reads like a file edit. The
    /// refusal must attribute that diff to the runtime-issued keys instead.
    #[tokio::test]
    async fn a_managed_keys_auth_diff_is_attributed_not_blamed_on_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        // Startup: the merged running config holds the operator key and the
        // store-issued key; the file below already lacks the store's id.
        write_auth_config(dir.path(), &[("laptop", PHC_A), ("issued", PHC_B)]);
        let state = state_for(dir.path())
            .await
            .with_runtime_issued_keys(vec!["issued".to_owned()]);

        // The re-read file keeps only the operator's own key.
        let path = write_auth_config(dir.path(), &[("laptop", PHC_A)]);

        let error = reload_registry(&state, &path, None)
            .await
            .expect_err("the merged config's extra ids are still restart-required");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("restart-required:"),
            "the refusal keeps the stable prefix: {message}"
        );
        assert!(
            message.contains("auth.clients"),
            "the diff names auth.clients: {message}"
        );
        assert!(
            message.contains("runtime-issued managed keys (issued)"),
            "the refusal attributes the diff to the store's key, not the file's own \
             clients: {message}"
        );
        assert!(
            message.contains("[auth.admin.key_store]"),
            "the attribution names where the runtime keys came from: {message}"
        );
    }

    /// A genuine file edit — an operator key removed from the file — must
    /// not wear the runtime-issued attribution: the id the running config
    /// has and the file lacks is not one the store issued.
    #[tokio::test]
    async fn a_removed_file_key_is_not_attributed_to_managed_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        write_auth_config(dir.path(), &[("laptop", PHC_A), ("phone", PHC_B)]);
        let state = state_for(dir.path())
            .await
            .with_runtime_issued_keys(vec!["issued".to_owned()]);

        // The operator removed the "phone" key from the file.
        let path = write_auth_config(dir.path(), &[("laptop", PHC_A)]);

        let error = reload_registry(&state, &path, None)
            .await
            .expect_err("a removed file key is restart-required");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("restart-required:") && message.contains("auth.clients"),
            "the refusal keeps its plain form: {message}"
        );
        assert!(
            !message.contains("runtime-issued"),
            "a removed file key is a file edit, not a managed-keys artifact: {message}"
        );
    }

    /// An added `[auth.admin] key_store` cannot be applied by a reload (the
    /// store is opened once at startup): the diff must refuse it as its own
    /// field and apply nothing.
    #[tokio::test]
    async fn a_key_store_edit_is_refused_as_restart_required() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (endpoint, _addr) = spawn_echo_upstream(Duration::ZERO, 3).await;
        write_provider(dir.path(), "prov", &endpoint, 1);
        let dir_path = dir.path().display().to_string();
        let config = |key_store: &str| {
            format!(
                r#"[server]
bind = "127.0.0.1:3477"
request_timeout = "300s"
log_level = "info"
hot_reload = true
server_name = "reload-test"

[auth]
mode = "client-keys"

[auth.clients]
laptop = "{PHC_A}"

[auth.admin]
token = "admin-tok"
{key_store}

[storage]
path = "{dir_path}/usage.db"
"#
            )
        };
        let path = {
            let p = dir.path().join("config.toml");
            std::fs::write(&p, config("")).expect("config file");
            p
        };

        let state = state_for(dir.path()).await;
        let engine_before = state.engine().expect("engine builds");

        // Add the key store under an unchanged admin token, touching
        // nothing else.
        let key_store_line = format!("key_store = \"{dir_path}/k.db\"");
        std::fs::write(&path, config(&key_store_line)).expect("config rewrite");

        let error = reload_registry(&state, &path, None)
            .await
            .expect_err("an added key store cannot be applied by a reload");
        let message = format!("{error:#}");
        assert!(
            message.starts_with("restart-required:"),
            "the refusal carries the stable prefix operators grep for: {message}"
        );
        assert!(
            message.contains("auth.admin.key_store"),
            "the refusal names the key store field: {message}"
        );
        let engine_after = state.engine().expect("engine");
        assert!(
            Arc::ptr_eq(&engine_before, &engine_after),
            "the engine was not swapped by a refused reload"
        );
    }
}
