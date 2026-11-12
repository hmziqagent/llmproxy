use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::serve::{Listener, ListenerExt};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::LogFormat;
#[cfg(not(feature = "admin-ui"))]
use llm_proxy_server::admin_router;
use llm_proxy_server::{build_router, shutdown_signal};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::info;

use crate::config_validation::validate_toml_extension;
use crate::paths::{config_dir, resolve_config};
use crate::pid::{is_process_running, read_pid, remove_pid, write_pid, write_pid_value};
use crate::state::init_tracing;

/// Hot reload (plan 07 step 11): the SIGHUP listener needs the state
/// before the router takes it.
fn start_hot_reload(state: &llm_proxy_server::AppState, path: PathBuf, port_override: Option<u16>) {
    let reload_state = state.clone();
    let hot_reload_enabled = state.app_config().server.hot_reload;
    if hot_reload_enabled {
        info!(
            "hot reload enabled: SIGHUP (or `llm-proxy reload`) re-reads the provider files; \
             [server]/[auth]/[storage] changes are refused with a restart-required error"
        );
    }
    spawn_reload_listener(reload_state, path, hot_reload_enabled, port_override);
}

/// Loads the TOML config and builds the serving state: the config loads
/// before the subscriber is built so `server.log_format` is honoured
/// (BUG-25, with `RUST_LOG_FORMAT` still winning), then the state —
/// providers, auth, ledger, replica stores — comes up from it.
async fn load_serve_state(
    path: &Path,
    port_override: Option<u16>,
) -> Result<llm_proxy_server::AppState> {
    let app_config = llm_proxy_core::load_app_config(path)
        .with_context(|| format!("loading TOML config from {}", path.display()))?;
    let app_config = apply_port_override(app_config, port_override);
    init_tracing(&crate::state::startup_log_format(&app_config));

    let adapter_registry = ProviderAdapterRegistry::builtin();
    crate::state::load_toml_state_from_config(app_config, path, adapter_registry).await
}

/// The `--background` path: the parent only logs spawn failures; the
/// daemon child runs this same command with `--daemonize` and initializes
/// tracing itself from the resolved config.
fn spawn_in_background(config_path: Option<&Path>, port_override: Option<u16>) -> Result<()> {
    init_tracing(&LogFormat::from_env());
    spawn_daemon(config_path, port_override)
}

/// Applies the `--port` CLI override to the loaded config (step 10's stated
/// precedence: CLI over config).
fn apply_port_override(
    mut app_config: llm_proxy_core::AppConfig,
    port_override: Option<u16>,
) -> llm_proxy_core::AppConfig {
    if let Some(port) = port_override {
        let mut bind = app_config.server.bind;
        bind.set_port(port);
        app_config.server.bind = bind;
    }
    app_config
}

/// Run the `serve` command.
///
/// Requires TOML configuration.
pub async fn cmd_serve(
    config_path: Option<PathBuf>,
    port_override: Option<u16>,
    background: bool,
    daemonize: bool,
) -> Result<()> {
    let path = resolve_config(config_path.as_deref());
    validate_toml_extension(&path)?;

    if background && !daemonize {
        return spawn_in_background(config_path.as_deref(), port_override);
    }

    let state = load_serve_state(&path, port_override).await?;

    let admin = bind_admin_listener(&state).await?;
    let mcp = bind_mcp_gateway(&state).await?;

    claim_pid_file()?;

    let bind_addr = state.bind_address();
    let shutdown_deadline = state.shutdown_timeout();
    let event_bus = std::sync::Arc::clone(state.event_bus());
    // Opt-in OTLP push of the runtime counters (plan 07 step 7): the guard
    // keeps the export loop alive until the server stops, then flushes.
    let otlp_guard = match state.app_config().server.metrics.otlp.as_ref() {
        Some(otlp) => Some(
            crate::otel_metrics::start_otlp_metrics(state.metrics(), &otlp.endpoint, otlp.interval)
                .with_context(|| "starting OTLP metric export from [server.metrics.otlp]")?,
        ),
        None => None,
    };
    // Hot reload (plan 07 step 11): the SIGHUP listener needs the state
    // before the router takes it.
    start_hot_reload(&state, path.clone(), port_override);

    let app = build_router(state);

    let listener = bind_listener(bind_addr).await?.tap_io(|tcp_stream| {
        if let Err(error) = tcp_stream.set_nodelay(true) {
            tracing::trace!(%error, "failed to set TCP_NODELAY on accepted connection");
        }
    });
    info!(
        addr = %listener.local_addr()?,
        version = env!("CARGO_PKG_VERSION"),
        "llm-proxy listening"
    );

    let shutdown = spawn_shutdown_listener();

    let serve = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(wait_for_shutdown(shutdown.clone()));

    let admin_serve = async {
        if let Some((listener, router)) = admin {
            if let Err(error) = axum::serve(listener, router.into_make_service())
                .with_graceful_shutdown(wait_for_shutdown(shutdown.clone()))
                .await
            {
                tracing::error!(%error, "admin surface listener failed");
            }
        }
    };
    let mcp_serve = async {
        if let Some((listener, router)) = mcp {
            if let Err(error) = axum::serve(listener, router.into_make_service())
                .with_graceful_shutdown(wait_for_shutdown(shutdown.clone()))
                .await
            {
                tracing::error!(%error, "MCP gateway listener failed");
            }
        }
    };

    let servers = async {
        let (main, (), ()) = tokio::join!(serve.into_future(), admin_serve, mcp_serve);
        main
    };
    let Some(result) = drain_within_deadline(
        servers,
        wait_for_shutdown(shutdown.clone()),
        shutdown_deadline,
    )
    .await
    else {
        force_exit_after_deadline(shutdown_deadline);
    };

    if let Err(e) = remove_pid() {
        tracing::warn!(error = %e, "failed to remove PID file during shutdown");
    }

    // Dropping the guard flushes the last OTLP export before the process
    // goes away.
    drop(otlp_guard);

    if let Err(close_err) = tokio::task::spawn_blocking(move || event_bus.close()).await {
        tracing::warn!(error = %close_err, "usage-ledger close task failed to join");
    }

    match result {
        Ok(()) => {
            info!("server stopped cleanly");
            Ok(())
        }
        Err(e) => {
            tracing::error!(error = %e, "server error");
            Err(anyhow::anyhow!("server error: {}", e))
        }
    }
}

/// Writes the daemon's PID file, clearing a stale entry when the recorded
/// process is gone and refusing when another server is alive.
fn claim_pid_file() -> Result<()> {
    if let Err(first_err) = write_pid() {
        if let Some(existing_pid) = read_pid()? {
            if is_process_running(existing_pid) {
                bail!(
                    "server already running with PID {} (PID file: {})",
                    existing_pid,
                    crate::paths::pid_file_path().display()
                );
            }
            info!("stale PID file found (PID {}), cleaning up", existing_pid);
        }
        remove_pid()?;
        write_pid().with_context(|| {
            format!(
                "failed to create PID file after removing stale entry: {}",
                first_err
            )
        })?;
    }
    Ok(())
}

/// Starts the one signal listener shared by every shutdown waiter, so the drain
/// deadline begins at the moment the servers begin to drain.
fn spawn_shutdown_listener() -> watch::Receiver<bool> {
    let (started, shutdown) = watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        started.send_replace(true);
    });
    shutdown
}

/// Listens for SIGHUP and reloads the provider and pool files (plan 07 step
/// 11). Without `server.hot_reload = true` the signal is acknowledged and
/// ignored, so an operator poking the daemon changes nothing silently.
/// `port_override` is the `--port` the server was started with, reapplied to
/// every re-read config so the CLI layer keeps outranking the file.
#[cfg(unix)]
fn spawn_reload_listener(
    state: llm_proxy_server::AppState,
    config_path: PathBuf,
    enabled: bool,
    port_override: Option<u16>,
) {
    tokio::spawn(async move {
        let mut hangup = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
        {
            Ok(hangup) => hangup,
            Err(error) => {
                tracing::warn!(%error, "cannot listen for SIGHUP; hot reload unavailable");
                return;
            }
        };
        while hangup.recv().await.is_some() {
            if !enabled {
                tracing::info!(
                    "SIGHUP ignored: server.hot_reload = false; restart the server to apply \
                     config changes"
                );
                continue;
            }
            match crate::reload::reload_registry(&state, &config_path, port_override).await {
                Ok(report) => {
                    if report.is_empty() {
                        info!("reload: nothing changed");
                    } else {
                        info!("{report}");
                    }
                }
                Err(error) => tracing::error!(
                    %error,
                    "reload refused; the running configuration stands"
                ),
            }
        }
    });
}

#[cfg(not(unix))]
fn spawn_reload_listener(
    _state: llm_proxy_server::AppState,
    _config_path: PathBuf,
    _enabled: bool,
    _port_override: Option<u16>,
) {
    // SIGHUP does not exist on this platform; `llm-proxy reload` says so too.
}

async fn wait_for_shutdown(mut shutdown: watch::Receiver<bool>) {
    if let Err(error) = shutdown.wait_for(|started| *started).await {
        tracing::error!(%error, "shutdown listener stopped; beginning graceful shutdown");
    }
}

/// Runs `servers` to completion, allowing `deadline` for the drain once
/// `shutdown` fires. Returns `None` when connections outlast the deadline; a
/// zero `deadline` waits for the drain however long it takes.
async fn drain_within_deadline<T>(
    servers: impl Future<Output = T>,
    shutdown: impl Future<Output = ()>,
    deadline: Duration,
) -> Option<T> {
    tokio::pin!(servers);
    tokio::select! {
        finished = &mut servers => return Some(finished),
        () = shutdown => {}
    }
    if deadline.is_zero() {
        return Some(servers.await);
    }
    tokio::time::timeout(deadline, servers).await.ok()
}

fn force_exit_after_deadline(deadline: Duration) -> ! {
    tracing::error!(
        ?deadline,
        "graceful-shutdown deadline elapsed; forcing exit with in-flight connections dropped"
    );
    if let Err(e) = remove_pid() {
        tracing::warn!(error = %e, "failed to remove PID file during shutdown");
    }
    std::process::exit(1);
}

async fn bind_admin_listener(
    state: &llm_proxy_server::AppState,
) -> Result<Option<(TcpListener, axum::Router)>> {
    let Some(posture) = state.app_config().admin_posture() else {
        return Ok(None);
    };
    if !state.app_config().server.dedicated_admin_listener() {
        tracing::info!(
            posture = posture.name(),
            "admin usage API served on the main listener only; no dedicated socket"
        );
        return Ok(None);
    }
    let Some(addr) = state.app_config().server.effective_admin_bind() else {
        bail!("admin API enabled but no admin bind could be derived");
    };
    // The full admin surface (plan 09 steps 2-3) when the feature is on:
    // key management, usage and config views, the web UI and MCP tools.
    // Otherwise the read-only /usage API stands.
    #[cfg(feature = "admin-ui")]
    let router = {
        use llm_proxy_admin::AdminSurface;
        use secrecy::ExposeSecret;
        let Some(admin) = state.app_config().auth.admin.as_ref() else {
            bail!("admin posture enabled but [auth.admin] is not configured");
        };
        let mut surface = AdminSurface::new(admin.token.expose_secret());
        if let Some(keys) = crate::state::managed_key_store_for_serve(state.app_config()) {
            surface = surface.keys(keys);
        }
        if let Some(storage) = state.app_config().storage.as_ref() {
            surface = surface.usage(llm_proxy_storage::UsageReader::new(storage.path.clone()));
        }
        surface
            .config(std::sync::Arc::clone(&state.app_config_arc()))
            .router()
    };
    #[cfg(not(feature = "admin-ui"))]
    let Some(router) = admin_router(state) else {
        bail!("admin API enabled but no /usage router could be built");
    };
    let listener = bind_listener(addr).await?;
    info!(
        addr = %addr,
        posture = posture.name(),
        "admin usage API listening (read-only, behind the admin token)"
    );
    Ok(Some((listener, router)))
}

/// The MCP gateway listener (plan 09 step 6): `[mcp]` is its own surface
/// with its own tokens, sharing the engine's admission and the ledger.
#[cfg(feature = "mcp-gateway")]
async fn bind_mcp_gateway(
    state: &llm_proxy_server::AppState,
) -> Result<Option<(TcpListener, axum::Router)>> {
    use std::sync::Arc;
    let Some(config) = state.app_config().mcp.clone() else {
        return Ok(None);
    };
    if config.tokens.is_empty() {
        bail!("[mcp] is configured with no tokens: the gateway would refuse everyone");
    }
    let admission = state
        .engine()
        .map(|engine| engine.shared_admission())
        .map_err(|error| anyhow::anyhow!("building the engine for the MCP gateway: {error}"))?;
    let gateway = llm_proxy_mcp::McpGateway::new(
        config.clone(),
        admission,
        Arc::clone(state.event_bus()),
        Arc::new(
            llm_proxy_transport::ProxyClient::try_new()
                .context("building the MCP gateway transport client")?,
        ),
    );
    let addr = gateway.bind();
    let router = gateway.router();
    let listener = bind_listener(addr).await?;
    info!(
        addr = %addr,
        upstreams = config.upstreams.len(),
        "MCP gateway listening (own tokens, shared admission and ledger)"
    );
    Ok(Some((listener, router)))
}

#[cfg(not(feature = "mcp-gateway"))]
async fn bind_mcp_gateway(
    state: &llm_proxy_server::AppState,
) -> Result<Option<(TcpListener, axum::Router)>> {
    if state.app_config().mcp.is_some() {
        bail!(
            "[mcp] is configured but this build lacks the mcp-gateway feature; \
             rebuild with --features mcp-gateway"
        );
    }
    Ok(None)
}

async fn bind_listener(bind_addr: std::net::SocketAddr) -> Result<TcpListener> {
    #[cfg(unix)]
    {
        use socket2::{Domain, Protocol, Socket, TcpKeepalive};
        let domain = Domain::for_address(bind_addr);
        let socket = Socket::new(domain, socket2::Type::STREAM, Some(Protocol::TCP))
            .with_context(|| format!("creating socket for {bind_addr}"))?;
        socket
            .set_reuse_address(true)
            .with_context(|| "setting SO_REUSEADDR")?;
        let keepalive = TcpKeepalive::new().with_time(std::time::Duration::from_secs(60));
        socket
            .set_tcp_keepalive(&keepalive)
            .with_context(|| "enabling TCP keepalive")?;
        socket
            .bind(&bind_addr.into())
            .with_context(|| format!("binding to {bind_addr}"))?;
        socket
            .listen(1024)
            .with_context(|| format!("listening on {bind_addr}"))?;
        socket
            .set_nonblocking(true)
            .with_context(|| "setting nonblocking on listener")?;
        let std_listener = std::net::TcpListener::from(socket);
        Ok(TcpListener::from_std(std_listener)?)
    }
    #[cfg(not(unix))]
    {
        TcpListener::bind(bind_addr)
            .await
            .with_context(|| format!("binding to {bind_addr}"))
    }
}

fn spawn_daemon(config_path: Option<&Path>, port_override: Option<u16>) -> Result<()> {
    let exe = std::env::current_exe().with_context(|| "resolving current executable")?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("serve").arg("--daemonize");

    if let Some(p) = config_path {
        cmd.arg("--config").arg(p);
    }
    if let Some(p) = port_override {
        cmd.arg("--port").arg(p.to_string());
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        let log_path = config_dir().join("llm-proxy.log");
        let log_file = crate::permissions::open_private_append(&log_path)
            .with_context(|| format!("opening log file {}", log_path.display()))?;
        cmd.stdout(
            log_file
                .try_clone()
                .with_context(|| "cloning log file handle")?,
        );
        cmd.stderr(log_file);
    }

    #[cfg(not(unix))]
    {
        println!(
            "warning: daemon log redirection is not supported on this platform; \
                   output will be lost"
        );
    }

    let mut child = cmd.spawn().with_context(|| "spawning daemon process")?;

    let pid = child.id();

    if let Err(e) = write_pid_value(pid) {
        if let Err(kill_err) = child.kill() {
            tracing::error!(
                error = %kill_err,
                pid,
                "failed to kill orphaned daemon child after PID-file write failure"
            );
        }
        match child.wait() {
            Ok(status) => {
                tracing::info!(%status, "reaped daemon child after PID-file write failure");
            }
            Err(wait_err) => {
                tracing::warn!(error = %wait_err, "failed to reap daemon child");
            }
        }
        bail!(
            "daemon child (PID {pid}) was spawned but writing its PID file failed: {e}; \
             the child has been killed to avoid leaving an unmanageable daemon"
        );
    }

    std::thread::sleep(std::time::Duration::from_millis(100));
    match child.try_wait() {
        Ok(Some(status)) => {
            if let Err(e) = crate::pid::remove_pid() {
                tracing::warn!(error = %e, "failed to remove PID file after daemon child exited immediately");
            }
            bail!(
                "daemon child exited immediately with status {}",
                status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".to_owned())
            );
        }
        Ok(None) => {}
        Err(e) => {
            bail!("failed to check daemon child status: {}", e);
        }
    }

    println!("llm-proxy started in background (PID {pid})");
    println!("  config dir: {}", config_dir().display());
    println!(
        "  log file:   {}",
        config_dir().join("llm-proxy.log").display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::future::{pending, ready};
    use std::time::Duration;

    use super::drain_within_deadline;

    async fn finishes_after(delay: Duration) -> u8 {
        tokio::time::sleep(delay).await;
        7
    }

    #[tokio::test]
    async fn deadline_does_not_run_before_shutdown() {
        let result = drain_within_deadline(
            finishes_after(Duration::from_millis(50)),
            pending::<()>(),
            Duration::from_millis(1),
        )
        .await;
        assert_eq!(result, Some(7));
    }

    #[tokio::test]
    async fn drain_that_outlasts_the_deadline_is_cut_off() {
        let result =
            drain_within_deadline(pending::<u8>(), ready(()), Duration::from_millis(10)).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn drain_within_the_deadline_returns_its_result() {
        let result = drain_within_deadline(
            finishes_after(Duration::from_millis(5)),
            ready(()),
            Duration::from_secs(60),
        )
        .await;
        assert_eq!(result, Some(7));
    }

    #[tokio::test]
    async fn zero_deadline_waits_for_the_whole_drain() {
        let result = drain_within_deadline(
            finishes_after(Duration::from_millis(30)),
            ready(()),
            Duration::ZERO,
        )
        .await;
        assert_eq!(result, Some(7));
    }
}
