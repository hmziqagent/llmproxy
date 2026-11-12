//! The `doctor` command (plan 07 step 9, backlog 26): live reachability
//! checks and optional one-token probes against every configured provider,
//! every failure carrying a named fix. Reachability costs nothing (DNS +
//! TCP); the one-token probe sends a real `max_tokens = 1` request through
//! the real transport and is therefore opt-in (`--probe`) — it can spend
//! money.

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_adapters::request::{AuthHeaders, ProxyRequest};
use llm_proxy_core::{
    AppConfig, ProviderConfig, ProviderRouteKind, provider_config::endpoint_without_query,
};
use tokio::net::TcpStream as TokioTcpStream;

use crate::paths::resolve_config;
use crate::state::load_registry_with_checks;

/// One doctor line: what was checked, how it ended, and the fix when it did
/// not pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckOutcome {
    /// The provider the check belongs to.
    pub(crate) provider: String,
    /// What was checked (`reachability <endpoint>`, `probe <route>`).
    pub(crate) check: String,
    /// PASS, FAIL or WARN.
    pub(crate) outcome: Outcome,
    /// What happened, in one line.
    pub(crate) detail: String,
    /// The named remedy for anything that is not a PASS.
    pub(crate) fix: Option<String>,
}

/// How a check ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The check passed.
    Pass,
    /// The check failed: the named fix is required.
    Fail,
    /// The check could not decide (unknown protocol for a probe, no route).
    Warn,
}

impl Outcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Warn => "WARN",
        }
    }
}

/// The whole report: one row per check, plus whether any provider failed.
#[derive(Debug, Clone, Default)]
pub(crate) struct DoctorReport {
    /// Every check, in provider order.
    pub(crate) rows: Vec<CheckOutcome>,
}

impl DoctorReport {
    /// Whether every check passed (warnings are not failures).
    #[must_use]
    pub(crate) fn all_passed(&self) -> bool {
        self.rows.iter().all(|row| row.outcome != Outcome::Fail)
    }

    /// The plain-text rendering the command prints.
    #[must_use]
    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        for row in &self.rows {
            out.push_str(&format!(
                "{:<outcome_width$} {:<provider_width$} {:<check_width$} {}\n",
                row.outcome.as_str(),
                row.provider,
                row.check,
                row.detail,
                outcome_width = 4,
                provider_width = 18,
                check_width = 34,
            ));
            if let Some(fix) = &row.fix {
                out.push_str(&format!("         fix: {fix}\n"));
            }
        }
        out
    }
}

/// Runs the doctor checks: config gates first (a broken config is the first
/// thing to fix), then per-provider reachability, then the optional
/// one-token probe.
///
/// # Errors
///
/// `anyhow` when the config itself cannot be loaded or fails its gates —
/// doctor cannot check a configuration the server would refuse to start
/// with, and that failure already carries its own fix in the error text.
pub(crate) async fn run_doctor(config_path: Option<&Path>, probe: bool) -> Result<DoctorReport> {
    let path = resolve_config(config_path);
    crate::config_validation::validate_toml_extension(&path)?;
    let app_config = llm_proxy_core::load_app_config(&path).context("loading config (doctor)")?;
    let adapter_registry = ProviderAdapterRegistry::builtin();
    // The same gates startup runs: an invalid provider set is reported as
    // the config error it is, not as per-provider noise.
    let registry = load_registry_with_checks(&app_config, &path, &adapter_registry)
        .context("validating config (doctor)")?;

    let mut rows = Vec::new();
    for provider in registry.iter() {
        rows.extend(reachability_checks(provider).await);
        if probe {
            rows.push(probe_check(provider, &app_config).await);
        }
    }
    Ok(DoctorReport { rows })
}

/// DNS + TCP reachability for every adapter and discovery endpoint.
async fn reachability_checks(provider: &ProviderConfig) -> Vec<CheckOutcome> {
    let mut rows = Vec::new();
    let mut targets: Vec<(&str, &str)> = provider
        .adapters
        .iter()
        .map(|(name, adapter)| (name.as_str(), adapter.endpoint.as_str()))
        .collect();
    if let Some(discovery) = &provider.discovery {
        targets.push(("discovery", discovery.endpoint.as_str()));
    }
    for (name, endpoint) in targets {
        rows.push(reachability(provider, name, endpoint).await);
    }
    rows
}

async fn reachability(provider: &ProviderConfig, name: &str, endpoint: &str) -> CheckOutcome {
    let check = format!("reachability {name}");
    let Some((host, port)) = host_port(endpoint) else {
        return CheckOutcome {
            provider: provider.name.clone(),
            check,
            outcome: Outcome::Fail,
            detail: format!("endpoint has no host:port: {endpoint}"),
            fix: Some("fix the endpoint URL in the provider's [provider.adapters]".to_owned()),
        };
    };
    // Resolve off the async runtime: DNS is blocking.
    let resolve_host = host.clone();
    let resolved = tokio::task::spawn_blocking(move || {
        (resolve_host.as_str(), port)
            .to_socket_addrs()
            .map(|addrs| addrs.collect::<Vec<SocketAddr>>())
    })
    .await
    .unwrap_or_else(|error| Err(std::io::Error::other(format!("join: {error}"))));

    let addresses = match resolved {
        Ok(addresses) if !addresses.is_empty() => addresses,
        Ok(_) => {
            return CheckOutcome {
                provider: provider.name.clone(),
                check,
                outcome: Outcome::Fail,
                detail: format!("\"{host}\" resolves to no address"),
                fix: Some("check the endpoint hostname and your DNS resolution".to_owned()),
            };
        }
        Err(error) => {
            return CheckOutcome {
                provider: provider.name.clone(),
                check,
                outcome: Outcome::Fail,
                detail: format!("cannot resolve \"{host}\": {error}"),
                fix: Some("check the endpoint hostname and your DNS resolution".to_owned()),
            };
        }
    };

    let timeout = Duration::from_secs(5);
    for address in &addresses {
        match tokio::time::timeout(timeout, TokioTcpStream::connect(address)).await {
            Ok(Ok(_)) => {
                return CheckOutcome {
                    provider: provider.name.clone(),
                    check,
                    outcome: Outcome::Pass,
                    detail: format!("reachable at {address}"),
                    fix: None,
                };
            }
            Ok(Err(error)) if error.kind() == std::io::ErrorKind::ConnectionRefused => {
                return CheckOutcome {
                    provider: provider.name.clone(),
                    check,
                    outcome: Outcome::Fail,
                    detail: format!("connection refused at {address}"),
                    fix: Some(format!(
                        "nothing is listening on {address}; check the endpoint port and that \
                         the upstream is running"
                    )),
                };
            }
            _ => continue,
        }
    }
    CheckOutcome {
        provider: provider.name.clone(),
        check,
        outcome: Outcome::Fail,
        detail: format!("no address for \"{host}\" accepted a connection within {timeout:?}"),
        fix: Some(
            "check your network egress to the upstream (and any [provider.egress] proxy)"
                .to_owned(),
        ),
    }
}

/// The one-token probe: a `max_tokens = 1` request through the real
/// transport with the provider's own credential, classified by what the
/// upstream answers.
async fn probe_check(provider: &ProviderConfig, config: &AppConfig) -> CheckOutcome {
    let check = "probe one-token".to_owned();
    let route_adapter = provider
        .routes
        .get(ProviderRouteKind::ChatCompletions)
        .or_else(|| provider.routes.get(ProviderRouteKind::Messages));
    let Some(adapter_name) = route_adapter else {
        return CheckOutcome {
            provider: provider.name.clone(),
            check,
            outcome: Outcome::Warn,
            detail: "no chat route configured; nothing to probe".to_owned(),
            fix: Some(
                "set [provider.routes] chat_completions or messages to enable probes".to_owned(),
            ),
        };
    };
    let Some(adapter) = provider.adapters.get(adapter_name) else {
        return CheckOutcome {
            provider: provider.name.clone(),
            check,
            outcome: Outcome::Warn,
            detail: format!("route names unknown adapter \"{adapter_name}\""),
            fix: Some("fix the adapter name in [provider.routes]".to_owned()),
        };
    };
    if !matches!(
        adapter.protocol.as_str(),
        "openai_chat_completions" | "anthropic_messages"
    ) {
        return CheckOutcome {
            provider: provider.name.clone(),
            check,
            outcome: Outcome::Warn,
            detail: format!("no one-token probe for protocol {}", adapter.protocol),
            fix: None,
        };
    }
    let model = probe_model(provider, &adapter.protocol);
    let body = format!(
        r#"{{"model":"{model}","max_tokens":1,"messages":[{{"role":"user","content":"ping"}}]}}"#
    );
    let request = ProxyRequest {
        url: adapter.endpoint.clone(),
        auth: AuthHeaders {
            style: provider.auth_style,
            api_key: provider.api_key.clone(),
        },
        body: body.into_bytes(),
        stream: false,
        extra_headers: std::collections::HashMap::new(),
        forwarded_headers: std::collections::HashMap::new(),
    };
    let client =
        match llm_proxy_transport::ProxyClient::try_with_timeouts(&config.server.upstream_timeouts)
        {
            Ok(client) => client,
            Err(error) => {
                return CheckOutcome {
                    provider: provider.name.clone(),
                    check,
                    outcome: Outcome::Fail,
                    detail: format!("building the transport client failed: {error}"),
                    fix: Some(
                        "check the [server.upstream_timeouts] phases and the proxy/TLS \
                         environment reqwest builds under"
                            .to_owned(),
                    ),
                };
            }
        };
    let detail_prefix = format!(
        "model {model} at {}",
        endpoint_without_query(&adapter.endpoint)
    );
    match client.send(request).await {
        Ok(_) => CheckOutcome {
            provider: provider.name.clone(),
            check,
            outcome: Outcome::Pass,
            detail: format!("{detail_prefix}: the upstream answered 2xx"),
            fix: None,
        },
        Err(error) => classify_probe_error(provider, check, &detail_prefix, &error),
    }
}

/// Turns a probe's failure into a row whose fix names the remedy the status
/// implies.
fn classify_probe_error(
    provider: &ProviderConfig,
    check: String,
    detail_prefix: &str,
    error: &llm_proxy_adapters::ProviderError,
) -> CheckOutcome {
    let row = |outcome, detail: String, fix: Option<&str>| CheckOutcome {
        provider: provider.name.clone(),
        check,
        outcome,
        detail,
        fix: fix.map(str::to_owned),
    };
    let llm_proxy_adapters::ProviderError::Api { status, .. } = error else {
        return row(
            Outcome::Fail,
            format!("{detail_prefix}: {error}"),
            Some(
                "the request never reached a usable answer; check the endpoint URL and your \
                 network egress",
            ),
        );
    };
    match status {
        401 | 403 => row(
            Outcome::Fail,
            format!("{detail_prefix}: rejected with {status}"),
            Some(&format!(
                "check this provider's api_key{}",
                if provider.passthrough_auth {
                    " (passthrough_auth forwards the caller's token, which doctor cannot present)"
                } else {
                    ""
                }
            )),
        ),
        429 => row(
            Outcome::Fail,
            format!("{detail_prefix}: rate limited (429)"),
            Some("the upstream is throttling or out of quota; check the provider account"),
        ),
        500..=599 => row(
            Outcome::Fail,
            format!("{detail_prefix}: upstream error ({status})"),
            Some("the upstream is unhealthy; check the provider's status page"),
        ),
        other => row(
            Outcome::Warn,
            format!("{detail_prefix}: answered {other}"),
            Some(
                "a non-2xx answer with valid transport usually means the endpoint path or the \
                 probed model id is wrong; check [provider.adapters] endpoint and the model ids",
            ),
        ),
    }
}

/// The model a probe should name: an alias target first (it is guaranteed to
/// exist upstream), then a priced model, then a protocol default. Both map
/// picks are by sorted key: HashMap iteration order would send repeated runs
/// at different models and make a "wrong model id" FAIL flaky.
fn probe_model(provider: &ProviderConfig, protocol: &str) -> String {
    if let Some((_, target)) = provider
        .model_aliases
        .iter()
        .min_by_key(|(alias, _)| *alias)
    {
        return target.clone();
    }
    if let Some(model) = provider.pricing.keys().min_by_key(|model| model.as_str()) {
        return model.as_str().to_owned();
    }
    if protocol == "anthropic_messages" {
        "claude-sonnet-4-6".to_owned()
    } else {
        "gpt-4o".to_owned()
    }
}

/// The `host:port` of an endpoint URL with scheme-derived default port.
fn host_port(endpoint: &str) -> Option<(String, u16)> {
    let (scheme, rest) = endpoint.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    let default_port = match scheme {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
        let (host, rest) = inner.split_once(']')?;
        let port = rest.strip_prefix(':').and_then(|p| p.parse::<u16>().ok());
        (host.to_owned(), port.unwrap_or(default_port))
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => (host.to_owned(), port.parse::<u16>().ok()?),
            None => (authority.to_owned(), default_port),
        }
    };
    (!host.is_empty()).then_some((host, port))
}

/// The CLI entry point: run, print, exit non-zero on any FAIL.
///
/// # Errors
///
/// `anyhow` when the config cannot be loaded or fails its gates.
pub async fn cmd_doctor(config_path: Option<&Path>, probe: bool) -> Result<()> {
    let report = run_doctor(config_path, probe).await?;
    print!("{}", report.render());
    if report.all_passed() {
        println!("doctor: no failures");
        return Ok(());
    }
    println!("doctor: failures found (see fixes above)");
    // A failing doctor is a failing command, so scripts can branch on it.
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(dir: &std::path::Path, endpoint: &str, name: &str) -> std::path::PathBuf {
        std::fs::create_dir_all(dir.join("providers")).expect("providers dir");
        std::fs::write(
            dir.join("providers/probe.toml"),
            format!(
                r#"[provider]
name = "{name}"
api_key = "doctor-test-key"
auth_style = "x-api-key"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "{endpoint}"
[provider.adapters.messages]
protocol = "anthropic_messages"
endpoint = "{endpoint}"
[provider.routes]
chat_completions = "chat"
messages = "messages"
"#
            ),
        )
        .expect("provider file");
        let path = dir.join("config.toml");
        std::fs::write(
            &path,
            r#"[server]
bind = "127.0.0.1:3499"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "doctor-test"
"#,
        )
        .expect("config file");
        path
    }

    async fn spawn_upstream(status: u16) -> String {
        use axum::Router;
        let app = Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || async move {
                (
                    axum::http::StatusCode::from_u16(status)
                        .unwrap_or(axum::http::StatusCode::BAD_REQUEST),
                    "{}",
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let serve = axum::serve(listener, app);
            if let Err(error) = serve.await {
                tracing::warn!(%error, "doctor test upstream stopped");
            }
        });
        format!("http://{addr}/v1/chat/completions")
    }

    #[tokio::test]
    async fn a_live_upstream_passes_both_checks() {
        let endpoint = spawn_upstream(200).await;
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(dir.path(), &endpoint, "healthy");
        let report = run_doctor(Some(&path), true).await.expect("doctor runs");
        assert!(
            report.all_passed(),
            "every check against a 2xx upstream passes: {}",
            report.render()
        );
        let probe = report
            .rows
            .iter()
            .find(|row| row.check == "probe one-token")
            .expect("the probe ran");
        assert_eq!(probe.outcome, Outcome::Pass);
        assert!(
            probe.detail.contains("2xx"),
            "the probe names what the upstream answered: {}",
            probe.detail
        );
    }

    #[tokio::test]
    async fn rejected_credentials_fail_with_a_named_fix() {
        let endpoint = spawn_upstream(401).await;
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(dir.path(), &endpoint, "badkey");
        let report = run_doctor(Some(&path), true).await.expect("doctor runs");
        let probe = report
            .rows
            .iter()
            .find(|row| row.check == "probe one-token")
            .expect("the probe ran");
        assert_eq!(probe.outcome, Outcome::Fail);
        let fix = probe.fix.as_deref().expect("a failure carries its fix");
        assert!(fix.contains("api_key"), "the fix names the remedy: {fix}");
        assert!(!report.all_passed());
    }

    #[tokio::test]
    async fn an_unreachable_endpoint_fails_reachability_with_a_named_fix() {
        // Bind then drop the listener: the port is very likely still free.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(dir.path(), &format!("http://{addr}/v1/x"), "dead");
        let report = run_doctor(Some(&path), false).await.expect("doctor runs");
        let reach = report
            .rows
            .iter()
            .find(|row| row.check.starts_with("reachability chat"))
            .expect("reachability ran");
        assert_eq!(reach.outcome, Outcome::Fail);
        let fix = reach.fix.as_deref().expect("the failure carries a fix");
        assert!(
            fix.contains("port") || fix.contains("listening"),
            "the fix names the remedy: {fix}"
        );
    }

    #[tokio::test]
    async fn every_row_is_a_pass_or_carries_a_named_fix() {
        // The Done-when property itself, on a mixed report: PASS rows carry
        // no fix; everything else does.
        let endpoint = spawn_upstream(404).await;
        let dir = tempfile::tempdir().unwrap();
        let path = write_config(dir.path(), &endpoint, "odd");
        let report = run_doctor(Some(&path), true).await.expect("doctor runs");
        assert!(!report.rows.is_empty());
        for row in &report.rows {
            match row.outcome {
                Outcome::Pass => assert!(row.fix.is_none(), "a pass needs no fix: {}", row.detail),
                Outcome::Fail | Outcome::Warn => assert!(
                    row.fix.as_deref().is_some_and(|fix| !fix.is_empty()),
                    "every non-pass names its fix: {} ({})",
                    row.check,
                    row.detail
                ),
            }
        }
    }

    /// Audit finding 7: several aliases used to make the probed model depend
    /// on HashMap iteration order, so a "wrong model id" FAIL was flaky.
    /// The probe must name the same model on every run: the target of the
    /// lexicographically first alias.
    #[test]
    fn probe_model_picks_the_first_alias_by_name_not_hash_order() {
        let provider = toml::from_str::<llm_proxy_core::ProviderFile>(
            r#"[provider]
name = "probe-determinism"
api_key = "k"
auth_style = "bearer"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"
[provider.routes]
chat_completions = "chat"
[provider.model_aliases]
"zz-alias" = "z-upstream-model"
"aa-alias" = "a-upstream-model"
"mm-alias" = "m-upstream-model"
"#,
        )
        .expect("provider parses")
        .provider;

        assert_eq!(
            probe_model(&provider, "openai_chat_completions"),
            "a-upstream-model",
            "the probe names the first alias by name, whatever the HashMap order"
        );
    }

    #[test]
    fn probe_model_falls_back_to_the_first_priced_model_deterministically() {
        let provider = toml::from_str::<llm_proxy_core::ProviderFile>(
            r#"[provider]
name = "priced-probe"
api_key = "k"
auth_style = "bearer"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://example.com/v1/chat/completions"
[provider.routes]
chat_completions = "chat"
[provider.pricing]
"zz-model" = { input = "0.001", output = "0.002" }
"aa-model" = { input = "0.001", output = "0.002" }
"#,
        )
        .expect("provider parses")
        .provider;

        assert_eq!(
            probe_model(&provider, "openai_chat_completions"),
            "aa-model",
            "without aliases the probe names the first priced model by name"
        );
    }
}
