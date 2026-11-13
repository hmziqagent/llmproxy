//! Config provenance (backlog 34): each resolved setting printed with the
//! layer it came from. One stated precedence — CLI > env > config file >
//! default — and this module attributes the config file and default layers;
//! the env layer appears when a `${VAR}` interpolation resolved a value or a
//! documented env override (like `RUST_LOG_FORMAT`) wins; the CLI layer
//! (`--port`) is applied by the serve command before this report could see
//! it and is announced in the header instead.
//!
//! Only operator-facing settings are reported: no secrets (client verifiers,
//! the admin token, provider keys) ever appear.

use llm_proxy_core::AppConfig;

/// Where one resolved setting came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provenance {
    /// Set in the config file (possibly through `${VAR}` interpolation, in
    /// which case [`Provenance::Env`] names the variable instead).
    Config,
    /// Not set anywhere; the documented default applies.
    Default,
    /// An environment variable supplied the value: `${VAR}` interpolation in
    /// the file, or a documented env override (carries the variable name).
    Env(String),
}

impl std::fmt::Display for Provenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config => f.write_str("config"),
            Self::Default => f.write_str("default"),
            Self::Env(var) => write!(f, "env {var}"),
        }
    }
}

/// One resolved setting with its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingProvenance {
    /// Dotted setting path (`server.bind`).
    pub setting: String,
    /// The resolved value, rendered for display (never a secret).
    pub value: String,
    /// The layer the value came from.
    pub source: Provenance,
}

/// The documented env overrides that outrank the config file, read by the
/// caller so the report itself stays pure and testable.
#[derive(Debug, Clone, Default)]
pub struct RuntimeEnvOverrides {
    /// `RUST_LOG_FORMAT`, when set: it overrides `server.log_format`.
    pub rust_log_format: Option<String>,
}

/// The one stated precedence for every setting.
pub const PRECEDENCE: &str = "CLI > env > config file > default";

fn table_value<'t>(table: &'t toml::Table, path: &[&str]) -> Option<&'t toml::Value> {
    let (last, parents) = path.split_last()?;
    let mut current = table;
    for key in parents {
        current = current.get(*key)?.as_table()?;
    }
    current.get(*last)
}

/// Names the first `${VAR}` reference in a raw string value, if any.
fn env_var_of(value: &toml::Value) -> Option<String> {
    let text = value.as_str()?;
    let (_, after_marker) = text.split_once("${")?;
    let (name, _) = after_marker.split_once('}')?;
    (!name.is_empty()).then(|| name.to_owned())
}

fn source_of(table: &toml::Table, path: &[&str]) -> Provenance {
    match table_value(table, path) {
        None => Provenance::Default,
        Some(value) => match env_var_of(value) {
            Some(var) => Provenance::Env(var),
            None => Provenance::Config,
        },
    }
}

fn line(setting: &str, value: String, source: Provenance) -> SettingProvenance {
    SettingProvenance {
        setting: setting.to_owned(),
        value,
        source,
    }
}

fn humantime(duration: std::time::Duration) -> String {
    format!("{duration:?}")
}

/// Builds the provenance report: every operator-facing server setting, the
/// auth mode, and the storage ledger presence, each with its source.
///
/// `raw_toml` is the file text *before* interpolation (so `${VAR}` references
/// are still visible); `config` is the parsed, interpolated result (so the
/// values shown are the resolved ones).
#[must_use]
pub fn provenance_report(
    raw_toml: &str,
    config: &AppConfig,
    env: &RuntimeEnvOverrides,
) -> Vec<SettingProvenance> {
    let table = raw_toml.parse::<toml::Table>().unwrap_or_default();
    let mut rows = server_rows(&table, &config.server, env);
    rows.extend(metrics_rows(&table, &config.server));
    rows.extend(feature_rows(&table, &config.server));
    rows.extend(auth_and_storage_rows(&table, config));
    if env.rust_log_format.is_some() {
        rows.push(line(
            "note",
            "RUST_LOG_FORMAT is set and overrides server.log_format".to_owned(),
            Provenance::Env("RUST_LOG_FORMAT".to_owned()),
        ));
    }
    rows
}

/// The `[server]` settings.
fn server_rows(
    table: &toml::Table,
    server: &llm_proxy_core::ServerConfig,
    env: &RuntimeEnvOverrides,
) -> Vec<SettingProvenance> {
    vec![
        line(
            "server.bind",
            server.bind.to_string(),
            source_of(table, &["server", "bind"]),
        ),
        line(
            "server.request_timeout",
            humantime(server.request_timeout),
            source_of(table, &["server", "request_timeout"]),
        ),
        line(
            "server.shutdown_timeout",
            humantime(server.shutdown_timeout),
            source_of(table, &["server", "shutdown_timeout"]),
        ),
        line(
            "server.log_level",
            server.log_level.clone(),
            source_of(table, &["server", "log_level"]),
        ),
        line(
            "server.log_format",
            match server.log_format {
                llm_proxy_core::LogFormat::Plain => "plain",
                llm_proxy_core::LogFormat::Json => "json",
            }
            .to_owned(),
            // The env override wins over whatever the file says.
            if env.rust_log_format.is_some() {
                Provenance::Env("RUST_LOG_FORMAT".to_owned())
            } else {
                source_of(table, &["server", "log_format"])
            },
        ),
        line(
            "server.hot_reload",
            server.hot_reload.to_string(),
            source_of(table, &["server", "hot_reload"]),
        ),
        line(
            "server.rate_limit_rpm",
            server.rate_limit_rpm.to_string(),
            source_of(table, &["server", "rate_limit_rpm"]),
        ),
        line(
            "server.trust_forwarded_headers",
            server.trust_forwarded_headers.to_string(),
            source_of(table, &["server", "trust_forwarded_headers"]),
        ),
        line(
            "server.server_name",
            server.server_name.clone(),
            source_of(table, &["server", "server_name"]),
        ),
        line(
            "server.allowed_origins",
            server
                .allowed_origins
                .as_ref()
                .map_or_else(|| "none".to_owned(), |origins| origins.join(", ")),
            source_of(table, &["server", "allowed_origins"]),
        ),
        line(
            "server.admin_bind",
            server.admin_bind.map_or_else(
                || "(derived loopback default)".to_owned(),
                |bind| bind.to_string(),
            ),
            source_of(table, &["server", "admin_bind"]),
        ),
        line(
            "server.admin_on_main_listener",
            server.admin_on_main_listener.to_string(),
            source_of(table, &["server", "admin_on_main_listener"]),
        ),
        line(
            "server.endpoint_families",
            render_endpoint_families(server),
            source_of(table, &["server", "endpoint_families"]),
        ),
    ]
}

/// The `[server.metrics]` export rows.
fn metrics_rows(
    table: &toml::Table,
    server: &llm_proxy_core::ServerConfig,
) -> Vec<SettingProvenance> {
    vec![
        line(
            "server.metrics.otlp.endpoint",
            server.metrics.otlp.as_ref().map_or_else(
                || "disabled".to_owned(),
                |otlp| llm_proxy_core::provider_config::endpoint_for_display(&otlp.endpoint),
            ),
            source_of(table, &["server", "metrics", "otlp", "endpoint"]),
        ),
        line(
            "server.metrics.otlp.interval",
            server
                .metrics
                .otlp
                .as_ref()
                .map_or_else(|| "disabled".to_owned(), |otlp| humantime(otlp.interval)),
            source_of(table, &["server", "metrics", "otlp", "interval"]),
        ),
    ]
}

/// `[server.endpoint_families]` rendered one family per entry, sorted by
/// family name so the row is stable across runs.
fn render_endpoint_families(server: &llm_proxy_core::ServerConfig) -> String {
    if server.endpoint_families.is_empty() {
        return "none".to_owned();
    }
    let mut families: Vec<(&String, &llm_proxy_core::EndpointFamilyConfig)> =
        server.endpoint_families.iter().collect();
    families.sort_by(|a, b| a.0.cmp(b.0));
    families
        .into_iter()
        .map(|(name, family)| {
            format!(
                "{name}: max_body_bytes={}, request_timeout={}",
                family.max_body_bytes,
                humantime(family.request_timeout)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Timeout-phase and feature-flag rows.
fn feature_rows(
    table: &toml::Table,
    server: &llm_proxy_core::ServerConfig,
) -> Vec<SettingProvenance> {
    vec![
        line(
            "server.upstream_timeouts.connect",
            humantime(server.upstream_timeouts.connect),
            source_of(table, &["server", "upstream_timeouts", "connect"]),
        ),
        line(
            "server.upstream_timeouts.first_byte",
            humantime(server.upstream_timeouts.first_byte),
            source_of(table, &["server", "upstream_timeouts", "first_byte"]),
        ),
        line(
            "server.upstream_timeouts.idle",
            humantime(server.upstream_timeouts.idle),
            source_of(table, &["server", "upstream_timeouts", "idle"]),
        ),
        line(
            "server.upstream_timeouts.total",
            server
                .upstream_timeouts
                .total
                .map_or_else(|| "none".to_owned(), humantime),
            source_of(table, &["server", "upstream_timeouts", "total"]),
        ),
        line(
            "server.bare_v1.enabled",
            server.bare_v1.enabled.to_string(),
            source_of(table, &["server", "bare_v1", "enabled"]),
        ),
        line(
            "server.preflight.mode",
            match server.preflight.mode {
                llm_proxy_core::PreflightMode::Off => "off",
                llm_proxy_core::PreflightMode::Warn => "warn",
                llm_proxy_core::PreflightMode::Enforce => "enforce",
            }
            .to_owned(),
            source_of(table, &["server", "preflight", "mode"]),
        ),
        line(
            "server.preflight.margin_tokens",
            server.preflight.margin_tokens.to_string(),
            source_of(table, &["server", "preflight", "margin_tokens"]),
        ),
        line(
            "server.cache.enabled",
            server.cache.enabled.to_string(),
            source_of(table, &["server", "cache", "enabled"]),
        ),
        line(
            "server.cache.ttl",
            humantime(server.cache.ttl),
            source_of(table, &["server", "cache", "ttl"]),
        ),
        line(
            "server.cache.max_entries",
            server.cache.max_entries.to_string(),
            source_of(table, &["server", "cache", "max_entries"]),
        ),
        line(
            "server.idempotency.enabled",
            server.idempotency.enabled.to_string(),
            source_of(table, &["server", "idempotency", "enabled"]),
        ),
        line(
            "server.idempotency.ttl",
            humantime(server.idempotency.ttl),
            source_of(table, &["server", "idempotency", "ttl"]),
        ),
        line(
            "server.idempotency.max_entries",
            server.idempotency.max_entries.to_string(),
            source_of(table, &["server", "idempotency", "max_entries"]),
        ),
        line(
            "server.bare_v1.models",
            render_bare_v1_models(server),
            source_of(table, &["server", "bare_v1", "models"]),
        ),
        line(
            "server.bare_v1.default_target",
            server
                .bare_v1
                .default_target
                .clone()
                .unwrap_or_else(|| "none".to_owned()),
            source_of(table, &["server", "bare_v1", "default_target"]),
        ),
    ]
}

/// `[server.bare_v1].models` rendered as sorted `model -> target` pairs.
fn render_bare_v1_models(server: &llm_proxy_core::ServerConfig) -> String {
    if server.bare_v1.models.is_empty() {
        return "none".to_owned();
    }
    let mut mappings: Vec<(&String, &String)> = server.bare_v1.models.iter().collect();
    mappings.sort_by(|a, b| a.0.cmp(b.0));
    mappings
        .into_iter()
        .map(|(model, target)| format!("{model} -> {target}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The `[auth]` mode/key count, the admin surface's presence, and the
/// `[storage]` ledger settings. Values only — no secret is ever rendered
/// (verifiers, tokens, keys).
fn auth_and_storage_rows(table: &toml::Table, config: &AppConfig) -> Vec<SettingProvenance> {
    vec![
        line(
            "auth.mode",
            match config.auth.mode {
                llm_proxy_core::AuthMode::None => "none",
                llm_proxy_core::AuthMode::ClientKeys => "client-keys",
            }
            .to_owned(),
            source_of(table, &["auth", "mode"]),
        ),
        line(
            "auth.client_keys",
            config.auth.clients.len().to_string(),
            source_of(table, &["auth", "clients"]),
        ),
        line(
            "auth.admin",
            if config.auth.admin.is_some() {
                "configured"
            } else {
                "absent"
            }
            .to_owned(),
            source_of(table, &["auth", "admin"]),
        ),
        line(
            "storage.path",
            config.storage.as_ref().map_or_else(
                || "(no ledger: nothing persisted)".to_owned(),
                |storage| storage.path.display().to_string(),
            ),
            source_of(table, &["storage", "path"]),
        ),
        line(
            "storage.max_age",
            config
                .storage
                .as_ref()
                .and_then(|storage| storage.max_age)
                .map_or_else(|| "forever".to_owned(), humantime),
            source_of(table, &["storage", "max_age"]),
        ),
        line(
            "storage.max_rows",
            config
                .storage
                .as_ref()
                .and_then(|storage| storage.max_rows)
                .map_or_else(|| "unlimited".to_owned(), |rows| rows.to_string()),
            source_of(table, &["storage", "max_rows"]),
        ),
        line(
            "storage.prune_interval",
            config.storage.as_ref().map_or_else(
                || "(no ledger)".to_owned(),
                |storage| humantime(storage.prune_interval),
            ),
            source_of(table, &["storage", "prune_interval"]),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_from(raw: &str) -> AppConfig {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        std::fs::write(file.path(), raw).expect("write config");
        llm_proxy_core::load_app_config(file.path()).expect("config parses and validates")
    }

    fn find<'a>(rows: &'a [SettingProvenance], setting: &str) -> &'a SettingProvenance {
        rows.iter()
            .find(|row| row.setting == setting)
            .unwrap_or_else(|| panic!("{setting} missing from report"))
    }

    #[test]
    fn unset_settings_report_the_default_layer() {
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"
"#;
        let config = config_from(raw);
        let rows = provenance_report(raw, &config, &RuntimeEnvOverrides::default());
        let rate = find(&rows, "server.rate_limit_rpm");
        assert_eq!(rate.source, Provenance::Default);
        assert_eq!(rate.value, "100", "the resolved default value is shown");
        let shutdown = find(&rows, "server.shutdown_timeout");
        assert_eq!(shutdown.source, Provenance::Default);
    }

    #[test]
    fn file_settings_report_the_config_layer_with_resolved_values() {
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"
rate_limit_rpm = 42
"#;
        let config = config_from(raw);
        let rows = provenance_report(raw, &config, &RuntimeEnvOverrides::default());
        let rate = find(&rows, "server.rate_limit_rpm");
        assert_eq!(rate.source, Provenance::Config);
        assert_eq!(rate.value, "42");
        let bind = find(&rows, "server.bind");
        assert_eq!(bind.source, Provenance::Config);
        assert_eq!(bind.value, "127.0.0.1:3456");
    }

    #[test]
    fn interpolated_settings_report_the_env_layer_naming_the_variable() {
        // The ${VAR} reference is visible in the raw file; the parsed config
        // carries the resolved value. Both are needed for the attribution, so
        // the "interpolation" here resolves the variable by hand instead of
        // touching the process environment.
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "${PROV_TEST_NAME}"
"#;
        let config = config_from(&raw.replace("${PROV_TEST_NAME}", "resolved-name"));
        let rows = provenance_report(raw, &config, &RuntimeEnvOverrides::default());
        let name = find(&rows, "server.server_name");
        assert_eq!(
            name.source,
            Provenance::Env("PROV_TEST_NAME".to_owned()),
            "an interpolated value came from the environment"
        );
        assert_eq!(name.value, "resolved-name");
    }

    #[test]
    fn a_set_rust_log_format_outranks_the_file_for_log_format() {
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
log_format = "json"
hot_reload = false
server_name = "t"
"#;
        let config = config_from(raw);
        let env = RuntimeEnvOverrides {
            rust_log_format: Some("plain".to_owned()),
        };
        let rows = provenance_report(raw, &config, &env);
        let format = find(&rows, "server.log_format");
        assert_eq!(
            format.source,
            Provenance::Env("RUST_LOG_FORMAT".to_owned()),
            "the documented env override wins over the file"
        );
    }

    #[test]
    fn no_secret_ever_appears_in_the_report() {
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"

[auth]
mode = "client-keys"
[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$c2FsdA$aGFzaA"
[auth.admin]
token = "admin-token-value"

[storage]
path = "usage.db"
"#;
        let config = config_from(raw);
        let rows = provenance_report(raw, &config, &RuntimeEnvOverrides::default());
        let rendered = rows
            .iter()
            .map(|row| format!("{} = {} [{}]", row.setting, row.value, row.source))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !rendered.contains("argon2id") && !rendered.contains("aGFzaA"),
            "verifiers must never be printed: {rendered}"
        );
        assert!(
            !rendered.contains("admin-token-value"),
            "the admin token must never be printed: {rendered}"
        );
        let keys = find(&rows, "auth.client_keys");
        assert_eq!(keys.value, "1", "the key count is the reportable fact");
        let admin = find(&rows, "auth.admin");
        assert_eq!(admin.value, "configured", "presence is the reportable fact");
    }

    /// Audit finding 3: the report used to print a curated subset. Every
    /// documented operator-facing setting must appear with its resolved value
    /// and its layer.
    #[test]
    fn every_documented_setting_has_a_row_with_its_resolved_value() {
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"

[server.preflight]
mode = "enforce"
margin_tokens = 128

[server.cache]
enabled = true
ttl = "5m"
max_entries = 250

[server.idempotency]
enabled = true
ttl = "12h"
max_entries = 50

[server.metrics.otlp]
endpoint = "http://127.0.0.1:4318/v1/metrics"
interval = "30s"

[server.bare_v1]
enabled = true
[server.bare_v1.models]
"claude-sonnet-4-6" = "anthropic"
"gpt-4o" = "openai"

[server.endpoint_families.messages]
max_body_bytes = 2048
request_timeout = "60s"

[storage]
path = "usage.db"
max_age = "30d"
max_rows = 1000
prune_interval = "2h"
"#;
        let config = config_from(raw);
        let rows = provenance_report(raw, &config, &RuntimeEnvOverrides::default());

        let cases = [
            ("server.preflight.margin_tokens", "128"),
            ("server.cache.ttl", "300s"),
            ("server.cache.max_entries", "250"),
            ("server.idempotency.ttl", "43200s"),
            ("server.idempotency.max_entries", "50"),
            (
                "server.metrics.otlp.endpoint",
                "http://127.0.0.1:4318/v1/metrics",
            ),
            ("server.metrics.otlp.interval", "30s"),
            (
                "server.bare_v1.models",
                "claude-sonnet-4-6 -> anthropic, gpt-4o -> openai",
            ),
            (
                "server.endpoint_families",
                "messages: max_body_bytes=2048, request_timeout=60s",
            ),
            ("storage.max_age", "2592000s"),
            ("storage.max_rows", "1000"),
            ("storage.prune_interval", "7200s"),
        ];
        for (setting, value) in cases {
            let row = find(&rows, setting);
            assert_eq!(
                row.value, value,
                "{setting} carries its resolved value [{:?}]",
                row.source
            );
            assert_eq!(
                row.source,
                Provenance::Config,
                "{setting} is set in the file"
            );
        }
    }

    /// The same rows exist when nothing is set, attributing the documented
    /// defaults instead of silently disappearing.
    #[test]
    fn unset_new_settings_still_have_rows_attributed_to_defaults() {
        let raw = r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"
"#;
        let config = config_from(raw);
        let rows = provenance_report(raw, &config, &RuntimeEnvOverrides::default());
        for setting in [
            "server.preflight.margin_tokens",
            "server.cache.ttl",
            "server.cache.max_entries",
            "server.idempotency.ttl",
            "server.idempotency.max_entries",
            "server.metrics.otlp.endpoint",
            "server.metrics.otlp.interval",
            "server.bare_v1.models",
            "server.bare_v1.default_target",
            "server.endpoint_families",
            "auth.admin",
            "storage.max_age",
            "storage.max_rows",
            "storage.prune_interval",
        ] {
            let row = find(&rows, setting);
            assert_eq!(
                row.source,
                Provenance::Default,
                "{setting} must be attributed to the default layer, got {:?}",
                row.source
            );
        }
        assert_eq!(find(&rows, "auth.admin").value, "absent");
        assert_eq!(
            find(&rows, "server.metrics.otlp.endpoint").value,
            "disabled"
        );
        assert_eq!(find(&rows, "storage.max_age").value, "forever");
        assert_eq!(find(&rows, "storage.max_rows").value, "unlimited");
    }
}
