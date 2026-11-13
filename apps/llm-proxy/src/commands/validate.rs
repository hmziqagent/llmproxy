use std::path::Path;

use anyhow::{Context, Result};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{AppConfig, load_app_config};

use crate::config_validation::validate_toml_extension;
use crate::paths::{providers_dir_for, resolve_config};
use crate::state::catalog_enforcement_warning;

/// Run the `validate` command.
///
/// Loads main TOML, loads provider files, validates provider adapter protocols
/// against builtins, and prints provider route table.
pub fn cmd_validate(config_path: Option<&Path>) -> Result<()> {
    let path = resolve_config(config_path);

    validate_toml_extension(&path)?;

    println!("validating config: {}", path.display());

    let app_config: AppConfig = load_app_config(&path)
        .with_context(|| format!("loading TOML config from {}", path.display()))?;

    println!();
    println!("=== Server Configuration ===");
    println!("  bind:            {}", app_config.server.bind);
    println!(
        "  request_timeout: {}s",
        app_config.server.request_timeout.as_secs()
    );
    println!("  log_level:       {}", app_config.server.log_level);
    println!("  hot_reload:      {}", app_config.server.hot_reload);
    println!("  rate_limit_rpm:  {}", app_config.server.rate_limit_rpm);
    println!(
        "  trust_forwarded_headers: {}",
        app_config.server.trust_forwarded_headers
    );
    println!("  server_name:     {}", app_config.server.server_name);

    // Config provenance (backlog 34): every resolved setting with the layer
    // it came from, so "why is my flag ignored?" is answerable from output.
    // The raw file text (pre-interpolation) is what attributes the env
    // layer; the parsed config supplies the resolved values.
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("re-reading config from {}", path.display()))?;
    let env = crate::provenance::RuntimeEnvOverrides {
        rust_log_format: std::env::var("RUST_LOG_FORMAT").ok(),
    };
    let report = crate::provenance::provenance_report(&raw, &app_config, &env);
    println!();
    println!(
        "=== Resolved settings ({}) ===",
        crate::provenance::PRECEDENCE
    );
    for row in &report {
        println!("  {} = {}  [{}]", row.setting, row.value, row.source);
    }

    let providers_dir = providers_dir_for(&path);
    // Every gate startup runs (protocols, self-loop, bare /v1 targets, the
    // auth/passthrough combination) runs here too, on the same shared path.
    let adapter_registry = ProviderAdapterRegistry::builtin();
    let registry = crate::state::load_registry_with_checks(&app_config, &path, &adapter_registry)?;

    println!();
    println!("=== Providers ({} loaded) ===", registry.len());
    if registry.is_empty() {
        println!("  (none)");
    } else {
        for provider in registry.iter() {
            println!(
                "  {} ({} adapters, {} routes, {} aliases)",
                provider.name,
                provider.adapters.len(),
                configured_routes(provider).len(),
                provider.model_aliases.len()
            );
        }
    }

    println!();
    println!("=== Provider Routes ===");
    if registry.is_empty() {
        println!("  (no providers loaded)");
    } else {
        for provider in registry.iter() {
            print_provider_routes(provider);
        }
    }

    let cache_dir = providers_dir.join(".catalog");
    let warnings: Vec<_> = registry
        .iter()
        .filter_map(|provider| catalog_enforcement_warning(provider, &cache_dir))
        .collect();
    if !warnings.is_empty() {
        println!();
        println!("=== Warnings ===");
        for warning in warnings {
            println!("  WARNING: {warning}");
        }
    }

    println!();
    println!("Config is valid!");
    Ok(())
}

/// Every route `provider` configures, as (`[provider.routes]` key, adapter
/// name), in declaration order.
fn configured_routes(provider: &llm_proxy_core::ProviderConfig) -> Vec<(&'static str, &str)> {
    llm_proxy_core::ProviderRouteKind::ALL
        .iter()
        .filter_map(|kind| {
            provider
                .routes
                .get(*kind)
                .map(|adapter| (kind.config_key(), adapter))
        })
        .collect()
}

/// Print every route `provider` configures as `route_key -> adapter/protocol`.
///
/// Display only: `validate_provider_config` already refused dangling or
/// empty route adapter names when the file was loaded, so this never runs
/// as the route-existence guard.
fn print_provider_routes(provider: &llm_proxy_core::ProviderConfig) {
    println!("{}:", provider.name);

    let route_entries = configured_routes(provider);

    if route_entries.is_empty() {
        println!("  (no routes configured)");
        return;
    }

    for (route_kind, adapter_name) in route_entries {
        match provider.adapters.get(adapter_name) {
            Some(adapter) => println!("  {route_kind} -> {adapter_name}/{}", adapter.protocol),
            None => println!("  {route_kind} -> {adapter_name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::DEFAULT_CONFIG_TOML;

    #[test]
    fn validate_lists_every_configured_route_kind() {
        // A provider routing a dialect route and two endpoint families:
        // all three are listed (and counted), not only chat and messages.
        let provider: llm_proxy_core::ProviderConfig = toml::from_str::<toml::Table>(
            r#"[provider]
name = "mixed"
api_key = "k"
auth_style = "bearer"
[provider.adapters.resp]
protocol = "openai_responses"
endpoint = "http://127.0.0.1:9/v1/responses"
[provider.adapters.embed]
protocol = "openai_chat_completions"
endpoint = "http://127.0.0.1:9/v1/embeddings"
[provider.routes]
responses = "resp"
embeddings = "embed"
batches = "embed"
"#,
        )
        .ok()
        .and_then(|table| table.get("provider").cloned())
        .and_then(|provider| provider.try_into().ok())
        .expect("provider parses");
        assert_eq!(
            configured_routes(&provider),
            vec![
                ("responses", "resp"),
                ("embeddings", "embed"),
                ("batches", "embed")
            ]
        );
    }

    #[test]
    fn cmd_validate_valid_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();

        let result = cmd_validate(Some(&config));
        result.unwrap();
    }

    #[test]
    fn cmd_validate_invalid_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, "not valid toml {{{{").unwrap();

        let result = cmd_validate(Some(&config));
        assert!(result.is_err());
    }

    #[test]
    fn cmd_validate_accepts_the_shipped_example_config() {
        let example =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config.toml.example");
        let raw = std::fs::read_to_string(&example)
            .unwrap()
            // The example references env vars only inside commented-out
            // lines, but config loading scans the raw text for them.
            .replace("${HOME}", "/tmp")
            .replace("${ADMIN_TOKEN}", "test-admin-token");
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, raw).unwrap();

        cmd_validate(Some(&config))
            .unwrap_or_else(|e| panic!("config.toml.example must validate as shipped: {e:#}"));
    }

    #[test]
    fn cmd_validate_rejects_an_upstream_that_resolves_to_the_own_listener() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/loopy.toml"),
            r#"[provider]
name = "loopy"
api_key = "k"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "http://localhost:3456/v1/chat/completions"

[provider.routes]
chat_completions = "chat"
"#,
        )
        .unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"
"#,
        )
        .unwrap();

        let err = cmd_validate(Some(&config))
            .expect_err("an upstream naming this proxy's own listener is invalid");
        let message = format!("{err:#}");
        assert!(
            message.contains("loopy") && message.contains("3456"),
            "the error names the provider and the listener: {message}"
        );
        assert!(
            message.contains("forever"),
            "the error says what would happen: {message}"
        );
    }

    #[test]
    fn cmd_validate_rejects_cleartext_credentials_to_a_remote_host() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/leaky.toml"),
            r#"[provider]
name = "leaky"
api_key = "k"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "http://api.example.com/v1/chat/completions"

[provider.routes]
chat_completions = "chat"
"#,
        )
        .unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let err = result.expect_err("cleartext credentials to a remote host are invalid");
        let message = format!("{err:#}");
        assert!(
            message.contains("leaky") && message.contains("api.example.com"),
            "the validate error names the provider and the host: {message}"
        );
        assert!(
            message.contains("cleartext"),
            "the validate error says what is wrong: {message}"
        );
    }

    #[test]
    fn cmd_validate_rejects_a_bare_v1_target_naming_no_provider_or_pool() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "t"

[server.bare_v1]
enabled = true
[server.bare_v1.models]
"claude-sonnet-4-6" = "ghost-pool"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let err = result.expect_err("a bare /v1 model mapping to an unknown target is invalid");
        let message = format!("{err:#}");
        assert!(
            message.contains("bare_v1") && message.contains("ghost-pool"),
            "the validate error names the section and the unknown target: {message}"
        );
    }
}
