use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use llm_proxy_core::{
    CatalogFile, CatalogFileMetadata, ProviderConfig, ProviderRegistry, StaticModelCatalogEntry,
    merge_catalog,
};
use llm_proxy_engine::ModelCatalogService;
use llm_proxy_engine::write_catalog_atomic;
use llm_proxy_transport::DiscoveryClient;

use crate::config_validation::validate_toml_extension;
use crate::paths::{providers_dir_for, resolve_config};

fn print_provider_routes(out: &mut dyn Write, provider: &ProviderConfig) -> std::io::Result<()> {
    let has_routes =
        provider.routes.chat_completions.is_some() || provider.routes.messages.is_some();
    writeln!(out, "    routes:")?;
    if let Some(ref adapter) = provider.routes.chat_completions {
        writeln!(out, "      chat_completions -> {adapter}")?;
    }
    if let Some(ref adapter) = provider.routes.messages {
        writeln!(out, "      messages -> {adapter}")?;
    }
    if !has_routes {
        writeln!(out, "      (none configured)")?;
    }
    Ok(())
}

fn discovered_catalog_file(
    provider: &ProviderConfig,
    discovered: Vec<StaticModelCatalogEntry>,
) -> CatalogFile {
    CatalogFile {
        catalog: CatalogFileMetadata {
            provider: provider.name.clone(),
            source: "live".to_owned(),
            generated_at: time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_owned()),
            models: discovered,
        },
    }
}

fn print_model_entries(
    out: &mut dyn Write,
    entries: &[StaticModelCatalogEntry],
) -> std::io::Result<()> {
    writeln!(out, "    models:")?;
    if entries.is_empty() {
        writeln!(out, "      (none available)")?;
    } else {
        for entry in entries {
            writeln!(out, "      {}", entry.id)?;
        }
    }
    Ok(())
}

/// Run the `models` command.
///
/// Lists provider routes and static, cached, or live model catalogs.
pub async fn cmd_models(
    config_path: Option<PathBuf>,
    provider_filter: Option<String>,
    live: bool,
    write_catalog: bool,
    require_success: bool,
) -> Result<()> {
    let mut out = std::io::stdout();
    run_models(
        &mut out,
        config_path,
        provider_filter,
        live,
        write_catalog,
        require_success,
    )
    .await
}

/// The body of the `models` command, writing its report into `out`.
///
/// Split from [`cmd_models`] so tests can capture the printed report without
/// seizing the process stdout.
///
/// # Errors
///
/// Returns config-loading errors, `--live` misconfiguration, and discovery
/// failures when `require_success` is set.
async fn run_models(
    out: &mut dyn Write,
    config_path: Option<PathBuf>,
    provider_filter: Option<String>,
    live: bool,
    write_catalog: bool,
    require_success: bool,
) -> Result<()> {
    let path = resolve_config(config_path.as_deref());

    validate_toml_extension(&path)?;

    let providers_dir = providers_dir_for(&path);
    let registry = if providers_dir.exists() {
        ProviderRegistry::load_from_dir(&providers_dir)
            .with_context(|| format!("loading providers from {}", providers_dir.display()))?
    } else {
        ProviderRegistry::from_providers(vec![])
            .with_context(|| "building empty provider registry")?
    };

    if registry.is_empty() {
        writeln!(out, "No providers configured.")?;
        writeln!(out, "Add provider files to {}", providers_dir.display())?;
        return Ok(());
    }

    let providers: Vec<_> = registry
        .iter()
        .filter(|p| {
            provider_filter
                .as_ref()
                .is_none_or(|filter| p.name == *filter)
        })
        .collect();

    if provider_filter.is_some() && providers.is_empty() {
        let name = provider_filter.as_deref().unwrap_or("?");
        bail!(
            "provider \"{name}\" not found in {}",
            providers_dir.display()
        );
    }

    let cache_dir = providers_dir.join(".catalog");
    let discovery =
        DiscoveryClient::try_new().context("building the HTTP client for model discovery")?;
    let catalogs = ModelCatalogService::new(Some(cache_dir.clone()), discovery.clone());

    writeln!(out, "Provider models (from {}):", providers_dir.display())?;
    writeln!(out)?;
    for provider in &providers {
        writeln!(out, "  {}:", provider.name)?;

        print_provider_routes(out, provider)?;

        let catalog_config = provider.catalog.clone().unwrap_or_default();
        let entries = if live {
            if provider.discovery.is_none() {
                bail!(
                    "provider \"{}\" has no discovery configuration for --live",
                    provider.name
                );
            }
            match discovery.discover(provider).await {
                Ok(discovered) => {
                    let entries = merge_catalog(&catalog_config, &discovered);
                    if write_catalog {
                        write_catalog_atomic(
                            &cache_dir,
                            &discovered_catalog_file(provider, discovered),
                        )?;
                    }
                    entries
                }
                Err(error) if require_success => return Err(error.into()),
                Err(error) => {
                    eprintln!(
                        "warning: live discovery failed for {}: {error}; using cached/static catalog",
                        provider.name
                    );
                    catalogs.catalog(provider, false).await?
                }
            }
        } else {
            catalogs.catalog(provider, false).await?
        };

        print_model_entries(out, &entries)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use axum::{Router, routing::get};

    use super::*;
    use crate::defaults::DEFAULT_CONFIG_TOML;

    #[tokio::test]
    async fn models_live_writes_discovered_catalog() {
        let app = Router::new().route(
            "/models",
            get(|| async {
                (
                    [("content-type", "application/json")],
                    r#"{"data":[{"id":"discovered-model"}]}"#,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();
        let providers = directory.path().join("providers");
        std::fs::create_dir(&providers).unwrap();
        std::fs::write(
            providers.join("test-provider.toml"),
            format!(
                r#"[provider]
name = "test-provider"
api_key = "test-key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "http://{address}/models"
"#
            ),
        )
        .unwrap();

        cmd_models(
            Some(config),
            Some("test-provider".to_owned()),
            true,
            true,
            true,
        )
        .await
        .unwrap();

        let raw =
            std::fs::read_to_string(providers.join(".catalog").join("test-provider.toml")).unwrap();
        let catalog: CatalogFile = toml::from_str(&raw).unwrap();
        assert_eq!(catalog.catalog.models[0].id, "discovered-model");
    }

    #[tokio::test]
    async fn models_live_fetches_even_in_static_catalog_mode() {
        let app = Router::new().route(
            "/models",
            get(|| async {
                (
                    [("content-type", "application/json")],
                    r#"{"data":[{"id":"discovered-model"}]}"#,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();
        let providers = directory.path().join("providers");
        std::fs::create_dir(&providers).unwrap();
        std::fs::write(
            providers.join("test-provider.toml"),
            format!(
                r#"[provider]
name = "test-provider"
api_key = "test-key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "http://{address}/models"

[provider.catalog]
mode = "static"
"#
            ),
        )
        .unwrap();

        cmd_models(
            Some(config),
            Some("test-provider".to_owned()),
            true,
            true,
            true,
        )
        .await
        .unwrap();

        let raw =
            std::fs::read_to_string(providers.join(".catalog").join("test-provider.toml")).unwrap();
        let catalog: CatalogFile = toml::from_str(&raw).unwrap();
        assert_eq!(catalog.catalog.models[0].id, "discovered-model");
    }

    #[tokio::test]
    async fn models_live_requires_discovery_configuration() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();
        let providers = directory.path().join("providers");
        std::fs::create_dir(&providers).unwrap();
        std::fs::write(
            providers.join("test-provider.toml"),
            r#"[provider]
name = "test-provider"
api_key = "test-key"
auth_style = "bearer"
"#,
        )
        .unwrap();

        let error = cmd_models(
            Some(config),
            Some("test-provider".to_owned()),
            true,
            false,
            false,
        )
        .await
        .expect_err("--live requires discovery configuration");

        assert!(error.to_string().contains("no discovery configuration"));
    }

    /// Serve `/models` with a bare 500 so `discover` fails after its retries.
    async fn start_erroring_discovery_server() -> std::net::SocketAddr {
        let app = Router::new().route(
            "/models",
            get(|| async { (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        address
    }

    fn write_provider_with_discovery_at(
        providers: &std::path::Path,
        name: &str,
        address: std::net::SocketAddr,
    ) {
        std::fs::write(
            providers.join(format!("{name}.toml")),
            format!(
                r#"[provider]
name = "{name}"
api_key = "test-key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "http://{address}/models"
"#
            ),
        )
        .unwrap();
    }

    fn write_catalog_cache(providers: &std::path::Path, name: &str, model_id: &str) -> String {
        let cache = format!(
            r#"[catalog]
provider = "{name}"
source = "live"
generated_at = "2026-01-01T00:00:00Z"

[[catalog.models]]
id = "{model_id}"
"#
        );
        let cache_dir = providers.join(".catalog");
        std::fs::create_dir(&cache_dir).unwrap();
        std::fs::write(cache_dir.join(format!("{name}.toml")), &cache).unwrap();
        cache
    }

    #[tokio::test]
    async fn models_without_live_never_touches_the_network() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();
        let providers = directory.path().join("providers");
        std::fs::create_dir(&providers).unwrap();
        std::fs::write(
            providers.join("no-net.toml"),
            // Discovery points at a port nothing listens on; the hybrid
            // default merges the static entry with the cached discovery.
            r#"[provider]
name = "no-net"
api_key = "test-key"
auth_style = "bearer"

[provider.discovery]
kind = "openai_compatible_models"
endpoint = "http://127.0.0.1:9/v1/models"

[[provider.catalog.models]]
id = "static-model"
"#,
        )
        .unwrap();
        write_catalog_cache(&providers, "no-net", "cached-model");

        let mut report = Vec::new();
        let started = std::time::Instant::now();
        run_models(&mut report, Some(config), None, false, false, false)
            .await
            .expect("listing without --live must succeed from static and cache alone");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "a non-live listing must not wait on the unreachable discovery \
             endpoint, took {:?}",
            started.elapsed()
        );
        let report = String::from_utf8(report).expect("the models report is UTF-8");
        assert!(
            report.contains("static-model"),
            "static entry listed: {report}"
        );
        assert!(
            report.contains("cached-model"),
            "cached entry listed: {report}"
        );
    }

    #[tokio::test]
    async fn models_live_require_success_fails_when_discovery_fails() {
        let address = start_erroring_discovery_server().await;
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();
        let providers = directory.path().join("providers");
        std::fs::create_dir(&providers).unwrap();
        write_provider_with_discovery_at(&providers, "failing", address);
        let cache_before = write_catalog_cache(&providers, "failing", "pre-existing");
        let cache_path = providers.join(".catalog").join("failing.toml");

        let error = cmd_models(Some(config), Some("failing".to_owned()), true, true, true)
            .await
            .expect_err("--require-success must fail the command when discovery fails");
        let message = format!("{error:#}");
        assert!(
            message.contains("500"),
            "the failure names the discovery status: {message}"
        );
        let cache_after =
            std::fs::read(&cache_path).expect("the pre-existing cache file must survive");
        assert_eq!(
            cache_after,
            cache_before.as_bytes(),
            "--write-catalog must not touch the cache when discovery failed"
        );
    }

    #[tokio::test]
    async fn models_live_without_require_success_falls_back_to_cache() {
        let address = start_erroring_discovery_server().await;
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();
        let providers = directory.path().join("providers");
        std::fs::create_dir(&providers).unwrap();
        write_provider_with_discovery_at(&providers, "fallback", address);
        let cache_before = write_catalog_cache(&providers, "fallback", "pre-existing");
        let cache_path = providers.join(".catalog").join("fallback.toml");

        cmd_models(Some(config), Some("fallback".to_owned()), true, true, false)
            .await
            .expect("without --require-success a discovery failure falls back to the cache");

        let cache_after = std::fs::read(&cache_path).unwrap();
        assert_eq!(
            cache_after,
            cache_before.as_bytes(),
            "the failed live refresh must leave the cached catalog untouched"
        );
    }
}
