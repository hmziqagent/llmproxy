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

fn print_provider_routes(provider: &ProviderConfig) {
    let has_routes =
        provider.routes.chat_completions.is_some() || provider.routes.messages.is_some();
    println!("    routes:");
    if let Some(ref adapter) = provider.routes.chat_completions {
        println!("      chat_completions -> {adapter}");
    }
    if let Some(ref adapter) = provider.routes.messages {
        println!("      messages -> {adapter}");
    }
    if !has_routes {
        println!("      (none configured)");
    }
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

fn print_model_entries(entries: &[StaticModelCatalogEntry]) {
    println!("    models:");
    if entries.is_empty() {
        println!("      (none available)");
    } else {
        for entry in entries {
            println!("      {}", entry.id);
        }
    }
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
        println!("No providers configured.");
        println!("Add provider files to {}", providers_dir.display());
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

    println!("Provider models (from {}):", providers_dir.display());
    println!();
    for provider in &providers {
        println!("  {}:", provider.name);

        print_provider_routes(provider);

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

        print_model_entries(&entries);
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
}
