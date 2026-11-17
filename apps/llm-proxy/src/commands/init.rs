use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::defaults::{
    DEFAULT_CONFIG_TOML, DEFAULT_PROVIDER_OPENCODE_GO, DEFAULT_PROVIDER_OPENCODE_ZEN,
};
use crate::paths::{config_dir, providers_dir_for};
use crate::permissions::{create_private_file, set_private_dir_permissions};

/// Run the `init` command against the default config directory.
pub fn cmd_init() -> Result<()> {
    cmd_init_at(&config_dir())
}

pub(super) fn cmd_init_at(base_dir: &Path) -> Result<()> {
    let config_path: PathBuf = base_dir.join("config.toml");
    if config_path.exists() {
        bail!("config file already exists at {}", config_path.display());
    }

    std::fs::create_dir_all(base_dir)
        .with_context(|| format!("creating directory {}", base_dir.display()))?;
    set_private_dir_permissions(base_dir)?;

    create_private_file(&config_path, DEFAULT_CONFIG_TOML.as_bytes())?;
    println!("created config at {}", config_path.display());

    let providers_dir = providers_dir_for(&config_path);
    std::fs::create_dir_all(&providers_dir)
        .with_context(|| format!("creating directory {}", providers_dir.display()))?;

    let go_path = providers_dir.join("opencode-go.toml");
    create_private_file(&go_path, DEFAULT_PROVIDER_OPENCODE_GO.as_bytes())?;
    println!("created provider config at {}", go_path.display());

    let zen_path = providers_dir.join("opencode-zen.toml");
    create_private_file(&zen_path, DEFAULT_PROVIDER_OPENCODE_ZEN.as_bytes())?;
    println!("created provider config at {}", zen_path.display());

    println!();
    println!("Next steps:");
    println!("  1. Set API keys via environment variables:");
    println!("     export LLM_PROXY_OPENCODE_GO_KEY=your-go-key");
    println!("     export LLM_PROXY_OPENCODE_ZEN_KEY=your-zen-key");
    println!("  2. Run `llm-proxy validate` to check your config");
    println!("  3. Run `llm-proxy serve` to start the proxy");

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::cmd_init_at;
    use llm_proxy_core::ProviderRegistry;

    /// Serialises tests that mutate the process environment.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Sets the `${...}` key placeholders the generated provider files carry,
    /// restoring the previous values (or absence) on drop.
    ///
    /// # Lock discipline (load-bearing invariant)
    ///
    /// [`std::env::set_var`] is unsafe since edition 2024: it is only sound
    /// while no other thread reads or writes the environment. The lock is
    /// acquired by the caller before [`KeyEnvGuard::new`] and still held when
    /// the guard drops (reverse drop order), giving this thread exclusive
    /// access for both operations.
    struct KeyEnvGuard {
        saved: [(String, Option<String>); 2],
    }

    impl KeyEnvGuard {
        fn new(keys: [&str; 2], value: &str) -> Self {
            let saved = keys.map(|key| {
                let prior = std::env::var(key).ok();
                #[expect(
                    unsafe_code,
                    reason = "std::env::set_var is unsafe since edition 2024; ENV_LOCK is held by the caller, see the SAFETY note below"
                )]
                // SAFETY: `set_var` is sound only when no other thread reads
                // or writes the environment concurrently. Every constructor
                // call site holds `ENV_LOCK` for the whole lifetime of the
                // returned guard, so this thread has exclusive access here.
                unsafe {
                    std::env::set_var(key, value)
                };
                (key.to_owned(), prior)
            });
            Self { saved }
        }
    }

    impl Drop for KeyEnvGuard {
        fn drop(&mut self) {
            for (key, prior) in &self.saved {
                #[expect(
                    unsafe_code,
                    reason = "restores the env var via unsafe set_var/remove_var while ENV_LOCK is still held, see the SAFETY notes below"
                )]
                match prior {
                    // SAFETY: the guard is dropped before the `ENV_LOCK`
                    // guard it shares a scope with (reverse drop order), so
                    // `ENV_LOCK` is still held and no other thread can touch
                    // the environment while the saved value is restored.
                    Some(value) => unsafe { std::env::set_var(key, value) },
                    // SAFETY: same invariant as the `Some` arm — `ENV_LOCK`
                    // is still held, giving this thread exclusive access.
                    None => unsafe { std::env::remove_var(key) },
                }
            }
        }
    }

    #[test]
    fn init_writes_server_only_main_config_and_provider_files_with_routes() {
        let dir = tempfile::tempdir().unwrap();
        cmd_init_at(dir.path()).expect("cmd_init_at should succeed on an empty base dir");

        // The main config holds server settings only: provider routing,
        // models and routes live in the per-provider files.
        let main_raw = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        let main: toml::Table =
            toml::from_str(&main_raw).expect("generated config.toml must be valid TOML");
        assert!(
            main.keys()
                .all(|key| matches!(key.as_str(), "server" | "auth" | "storage")),
            "generated config.toml must hold only server/auth/storage settings, found: {:?}",
            main.keys().collect::<Vec<_>>()
        );
        assert!(
            main.contains_key("server"),
            "generated config.toml must have a [server] section"
        );

        // Every provider file routes through adapters it declares itself and
        // offers discovery as a commented-out block.
        let providers_dir = dir.path().join("providers");
        let mut provider_files = 0;
        for entry in std::fs::read_dir(&providers_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
                continue;
            }
            provider_files += 1;
            let raw = std::fs::read_to_string(&path).unwrap();
            let parsed: toml::Table =
                toml::from_str(&raw).expect("generated provider file must be valid TOML");
            let routes = parsed
                .get("provider")
                .and_then(|provider| provider.get("routes"))
                .and_then(|routes| routes.as_table())
                .expect("generated provider file must have [provider.routes]");
            let adapters = parsed
                .get("provider")
                .and_then(|provider| provider.get("adapters"))
                .and_then(|adapters| adapters.as_table())
                .expect("generated provider file must have [provider.adapters]");
            assert!(
                !routes.is_empty(),
                "{} must route at least one endpoint family",
                path.display()
            );
            for (route_kind, adapter_name) in routes {
                let adapter_name = adapter_name
                    .as_str()
                    .unwrap_or_else(|| panic!("route {route_kind} must name an adapter"));
                assert!(
                    adapters.contains_key(adapter_name),
                    "route {route_kind} in {} names adapter {adapter_name:?}, \
                     which [provider.adapters] does not declare",
                    path.display()
                );
            }
            assert!(
                raw.contains("# [provider.discovery]"),
                "{} should offer discovery as a commented-out block",
                path.display()
            );
        }
        assert_eq!(
            provider_files, 2,
            "init writes one file per generated provider"
        );

        // The generated files are valid inputs to the real loader once the
        // `${...}` key placeholders they carry resolve.
        let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _keys = KeyEnvGuard::new(
            ["LLM_PROXY_OPENCODE_GO_KEY", "LLM_PROXY_OPENCODE_ZEN_KEY"],
            "test-key",
        );
        let registry =
            ProviderRegistry::load_from_dir(&providers_dir).expect("generated providers load");
        assert_eq!(registry.len(), 2, "both generated providers load");
        for name in ["opencode-go", "opencode-zen"] {
            assert!(
                registry.iter().any(|provider| provider.name == name),
                "the loader must surface generated provider {name}"
            );
        }
    }

    #[test]
    fn cmd_init_creates_files_with_permissions() {
        let dir = tempfile::tempdir().unwrap();
        cmd_init_at(dir.path()).expect("cmd_init_at should succeed on an empty base dir");

        let config_path = dir.path().join("config.toml");
        let providers_dir = dir.path().join("providers");
        let go_path = providers_dir.join("opencode-go.toml");
        let zen_path = providers_dir.join("opencode-zen.toml");

        assert!(config_path.exists(), "config.toml was created");
        assert!(go_path.exists(), "opencode-go.toml was created");
        assert!(zen_path.exists(), "opencode-zen.toml was created");

        let main_content = std::fs::read_to_string(&config_path).unwrap();
        let parsed: toml::Value =
            toml::from_str(&main_content).expect("generated config.toml must be valid TOML");
        assert!(
            parsed.get("server").is_some(),
            "generated config.toml must have a [server] section"
        );

        let go_content = std::fs::read_to_string(&go_path).unwrap();
        let parsed: toml::Value = toml::from_str(&go_content).unwrap();
        assert!(parsed.get("provider").is_some());
        assert_eq!(parsed["provider"]["name"].as_str(), Some("opencode-go"));

        let zen_content = std::fs::read_to_string(&zen_path).unwrap();
        let parsed: toml::Value = toml::from_str(&zen_content).unwrap();
        assert_eq!(parsed["provider"]["name"].as_str(), Some("opencode-zen"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = config_path.metadata().unwrap().permissions().mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "config.toml should be owner-only (0600)"
            );
            let mode = go_path.metadata().unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
            let mode = zen_path.metadata().unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[test]
    fn cmd_init_rejects_existing_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(&config_path, "existing").unwrap();

        let result = cmd_init_at(dir.path());
        let err = result.expect_err("cmd_init_at must error when config.toml already exists");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already exists"),
            "error should mention the config already exists, got: {msg}"
        );

        assert_eq!(
            std::fs::read_to_string(&config_path).unwrap(),
            "existing",
            "cmd_init_at must not overwrite an existing config"
        );

        let go_path = dir.path().join("providers").join("opencode-go.toml");
        assert!(
            !go_path.exists(),
            "no provider files should be written on bail"
        );
    }

    #[test]
    fn cmd_init_is_idempotent_reject() {
        let dir = tempfile::tempdir().unwrap();
        cmd_init_at(dir.path()).expect("first init should succeed");
        let second = cmd_init_at(dir.path());
        assert!(
            second.is_err(),
            "a second init against a populated dir must bail, not clobber"
        );
    }
}
