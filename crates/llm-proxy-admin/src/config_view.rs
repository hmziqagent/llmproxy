//! The sanitized config view: the running [`AppConfig`] as JSON with
//! every secret value redacted. The redaction walks the serialized tree
//! and replaces the values of secret-bearing keys, so a field added
//! upstream is redacted by name rather than by position.

use llm_proxy_core::AppConfig;
use serde_json::Value;

/// Keys whose values are secrets, matched exactly at any depth. Anything
/// named like a credential is redacted even if it is not one — the view
/// is for operators, and a false redaction costs nothing while a leak
/// costs everything.
const SECRET_KEYS: &[&str] = &[
    "api_key",
    "api_keys",
    "token",
    "tokens",
    "verifier",
    "verifiers",
    "secret",
    "url",
];

/// The placeholder a redacted value is replaced with.
const REDACTED: &str = "[REDACTED]";

/// Serializes `config` to JSON with secrets redacted. Connection URLs
/// (`[replica]`, `[mcp]`) can embed credentials, so they are redacted as
/// a whole.
///
/// # Panics
///
/// Never: `AppConfig` serializes to a JSON object by construction.
pub fn sanitize_config(config: &AppConfig) -> Value {
    let mut value = serde_json::to_value(config).unwrap_or(Value::Null);
    redact(&mut value);
    value
}

fn redact(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, entry) in map.iter_mut() {
                if SECRET_KEYS.contains(&key.as_str()) {
                    *entry = Value::String(REDACTED.to_owned());
                } else {
                    redact(entry);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn config_with_secrets() -> AppConfig {
        AppConfig {
            server: llm_proxy_core::ServerConfig {
                bind: "127.0.0.1:8080".parse().expect("addr"),
                request_timeout: std::time::Duration::from_secs(30),
                shutdown_timeout: std::time::Duration::from_secs(5),
                log_level: "info".to_owned(),
                hot_reload: false,
                allowed_origins: None,
                server_name: "sanitize-test".to_owned(),
                rate_limit_rpm: 100,
                trust_forwarded_headers: false,
                log_format: Default::default(),
                admin_bind: None,
                admin_on_main_listener: false,
                endpoint_families: Default::default(),
                upstream_timeouts: Default::default(),
                bare_v1: Default::default(),
                preflight: Default::default(),
                semantic_cache: Default::default(),
                cache: Default::default(),
                idempotency: Default::default(),
                metrics: Default::default(),
            },
            auth: llm_proxy_core::AuthConfig {
                mode: llm_proxy_core::AuthMode::ClientKeys,
                clients: HashMap::from([(
                    "laptop".to_owned(),
                    llm_proxy_core::ClientKeySpec::Verifier(secrecy::SecretString::from(
                        "phc-sentinel-verifier",
                    )),
                )]),
                admin: Some(llm_proxy_core::AdminAuthConfig {
                    token: secrecy::SecretString::from("admin-sentinel-token"),
                    key_store: None,
                }),
                legacy_bare_tokens: false,
            },
            storage: Some(llm_proxy_core::StorageConfig {
                path: "usage.db".into(),
                max_age: None,
                max_rows: None,
                prune_interval: std::time::Duration::from_secs(3600),
            }),
            replica: Some(llm_proxy_core::ReplicaConfig {
                backend: llm_proxy_core::ReplicaBackend::Redis,
                url: secrecy::SecretString::from("redis://:password-sentinel@redis.internal:6379"),
                prefix: "fleet".to_owned(),
            }),
            mcp: Some(llm_proxy_core::McpGatewayConfig {
                bind: "127.0.0.1:3913".parse().expect("addr"),
                tokens: vec![secrecy::SecretString::from("mcp-sentinel-token")],
                upstreams: vec![llm_proxy_core::McpUpstreamConfig {
                    name: "docs".to_owned(),
                    url: "https://mcp.example.com/mcp".to_owned(),
                    token: Some(secrecy::SecretString::from("upstream-sentinel")),
                }],
                session_ttl: std::time::Duration::from_secs(1800),
            }),
        }
    }

    #[test]
    fn no_secret_value_survives_the_view() {
        let json = sanitize_config(&config_with_secrets()).to_string();
        for sentinel in [
            "phc-sentinel-verifier",
            "admin-sentinel-token",
            "password-sentinel",
            "mcp-sentinel-token",
            "upstream-sentinel",
        ] {
            assert!(
                !json.contains(sentinel),
                "the config view must not contain {sentinel}, got: {json}"
            );
        }
        assert!(
            json.contains("[REDACTED]"),
            "redactions are visible as placeholders"
        );
        // Non-secret settings survive.
        assert!(json.contains("sanitize-test"));
        assert!(json.contains("fleet"));
        assert!(json.contains("8080"));
    }
}
