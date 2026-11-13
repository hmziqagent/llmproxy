//! The sanitized config view: the running [`AppConfig`] as JSON with
//! every secret value redacted. Redaction is shape-based, not a list of
//! exact key names: a key named like a credential is redacted at any
//! depth, and a string value embedding `scheme://user:pass@` is redacted
//! wherever it appears, so a field added upstream is covered without
//! anyone remembering to extend a list.

use llm_proxy_core::AppConfig;
use serde_json::Value;

/// Lowercased substrings that make a key name secret-shaped. Anything
/// named like a credential is redacted even if it is not one — the view
/// is for operators, and a false redaction costs nothing while a leak
/// costs everything.
const SECRET_KEY_MARKERS: &[&str] = &[
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "verifier",
    "credential",
    "authorization",
    "cookie",
    "url",
    "dsn",
    "proxy",
];

/// Exact (lowercased) keys that match a marker by accident and hold no
/// secret, so the view stays useful. Audited against the serialized
/// `AppConfig` fields in `crates/llm-proxy-core/src/provider_config.rs`:
/// `key_store` is the managed-key-store path (`AdminAuthConfig`, line
/// 1067), `legacy_bare_tokens` is the accept-bare-tokens flag
/// (`AuthConfig`, line 717) and `margin_tokens` is the preflight
/// token-count margin (`PreflightConfig`, line 883). Values under these
/// keys still go through the userinfo pass.
const ALLOWED_KEYS: &[&str] = &["key_store", "legacy_bare_tokens", "margin_tokens"];

/// The placeholder a redacted value is replaced with.
const REDACTED: &str = "[REDACTED]";

/// Serializes `config` to JSON with secrets redacted: secret-shaped keys
/// at any depth, and any string value embedding `scheme://user[:pass]@`
/// credentials (connection URLs, headers, notes).
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
                if is_secret_key(key) {
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
        Value::String(text) if has_userinfo_url(text) => {
            *value = Value::String(REDACTED.to_owned());
        }
        _ => {}
    }
}

fn is_secret_key(key: &str) -> bool {
    let lowered = key.to_lowercase();
    !ALLOWED_KEYS.contains(&lowered.as_str())
        && SECRET_KEY_MARKERS
            .iter()
            .any(|marker| lowered.contains(marker))
}

/// Whether `text` embeds a URL with userinfo (`scheme://user[:pass]@`):
/// some `://` whose authority — up to the next `/` — contains an `@`.
/// Occurrences after prose or an earlier clean URL are still found.
fn has_userinfo_url(text: &str) -> bool {
    let mut rest = text;
    while let Some((_, after)) = rest.split_once("://") {
        if let Some(authority) = after.split('/').next() {
            if authority.contains('@') {
                return true;
            }
        }
        rest = after;
    }
    false
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

    #[test]
    fn unlisted_secret_shaped_keys_and_userinfo_urls_are_redacted() {
        let mut value = serde_json::json!({
            "db_password": "hunter2",
            "client_secret": "s",
            "egress": {"proxy": "http://u:p@h:3128"},
            "note": "https://a:b@example.com/x",
            "bind": "127.0.0.1:8080",
            "server_name": "x",
        });
        redact(&mut value);
        let json = value.to_string();

        for leaked in ["hunter2", "u:p", "a:b"] {
            assert!(
                !json.contains(leaked),
                "the config view must not contain {leaked}, got: {json}"
            );
        }
        assert_eq!(value["db_password"], "[REDACTED]", "got: {json}");
        assert_eq!(value["client_secret"], "[REDACTED]", "got: {json}");
        assert_eq!(value["egress"]["proxy"], "[REDACTED]", "got: {json}");
        assert_eq!(value["note"], "[REDACTED]", "got: {json}");
        assert_eq!(value["bind"], "127.0.0.1:8080", "got: {json}");
        assert_eq!(value["server_name"], "x", "got: {json}");
    }

    #[test]
    fn key_matching_ignores_case_and_userinfo_is_found_anywhere_in_a_value() {
        let mut value = serde_json::json!({
            "Authorization": "Bearer header-sentinel",
            "SESSION_COOKIE": "cookie-sentinel",
            "pin": {"Passwd": 4242},
            "endpoints": [
                "https://plain.example.com/v1",
                "first https://plain.example.com/a@b then redis://:array-sentinel@cache:6379/0",
            ],
            "contact": "ops@example.com",
        });
        redact(&mut value);
        let json = value.to_string();

        for leaked in [
            "header-sentinel",
            "cookie-sentinel",
            "4242",
            "array-sentinel",
        ] {
            assert!(
                !json.contains(leaked),
                "the config view must not contain {leaked}, got: {json}"
            );
        }
        // A URL without userinfo and a bare address are not credentials.
        assert_eq!(
            value["endpoints"][0], "https://plain.example.com/v1",
            "got: {json}"
        );
        assert_eq!(value["contact"], "ops@example.com", "got: {json}");
    }

    #[test]
    fn non_secret_settings_named_like_secrets_stay_readable() {
        let mut config = config_with_secrets();
        config.auth.legacy_bare_tokens = true;
        config.auth.admin = Some(llm_proxy_core::AdminAuthConfig {
            token: secrecy::SecretString::from("admin-sentinel-token"),
            key_store: Some("managed-keys.db".into()),
        });
        config.server.preflight.margin_tokens = 321;

        let value = sanitize_config(&config);
        let json = value.to_string();

        assert_eq!(
            value["auth"]["admin"]["key_store"], "managed-keys.db",
            "got: {json}"
        );
        assert_eq!(value["auth"]["legacy_bare_tokens"], true, "got: {json}");
        assert_eq!(
            value["server"]["preflight"]["margin_tokens"], 321,
            "got: {json}"
        );
        assert!(
            !json.contains("admin-sentinel-token"),
            "the admin token next to the key store is still redacted, got: {json}"
        );

        // The exemption is by name only: a credential in such a value is
        // still caught.
        let mut exempt = serde_json::json!({"key_store": "sqlite://u:exempt-sentinel@host/keys"});
        redact(&mut exempt);
        assert!(
            !exempt.to_string().contains("exempt-sentinel"),
            "got: {exempt}"
        );
    }
}
