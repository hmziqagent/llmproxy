//! Integration tests for the `validate` command: valid TOML, invalid TOML,
//! wrong extension, and the admin/auth validation rules.

#[cfg(test)]
mod tests {

    use llm_proxy_app::commands::cmd_validate;
    use llm_proxy_app::defaults::DEFAULT_CONFIG_TOML;

    #[test]
    fn validate_accepts_default_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, DEFAULT_CONFIG_TOML).unwrap();

        let result = cmd_validate(Some(&config));
        assert!(
            result.is_ok(),
            "default config should validate: {:?}",
            result
        );
    }

    #[test]
    fn validate_rejects_invalid_toml() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("bad.toml");
        std::fs::write(&config, "not valid toml {{{{").unwrap();

        let result = cmd_validate(Some(&config));
        assert!(result.is_err(), "malformed TOML should fail");
    }

    #[test]
    fn validate_rejects_json_extension() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, "{}").unwrap();

        let result = cmd_validate(Some(&config));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unsupported") || msg.contains("extension"),
            "error should mention unsupported extension: {msg}"
        );
    }

    #[test]
    fn validate_rejects_non_loopback_bind_without_client_keys() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "0.0.0.0:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let msg = format!(
            "{:#}",
            result.expect_err("non-loopback anonymous bind must be rejected")
        );
        assert!(
            msg.contains("not loopback"),
            "error should explain the loopback restriction: {msg}"
        );
    }

    #[test]
    fn validate_accepts_client_keys_on_non_loopback_bind() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "0.0.0.0:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[auth]
mode = "client-keys"

[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$eEnjdfBL8hfvU2QuGOpow8Yq2Jyrr8GXKYgtAAsscSs"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        assert!(
            result.is_ok(),
            "client-keys on a non-loopback bind should validate: {result:?}"
        );
    }

    #[test]
    fn validate_rejects_passthrough_provider_with_client_keys() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/pt.toml"),
            r#"[provider]
name = "pt"
api_key = "unused"
passthrough_auth = true
auth_style = "bearer"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "http://127.0.0.1:9/v1/chat/completions"
[provider.routes]
chat_completions = "chat"
"#,
        )
        .unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "0.0.0.0:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[auth]
mode = "client-keys"

[auth.clients]
laptop = "$argon2id$v=19$m=8,t=1,p=1$MDEyMzQ1Njc4OWFiY2RlZg$eEnjdfBL8hfvU2QuGOpow8Yq2Jyrr8GXKYgtAAsscSs"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let err = result.expect_err("client-keys with a passthrough_auth provider must be refused");
        let message = format!("{err:#}");
        assert!(
            message.contains("pt") && message.contains("client key"),
            "the refusal names the provider and the client key it would leak: {message}"
        );
    }

    #[test]
    fn validate_accepts_passthrough_provider_with_anonymous_auth() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("providers")).unwrap();
        std::fs::write(
            dir.path().join("providers/pt.toml"),
            r#"[provider]
name = "pt"
api_key = "unused"
passthrough_auth = true
auth_style = "bearer"
[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "http://127.0.0.1:9/v1/chat/completions"
[provider.routes]
chat_completions = "chat"
"#,
        )
        .unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        assert!(
            result.is_ok(),
            "passthrough_auth under anonymous auth must still validate: {result:?}"
        );
    }

    #[test]
    fn validate_accepts_admin_api_with_storage() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[auth.admin]
token = "admin-secret"

[storage]
path = "usage.db"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        assert!(
            result.is_ok(),
            "admin + storage should validate: {result:?}"
        );
    }

    #[test]
    fn validate_rejects_admin_token_without_storage() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"

[auth.admin]
token = "admin-secret"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let msg = format!(
            "{:#}",
            result.expect_err("admin without storage must be rejected")
        );
        assert!(
            msg.contains("[auth.admin] requires"),
            "error should point at the missing ledger: {msg}"
        );
    }

    #[test]
    fn validate_rejects_admin_on_main_listener_without_admin_token() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"
admin_on_main_listener = true

[storage]
path = "usage.db"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let msg = format!("{:#}", result.expect_err("dead opt-in must be rejected"));
        assert!(
            msg.contains("admin_on_main_listener"),
            "error should name the dead flag: {msg}"
        );
    }

    #[test]
    fn validate_rejects_admin_bind_equal_to_main_bind() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            r#"
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
server_name = "test"
admin_bind = "127.0.0.1:3456"

[auth.admin]
token = "admin-secret"

[storage]
path = "usage.db"
"#,
        )
        .unwrap();

        let result = cmd_validate(Some(&config));
        let msg = format!("{:#}", result.expect_err("same-socket bind rejected"));
        assert!(
            msg.contains("must differ from server.bind"),
            "error should explain the socket conflict: {msg}"
        );
    }
}
