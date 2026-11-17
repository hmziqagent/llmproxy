//! Integration tests for the `reload` command, driven against a tempdir-backed
//! [`CommandPaths`]: missing PID file, stale PID, and the SIGHUP delivered to
//! the recorded process. Unix only — the signal is SIGHUP.

#![cfg(unix)]

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::path::Path;
    use std::process::Command;

    use llm_proxy_app::commands::cmd_reload;
    use llm_proxy_app::paths::CommandPaths;
    use llm_proxy_app::pid::PidManager;

    fn tempdir_paths(dir: &Path) -> CommandPaths {
        CommandPaths::new(PidManager::new(dir), dir.join("config.toml"))
    }

    #[test]
    fn reload_without_a_pid_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let paths = tempdir_paths(dir.path());

        let err = cmd_reload(&paths).expect_err("no PID file -> Err");
        assert!(
            err.to_string().contains("no PID file"),
            "expected a no-PID-file error, got: {err}"
        );
    }

    #[test]
    fn reload_with_a_stale_pid_errors() {
        let dir = tempfile::tempdir().unwrap();
        let paths = tempdir_paths(dir.path());

        std::fs::write(paths.pid_manager().pid_file(), "299999999").unwrap();

        let err = cmd_reload(&paths).expect_err("stale PID -> Err");
        assert!(
            err.to_string().contains("stale PID file"),
            "expected a stale-PID-file error, got: {err}"
        );
    }

    #[test]
    fn reload_signals_the_recorded_process() {
        let dir = tempfile::tempdir().unwrap();
        let paths = tempdir_paths(dir.path());

        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        std::fs::write(paths.pid_manager().pid_file(), child.id().to_string()).unwrap();

        cmd_reload(&paths).expect("live PID -> Ok");

        let status = child.wait().unwrap();
        assert_eq!(
            status.signal(),
            Some(libc::SIGHUP),
            "child should have been terminated by the delivered SIGHUP, got {status:?}"
        );
    }
}
