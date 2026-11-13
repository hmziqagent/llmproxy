//! Host and Origin validation under anonymous auth (backlog 11, CWE-346).
//!
//! `mode = "none"` is only permitted on a loopback bind, which stops remote
//! callers and nothing else. A web page reaches a loopback port two ways: by
//! rebinding a hostname it controls to a loopback address, or by a
//! cross-origin request that never meets a CORS preflight (a "simple" POST
//! with a `text/plain` body is enough to make the proxy spend upstream
//! tokens).
//!
//! Three signals, one gate:
//!
//! - **`Host` must name this listener** (a loopback spelling or `localhost`),
//!   unconditionally. This is what refuses a rebound name: after rebind, the
//!   page's own origin *is* the attacker's hostname, so `Sec-Fetch-Site`
//!   reads `same-origin` and nothing but the `Host` betrays it.
//! - **`Origin` must name this proxy or an operator-listed origin** — but
//!   only on requests that carry Fetch Metadata (`Sec-Fetch-*`). Browsers
//!   attach Fetch Metadata to every scripted request; some non-browser
//!   clients (captured Claude Code traffic among them) attach an `Origin`
//!   header with no browser markers at all, and refusing those would break
//!   real clients. The browser is the threat, so the browser is what the
//!   check discriminates on.
//! - **`Sec-Fetch-Site` must be `same-origin` or `none`** on a request that
//!   carries no `Origin` at all: `same-site` (another loopback port's dev
//!   server) and `cross-site` are another site's script. When an `Origin` is
//!   present it decides alone — a fetch from an operator-listed origin is
//!   legitimately `cross-site`.
//!
//! The gate is installed only under [`llm_proxy_core::AuthMode::None`]: with
//! client keys enforced, a browser page has no credential to present, so
//! rebinding buys it a 401.

use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::state::AppState;
use llm_proxy_core::AuthMode;

/// The port this listener serves on is part of every derived origin, so a
/// page served from a *different* loopback port (a dev server on
/// `localhost:3000`) is still foreign: same-site is not same-origin.
#[derive(Debug, Clone)]
pub(crate) struct HostOriginPolicy {
    port: u16,
    allowed_origins: Vec<String>,
}

impl HostOriginPolicy {
    /// Derive the policy from the app config: the listener port plus the
    /// operator's CORS origin list, normalized once.
    #[must_use]
    pub(crate) fn from_config(config: &llm_proxy_core::AppConfig) -> Self {
        let allowed_origins = config
            .server
            .allowed_origins
            .iter()
            .flatten()
            .filter_map(|origin| normalize_origin(origin))
            .collect();
        Self {
            port: config.server.bind.port(),
            allowed_origins,
        }
    }
}

/// Normalizes an origin string to `scheme://host:port` with the scheme's
/// default port made explicit and IPv6 hosts bracketed. `None` when the value
/// is not an absolute `http(s)://host[:port]` origin.
fn normalize_origin(raw: &str) -> Option<String> {
    let lowered = raw.trim().trim_end_matches('/').to_ascii_lowercase();
    let (scheme, rest) = lowered.split_once("://")?;
    if !matches!(scheme, "http" | "https") {
        return None;
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    if authority.is_empty() {
        return None;
    }
    let (host, port) = split_host_port(authority)?;
    let default_port = if scheme == "https" { 443u16 } else { 80u16 };
    let port = port.unwrap_or(default_port);
    let bracketed = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_owned()
    };
    Some(format!("{scheme}://{bracketed}:{port}"))
}

/// Splits `host[:port]` with IPv6 bracket handling. A bare bracketed IPv6
/// literal has no port (an unbracketed port would be ambiguous).
fn split_host_port(authority: &str) -> Option<(&str, Option<u16>)> {
    if let Some(inner) = authority.strip_prefix('[') {
        let (host, rest) = inner.split_once(']')?;
        let port = rest
            .strip_prefix(':')
            .and_then(|port| port.parse::<u16>().ok());
        return Some((host, port));
    }
    match authority.rsplit_once(':') {
        Some((host, port)) => Some((host, port.parse::<u16>().ok())),
        None => Some((authority, None)),
    }
}

/// A `Host` value names this proxy when its bare host is a loopback address
/// or `localhost`. DNS rebinding needs a hostname to rebind, so admitting
/// loopback spellings costs nothing the gate was built to stop; a client
/// reaching the proxy through an SSH tunnel should use its address, not a
/// tunnel hostname.
fn host_is_allowed(host_header: &str) -> bool {
    let bare = host_header.trim();
    let Some((host, _port)) = split_host_port(bare) else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// Whether a browser-supplied `Origin` names this proxy or an operator-listed
/// origin.
fn origin_is_allowed(origin: &str, policy: &HostOriginPolicy) -> bool {
    let Some(candidate) = normalize_origin(origin) else {
        return false;
    };
    if policy.allowed_origins.contains(&candidate) {
        return true;
    }
    for host in ["127.0.0.1", "localhost", "[::1]", "::1"] {
        if candidate == format!("http://{host}:{}", policy.port) {
            return true;
        }
    }
    false
}

/// `Sec-Fetch-Site` values a browser may send here: `same-origin` is this
/// proxy's own page, `none` is a user-initiated navigation. Everything else
/// (`same-site`, `cross-site`) is another site's script.
fn sec_fetch_site_is_allowed(site: &str) -> bool {
    matches!(site, "same-origin" | "none")
}

/// Whether the request carries Fetch Metadata, which every scripted browser
/// request does and no captured non-browser client does. `Origin` is judged
/// only when this is true: a CLI may send an `Origin` header of its own with
/// no browser markers, and refusing it would break real clients.
fn request_carries_fetch_metadata(headers: &axum::http::HeaderMap) -> bool {
    ["sec-fetch-site", "sec-fetch-mode", "sec-fetch-dest"]
        .iter()
        .any(|name| headers.contains_key(*name))
}

/// The middleware. Installs nothing under client-keys auth; see the module
/// docs. A request whose only foreign marker is an `Origin` header with no
/// Fetch Metadata is admitted on purpose (captured CLI traffic sends one);
/// `a_non_browser_request_with_an_origin_header_passes` pins that admission.
pub(crate) async fn enforce_host_origin(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let config = state.app_config();
    if config.auth.mode != AuthMode::None {
        return next.run(req).await;
    }
    let policy = HostOriginPolicy::from_config(config);

    let refusal = |which: &str, value: &str, fix: &str| {
        tracing::warn!(
            header = which,
            %value,
            "refusing request from a foreign source under anonymous auth (DNS-rebinding defense)"
        );
        (
            StatusCode::FORBIDDEN,
            axum::Json(json!({
                "error": {
                    "type": "forbidden_request_source",
                    "message": format!(
                        "refused: the {which} header does not name this proxy ({value}). {fix}"
                    ),
                }
            })),
        )
            .into_response()
    };

    if let Some(host) = req.headers().get(header::HOST)
        && let Ok(value) = host.to_str()
        && !host_is_allowed(value)
    {
        return refusal(
            "Host",
            value,
            "Anonymous auth serves loopback callers only; use a loopback address or \
             \"localhost\" as the host, not a rebound or tunneled hostname.",
        );
    }
    // A browser request: an allowed Origin is admitted whatever Fetch
    // Metadata says (a fetch from an operator-listed origin is legitimately
    // `cross-site`), a foreign one is refused. With no Origin at all, Fetch
    // Metadata must describe this proxy's own page or a user navigation. A
    // request with no Fetch Metadata at all may still carry an Origin (some
    // CLI clients do); the Host check above already pinned it to this
    // listener's address.
    if request_carries_fetch_metadata(req.headers()) {
        let origin = req
            .headers()
            .get(header::ORIGIN)
            .and_then(|value| value.to_str().ok());
        let origin_allowed = origin.is_some_and(|value| origin_is_allowed(value, &policy));
        if let Some(value) = origin
            && !origin_allowed
        {
            return refusal(
                "Origin",
                value,
                "Cross-origin browser requests are not permitted; list the origin in \
                 server.allowed_origins if it is yours.",
            );
        }
        if origin.is_none()
            && let Some(site) = req.headers().get("sec-fetch-site")
            && let Ok(value) = site.to_str()
            && !sec_fetch_site_is_allowed(value)
        {
            return refusal(
                "Sec-Fetch-Site",
                value,
                "Requests scripted by another site are not permitted; call the proxy from \
                 a non-browser client or its own origin.",
            );
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(port: u16, allowed: &[&str]) -> HostOriginPolicy {
        HostOriginPolicy {
            port,
            allowed_origins: allowed
                .iter()
                .filter_map(|origin| normalize_origin(origin))
                .collect(),
        }
    }

    #[test]
    fn loopback_hosts_are_allowed_including_all_spellings() {
        for host in [
            "127.0.0.1:8787",
            "127.0.0.5:8787",
            "localhost:8787",
            "LOCALHOST:8787",
            "[::1]:8787",
            "127.0.0.1",
        ] {
            assert!(host_is_allowed(host), "Host {host:?} must be allowed");
        }
    }

    #[test]
    fn rebound_and_foreign_hosts_are_refused() {
        for host in [
            "rebind.attacker.example:8787",
            "attacker.example",
            "10.0.0.5:8787",
            "192.168.1.9:8787",
            "[fd00::1]:8787",
            "",
        ] {
            assert!(
                !host_is_allowed(host),
                "Host {host:?} must be refused (a rebound name reaches here as the attacker's hostname)"
            );
        }
    }

    #[test]
    fn origins_naming_this_listener_are_allowed() {
        let policy = policy(8787, &[]);
        for origin in [
            "http://127.0.0.1:8787",
            "http://localhost:8787",
            "http://[::1]:8787",
            "HTTP://LOCALHOST:8787/",
            "http://localhost:8787/some/path",
        ] {
            assert!(
                origin_is_allowed(origin, &policy),
                "Origin {origin:?} names this listener and must be allowed"
            );
        }
    }

    #[test]
    fn foreign_origins_are_refused_even_on_other_loopback_ports() {
        let policy = policy(8787, &[]);
        for origin in [
            "http://localhost:3000",
            "https://app.attacker.example",
            "http://rebind.attacker.example:8787",
            "null",
            "",
        ] {
            assert!(
                !origin_is_allowed(origin, &policy),
                "Origin {origin:?} must be refused: a page on another origin must not drive the proxy"
            );
        }
    }

    #[test]
    fn operator_listed_origins_are_allowed_including_non_loopback() {
        let policy = policy(8787, &["https://app.mycompany.example"]);
        assert!(origin_is_allowed("https://app.mycompany.example", &policy));
        // Default ports normalize, so a browser omitting :443 still matches.
        assert!(origin_is_allowed(
            "https://app.mycompany.example:443",
            &policy
        ));
        assert!(!origin_is_allowed("https://other.example", &policy));
    }

    #[test]
    fn fetch_metadata_allows_only_same_origin_and_user_navigation() {
        assert!(sec_fetch_site_is_allowed("same-origin"));
        assert!(sec_fetch_site_is_allowed("none"));
        assert!(!sec_fetch_site_is_allowed("same-site"));
        assert!(!sec_fetch_site_is_allowed("cross-site"));
    }

    #[test]
    fn origin_normalization_rules() {
        assert_eq!(
            normalize_origin("https://Example.COM"),
            Some("https://example.com:443".to_owned())
        );
        assert_eq!(
            normalize_origin("http://127.0.0.1:80"),
            Some("http://127.0.0.1:80".to_owned())
        );
        assert_eq!(
            normalize_origin("http://user:pass@localhost:9000/x"),
            Some("http://localhost:9000".to_owned())
        );
        assert_eq!(normalize_origin("ftp://localhost:9000"), None);
        assert_eq!(normalize_origin("localhost:9000"), None);
        assert_eq!(normalize_origin("http://"), None);
    }
}
