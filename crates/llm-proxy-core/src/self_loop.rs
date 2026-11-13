//! Self-loop prevention (backlog 32): these proxies are configured via
//! base-URL swaps, and a URL that points back at the proxy's own listener
//! forwards to itself forever. Two defenses, one module:
//!
//! - **Egress hop stamp**: every upstream request carries
//!   [`HOP_HEADER`]: `1`. A request that arrives at any llm-proxy with that
//!   header already set is inside such a loop and is answered
//!   `508 Loop Detected` (the ingress check lives in the server crate).
//! - **Config-time refusal**: [`self_loop_problems`] resolves every
//!   configured upstream endpoint and reports the ones that land on one of
//!   this instance's own listeners. Startup and `validate` refuse to
//!   continue while any problem is reported.

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

use crate::provider_config::ProviderConfig;

/// Header name stamped on every upstream request and refused on ingress.
pub const HOP_HEADER: &str = "x-llm-proxy-hop";

/// The value written on outbound requests.
pub const HOP_HEADER_VALUE: &str = "1";

/// One upstream endpoint to check, with its owner named for the message.
struct UpstreamEndpoint<'a> {
    provider: &'a str,
    adapter: &'a str,
    url: &'a str,
}

fn endpoints_of(provider: &ProviderConfig) -> Vec<UpstreamEndpoint<'_>> {
    let mut endpoints: Vec<UpstreamEndpoint<'_>> = provider
        .adapters
        .iter()
        .map(|(name, adapter)| UpstreamEndpoint {
            provider: &provider.name,
            adapter: name,
            url: &adapter.endpoint,
        })
        .collect();
    if let Some(discovery) = &provider.discovery {
        endpoints.push(UpstreamEndpoint {
            provider: &provider.name,
            adapter: "discovery",
            url: &discovery.endpoint,
        });
    }
    endpoints
}

/// The `host:port` an endpoint URL names, with the scheme's default port
/// when none is spelled out. `None` when the URL has no usable authority.
fn host_port_of(url: &str) -> Option<(String, u16)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    let default_port = match scheme.to_ascii_lowercase().as_str() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
        let (host, rest) = inner.split_once(']')?;
        let port = rest.strip_prefix(':').and_then(|p| p.parse::<u16>().ok());
        (host.to_owned(), port.unwrap_or(default_port))
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => (host.to_owned(), port.parse::<u16>().ok()?),
            None => (authority.to_owned(), default_port),
        }
    };
    (!host.is_empty()).then_some((host, port))
}

/// Whether `addr` is one of `listeners`: the same address and port, the same
/// port on loopback (the default deployment binds loopback, whatever loopback
/// spelling the endpoint resolved to), or — for a wildcard bind (`0.0.0.0` or
/// `[::]`) — any loopback (or wildcard) address on that port, since a wildcard
/// listener accepts every loopback spelling of either family (`[::]` is
/// dual-stack on the platforms we ship). Residual: an upstream naming one of
/// the host's own non-loopback interface addresses on a wildcard port is not
/// caught here and is still only stopped by the runtime `508` hop-header check.
fn lands_on_listener(addr: SocketAddr, listeners: &[SocketAddr]) -> Option<SocketAddr> {
    listeners
        .iter()
        .copied()
        .find(|listener| addr.port() == listener.port() && addr.ip() == listener.ip())
        .or_else(|| {
            listeners.iter().copied().find(|listener| {
                listener.ip().is_unspecified()
                    && addr.port() == listener.port()
                    && (addr.ip().is_loopback() || addr.ip().is_unspecified())
            })
        })
        .or_else(|| {
            listeners.iter().copied().find(|listener| {
                addr.ip().is_loopback()
                    && listener.ip().is_loopback()
                    && addr.port() == listener.port()
            })
        })
}

/// Reports every configured upstream endpoint that resolves to one of
/// `listeners` (this instance's own main and admin binds), one human-readable
/// message per endpoint. DNS resolution happens at call time: a host that
/// cannot be resolved is skipped (offline validation must still pass), and
/// any address it does resolve to is checked.
///
/// Pools are covered by their member providers.
#[must_use]
pub fn self_loop_problems<'a>(
    providers: impl IntoIterator<Item = &'a ProviderConfig>,
    listeners: &[SocketAddr],
) -> Vec<String> {
    let mut problems = Vec::new();
    if listeners.is_empty() {
        return problems;
    }
    for provider in providers {
        for endpoint in endpoints_of(provider) {
            let Some((host, port)) = host_port_of(endpoint.url) else {
                continue;
            };
            // A literal address is checked as itself; a name is resolved —
            // that is what DNS rebinding of an upstream name would hit.
            let resolved: Vec<SocketAddr> = match host.parse::<IpAddr>() {
                Ok(ip) => vec![SocketAddr::new(ip, port)],
                Err(_) => match (host.as_str(), port).to_socket_addrs() {
                    Ok(addrs) => addrs.collect(),
                    Err(_) => Vec::new(),
                },
            };
            for addr in resolved {
                if let Some(listener) = lands_on_listener(addr, listeners) {
                    problems.push(format!(
                        "provider \"{}\": adapter \"{}\" endpoint \"{}\" resolves to {}, which is \
                         this proxy's own listener — the proxy would forward to itself forever. \
                         Point the endpoint at the upstream provider, not at this proxy",
                        endpoint.provider,
                        endpoint.adapter,
                        crate::provider_config::endpoint_without_query(endpoint.url),
                        listener
                    ));
                    break;
                }
            }
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn provider_with_endpoint(name: &str, endpoint: &str) -> ProviderConfig {
        crate::provider_config::ProviderFile {
            provider: ProviderConfig {
                name: name.to_owned(),
                api_key: secrecy::SecretString::from("k"),
                auth_style: crate::provider_config::AuthStyle::Bearer,
                passthrough_auth: false,
                api_keys: None,
                retry: None,
                egress: None,
                adapters: HashMap::from([(
                    "chat".to_owned(),
                    crate::provider_config::ProviderAdapterConfig {
                        protocol: "openai_chat_completions".to_owned(),
                        endpoint: endpoint.to_owned(),
                        headers: std::sync::Arc::new(HashMap::new()),
                        forward_inbound_headers: None,
                        lossy: false,
                    },
                )]),
                routes: crate::provider_config::ProviderRoutesConfig::default(),
                model_aliases: HashMap::new(),
                discovery: None,
                catalog: None,
                pricing: HashMap::new(),
                concurrency: None,
                pricing_sync: None,
                prompt_cache: None,
            },
        }
        .provider
    }

    fn listeners_of(ports: &[u16]) -> Vec<SocketAddr> {
        ports
            .iter()
            .map(|&port| SocketAddr::new(IpAddr::from([127, 0, 0, 1]), port))
            .collect()
    }

    #[test]
    fn an_upstream_on_the_own_listener_port_is_reported() {
        let providers = [provider_with_endpoint(
            "loopy",
            "http://127.0.0.1:8080/v1/chat/completions",
        )];
        let problems = self_loop_problems(&providers, &listeners_of(&[8080, 8081]));
        assert_eq!(problems.len(), 1, "exactly one problem: {problems:?}");
        assert!(
            problems[0].contains("loopy")
                && problems[0].contains("chat")
                && problems[0].contains("forever"),
            "the message names the provider, adapter and the fix: {}",
            problems[0]
        );
    }

    #[test]
    fn a_hostname_resolving_to_the_own_listener_is_reported() {
        // `localhost` resolves to a loopback address on every test host.
        let providers = [provider_with_endpoint(
            "byname",
            "http://localhost:9333/v1/chat/completions",
        )];
        let problems = self_loop_problems(&providers, &listeners_of(&[9333]));
        assert_eq!(
            problems.len(),
            1,
            "a name that resolves to the listener is still a loop: {problems:?}"
        );
    }

    #[test]
    fn other_ports_and_hosts_are_clean() {
        let providers = [provider_with_endpoint(
            "fine",
            "http://127.0.0.1:9999/v1/chat/completions",
        )];
        assert!(self_loop_problems(&providers, &listeners_of(&[8080, 8081])).is_empty());

        // A remote host, same port: not this machine's listener.
        let providers = [provider_with_endpoint(
            "remote",
            "https://api.example.com:8080/v1/chat/completions",
        )];
        assert!(
            self_loop_problems(&providers, &listeners_of(&[8080])).is_empty(),
            "the check must not report an upstream on another host"
        );
    }

    #[test]
    fn a_wildcard_bind_catches_loopback_upstreams_on_its_port() {
        let wildcard_cases = ["0.0.0.0:8080", "[::]:8080"];
        for bind in wildcard_cases {
            let listeners: Vec<SocketAddr> = vec![bind.parse().unwrap()];
            for (name, endpoint) in [
                ("v4-loopback", "http://127.0.0.1:8080/v1/chat/completions"),
                ("by-name", "http://localhost:8080/v1/chat/completions"),
                ("v6-loopback", "http://[::1]:8080/v1/chat/completions"),
            ] {
                let providers = [provider_with_endpoint("loopy", endpoint)];
                let problems = self_loop_problems(&providers, &listeners);
                assert_eq!(
                    problems.len(),
                    1,
                    "bind {bind}: endpoint {name} at {endpoint} lands on the wildcard listener: \
                     {problems:?}"
                );
            }
            for (name, endpoint) in [
                ("other-port", "http://127.0.0.1:8081/v1/chat/completions"),
                ("other-host", "http://10.1.2.3:8080/v1/chat/completions"),
            ] {
                let providers = [provider_with_endpoint("fine", endpoint)];
                let problems = self_loop_problems(&providers, &listeners);
                assert!(
                    problems.is_empty(),
                    "bind {bind}: endpoint {name} at {endpoint} must not be reported: {problems:?}"
                );
            }
        }
    }

    #[test]
    fn an_unresolvable_hostname_is_skipped_not_fatal() {
        let providers = [provider_with_endpoint(
            "offline",
            "http://no-such-host.invalid:8080/v1/chat/completions",
        )];
        assert!(self_loop_problems(&providers, &listeners_of(&[8080])).is_empty());
    }

    #[test]
    fn default_ports_are_derived_from_the_scheme() {
        assert_eq!(
            host_port_of("https://api.example.com/v1/x"),
            Some(("api.example.com".to_owned(), 443))
        );
        assert_eq!(
            host_port_of("http://api.example.com:9000/v1/x"),
            Some(("api.example.com".to_owned(), 9000))
        );
        assert_eq!(
            host_port_of("http://[::1]:9000/v1/x"),
            Some(("::1".to_owned(), 9000))
        );
        assert_eq!(host_port_of("not-a-url"), None);
    }
}
