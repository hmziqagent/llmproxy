//! The optional MCP gateway (plan 09 step 6, audit conflict 17): MCP
//! clients reach configured upstream MCP servers through the proxy, with
//! **its own auth and session model** — a separate token list and
//! listener, never the proxy's client keys — while **sharing the proxy's
//! admission and accounting**: every `tools/call` reserves against the
//! same [`Admission`] step the chat pipeline uses and settles under the
//! same rules, and every call writes the same [`ProxyEvent`] usage rows.
//!
//! Client-facing framing follows the MCP Streamable HTTP transport,
//! revision 2026-07-28: one `POST /mcp` per JSON-RPC message, replies as
//! a single `application/json` object, notifications acknowledged with
//! `202`, no protocol-level sessions minted or echoed, `405` for GET.
//! Upstream calls carry the revision's required headers
//! (`MCP-Protocol-Version`, `Mcp-Method`, `Mcp-Name`).
//!
//! The gateway's *own* session model (which the protocol does not
//! define) is per-token: each presented gateway token identifies a
//! client whose session ages out after the configured TTL, and admission
//! is charged per session caller (`mcp:<token-id>`), so one MCP client
//! cannot outspend the limits the operator set for it.
//!
//! Provenance for wire shapes: the MCP Streamable HTTP transport
//! specification, revision 2026-07-28
//! (`https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http.md`),
//! and the tools schema pages under `.../server/tools`. The upstream
//! integration is exercised against in-repo fake MCP servers, not live
//! ones (none exist in this environment).

#![deny(missing_docs)]

mod accounting;
mod session;
mod upstream;

pub use session::{GatewaySession, SessionTable};
pub use upstream::UpstreamMcp;

use std::sync::{Arc, Mutex};

use axum::Router;
use llm_proxy_core::McpGatewayConfig;
use llm_proxy_engine::Admission;
use llm_proxy_storage::EventBus;
use llm_proxy_transport::ProxyClient;

/// The gateway: config plus the shared admission step, usage sink and
/// transport, and its own session table.
pub struct McpGateway {
    pub(crate) config: McpGatewayConfig,
    pub(crate) admission: Arc<dyn Admission>,
    pub(crate) bus: Arc<dyn EventBus>,
    pub(crate) transport: Arc<ProxyClient>,
    sessions: Mutex<SessionTable>,
    upstreams: Vec<UpstreamMcp>,
}

impl std::fmt::Debug for McpGateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpGateway")
            .field("bind", &self.config.bind)
            .field(
                "upstreams",
                &self
                    .upstreams
                    .iter()
                    .map(|u| u.name.clone())
                    .collect::<Vec<_>>(),
            )
            .field("sessions", &self.sessions)
            .finish_non_exhaustive()
    }
}

impl McpGateway {
    /// A gateway over `config`, sharing `admission` (the same step the
    /// chat pipeline reserves through) and `bus` (the same usage ledger).
    /// Open inside a tokio runtime.
    #[must_use]
    pub fn new(
        config: McpGatewayConfig,
        admission: Arc<dyn Admission>,
        bus: Arc<dyn EventBus>,
        transport: Arc<ProxyClient>,
    ) -> Self {
        let upstreams = config
            .upstreams
            .iter()
            .map(UpstreamMcp::from_config)
            .collect();
        Self {
            config,
            admission,
            bus,
            transport,
            sessions: Mutex::new(SessionTable::default()),
            upstreams,
        }
    }

    /// The configured listener address.
    #[must_use]
    pub fn bind(&self) -> std::net::SocketAddr {
        self.config.bind
    }

    /// The gateway's axum router (`POST /mcp`, `405` otherwise), with its
    /// own bearer-token auth from `[mcp]` applied.
    #[must_use = "the Router must be used with a server"]
    pub fn router(self) -> Router {
        axum::Router::new()
            .route("/mcp", axum::routing::post(upstream::gateway_endpoint))
            .route("/mcp", axum::routing::get(upstream::method_not_allowed))
            .with_state(Arc::new(self))
            .layer(tower_http::trace::TraceLayer::new_for_http())
            .layer(tower_http::catch_panic::CatchPanicLayer::new())
    }

    /// Looks the presented bearer token up in the gateway's own token
    /// list: token *ids* are the list index (`mcp-token-<n>`), which is
    /// what admission charges and usage rows record.
    pub(crate) fn token_id(&self, presented: Option<&str>) -> Option<String> {
        use secrecy::ExposeSecret;
        let presented = presented?;
        self.config
            .tokens
            .iter()
            .position(|token| token.expose_secret() == presented)
            .map(|index| format!("mcp-token-{index}"))
    }

    /// Expires sessions past the TTL and returns the live one for
    /// `token_id`, creating it on first sight.
    pub(crate) fn session_for(&self, token_id: &str) -> GatewaySession {
        let mut table = self
            .sessions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        table.expire(self.config.session_ttl);
        table.touch(token_id)
    }

    /// The upstream a namespaced tool name (`name__tool`) belongs to,
    /// with the upstream tool name.
    pub(crate) fn resolve_tool(&self, name: &str) -> Option<(&UpstreamMcp, String)> {
        let (server, tool) = name.split_once("__")?;
        self.upstreams
            .iter()
            .find(|upstream| upstream.name == server)
            .map(|upstream| (upstream, tool.to_owned()))
    }

    /// Every configured upstream.
    pub(crate) fn upstreams(&self) -> &[UpstreamMcp] {
        &self.upstreams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ids_map_to_stable_ids() {
        use secrecy::SecretString;
        let config = McpGatewayConfig {
            bind: "127.0.0.1:3913".parse().expect("addr"),
            tokens: vec![
                SecretString::from("alpha-token"),
                SecretString::from("beta-token"),
            ],
            upstreams: Vec::new(),
            session_ttl: std::time::Duration::from_secs(60),
        };
        let gateway = McpGateway::new(
            config,
            Arc::new(llm_proxy_engine::AllowAllAdmission),
            Arc::new(llm_proxy_storage::NoopBus),
            Arc::new(ProxyClient::try_new().expect("client")),
        );
        assert_eq!(
            gateway.token_id(Some("alpha-token")).as_deref(),
            Some("mcp-token-0")
        );
        assert_eq!(
            gateway.token_id(Some("beta-token")).as_deref(),
            Some("mcp-token-1")
        );
        assert_eq!(gateway.token_id(Some("wrong")), None);
        assert_eq!(gateway.token_id(None), None);
    }

    #[test]
    fn an_empty_token_list_refuses_everyone() {
        let config = McpGatewayConfig {
            bind: "127.0.0.1:3913".parse().expect("addr"),
            tokens: Vec::new(),
            upstreams: Vec::new(),
            session_ttl: std::time::Duration::from_secs(60),
        };
        let gateway = McpGateway::new(
            config,
            Arc::new(llm_proxy_engine::AllowAllAdmission),
            Arc::new(llm_proxy_storage::NoopBus),
            Arc::new(ProxyClient::try_new().expect("client")),
        );
        assert_eq!(gateway.token_id(Some("anything")), None);
    }
}
