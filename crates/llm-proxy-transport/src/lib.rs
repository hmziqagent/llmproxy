//! llm-proxy-transport: HTTP egress for the proxy — the pooled upstream
//! client, redirect and timeout policy, forward-proxy (egress) support, and
//! model discovery. It sends [`ProxyRequest`]s produced by the
//! `llm-proxy-adapters` codecs and knows nothing about protocol fields.

#![deny(missing_docs)]

/// Protocol-neutral HTTP client for upstream providers.
pub mod client;
/// Provider model discovery client and parsers.
pub mod discovery;

pub use client::ProxyClient;
pub use discovery::DiscoveryClient;
