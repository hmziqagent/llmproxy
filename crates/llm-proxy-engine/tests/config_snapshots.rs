//! Config-snapshot swap tests (plan 07 step 10, audit conflict 11): the
//! engine reads its configuration through the [`ConfigSource`] per pipeline
//! stage, so swapping the whole immutable snapshot takes effect for later
//! requests without rebuilding anything.

#![cfg(test)]

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::http::{StatusCode, header};
use axum::routing::post;
use llm_proxy_adapters::client_wire::{ClientWireFormat, ClientWireRequest};
use llm_proxy_core::provider_config::StaticModelCatalogEntry;
use llm_proxy_core::{
    AppConfig, AuthStyle, PreflightConfig, PreflightMode, ProviderAdapterConfig,
    ProviderCatalogConfig, ProviderConfig, ProviderRegistry, ProviderRouteKind,
    ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_engine::{
    ClientDialect, ConfigSource, Engine, EngineBuilder, EngineRequest, InboundRequest,
    SwappingConfigSource,
};
use llm_proxy_protocol::core::{CoreRequest, ModelRef, SamplingOptions};
use llm_proxy_storage::{EventBus, NoopBus};

fn config_with_preflight(mode: PreflightMode) -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind addr"),
            request_timeout: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "swap-test".to_owned(),
            rate_limit_rpm: 1000,
            trust_forwarded_headers: false,
            cache: Default::default(),
            idempotency: Default::default(),
            metrics: Default::default(),
            log_format: Default::default(),
            admin_bind: None,
            admin_on_main_listener: false,
            endpoint_families: Default::default(),
            upstream_timeouts: Default::default(),
            bare_v1: Default::default(),
            preflight: PreflightConfig {
                mode,
                margin_tokens: 0,
            },
            semantic_cache: Default::default(),
        },
        auth: Default::default(),
        storage: None,
        replica: None,
        mcp: None,
    }
}

fn provider(endpoint: String) -> ProviderConfig {
    let mut adapters = std::collections::HashMap::new();
    adapters.insert(
        "messages".to_owned(),
        ProviderAdapterConfig {
            protocol: "anthropic_messages".to_owned(),
            endpoint,
            headers: Arc::new(std::collections::HashMap::new()),
            forward_inbound_headers: None,
            lossy: false,
        },
    );
    ProviderConfig {
        name: "fake".to_owned(),
        api_key: secrecy::SecretString::from("upstream-key"),
        auth_style: AuthStyle::XApiKey,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes: ProviderRoutesConfig {
            messages: Some("messages".to_owned()),
            chat_completions: None,
            ..Default::default()
        },
        model_aliases: Default::default(),
        discovery: None,
        // A tiny context window makes the preflight enforce-mode decision
        // observable: a request longer than the window is rejected before
        // egress.
        catalog: Some(ProviderCatalogConfig {
            models: vec![StaticModelCatalogEntry {
                id: "claude-sonnet-4-6".to_owned(),
                display_name: None,
                supports: Vec::new(),
                context_length: Some(4),
            }],
            ..ProviderCatalogConfig::default()
        }),
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn request(text: &str) -> EngineRequest {
    EngineRequest::new(
        "fake",
        ProviderRouteKind::Messages,
        ClientDialect::Anthropic,
        CoreRequest {
            model: ModelRef {
                requested: "claude-sonnet-4-6".to_owned(),
                upstream: None,
            },
            messages: vec![llm_proxy_protocol::core::CoreMessage {
                role: llm_proxy_protocol::core::CoreRole::User,
                content: vec![llm_proxy_protocol::core::CoreContent::Text {
                    text: text.to_owned(),
                    cache: None,
                }],
            }],
            system: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            sampling: SamplingOptions::default(),
            stream: false,
            metadata: Default::default(),
            provider_hints: Default::default(),
        },
        InboundRequest {
            auth: None,
            headers: Default::default(),
            wire: ClientWireRequest::new(
                ClientWireFormat::AnthropicMessages,
                serde_json::json!({}),
            ),
        },
    )
    .with_request_id("req-swap-test")
    .with_body_hash("hash")
}

async fn spawn_upstream() -> String {
    let app = Router::new().route(
        "/{*path}",
        post(|| async {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"claude-sonnet-4-6","stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":4}}"#,
            )
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::trace!(%error, "swap-test upstream stopped");
        }
    });
    format!("http://{addr}/v1/messages")
}

fn engine_over(source: Arc<SwappingConfigSource>, endpoint: String) -> Engine {
    let registry =
        ProviderRegistry::from_providers(vec![provider(endpoint)]).expect("provider registry");
    EngineBuilder::from_parts(
        source as Arc<dyn ConfigSource>,
        Arc::new(registry),
        Arc::new(llm_proxy_adapters::ProviderAdapterRegistry::builtin()),
        Arc::new(llm_proxy_transport::ProxyClient::try_new().expect("client")),
    )
    .event_bus(Arc::new(NoopBus) as Arc<dyn EventBus>)
    .build()
    .expect("engine builds")
}

/// The observable property of a swappable snapshot (plan 07 step 10): the
/// same engine, built once, serves per the snapshot each request reads. With
/// enforce-preflight the oversized request is rejected; after swapping the
/// whole snapshot for an off-preflight one, the identical request passes to
/// the upstream. Nothing about the engine is rebuilt.
#[tokio::test]
async fn swapping_the_snapshot_changes_what_later_requests_see() {
    let endpoint = spawn_upstream().await;
    let source = Arc::new(SwappingConfigSource::new(Arc::new(config_with_preflight(
        PreflightMode::Enforce,
    ))));
    let engine = engine_over(Arc::clone(&source), endpoint.clone());

    let rejected = engine
        .execute(request("this message is much longer than four tokens"))
        .await
        .expect_err("enforce-mode preflight rejects the oversized request");
    assert!(
        matches!(rejected, llm_proxy_engine::EngineError::InvalidRequest(_)),
        "the refusal is the preflight's client error, got: {rejected:?}"
    );

    // Swap the whole snapshot; the engine itself is untouched.
    let previous = source.swap(Arc::new(config_with_preflight(PreflightMode::Off)));
    assert_eq!(
        previous.server.preflight.mode,
        PreflightMode::Enforce,
        "swap returns the snapshot it replaced"
    );
    let response = engine
        .execute(request("this message is much longer than four tokens"))
        .await
        .expect("the identical request passes under the swapped snapshot");
    assert_eq!(response.core.usage.input_tokens, 10);
}

/// The swap replaces, never merges: the swapped-in snapshot's own values
/// stand for every setting, and nothing is inherited from the previous one.
#[tokio::test]
async fn a_swap_replaces_the_whole_snapshot_rather_than_merging() {
    let source = Arc::new(SwappingConfigSource::new(Arc::new(config_with_preflight(
        PreflightMode::Enforce,
    ))));
    // A swapped-in snapshot whose preflight differs: had the swap merged
    // field-wise, the enforce mode of the first snapshot would survive.
    // The replaced snapshot's fate is irrelevant to the merge question.
    drop(source.swap(Arc::new(config_with_preflight(PreflightMode::Warn))));
    let loaded = ConfigSource::load(&*source).expect("load");
    assert_eq!(
        loaded.server.preflight.mode,
        PreflightMode::Warn,
        "the swapped-in snapshot's own value stands; nothing merged in"
    );
}

/// Holders of an old snapshot keep it: `load` hands out clones of the `Arc`.
#[test]
fn old_snapshot_holders_keep_their_arc_alive_across_a_swap() {
    let first = Arc::new(config_with_preflight(PreflightMode::Enforce));
    let source = SwappingConfigSource::new(Arc::clone(&first));
    let held = ConfigSource::load(&source).expect("initial load");
    // The replaced snapshot is intentionally dropped here: this test only
    // asks what a pre-swap holder still sees.
    drop(source.swap(Arc::new(config_with_preflight(PreflightMode::Off))));
    assert!(
        Arc::ptr_eq(&held, &first),
        "a load taken before the swap still names the old snapshot"
    );
    let after = ConfigSource::load(&source).expect("load after swap");
    assert_eq!(after.server.preflight.mode, PreflightMode::Off);
}
