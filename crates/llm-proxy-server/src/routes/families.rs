//! Non-chat endpoint families (plan 09 step 7, audit conflict 16):
//! embeddings, rerank, audio, images, files and batch, each served by a
//! family-specific handler around the same generic spine as chat —
//! caller auth, per-family body limit and timeout, admission
//! reserve/settle, routing through the provider registry, and usage rows
//! on the shared ledger. These are *not* forced through the chat core:
//! requests pass through in the client's dialect with the edits the
//! proxy needs (model aliasing), which is exactly what these endpoints
//! need and what `docs/research/endpoints.md` prescribes per family.
//!
//! Wire shapes follow each family's documented API (OpenAI `/v1/embeddings`,
//! `/v1/audio/*`, `/v1/images/generations`, `/v1/files`, `/v1/batches`;
//! the Cohere-style `/v1/rerank`), verified against real implementations
//! (`tests/live_families.rs`: llama.cpp, speaches, stable-diffusion.cpp
//! and Llama Stack, plus the official OpenAI and Cohere SDKs against the
//! daemon):
//!
//! - The client's method and query reach the upstream: files and batches
//!   are a lifecycle of `GET` lists and downloads, `DELETE`s and `POST`s,
//!   not one upload. A resource path (`/files/{id}/content`) goes between
//!   the configured endpoint and its own query, which survives.
//! - A resource path is plain identifier segments. Anything a URL parser
//!   would resolve (`..`, `%2e%2e`) could leave the family's endpoint on
//!   the operator's credential, so it is refused before any upstream call.
//! - The answer keeps the upstream's status, `Content-Type` and
//!   `Content-Disposition`: speech is audio, a transcript may be plain
//!   text, a download is the file.
//! - The credential and transport are the provider's, as on the chat
//!   routes: `passthrough_auth` sends the client's own key,
//!   `[provider.egress]` proxies the family's egress too.
//! - Failures follow the one client-status policy
//!   ([`llm_proxy_engine::map_upstream_status`]): the upstream's error body
//!   never reaches the client. One stated addition: an upstream `404` for
//!   a request that names a resource (`/files/{id}`, `/batches/{id}`) is
//!   that resource's absence — the client's fixable `404`, not an upstream
//!   fault. A collection endpoint that `404`s is still a misconfigured
//!   upstream (`502`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use llm_proxy_adapters::ProviderError;
use llm_proxy_adapters::request::{AuthHeaders, ProxyRequest};
use llm_proxy_core::ProviderRouteKind;
use llm_proxy_engine::EngineError;
use llm_proxy_protocol::core::{ModelRef, Usage};
use llm_proxy_storage::{EventBus, ProxyEvent, RequestReceived, ResponseCompleted};
use llm_proxy_transport::ProxyClient;
use llm_proxy_transport::client::UpstreamResponse;
use secrecy::SecretString;
use sha2::{Digest as _, Sha256};
use time::OffsetDateTime;

use super::core_pipeline::{
    emit_response_failed, extract_inbound_auth, inbound_headers_for_engine,
};
use super::error_response::{ClientProtocol, RouteError, route_error_response};
use crate::middleware::RequestMeta;
use crate::state::AppState;

/// Per-family wiring carried in request extensions.
#[derive(Clone)]
pub(crate) struct FamilySpec {
    /// Which route kind this family resolves.
    pub(crate) kind: ProviderRouteKind,
    /// Whether request bodies are JSON (model-aliasable) or opaque bytes
    /// (multipart uploads). Speech is JSON: only its answer is audio.
    pub(crate) json: bool,
}

/// Family names, also the `[server.endpoint_families]` keys.
pub(crate) const FAMILY_EMBEDDINGS: &str = "embeddings";
pub(crate) const FAMILY_RERANK: &str = "rerank";
pub(crate) const FAMILY_AUDIO_TRANSCRIPTIONS: &str = "audio_transcriptions";
pub(crate) const FAMILY_AUDIO_SPEECH: &str = "audio_speech";
pub(crate) const FAMILY_IMAGES_GENERATIONS: &str = "images_generations";
pub(crate) const FAMILY_FILES: &str = "files";
pub(crate) const FAMILY_BATCHES: &str = "batches";

/// A family's collection route (`/v1/embeddings`, `/v1/files`, ...).
#[expect(
    clippy::too_many_arguments,
    reason = "axum handlers take one argument per extractor; this route reads the method, URI and headers"
)]
pub(crate) async fn handle_family(
    State(state): State<AppState>,
    Path(provider): Path<String>,
    meta: RequestMeta,
    family: axum::Extension<FamilySpec>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    run_family(
        &state,
        &meta,
        FamilyCall {
            provider: &provider,
            spec: &family.0,
            method,
            query: uri.query(),
            headers: &headers,
            body,
            resource: "",
        },
    )
    .await
}

/// A family's resource route (`/v1/files/{id}[/content]`,
/// `/v1/batches/{id}[/cancel]`): the captured path is appended to the
/// configured family endpoint.
#[expect(
    clippy::too_many_arguments,
    reason = "axum handlers take one argument per extractor; this route reads the method, URI and headers"
)]
pub(crate) async fn handle_family_resource(
    State(state): State<AppState>,
    Path((provider, resource)): Path<(String, String)>,
    meta: RequestMeta,
    family: axum::Extension<FamilySpec>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    run_family(
        &state,
        &meta,
        FamilyCall {
            provider: &provider,
            spec: &family.0,
            method,
            query: uri.query(),
            headers: &headers,
            body,
            resource: &resource,
        },
    )
    .await
}

/// The inbound side of one family call.
struct FamilyCall<'a> {
    provider: &'a str,
    spec: &'a FamilySpec,
    method: Method,
    query: Option<&'a str>,
    headers: &'a HeaderMap,
    body: Bytes,
    /// The resource path after the family endpoint; empty for the
    /// collection itself.
    resource: &'a str,
}

/// Everything one family call carries through the spine.
struct FamilyContext<'a> {
    request_id: String,
    caller: String,
    provider: &'a str,
    family: &'static str,
    kind: ProviderRouteKind,
    requested_model: String,
    body_hash: String,
    start: Instant,
}

/// The prepared egress side of one family call.
struct Outbound<'a> {
    method: Method,
    url: String,
    body: Vec<u8>,
    content_type: Option<String>,
    headers: &'a HeaderMap,
}

/// The generic family spine: resolve, admit, forward, account.
async fn run_family(state: &AppState, meta: &RequestMeta, call: FamilyCall<'_>) -> Response {
    let FamilyCall {
        provider,
        spec,
        method,
        query,
        headers,
        body,
        resource,
    } = call;
    let mut ctx = FamilyContext {
        request_id: meta.request_id.0.clone(),
        caller: meta.caller.as_str().to_owned(),
        provider,
        family: family_name(spec.kind),
        kind: spec.kind,
        requested_model: String::new(),
        body_hash: format!("{:x}", Sha256::digest(&body)),
        start: Instant::now(),
    };

    // Prepare the egress body first, then write the request row before
    // any refusal can occur — the chat lifecycle and the MCP gateway
    // emit theirs before admitting, so a refused family call leaves an
    // attributed pair of rows (caller, provider, route kind) instead of
    // an unattributable failure row alone.
    let registry = state.providers();
    let (requested_model, outbound_body, outbound_content_type) =
        prepare_body(spec, &body, headers, &registry, provider);
    ctx.requested_model = requested_model;
    emit_family_request(state.event_bus(), &ctx);

    if !resource.is_empty() && !is_plain_resource_path(resource) {
        return family_error(
            state.event_bus(),
            &ctx,
            &RouteError::InvalidRequest(
                "a resource path is identifier segments ([A-Za-z0-9._~-], no dot segments)"
                    .to_owned(),
            ),
        );
    }

    // Resolve the route through the same registry the chat pipeline uses.
    let target = match registry.resolve_provider_route(provider, spec.kind, &ctx.requested_model) {
        Ok(target) => target,
        Err(error) => {
            let refusal = RouteError::from(EngineError::from_route_error(error));
            return family_error(state.event_bus(), &ctx, &refusal);
        }
    };

    // The credential, as the chat routes pick it: the provider's key, or
    // on a `passthrough_auth` provider the client's own, which it must
    // bring. (A multi-key pool's extra keys are not rotated here: family
    // calls use the base key.)
    let passthrough = registry
        .get(&target.provider_name)
        .is_some_and(|config| config.passthrough_auth);
    let api_key = if passthrough {
        match extract_inbound_auth(headers) {
            Some(key) => key,
            None => return family_error(state.event_bus(), &ctx, &RouteError::Unauthorized),
        }
    } else {
        target.api_key.clone()
    };

    // Admission: the same reserve/settle step the chat pipeline runs. A
    // refusal never reaches the upstream. The reservation itself carries
    // no token estimate (families have no chat-style estimator); the
    // settlement in `account` charges the provider-reported usage, so
    // per-key TPM windows and priced budgets count family traffic.
    let engine = match state.engine() {
        Ok(engine) => engine,
        Err(error) => {
            let refusal = RouteError::Internal(error.to_string());
            return family_error(state.event_bus(), &ctx, &refusal);
        }
    };
    let admission = engine.shared_admission();
    let claim = llm_proxy_engine::AdmissionClaim {
        caller: &ctx.caller,
        provider,
        member: &target.provider_name,
        model: &target.upstream_model,
        requested_model: &ctx.requested_model,
        body_hash: &ctx.body_hash,
        estimated_input_tokens: 0,
        max_output_tokens: None,
    };
    let ticket = match admission.reserve(&claim).await {
        Ok(ticket) => ticket,
        Err(error) => return family_error(state.event_bus(), &ctx, &RouteError::from(error)),
    };
    // The reservation lives inside this handler future: a drop while the
    // upstream call runs (client disconnect, the family timeout) must
    // release the lease. `account` settles in both arms without suspending,
    // so the guard disarms right before it.
    let mut settle_guard =
        llm_proxy_engine::UnsettledLeaseGuard::new(Arc::clone(&admission), &ticket);

    // Forward on the provider's own transport (its `[provider.egress]`
    // client when it has one) and account.
    let outbound = Outbound {
        method,
        url: family_url(&target.endpoint, resource, query),
        body: outbound_body,
        content_type: outbound_content_type,
        headers,
    };
    let client = engine.transport().client_for(&target.provider_name);
    let upstream = forward(client, &target, api_key, outbound).await;
    settle_guard.disarm();
    account(
        state,
        admission.as_ref(),
        &ticket,
        &ctx,
        &target,
        upstream.map_err(|error| upstream_refusal(&error, !resource.is_empty(), passthrough)),
    )
    .await
}

/// Whether `path` is one or more identifier segments: unreserved URL
/// characters only and no `.`/`..` segment, so no URL parser can resolve
/// it outside the family endpoint it is appended to.
fn is_plain_resource_path(path: &str) -> bool {
    path.split('/').all(|segment| {
        !segment.is_empty()
            && segment != "."
            && segment != ".."
            && segment
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'~' | b'-'))
    })
}

/// The upstream URL for one call: the configured endpoint with the
/// resource path appended before the endpoint's own query, then the
/// client's query after it.
fn family_url(endpoint: &str, resource: &str, client_query: Option<&str>) -> String {
    let (base, endpoint_query) = match endpoint.split_once('?') {
        Some((base, query)) => (base, Some(query)),
        None => (endpoint, None),
    };
    let mut url = base.to_owned();
    if !resource.is_empty() {
        if !url.ends_with('/') {
            url.push('/');
        }
        url.push_str(resource);
    }
    let queries: Vec<&str> = [endpoint_query, client_query]
        .into_iter()
        .flatten()
        .filter(|query| !query.is_empty())
        .collect();
    if !queries.is_empty() {
        url.push('?');
        url.push_str(&queries.join("&"));
    }
    url
}

/// Sends the prepared family call to the resolved endpoint: the client's
/// method, the picked credential, the provider's static headers, the
/// allowlisted inbound headers, and the caller's content type.
async fn forward(
    client: &ProxyClient,
    target: &llm_proxy_core::ProviderAdapterTargetConfig,
    api_key: SecretString,
    outbound: Outbound<'_>,
) -> Result<UpstreamResponse, ProviderError> {
    let mut extra_headers: HashMap<String, String> = target
        .headers
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if let Some(content_type) = outbound.content_type {
        extra_headers.insert(header::CONTENT_TYPE.to_string(), content_type);
    }
    let inbound = inbound_headers_for_engine(outbound.headers);
    for (name, value) in llm_proxy_engine::headers::select_forwarded_headers(
        &inbound,
        &target.forward_inbound_headers,
    ) {
        extra_headers.insert(name, value);
    }
    let request = ProxyRequest {
        url: outbound.url,
        auth: AuthHeaders {
            style: target.auth_style,
            api_key,
        },
        body: outbound.body,
        stream: false,
        extra_headers,
        forwarded_headers: HashMap::new(),
    };
    client.send_raw(outbound.method, request).await
}

/// The client-facing refusal for an upstream failure: the one status
/// policy, plus the resource rule — a request that names a resource and
/// is answered `404` asked for something the provider does not have.
fn upstream_refusal(
    error: &ProviderError,
    names_a_resource: bool,
    passthrough: bool,
) -> RouteError {
    if names_a_resource && matches!(error, ProviderError::Api { status: 404, .. }) {
        return RouteError::NotFound;
    }
    RouteError::from(llm_proxy_engine::map_provider_error(error, passthrough))
}

/// Settles the reservation and writes the terminal usage row: the
/// successful call passes the provider's answer through, the failed one
/// answers the policy's refusal.
async fn account(
    state: &AppState,
    admission: &dyn llm_proxy_engine::Admission,
    ticket: &llm_proxy_engine::AdmissionTicket,
    ctx: &FamilyContext<'_>,
    target: &llm_proxy_core::ProviderAdapterTargetConfig,
    upstream: Result<UpstreamResponse, RouteError>,
) -> Response {
    match upstream {
        Ok(answer) => {
            // The families that report request-scoped token usage do so
            // in their JSON answers; it drives settlement (per-key TPM
            // windows and, where the model is priced, USD budgets) and
            // the ledger row. Everything else reports usage in its own
            // units (rows, seconds, images) and stays synthetic zero
            // rather than a fabricated number.
            let usage = family_usage(ctx.kind, &answer.body);
            let cost = usage.as_ref().and_then(|usage| {
                llm_proxy_engine::lifecycle::billed_cost(
                    usage,
                    state
                        .providers()
                        .pricing_for(&target.provider_name, &target.upstream_model),
                )
            });
            admission.settle(
                ticket,
                &llm_proxy_engine::AdmissionOutcome {
                    succeeded: true,
                    usage: usage.as_ref(),
                    cost: cost.as_ref(),
                },
            );
            state
                .event_bus()
                .emit(&ProxyEvent::ResponseCompleted(ResponseCompleted {
                    request_id: ctx.request_id.clone(),
                    timestamp: OffsetDateTime::now_utc(),
                    provider: target.provider_name.clone(),
                    upstream_message_id: None,
                    model: ModelRef {
                        requested: ctx.requested_model.clone(),
                        upstream: None,
                    },
                    usage: usage.unwrap_or_else(Usage::synthetic_zero),
                    cost,
                    stop_reason: llm_proxy_protocol::core::StopReason::EndTurn,
                    latency_ms: elapsed_ms(ctx.start),
                    credential_id: None,
                    cache_hit: false,
                }));
            pass_through(answer)
        }
        Err(refusal) => {
            admission.settle(
                ticket,
                &llm_proxy_engine::AdmissionOutcome {
                    succeeded: false,
                    usage: None,
                    cost: None,
                },
            );
            family_error(state.event_bus(), ctx, &refusal)
        }
    }
}

/// The provider-reported token usage in one family answer, for the
/// families whose documented APIs report request-scoped tokens in their
/// JSON responses: embeddings and image generations (the OpenAI
/// `usage.prompt_tokens` / `usage.total_tokens` and
/// `usage.input_tokens` / `usage.output_tokens` shapes), and rerank —
/// the flat `usage` variants plus Cohere's Rerank v2 envelope, which has
/// no top-level `usage` and reports under `meta.tokens`, falling back to
/// `meta.billed_units`. Files and batch answers carry aggregate
/// lifecycle usage, not this call's, and audio answers are not JSON —
/// both stay [`UsageProvenance::SyntheticZero`]. The answer is
/// deserialized into a struct naming only the usage fields, so a large
/// vector payload is skipped, never materialized.
fn family_usage(kind: ProviderRouteKind, body: &[u8]) -> Option<Usage> {
    if !matches!(
        kind,
        ProviderRouteKind::Embeddings
            | ProviderRouteKind::Rerank
            | ProviderRouteKind::ImagesGenerations
    ) {
        return None;
    }
    let answer: FamilyAnswer = serde_json::from_slice(body).ok()?;
    if let Some(usage) = answer.usage.as_ref().and_then(flat_usage_pair) {
        return Some(Usage::provider_reported(usage.0, usage.1));
    }
    // Cohere's rerank envelope: usage under meta.tokens, falling back to
    // meta.billed_units when meta.tokens is absent or carries no input.
    if kind == ProviderRouteKind::Rerank {
        let meta = answer.meta.as_ref()?;
        let counts = meta
            .tokens
            .as_ref()
            .filter(|tokens| tokens.input_tokens.is_some())
            .or(meta.billed_units.as_ref())?;
        let input = counts.input_tokens?;
        return Some(Usage::provider_reported(
            count_i32(input),
            count_i32(counts.output_tokens.unwrap_or(0)),
        ));
    }
    None
}

/// The input/output pair of a top-level `usage` object, when it carries
/// any token count at all.
fn flat_usage_pair(usage: &FlatUsage) -> Option<(i32, i32)> {
    let input = usage
        .prompt_tokens
        .or(usage.input_tokens)
        .or(usage.billed_tokens)?;
    let output = usage
        .output_tokens
        .or(usage.completion_tokens)
        .or_else(|| usage.total_tokens.map(|total| total.saturating_sub(input)))
        .unwrap_or(0);
    Some((count_i32(input), count_i32(output)))
}

fn count_i32(count: u64) -> i32 {
    i32::try_from(count).unwrap_or(i32::MAX)
}

/// A family answer, reduced to the fields the proxy reads: deserializing
/// into this shape skips everything else (embedding vectors, ranked
/// documents, image payloads) without allocating values for it.
#[derive(serde::Deserialize)]
struct FamilyAnswer {
    #[serde(default)]
    usage: Option<FlatUsage>,
    #[serde(default)]
    meta: Option<CohereMeta>,
}

/// A top-level `usage` object: the OpenAI embeddings/images fields plus
/// the flat variants rerank servers send.
#[derive(serde::Deserialize)]
struct FlatUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    billed_tokens: Option<u64>,
}

/// Cohere's per-response metadata envelope (Rerank v2).
#[derive(serde::Deserialize)]
struct CohereMeta {
    #[serde(default)]
    tokens: Option<TokenCounts>,
    #[serde(default)]
    billed_units: Option<TokenCounts>,
}

/// An `{input_tokens, output_tokens}` counts object.
#[derive(serde::Deserialize)]
struct TokenCounts {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
}

/// Rewrites the family body for egress: JSON families get their `model`
/// field aliased (the one surgical edit); byte families pass through
/// verbatim with their original content type, and a bodiless call
/// (a `GET`, a `DELETE`, a cancel) stays bodiless. Returns the requested
/// model name (for routing, admission and usage rows).
fn prepare_body(
    spec: &FamilySpec,
    body: &Bytes,
    headers: &HeaderMap,
    registry: &llm_proxy_core::ProviderRegistry,
    provider: &str,
) -> (String, Vec<u8>, Option<String>) {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    if !spec.json || body.is_empty() {
        return (String::new(), body.to_vec(), content_type);
    }
    let Ok(mut value) = serde_json::from_slice::<serde_json::Value>(body) else {
        return (String::new(), body.to_vec(), content_type);
    };
    let requested = value
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if !requested.is_empty()
        && let Some(upstream_model) = registry
            .get(provider)
            .and_then(|p| p.model_aliases.get(&requested))
    {
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "model".to_owned(),
                serde_json::Value::String(upstream_model.as_str().to_owned()),
            );
        }
    }
    let encoded = serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec());
    (requested, encoded, Some("application/json".to_owned()))
}

/// The provider's answer as the client receives it: its status, the
/// headers that describe the body, and the body. Family answers are
/// provider-shaped (JSON envelopes, audio, files); the proxy has no
/// family decoders and relabels nothing.
fn pass_through(answer: UpstreamResponse) -> Response {
    let status = StatusCode::from_u16(answer.status).unwrap_or(StatusCode::OK);
    let mut response = (status, answer.body).into_response();
    for (name, value) in [
        (header::CONTENT_TYPE, answer.content_type),
        (header::CONTENT_DISPOSITION, answer.content_disposition),
    ] {
        if let Some(value) = value.and_then(|value| HeaderValue::from_str(&value).ok()) {
            response.headers_mut().insert(name, value);
        }
    }
    response
}

/// One refusal, rendered once: the failure row, then the OpenAI-dialect
/// error body with the policy's `x-should-retry` and `Retry-After`.
fn family_error(bus: &Arc<dyn EventBus>, ctx: &FamilyContext<'_>, error: &RouteError) -> Response {
    let model = ModelRef {
        requested: ctx.requested_model.clone(),
        upstream: None,
    };
    emit_response_failed(
        bus,
        &ctx.request_id,
        Some(ctx.provider),
        Some(&model),
        error,
        ctx.start,
    );
    route_error_response(ClientProtocol::OpenAiChat, error)
}

fn emit_family_request(bus: &Arc<dyn EventBus>, ctx: &FamilyContext<'_>) {
    bus.emit(&ProxyEvent::RequestReceived(RequestReceived {
        request_id: ctx.request_id.clone(),
        timestamp: OffsetDateTime::now_utc(),
        caller: ctx.caller.clone(),
        provider: ctx.provider.to_owned(),
        route_kind: ctx.family.to_owned(),
        client_protocol: ctx.family.to_owned(),
        model: ModelRef {
            requested: ctx.requested_model.clone(),
            upstream: None,
        },
        streaming: false,
        body_hash: ctx.body_hash.clone(),
        session_id: None,
    }));
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn family_name(kind: ProviderRouteKind) -> &'static str {
    match kind {
        ProviderRouteKind::Embeddings => FAMILY_EMBEDDINGS,
        ProviderRouteKind::Rerank => FAMILY_RERANK,
        ProviderRouteKind::AudioTranscriptions => FAMILY_AUDIO_TRANSCRIPTIONS,
        ProviderRouteKind::AudioSpeech => FAMILY_AUDIO_SPEECH,
        ProviderRouteKind::ImagesGenerations => FAMILY_IMAGES_GENERATIONS,
        ProviderRouteKind::Files => FAMILY_FILES,
        ProviderRouteKind::Batches => FAMILY_BATCHES,
        _ => "chat",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_proxy_protocol::core::UsageProvenance;

    #[test]
    fn the_resource_goes_between_the_endpoint_and_its_query() {
        assert_eq!(
            family_url("https://h/v1/files", "file-1/content", None),
            "https://h/v1/files/file-1/content"
        );
        assert_eq!(
            family_url(
                "https://h/v1/files?api-version=1",
                "file-1",
                Some("limit=2")
            ),
            "https://h/v1/files/file-1?api-version=1&limit=2"
        );
        assert_eq!(
            family_url("https://h/v1/batches/", "b_1/cancel", Some("")),
            "https://h/v1/batches/b_1/cancel",
            "no doubled slash, no empty query"
        );
        assert_eq!(
            family_url("https://h/v1/files", "", Some("purpose=batch")),
            "https://h/v1/files?purpose=batch"
        );
    }

    #[test]
    fn only_identifier_segments_form_a_resource_path() {
        for accepted in ["file-abc", "file-abc/content", "batch_1/cancel", "v1.2~x"] {
            assert!(is_plain_resource_path(accepted), "{accepted}");
        }
        for refused in [
            "..", "a/../b", "./a", "a//b", "a/", "%2e%2e", "a/%2E%2e", "a?b=c", "a#b", "a b",
            "a\\..\\b", "",
        ] {
            assert!(!is_plain_resource_path(refused), "{refused:?}");
        }
    }

    #[test]
    fn the_documented_usage_shapes_become_provider_reported_usage() {
        // OpenAI embeddings: prompt_tokens + total_tokens.
        let openai = family_usage(
            ProviderRouteKind::Embeddings,
            br#"{"data":[],"usage":{"prompt_tokens":5,"total_tokens":5}}"#,
        )
        .expect("usage parsed");
        assert_eq!(
            (openai.input_tokens, openai.output_tokens),
            (5, 0),
            "total == prompt, so output is zero"
        );
        assert_eq!(openai.provenance, UsageProvenance::ProviderReported);

        // Cohere Rerank v2: no top-level usage; meta.tokens preferred.
        let cohere = family_usage(
            ProviderRouteKind::Rerank,
            br#"{"results":[],"id":"9f75d94c","meta":{
                "api_version":{"version":"2"},
                "billed_units":{"input_tokens":7},
                "tokens":{"input_tokens":7,"output_tokens":3}}}"#,
        )
        .expect("usage parsed");
        assert_eq!(
            (cohere.input_tokens, cohere.output_tokens),
            (7, 3),
            "meta.tokens wins over billed_units"
        );
        assert_eq!(cohere.provenance, UsageProvenance::ProviderReported);

        // Cohere's documented example body reports only billed_units.
        let billed_only = family_usage(
            ProviderRouteKind::Rerank,
            br#"{"results":[],"meta":{"api_version":{"version":"2"},
                "billed_units":{"input_tokens":192}}}"#,
        )
        .expect("usage parsed");
        assert_eq!(
            (billed_only.input_tokens, billed_only.output_tokens),
            (192, 0)
        );

        // Flat rerank variants and image generations' shape.
        let billed = family_usage(
            ProviderRouteKind::Rerank,
            br#"{"results":[],"usage":{"billed_tokens":9}}"#,
        )
        .expect("usage parsed");
        assert_eq!((billed.input_tokens, billed.output_tokens), (9, 0));
        let images = family_usage(
            ProviderRouteKind::ImagesGenerations,
            br#"{"data":[],"usage":{"input_tokens":11,"output_tokens":423,"total_tokens":434}}"#,
        )
        .expect("usage parsed");
        assert_eq!((images.input_tokens, images.output_tokens), (11, 423));
    }

    #[test]
    fn families_without_request_scoped_usage_stay_synthetic_zero() {
        // A files/batches answer carries lifecycle aggregates, not this
        // call's tokens; audio answers are not JSON. Neither is priced or
        // charged into TPM windows from numbers that are not this call's.
        let batch_list = br#"{"data":[{"usage":{"total_tokens":1234}}]}"#;
        assert!(family_usage(ProviderRouteKind::Files, batch_list).is_none());
        assert!(family_usage(ProviderRouteKind::Batches, batch_list).is_none());
        assert!(
            family_usage(
                ProviderRouteKind::AudioSpeech,
                b"RIFF\x24\x00\x00\x00WAVEfmt "
            )
            .is_none()
        );
        // A JSON answer from a token-reporting family without a usage
        // object (an unpriced self-hosted server) is not fabricated.
        assert!(family_usage(ProviderRouteKind::Embeddings, br#"{"data":[]}"#).is_none());
        // A rerank meta with no token counts anywhere reports nothing.
        assert!(
            family_usage(
                ProviderRouteKind::Rerank,
                br#"{"results":[],"meta":{"api_version":{"version":"2"},"warnings":[]}}"#
            )
            .is_none()
        );
    }
}
