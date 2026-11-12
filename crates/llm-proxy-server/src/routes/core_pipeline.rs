//! The server side of the core pipeline: HTTP-shaped glue around the
//! engine. Everything upstream of the client codec — admission, routing,
//! credentials, egress, settlement — lives in `llm-proxy-engine`; this
//! module decodes the client dialect into a [`EngineRequest`], runs the
//! engine, and encodes the client's response or SSE stream.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderMap, Response, StatusCode, header};
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{BoxStream, StreamExt};
use llm_proxy_engine::{
    ClientDialect, Engine, EngineError, EngineRequest, InboundRequest as EngineInbound,
};
use llm_proxy_protocol::client::anthropic;
use llm_proxy_protocol::client::anthropic::StreamEncoder as AnthropicStreamEncoder;
use llm_proxy_protocol::client::gemini;
use llm_proxy_protocol::client::gemini::StreamEncoder as GeminiStreamEncoder;
use llm_proxy_protocol::client::ollama;
use llm_proxy_protocol::client::ollama::StreamEncoder as OllamaStreamEncoder;
use llm_proxy_protocol::client::openai_chat;
use llm_proxy_protocol::client::openai_chat::StreamEncoder as OpenAiStreamEncoder;
use llm_proxy_protocol::client::responses;
use llm_proxy_protocol::client::responses::StreamEncoder as ResponsesStreamEncoder;
use llm_proxy_protocol::core::{CoreEvent, CoreRequest, CoreStreamError, ModelRef};
use llm_proxy_storage::{EventBus, ProxyEvent, ResponseFailed};
use secrecy::SecretString;
use tracing::warn;

use crate::middleware::get_client_ip;
use crate::state::AppState;

use super::error_response::{
    ClientProtocol, RouteError, StreamWire, extract_error_fields, route_error_variant_name,
};

const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(3);

/// The per-request context the server assembles before calling the engine.
#[derive(Debug)]
pub(crate) struct RequestContext {
    pub(crate) request_id: String,
    pub(crate) start: Instant,
    pub(crate) caller: String,
    pub(crate) body_hash: String,
}

/// Applies caller limits (rate limiting — pipeline slot 2) before the engine
/// takes over.
pub(crate) fn prepare_request(
    state: &AppState,
    meta: &crate::middleware::RequestMeta,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<RequestContext, RouteError> {
    let caller = &meta.caller;
    let connect_info = meta.connect_info.as_ref();
    let trust_forwarded_headers = state.trust_forwarded_headers();
    let client_ip = get_client_ip(headers, connect_info, trust_forwarded_headers);
    let direct_loopback = !trust_forwarded_headers
        && !caller.is_authenticated()
        && connect_info.is_some_and(|address| address.ip().is_loopback());

    let rate_limit_key = if caller.is_authenticated() {
        caller.as_str().to_owned()
    } else {
        client_ip
    };
    if !direct_loopback && !state.rate_limiter.is_allowed(&rate_limit_key) {
        state.metrics.record_rate_limited();
        return Err(RouteError::RateLimited {
            message: "rate limit exceeded".to_owned(),
            retry_after_secs: Some(1),
        });
    }

    Ok(RequestContext {
        request_id: meta.request_id.0.clone(),
        start: Instant::now(),
        caller: caller.as_str().to_owned(),
        body_hash: body_hash_of_bytes(body),
    })
}

/// Validates a provider name from the URL path (engine rule, server error).
pub(super) fn validate_provider_name(name: &str) -> Result<(), RouteError> {
    llm_proxy_engine::validate_provider_name(name).map_err(RouteError::from)
}

/// Whether `name` addresses a configured provider or pool (plan 04 step 4):
/// both live under `/providers/{name}/v1/...`.
pub(crate) fn route_target_exists(state: &AppState, name: &str) -> bool {
    state.providers().contains_route(name)
}

/// Bare `/v1/*` model resolution (plan 04 step 11): the configured model map
/// first, then the default target; an unresolvable model is a 404 naming it.
/// The caller has already checked that `server.bare_v1.enabled` is on.
pub(crate) fn resolve_bare_target(state: &AppState, model: &str) -> Result<String, RouteError> {
    let bare = &state.app_config().server.bare_v1;
    let target = bare
        .models
        .get(model)
        .cloned()
        .or_else(|| bare.default_target.clone())
        .ok_or_else(|| RouteError::UnknownProvider(model.to_owned()))?;
    if !route_target_exists(state, &target) {
        return Err(RouteError::UnknownProvider(target));
    }
    Ok(target)
}

/// The `model` field of a raw client body, for bare-route resolution.
pub(crate) fn bare_model_of(body: &[u8]) -> Result<String, RouteError> {
    let raw: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| RouteError::InvalidRequest(format!("invalid JSON: {e}")))?;
    raw.get("model")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            RouteError::InvalidRequest(
                "the bare /v1 routes need a \"model\" field to resolve the target".to_owned(),
            )
        })
}

pub(super) fn validate_json_content_type(headers: &HeaderMap) -> Result<(), RouteError> {
    let Some(value) = headers.get(header::CONTENT_TYPE) else {
        return Ok(());
    };
    let Ok(ct) = value.to_str() else {
        return Ok(());
    };
    let essence = ct
        .split(';')
        .next()
        .unwrap_or(ct)
        .trim()
        .to_ascii_lowercase();
    let is_json = essence == "application/json" || essence.ends_with("+json");
    if !is_json {
        return Err(RouteError::InvalidRequest(format!(
            "expected application/json Content-Type, got {ct}"
        )));
    }
    Ok(())
}

/// Writes the [`ProxyEvent::ResponseFailed`] row for a server-mapped error
/// (engine failures record their own rows).
pub(crate) fn emit_response_failed(
    event_bus: &Arc<dyn EventBus>,
    request_id: &str,
    provider: Option<&str>,
    model: Option<&ModelRef>,
    error: &RouteError,
    start: Instant,
) {
    let provider = provider.or(match error {
        RouteError::UnknownProvider(name) => Some(name.as_str()),
        _ => None,
    });
    if let RouteError::Upstream {
        status: upstream_status,
        body,
        auth_owner,
        retry_after_secs: _,
    } = error
    {
        warn!(
            upstream_status = upstream_status.as_u16(),
            auth_owner = ?auth_owner,
            upstream_body = %body,
            "upstream error mapped for the client; real body logged here only"
        );
    }
    let (status, _client_error_type, message) = extract_error_fields(error);
    let error_kind = route_error_variant_name(error);
    event_bus.emit(&ProxyEvent::ResponseFailed(ResponseFailed {
        request_id: request_id.to_owned(),
        timestamp: time::OffsetDateTime::now_utc(),
        provider: provider.map(str::to_owned),
        model: model.cloned(),
        error_kind: error_kind.to_owned(),
        message,
        http_status: status.as_u16(),
        latency_ms: start.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
    }));
}

/// Bearer/x-api-key token from the client, if any (passthrough auth).
pub(crate) fn extract_inbound_auth(headers: &HeaderMap) -> Option<SecretString> {
    if let Some(value) = headers.get(header::AUTHORIZATION) {
        if let Ok(s) = value.to_str() {
            let mut parts = s.splitn(2, ' ');
            match (parts.next(), parts.next()) {
                (Some(scheme), Some(token)) if scheme.eq_ignore_ascii_case("bearer") => {
                    let token = token.trim();
                    if !token.is_empty() {
                        return Some(SecretString::from(token.to_owned()));
                    }
                }
                _ => {}
            }
        }
    }
    for name in ["x-api-key", "x-goog-api-key"] {
        if let Some(value) = headers.get(name) {
            if let Ok(s) = value.to_str() {
                if !s.is_empty() {
                    return Some(SecretString::from(s.to_owned()));
                }
            }
        }
    }
    None
}

/// Inbound headers as the engine sees them: lowercased names, UTF-8 values,
/// first value per name (the forwarding allowlist semantics it applies).
pub(crate) fn inbound_headers_for_engine(
    headers: &HeaderMap,
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for (name, value) in headers.iter() {
        let Ok(text) = value.to_str() else {
            tracing::trace!(header = %name.as_str(), "inbound header value is not UTF-8; not forwarded");
            continue;
        };
        map.entry(name.as_str().to_ascii_lowercase())
            .or_insert_with(|| text.to_owned());
    }
    map
}

pub(crate) fn body_hash_of_bytes(body: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(body);
    format!("{:x}", hasher.finalize())
}

/// Everything the engine needs about the inbound request beyond the decoded
/// core: auth token, headers, raw wire body.
pub(crate) struct InboundRequest {
    /// Bearer/x-api-key token from the client, if any.
    pub(crate) auth: Option<SecretString>,
    /// The client's request headers.
    pub(crate) headers: HeaderMap,
    /// The client's raw wire request (dialect + body).
    pub(crate) wire: llm_proxy_adapters::ClientWireRequest,
}

fn engine_request(
    ctx: &RequestContext,
    provider_name: &str,
    route_kind: llm_proxy_core::ProviderRouteKind,
    client_protocol: ClientProtocol,
    core: CoreRequest,
    inbound: InboundRequest,
) -> EngineRequest {
    let InboundRequest {
        auth,
        headers,
        wire,
    } = inbound;
    let dialect = match client_protocol {
        ClientProtocol::Anthropic => ClientDialect::Anthropic,
        ClientProtocol::OpenAiChat => ClientDialect::OpenAiChat,
        ClientProtocol::OpenAiResponses => ClientDialect::OpenAiResponses,
        ClientProtocol::Gemini => ClientDialect::Gemini,
        ClientProtocol::Ollama => ClientDialect::Ollama,
    };
    EngineRequest::new(
        provider_name,
        route_kind,
        dialect,
        core,
        EngineInbound {
            auth,
            headers: inbound_headers_for_engine(&headers),
            wire,
        },
    )
    .with_request_id(ctx.request_id.clone())
    .with_caller(ctx.caller.clone())
    .with_body_hash(ctx.body_hash.clone())
    .with_start(ctx.start)
}

fn map_engine_error(error: EngineError) -> RouteError {
    RouteError::from(error)
}

/// Non-streaming: run the engine, then hand the client its dialect.
pub(crate) async fn handle_core_once(
    state: AppState,
    ctx: RequestContext,
    dispatch: CoreDispatch<'_>,
    core: CoreRequest,
    inbound: InboundRequest,
) -> Result<Response<Body>, RouteError> {
    let client_protocol = dispatch.client_protocol;
    let request = engine_request(
        &ctx,
        dispatch.provider_name,
        dispatch.route_kind,
        client_protocol,
        core,
        inbound,
    );
    let engine = state.engine().map_err(RouteError::from)?;
    run_once(&engine, request, client_protocol, dispatch.ollama_generate).await
}

async fn run_once(
    engine: &Arc<Engine>,
    request: EngineRequest,
    client_protocol: ClientProtocol,
    ollama_generate: bool,
) -> Result<Response<Body>, RouteError> {
    let request_id = request.request_id.clone();
    let outcome = engine.execute(request).await;
    let core_resp = match outcome {
        Ok(response) => response,
        Err(error) => return Err(map_engine_error(error)),
    };
    // Same-format passthrough: the engine hands back the exact upstream
    // bytes; the decode only read usage and the stop reason for the ledger.
    let response_body = match core_resp.raw_body {
        Some(bytes) => bytes,
        None => {
            let encoded: Result<Vec<u8>, String> = match client_protocol {
                ClientProtocol::Anthropic => anthropic::encode_response(core_resp.core)
                    .map_err(|e| e.to_string())
                    .and_then(|msg_resp| serde_json::to_vec(&msg_resp).map_err(|e| e.to_string())),
                ClientProtocol::OpenAiChat => openai_chat::encode_response(core_resp.core)
                    .map_err(|e| e.to_string())
                    .and_then(|chat_resp| {
                        serde_json::to_vec(&chat_resp).map_err(|e| e.to_string())
                    }),
                ClientProtocol::OpenAiResponses => responses::encode_response(&core_resp.core)
                    .map_err(|e| e.to_string())
                    .and_then(|resp| serde_json::to_vec(&resp).map_err(|e| e.to_string())),
                ClientProtocol::Gemini => gemini::encode_response(&core_resp.core)
                    .map_err(|e| e.to_string())
                    .and_then(|resp| serde_json::to_vec(&resp).map_err(|e| e.to_string())),
                ClientProtocol::Ollama => {
                    let encoded = if ollama_generate {
                        ollama::encode_generate_response(&core_resp.core)
                    } else {
                        ollama::encode_chat_response(&core_resp.core)
                    };
                    encoded
                        .map_err(|e| e.to_string())
                        .and_then(|resp| serde_json::to_vec(&resp).map_err(|e| e.to_string()))
                }
            };
            match encoded {
                Ok(bytes) => bytes,
                Err(e) => {
                    // The upstream response completed (and is settled in the
                    // ledger); only the client re-encode failed.
                    warn!(error = %e, "client response encode failed after settlement");
                    return Err(RouteError::Internal(format!("client encode: {e}")));
                }
            }
        }
    };

    let mut response = (StatusCode::OK, Body::from(response_body)).into_response();
    response.headers_mut().insert(
        "x-request-id",
        request_id
            .parse()
            .unwrap_or_else(|_| axum::http::HeaderValue::from_static("unknown")),
    );
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );

    Ok(response)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CoreDispatch<'a> {
    pub(crate) provider_name: &'a str,
    pub(crate) route_kind: llm_proxy_core::ProviderRouteKind,
    pub(crate) client_protocol: ClientProtocol,
    /// Ollama framing selector: `/api/generate` (true) or `/api/chat`.
    pub(crate) ollama_generate: bool,
}

#[expect(
    clippy::large_enum_variant,
    reason = "the encoders are built once per request and moved into the stream task; \
              boxing them would add an allocation per dialect for no size win"
)]
enum ClientStreamEncoder {
    Anthropic(AnthropicStreamEncoder),
    OpenAi(OpenAiStreamEncoder),
    Responses(ResponsesStreamEncoder),
    Gemini(GeminiStreamEncoder),
    Ollama(OllamaStreamEncoder),
}
enum ClientEncodedEvent {
    /// An SSE event with an `event:` line (Anthropic, Responses).
    Named { event_type: String, json: String },
    /// An SSE data-only event (OpenAI Chat, Gemini `alt=sse`).
    Data { json: String },
    /// One raw JSON line (Ollama NDJSON; also the Gemini JSON-array chunk).
    Line { json: String },
}

impl ClientEncodedEvent {
    fn into_line(self) -> String {
        match self {
            Self::Named { json, .. } | Self::Data { json } | Self::Line { json } => json,
        }
    }
}

impl ClientStreamEncoder {
    fn new(
        protocol: ClientProtocol,
        msg_id: String,
        model: String,
        core: &CoreRequest,
        ollama_generate: bool,
    ) -> Self {
        match protocol {
            ClientProtocol::Anthropic => {
                Self::Anthropic(AnthropicStreamEncoder::new(msg_id, model))
            }
            ClientProtocol::OpenAiChat => {
                let include_usage = core
                    .provider_hints
                    .raw
                    .get("stream_options")
                    .and_then(|v| v.get("include_usage"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let created = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                Self::OpenAi(OpenAiStreamEncoder::new(
                    msg_id,
                    model,
                    created,
                    include_usage,
                ))
            }
            ClientProtocol::OpenAiResponses => {
                Self::Responses(ResponsesStreamEncoder::new(msg_id, model))
            }
            ClientProtocol::Gemini => Self::Gemini(GeminiStreamEncoder::new(model)),
            ClientProtocol::Ollama => {
                Self::Ollama(OllamaStreamEncoder::new(model, ollama_generate))
            }
        }
    }
    fn encode_event(
        &mut self,
        event: CoreEvent,
    ) -> Result<Vec<ClientEncodedEvent>, llm_proxy_protocol::client::ProtocolError> {
        match self {
            Self::Anthropic(enc) => {
                let msg_events = enc.encode_event(event)?;
                Ok(wrap_anthropic_events(msg_events))
            }
            Self::OpenAi(enc) => {
                let chunks = enc.encode_event(event)?;
                Ok(wrap_openai_events(chunks))
            }
            Self::Responses(enc) => {
                let events = enc.encode_event(event)?;
                Ok(wrap_named_json(events))
            }
            Self::Gemini(enc) => {
                let chunks = enc.encode_event(event)?;
                Ok(wrap_data_json(chunks))
            }
            Self::Ollama(enc) => {
                let objects = enc.encode_event(event)?;
                Ok(wrap_data_json(objects))
            }
        }
    }
    fn finish(
        &mut self,
    ) -> Result<Vec<ClientEncodedEvent>, llm_proxy_protocol::client::ProtocolError> {
        match self {
            Self::Anthropic(enc) => {
                let msg_events = enc.finish()?;
                Ok(wrap_anthropic_events(msg_events))
            }
            Self::OpenAi(enc) => {
                let chunks = enc.finish()?;
                Ok(wrap_openai_events(chunks))
            }
            Self::Responses(enc) => {
                let events = enc.finish()?;
                Ok(wrap_named_json(events))
            }
            Self::Gemini(enc) => {
                let chunks = enc.finish()?;
                Ok(wrap_data_json(chunks))
            }
            Self::Ollama(enc) => {
                let objects = enc.finish()?;
                Ok(wrap_data_json(objects))
            }
        }
    }
    fn mark_finished(&mut self) {
        match self {
            Self::Anthropic(enc) => enc.mark_finished(),
            Self::OpenAi(enc) => enc.mark_finished(),
            Self::Responses(enc) => enc.mark_finished(),
            Self::Gemini(enc) => enc.mark_finished(),
            Self::Ollama(enc) => enc.mark_finished(),
        }
    }
}

fn wrap_anthropic_events(
    events: Vec<llm_proxy_protocol::anthropic::MessageEvent>,
) -> Vec<ClientEncodedEvent> {
    events
        .into_iter()
        .filter_map(|me| match serde_json::to_string(&me) {
            Ok(json) => {
                let event_type = me.r#type;
                Some(ClientEncodedEvent::Named { event_type, json })
            }
            Err(e) => {
                warn!(error = %e, "failed to serialize Anthropic SSE event; dropping");
                None
            }
        })
        .collect()
}
fn wrap_openai_events<T: serde::Serialize>(chunks: Vec<T>) -> Vec<ClientEncodedEvent> {
    chunks
        .into_iter()
        .filter_map(|chunk| match serde_json::to_string(&chunk) {
            Ok(json) => Some(ClientEncodedEvent::Data { json }),
            Err(e) => {
                warn!(error = %e, "failed to serialize OpenAI SSE chunk; dropping");
                None
            }
        })
        .collect()
}
fn wrap_named_json(events: Vec<responses::ResponsesSseEvent>) -> Vec<ClientEncodedEvent> {
    events
        .into_iter()
        .filter_map(|event| match serde_json::to_string(&event.payload) {
            Ok(json) => Some(ClientEncodedEvent::Named {
                event_type: event.event_type,
                json,
            }),
            Err(e) => {
                warn!(error = %e, "failed to serialize Responses SSE event; dropping");
                None
            }
        })
        .collect()
}
fn wrap_data_json(values: Vec<serde_json::Value>) -> Vec<ClientEncodedEvent> {
    values
        .into_iter()
        .filter_map(|value| match serde_json::to_string(&value) {
            Ok(json) => Some(ClientEncodedEvent::Data { json }),
            Err(e) => {
                warn!(error = %e, "failed to serialize stream chunk; dropping");
                None
            }
        })
        .collect()
}
fn client_event_to_sse(encoded: ClientEncodedEvent) -> Event {
    match encoded {
        ClientEncodedEvent::Named { event_type, json } => {
            Event::default().event(&event_type).data(json)
        }
        ClientEncodedEvent::Data { json } => Event::default().data(json),
        ClientEncodedEvent::Line { json } => Event::default().data(json),
    }
}
/// The in-band error framing for dialects whose stream chunks cannot carry
/// an error themselves. The Anthropic and Responses encoders render the
/// error through their own event shapes.
fn dialect_error_chunk(
    protocol: ClientProtocol,
    error: &CoreStreamError,
) -> Option<ClientEncodedEvent> {
    match protocol {
        ClientProtocol::OpenAiChat => openai_chat::stream_error_sse_payload(error)
            .map(|json| ClientEncodedEvent::Data { json }),
        ClientProtocol::Gemini => {
            let payload = gemini::error_payload(500, error.message());
            let json = serde_json::to_string(&payload)
                .unwrap_or_else(|_| "{\"error\":{\"code\":500}}".to_owned());
            Some(ClientEncodedEvent::Data { json })
        }
        ClientProtocol::Ollama => {
            let payload = ollama::error_payload(error.message());
            let json = serde_json::to_string(&payload)
                .unwrap_or_else(|_| "{\"error\":\"stream failed\"}".to_owned());
            Some(ClientEncodedEvent::Line { json })
        }
        ClientProtocol::Anthropic | ClientProtocol::OpenAiResponses => None,
    }
}
/// Extracts the upstream message id from the buffered prefix, falling back
/// to a dialect-shaped generated id (client codec policy, not engine).
fn message_id_for(events: &[CoreEvent], protocol: ClientProtocol) -> String {
    events
        .iter()
        .find_map(|event| match event {
            CoreEvent::MessageStart { id: Some(id), .. } => Some(id.clone()),
            _ => None,
        })
        .unwrap_or_else(|| match protocol {
            ClientProtocol::OpenAiChat => format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            ClientProtocol::Anthropic => format!("msg_{}", uuid::Uuid::new_v4()),
            ClientProtocol::OpenAiResponses => format!("resp_{}", uuid::Uuid::new_v4().simple()),
            // Gemini and Ollama chunks carry no message id.
            ClientProtocol::Gemini | ClientProtocol::Ollama => String::new(),
        })
}

/// Streaming: run the engine to the first committed byte, then encode the
/// core events into the client's dialect framing (`wire`; SSE for the
/// Anthropic/OpenAI/Gemini-SSE dialects, NDJSON for Ollama, a JSON array for
/// Gemini without `alt=sse`). Settlement is driven by the engine as the
/// returned stream completes or is dropped.
pub(crate) async fn handle_core_stream(
    state: AppState,
    ctx: RequestContext,
    dispatch: CoreDispatch<'_>,
    core: CoreRequest,
    inbound: InboundRequest,
    wire: StreamWire,
) -> Result<Response<Body>, RouteError> {
    let client_protocol = dispatch.client_protocol;
    let request = engine_request(
        &ctx,
        dispatch.provider_name,
        dispatch.route_kind,
        client_protocol,
        core.clone(),
        inbound,
    );
    let request_id = request.request_id.clone();
    let engine = state.engine().map_err(RouteError::from)?;
    let stream = match engine.execute_stream(request).await {
        Ok(stream) => stream,
        Err(error) => return Err(map_engine_error(error)),
    };

    let mut encoder = ClientStreamEncoder::new(
        client_protocol,
        message_id_for(&stream.head, client_protocol),
        core.model.requested.clone(),
        &core,
        dispatch.ollama_generate,
    );
    let head_events = encode_head(&mut encoder, client_protocol, stream.head);

    let tail = encode_tail(encoder, client_protocol, stream.tail);

    let all = futures::stream::iter(head_events).chain(tail);
    let (mut parts, body) = match wire {
        StreamWire::SseNamed | StreamWire::SseData => {
            let sse = Sse::new(
                all.map(client_event_to_sse)
                    .map(Ok::<_, std::convert::Infallible>),
            )
            .keep_alive(KeepAlive::new().interval(HEARTBEAT_INTERVAL));
            let response = sse.into_response();
            let (mut parts, body) = response.into_parts();
            parts.headers.insert(
                header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("text/event-stream"),
            );
            parts.headers.insert(
                header::CACHE_CONTROL,
                axum::http::HeaderValue::from_static("no-cache"),
            );
            parts.headers.insert(
                header::CONNECTION,
                axum::http::HeaderValue::from_static("keep-alive"),
            );
            parts.headers.insert(
                "x-accel-buffering",
                axum::http::HeaderValue::from_static("no"),
            );
            (parts, body)
        }
        StreamWire::Ndjson => {
            let lines = all.map(|event| {
                Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
                    "{}\n",
                    event.into_line()
                )))
            });
            let response = Response::new(Body::from_stream(lines));
            let (mut parts, body) = response.into_parts();
            parts.headers.insert(
                header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/x-ndjson"),
            );
            (parts, body)
        }
        StreamWire::JsonArray => {
            let response = Response::new(Body::from_stream(json_array_stream(all)));
            let (mut parts, body) = response.into_parts();
            parts.headers.insert(
                header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            (parts, body)
        }
    };
    parts.headers.insert(
        "x-request-id",
        request_id
            .parse()
            .unwrap_or_else(|_| axum::http::HeaderValue::from_static("unknown")),
    );
    tracing::debug!(request_id = %request_id, "streaming started");
    Ok(Response::from_parts(parts, body))
}

/// A JSON array of chunks: `"["`, the chunks separated by commas, `"]"` (the
/// Gemini `streamGenerateContent` framing without `alt=sse`).
fn json_array_stream<S>(
    all: S,
) -> impl futures::Stream<Item = Result<axum::body::Bytes, std::convert::Infallible>>
where
    S: futures::Stream<Item = ClientEncodedEvent>,
{
    futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(b"["))
    })
    .chain(all.enumerate().map(|(index, event)| {
        let comma = if index == 0 { "" } else { "," };
        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(format!(
            "{comma}{}",
            event.into_line()
        )))
    }))
    .chain(futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(axum::body::Bytes::from_static(b"]"))
    }))
}

/// Encodes the buffered prefix, applying the client-dialect error framing
/// (an OpenAI leading error crosses as a bare error object plus `[DONE]`).
fn encode_head(
    encoder: &mut ClientStreamEncoder,
    protocol: ClientProtocol,
    head: Vec<CoreEvent>,
) -> Vec<ClientEncodedEvent> {
    let mut events = Vec::new();
    for core_event in head {
        match &core_event {
            CoreEvent::Error { error } => {
                // A leading error crosses the first-byte boundary in the
                // dialect's error framing. The encoder first: Anthropic and
                // Responses render the error through their own event shapes,
                // the OpenAI codec flushes a buffered finish chunk when one
                // exists (its refusal otherwise is documented behaviour, not
                // a failure), Gemini and Ollama self-finish and render
                // nothing.
                match encoder.encode_event(core_event.clone()) {
                    Ok(mut encoded) => events.append(&mut encoded),
                    Err(e) => {
                        tracing::trace!(
                            error = %e,
                            "codec refused the leading error event; error framing follows"
                        );
                    }
                }
                if let Some(chunk) = dialect_error_chunk(protocol, error) {
                    events.push(chunk);
                }
                if matches!(protocol, ClientProtocol::OpenAiChat) {
                    events.push(ClientEncodedEvent::Data {
                        json: "[DONE]".to_owned(),
                    });
                }
                encoder.mark_finished();
                return events;
            }
            _ => match encoder.encode_event(core_event) {
                Ok(mut encoded) => events.append(&mut encoded),
                Err(e) => {
                    warn!(error = %e, "failed to encode buffered head event; dropping");
                }
            },
        }
    }
    events
}

/// Encodes one core event through the client codec.
///
/// In-band errors after commit never touch the OpenAI codec: its chunk
/// schema cannot carry an error (the encoder would refuse and self-finish),
/// so an OpenAI client gets the bare error object instead, and the synthetic
/// terminal plus `[DONE]` still follow from the encoder's own flush. An
/// Anthropic client gets a real `error` SSE event; a Responses client gets
/// `response.failed`; Gemini and Ollama get their error chunk shapes.
fn encode_one(
    encoder: &mut ClientStreamEncoder,
    protocol: ClientProtocol,
    core_event: CoreEvent,
) -> Vec<ClientEncodedEvent> {
    if let CoreEvent::Error { error } = &core_event
        && matches!(protocol, ClientProtocol::OpenAiChat)
    {
        let mut events = Vec::new();
        if let Some(chunk) = dialect_error_chunk(protocol, error) {
            events.push(chunk);
        }
        return events;
    }
    let mut events = Vec::new();
    // The encoder first: with an in-band error, Anthropic and Responses
    // render it through their own event shapes, while Gemini and Ollama
    // self-finish and render nothing — so their dialect error chunk follows
    // here, the same framing `encode_terminal_error` adds when the error
    // ends the stream. Without it the error would vanish whenever the stream
    // carries any event past the error (the decoder's end-of-stream stop).
    let in_band_error = if let CoreEvent::Error { error } = &core_event {
        Some(error.clone())
    } else {
        None
    };
    match encoder.encode_event(core_event) {
        Ok(mut encoded) => events.append(&mut encoded),
        Err(e) => {
            if in_band_error.is_some() {
                tracing::trace!(
                    error = %e,
                    "codec refused the in-band error event; error framing follows"
                );
            } else {
                warn!(error = %e, "failed to encode stream event; dropping");
            }
        }
    }
    if let Some(error) = in_band_error.as_ref()
        && let Some(chunk) = dialect_error_chunk(protocol, error)
    {
        events.push(chunk);
    }
    events
}

/// An in-band error as the stream's last item (transport/decode failure
/// after commit): the codec renders what it can, then the dialect's error
/// framing; the encoder is marked finished so no synthetic terminal follows.
fn encode_terminal_error(
    encoder: &mut ClientStreamEncoder,
    protocol: ClientProtocol,
    error: &llm_proxy_protocol::core::CoreStreamError,
) -> Vec<ClientEncodedEvent> {
    let mut events = Vec::new();
    // The encoder first: with a terminal buffered it flushes the finish
    // chunk the client is still owed ahead of the error; without one it
    // self-finishes and returns nothing. The OpenAI codec refuses the bare
    // error event itself by design — the dialect error chunk and [DONE]
    // below carry the error — so a refusal is traced, not warned.
    match encoder.encode_event(CoreEvent::Error {
        error: error.clone(),
    }) {
        Ok(mut encoded) => events.append(&mut encoded),
        Err(e) => {
            tracing::trace!(
                error = %e,
                "codec refused the terminal error event; error framing follows"
            );
        }
    }
    if let Some(chunk) = dialect_error_chunk(protocol, error) {
        events.push(chunk);
    }
    if matches!(protocol, ClientProtocol::OpenAiChat) {
        events.push(ClientEncodedEvent::Data {
            json: "[DONE]".to_owned(),
        });
    }
    encoder.mark_finished();
    events
}

/// The terminal flush: `encoder.finish()` plus `[DONE]` for OpenAI.
fn encode_finish(
    encoder: &mut ClientStreamEncoder,
    protocol: ClientProtocol,
) -> Vec<ClientEncodedEvent> {
    let mut events = Vec::new();
    match encoder.finish() {
        Ok(mut final_encoded) => {
            events.append(&mut final_encoded);
            if matches!(protocol, ClientProtocol::OpenAiChat) {
                events.push(ClientEncodedEvent::Data {
                    json: "[DONE]".to_owned(),
                });
            }
        }
        Err(e) => {
            warn!(error = %e, "client encoder finish error");
        }
    }
    events
}

/// Lazily encodes the engine's continuation stream into client chunks.
///
/// One-item lookahead on error events preserves the pre-split framing
/// contract: an error followed by more events renders in-band only (the
/// synthetic terminal and `[DONE]` still follow at the natural end); an
/// error as the last item terminates the stream with the error framing
/// itself and no synthetic terminal.
fn encode_tail(
    encoder: ClientStreamEncoder,
    protocol: ClientProtocol,
    tail: BoxStream<'static, CoreEvent>,
) -> BoxStream<'static, ClientEncodedEvent> {
    struct EncodeState {
        encoder: ClientStreamEncoder,
        protocol: ClientProtocol,
        tail: BoxStream<'static, CoreEvent>,
        /// Set once the terminal flush (or terminal error) has been emitted.
        finished: bool,
    }
    futures::stream::unfold(
        EncodeState {
            encoder,
            protocol,
            tail,
            finished: false,
        },
        |mut state| async move {
            if state.finished {
                return None;
            }
            let Some(core_event) = state.tail.next().await else {
                let events = encode_finish(&mut state.encoder, state.protocol);
                state.finished = true;
                return Some((events, state));
            };
            if !matches!(core_event, CoreEvent::Error { .. }) {
                let events = encode_one(&mut state.encoder, state.protocol, core_event);
                return Some((events, state));
            }
            // Error event: hold it back until we know whether the stream
            // continues past it. The `else` branch above handled everything
            // else, so the refutable pattern is total here.
            let CoreEvent::Error { error } = core_event else {
                return None;
            };
            match state.tail.next().await {
                Some(next) => {
                    let mut events = encode_one(
                        &mut state.encoder,
                        state.protocol,
                        CoreEvent::Error { error },
                    );
                    events.extend(encode_one(&mut state.encoder, state.protocol, next));
                    Some((events, state))
                }
                None => {
                    let events = encode_terminal_error(&mut state.encoder, state.protocol, &error);
                    state.finished = true;
                    Some((events, state))
                }
            }
        },
    )
    .map(futures::stream::iter)
    .flatten()
    .boxed()
}

pub(crate) fn protocol_error_to_route(e: llm_proxy_protocol::client::ProtocolError) -> RouteError {
    match e {
        llm_proxy_protocol::client::ProtocolError::InvalidRequest(msg) => {
            RouteError::InvalidRequest(msg)
        }
        llm_proxy_protocol::client::ProtocolError::Decode(msg) => RouteError::InvalidRequest(msg),
        other => RouteError::Internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_hash_is_stable_and_distinguishes_requests() {
        let a1 = br#"{"model":"gpt-4o","messages":[]}"#;
        let a2 = br#"{"model":"gpt-4o","messages":[]}"#;
        let b = br#"{"model":"gpt-3.5-turbo","messages":[]}"#;
        assert_eq!(body_hash_of_bytes(a1), body_hash_of_bytes(a2));
        assert_ne!(body_hash_of_bytes(a1), body_hash_of_bytes(b));
    }

    #[test]
    fn extract_inbound_auth_bearer_scheme_is_case_insensitive() {
        use secrecy::ExposeSecret;
        fn extracted_token(auth_header: &str) -> Option<String> {
            let mut headers = HeaderMap::new();
            headers.insert(header::AUTHORIZATION, auth_header.parse().unwrap());
            extract_inbound_auth(&headers).map(|s| s.expose_secret().to_owned())
        }
        assert_eq!(extracted_token("Bearer abc123").as_deref(), Some("abc123"));
        assert_eq!(extracted_token("bearer abc123").as_deref(), Some("abc123"));
        assert_eq!(extracted_token("BEARER abc123").as_deref(), Some("abc123"));
        assert_eq!(extracted_token("BeArEr abc123").as_deref(), Some("abc123"));
        assert_eq!(
            extracted_token("Bearer   abc123   ").as_deref(),
            Some("abc123")
        );
        assert_eq!(extracted_token("Basic dXNlcjpwYXNz"), None);
        assert_eq!(extracted_token("Bearer "), None);
    }

    #[test]
    fn forwarded_headers_honour_the_allowlist_end_to_end() {
        let mut headers = HeaderMap::new();
        headers.insert("Anthropic-Beta", "context-1m-2025-08-07".parse().unwrap());
        headers.insert("x-claude-code-session-id", "sess-1".parse().unwrap());
        headers.insert("user-agent", "probe/1".parse().unwrap());
        let allow = vec![
            "Anthropic-Beta".to_owned(),
            "x-claude-code-session-id".to_owned(),
        ];
        let forwarded = llm_proxy_engine::select_forwarded_headers(
            &inbound_headers_for_engine(&headers),
            &allow,
        );
        assert_eq!(
            forwarded.get("anthropic-beta").map(String::as_str),
            Some("context-1m-2025-08-07")
        );
        assert_eq!(
            forwarded
                .get("x-claude-code-session-id")
                .map(String::as_str),
            Some("sess-1")
        );
        assert_eq!(forwarded.len(), 2, "unlisted headers must not forward");
    }

    #[test]
    fn client_auth_origin_and_browser_headers_are_never_forwarded() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer client-secret".parse().unwrap());
        headers.insert("x-api-key", "client-key".parse().unwrap());
        headers.insert("origin", "https://evil.example".parse().unwrap());
        headers.insert("user-agent", "probe/1".parse().unwrap());
        headers.insert("sec-fetch-mode", "cors".parse().unwrap());
        headers.insert("x-request-id", "client-id".parse().unwrap());
        let allow: Vec<String> = headers
            .keys()
            .map(|name| name.as_str().to_owned())
            .collect();
        let forwarded = llm_proxy_engine::select_forwarded_headers(
            &inbound_headers_for_engine(&headers),
            &allow,
        );
        assert!(
            forwarded.is_empty(),
            "client auth, Origin, browser headers and client request ids must never forward, got {forwarded:?}"
        );
    }

    #[test]
    fn inbound_headers_lowercase_names_and_first_value() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Api-Key", "first".parse().unwrap());
        headers.append("X-Api-Key", "second".parse().unwrap());
        let map = inbound_headers_for_engine(&headers);
        assert_eq!(map.get("x-api-key").map(String::as_str), Some("first"));
    }

    #[test]
    fn engine_error_maps_to_route_error_status() {
        assert!(matches!(
            map_engine_error(EngineError::Unauthorized),
            RouteError::Unauthorized
        ));
        assert!(matches!(
            map_engine_error(EngineError::InvalidRequest("x".into())),
            RouteError::InvalidRequest(_)
        ));
    }
}
