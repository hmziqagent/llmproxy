//! Plan 09 step 7 against **real** upstream implementations of each
//! family's wire API, run locally — not loopback fakes. A client talks to
//! the proxy exactly as it would talk to the provider (the documented
//! OpenAI `/v1/embeddings`, `/v1/audio/*`, `/v1/images/generations`,
//! `/v1/files` and `/v1/batches` shapes and the Cohere/Jina-style
//! `/v1/rerank`), and the proxy forwards to a real server:
//!
//! - embeddings and rerank: llama.cpp's `llama-server`;
//! - audio speech and transcriptions: speaches (Kokoro TTS and
//!   faster-whisper);
//! - images: stable-diffusion.cpp's `sd-server` (SD-Turbo);
//! - files and batches: Llama Stack (`inline::localfs` files,
//!   `inline::reference` batches answered by a llama.cpp chat model).
//!
//! Every test is `#[ignore]`d and does nothing unless its family's
//! variable names a live server, so run any subset. The reference setup
//! (CPU only; GGUF models from Hugging Face in `$MODELS`):
//!
//! ```text
//! docker run -d -p 18081:8080 -v $MODELS:/models ghcr.io/ggml-org/llama.cpp:server \
//!     -m /models/bge-small-en-v1.5-q8_0.gguf --embeddings --host 0.0.0.0 --port 8080
//! docker run -d -p 18082:8080 -v $MODELS:/models ghcr.io/ggml-org/llama.cpp:server \
//!     -m /models/bge-reranker-v2-m3-Q4_K_M.gguf --reranking --host 0.0.0.0 --port 8080
//! docker run -d -p 18084:8000 ghcr.io/speaches-ai/speaches:latest-cpu
//! curl -X POST localhost:18084/v1/models/Systran/faster-whisper-base.en
//! curl -X POST localhost:18084/v1/models/speaches-ai/Kokoro-82M-v1.0-ONNX-int8
//! sd-server -m $MODELS/sd_turbo-f16-q8_0.gguf --listen-port 18085 --steps 1 --cfg-scale 1
//! llama stack run run.yaml   # port 18086: inference remote::vllm -> a llama.cpp chat
//!                            # server, files inline::localfs, batches inline::reference
//!                            # (llama-stack 0.7.3 needs `pip install "mcp<2"`)
//!
//! LLM_PROXY_TEST_EMBEDDINGS_URL=http://127.0.0.1:18081/v1 \
//! LLM_PROXY_TEST_RERANK_URL=http://127.0.0.1:18082/v1 \
//! LLM_PROXY_TEST_AUDIO_URL=http://127.0.0.1:18084/v1 \
//! LLM_PROXY_TEST_IMAGES_URL=http://127.0.0.1:18085/v1 \
//! LLM_PROXY_TEST_FILES_URL=http://127.0.0.1:18086/v1 \
//!     cargo test -p llm-proxy-server --features endpoint-families \
//!         --test live_families -- --ignored
//! ```
//!
//! Upstream model names default to that setup and can be overridden with
//! `LLM_PROXY_TEST_EMBEDDINGS_MODEL`, `LLM_PROXY_TEST_RERANK_MODEL`,
//! `LLM_PROXY_TEST_SPEECH_MODEL`, `LLM_PROXY_TEST_SPEECH_VOICE`,
//! `LLM_PROXY_TEST_TRANSCRIPTION_MODEL`, `LLM_PROXY_TEST_IMAGES_MODEL` and
//! `LLM_PROXY_TEST_BATCH_MODEL`; `LLM_PROXY_TEST_FAMILIES_KEY` is sent as
//! the upstream bearer key. The embeddings, rerank and speech clients name
//! an alias, and their servers echo or require the upstream name, so those
//! tests also prove the model rewrite reached the server.

#![cfg(test)]
#![cfg(feature = "endpoint-families")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, StatusCode, header};
use llm_proxy_adapters::ProviderAdapterRegistry;
use llm_proxy_core::{
    AppConfig, AuthStyle, ProviderAdapterConfig, ProviderConfig, ProviderRegistry,
    ProviderRouteKind, ProviderRoutesConfig, ServerConfig,
};
use llm_proxy_server::{AppState, BuildInfo, build_router};
use llm_proxy_storage::{ProxyEvent, RecordingBus};
use llm_proxy_transport::ProxyClient;
use tower::ServiceExt;

const PROVIDER: &str = "live";
const MAX_BODY: usize = 64 << 20;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn env_or(name: &str, default: &str) -> String {
    env(name).unwrap_or_else(|| default.to_owned())
}

/// The live server for one test, or `None` (with a note) when unset:
/// `--ignored` runs every test, and only the configured ones have a
/// server to talk to.
fn live_base(var: &str) -> Option<String> {
    let base = env(var);
    if base.is_none() {
        eprintln!("skipped: {var} is not set");
    }
    base.map(|base| base.trim_end_matches('/').to_owned())
}

/// One provider whose every family route points under `base` (an
/// OpenAI-style `/v1` root), with the given client-facing aliases.
fn provider(base: &str, aliases: &[(&str, &str)]) -> ProviderConfig {
    let families = [
        (ProviderRouteKind::Embeddings, "embeddings"),
        (ProviderRouteKind::Rerank, "rerank"),
        (
            ProviderRouteKind::AudioTranscriptions,
            "audio/transcriptions",
        ),
        (ProviderRouteKind::AudioSpeech, "audio/speech"),
        (ProviderRouteKind::ImagesGenerations, "images/generations"),
        (ProviderRouteKind::Files, "files"),
        (ProviderRouteKind::Batches, "batches"),
    ];
    let adapters = families
        .iter()
        .map(|(kind, path)| {
            (
                format!("{kind:?}"),
                ProviderAdapterConfig {
                    protocol: "openai_chat_completions".to_owned(),
                    endpoint: format!("{base}/{path}"),
                    headers: Arc::new(HashMap::new()),
                    forward_inbound_headers: None,
                    lossy: false,
                },
            )
        })
        .collect();
    let route = |kind: ProviderRouteKind| Some(format!("{kind:?}"));
    ProviderConfig {
        name: PROVIDER.to_owned(),
        api_key: secrecy::SecretString::from(env_or(
            "LLM_PROXY_TEST_FAMILIES_KEY",
            "sk-live-families-test",
        )),
        auth_style: AuthStyle::Bearer,
        passthrough_auth: false,
        api_keys: None,
        retry: None,
        egress: None,
        adapters,
        routes: ProviderRoutesConfig {
            messages: None,
            chat_completions: None,
            responses: None,
            generate_content: None,
            embeddings: route(ProviderRouteKind::Embeddings),
            rerank: route(ProviderRouteKind::Rerank),
            audio_transcriptions: route(ProviderRouteKind::AudioTranscriptions),
            audio_speech: route(ProviderRouteKind::AudioSpeech),
            images_generations: route(ProviderRouteKind::ImagesGenerations),
            files: route(ProviderRouteKind::Files),
            batches: route(ProviderRouteKind::Batches),
        },
        model_aliases: aliases
            .iter()
            .map(|(alias, upstream)| ((*alias).to_owned(), (*upstream).to_owned()))
            .collect(),
        discovery: None,
        catalog: None,
        pricing: Default::default(),
        concurrency: None,
        pricing_sync: None,
        prompt_cache: None,
    }
}

fn config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            bind: "127.0.0.1:0".parse().expect("bind"),
            request_timeout: Duration::from_secs(300),
            shutdown_timeout: Duration::from_secs(5),
            log_level: "info".to_owned(),
            hot_reload: false,
            allowed_origins: None,
            server_name: "live-families".to_owned(),
            rate_limit_rpm: 10_000,
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
        auth: Default::default(),
        storage: None,
        replica: None,
        mcp: None,
    }
}

/// The proxy under test over one live server.
fn proxy(base: &str, aliases: &[(&str, &str)]) -> (Router, Arc<RecordingBus>) {
    let registry =
        ProviderRegistry::from_providers(vec![provider(base, aliases)]).expect("registry builds");
    let bus = Arc::new(RecordingBus::new());
    let state = AppState::new(
        config(),
        registry,
        ProviderAdapterRegistry::builtin(),
        ProxyClient::try_new()
            .expect("client")
            .with_upstream_headers_timeout(Duration::from_secs(300)),
        BuildInfo {
            name: "live-families",
            version: "0.0.0",
            target: "test",
            git_sha: "test",
        },
    )
    .expect("app state")
    .with_event_bus(Arc::clone(&bus) as Arc<dyn llm_proxy_storage::EventBus>);
    (build_router(state), bus)
}

/// One answered request through the proxy.
struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
}

impl Reply {
    fn content_type(&self) -> &str {
        self.headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
    }

    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|error| {
            panic!(
                "expected JSON ({error}), got {} {:?}: {}",
                self.status,
                self.content_type(),
                String::from_utf8_lossy(&self.body)
            )
        })
    }

    fn describe(&self) -> String {
        format!(
            "{} {:?}: {}",
            self.status,
            self.content_type(),
            String::from_utf8_lossy(self.body.get(..self.body.len().min(400)).unwrap_or(&[]))
        )
    }
}

async fn send(
    app: &Router,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: Vec<u8>,
) -> Reply {
    let mut request = Request::builder()
        .method(method)
        .uri(format!("/providers/{PROVIDER}{path}"));
    if let Some(content_type) = content_type {
        request = request.header(header::CONTENT_TYPE, content_type);
    }
    let response = app
        .clone()
        .oneshot(request.body(Body::from(body)).expect("request"))
        .await
        .expect("served");
    let status = response.status();
    let headers = response.headers().clone();
    let body = axum::body::to_bytes(response.into_body(), MAX_BODY)
        .await
        .expect("body");
    Reply {
        status,
        headers,
        body,
    }
}

async fn post_json(app: &Router, path: &str, body: &serde_json::Value) -> Reply {
    send(
        app,
        "POST",
        path,
        Some("application/json"),
        body.to_string().into_bytes(),
    )
    .await
}

/// A multipart/form-data body with the client's own boundary: text
/// fields, then one file part.
fn multipart(boundary: &str, fields: &[(&str, &str)], file: (&str, &str, &[u8])) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    let (filename, content_type, bytes) = file;
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

fn floats(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .expect("embedding array")
        .iter()
        .map(|x| x.as_f64().expect("float component"))
        .collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

/// The family's request row and terminal row, in order.
fn family_rows(bus: &RecordingBus) -> Vec<(String, String)> {
    bus.snapshot()
        .iter()
        .filter_map(|event| match event {
            ProxyEvent::RequestReceived(row) => {
                Some(("request".to_owned(), row.route_kind.clone()))
            }
            ProxyEvent::ResponseCompleted(row) => {
                Some(("completed".to_owned(), row.model.requested.clone()))
            }
            ProxyEvent::ResponseFailed(row) => Some(("failed".to_owned(), row.error_kind.clone())),
            _ => None,
        })
        .collect()
}

#[tokio::test]
#[ignore = "needs a live embeddings server; set LLM_PROXY_TEST_EMBEDDINGS_URL"]
async fn embeddings_reach_a_real_server_through_the_alias() {
    let Some(base) = live_base("LLM_PROXY_TEST_EMBEDDINGS_URL") else {
        return;
    };
    let model = env_or("LLM_PROXY_TEST_EMBEDDINGS_MODEL", "bge-small-en-v1.5");
    let (app, bus) = proxy(&base, &[("live-embed", &model)]);

    let reply = post_json(
        &app,
        "/v1/embeddings",
        &serde_json::json!({
            "model": "live-embed",
            "input": [
                "The cat sat on the mat.",
                "A kitten was resting on the rug.",
                "Quarterly revenue grew four percent."
            ],
            "encoding_format": "float"
        }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.describe());
    assert!(reply.content_type().starts_with("application/json"));
    let body = reply.json();
    assert_eq!(
        body["model"].as_str(),
        Some(model.as_str()),
        "the server answered for the aliased upstream model"
    );
    let data = body["data"].as_array().expect("data");
    assert_eq!(data.len(), 3, "one vector per input");
    let indices: Vec<u64> = data
        .iter()
        .map(|item| item["index"].as_u64().expect("index"))
        .collect();
    assert_eq!(indices, vec![0, 1, 2], "input order is preserved");
    let vectors: Vec<Vec<f64>> = data.iter().map(|item| floats(&item["embedding"])).collect();
    let dimension = vectors.first().map(Vec::len).unwrap_or_default();
    assert!(dimension > 0);
    assert!(vectors.iter().all(|v| v.len() == dimension));
    assert!(
        cosine(&vectors[0], &vectors[1]) > cosine(&vectors[0], &vectors[2]),
        "a real model ranks the paraphrase closer than the unrelated sentence"
    );
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0);
    assert_eq!(
        family_rows(&bus),
        vec![
            ("request".to_owned(), "embeddings".to_owned()),
            ("completed".to_owned(), "live-embed".to_owned()),
        ]
    );
}

#[tokio::test]
#[ignore = "needs a live rerank server; set LLM_PROXY_TEST_RERANK_URL"]
async fn rerank_reaches_a_real_reranker_through_the_alias() {
    let Some(base) = live_base("LLM_PROXY_TEST_RERANK_URL") else {
        return;
    };
    let model = env_or("LLM_PROXY_TEST_RERANK_MODEL", "bge-reranker-v2-m3");
    let (app, bus) = proxy(&base, &[("live-rerank", &model)]);

    let reply = post_json(
        &app,
        "/v1/rerank",
        &serde_json::json!({
            "model": "live-rerank",
            "query": "What is the capital of France?",
            "documents": [
                "Bananas are rich in potassium.",
                "Paris is the capital and largest city of France.",
                "The Danube flows through Vienna."
            ],
            "top_n": 2
        }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.describe());
    let body = reply.json();
    assert_eq!(body["model"].as_str(), Some(model.as_str()));
    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), 2, "top_n is honoured");
    assert_eq!(
        results[0]["index"].as_u64(),
        Some(1),
        "the relevant document ranks first: {results:?}"
    );
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["relevance_score"].as_f64().expect("score"))
        .collect();
    assert!(scores[0] > scores[1], "scores descend: {scores:?}");
    assert_eq!(
        family_rows(&bus).first(),
        Some(&("request".to_owned(), "rerank".to_owned()))
    );
}

#[tokio::test]
#[ignore = "needs a live audio server; set LLM_PROXY_TEST_AUDIO_URL"]
#[expect(
    clippy::too_many_lines,
    reason = "one real audio round trip: speech, then its two transcription formats, each with asserts"
)]
async fn speech_and_transcription_round_trip_through_a_real_audio_server() {
    let Some(base) = live_base("LLM_PROXY_TEST_AUDIO_URL") else {
        return;
    };
    let speech_model = env_or(
        "LLM_PROXY_TEST_SPEECH_MODEL",
        "speaches-ai/Kokoro-82M-v1.0-ONNX-int8",
    );
    let voice = env_or("LLM_PROXY_TEST_SPEECH_VOICE", "af_heart");
    let transcription_model = env_or(
        "LLM_PROXY_TEST_TRANSCRIPTION_MODEL",
        "Systran/faster-whisper-base.en",
    );
    let (app, bus) = proxy(&base, &[("live-tts", &speech_model)]);

    // Speech: a JSON request (aliased) answered with audio bytes.
    let speech = post_json(
        &app,
        "/v1/audio/speech",
        &serde_json::json!({
            "model": "live-tts",
            "voice": voice,
            "input": "The quick brown fox jumps over the lazy dog.",
            "response_format": "wav"
        }),
    )
    .await;
    assert_eq!(speech.status, StatusCode::OK, "{}", speech.describe());
    assert!(
        speech.content_type().starts_with("audio/"),
        "the audio keeps the server's content type, got {:?}",
        speech.content_type()
    );
    let wav = speech.body.to_vec();
    assert!(
        wav.starts_with(b"RIFF") && wav.get(8..12) == Some(b"WAVE".as_slice()),
        "the body is the server's WAV bytes"
    );

    // Transcription: the client's own multipart body, then the server's
    // JSON answer and its plain-text answer.
    let boundary = "live-families-boundary-7d1f";
    let content_type = format!("multipart/form-data; boundary={boundary}");
    let json_form = multipart(
        boundary,
        &[("model", &transcription_model), ("response_format", "json")],
        ("speech.wav", "audio/wav", &wav),
    );
    let transcript = send(
        &app,
        "POST",
        "/v1/audio/transcriptions",
        Some(&content_type),
        json_form,
    )
    .await;
    assert_eq!(
        transcript.status,
        StatusCode::OK,
        "{}",
        transcript.describe()
    );
    let text = transcript.json()["text"]
        .as_str()
        .unwrap_or_default()
        .to_lowercase();
    assert!(
        text.contains("quick brown fox"),
        "the server heard the proxied speech, got {text:?}"
    );

    let text_form = multipart(
        boundary,
        &[("model", &transcription_model), ("response_format", "text")],
        ("speech.wav", "audio/wav", &wav),
    );
    let plain = send(
        &app,
        "POST",
        "/v1/audio/transcriptions",
        Some(&content_type),
        text_form,
    )
    .await;
    assert_eq!(plain.status, StatusCode::OK, "{}", plain.describe());
    assert!(
        plain.content_type().starts_with("text/plain"),
        "a text transcript is not relabelled as JSON, got {:?}",
        plain.content_type()
    );
    assert!(
        String::from_utf8_lossy(&plain.body)
            .to_lowercase()
            .contains("lazy dog")
    );

    let kinds: Vec<String> = family_rows(&bus)
        .into_iter()
        .filter(|(row, _)| row == "request")
        .map(|(_, kind)| kind)
        .collect();
    assert_eq!(
        kinds,
        vec![
            "audio_speech".to_owned(),
            "audio_transcriptions".to_owned(),
            "audio_transcriptions".to_owned(),
        ]
    );
}

#[tokio::test]
#[ignore = "needs a live image server; set LLM_PROXY_TEST_IMAGES_URL"]
async fn image_generation_reaches_a_real_diffusion_server() {
    let Some(base) = live_base("LLM_PROXY_TEST_IMAGES_URL") else {
        return;
    };
    let model = env_or("LLM_PROXY_TEST_IMAGES_MODEL", "sd-turbo");
    let (app, bus) = proxy(&base, &[("live-image", &model)]);

    let reply = post_json(
        &app,
        "/v1/images/generations",
        &serde_json::json!({
            "model": "live-image",
            "prompt": "a red circle on a white background",
            "n": 1,
            "size": "256x256",
            "response_format": "b64_json"
        }),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK, "{}", reply.describe());
    let body = reply.json();
    let image = body["data"][0]["b64_json"].as_str().unwrap_or_default();
    assert!(
        image.starts_with("iVBORw0KGgo"),
        "the server returned a base64 PNG (its signature), got {:?}",
        image.get(..16)
    );
    assert_eq!(
        family_rows(&bus),
        vec![
            ("request".to_owned(), "images_generations".to_owned()),
            ("completed".to_owned(), "live-image".to_owned()),
        ]
    );
}

#[tokio::test]
#[ignore = "needs a live files and batches server; set LLM_PROXY_TEST_FILES_URL"]
#[expect(
    clippy::too_many_lines,
    reason = "one real files-and-batches lifecycle: upload, retrieve, list, download, batch to completion, output, delete"
)]
async fn files_and_batches_run_their_whole_lifecycle_through_a_real_server() {
    let Some(base) = live_base("LLM_PROXY_TEST_FILES_URL") else {
        return;
    };
    let model = env_or("LLM_PROXY_TEST_BATCH_MODEL", "vllm/smollm2-135m");
    let (app, bus) = proxy(&base, &[]);

    // Upload the batch input: the client's multipart body.
    let lines: String = ["live-1", "live-2"]
        .iter()
        .map(|id| {
            format!(
                "{}\n",
                serde_json::json!({
                    "custom_id": id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 8
                    }
                })
            )
        })
        .collect();
    let boundary = "live-files-boundary-3c9e";
    let upload = send(
        &app,
        "POST",
        "/v1/files",
        Some(&format!("multipart/form-data; boundary={boundary}")),
        multipart(
            boundary,
            &[("purpose", "batch")],
            ("input.jsonl", "application/jsonl", lines.as_bytes()),
        ),
    )
    .await;
    assert_eq!(upload.status, StatusCode::OK, "{}", upload.describe());
    let file = upload.json();
    let file_id = file["id"].as_str().expect("file id").to_owned();
    assert_eq!(file["purpose"].as_str(), Some("batch"));
    assert_eq!(file["bytes"].as_u64(), Some(lines.len() as u64));

    // Retrieve, list (with the client's query) and download it.
    let retrieved = send(&app, "GET", &format!("/v1/files/{file_id}"), None, vec![]).await;
    assert_eq!(retrieved.status, StatusCode::OK, "{}", retrieved.describe());
    assert_eq!(retrieved.json()["id"].as_str(), Some(file_id.as_str()));

    let listed = send(
        &app,
        "GET",
        "/v1/files?purpose=batch&limit=100",
        None,
        vec![],
    )
    .await;
    assert_eq!(listed.status, StatusCode::OK, "{}", listed.describe());
    let listed = listed.json();
    let ids: Vec<&str> = listed["data"]
        .as_array()
        .expect("file list")
        .iter()
        .filter_map(|f| f["id"].as_str())
        .collect();
    assert!(ids.contains(&file_id.as_str()), "listed: {ids:?}");

    let content = send(
        &app,
        "GET",
        &format!("/v1/files/{file_id}/content"),
        None,
        vec![],
    )
    .await;
    assert_eq!(content.status, StatusCode::OK, "{}", content.describe());
    assert_eq!(
        content.body.as_ref(),
        lines.as_bytes(),
        "the download is the uploaded bytes"
    );
    assert!(
        !content.content_type().starts_with("application/json"),
        "the file keeps the server's content type, got {:?}",
        content.content_type()
    );

    // Create a batch over it and follow it to completion.
    let created = post_json(
        &app,
        "/v1/batches",
        &serde_json::json!({
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK, "{}", created.describe());
    let batch_id = created.json()["id"].as_str().expect("batch id").to_owned();

    let started = Instant::now();
    let batch = loop {
        let polled = send(
            &app,
            "GET",
            &format!("/v1/batches/{batch_id}"),
            None,
            vec![],
        )
        .await;
        assert_eq!(polled.status, StatusCode::OK, "{}", polled.describe());
        let batch = polled.json();
        match batch["status"].as_str() {
            Some("completed") => break batch,
            Some("failed" | "expired" | "cancelled") => panic!("batch ended badly: {batch}"),
            _ => {}
        }
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "batch never completed: {batch}"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    assert_eq!(batch["request_counts"]["completed"].as_u64(), Some(2));

    let output_id = batch["output_file_id"].as_str().expect("output file");
    let output = send(
        &app,
        "GET",
        &format!("/v1/files/{output_id}/content"),
        None,
        vec![],
    )
    .await;
    assert_eq!(output.status, StatusCode::OK, "{}", output.describe());
    let mut answered: Vec<String> = String::from_utf8_lossy(&output.body)
        .lines()
        .map(|line| {
            let row: serde_json::Value = serde_json::from_str(line).expect("jsonl row");
            assert_eq!(row["response"]["status_code"].as_u64(), Some(200), "{row}");
            row["custom_id"].as_str().unwrap_or_default().to_owned()
        })
        .collect();
    answered.sort();
    assert_eq!(answered, vec!["live-1".to_owned(), "live-2".to_owned()]);

    let batches = send(&app, "GET", "/v1/batches?limit=100", None, vec![]).await;
    assert_eq!(batches.status, StatusCode::OK, "{}", batches.describe());
    assert!(
        batches.json()["data"]
            .as_array()
            .expect("batch list")
            .iter()
            .any(|b| b["id"].as_str() == Some(batch_id.as_str()))
    );

    // Delete the input file; the server then reports it unknown, and the
    // client sees that as the 404 it is, not an upstream fault.
    let deleted = send(
        &app,
        "DELETE",
        &format!("/v1/files/{file_id}"),
        None,
        vec![],
    )
    .await;
    assert_eq!(deleted.status, StatusCode::OK, "{}", deleted.describe());
    assert_eq!(deleted.json()["deleted"].as_bool(), Some(true));
    let gone = send(&app, "GET", &format!("/v1/files/{file_id}"), None, vec![]).await;
    assert_eq!(gone.status, StatusCode::NOT_FOUND, "{}", gone.describe());

    let kinds: Vec<String> = family_rows(&bus)
        .into_iter()
        .filter(|(row, _)| row == "request")
        .map(|(_, kind)| kind)
        .collect();
    assert!(kinds.iter().any(|kind| kind == "files"));
    assert!(kinds.iter().any(|kind| kind == "batches"));
}
