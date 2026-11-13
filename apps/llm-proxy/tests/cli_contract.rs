//! Binary-level contract replay (fix-brief 03-9): two plan-02 contract
//! fixtures — one non-streaming, one streaming — replay through the
//! spawned `llm-proxy` binary against a loopback fake upstream. The
//! request reaching the fake upstream must equal the fixture body apart
//! from the same allowed edits the in-process suite
//! (`crates/llm-proxy-server/tests/contract_replay.rs`) permits; for
//! these two Anthropic fixtures the only allowed edit is the model
//! alias. Fixture provenance lives in
//! `crates/llm-proxy-server/tests/fixtures/contract/README.md` (the
//! fixtures are frozen; nothing here writes to them).

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::path::Path;
    use std::process::{Child, Command, Stdio};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use axum::{response::IntoResponse, routing::post};
    use serde_json::Value;

    const WAIT_LIMIT: Duration = Duration::from_secs(15);
    const UPSTREAM_MODEL: &str = "claude-sonnet-4-5-20250929";

    /// The canned non-streaming reply the fake upstream sends; the same
    /// body `contract_replay.rs` uses for Anthropic Messages fixtures.
    fn anthropic_json_reply() -> String {
        serde_json::to_string(&serde_json::json!({
            "id": "msg_fixture_01",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": "fixture reply" }],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 64,
                "cache_read_input_tokens": 32
            }
        }))
        .expect("serialize anthropic fixture response")
    }

    /// The canned streaming reply; the same SSE sequence
    /// `contract_replay.rs` sends for Anthropic stream fixtures.
    const ANTHROPIC_STREAM_REPLY: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

"#;

    type Capture = Arc<Mutex<Option<String>>>;

    /// A loopback fake upstream serving `/v1/messages`: it captures the
    /// request body and answers with the canned Anthropic body (JSON for
    /// non-streaming, SSE for streaming).
    async fn spawn_fake_upstream(streaming: bool) -> (SocketAddr, Capture) {
        let capture: Capture = Arc::new(Mutex::new(None));
        let capture_for_handler = Arc::clone(&capture);

        let app = axum::Router::new().route(
            "/v1/messages",
            post(move |body: axum::body::Bytes| {
                let capture = Arc::clone(&capture_for_handler);
                async move {
                    *capture.lock().expect("capture lock") =
                        Some(String::from_utf8_lossy(&body).into_owned());
                    if streaming {
                        (
                            axum::http::StatusCode::OK,
                            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                            ANTHROPIC_STREAM_REPLY,
                        )
                            .into_response()
                    } else {
                        (
                            axum::http::StatusCode::OK,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            anthropic_json_reply(),
                        )
                            .into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("fake upstream bind");
        let addr = listener.local_addr().expect("fake upstream addr");
        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("fake upstream serves");
        });
        (addr, capture)
    }

    fn captured_request(capture: &Capture) -> String {
        capture
            .lock()
            .expect("capture lock")
            .clone()
            .expect("fake upstream received a request")
    }

    /// contract_replay.rs's comparison rule, copied rather than exported:
    /// the upstream body must equal the fixture apart from the allowed
    /// edits; for these two Anthropic fixtures the only allowed edit is
    /// the model alias.
    fn expected_upstream_body(fixture_body: Value, upstream_model: &str) -> Value {
        let mut expected = fixture_body;
        expected["model"] = Value::String(upstream_model.to_owned());
        expected
    }

    fn fixture_body(relative: &str) -> Value {
        let path = format!(
            "{}/../../crates/llm-proxy-server/tests/fixtures/contract/{relative}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse fixture {relative}: {e}"))
    }

    struct RunningServer {
        child: Child,
        addr: SocketAddr,
        _dir: tempfile::TempDir,
    }

    fn write_provider_file(dir: &Path, upstream: SocketAddr) {
        std::fs::create_dir(dir.join("providers")).expect("providers dir");
        std::fs::write(
            dir.join("providers").join("fixture-provider.toml"),
            format!(
                r#"[provider]
name = "fixture-provider"
api_key = "test-upstream-key-not-real"
auth_style = "bearer"

[provider.adapters.messages]
protocol = "anthropic_messages"
endpoint = "http://{upstream}/v1/messages"

[provider.routes]
messages = "messages"

[provider.model_aliases]
claude-sonnet-4-5 = "{UPSTREAM_MODEL}"
"#
            ),
        )
        .expect("write provider file");
    }

    fn write_config(dir: &Path, bind: SocketAddr) -> std::path::PathBuf {
        let config = dir.join("config.toml");
        std::fs::write(
            &config,
            format!(
                r#"[server]
bind = "{bind}"
request_timeout = "60s"
shutdown_timeout = "1s"
log_level = "warn"
hot_reload = false
rate_limit_rpm = 0
trust_forwarded_headers = false
server_name = "cli-contract-test"
[auth]
mode = "none"
[storage]
path = "{}"
"#,
                dir.join("usage.db").display(),
            ),
        )
        .expect("write config");
        config
    }

    impl RunningServer {
        /// Starts the binary against the fake upstream and returns once it
        /// answers `/health` with its own 200 (the same readiness rule
        /// `cli_serve.rs` uses, so a port squatter cannot pass).
        fn start(upstream: SocketAddr) -> Self {
            let dir = tempfile::tempdir().expect("tempdir");
            let addr = TcpListener::bind("127.0.0.1:0")
                .expect("bind probe")
                .local_addr()
                .expect("probe addr");
            write_provider_file(dir.path(), upstream);
            let config = write_config(dir.path(), addr);

            let child = Command::new(env!("CARGO_BIN_EXE_llm-proxy"))
                .args(["serve", "--config"])
                .arg(&config)
                .env("HOME", dir.path())
                .env("NO_PROXY", "127.0.0.1")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn llm-proxy serve");
            let mut server = Self {
                child,
                addr,
                _dir: dir,
            };
            server.wait_until_listening();
            server
        }

        fn wait_until_listening(&mut self) {
            let started = Instant::now();
            loop {
                if self.serves_health() {
                    return;
                }
                let exited = self.child.try_wait().expect("child state");
                assert!(exited.is_none(), "server exited during startup: {exited:?}");
                assert!(started.elapsed() < WAIT_LIMIT, "server never started");
                std::thread::sleep(Duration::from_millis(20));
            }
        }

        /// True only when this server answers `/health` with its own
        /// `HTTP/1.1 200` (see `cli_serve.rs` for why a bare TCP connect
        /// does not count).
        fn serves_health(&self) -> bool {
            let Ok(mut stream) = TcpStream::connect(self.addr) else {
                return false;
            };
            if stream
                .set_read_timeout(Some(Duration::from_millis(500)))
                .is_err()
            {
                return false;
            }
            // The host guard under anonymous auth refuses a Host that does
            // not name this listener, so the readiness probe spells the
            // bound address (same rule as `post_json` and `cli_serve.rs`).
            let request = format!(
                "GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n",
                addr = self.addr
            );
            if stream.write_all(request.as_bytes()).is_err() {
                return false;
            }
            let mut reply = String::new();
            stream.read_to_string(&mut reply).is_ok() && reply.starts_with("HTTP/1.1 200")
        }
    }

    impl Drop for RunningServer {
        fn drop(&mut self) {
            if let Err(error) = self.child.kill().and_then(|()| self.child.wait().map(drop)) {
                eprintln!("failed to stop the cli-contract server: {error}");
            }
        }
    }

    /// A reply header as (lower-cased name, value).
    type Header = (String, String);

    /// Blocking HTTP POST of a JSON body through the spawned binary — the
    /// same raw-`TcpStream` client `cli_serve.rs` uses. The host guard
    /// under anonymous auth refuses a Host that does not name this
    /// listener, so the request spells its address.
    fn post_json(bind: SocketAddr, path: &str, body: &str) -> (u16, Vec<Header>, String) {
        let mut stream = TcpStream::connect(bind).expect("connect to the spawned binary");
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .expect("read timeout");
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: {bind}\r\n\
             Content-Type: application/json\r\nContent-Length: {length}\r\n\
             Connection: close\r\n\r\n{body}",
            length = body.len(),
        );
        stream.write_all(request.as_bytes()).expect("write request");
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).expect("read reply");
        let raw = String::from_utf8_lossy(&raw).into_owned();
        let (head, body) = raw
            .split_once("\r\n\r\n")
            .unwrap_or_else(|| panic!("reply has a header block: {raw:.200}"));
        let mut lines = head.lines();
        let status_line = lines.next().unwrap_or_default();
        assert!(
            status_line.starts_with("HTTP/1.1 "),
            "not an HTTP/1.1 reply: {status_line}"
        );
        let status: u16 = status_line
            .strip_prefix("HTTP/1.1 ")
            .and_then(|rest| rest.split(' ').next())
            .and_then(|code| code.parse().ok())
            .unwrap_or_else(|| panic!("status line without a code: {status_line}"));
        let headers: Vec<Header> = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_owned()))
            .collect();
        let chunked = headers.iter().any(|(name, value)| {
            name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked")
        });
        let body = if chunked {
            String::from_utf8_lossy(&decode_chunked(body.as_bytes())).into_owned()
        } else {
            body.to_owned()
        };
        (status, headers, body)
    }

    /// De-chunks a `transfer-encoding: chunked` body so the SSE and JSON
    /// assertions see the payload the server sent.
    fn decode_chunked(raw: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        while let Some(line_end) = find_subslice(raw, b"\r\n", pos) {
            let size_text = std::str::from_utf8(&raw[pos..line_end]).unwrap_or("0");
            let size = size_text
                .trim()
                .split(';')
                .next()
                .and_then(|hex| usize::from_str_radix(hex, 16).ok())
                .unwrap_or(0);
            if size == 0 {
                break;
            }
            let start = line_end + 2;
            let end = start + size;
            if end > raw.len() {
                break;
            }
            out.extend_from_slice(&raw[start..end]);
            pos = end + 2;
        }
        out
    }

    fn find_subslice(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
        haystack[from..]
            .windows(needle.len())
            .position(|window| window == needle)
            .map(|at| at + from)
    }

    /// The binary-level replay of ONE non-streaming contract fixture
    /// (`anthropic-sdk/tool-replay-request.json`): through config
    /// loading, startup wiring and a real listener, the request must
    /// reach the fake upstream equal to the fixture apart from the model
    /// alias, and the client response must parse.
    #[tokio::test(flavor = "multi_thread")]
    async fn contract_fixture_replays_through_the_spawned_binary() {
        // The blocking HTTP posts below run on this thread; the fake
        // upstream needs the runtime's other workers to answer.
        let (upstream_addr, capture) = spawn_fake_upstream(false).await;
        let server = RunningServer::start(upstream_addr);

        let body = fixture_body("anthropic-sdk/tool-replay-request.json");
        let (status, _headers, reply) = post_json(
            server.addr,
            "/providers/fixture-provider/v1/messages",
            &body.to_string(),
        );
        assert_eq!(
            status, 200,
            "fixture replay must succeed, reply: {reply:.400}"
        );

        let upstream_body: Value =
            serde_json::from_str(&captured_request(&capture)).expect("upstream body is JSON");
        assert_eq!(
            upstream_body,
            expected_upstream_body(body, UPSTREAM_MODEL),
            "the request reaching upstream must be the fixture body modulo the model alias"
        );

        let client_response: Value = serde_json::from_str(&reply).expect("client response parses");
        assert_eq!(
            client_response["type"], "message",
            "reply: {client_response}"
        );
        assert_eq!(client_response["content"][0]["text"], "fixture reply");
    }

    /// The binary-level replay of ONE streaming contract fixture
    /// (`captured/anthropic-sdk/stream-request.json`): same rules, with
    /// the upstream answering SSE and the client receiving SSE that
    /// parses event by event.
    #[tokio::test(flavor = "multi_thread")]
    async fn contract_fixture_replays_streaming_through_the_spawned_binary() {
        let (upstream_addr, capture) = spawn_fake_upstream(true).await;
        let server = RunningServer::start(upstream_addr);

        let body = fixture_body("captured/anthropic-sdk/stream-request.json");
        let (status, headers, reply) = post_json(
            server.addr,
            "/providers/fixture-provider/v1/messages",
            &body.to_string(),
        );
        assert_eq!(
            status, 200,
            "stream replay must succeed, reply: {reply:.400}"
        );
        assert!(
            headers.iter().any(|(name, value)| name == "content-type"
                && value.to_ascii_lowercase().contains("text/event-stream")),
            "the client must receive an event stream, headers: {headers:?}"
        );

        let upstream_body: Value =
            serde_json::from_str(&captured_request(&capture)).expect("upstream body is JSON");
        assert_eq!(
            upstream_body,
            expected_upstream_body(body, UPSTREAM_MODEL),
            "the streamed request reaching upstream must be the fixture body modulo the model alias"
        );
        assert_eq!(upstream_body["stream"], true);

        // The client response parses: every `data:` line is JSON and the
        // terminal event arrives.
        let data_events: Vec<Value> = reply
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| {
                serde_json::from_str(data)
                    .unwrap_or_else(|e| panic!("SSE data line must be JSON ({e}): {data:.200}"))
            })
            .collect();
        assert!(
            data_events
                .iter()
                .any(|event| event["type"] == "message_stop"),
            "the terminal message_stop event must reach the client: {reply:.400}"
        );
    }
}
