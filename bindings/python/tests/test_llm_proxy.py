"""The shared binding contract, run through the Python package (plan 10,
step 4): the plan-02 contract fixtures from bindings/shared/cases.json replay
through ``llm_proxy`` against a local fake upstream, and the results must
equal the checked-in snapshots the Rust bridge test generated — the same
fixtures, the same results, in a third language. Streaming and cancellation
behave exactly as in Rust.

Run against an installed wheel:

    python -m venv .venv-test && .venv-test/bin/pip install dist/*.whl pytest
    .venv-test/bin/python -m pytest tests -q
"""

import gc
import http.server
import json
import pathlib
import queue
import threading
import time

import pytest

from llm_proxy import Proxy, ProxyError

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
SHARED = REPO_ROOT / "bindings" / "shared"


def load_shared():
    return json.loads((SHARED / "cases.json").read_text())


class CannedUpstream:
    """Serves one canned response (JSON or SSE, with gaps) and captures the
    last request, like the Rust harness's axum fake upstream."""

    def __init__(self, canned, gap_ms=0):
        self.canned = canned
        self.gap = gap_ms / 1000.0
        self.capture = None
        self.lock = threading.Lock()

        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib naming
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = {}
                with upstream.lock:
                    upstream.capture = {
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                if upstream.canned["kind"] == "json":
                    stall = upstream.canned.get("stall_ms") or 0
                    if stall:
                        time.sleep(stall / 1000.0)
                    payload = json.dumps(upstream.canned["body"]).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for chunk in upstream.canned["events"]:
                        self.wfile.write(chunk.encode())
                        self.wfile.flush()
                        if upstream.gap:
                            time.sleep(upstream.gap)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    def captured(self):
        with self.lock:
            return self.capture


def provider_for(shared, name, base_url):
    template = shared["providers"][name]
    return json.loads(json.dumps(template).replace("{upstream}", base_url))


def header_subset(shared, headers):
    return {
        name: headers[name]
        for name in shared["header_subset"]
        if name in headers
    }


def error_info(failure):
    return {
        "kind": failure.kind,
        "status": failure.status,
        "message": failure.message,
        "retry_after_secs": failure.retry_after_seconds,
        "should_retry": failure.should_retry,
    }


def normalize(value, key="created", marker="<created-timestamp>"):
    """The shared normalization: JSON numbers under `created` are wall-clock
    stamps the chat encoder writes."""
    if isinstance(value, dict):
        return {
            k: (marker if k == key and isinstance(v, (int, float)) else normalize(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def make_proxy(shared, providers, record_usage=False):
    config = {"providers": providers, "record_usage": record_usage}
    return Proxy(config)


@pytest.mark.parametrize("case", load_shared()["cases"], ids=lambda c: c["name"])
def test_case_matches_the_shared_snapshot(case, shared=load_shared()):
    canned = shared["upstream_responses"][case["upstream_response"]]
    upstream = CannedUpstream(canned)
    provider = provider_for(shared, case["provider"], upstream.base_url)
    proxy = make_proxy(shared, [provider])

    fixture = (REPO_ROOT / shared["fixture_root"] / case["fixture"]).read_text()
    body = json.loads(fixture)

    error = None
    response = None
    events = None
    if case["stream"]:
        try:
            events = list(
                proxy.execute_stream(
                    case["provider"], case["dialect"], body, case.get("options")
                )
            )
        except ProxyError as failure:
            error = error_info(failure)
    else:
        try:
            response = proxy.execute(
                case["provider"], case["dialect"], body, case.get("options")
            )
        except ProxyError as failure:
            error = error_info(failure)

    captured = upstream.captured()
    snapshot = {
        "upstream_request": captured["body"] if captured else None,
        "upstream_headers": header_subset(shared, captured["headers"]) if captured else None,
        "response": response,
        "stream_events": events,
        "error": error,
    }
    expected = json.loads((SHARED / "snapshots" / ("%s.json" % case["name"])).read_text())
    assert normalize(snapshot) == expected


@pytest.mark.parametrize("case", load_shared()["translates"], ids=lambda c: c["name"])
def test_translate_matches_the_shared_snapshot(case, shared=load_shared()):
    config = {
        "providers": [
            {
                "name": "offline",
                "api_key": "k",
                "auth_style": "bearer",
                "adapters": {
                    "a": {
                        "protocol": "anthropic_messages",
                        "endpoint": "http://127.0.0.1:9/v1/messages",
                    }
                },
                "routes": {"messages": "a"},
            }
        ]
    }
    proxy = Proxy(config)
    if "fixture" in case:
        body = json.loads((REPO_ROOT / shared["fixture_root"] / case["fixture"]).read_text())
    else:
        body = case["body"]

    error = None
    try:
        if case["mode"] == "response":
            result = proxy.translate_response(body, case["from"], case["to"])
        else:
            result = proxy.translate_request(body, case["from"], case["to"])
    except ProxyError as failure:
        result = None
        error = error_info(failure)

    expected = json.loads((SHARED / "snapshots" / ("%s.json" % case["name"])).read_text())
    assert normalize({"result": result, "error": error}) == expected


def test_failures_raise_proxy_error_with_status_and_kind(shared=load_shared()):
    upstream = CannedUpstream(shared["upstream_responses"]["anthropic_json"])
    provider = provider_for(shared, "anthropic-upstream", upstream.base_url)
    proxy = make_proxy(shared, [provider])
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }

    with pytest.raises(ProxyError) as unknown:
        proxy.execute("nope", "anthropic_messages", body)
    assert unknown.value.kind == "UnknownProvider"
    assert unknown.value.status == 404
    assert unknown.value.should_retry is False

    with pytest.raises(ProxyError) as bad_body:
        proxy.execute("anthropic-upstream", "anthropic_messages", {"no": "model"})
    assert bad_body.value.status == 400
    assert bad_body.value.kind == "InvalidRequest"


def test_request_timeout_is_enforced(shared=load_shared()):
    """``request_timeout_secs`` bounds the whole request: an upstream that
    accepts the connection and then stalls for 30 s must surface as a
    ProxyError within the budget, not wait out the stall."""
    upstream = CannedUpstream(
        {
            "kind": "json",
            "stall_ms": 30_000,
            "body": shared["upstream_responses"]["anthropic_json"]["body"],
        }
    )
    provider = provider_for(shared, "anthropic-upstream", upstream.base_url)
    proxy = Proxy(
        {"providers": [provider], "record_usage": False, "request_timeout_secs": 1}
    )
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }

    started = time.monotonic()
    with pytest.raises(ProxyError) as stalled:
        proxy.execute("anthropic-upstream", "anthropic_messages", body)
    assert time.monotonic() - started < 5, "the whole-request deadline ended the stall"
    assert stalled.value.kind == "UpstreamTimeout"
    assert stalled.value.status == 504
    assert stalled.value.should_retry is True


def test_cancellation_stops_delivery_and_settles_client_cancelled(shared=load_shared()):
    scenario = shared["cancellation"]
    gap_ms = scenario.get("gap_ms", 250)
    settle_timeout = scenario.get("settle_timeout_secs", 5)
    sse = CannedUpstream(shared["upstream_responses"][scenario["upstream_response"]], gap_ms)
    json_upstream = CannedUpstream(shared["upstream_responses"]["anthropic_json"])

    proxy = make_proxy(
        shared,
        [
            provider_for(shared, scenario["stream_provider"], sse.base_url),
            provider_for(shared, scenario["followup_provider"], json_upstream.base_url),
        ],
        record_usage=True,
    )

    stream = proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], scenario["request"]
    )
    first = next(iter(stream))
    assert "MessageStart" in first

    stream.cancel()
    assert stream.cancelled
    with pytest.raises(StopIteration):
        next(iter(stream))

    deadline = time.monotonic() + settle_timeout
    while time.monotonic() < deadline:
        rows = proxy.usage_events()
        if any(
            row.get("kind") == "response_failed"
            and row.get("error_kind") == "ClientCancelled"
            for row in rows
        ):
            break
        time.sleep(0.05)
    else:
        pytest.fail("the ClientCancelled row must land after cancel: %r" % proxy.usage_events())

    followup = proxy.execute(
        scenario["followup_provider"],
        scenario["dialect"],
        {"model": "claude-sonnet-4-5", "max_tokens": 64,
         "messages": [{"role": "user", "content": "again"}]},
    )
    assert "id" in followup


def test_cancel_from_another_thread_unblocks_a_blocked_next(shared=load_shared()):
    """The README documents GIL-releasing waits, so a reader thread can be
    blocked in next() while the main thread cancels: cancel must not raise
    'Already borrowed', the blocked reader must be released, and the request
    must settle as ClientCancelled."""
    scenario = shared["cancellation"]
    # One SSE event, then nothing for an hour: the reader parks on a stalled
    # upstream.
    sse = CannedUpstream(
        shared["upstream_responses"][scenario["upstream_response"]], 3_600_000
    )
    json_upstream = CannedUpstream(shared["upstream_responses"]["anthropic_json"])

    proxy = make_proxy(
        shared,
        [
            provider_for(shared, scenario["stream_provider"], sse.base_url),
            provider_for(shared, scenario["followup_provider"], json_upstream.base_url),
        ],
        record_usage=True,
    )
    stream = proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], scenario["request"]
    )

    events = queue.Queue()

    def read_until_end():
        for event in stream:
            events.put(event)
        events.put(None)

    reader = threading.Thread(target=read_until_end, daemon=True)
    reader.start()

    first = events.get(timeout=3)
    assert "MessageStart" in first
    with pytest.raises(queue.Empty):
        # the reader is parked on the stalled upstream
        events.get(timeout=0.3)

    stream.cancel()
    assert stream.cancelled, "the getter is readable while next() is blocked"
    assert events.get(timeout=5) is None, "cancel unblocks the parked reader"
    reader.join(timeout=5)
    assert not reader.is_alive()

    deadline = time.monotonic() + scenario.get("settle_timeout_secs", 5)
    while time.monotonic() < deadline:
        rows = proxy.usage_events()
        if any(
            row.get("kind") == "response_failed"
            and row.get("error_kind") == "ClientCancelled"
            for row in rows
        ):
            break
        time.sleep(0.05)
    else:
        pytest.fail("the ClientCancelled row must land after cancel: %r" % proxy.usage_events())

    followup = proxy.execute(
        scenario["followup_provider"],
        scenario["dialect"],
        {"model": "claude-sonnet-4-5", "max_tokens": 64,
         "messages": [{"role": "user", "content": "again"}]},
    )
    assert "id" in followup


def stalled_stream_proxy(shared):
    """A recording proxy against a stalled SSE upstream (one event, then
    nothing for an hour) plus a fast JSON followup upstream."""
    scenario = shared["cancellation"]
    sse = CannedUpstream(
        shared["upstream_responses"][scenario["upstream_response"]], 3_600_000
    )
    json_upstream = CannedUpstream(shared["upstream_responses"]["anthropic_json"])
    proxy = make_proxy(
        shared,
        [
            provider_for(shared, scenario["stream_provider"], sse.base_url),
            provider_for(shared, scenario["followup_provider"], json_upstream.base_url),
        ],
        record_usage=True,
    )
    return proxy, scenario


def wait_for_client_cancelled(proxy, timeout, why):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = proxy.usage_events()
        if any(
            row.get("kind") == "response_failed"
            and row.get("error_kind") == "ClientCancelled"
            for row in rows
        ):
            return
        time.sleep(0.05)
    pytest.fail("%s: %r" % (why, proxy.usage_events()))


def test_leaving_a_with_block_cancels(shared=load_shared()):
    proxy, scenario = stalled_stream_proxy(shared)

    with proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], scenario["request"]
    ) as stream:
        first = next(iter(stream))
        assert "MessageStart" in first

    assert stream.cancelled, "leaving the with block must cancel the stream"
    wait_for_client_cancelled(
        proxy,
        scenario.get("settle_timeout_secs", 5),
        "the ClientCancelled row must land after the with block",
    )


def test_breaking_out_of_a_loop_over_a_temporary_stream_cancels(shared=load_shared()):
    proxy, scenario = stalled_stream_proxy(shared)

    for _ in proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], scenario["request"]
    ):
        break

    gc.collect()
    wait_for_client_cancelled(
        proxy,
        scenario.get("settle_timeout_secs", 5),
        "the ClientCancelled row must land after the temporary stream is released",
    )


def test_deleting_the_stream_cancels(shared=load_shared()):
    proxy, scenario = stalled_stream_proxy(shared)
    stream = proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], scenario["request"]
    )
    first = next(iter(stream))
    assert "MessageStart" in first

    del stream
    gc.collect()
    wait_for_client_cancelled(
        proxy,
        scenario.get("settle_timeout_secs", 5),
        "the ClientCancelled row must land after the stream is deleted",
    )


def test_a_fully_consumed_stream_that_is_released_stays_completed(shared=load_shared()):
    def completed_row_lands(proxy, timeout, why):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rows = proxy.usage_events()
            if any(row.get("kind") == "response_completed" for row in rows):
                return rows
            time.sleep(0.05)
        pytest.fail("%s: %r" % (why, proxy.usage_events()))

    scenario = shared["cancellation"]
    body = scenario["request"]

    # Drained to the end (StopIteration), then released.
    fast = CannedUpstream(shared["upstream_responses"][scenario["upstream_response"]])
    proxy = make_proxy(
        shared,
        [provider_for(shared, scenario["stream_provider"], fast.base_url)],
        record_usage=True,
    )
    with proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], body
    ) as stream:
        with pytest.raises(StopIteration):
            while True:
                next(iter(stream))
    rows = completed_row_lands(
        proxy,
        scenario.get("settle_timeout_secs", 5),
        "the completed row must land after a drained stream is released",
    )
    assert not any(row.get("kind") == "response_failed" for row in rows), rows
    assert not stream.cancelled

    # Released right after the terminal event, without reading to the end.
    fast = CannedUpstream(shared["upstream_responses"][scenario["upstream_response"]])
    proxy = make_proxy(
        shared,
        [provider_for(shared, scenario["stream_provider"], fast.base_url)],
        record_usage=True,
    )
    for event in proxy.execute_stream(
        scenario["stream_provider"], scenario["dialect"], body
    ):
        if "MessageStop" in event:
            break
    gc.collect()
    rows = completed_row_lands(
        proxy,
        scenario.get("settle_timeout_secs", 5),
        "the completed row must land after a release at the terminal event",
    )
    assert not any(row.get("kind") == "response_failed" for row in rows), rows


def test_usage_events_record_the_lifecycle_when_opted_in(shared=load_shared()):
    upstream = CannedUpstream(shared["upstream_responses"]["anthropic_json"])
    provider = provider_for(shared, "anthropic-upstream", upstream.base_url)
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }

    recording = make_proxy(shared, [provider], record_usage=True)
    recording.execute("anthropic-upstream", "anthropic_messages", body)
    kinds = [row.get("kind") for row in recording.usage_events()]
    assert "request_received" in kinds
    assert "response_completed" in kinds

    silent = make_proxy(shared, [provider], record_usage=False)
    silent.execute("anthropic-upstream", "anthropic_messages", body)
    assert silent.usage_events() == []
