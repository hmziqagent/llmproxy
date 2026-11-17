# llm-proxy (Python)

Python bindings for the llm-proxy engine: translation between wire dialects,
request execution through the full pipeline, streaming events with
cancellation, and configuration from code. The extension wraps the shared
bridge crate (`crates/llm-proxy-bridge`) and adds no logic of its own.

## Install

Prebuilt wheels install with no Rust toolchain:

```
pip install llm_proxy-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl
```

Every wheel is built in CI by the `python-wheels` workflow
(`.github/workflows/python-wheels.yml`): Linux x86_64 manylinux and
musllinux_1_2, Linux aarch64, Windows x64 and macOS universal2, plus an
sdist. Each OS wheel is pip-installed into a fresh venv with no Rust
toolchain and the full pytest suite runs against the install on Linux,
Windows and macOS for CPython 3.9 and 3.13; a final clean-install job
removes every Rust toolchain from the runner and installs the wheel with
`--no-index --only-binary=:all:` so nothing can compile. The wheels are
abi3 and support CPython >= 3.9.

## Use

```python
from llm_proxy import Proxy, ProxyError

proxy = Proxy({
    "providers": [
        {
            "name": "acme",
            "api_key": "sk-...",
            "auth_style": "bearer",
            "adapters": {
                "messages": {
                    "protocol": "anthropic_messages",
                    "endpoint": "https://api.example.com/v1/messages",
                }
            },
            "routes": {"messages": "messages"},
            "model_aliases": {"claude-sonnet-4-5": "claude-sonnet-4-5-20250929"},
        }
    ],
    "record_usage": True,
})

reply = proxy.execute("acme", "anthropic_messages", request_body)

with proxy.execute_stream("acme", "anthropic_messages", stream_body) as stream:
    for event in stream:
        ...  # event dicts; leaving the with block cancels the upstream

translated = proxy.translate_request(body, "anthropic_messages", "openai_chat_completions")
```

A stream stops when you leave its `with` block, call `stream.cancel()`
(or `stream.close()`), or release its last reference; breaking out of a
loop over a stream you still hold keeps the request running until the
stream is released. A stream that has run to its end is not cancelled on
release: its request settles as completed.

Dialect names: `anthropic_messages`, `openai_chat_completions`,
`openai_responses`, `gemini_generate_content`, `ollama_chat`,
`ollama_generate`. Provider protocols: `anthropic_messages`,
`openai_chat_completions`, `openai_responses`, `gemini_generate_content`.

`execute` blocks and releases the GIL while waiting. Failures raise
`ProxyError` with `kind`, `status`, `message`, `retry_after_seconds` and
`should_retry`.

recorded usage rows are kept in memory for the life of the process; drain
with `usage_events()` periodically, or leave `record_usage` off in
long-running services.

## Build and test

```
python -m venv .venv && .venv/bin/pip install maturin pytest
.venv/bin/maturin build --release -o dist
python -m venv .venv-test && .venv-test/bin/pip install dist/*.whl pytest
.venv-test/bin/python -m pytest tests -q
```

The test suite replays the plan-02 contract fixtures from
`bindings/shared/cases.json` and compares against the snapshots the Rust
bridge test generates: the same fixtures must give the same results in Rust
and Python.
