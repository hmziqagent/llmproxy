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

Three wheels are produced here: the Linux x86_64 manylinux wheel (built and
install-tested), a Windows x64 wheel
(`llm_proxy-0.1.0-cp39-abi3-win_amd64.whl`, mingw-w64 cross-compilation via
`RUSTC_WRAPPER= .venv/bin/maturin build --release --target
x86_64-pc-windows-gnu -o dist`), and a macOS universal2 wheel
(`llm_proxy-0.1.0-cp39-abi3-macosx_10_12_universal2.macosx_10_12_x86_64.macosx_11_0_arm64.whl`,
zig cross-compilation via `RUSTC_WRAPPER= .venv/bin/maturin build --release
--zig --target universal2-apple-darwin -o dist`; zig supplies the macOS libc
stubs and `llvm-lipo` merges the two slices). The Windows wheel is
run-verified with the embeddable Windows CPython 3.13.5 under wine 10.0 —
every shared-contract case, the error shape, streaming, cancellation and the
ClientCancelled settlement pass (see `tests/windows_driver.py`). The macOS
wheel's extension is a fat `x86_64 + arm64` Mach-O whose undefined `Py_*`
symbols resolve from the host interpreter and which links only system
libraries (`libSystem`, `libiconv`, `libcharset`). macOS runs happen in CI:
the `bindings-macos` job builds the universal2 wheel on a macOS runner,
pip-installs it into a fresh venv (no Rust toolchain) and runs the full
pytest suite against the install. The abi3 wheels support CPython 3.9+.

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
