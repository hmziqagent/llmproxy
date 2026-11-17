# llm-proxy

A provider-scoped LLM proxy with OpenAI Chat Completions and Anthropic
Messages client endpoints, normalized protocol translation, streaming, model
catalogs, and optional live model discovery.

## Layout

~~~
apps/llm-proxy                 daemon and CLI (binary: llm-proxy)
crates/llm-proxy-core          config, routing, registry, metrics
crates/llm-proxy-protocol      OpenAI / Claude / Gemini wire types
crates/llm-proxy-adapters      provider codecs (no I/O)
crates/llm-proxy-transport     HTTP egress, credentials, discovery
crates/llm-proxy-engine        request lifecycle (no HTTP) + traits
crates/llm-proxy-bridge        JSON surface the Python/Node bindings wrap
crates/llm-proxy-storage       usage ledger
crates/llm-proxy-api           read-only /usage API
crates/llm-proxy-server        axum routes and error mapping
crates/llm-proxy-sdk           facade crate (published as llm-proxy-sdk)

# Optional crates (plan 09) — not in the default build; enable with the
# daemon feature of the same name:
crates/llm-proxy-store-redis       [replica] shared limits/budgets in Redis
crates/llm-proxy-store-postgres    [replica] shared limits/budgets in Postgres
crates/llm-proxy-admin             admin surface: keys, usage, config, UI, MCP tools
crates/llm-proxy-semantic-cache    [server.semantic_cache]
crates/llm-proxy-mcp               MCP gateway ([mcp], own tokens)
# No daemon feature: embedders wire guards through
# EngineBuilder::guard (llm-proxy-guardrails never becomes a daemon
# dependency).
crates/llm-proxy-guardrails        guards with modes and fail policies

# Language bindings (plan 10) — standalone workspaces packaged for PyPI/npm:
bindings/python                 llm-proxy wheel (pyo3 + maturin)
bindings/node                   @freeoxide/llm-proxy (napi-rs)
bindings/shared                 binding contract cases and snapshots
~~~

A bare `cargo build` compiles none of them (they sit outside
`default-members`); `cargo build --workspace` builds everything.

The non-chat endpoint families (embeddings, rerank, audio, images, files
and batches) are a daemon feature too: build with
`--features endpoint-families` and map them in `[provider.routes]`
(`embeddings`, `rerank`, `audio_transcriptions`, `audio_speech`,
`images_generations`, `files`, `batches`). A provider that routes a
family in a build without the feature refuses to start.

Live integration suites for the replica stores, the MCP gateway and the
endpoint families ship `#[ignore]`d so the default test run needs no
servers; set `LLM_PROXY_TEST_REDIS`, `LLM_PROXY_TEST_POSTGRES`,
`LLM_PROXY_TEST_MCP_UPSTREAM` (plus `LLM_PROXY_TEST_MCP_CLIENT` to also
drive the gateway from the official MCP Inspector CLI) or the
`LLM_PROXY_TEST_*_URL` variables listed in
`crates/llm-proxy-server/tests/live_families.rs`, and run the matching
crate's tests with `-- --ignored`.

## Build

~~~
cargo build --release
~~~

## Run

~~~
./target/release/llm-proxy validate --config ./config.toml
./target/release/llm-proxy serve --config ./config.toml
~~~

Every API request names its provider in the path:

~~~text
POST /providers/{provider}/v1/chat/completions
POST /providers/{provider}/v1/messages
POST /providers/{provider}/v1/messages/count_tokens
GET  /providers/{provider}/v1/models
~~~

See `config.toml.example` and the files under `providers/` for configuration
examples.

See the [documentation index](docs/README.md) for the project goal, plans,
protocol design and research.

## Admin surface

Keys, config view, UI and MCP tools are served only on the dedicated admin
listener; `admin_on_main_listener = true` exposes the read-only usage API only.

## References

These shaped the layout and scope:

- LeenHawk/gproxy
- modpotatodotdev/LLMG
- x5iu/openproxy
- habibi-dev/rust-llm-proxy
