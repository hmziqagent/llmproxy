# 09. Optional crates

- **Status:** Completed (2026-11-07)
- **Depends on:** 05; the semantic cache also needs 06
- **Audit rows:** None
- **Code-audit bugs:** None
- **Backlog:** 35
- **Conflicts:** 7, 9, 13, 16, 17

Build the large features as opt-in crates behind cargo features. Each uses the engine's
traits for admission and accounting instead of its own copy. Take them in this order
unless real use says otherwise.

## To do

1. **Replica stores (conflict 13).** `llm-proxy-store-postgres` and
   `llm-proxy-store-redis` hold limits, cooldowns, budgets and usage across instances.
   Multi-tenant hosting is assessed only after this works.
2. **Admin API and web UI.** `llm-proxy-admin` serves key management, usage and config
   views on the dedicated admin listener.
3. **Admin MCP tools (backlog 35).** Expose the admin API as MCP tools, reusing its auth.
4. **Guardrails (conflict 9).** Each guard declares a mode (pre-call, buffered or
   post-hoc) and whether it fails open or closed. Masking on streams needs placeholder
   handling across chunks, so a guard that can't do this rejects instead.
5. **Semantic cache (conflict 7).** Off by default, with caller-scoped keys and the data
   policy from plan 06.
6. **MCP gateway (conflict 17).** `llm-proxy-mcp` has its own auth and session model, and
   shares admission and accounting with the proxy.
7. **Non-chat endpoints (conflict 16).** Embeddings, rerank, audio, images, files and
   batch, each as a family-specific handler using the plan 03 engine. See
   [`endpoints.md`](../research/endpoints.md) for per-family notes.

## Done when

- The default build compiles none of these crates.
- Each crate can be enabled alone, and the workspace tests pass with all of them on and
  with all of them off.
- Two instances sharing the replica stores enforce one budget between them.

## Done

- The workspace has `default-members` (the pre-plan-09 crates) plus the
  optional members `llm-proxy-store-redis`, `llm-proxy-store-postgres`,
  `llm-proxy-admin`, `llm-proxy-guardrails`, `llm-proxy-semantic-cache`
  and `llm-proxy-mcp`: a bare `cargo build` compiles none of them
  (verified: `cargo tree` over the default build graph shows zero
  optional crates), each builds alone, and `cargo test --workspace
  --all-features` / `cargo test --workspace` both pass.
- The admission counters moved behind one engine trait
  (`AdmissionBackend`), the cooldown view behind `Cooldowns` (now
  wall-clock), so the replica stores implement engine seams instead of
  their own copy of the rules. Redis: every counter move is one atomic
  `INCR`, refusals roll back; unsettled leases are reaped past their
  deadline (a deadline index plus a 60 s reaper on every replica,
  exactly one release per lease); two instances over one store enforce
  one budget between them, concurrently, and a refund on one releases
  budget for the other.
  Settlement charges spend before releasing the reservation (settle-vs-
  admit is ordering, not a transaction), and one lease settles its
  counters exactly once.
  Postgres: one transaction per admission (windows, caps and the lease
  commit together or not at all), deadpool-managed pool, a purge timer;
  the same done-when holds. Both suites also ran against the real
  servers (see the live-server verification paragraph below) and are
  `#[ignore]`d behind their env vars. Only Redis has an in-repo loopback
  fake, so the default test run exercises Redis; the Postgres store's
  tests all need `LLM_PROXY_TEST_POSTGRES` and never run by default.
  `[replica]` wires either store into the daemon
  and refuses to start when the section names a backend whose feature is
  off.
- `llm-proxy-admin` (steps 2-3, backlog 35) serves on the dedicated
  admin listener behind `[auth.admin]`: managed client keys (issue, list,
  revoke; argon2id verifiers, the token shown once, merged into
  `ClientAuth` at startup behind the daemon's `admin-ui` feature), the
  sanitized running config, the read-only usage views, and an embedded
  web UI over the same JSON API. `POST /mcp` exposes the admin API as
  MCP tools under protocol revision 2026-07-28 — single-JSON responses,
  202 for notifications, `-32020` header validation, 404/`-32601` for
  unknown methods, `initialize` answered for older-era clients — behind
  the same admin token.
- `llm-proxy-guardrails` (step 4, conflict 9): every guard declares a
  mode (pre-call, buffered, post-hoc) and a fail policy (open or
  closed); the engine's `Guard` trait gained in-place
  `transform_request`/`transform_response` hooks the wrapper masks
  through. A masking guard that cannot keep placeholders consistent
  across stream chunks rejects streaming requests instead of masking
  them unsafely — the plan's rule — and when a guard rewrites a request
  the engine refuses the same-format passthrough shortcut, so the
  redacted core, not the raw client bytes, reaches the upstream.
- `llm-proxy-semantic-cache` (step 5, conflict 7) implements the
  engine's `Cache` trait: caller-scoped buckets, prompts embedded
  through an `Embedder` trait (default: the offline deterministic
  trigram embedder), threshold-gated cosine matching, TTL and per-caller
  bounds, complete non-streaming responses only. The engine's `CacheKey`
  now carries the normalized request so the cache embeds the prompt
  instead of the body hash. Wired from `[server.semantic_cache]` behind
  the `semantic-cache` feature, off by default; the data-class policy
  page covers what it stores.
- `llm-proxy-mcp` (step 6, conflict 17): MCP clients reach configured
  upstream MCP servers through the proxy on its own listener with its
  own token list and a per-token session table aged out by TTL. Every
  `tools/call` reserves and settles through the engine's shared
  `Admission` step and writes the same usage rows (route kind `mcp`),
  refused calls included. Upstream calls carry the revision's required
  headers. The daemon binds it from `[mcp]` behind `mcp-gateway`;
  tests run against in-repo fake upstream MCP servers.
- Non-chat families (step 7, conflict 16) behind the server's
  `endpoint-families` feature (also exposed by the daemon and the SDK;
  a provider that routes a family in a build without it refuses to
  start): embeddings, rerank, audio transcriptions/speech, images, files
  and batch, each a family-specific handler (new `ProviderRouteKind`
  variants, per-family `[server.endpoint_families]` limits) around the
  generic spine — the same registry routing, admission, credential
  choice (including `passthrough_auth`), `[provider.egress]` transport
  and ledger as chat, never forced through the chat core. JSON families
  (speech included) get the one surgical edit (model aliasing, unknown
  fields preserved); multipart uploads pass through verbatim with the
  client's content type. The client's method and query reach the
  upstream (`GET`/`DELETE` on files, list/retrieve/cancel on batches); a
  resource path must be plain identifier segments, so nothing can resolve
  outside the family's endpoint; answers keep the upstream's status,
  `Content-Type` and `Content-Disposition`. Failures follow the one
  client-status policy, plus one stated rule: an upstream `404` on a
  request that names a resource is that resource's absence (`404`), not
  an upstream fault. Family calls use the provider's base key; extra
  pool keys are not rotated on these routes.
- Live-server verification: a Docker daemon was available, so both
  replica stores ran against the real servers — Redis 7
  (`LLM_PROXY_TEST_REDIS`, `tests/live_server.rs`: two instances enforce
  one budget between them concurrently, a refund crosses replicas,
  windows and cooldowns are shared, usage events reach the stream) and
  Postgres 16 (`LLM_PROXY_TEST_POSTGRES`, `tests/replica_state.rs`: the
  same done-when). The real Postgres immediately exposed a concurrency
  bug the in-repo fakes could not — parallel `CREATE TABLE IF NOT
  EXISTS` races in `pg_type` (SQLSTATE 23505) — fixed by serializing
  schema creation behind a transaction-scoped advisory lock
  (`pg_advisory_xact_lock`), verified by three consecutive concurrent
  clean runs. Both suites are `#[ignore]`d without their env vars; the
  default test run still uses the in-repo loopback fakes.
- Live MCP verification (both directions), against the official
  reference implementations run locally: the reference "everything"
  server (`@modelcontextprotocol/server-everything` 2.2.1, Streamable
  HTTP) as the gateway's upstream (`LLM_PROXY_TEST_MCP_UPSTREAM`,
  `tests/live_upstream.rs`: tools aggregated and namespaced, a real
  echo tool called end to end, usage rows written), and the official
  MCP Inspector CLI (`@modelcontextprotocol/inspector` 2.14.0) as a
  real client against the gateway's own `POST /mcp`
  (`LLM_PROXY_TEST_MCP_CLIENT`, `tests/live_client.rs`: the inspector
  negotiates initialize, lists the gateway's namespaced tools and calls
  one through to the reference server). Both live runs found real bugs
  the in-repo fakes could not, both fixed with red-green regression
  tests: the gateway's upstream requests were missing the revision's
  per-request `_meta` envelope (the reference server rejects with
  -32602), and both MCP endpoints answered `initialize` with
  2026-07-28 regardless of the version the client negotiated — an
  initialization-era client (like the Inspector) rejects such a server,
  so the negotiated revision is now echoed, exactly as the reference
  server does. The admin `/mcp` endpoint gained the same negotiation.
- Live family verification, against real implementations of each
  family's wire API run locally: llama.cpp `llama-server` (embeddings,
  rerank), speaches (Kokoro speech, faster-whisper transcription),
  stable-diffusion.cpp `sd-server` (images) and Llama Stack 0.7.3
  (`inline::localfs` files, `inline::reference` batches answered by a
  llama.cpp chat model). `tests/live_families.rs` runs every family
  through the router (`LLM_PROXY_TEST_EMBEDDINGS_URL`,
  `LLM_PROXY_TEST_RERANK_URL`, `LLM_PROXY_TEST_AUDIO_URL`,
  `LLM_PROXY_TEST_IMAGES_URL`, `LLM_PROXY_TEST_FILES_URL`; `#[ignore]`d
  without them), and the daemon binary was driven end to end by the
  official OpenAI Python SDK (every family, including the whole
  upload/list/download/batch/delete lifecycle) and the Cohere SDK
  (rerank). The live runs found bugs the loopback fakes could not, each
  fixed with a regression test that failed first: files and batches
  served only `POST`, so their lifecycle was unreachable; speech bodies
  were not aliased and its audio was relabelled `application/json`;
  every upstream failure became a `502` carrying the upstream's error
  body; a resource path such as `..%2F..%2Fchat%2Fcompletions` reached
  other upstream paths on the operator's key; family calls ignored
  `[provider.egress]` and `passthrough_auth`; a caller content type was
  sent twice; the daemon had no way to enable the families; and
  `llm-proxy validate` listed only chat and messages routes.
- Not verified: hosted providers' own endpoints (OpenAI, Cohere and the
  like). This environment has no upstream API keys, so the families were
  verified against the self-hosted servers above, which implement the
  same documented wire APIs.
