# 03. Engine and crate split

- **Status:** Completed (2026-11-05)
- **Depends on:** 02
- **Audit rows:** None
- **Code-audit bugs:** BUG-23
- **Backlog:** None
- **Conflicts:** 13, 16

Make one request engine that the server, the SDK and the bindings all share. This
comes before features so they don't get built into the axum layer, where SDK users
can't reach them.

## Target crates

```text
llm-proxy-protocol     wire types and client codecs; no I/O
llm-proxy-adapters     provider codecs; no I/O (split from llm-proxy-provider)
llm-proxy-transport    HTTP, credentials and egress
llm-proxy-engine       request lifecycle without HTTP, plus the traits
llm-proxy-store-*      optional: sqlite, postgres, redis
llm-proxy-server       axum binding: routes and error mapping only
llm-proxy-admin        optional: admin API and web UI
llm-proxy-mcp          optional
llm-proxy              facade crate with cargo features
apps/llm-proxy         daemon and CLI
```

The facade name is taken by the binary's package today. Rename that package and
keep the binary named `llm-proxy`. Confirm each new name is free on crates.io.

## Pipeline slots

Every feature takes one of these places, in this order:

1. Authenticate the caller.
2. Apply caller limits and key allowlists.
3. Resolve the route to a provider or pool.
4. Claim the idempotency key.
5. Look up the cache and join any single-flight request.
6. Run preflight and reserve budget.
7. Run pre-call guards.
8. Apply request mutations in their fixed phases.
9. Run the attempt loop: pick a member and credential, encode, send, retry or
   fail over.
10. Hold the stream until it can be committed to the client.
11. Run streaming and post-call guards.
12. Encode for the client.
13. Settle budget, write usage rows and store cache and idempotency results.

## To do

1. **Split adapters from transport** so pure encoding doesn't need `reqwest`.
2. **Extract the engine.** Move the lifecycle out of
   `crates/llm-proxy-server/src/routes/core_pipeline.rs`. It takes core types and
   returns a `CoreResponse` or a stream of `CoreEvent`, with no axum types.
3. **Add traits with local defaults (conflict 13):** `Router`, `Admission`, `Cache`, `Guard`,
   `CredentialSource`, `UsageSink` and `ConfigSource`. `EventBus` and `Tokenizer`
   already exist.
4. **Open the provider set.** Replace the closed `ProviderProtocol` and
   `ProviderAdapter` enums with registration, so an SDK user can add a provider.
5. **Support config from code.** Add a builder, and make TOML `AppConfig` one
   `ConfigSource`.
6. **Shrink the server.** It keeps routes and error mapping, and `build_router`
   stays public behind a small, stable API.
7. **Clean up boundaries.** Move admin reads off direct SQLite access in
   `crates/llm-proxy-api/src/reader.rs`, and move `crates/llm-proxy-core/src/pid.rs`
   to the app.
8. **Support endpoint families (conflict 16).** Keep admission, routing and
   accounting generic around family-specific handlers. `MAX_BODY_BYTES` and the
   `TimeoutLayer` become per-family settings.
9. **Add cargo features.** The server builds without `rusqlite`, the core without
   `tiktoken-rs`, and `reqwest` uses one TLS stack (BUG-23).
10. **Make it publishable.** Remove the exact `time` pin from `Cargo.toml` and add
    versions to the 14 path dependencies.

## Done when

- A Rust program runs the full pipeline against a fake upstream without binding a
  port.
- An axum app mounts the proxy router in a test.
- `llm-proxy-protocol` and `llm-proxy-adapters` build with no network dependencies.
- The contract tests from plan 02 pass unchanged through the real router
  in-process; two fixtures additionally through the spawned binary.
- `cargo publish --dry-run` passes for every published crate.

## Done

- **Crate split (steps 1, 10).** `llm-proxy-provider` is gone:
  `llm-proxy-adapters` (provider codecs, pure encoding) and
  `llm-proxy-transport` (HTTP egress, credentials, discovery) replace it;
  `llm-proxy-engine` holds the extracted lifecycle; `llm-proxy-sdk` is the
  facade (published as `llm-proxy-sdk`, not `llm-proxy`, because crates.io
  `llm-proxy` is an unrelated third-party placeholder from 2023 —
  crates.io does not reclaim placeholder names, so a rename back is not
  available). The app package is `llm-proxy-app`; the binary stays
  `llm-proxy`.
- **Engine (steps 2, 3).** `crates/llm-proxy-engine` runs the pipeline
  (admission → routing → cache → guards → attempt → settlement) with no
  axum types: it takes core types and returns a `CoreResponse` (plus the
  same-format passthrough body) or a stream of `CoreEvent`. Traits with
  local defaults: `Router`, `Admission`, `Cache`, `Guard`,
  `CredentialSource`, `UsageSink`, `ConfigSource` (`EventBus` and
  `Tokenizer` already existed).
- **Open provider set (step 4).** The closed `ProviderProtocol` and
  `ProviderAdapter` enums are replaced by `ProtocolName` plus trait-object
  registration (`ProviderAdapterRegistry::register`); a custom protocol
  implemented against the public traits routes through the full pipeline
  (`crates/llm-proxy-engine/tests/open_provider_set.rs`).
- **Config from code (step 5).** `EngineBuilder` and
  `llm_proxy_sdk::ProxyBuilder` assemble the stack without TOML; a loaded
  TOML `AppConfig` is one `ConfigSource` (`StaticConfigSource`).
- **Server (step 6).** `llm-proxy-server` keeps routes, middleware and
  error mapping; `build_router(AppState)` is unchanged. `AppState`
  materializes the engine on first use.
- **Boundaries (step 7).** Admin reads go through
  `llm-proxy-storage::UsageReader` (the api crate no longer opens SQLite);
  `PidManager` moved to the app (`apps/llm-proxy/src/pid/`).
- **Endpoint families (step 8).** `[server.endpoint_families.<name>]`
  sets `max_body_bytes` and `request_timeout` per family; the router layers
  them per family (`crates/llm-proxy-server/tests/endpoint_families.rs`).
- **Features (step 9).** The server builds without `rusqlite`
  (`--no-default-features`; the admin API is behind `admin`), the core
  without `tiktoken-rs` (`--no-default-features`), and reqwest uses
  rustls only — no native-tls/openssl anywhere in the transport tree
  (BUG-23). `tiktoken-rs` is consumer-visible opt-in: the workspace
  `llm-proxy-core` dependency keeps its defaults off, and
  `llm-proxy-engine`'s default `tiktoken` feature (forwarded by
  `llm-proxy-server`, `llm-proxy-sdk` and the app) is the only thing
  enabling it, so `llm-proxy-adapters`, `-transport` and `-protocol`
  never compile it.
- **Publishable (step 10).** The `time = "=0.3.44"` pin is gone; all path
  dependencies carry `version = "0.1.0"`. Every crate passes
  `cargo publish --dry-run` (full package + verification build + upload
  prep). Against live crates.io that is only possible in dependency order —
  cargo requires a crate's registry dependencies to already exist — so the
  two leaf crates (`llm-proxy-protocol`, `llm-proxy-core`) were dry-run
  against crates.io directly, and the full set was verified against a local
  registry reproducing crates.io's post-publication state: all 413 external
  dependencies vendored from the lockfile (real `.crate` files, real
  checksums, `cargo-local-registry` 0.2.12) plus each workspace crate's
  actual packaged artifact indexed in order
  (protocol → core → adapters → transport → storage → engine → api →
  server → sdk, then the `llm-proxy-app` binary package). The real
  crates.io dry-runs for the dependent crates become passable the moment
  the leaves are actually uploaded — a release action, not a code change.

**Done-when verification:** a Rust program runs the full pipeline against a
loopback fake upstream with the proxy binding no port
(`crates/llm-proxy-engine/tests/fake_upstream.rs`); an axum app mounts
`build_router` next to its own routes in a test
(`crates/llm-proxy-sdk/tests/mount_and_build.rs`); `cargo tree` shows no
network dependency for `llm-proxy-protocol` or `llm-proxy-adapters`; the
plan-02 contract fixtures replay unchanged (their assertions and fixtures
untouched; import paths updated from `llm_proxy_provider` to the split
crate names, which the split itself forces). Additionally, one
non-streaming and one streaming contract fixture replay through the
spawned binary against a loopback fake upstream
(`apps/llm-proxy/tests/cli_contract.rs`:
`contract_fixture_replays_through_the_spawned_binary` and
`contract_fixture_replays_streaming_through_the_spawned_binary`).

**Known behavior deltas** (in paths unreachable before the refactor):
client re-encode failure after a completed non-streaming response
previously returned a 500 with a `ResponseFailed` ledger row and no
completed row; it now returns the same 500 while the ledger keeps the
truthful `ResponseCompleted` row and writes no failed row. Nothing else
in the 1,405-test workspace suite changed.
