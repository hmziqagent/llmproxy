# 07. Hardening and operations

- **Status:** Completed
- **Depends on:** Nothing; steps 10 and 11 need 03
- **Audit rows:** None
- **Code-audit bugs:** BUG-25
- **Backlog:** 11, 21, 23, 25, 26, 27, 32, 34, 37
- **Conflicts:** 11

Make the proxy safer to expose and easier to run. Items 1 to 6 are small and can ship
at any time.

## To do

1. **Host and Origin validation (backlog 11).** Defend against DNS rebinding in
   `mode = "none"`. Refusing non-loopback binds without auth already exists.
2. **Cleartext credentials (backlog 25).** Refuse an API key on an `http://` URL to an
   off-machine host, with a loopback exemption.
3. **Admin caching (backlog 37).** Send `Cache-Control: no-store` on all admin API
   responses.
4. **Self-loop prevention (backlog 32).** Refuse an upstream that resolves to the
   proxy's own listener, and add a hop header on egress.
5. **`log_format` (BUG-25).** The config value is ignored at startup in
   `apps/llm-proxy/src/commands/serve.rs`. Honour it, with `RUST_LOG_FORMAT` still
   winning as documented.
6. **Config provenance (backlog 34).** Print each resolved setting with its source.
7. **Metrics (backlog 21).** Export the existing counters through Prometheus and OTel.
8. **In-flight visibility (backlog 27).** Add state and elapsed time to the in-flight
   rows the admin API already returns.
9. **`doctor` command (backlog 26).** Run live reachability checks and one-token
   probes, each failure naming its fix.
10. **Config snapshots (conflict 11).** Load config through `ConfigSource` from plan 03
    into an immutable snapshot that can be swapped, with one stated precedence.
11. **Hot reload (backlog 23).** Swap one provider at a time without touching other
    connections, and refuse changes that need a restart rather than half-applying them.

## Done when

- A request with a foreign `Host`, or a browser request (one carrying
  `Sec-Fetch-*`) with a foreign `Origin` or a cross-site `Sec-Fetch-Site`,
  is refused in `mode = "none"`; a non-browser client's stray `Origin`
  header is admitted because the Host check already pins it to this
  listener.
- `validate` refuses cleartext credentials to a remote host.
- A reload that changes one provider leaves in-flight requests on other providers
  running.
- `doctor` reports a pass or a named fix for each configured provider.

## Done

All eleven steps landed, each with regression tests:

1. `crates/llm-proxy-server/src/host_guard.rs` — under `auth.mode = "none"` the
   API routes and the lightweight `/health`, `/ready`, `/version` routes
   refuse a `Host` that does not name this listener, a browser
   `Origin` that is neither this proxy nor an operator-listed origin, and a
   `Sec-Fetch-Site` from another site. `Origin` is judged only on requests
   carrying Fetch Metadata, because captured CLI traffic sends an `Origin`
   with no browser markers. Client-keys mode stands down (probes stay
   unauthenticated).
2. `ConfigValidationError::CleartextCredentials` — `http://` adapter and
   discovery endpoints to off-machine hosts are refused at load (loopback
   literals and `localhost` exempt); `validate` and startup share the check.
3. The `/usage` router layers `Cache-Control: no-store` (overriding) and
   `Pragma: no-cache` on every response, the 401 refusal included.
4. `llm-proxy-core::self_loop` — every provider upstream request (chat,
   discovery) carries `x-llm-proxy-hop: 1`; any request arriving with it is
   answered `508 Loop Detected`; startup and `validate` resolve every
   adapter and discovery endpoint (DNS included) and refuse those landing
   on the proxy's own listeners.
5. `LogFormat::effective` + config-before-subscriber init in `cmd_serve`:
   `server.log_format` is honoured, `RUST_LOG_FORMAT` still wins — verified
   against the real binary.
6. `llm_proxy_app::provenance` — `validate` prints every resolved setting
   with its layer (config / default / env) under one stated precedence,
   CLI > env > config file > default. Secrets never appear.
7. `Snapshot::to_prometheus_text` + the token-gated `/metrics` route on the
   admin surface; opt-in `[server.metrics.otlp]` pushes the same counters
   as OTLP/HTTP JSON (tested against a live local collector).
8. `/usage/requests` rows carry a derived `state`
   (`in_flight`/`failed`/`completed`) and in-flight rows carry `elapsed_ms`
   measured at query time.
9. `llm-proxy doctor [--probe]` — DNS/TCP reachability per endpoint (at the
   `[provider.egress]` proxy for an egress provider), plus an opt-in one-token
   probe through the real transport (egress proxy included); every non-PASS row
   names its fix; failures exit 1.
10. `llm_proxy_engine::SwappingConfigSource` — whole-snapshot swaps, never
    merges; one stated precedence documented on the `ConfigSource` trait;
    `AppState` wires the engine over it.
11. `server.hot_reload = true` + SIGHUP or `llm-proxy reload`: providers and
    pools reload through the startup gates and the engine slot swaps;
    in-flight requests finish on the engine they hold; anything under
    `[server]`, `[auth]` (including `[auth.admin] key_store`), `[storage]`,
    `[replica]` or `[mcp]` is refused whole with a stable `restart-required:`
    error naming the fields.
