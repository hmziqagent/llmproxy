# 06. Cache and dedup

- **Status:** Completed
- **Depends on:** 05
- **Audit rows:** 14
- **Code-audit bugs:** BUG-13
- **Backlog:** 15, 24
- **Conflicts:** 4, 5, 7

Replace the dedup window with mechanisms that fit together, and protect provider
prompt caches from the proxy's own edits.

## To do

1. **Retire `dedup_window` (row 14, BUG-13, conflict 4).** It runs first on raw bytes, ignores the caller, covers
   only 500 ms from first sighting and answers 409, which stock SDKs retry. It's in
   `crates/llm-proxy-server/src/middleware.rs` and off by default.
2. **Per-caller single-flight.** Identical concurrent requests from one caller share
   one upstream call.
3. **`Idempotency-Key` (backlog 24).** Opt-in. Claim the key before budget checks,
   replay the stored response on a matching retry, and return 409 only while the
   first request is in flight.
4. **Response cache (conflict 7).** Opt-in and exact-match, with keys scoped to the
   caller. Lookup comes before single-flight and upstream calls. A hit records usage
   as a hit under the rule in plan 05.
5. **Data-class policy (conflict 7).** State what each opt-in stores and for how long.
   Prompts and responses stay unstored by default.
6. **Mutation phases (conflict 5).** Run request edits in a fixed order. Nothing
   changes bytes before the last cache breakpoint without an explicit opt-in.
7. **Prompt-cache hygiene (backlog 15).** Opt-in removal of volatile bytes and marker
   injection, at the provider edge. This depends on the marker fix in
   [plan 01](../completed/01-bug-fixes.md), which is done, and excludes
   fingerprinting.

## Done when

- With every opt-in off, the proxy stores no prompt or response, and a test checks
  this.
- A cache hit succeeds where dedup used to answer 409.
- One caller's cached response can't be returned to another caller.
- A prefix-stability test shows that no enabled feature changes bytes before the last
  breakpoint unless opted in.

## Done

- The `RequestDeduplicator`, `server.dedup_window` (now rejected as an unknown
  field) and the dedup metric are gone; identical requests are always answered.
- Per-caller single-flight lives in the engine (`llm_proxy_engine::cache`):
  identical concurrent requests from one caller share one upstream call.
  Non-streaming followers await the leader's outcome; streaming followers
  replay the leader's collected events. Nothing is retained after completion.
  A leader that is refused or fails before its first byte fails its followers
  with the same error; a leader that vanishes before its first byte makes its
  followers (streaming and not) re-run the request once. Known limitation:
  once a shared stream has committed, a disconnecting leader ends it for its
  joiners, who receive the events collected so far and an in-band error
  rather than the remainder of the response.
- `Idempotency-Key` (opt-in, `[server.idempotency]`) is claimed at slot 4,
  before cache, single-flight and budget checks; a matching retry replays the
  stored response until the TTL, 409 only while the first request is in
  flight, 400 for a key reused with a different request (body or provider
  route), and failures release the key so a retry executes.
- The response cache (opt-in, `[server.cache]`) is an exact-match,
  caller-scoped, TTL-aged, capacity-bounded `MemoryCache` behind the `Cache`
  trait; hits and served copies record a row under plan 05's rule (usage
  kept for the record, nothing charged) and bump a cache-hit metric.
- The data-class policy (`docs/data-class-policy.md`) states what each
  opt-in stores, where and for how long; prompts and responses are stored
  nowhere by default, and a test asserts no ledger row carries their bytes.
- Request mutations run in the fixed order documented in
  `crates/llm-proxy-adapters/src/mutation.rs`: route edits, dialect edits,
  then operator hygiene at the provider edge. Prefix-stability tests hold
  the invariant: the marked prefix reaches the upstream unchanged unless the
  operator opted in.
- `[provider.prompt_cache]` strips configured volatile JSON pointers and
  injects Anthropic `cache_control` markers within the four-breakpoint budget
  (newest message never marked, client markers kept). Markers go only into
  array-form `system`/`content`; string-form prompts are passed through
  untouched. Malformed pointers are refused at config load. No fingerprinting.
- Fix that fell out: `AppState`'s engine cell is `Arc`-shared across state
  clones — axum clones the state per request, and a per-clone cell gave every
  request a fresh engine, which would have silently emptied the cache,
  single-flight and idempotency stores on every request.
