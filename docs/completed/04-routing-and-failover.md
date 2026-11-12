# 04. Routing and failover

- **Status:** Completed (2026-11-05)
- **Depends on:** 03
- **Audit rows:** None
- **Code-audit bugs:** BUG-14
- **Backlog:** 1, 2, 8, 14, 17, 20, 31, 38
- **Conflicts:** 1, 3, 6, 8

Add pools, retries and failover on top of the routing decision in the
[project goal](../goal.md#routing).

## To do

1. **Attempt-level usage rows (conflict 8).** Add one row per upstream attempt,
   linked to its request. Failed attempts can still cost money.
2. **Error classification (backlog 1).** Sort transient from terminal errors in one
   place and honour `Retry-After`. Retries wrap the send itself, because 429s, 5xx
   responses and connect errors return before any stream exists.
3. **Retry ladder (backlog 1).** Add per-provider cooldowns and one deadline across
   all attempts.
4. **Pools (conflict 1).** Add config for named pools and serve them at
   `/providers/{pool}/v1/...`. Refuse a pool name that matches a provider name at
   config validation. Update the source-guard tests in
   `crates/llm-proxy-server/src/routes/messages.rs` to allow this path only.
5. **Credential pools (backlog 31).** Support several keys per provider, with
   selection and failure feedback in the attempt loop and a ledger column for the
   key used.
6. **Pre-output failover (backlog 2, conflict 3).** Each route gets a hold budget:
   buffer for up to N seconds, then commit and send heartbeats. Failover and status
   codes are possible only before the commit, and errors are in-band after it.
7. **Heartbeats and terminal usage (backlog 14).** Cover silence before the commit,
   and guarantee a terminal usage event on every stream. Heartbeats after the first
   event already exist.
8. **Timeouts (backlog 8, BUG-14).** Make connect, first-byte, idle and total
   timeouts configurable and validate them at config load. This replaces the fixed
   `DEFAULT_UPSTREAM_HEADERS_TIMEOUT`, and the total deadline is new.
9. **Health and quota (backlog 20).** Turn 429 and rate-limit headers into cooldowns
   based on real reset times, with an optional synthetic probe.
10. **Session affinity (backlog 17, conflict 6).** Record session ids on usage rows,
    and let routing take an affinity input. Failover strips state tied to one account
    or provider, such as signed thinking and file ids, or refuses the switch.
11. **Bare `/v1/*`.** Off by default. When enabled, it resolves a model name to a
    provider or pool and runs through the same engine.
12. **Retry signals (backlog 38).** After retries run out, send
    `x-should-retry: false` and a computed `Retry-After`.

## Done when

- `/providers/{provider}/v1/...` behaves exactly as before and never fails over.
- A pool request survives a 429, a 5xx and a connect error from its first member,
  and the ledger shows every attempt.
- A stream that fails before commit fails over, and one that fails after commit ends
  with an in-band error and a failed ledger row.
- With bare `/v1/*` left off, those routes return 404.

## Done

- `llm_proxy_engine::classification` is the one transient/terminal sort,
  with `Retry-After` parsing for both header spellings, Go-style durations
  and RFC 3339 reset timestamps — scoped by header name (`retry-after`, or
  names containing both `ratelimit` and `reset`), longest advertised reset
  wins; the ladder consumes it through `next_wait` for backoff and
  cooldown decisions.
- Pools live in `[pool.*]` config and serve `/providers/{pool}/v1/...`:
  config validation rejects a duplicate member, an empty member list and a
  pool name colliding with a provider (`validate_pool_config_against`,
  `provider_registry.rs`), and every attempt of a pool's ladder writes its
  own `AttemptCompleted` row — `pool_survives_a_429…`, `…a_5xx_then_a_
  connect_error…` and `a_direct_provider_route_never_fails_over…`
  (`crates/llm-proxy-engine/tests/pools_and_failover.rs`) pin the done-when
  lines, over HTTP in `crates/llm-proxy-server/tests/routing_pools.rs`.
- One ladder deadline spans all attempts (`ladder_deadline`, the tighter of
  the route's `total_timeout` and `server.upstream_timeouts.total`;
  `the_ladder_deadline_spans_all_attempts`), with per-member attempts,
  backoff, and passive cooldowns on the `Cooldowns` trait — a 429 with a
  reset time cools the member until it (`upstream_429_with_retry_after`)
  and the optional synthetic probe clears it on recovery
  (`the_health_probe_clears_a_members_cooldown_when_it_recovers`).
- Credential pools rotate `api_keys` with failure feedback: a rejected key
  stays out of rotation while it cools and the used key id lands on the
  attempt and usage rows (`a_rejected_key_stays_out_of_rotation…`,
  `a_rejected_key_cools_the_key_not_the_member`).
- Pre-output failover is the hold budget: the buffer phase holds the
  response for the configured window (committing an empty head when it
  expires in silence, heartbeats covering the gap), fails over before the
  first byte, and after commit every failure is in-band with a terminal
  event and a failed row (`a_stream_that_fails_before_commit_fails_over`,
  `a_stream_that_fails_after_commit_ends_with_an_in_band_error…`,
  `the_hold_budget_commits_a_silent_stream_and_covers_the_gap`).
- Connect/first-byte/idle/total timeouts come from
  `server.upstream_timeouts` (per-family overrides in
  `server.endpoint_families`), validated at config load
  (`validate_upstream_timeouts`).
- Sticky pools pin a session (the `x-claude-code-session-id` header) to the
  member that served it, reordering the next ladder; the pin never applies
  to a direct route or a non-sticky pool, and the refuse policy holds the
  session on its pinned member even mid-cooldown
  (`sticky_affinity_pins_a_session_to_the_member_that_served_it`,
  `a_direct_route_does_not_pin_the_session`,
  `refuse_policy_holds_when_the_pinned_member_is_already_cooling`). A
  provider switch strips provider-bound state (`strip_provider_bound_state`):
  provider hints outside the kept allowlist, tool-call signatures, and
  thinking blocks from earlier assistant turns, and re-encodes from the
  stripped core instead of passing the raw body through. It refuses the
  switch when the request carries state no other provider can honour — a
  media block referencing a provider file, or signed thinking in an open
  tool loop (`failover_drops_earlier_turn_thinking_and_tool_signatures`,
  `failover_refuses_signed_thinking_in_an_open_tool_loop`,
  `failover_refuses_to_move_a_file_reference_between_members`,
  `a_request_with_nothing_to_strip_keeps_same_format_passthrough`).
- Bare `/v1/*` is off by default and 404s while off
  (`bare_v1_routes_return_404_when_left_off`); enabled it resolves a model
  to a provider or pool and runs the same failover
  (`bare_v1_resolves_a_model_to_a_provider_when_enabled`, `…_to_a_pool…`).
- Retry exhaustion answers with the status policy's `x-should-retry: false`
  plus a computed `Retry-After` (`llm_proxy_engine::error_policy`, asserted
  on the HTTP surface in `routing_pools.rs`).
