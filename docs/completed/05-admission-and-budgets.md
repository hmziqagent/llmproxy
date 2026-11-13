# 05. Admission and budgets

- **Status:** Completed (2026-11-07)
- **Depends on:** 04
- **Audit rows:** None
- **Code-audit bugs:** BUG-24
- **Backlog:** 4, 6, 18, 22, 30
- **Conflicts:** 2, 8, 12, 14, 15

Decide in one place whether a request may run, and what it costs. The research
proposes three different admission models, and this plan replaces them with one.

## To do

1. **Error policy (conflict 14).** Keep one table in one module for 402, 408, 409,
   429, 503 and `x-should-retry`. Stock SDKs retry some of these automatically, so
   each code needs a deliberate choice.
2. **Key lookup (conflict 12).** A cache miss in
   `crates/llm-proxy-server/src/middleware.rs` tries every key in turn, so 1,000 keys
   means about 1,000 argon2 runs per bad token. Use a key-id prefix and one
   verification.
3. **Per-key allowlists (backlog 18).** Add model and provider globs, and fail closed
   when a request has no identifiable model.
4. **One admission step (backlog 6, conflict 2).** Reserve, then settle or refund. A
   cost that can't be estimated goes to an explicit per-key policy. This lives in the
   `Admission` trait from plan 03, not in `llm-proxy-core`.
5. **Limit windows (backlog 6).** Add RPM, TPM, RPD and per-key budgets. Per-key
   budgets and the rpd window are rebuilt from `requests` rows at startup when
   `[storage]` is set; rpm/tpm windows restart empty; without `[storage]` everything
   resets on restart.
6. **What counts (conflict 8).** Budgets charge what providers bill, per attempt.
   Cache hits charge nothing and are recorded as hits.
7. **Concurrency caps (backlog 30).** Limit in-flight requests per provider and model.
   When saturated, return 503 with a short `Retry-After`, not 429.
8. **Preflight (backlog 4, conflict 15).** Warn by default. Reject only with a safety
   margin or a provider token count, because non-OpenAI estimates use chars/4 and
   counting ignores tools, tool results and images.
9. **Pricing (backlog 22, BUG-24).** Add an opt-in sync from models.dev and an explicit
   unpriced state. Unpriced rows never show as $0 in summaries.

## Done when

- Concurrent requests can't overspend a budget by more than the reserved amount.
- A refund restores the budget when a request fails before any cost.
- Lookup cost for a bad token doesn't grow with the number of keys.
- Every status code the proxy produces is in the policy table, and a test enforces it.

## Done

- `llm_proxy_engine::error_policy` holds the one status table; the server renders
  its `x-should-retry` row on every error plus any `Retry-After` advice, and
  tests on both sides enforce that every engine and server status is a row.
- Client tokens are `<key-id>.<secret>`: one argon2 verification per bad
  prefixed token, zero for an unknown id, at any key count. A bare secret
  without a dot is refused unless `[auth] legacy_bare_tokens = true`; the
  opt-in scan is capped, and recently failed token digests are briefly
  negative-cached so a repeated bad token costs one verification.
- `PolicyAdmission` implements the `Admission` trait with the four rules in
  one atomic reservation: glob allowlists (403, fail closed without a model),
  rpm/tpm/rpd windows (429 with the reset), a `usd_per_day` budget that
  reserves first (402) with an explicit unestimatable policy, and
  per-provider and per-model in-flight caps (503 with a one-second
  `Retry-After`). The allowlist verdict also runs on its own at pipeline
  slot 2 (`Admission::authorize`), ahead of the idempotency replay, the
  cache and single-flight, so a key restricted at runtime is not served an
  answer stored for it earlier. Settlement charges exactly what the provider
  billed and refunds a failure before any usage.
- Preflight estimates input tokens once (shared with admission) and warns by
  default; enforce mode rejects only past the configured margin.
- `llm-proxy pricing-sync` writes per-token pricing from models.dev into
  opted-in provider files, and unpriced rows report `cost: null` with an
  unpriced count in usage summaries (never `$0`).
