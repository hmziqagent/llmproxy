# Repository intelligence — fourth research wave (November 2026)

Method: a fourth research wave, run 2026-11-04, harvesting **63
repositories** in two tiers. Twenty-three were researched in full along four
angles — **code** (capabilities read from clones), **issues** (user-reported
pain points), **pulls** (where each project is investing), and **gaps**
(verified absences on the llm-proxy side) — and forty more received
**issue-tracker sweeps** (recent issues and PRs mined for pain points and
directions). Tier assignment follows the harvest's data, not a judgment of
importance: the 23 full-angle repositories are exactly those with a code
clone in this wave — 22 under `research-repos/` (`ls research-repos/ | wc -l`
→ 22, run this session) plus mistral.rs at `/tmp/intel-mistral/mistral.rs` —
while the 40 sweep repositories were harvested through their issue/PR
trackers only, via the snapshots at
`research-issues/<owner>__<repo>/{issues,pulls}.json`. Every claim below
carries the harvest's evidence: a GitHub URL or a clone path. Fully-qualified
clone paths start with `research-repos/<owner>__<repo>/`; paths written
clone-relative (`src/…`, `crates/…`, `backend/…`, `packages/…`, …) refer to
that section's repository clone under the same root, and mistral.rs paths
are relative to `/tmp/intel-mistral/mistral.rs` @ 52b3c49. Maintenance
statistics (merge counts, issue tallies, dates, star counts) are reads of
those same `research-issues/` snapshots and clone refs captured by the
harvest; they carry no per-claim URL beyond that provenance.

Statements and `path:line` citations about llm-proxy's own code describe
commit `79cf406` (2026-11-05). Plans 01–10 have since rebuilt most of it;
current behaviour is in [docs/goal.md](../goal.md) and
[docs/completed/](../completed/).

Nothing was built or run; llm-proxy-side statements come from the wave's
provided inventory plus spot-checks verified this session at llm-proxy
commit `6bcf1ef4e4d32fc0eae69b528f0e10cfea3e433b` (`git rev-parse HEAD`,
2026-11-04): no request-path retry outside
`crates/llm-proxy-provider/src/discovery.rs:145`; flat `TimeoutLayer` at
`crates/llm-proxy-server/src/routes/mod.rs:64-67`; `MAX_BODY_BYTES` 32 MiB
at `routes/mod.rs:37`; no `prompt_tokens_details` / `cached_tokens` anywhere
  in `crates/` or `apps/`; core usage already carries `cache_creation`/
  `cache_read` at `crates/llm-proxy-protocol/src/core.rs:683-685` (cite
  corrected 2026-11-08 per research audit; `:639-641` is `Cost`); and the
rate limiter keys by caller key id when authenticated, client IP otherwise
(`crates/llm-proxy-server/src/routes/core_pipeline.rs:173-176`) — per-key
RPM already exists, matching the compendium's narrowed gap #3.

Companion file: [`compensation-backlog.json`](compensation-backlog.json) —
the machine-readable, priority-sorted list of features/fixes this record
implies for llm-proxy. Earlier waves: the gap register (#1–#20) lives in
[`competitive-landscape-compendium.md`](competitive-landscape-compendium.md);
wave-3 niche deep dives in [`niche-repos.md`](niche-repos.md). Wave 4 is
deliberately **not** folded into
[`repos-registry.json`](repos-registry.json): the registry remains the
three-wave capability-row artifact, and this wave's machine-readable output
is the compensation backlog (candidates mapped to crates rather than
capability rows).

From record to backlog: a Part-I gap or sweep finding becomes a compensation
candidate when the need recurs across three or more harvested repositories
(the recurring-needs threshold below) or is a single-repo capability with
direct user validation and no cheaper substitute. Deliberate exclusions:
ThinkWatch's minimum-generation-speed stream cutting
(`research-repos/ThinkWatchProject__ThinkWatch/crates/gateway/src/proxy/min_speed.rs:1-88`)
presupposes streaming tokens/sec telemetry llm-proxy does not yet collect —
it is a follow-on to the phase-timeout and usage-estimation candidates, not
an independent one; mistral.rs's stateful Responses machinery and
session-portability API (`mistralrs-server-core/src/cached_responses.rs:14-27`,
`handlers.rs:634-691` per the harvest) are design inputs to the Responses
client-endpoint candidate — server-side state is a separate decision once
the endpoint exists; engine-local surfaces (model load/unload, KV-cache
management, GPU residency) belong to inference servers, not a translation
proxy, and stay recorded as context only.

## Contents

- Part I — Full four-angle harvests (23 repositories)
- Part II — Issue-tracker sweeps (40 repositories)
- Pain-point index — theme → repos and issues
- Direction index — where the niche ecosystem is heading
- Recurring unmet needs — patterns in 3+ repositories

---

## Part I — Full four-angle harvests

### EricLBuehler/mistral.rs

**Identity · Language · Form factor.** Rust workspace (edition 2021, MSRV
1.95, 15+ crates) built on candle: a local LLM inference engine whose
`mistralrs serve` binary doubles as an OpenAI- and Anthropic-compatible API
server with an embedded web UI.

**Maintenance.** Very active — HEAD 52b3c49b dated 2026-11-02 ("per-request
context limits for coding agents"), PRs merged daily.

**Capabilities (code).** Three client dialects on one server (OpenAI Chat,
OpenAI Responses, Anthropic Messages) with count_tokens implemented by
round-tripping the real loaded tokenizer
(`/tmp/intel-mistral/mistral.rs/mistralrs-server-core/src/anthropic.rs:1834,1938,2006`);
per-request serving context = min(model context, paged-KV capacity,
`--max-model-len`) with a typed `ContextLengthExceeded` mapped per dialect
(`mistralrs-server-core/src/handler_core.rs:27,144`); one `observe_http`
middleware exporting Prometheus + TTFT/inter-token-latency histograms,
request-id injection, and SSE usage deferred until the body stream ends
(`mistralrs-server-core/src/metrics.rs:45-98,214-331`); client-disconnect
cancellation wired end-to-end into the scheduler
(`mistralrs-core/src/scheduler/default_scheduler.rs:364-370`); runtime model
lifecycle over HTTP plus confined hot LoRA swap (`mistralrs-server-core/src/handlers.rs:419-529`);
MCP in both directions (`mistralrs-server-core/src/mcp_server.rs:13-35`);
human-in-the-loop tool approvals over SSE (`mistralrs-server-core/src/approvals.rs:53-130`);
OS-level subprocess sandbox (Landlock/seccomp/Seatbelt, `mistralrs-sandbox/src/lib.rs:1-40`);
OpenAI-compatible `/v1/files` and versioned `/v1/skills`
(`mistralrs-server-core/src/skills.rs:47-282`).

**Pain points (issues).** Default `0.0.0.0:1234` with no auth exposed the
whole control surface; five recent GHSAs were all reachable unauthenticated
(https://github.com/EricLBuehler/mistral.rs/issues/2490); tool-call parsing
is the most bug-dense area (https://github.com/EricLBuehler/mistral.rs/issues/2466,
/issues/2520, /issues/2528); `thinking.budget_tokens` accepted but never
enforced (https://github.com/EricLBuehler/mistral.rs/issues/2522); no
per-request prefix-cache isolation — `CacheSalt` exists but is never
constructed (https://github.com/EricLBuehler/mistral.rs/issues/2523);
auto KV sizing starves admission on small GPUs with a 503 that never names
memory pressure (https://github.com/EricLBuehler/mistral.rs/issues/2502).

**Directions (pulls).** The server is being tuned as a local backend for
coding agents: per-request context limits with native per-API overflow
errors and `max_model_len` on `/v1/models`
(https://github.com/EricLBuehler/mistral.rs/pull/2538); per-request
`cache_salt` cache isolation open (https://github.com/EricLBuehler/mistral.rs/pull/2524);
periodic defensive-security passes (https://github.com/EricLBuehler/mistral.rs/pull/2447);
Apple Silicon and quantization as the performance bets
(https://github.com/EricLBuehler/mistral.rs/pull/2418, /pull/2512).

**Gaps vs llm-proxy.** Context-limit preflight is a cheap win for llm-proxy,
which surfaces `context_length` on `/v1/models`
(`crates/llm-proxy-server/src/routes/models.rs:35,142`) but never
preflight-checks requests (gap evidence: mistral.rs PR #2538 +
`mistralrs-core/src/engine/add_request.rs:523-537` @ 52b3c49); also absent
from llm-proxy: legacy `/v1/completions`, `/v1/skills`, session-portability
API, stateful Responses machinery (background tasks + retrieval/cancel +
`previous_response_id` chaining), served OpenAPI spec, live model-ops over
HTTP, and a client chat UI (`mistralrs-server-core/src/route_registry.rs:27-120`).

### MikkoParkkola/mcp-gateway

**Identity · Language · Form factor.** Rust single-binary MCP gateway
(edition 2024, MSRV 1.95, v4.0.0, ~302k lines) exposing a compact ~20-tool
`gateway_*` meta-surface over MCP backends plus 125 prebuilt REST
capabilities. PolyForm-NC with a commercial tier.

**Maintenance.** Extremely active and effectively solo — 69 of the last 100
PRs merged on 2026-11-04 alone; nearly all 394 true issues are owner-filed
severity-tagged review findings.

**Capabilities (code).** Per-backend `Failsafe` composing circuit breaker,
governor token bucket, backon retry, and latency health — with 429s excluded
from failure accounting (`research-repos/MikkoParkkola__mcp-gateway/src/failsafe/mod.rs`);
three-layer kill switch: operator disable meta-tool, per-backend sliding-window
error budgets, per-capability budgets with cooldown auto-recovery
(`src/kill_switch/mod.rs`); pre-invoke `BudgetEnforcer` in micro-USD atomics
with UTC-day CAS reset (`src/cost_accounting/enforcer.rs`); structural-diff
hot reload that refuses listener changes (`src/config_reload/mod.rs`);
canonical-JSON idempotency with hard resource fences
(`src/idempotency/admission.rs`); credential injection with forced overwrite
and audit (`src/secret_injection.rs`); TF-IDF tool discovery behind the
meta-surface (`src/semantic_search/mod.rs`).

**Pain points (issues).** Rate limiter and breaker keyed by API-key display
name so renamed/shared keys share buckets
(https://github.com/MikkoParkkola/mcp-gateway/issues/3950); stdio pipe
backpressure wedged the whole gateway
(https://github.com/MikkoParkkola/mcp-gateway/issues/2531); the idempotent
cache answered before policy so a firewalled tool still served from cache
(https://github.com/MikkoParkkola/mcp-gateway/issues/2445);
credential-echo redaction kept missing stored-answer paths
(https://github.com/MikkoParkkola/mcp-gateway/pull/4552).

**Directions (pulls).** Zero-restart policy reload under per-request pinned
config generations (https://github.com/MikkoParkkola/mcp-gateway/pull/4508);
rotation-safe auth state — rate-limit windows keyed to principal not digest
(https://github.com/MikkoParkkola/mcp-gateway/pull/4501); credential-echo
scrubbing as a core boundary over every answer path
(https://github.com/MikkoParkkola/mcp-gateway/pull/4561); append-only
transparency log (https://github.com/MikkoParkkola/mcp-gateway/pull/4598).

**Gaps vs llm-proxy.** Always-on DNS-rebinding/Origin guard (CWE-346 fix)
versus llm-proxy's opt-in-only CORS and absent Host validation
(`src/gateway/router/origin_guard.rs:316-400` vs
`crates/llm-proxy-server/src/routes/mod.rs:111-119`); cleartext-credential
refusal at config load (`src/config/mod.rs:1144-1190`); per-API-key tool
scoping with globs (`examples/per-client-tool-scopes.yaml:17-52`); `doctor`
diagnostics with fix/verify/rollback commands (`src/commands/doctor.rs:47-105`);
client mTLS (`src/mtls/mod.rs`); hot reload with auth-weakening refusal
semantics (`src/config_reload/mod.rs:93-135`); idempotency replay where
llm-proxy's dedup only 409-rejects concurrent duplicates
(`src/idempotency/admission.rs`).

### NVIDIA-NeMo/Switchyard

**Identity · Language · Form factor.** Apache-2.0 Rust workspace (edition
2024, MSRV 1.96.1, v0.3.0) whose core is `libsy`, an embeddable
model-routing library; the OpenAI/Anthropic/Responses-compatible server is
explicitly a local reference implementation, with Python/Go/TS/WASM
bindings.

**Maintenance.** Very active — 65 of 100 harvested PRs merged 2026-10-23 →
2026-11-03 (~6/day), NVIDIA-engineer-driven.

**Capabilities (code).** Host-driven Step-protocol routing decoupled from
transport, panic-contained and cancellation-safe
(`research-repos/NVIDIA-NeMo__Switchyard/crates/libsy/src/core/algorithm.rs:613-716`);
per-backend retry budget covering transport failures, timeouts, 408/429/5xx
with capped Retry-After and one deadline spanning attempts and stream reads,
first decoded event checked inside the retry boundary
(`crates/libsy-llm-client/src/client.rs:400-528,886-944`); ordered fallback
candidates with per-candidate rewrites, advancing only on gated reasons
(`crates/libsy/src/core/algorithm.rs:257-323`); per-route capability gating
pre-dispatch (`crates/switchyard-server/src/capabilities.rs:12-62`); durable
JSONL routing log + `GET /v1/routing/session-stats`
(`crates/switchyard-server/src/routing_log.rs:27-51,325-371`); dual-source
cost accounting with `CostSource` tags
(`crates/switchyard-costs/src/lib.rs:46-123`); Prometheus export via an
opentelemetry_prometheus reader with per-metric histogram buckets plus
routing-specific instruments (`classifier_fail_open`, `retry_recovered`,
`client_disconnected` kept separate from served responses)
(`crates/switchyard-server/src/metrics.rs:24-33,50-77,119-165`); credential
redaction of
headers/JSON/SSE (`crates/switchyard-server/src/redaction.rs:16-70`);
loss-aware `TranslationPolicy` with structured diagnostics
(`crates/switchyard-translation/README.md:91-123`).

**Pain points (issues).** An OpenAI upstream can return HTTP 200 and reject
the stream in its first SSE frame; treated as success, no fallback
(https://github.com/NVIDIA-NeMo/Switchyard/issues/1132); codecs silently
drop request fields with no diagnostics channel on the stream path
(https://github.com/NVIDIA-NeMo/Switchyard/issues/1215); slow first byte
(~414 s) trips client/proxy timeouts — early response headers plus an
Anthropic `event: ping` keepalive requested, none exists
(https://github.com/NVIDIA-NeMo/Switchyard/issues/1168).

**Directions (pulls).** Library-first positioning
(https://github.com/NVIDIA-NeMo/Switchyard/pull/1155); in-stream pre-answer
fallback for 200-then-error with a held prelude
(https://github.com/NVIDIA-NeMo/Switchyard/pull/1133); cross-wire reasoning
normalization with explicit loss diagnostics
(https://github.com/NVIDIA-NeMo/Switchyard/pull/1216); session-affinity
stability including preserving the prompt-cache-reusable prefix
(https://github.com/NVIDIA-NeMo/Switchyard/pull/1213); decision models
spreading through every algorithm
(https://github.com/NVIDIA-NeMo/Switchyard/pull/1195).

**Gaps vs llm-proxy.** Execution-signal stage routing from tool-result
history (`crates/libsy/src/algorithms/util/tool_signals.rs:34-120`); LLM-judge
task-classifier routing (`crates/switchyard-server/README.md:176-190`);
learned prefill routing (`crates/prefill-router/src/`); verification-gated
routing with shadow mode and kill switch (`crates/libsy/src/algorithms/vgr.rs`);
first-party evaluation/soak harness (`benchmark/README.md:8-19`);
provider-credential redaction at the output boundary
(`crates/switchyard-runner/src/provider_key_redactor.rs:19-43`); per-target
request augmentation (system prompt prepend, `extra_body` merge)
(`crates/switchyard-server/README.md:106-107`).

### ThinkWatchProject/ThinkWatch

**Identity · Language · Form factor.** Rust workspace (6 crates: auth,
common, gateway, mcp-gateway, server, test-support) plus a React console —
a self-hosted AI API + MCP gateway for organizations over PostgreSQL,
Redis, and ClickHouse. BUSL-1.1.

**Maintenance.** Extremely active — v3.11.0 committed 2026-11-03, 93 of the
last 100 PRs merged; organic user signal thin (only 5 true issues ever, two
of them vendor noise).

**Capabilities (code).** Strategy-driven weighted routing with Redis-shared
EWMA health and CAS-only breaker state transitions
(`research-repos/ThinkWatchProject__ThinkWatch/crates/gateway/src/strategy.rs:1-120`,
`src/health.rs:1-119`); deadline-budget retry with precise classification —
transient = 5xx/529/connection/stream-dead-before-output, never 429/other
4xx/after-content — with SSE `: keep-alive` frames while waiting
(`crates/gateway/src/proxy/retry.rs:1-130`); sealed-history provider pinning
for Responses `store:false` (30-day Redis, plain-language refusal when the
provider has no route, `crates/gateway/src/proxy/sealed.rs:1-92`);
per-provider minimum generation speed that cuts dribbling streams with 503
and drops affinity (`crates/gateway/src/proxy/min_speed.rs:1-88`); Redis Lua
bucketed sliding windows for requests *and* tokens with admit/record split
and fail-open/closed switch (`crates/common/src/limits/sliding.rs:1-46`);
exact-match response cache keyed on the whole request body whose hits still
debit limits (`crates/gateway/src/cache.rs:1-110`); usage estimation when
upstream usage never arrives (`crates/gateway/src/usage_estimate.rs:1-15`);
early-cancel 499 audit rows via a drop guard armed at auth
(`crates/gateway/src/proxy/early_cancel.rs:1-24`, merged as the pre-stream
cancellation rows of https://github.com/ThinkWatchProject/ThinkWatch/pull/54
and the drop-guard hardening of
https://github.com/ThinkWatchProject/ThinkWatch/pull/62); self-healing
protocol relearn persisted per route
(`crates/gateway/src/proxy/protocol_relearn.rs:1-24`).

**Pain points (issues).** OIDC multi-audience ID tokens rejected
out-of-the-box (https://github.com/ThinkWatchProject/ThinkWatch/issues/10);
third-party MCP-server admission scanning declined not-planned
(https://github.com/ThinkWatchProject/ThinkWatch/issues/113).

**Directions (pulls).** Failover accounting decided by upstream error class
so one caller's bad request cannot open breakers
(https://github.com/ThinkWatchProject/ThinkWatch/pull/38); transient retry
across all candidate routes under a 120 s ceiling
(https://github.com/ThinkWatchProject/ThinkWatch/pull/122); per-bucket
pricing with cache-read/write weights
(https://github.com/ThinkWatchProject/ThinkWatch/pull/40); limits hardening
with lineage-surviving key counters and computed `Retry-After` +
`x-should-retry: false` (https://github.com/ThinkWatchProject/ThinkWatch/pull/81);
Gemini client surface plus Responses over WebSocket through one pipeline
(https://github.com/ThinkWatchProject/ThinkWatch/pull/43).

**Gaps vs llm-proxy.** Minimum-generation-speed stream cutting (not in the
register); org identity — OIDC/TOTP/RBAC with deny-wins policy union
(`crates/auth/src/oidc.rs:1-50`, `src/rbac.rs:16-33`, `src/totp.rs:7-23`);
stream-safe reversible redaction with a placeholder ledger and scan cache
(`crates/gateway/src/redaction.rs:1-79`); tool-call inspection against an
upstream-MITM threat model (`crates/gateway/src/tool_inspection.rs:1-40`);
client-facing `GET /v1/usage` for the caller's own key
(`crates/server/src/handlers/key_usage.rs:1-24`); SIEM forwarding with a
durable outbox (`crates/common/src/audit/outbox.rs:1-35`).

### Wei-Shaw/sub2api

**Identity · Language · Form factor.** Go 1.27 monolith (Gin + Ent +
PostgreSQL/Redis + Vue 3 admin) pooling upstream AI subscription/OAuth
accounts (Claude, Codex, Gemini, Grok, Antigravity, CN providers) and
redistributing quota via billed API keys. 1,448 non-test Go files.

**Maintenance.** Extremely active — HEAD 2026-11-04; the issue tracker is
high-volume and code-level (34 true issues in five days at harvest).

**Capabilities (code).** Request-path failover loop with bounded account
switches, failed-account exclusion, livelock guards, and client-gone
cancellation semantics (`research-repos/Wei-Shaw__sub2api/backend/internal/handler/failover_loop.go:33-267`);
sticky sessions in Redis (1 h) with re-billing of input tokens as cache
reads when a switch abandons a warm prompt cache — a cost-correctness
detail unique to multi-account pools (`backend/internal/service/gateway_service.go:462-484`);
per-account quota-window pausing from real upstream reset snapshots
(`backend/internal/service/account_scheduling_threshold_eval.go:22-60`);
per-platform OAuth token providers with refresh skew
(`backend/internal/service/claude_token_provider.go:13-73`); first-output
timeout staging for streaming failover — SSE buffered 64 KiB→temp file
until the first output event so a hung upstream can be failed over before
any byte is committed (`backend/internal/service/openai_first_output_timeout.go:23-55`);
in-flight balance reservation from max-output-token estimates
(`backend/internal/handler/gateway_inflight_reservation.go:13-70`); channel
monitor v2 synthetic probes with error-taxonomy rollups
(`backend/internal/service/channel_monitor_v2.go:322-621`); a management MCP
server exposing the platform's own REST admin/user API as tools over
stateless Streamable HTTP, built as four immutable pre-configured servers
(user/admin scope × read-only/write) so no credentials persist across HTTP
requests (`backend/internal/managementmcp/server.go:19-60`; landed via
https://github.com/Wei-Shaw/sub2api/pull/8389).

**Pain points (issues).** Request-level upstream 403s misclassified as
account faults and failover-replayed until every account is disabled
(https://github.com/Wei-Shaw/sub2api/issues/8442); cooldown scope/duration
mismatched to the upstream signal
(https://github.com/Wei-Shaw/sub2api/issues/8439, /issues/8502, /issues/8479);
abnormal stream terminations masked as normal completion — synthetic
`end_turn` + `[DONE]` silently drop tool calls
(https://github.com/Wei-Shaw/sub2api/issues/8505); WebSocket failover
replayed the conversation's first request, stripping a running agent's
context (https://github.com/Wei-Shaw/sub2api/issues/8513); cross-protocol
conversion loses session IDs and agent semantics
(https://github.com/Wei-Shaw/sub2api/issues/8514, /issues/8461); key-to-group
and model scoping not enforced end-to-end — keys bound to one group
scheduled to other groups' accounts, and account-level model names fail to
constrain passthrough (https://github.com/Wei-Shaw/sub2api/issues/8480,
/issues/8492, /issues/8506); account connectivity testing does not scale —
one account can expose hundreds of models testable only one-by-one, with a
fixed prompt and no thinking-effort control
(https://github.com/Wei-Shaw/sub2api/issues/8454, /issues/8494).

**Directions (pulls).** Prompt-cache engineering by
mimicking first-party client fingerprints — cached tokens 0 → 34,304 by
turn 4 (https://github.com/Wei-Shaw/sub2api/pull/8464); full pricing matrix
at model-add time (https://github.com/Wei-Shaw/sub2api/pull/8364); durable
outbox reworked after a production incident
(https://github.com/Wei-Shaw/sub2api/pull/8468); least-privilege admin keys
(https://github.com/Wei-Shaw/sub2api/pull/8476).

Citation corrected 2026-11-08 per research audit: this list previously
opened with "quota-exhaustion semantics — final non-retryable usage-limit
errors carrying the real reset time" cited to
`sub2api/pull/7239`, but that PR fixes a registration promo-code field; the
described behaviour is CLIProxyAPI's
(https://github.com/router-for-me/CLIProxyAPI/pull/5978, which parses the
recovery time out of a 429 body).

**Gaps vs llm-proxy.** Background OAuth refresh pool
(`backend/internal/service/token_refresh_service.go:16-34`); upstream
client-identity emulation (`claude_code_version_sync_service.go:8-27`);
rule-based alerting engine (`ops_alert_evaluator_service.go:15-65`); egress
proxy pool with cycle-safe fallback (`proxy_fallback.go:7-40`); Redis leader
election for single-flight background jobs (`leader_lock.go:9-30`).

### api7/aisix

**Identity · Language · Form factor.** Apache-2.0 Rust AI gateway from the
original APISIX creators — 18-crate workspace (edition 2021, MSRV 1.93, axum
0.7), one static binary; OSS data plane beneath commercial AISIX
Enterprise/Cloud control planes; declarative `resources.yaml` (SIGHUP) or
etcd for clusters.

**Maintenance.** Exceptionally active — 92 of 100 harvested PRs merged
2026-10-16 → 2026-11-04; 183 e2e scenario files against real gateway
processes.

**Capabilities (code).** Six LB strategies incl. least-cost/least-latency
(TTFT EWMA) with retry budgets and configurable `fallback_on_statuses`
(`research-repos/api7__aisix/crates/aisix-core/src/models/routing.rs:46-63,301-332`);
two-phase rate limiting — RPM check-and-increment up front, TPM committed
after the response lands (`crates/aisix-ratelimit/src/lib.rs:1-16`);
ensemble models (fan-out + judge synthesis) (`crates/aisix-proxy/src/ensemble.rs:1-35`);
semantic routing over cached example embeddings
(`crates/aisix-proxy/src/semantic.rs:1-15`); guardrail chains incl.
quickjs-sandboxed custom scripts with a FAULT verdict distinguishable from
content blocks (`crates/aisix-guardrails/src/custom.rs:1-30`); MCP gateway
across revisions + OpenAPI-as-tools (`crates/aisix-mcp/src/gateway.rs:5-33`);
exact + semantic response cache and Anthropic `auto_prompt_caching`
breakpoint injection (`crates/aisix-cache/src/lib.rs:1-22`,
`crates/aisix-provider-anthropic/src/wire.rs:1322`); inbound OIDC/JWT with
JWKS rotation mapped onto API keys (`crates/aisix-proxy/src/jwt.rs:1-45`);
BPE usage-estimation fallback flagged `usage_estimated`
(`crates/aisix-proxy/src/token_estimate.rs:1-16`); ArcSwap lock-free config
snapshots (`crates/aisix-core/src/snapshot.rs:1-27`); resolver-level SSRF
guard with DNS pinning (`crates/aisix-gateway/src/ssrf.rs:1-14`).

**Pain points (issues).** The single largest defect class is scan-vs-forward
asymmetry — guardrails scan one text but forward another, silently
(https://github.com/api7/aisix/issues/1016, /issues/696, /issues/1026, /issues/1103, /issues/1106);
rate limits and policies silently unenforced on non-chat endpoints
(https://github.com/api7/aisix/issues/775, /issues/697, /issues/715); usage
blind spots on secondary endpoints (https://github.com/api7/aisix/issues/748, /issues/699, /issues/688);
an external researcher filed three P2 policy-bypass bugs in one day —
Host-header route shadowing, `forward_client_headers` globs leaking
credentials, dual MCP-name parsers (https://github.com/api7/aisix/issues/1401, /issues/1403, /issues/1404);
credential rotation lagging ~59 minutes behind key replacement
(https://github.com/api7/aisix/issues/779); Retry-After up to 86,000 s
uncapped (https://github.com/api7/aisix/issues/1157).

**Directions (pulls).** MCP per-user OAuth with keyless sign-in
(https://github.com/api7/aisix/pull/1394); native protocol fidelity beating
hub-and-spoke translation on Bedrock/Vertex
(https://github.com/api7/aisix/pull/1318); budget enforcement pushed to the
gateway so caps survive control-plane outages
(https://github.com/api7/aisix/pull/1366); OIDC `tag_claims` driving
chargeback attribution (https://github.com/api7/aisix/pull/1407); CI-enforced
per-route metric coverage (https://github.com/api7/aisix/pull/1367);
stateless session stickiness via tagged `encrypted_content`
(https://github.com/api7/aisix/pull/1332).

**Gaps vs llm-proxy.** Inbound OIDC/JWT caller auth; ensemble and semantic
routing; A2A agent gateway (`crates/aisix-proxy/src/a2a.rs:1-13`); etcd
multi-replica config; object-store NDJSON telemetry sink with
content-addressed keys (`crates/aisix-obs/src/sink/object_store.rs:1-18`);
explicit passthrough routes that still detect the LLM envelope
(`crates/aisix-proxy/src/passthrough_route.rs:1-22`); PROXY protocol v1/v2
(`crates/aisix-server/src/proxy_protocol.rs:1-12`); async video
submit→poll→credential-safe-fetch lifecycle (`crates/aisix-proxy/src/videos.rs:1-20`).

### katanemo/plano

**Identity · Language · Form factor.** Envoy-based AI-native data plane for
agentic apps: Rust sidecar (brightstaff) + proxy-wasm filters (llm_gateway,
prompt_gateway) + shared crates, driven by a Python `planoai` CLI that
renders Envoy config; hot path runs in sandboxed Wasm inside Envoy.

**Maintenance.** Active — clone at release 0.4.39 (2026-10-24), 60 merged
PRs May–Oct 2026.

**Capabilities (code).** Token-budget rate limiting keyed by a configurable
header selector (`research-repos/katanemo__plano/crates/common/src/ratelimit.rs:99-161`);
cost/latency-aware model ranking from live feeds (models.dev/DigitalOcean
pricing, Prometheus latency) (`crates/brightstaff/src/router/model_metrics.rs:84-232`);
session-affinity pinning with a routing budget — warm-cache loss priced,
paid switches capped at `max_switch_spend_pct` of a never-switch baseline
(`crates/brightstaff/src/handlers/llm/session_router.rs:176-216,333-430`);
automatic prompt-cache marker injection per (provider × model family ×
upstream API), threshold-guarded and idempotent against client markers
(`crates/brightstaff/src/handlers/llm/prompt_caching.rs:25-70`); Envoy
RING_HASH prefix affinity for KV-cache locality
(`config/envoy.template.yaml:276-281,1005-1008`); SSE framing with
line-holdback, 1 MiB cap, and UTF-8 boundary safety
(`crates/hermesllm/src/apis/streaming_shapes/sse_chunk_processor.rs:9-57`);
Agentic Signals behavioral-quality analyzer
(`crates/brightstaff/src/signals/mod.rs:1-19`).

**Pain points (issues).** Output-filter guardrails stateless per SSE chunk
and fail open — a secret split across two events passes
(https://github.com/katanemo/plano/issues/1036); multibyte UTF-8 split at
chunk boundaries silently truncated Anthropic streams in production
(https://github.com/katanemo/plano/issues/980); request-derived data can
panic the entire Envoy WASM VM (https://github.com/katanemo/plano/issues/999,
/issues/1030); session traces unattributable because session id comes only
from a Plano-internal header (https://github.com/katanemo/plano/issues/979);
pricing-feed degradation silently weakens the routing budget
(https://github.com/katanemo/plano/issues/992); switch-cost math ignores
cross-model tokenizer variance (https://github.com/katanemo/plano/issues/991).

**Directions (pulls).** The cost-aware switching arc
(https://github.com/katanemo/plano/pull/982, /pull/1016); server-side turn
semantics for coding agents (`route_on_user_only` replays the prior decision
on tool-result turns, https://github.com/katanemo/plano/pull/1015);
Responses SSE byte-level framing hardening
(https://github.com/katanemo/plano/pull/988); usage-metering completeness —
`Accept-Encoding: identity` forced and `stream_options.include_usage`
injected so usage parses (https://github.com/katanemo/plano/pull/1043).

**Gaps vs llm-proxy.** Agent orchestration listeners
(`crates/brightstaff/src/handlers/agents/selector.rs:21-26`); Agentic
Signals OTel taxonomy; per-session switch-spend budget; cache-marker
auto-injection; replica-level prefix-hash affinity; shared Redis session
state; live external metrics feeds; stateful Responses emulation with
PostgreSQL persistence (`crates/brightstaff/src/state/postgresql.rs:10-16`);
PostHog product-analytics exporter
(`crates/brightstaff/src/tracing/posthog_exporter.rs`).

### mozilla-ai/otari

**Identity · Language · Form factor.** Python 3.13 FastAPI modular monolith
(gateway + management API + React dashboard in one process) delegating to
any-llm for 40+ providers; PostgreSQL in compose, SQLite for dev, 142
alembic migrations; Apache-2.0 open core with a hosted otari.ai edition.

**Maintenance.** Extremely active — v0.36.0 → v0.38.0 released
2026-11-03/04; 73 of the last 100 PRs merged within three days.

**Capabilities (code).** Routing compiled to an ordered plan where every
dropped candidate carries a machine-readable reason
(`research-repos/mozilla-ai__otari/src/gateway/services/routing/compiler.py:118-150`);
attempt walker with retryable classification, per-attempt credential
re-binding, and tool-loop lock-in (fallback stops after the first assistant
message because provider-specific tool ids cannot be replayed)
(`src/gateway/api/routes/_attempts.py:235-314,414-474`); pluggable routers
under a hard 5 s deadline that fall back to the default target
(`src/gateway/services/routing/decide.py:49-52`); exact sliding-window log
rate limiting with cost-carrying entries admitted then settled to true cost
(`src/gateway/rate_limit.py:63-124`); reserve-then-settle budgets in BIGINT
with `BudgetReservation` rows swept back on every axis
(`src/gateway/models/budgets.py:200-296`); Idempotency-Key claims before
budget reservation, replays stored encrypted with request-shaping headers
in the fingerprint (`src/gateway/api/routes/_idempotency.py:50-79`);
provider-neutral cost math reconciling OpenAI vs Anthropic cache-token
conventions with `require_pricing` refusal
(`src/gateway/core/metered_pricing.py:17-35`); format-neutral SSE plumbing
with keepalives and settlement decoupled from streaming
(`src/gateway/streaming.py:34-238`).

**Pain points (issues).** Guardrail in block mode with `on_unavailable:
block` served the request when the external guardrail returned an
inconclusive verdict (https://github.com/mozilla-ai/otari/issues/2788);
fallback candidates are not interchangeable — Bedrock 400s a field the
first candidate accepted (https://github.com/mozilla-ai/otari/issues/2790);
every usage/budget read API requires the master key hosted tenants never
have (https://github.com/mozilla-ai/otari/issues/2863 and
/issues/2792 — apps must poll one request at a time or discover limits by
triggering refusals); the code-execution
tool is billed but surfaces no cost (https://github.com/mozilla-ai/otari/issues/2864);
models.dev catalog half-used — model type guessed from names, mislabeling
96 transcription models as chat (https://github.com/mozilla-ai/otari/issues/2782).

**Directions (pulls).** models.dev as canonical catalog, persisted and
refusal-on-unreadable (https://github.com/mozilla-ai/otari/pull/2771, /pull/2787);
pricing schema converging on models.dev's exact unit vocabulary
(https://github.com/mozilla-ai/otari/pull/2812, /pull/2820); tool calls as
first-class billable units (https://github.com/mozilla-ai/otari/pull/2831);
model-free tool endpoints (`/api/v1/code/execute`, `/api/v1/fetch`)
(https://github.com/mozilla-ai/otari/pull/2803); learned kNN routers removed
for privacy after teaching stored tenant prompts in plaintext
(https://github.com/mozilla-ai/otari/pull/2829); estimated usage row when a
client abandons a stream (https://github.com/mozilla-ai/otari/pull/2869).

**Gaps vs llm-proxy.** Agent Guardrails laptop-side CLI enforcing
repo-committed gates at PreToolUse/Stop (`docs/agent-guardrails.md:3-91`);
workload-identity federation — OIDC identity tokens + jwt-bearer exchange
compatible with Claude Code federation env vars (`docs/workload-identity.md:21-48`);
external usage ingestion (`docs/external-usage.md:23-37`); OTLP ingest
booking tokens from exported traces (`docs/external-usage.md:86-116`);
models.dev default pricing with timestamp-effective rates
(`src/gateway/services/pricing_refresh_service.py:52-53`); replica-shared
Redis rate-limit store (`src/gateway/adapters/rate_limit_store_adapter.py:27-63`).

### musistudio/claude-code-router

**Identity · Language · Form factor.** TypeScript/Node pnpm monorepo
(@musistudio/claude-code-router v3.1.3): Electron desktop app, CLI, and
Docker — a local gateway/control-plane for coding-agent CLIs (Claude Code,
Codex, Grok, Kimi, OpenCode, Pi, ZCode) with a web management UI; the HTTP
core is an external plugin runtime supervised as a child process.

**Maintenance.** Active but backlogged — last merge 2026-10-10 with 55 open
PRs queued; the harvest window shows deep, well-investigated bug reports
against 3.1.1–3.1.3.

**Capabilities (code).** Config compiled up front with per-rule
diagnostics — a bad rule goes inactive while the rest keep serving
(`research-repos/musistudio__claude-code-router/packages/core/src/routing/config-compiler.ts:46-70`);
fallback as a precomputed execution plan (bounded same-model retry or
ordered model chain) driven by a status-code classifier with retry-after
clamped to 60 s (`packages/core/src/routing/execution-plan.ts:24-45`,
`failure-classifier.ts:10-31`); user-supplied JS routing rules in a
sandboxed worker with code generation disabled and byte budgets
(`packages/core/src/routing/route-script-worker.ts:12-58`); upstream
credential pools with per-credential cooldown and spillover
(`packages/core/src/providers/credential-pool.ts:16-65`); client API keys
with expiry, constant-time comparison, and pre-request reservations across
rpm/rph/rpd/tpm/tph/tpd windows estimated from the request body
(`packages/core/src/gateway/auth/api-key-authorizer.ts:81-162`); five client
protocol surfaces incl. Gemini (`packages/core/src/routing/protocol-endpoints.ts:3-21`);
SQLite usage ledger with cache-token splits and virtual-model attribution
(`packages/core/src/usage/store.ts:40-72`); stream-experience metering
(TTFS/TTFT, inter-event gap p95) (`packages/core/src/observability/stream-experience.ts:27-90`).

**Pain points (issues).** Fallback chains silently didn't execute — an
ingress header overrode the per-attempt body on every hop
(https://github.com/musistudio/claude-code-router/issues/1850); fallback
outcomes misreported on every surface — dashboards showed 65% success when
clients saw 98.4% (https://github.com/musistudio/claude-code-router/issues/1870, /issues/1875, /issues/1927);
the Node process died under real Claude Code traffic (2–20 MB bodies) via
unhandled error events and log-spool OOMs (https://github.com/musistudio/claude-code-router/issues/1846, /issues/1867);
translation dropped `max_tokens` and vendor `thinking` into upstream 400s
(https://github.com/musistudio/claude-code-router/issues/1843, /issues/1937);
a truncated upstream stream surfaced as a bare socket destroy with no
terminal event (https://github.com/musistudio/claude-code-router/issues/1917);
Codex OAuth import structurally broken by scope demands and per-profile
refresh-token reuse (https://github.com/musistudio/claude-code-router/issues/1941, /issues/1879);
same-provider credential failover is unclear rather than absent — users
model every API
key as a separate provider, and the filed feature request asks for pool
rotation on 429/401/403/5xx/timeout with Retry-After awareness and
quarantine of persistently unauthorized keys
(https://github.com/musistudio/claude-code-router/issues/1923, which
describes the failover story as unclear, not absent — corrected 2026-11-08
per research audit).

**Directions (pulls).** Multi-attempt diagnosability — per-attempt
`[stage|status]` summaries appended to `error.message` because Claude Code
renders only that (https://github.com/musistudio/claude-code-router/pull/1773);
prompt-cache affinity headers on Codex legs
(https://github.com/musistudio/claude-code-router/pull/1866); `/compact`
tool_result preservation (https://github.com/musistudio/claude-code-router/pull/1824);
pricing keyed to the routed physical model
(https://github.com/musistudio/claude-code-router/pull/1790); request-log
prune ordering fixed after 13 GB of orphaned bodies
(https://github.com/musistudio/claude-code-router/pull/1931).

**Gaps vs llm-proxy.** Declarative + scriptable routing policy engine
(`docs/.../routing.md:64-135`); agent-aware subagent model selection via a
`<CCR-SUBAGENT-MODEL>` tag; Fusion capability-composition virtual models
(vision pre-digestion, hosted search); ToolHub lazy MCP tool loading
(`packages/core/src/mcp/toolhub-mcp.ts:185-352`); interactive OAuth login
import from local agent CLIs (`docs/.../providers.md:8-78`); bundled
generated price/context catalog (`packages/core/models.json`); agent launch
profiles that write the client's own config (`docs/.../profiles.md:10-37`).

### raine/claude-code-proxy

**Identity · Language · Form factor.** Single-crate Rust binary (edition
2024, axum 0.8, ~17.5k LoC, v0.1.48) presenting an Anthropic-Messages
endpoint and translating for consumer-subscription backends: Codex/ChatGPT,
Kimi, Grok, Cursor Agent, OpenCode Go. One Provider trait with per-provider
modules rather than a hub-and-spoke core.

**Maintenance.** Very active — weekly-ish releases (v0.1.48 on 2026-11-01),
62 of 100 harvested PRs merged Jul–Oct 2026.

**Capabilities (code).** Per-session provider affinity from validated
session headers in a bounded LRU (`research-repos/raine__claude-code-proxy/src/session.rs:49-115`);
stateless reasoning replay — upstream encrypted reasoning base64-packed
into a synthetic `ccp:codex:v1:` thinking signature the client echoes back
(`src/providers/codex/translate/reasoning_signature.rs:26-60`); dual-transport
upstream with a pooled Responses-over-WebSocket protocol plus automatic HTTP
degradation (`src/providers/codex/websocket.rs:40-58`); budget-aware retry
on 429/5xx with Retry-After honored and in-band SSE errors also classified
(`src/retry.rs:4-41`); full consumer OAuth — PKCE browser login, device
code, single-flight refresh, keychain-encrypted token files
(`src/providers/codex/auth/manager.rs:16-105`); live ratatui monitor plus a
loopback JSON `/monitor` endpoint for remote dashboards
(`src/monitor.rs:71-136`, `src/monitor/remote.rs:13-52`); opt-in traffic
capture with 15-key redaction (`src/traffic.rs:44-105`); auxiliary-request
rerouting (Claude Code's background security-review requests detected and
routed to a cheap model) (`src/server.rs:52-108`); subscription quota
telemetry parsed out-of-band into the usage block (`src/providers/codex/client.rs:1787-1796`).

**Pain points (issues).** No per-backend tool-schema sanitization —
Codex/Grok hard-400 on Claude Code tool fields
(https://github.com/raine/claude-code-proxy/issues/142, /issues/109);
in-memory session state mis-keyed — subagents sharing a parent session id
destroyed the main conversation's state (74% 409 rate measured)
(https://github.com/raine/claude-code-proxy/issues/163, /issues/131);
exhausted 503 without `x-should-retry: false` stacked Claude Code's retry
loop on the proxy's own — one turn hung ~30 minutes
(https://github.com/raine/claude-code-proxy/issues/174); the model registry
is hardcoded so every new model needs a release
(https://github.com/raine/claude-code-proxy/issues/198, /issues/169);
subscription upstreams change protocol details without notice and break the
proxy same-day (https://github.com/raine/claude-code-proxy/issues/137, /issues/180).

**Directions (pulls).** "Semantic commit point" unification so retries only
replay before visible output (https://github.com/raine/claude-code-proxy/pull/103);
`x-should-retry: false` terminality signals (https://github.com/raine/claude-code-proxy/pull/175);
quota/reset-clock surfacing into the Anthropic usage block
(https://github.com/raine/claude-code-proxy/pull/149, /pull/139); runtime
model catalog from the vendor CLI's own cache
(https://github.com/raine/claude-code-proxy/pull/130); 64 MiB body limit
because image-heavy sessions re-send full history every turn
(https://github.com/raine/claude-code-proxy/pull/152); `SSL_CERT_FILE` /
native trust roots (https://github.com/raine/claude-code-proxy/pull/143).

**Gaps vs llm-proxy.** OAuth subscription login; server-side compaction
with a retained-token budget (`src/providers/codex/compaction.rs:16-59`);
incremental turn continuation via `previous_response_id` + semantic prefix
validation (`src/providers/codex/continuation.rs:8-62`); encrypted-reasoning
replay (llm-proxy's `Thinking.signature` is passthrough-only — a grep for
`encrypted_content` across llm-proxy crates returns nothing); upstream
WebSocket transport; live attachable monitor; XML tool-call synthesis for
models without native function calling (`src/providers/cursor/tool_use_xml.rs:1-50`);
corporate TLS trust-store support. (Parenthetical corrected 2026-11-08 per
research audit: llm-proxy did not build reqwest with bundled roots only at
the audited commit — reqwest's default features were on, so native-tls and
the OS trust store were active and corporate roots already worked. Plan 03
later disabled the defaults, making rustls with bundled webpki roots the
single stack in `crates/llm-proxy-transport/Cargo.toml`, which is what
makes trust-store support a real gap.)

### router-for-me/CLIProxyAPI

**Identity · Language · Form factor.** Go 1.26 single-binary LLM gateway
(module v8, v8.0.54) fronting CLI-subscription OAuth accounts (Codex,
Claude Code, Gemini/Antigravity, Grok, Kimi, Meta, Devin) plus API-key
providers behind OpenAI/Anthropic/Gemini-compatible endpoints; ships an
embedding SDK, c-shared plugin host, bubbletea TUI, and web panel.

**Maintenance.** Extremely active — 22 PRs merged 2026-10-27 → 2026-11-04;
issues closed within a day.

**Capabilities (code).** Multi-strategy credential scheduler (round-robin,
fill-first, smooth WRR, plugin strategies) with cooldown queues
(`research-repos/router-for-me__CLIProxyAPI/sdk/cliproxy/auth/scheduler.go:21-24`);
request-path retry/failover conductor with per-credential error rules
(stop / stop-and-cooldown / continue) (`sdk/cliproxy/auth/conductor_execution.go:123-183`);
session-sticky routing keyed from Claude/Codex session headers,
`prompt_cache_key`, and conversation IDs, with subagent inheritance for
KV-cache locality (`sdk/cliproxy/auth/session_cache.go:31-45`);
in-band overload failover via stream bootstrap buffering — handshake/heartbeat
frames held within a 48-frame/1 MiB budget so a `server_is_overloaded`
rejection inside an HTTP 200 stream can still be failed over before
downstream headers commit (`internal/runtime/executor/codex_executor_stream.go:142-200`);
single-port protocol multiplexer (Redis RESP SUBSCRIBE for live usage)
(`internal/api/protocol_multiplexer.go:125-207`); realtime WebRTC voice
relay (`internal/client/codex/live/media.go:24-58`); uTLS client
impersonation and CLI device profiles (`internal/runtime/executor/helps/utls_client.go:27-31`);
pairwise translator registry rather than hub-and-spoke
(`sdk/translator/registry.go:29-48`); declarative gjson/sjson payload
rewriting (`config.example.yaml:270-336`); debounced hot reload
(`internal/watcher/config_reload.go:1-43`); management API v8 over the whole
config tree (`internal/api/server_management_v8.go:13-69`).

**Pain points (issues).** Cross-dialect translation loses semantics —
omitted Gemini roles, split parallel tool batches, dropped Devin reasoning
(https://github.com/router-for-me/CLIProxyAPI/issues/7186, /issues/7177, /issues/7228);
coarse error classification takes whole credentials out of service — a
model-scoped 429 cooled an entire Claude credential ~6 days
(https://github.com/router-for-me/CLIProxyAPI/issues/7197, /issues/7241);
OAuth refresh destroys or spins credentials when the store is read-only
(https://github.com/router-for-me/CLIProxyAPI/issues/7214); mid-stream
failures unrecoverable once bytes were forwarded
(https://github.com/router-for-me/CLIProxyAPI/issues/7169); usage conflates
unknown with zero — cache fields absent upstream log as 0% hit rate
(https://github.com/router-for-me/CLIProxyAPI/issues/7157).

**Directions (pulls).** Quota-reset-aware `soonest-reset` scheduling from
passively observed quota headers (https://github.com/router-for-me/CLIProxyAPI/pull/7246);
final non-retryable usage-limit errors carrying reset times
(https://github.com/router-for-me/CLIProxyAPI/pull/7239); per-credential
connection pooling (https://github.com/router-for-me/CLIProxyAPI/pull/7101);
fine-grained cache-bucket usage accounting (https://github.com/router-for-me/CLIProxyAPI/pull/7074);
embeddability hooks (https://github.com/router-for-me/CLIProxyAPI/pull/7124);
per-key model policies (https://github.com/router-for-me/CLIProxyAPI/pull/7117).

**Gaps vs llm-proxy.** OAuth login flows for CLI-subscription providers
(`internal/auth/`); shipped credential-pool failover with persisted cooldown
state; session-affinity sticky routing; a management write API (llm-proxy's
admin API is read-only usage); hot reload; client impersonation; a dedicated
reasoning/thinking translation engine converting effort levels to provider
budgets (`internal/thinking/convert.go:8-45`); cache-preserving replay
caches; the payload rules engine; the plugin platform + public SDK.

### smg-project/smg

**Identity · Language · Form factor.** Apache-2.0 Rust workspace (26
members; `model_gateway` binary `smg` plus protocol/parser/router/mesh/wasm
crates and Python/Go bindings; axum 0.8 + tonic 0.14) — an engine-agnostic
model-routing gateway for self-hosted engines (vLLM, SGLang, TensorRT-LLM,
MLX) plus cloud providers; binary, Docker, Helm, PyPI.

**Maintenance.** Extremely active — HEAD 2026-11-05; 67 of 100 harvested
PRs merged in November 2026; nightly fuzz triage and CI-health bots.

**Capabilities (code).** Ten LB policies incl. cache-aware routing that
mirrors worker KV-cache radix trees with cache namespaces
(`research-repos/smg-project__smg/model_gateway/src/policies/mod.rs:19-42`,
`src/policies/cache_aware.rs:1-60`); priority-aware admission scheduler with
preemption and distinct rejections (`src/middleware/scheduler/engine.rs:30-50`);
retry + per-endpoint breakers with an attempt ledger charging a worker at
most one failure per request (`crates/external_router/src/retry.rs:96-104`);
per-tenant reserve/settle token limits (`src/rate_limit/mod.rs:1-9`);
21 tool-call parsers and 16 reasoning parsers with auto model detection
(`crates/tool_parser/src/factory.rs:447`); WASM plugin middleware over a
runtime REST API (`model_gateway/src/wasm/route.rs:1-10`); gossip mesh HA
with CRDT state replication and mTLS (`crates/mesh/src/lib.rs:1-24`);
OTel + 340 Prometheus sites (`src/observability/otel_trace.rs:1-4`);
Kubernetes/Consul/Slurm/file discovery (`src/service_discovery/mod.rs:1-32`);
two-level cached tokenizer with chat templates (`crates/tokenizer/src/lib.rs:20-35`).

**Pain points (issues).** Hardcoded multimodal image-count limits reject
requests the model would serve (https://github.com/smg-project/smg/issues/5044);
a streaming reasoning parser replays the whole already-streamed buffer as
content — clients receive the reply twice at ~1% of long-session messages
(https://github.com/smg-project/smg/issues/5038); the suggested mitigation
knob is dead config for that parser (same issue); nightly fuzz found a
JSON-prefix breach class in the pythonic tool-call parser
(https://github.com/smg-project/smg/issues/5056).

**Directions (pulls).** Hand-written one-pass SSE serializers replacing
serde on every stream family, byte-pinned against fixtures
(https://github.com/smg-project/smg/pull/4987, /pull/5043, /pull/5031, /pull/5035);
load telemetry moving from polling to push
(https://github.com/smg-project/smg/pull/4971); metrics-correctness pass
closing per-endpoint coverage (https://github.com/smg-project/smg/pull/5017);
MCP tool execution first-class in the Responses path
(https://github.com/smg-project/smg/pull/5045); chat-template byte parity
with the serving engine (https://github.com/smg-project/smg/pull/5052).

**Gaps vs llm-proxy.** Tool-call/reasoning text parsers for self-hosted
models; KV-state-aware cache routing; WASM runtime-extension API;
infrastructure service discovery; multi-node mesh HA; pluggable chat-history
storage with migrations; priority admission with preemption
(`src/middleware/scheduler/admission.rs:1`); a real tokenizer stack —
llm-proxy's offline tiktoken with a ~4-chars/token fallback silently
mismatches non-OpenAI vocabularies; OIDC/JWKS caller auth
(`crates/auth/src/jwks.rs:1-13`); MCP approval engine with audit ring
(`crates/mcp/src/approval/policy.rs:1-22`).

### zilliztech/GPTCache

**Identity · Language · Form factor.** Python library (pip `gptcache`) plus
an optional FastAPI server implementing a modular semantic cache for LLM
responses — pluggable scalar/vector stores, embedders, similarity
evaluators, eviction policies. MIT, from Zilliz.

**Maintenance.** Caretaker-mode with a Sept 2026 revival — v0.1.44 shipped
Aug 2024, then collaborator xiaofan-luan merged adapters + an LLM-judged
evaluator + a reuse benchmark on 2026-09-22 and opened the v0.2.0 release
PR; 41 of 56 harvested true issues sit open, some unanswered since 2023.

**Capabilities (code).** A composable `adapt()` pipeline (pre-process →
embed → ANN search → re-rank → post-process → save) where each stage is an
injected callable (`research-repos/zilliztech__GPTCache/gptcache/adapter/adapter.py:12-107`);
every evaluator declares its score range so one user threshold works across
scorers (`gptcache/similarity_evaluation/similarity_evaluation.py:21-41`);
streaming cached without changing the stream contract — hits replay a
synthetic three-chunk SSE sequence (`gptcache/adapter/openai.py:78-106`);
session-scoped isolation with an anti-repeat hit filter
(`gptcache/session.py:48-76`); two-tier eviction with vector-index rebuild
(`gptcache/manager/eviction_manager.py:11-46`); two-store consistency
self-heal under `data_check` (`gptcache/adapter/adapter.py:122-131`);
L1/L2 chaining via `next_cache` (`gptcache/core.py:44,80`); template-aware
cache keys (`gptcache/processor/pre.py:77-135`); LLM-as-verifier hit gate
(`gptcache/processor/post.py:93-205`).

**Pain points (issues).** The server answered the first question's answer
for every subsequent unrelated question under OpenAI embeddings — ~100%
false hits (https://github.com/zilliztech/GPTCache/issues/612); unsafe YAML
loading gave config-file RCE, reported twice and left open
(https://github.com/zilliztech/GPTCache/issues/676, /issues/687);
`BufferedReader.peek()` hashed only ~8 KiB of files so different files
collide — cross-user disclosure (https://github.com/zilliztech/GPTCache/issues/684);
a timing side channel distinguishes hits from misses at 85.3% accuracy
(https://github.com/zilliztech/GPTCache/issues/652); multi-node deployments
unsupported (https://github.com/zilliztech/GPTCache/issues/666); no
read-only mode (https://github.com/zilliztech/GPTCache/issues/656).

**Directions (pulls).** The revival: Anthropic Messages + modern OpenAI
adapters plus JEV (https://github.com/zilliztech/GPTCache/pull/701);
cache-hit judgment moving from embedding similarity to an LLM judge asking
five atomic reuse questions at a calibrated 0.70 threshold
(https://github.com/zilliztech/GPTCache/pull/701,
`gptcache/similarity_evaluation/jev.py:186-297`); a frozen 1,536-case
cache-reuse benchmark — JEV 99.7% precision vs 42.2% for the default
reranker at 0.80 (https://github.com/zilliztech/GPTCache/pull/702);
cost-aware eviction research PRs open (W-TinyLFU, GDSF, ARC, CARMA)
(https://github.com/zilliztech/GPTCache/pull/680, /pull/689, /pull/692, /pull/696).

**Gaps vs llm-proxy.** LLM-judged reuse gating and a public hit-policy
correctness dataset; per-request cache controls on the wire (`cache_skip`,
temperature-derived bypass, `cache_factor`); session anti-repeat policy;
pluggable key preprocessing separating store key from match key; multi-tier
chaining; two-store consistency self-heal; hit-audit accounting with
client-visible token savings
(`research-repos/zilliztech__GPTCache/gptcache/adapter/openai.py:111-122`).

### eliasbiondo/antirion

**Identity · Language · Form factor.** Apache-2.0 Rust workspace (~28
crates, edition 2024, MSRV 1.98, v0.1.1) shipping one binary with an
embedded Next.js admin console; standalone (SQLite) or clustered
(PostgreSQL + Valkey); positioned as "the fastest open-source AI gateway"
with a reproducible benchmark harness vs Bifrost and LiteLLM.

**Maintenance.** Young and single-author but disciplined — public since
2026-10-17, nightly supply-chain and per-parser fuzzing CI gating every PR;
human reports triaged within days.

**Capabilities (code).** Routing trees compiled to an arena at config time —
nine node kinds incl. weighted, bandit, least-latency, conditional,
classifier, tier (`research-repos/eliasbiondo__antirion/crates/antirion-router/src/tree/mod.rs:80-152`);
selection state (PeakEWMA latency, success EWMA, in-flight) keyed by stable
identities that survive config swaps (`crates/antirion-router/src/state.rs:1-38`);
request hedging plus a cross-request retry budget (ratio 0.2 over 10 s)
(`crates/antirion-resilience/src/hedging.rs:1-4`, `src/budget.rs:1-32`);
exact integer GCRA mirrored in Lua with a differential test
(`crates/antirion-limits/src/math.rs:1-70`); RFC 8785 JCS canonical cache
keys (`crates/antirion-cache/src/jcs.rs:1-3`); single-flight cache misses
via watch channels (`crates/antirion-cache/src/flights.rs:1-12`); semantic
cache with a negation lexicon so "don't enable X" cannot match "enable X"
(`crates/antirion-cache/src/semantic.rs:1-25`); guardrails with
format-preserving redaction (`crates/antirion-guardrails/src/preserve.rs:1-19`);
MCP gateway bridging both protocol eras with stateless XChaCha20-Poly1305
session tokens (`crates/antirion-mcp/src/token.rs:1-7`); WASI 0.2 plugin
extensions sha256-pinned (`crates/antirion-ext/src/wasm/mod.rs:1-13`);
three-class egress SSRF catalogue (`crates/antirion-egress/src/ssrf.rs:1-23`);
thread-per-core data plane with a workspace `deny(unsafe_code)`
(`crates/antirion-server/src/threads.rs:1-38`); disk-spooled idempotent
spend ledger — a full spool refuses budgeted requests with 503 rather than
losing money (`crates/antirion-store/src/lib.rs:3`).

**Pain points (issues).** Nightly cargo-deny/audit failures outpace
remediation — 12 bot issues Oct 21–Nov 4, five open with no triage
(https://github.com/eliasbiondo/antirion/issues/63); a fuzz crash found
Nov 1 sits open and gates the release workflow
(https://github.com/eliasbiondo/antirion/issues/60); retry predicates
cannot branch on response payload values
(https://github.com/eliasbiondo/antirion/issues/47); provider model
listing stops at a hard 20-page cap with no partial-catalog signal
(https://github.com/eliasbiondo/antirion/issues/41).

**Directions (pulls).** A typed decisions endpoint (`POST /v1/systemone`)
as a new endpoint family where routing defaults to refusing the unknown
operation (https://github.com/eliasbiondo/antirion/pull/48); supply-chain
sweeps as merged maintenance work (https://github.com/eliasbiondo/antirion/pull/34, /pull/44);
build-time gzip embedding of the console (https://github.com/eliasbiondo/antirion/pull/45).

**Gaps vs llm-proxy.** A2A agent proxy (`crates/antirion-mcp/src/a2a.rs:1-5`);
WASM plugin extensions; dry-run simulation endpoints for routes and limits
(`crates/antirion-admin/src/simulate.rs:1`); shadow-traffic mirroring with
measured divergence (`crates/antirion-router/src/shadow.rs:1-28`);
multi-node cluster mode; enterprise SSO/SAML/SCIM/RBAC; tenant erasure as a
store operation (`crates/antirion-store/src/erasure.rs:1-16`); signed
attested releases (SBOM + Sigstore + SLSA L3, `.github/workflows/release.yml:2-3`);
in-repo fuzzing of every untrusted parser (`fuzz/fuzz_targets/`); Gemini
and Bedrock client dialects (`crates/antirion-pipeline/src/accept.rs:382-391`).

### deanxv/done-hub

**Identity · Language · Form factor.** Go 1.26 monolith (Gin + GORM +
embedded React 18) forked from one-hub/one-api — a full commercial LLM
gateway with ~45 upstream adapters incl. dedicated client-reverse-proxy
channels (Claude Code, Codex, Gemini CLI, Antigravity), payments, and an
MCP server; single Docker image, MySQL/PostgreSQL/SQLite.

**Maintenance.** Active (last commit 2026-10-27); community-PR intake is
weekend-cadence with maintainer-side equivalents landing directly.

**Capabilities (code).** Failover loop capping retries at available-channel
count, accumulating skip ids, and freezing a failed (channel, model) with a
cooldown preferring the upstream's exact reset time
(`research-repos/deanxv__done-hub/relay/main.go:74-176`,
`model/balancer.go:95-163`); layered admission control — per-token model
allowlists enforced pre-routing (`CheckLimitModel` over
`token.setting.limits.models[]`, `relay/common.go:72-111`), token-group
validation, a per-user-group Redis API limiter, dual user/IP rate buckets,
and CIDR whitelisting (`middleware/distributor.go:22-70`,
`middleware/rate-limit.go:29-113`); Redis sticky sessions keyed per CLI-client
protocol (Claude `metadata.user_id`, Gemini UA+IP, Codex `session_id`)
(`model/balancer.go:194-309`); raw-upstream passthrough preserving unknown
fields and upstream rate-limit headers while final errors are collapsed to
hide upstream identity but preserve 429 for client backoff
(`relay/common.go:416-468,796-877`); SSE `::PING` heartbeats after idle and
post-disconnect drain so usage/billing stays accurate
(`relay/relay_util/heartbeat.go:14-175`); auto channel disabling with
keyword exemptions (middleware + `relay/common.go:764-792`); OAuth account
channels with auto-refresh persisted back to the database
(`providers/claudecode/base.go:56-89`); the gateway itself as an MCP server
with a per-user dashboard tool (`mcp/server.go:14-75`); scheduled price
sync from models.dev (`model/pricing_modelsdev.go:18-108`).

**Pain points (issues).** Strictly-typed unmarshalling breaks streams when
real clients send object-vs-string fields (https://github.com/deanxv/done-hub/issues/58, /issues/25);
Codex/Responses compatibility churns release-over-release
(https://github.com/deanxv/done-hub/issues/54); upstream
Claude-Code-only providers fingerprint and reject relayed requests —
byte-exact passthrough requested (https://github.com/deanxv/done-hub/issues/16);
cache-token accounting lossy across dialects (https://github.com/deanxv/done-hub/issues/51, /issues/55);
no request/response observability of what was actually sent upstream
(https://github.com/deanxv/done-hub/issues/32); no multi-key channels — one
key per channel forces duplication (https://github.com/deanxv/done-hub/issues/57);
no self-hosted/personal mode with the commercial layer off
(https://github.com/deanxv/done-hub/issues/14).

**Directions (pulls).** Wire-exact Responses compatibility — removing
`omitempty` on `usage.input_tokens_details.cached_tokens` fixed Codex CLI
parsing (https://github.com/deanxv/done-hub/pull/39); Anthropic cache fields
parsed on OpenAI-compatible channels (https://github.com/deanxv/done-hub/pull/52);
a JSON-Schema→Gemini-Schema structural converter because Antigravity
silently ignores `parametersJsonSchema` (https://github.com/deanxv/done-hub/pull/59);
nested `remove_params` paths for per-channel body surgery
(https://github.com/deanxv/done-hub/pull/44).

**Gaps vs llm-proxy.** Gemini-native client routes
(`router/relay-router.go:120-127`); consumer-OAuth CLI channels with
impersonation headers (`providers/codex/base.go:28-148`); Gemini 3
`thoughtSignature` recovery via the official sentinel + one idempotent
retry (`relay/common.go:735-759`, issue #28); cross-dialect cached-token
normalization into one Usage type (`types/common.go:10-135`); channel-level
parameter policy (`providers/base/common.go:608-705`); client-header
passthrough rules (`providers/base/common.go:160-266`);
`context.WithoutCancel` disconnect settlement (`providers/base/common.go:276-301`);
no-cache
headers on the admin surface after a CDN-cached authorization bypass
(`middleware/cache.go:8-16`); untranslated passthrough relay for whole
endpoint families (`router/relay-router.go:53-65`). (`{model}` URL
templating was removed from this gap list 2026-11-08 per research audit:
llm-proxy already implements it, `crates/llm-proxy-provider/src/adapter/mod.rs:283-317`,
used by `providers/opencode-zen.toml.example:25`.)

### messkan/prompt-cache

**Identity · Language · Form factor.** Go 1.24 single-binary semantic cache
(Gin + BadgerDB + in-memory ANN) exposing an OpenAI-compatible
`/v1/chat/completions` that serves near-duplicate prompts from an
embedding-similarity cache; OpenAI, Mistral, and Claude+Voyage providers.

**Maintenance.** Maintainer-driven, moderate — v0.4.0 (2026-04-25), last
merge 2026-08-18; two external PRs (ANN staleness fix, MiniMax provider)
open and uncommented since Jul/Sep 2026.

**Capabilities (code).** Two-threshold matching — ≥0.70 hit, <0.30 miss,
gray zone deferred to an LLM "semantic judge"
(`research-repos/messkan__prompt-cache/internal/semantic/semantic.go:256-336`);
SSE transparency both ways — misses pipe verbatim while reassembling,
hits re-synthesize a fresh chunk sequence
(`internal/semantic/providers.go:40-111`, `cmd/api/main.go:198-273`);
cross-protocol translation inside the cache provider
(`internal/semantic/providers.go:815-1103`); per-caller namespace
partitioning where namespaced lookups use filtered scans so candidates
never cross a partition (`internal/cachecontext/namespace.go:10-83`);
persistent two-tier cache with an hourly TTL janitor
(`internal/cache/cache.go:76-143`); runtime threshold reconfiguration and
provider hot-swap (`cmd/api/main.go:383-443`); cache warming API
(`cmd/api/main.go:492-560`); retryable upstream client for 429/5xx
(`internal/http/client.go:74-146`); `X-Cache`/`X-Similarity-Score` response
headers + Prometheus (`internal/metrics/metrics.go:86-186`); multimodal
cache-key hygiene via media digests (`cmd/api/content.go:40-105`).

**Pain points (issues).** Cache invalidation is TTL-only so RAG users get
stale answers when the corpus changes (https://github.com/messkan/prompt-cache/issues/8);
no multi-tenancy — one shared token, and the README itself warns "semantic
similarity is not an authorization mechanism" (https://github.com/messkan/prompt-cache/issues/9);
an adopter asked when v1.0.0 ships before committing — the owner replied
within days, so this is not an unanswered months-old request (corrected
2026-11-08 per research audit)
(https://github.com/messkan/prompt-cache/issues/16); OpenShift non-root
execution broken at release time (https://github.com/messkan/prompt-cache/issues/10).

**Directions (pulls).** Namespace isolation landed claims-first — docs
disclosure before the mechanism, then tests
(https://github.com/messkan/prompt-cache/pull/13, /pull/14, /pull/15);
multimodal content parts with media digests in the cache key
(https://github.com/messkan/prompt-cache/pull/17); container hardening for
arbitrary-UID security contexts (https://github.com/messkan/prompt-cache/pull/12).

**Gaps vs llm-proxy.** A two-threshold semantic cache with an LLM gray-zone
verifier; persistent TTL cache with LRU; cache-aware streaming playback;
multimodal cache-key derivation; per-tenant partitioning of shared state;
mutating admin API (`PATCH /v1/config`); cache lifecycle admin
(list/delete/warm); Prometheus exposition; request-path HTTP retry.

### EeroEternal/xlb

**Identity · Language · Form factor.** Single-crate Rust (edition 2021,
tokio + httparse + rustls, ~4.8k LoC) streaming-first thin-L7/L4 load
balancer that fronts AI gateways (LiteLLM, New API, xlite, AISIX) — it
forwards bytes, not tokens, with zero payload inspection; prebuilt musl
tarballs and Docker images.

**Maintenance.** Very young and dormant — all 10 PRs authored and merged
2026-10-14 → 2026-10-19 (v0.0.1–v0.0.9), then silent; solo maintainer; zero
user-filed issues (all tracker signal is the maintainer's own PR bodies).

**Capabilities (code).** Framing-only forwarding that pools completed
streaming POSTs at proven message boundaries
(`research-repos/EeroEternal__xlb/src/proxy/l7.rs:1-13`,
`src/pool.rs:60-147`); phase-shaped timeout hierarchy — connect /
first_byte (TTFB) / idle (per-direction silence) / total (3600 s default) /
fin, cross-validated (`src/config.rs:73-111`); pre-first-byte status
rerouting with the request buffer dropped the moment response headers are
written (`src/proxy/l7.rs:117-163`, `src/proxy/http1.rs:846-923`);
non-idempotent retry boundary — POST/PUT/PATCH reroute only on unambiguous
pre-execution refusals, `respect_idempotency_key` restores the full list
(`src/config.rs:253-298`); one wall-clock retry budget spanning connect
retries, stale-socket failovers, and status reroutes (`src/config.rs:241-248`);
automatic L4 fallback via protocol sniffing (`src/proxy/l7.rs:61-106`);
active+passive health with hysteresis and authenticated probes
(`src/health.rs:58-100`); client-abort propagation during silent prefill
(`src/proxy/http1.rs:133-228`); proxy-added-latency TTFB-delta metric
(`src/metrics.rs:10-60`); validated LIFO pool with transparent stale-socket
redial incl. POSTs (`src/pool.rs:62-97`).

**Pain points (from the maintainer's PR bodies — no user issues exist).**
Non-idempotent POSTs replayed on in-doubt 500/502/504 risk double billing
by default (https://github.com/EeroEternal/xlb/pull/10); retries had no
total ceiling until `retry.timeout` (https://github.com/EeroEternal/xlb/pull/10);
client aborts during silent prefill held upstream sockets for the full
timeout (https://github.com/EeroEternal/xlb/pull/6); client-sent
`Connection: close` poisoned backend pooling (https://github.com/EeroEternal/xlb/pull/7);
credential-like request ids landed in correlation logs until auto-redaction
(https://github.com/EeroEternal/xlb/pull/10).

**Directions (pulls).** Timeout phases reshaped around LLM stream
lifecycles (https://github.com/EeroEternal/xlb/pull/2); per-backend
concurrency caps with bounded queueing (https://github.com/EeroEternal/xlb/pull/3);
data-plane circuit breaking via passive dial-failure ejection
(https://github.com/EeroEternal/xlb/pull/4); privacy-preserving multi-attempt
tracing (https://github.com/EeroEternal/xlb/pull/10); zero-dependency
distribution (https://github.com/EeroEternal/xlb/pull/5).

**Gaps vs llm-proxy.** Timeout hierarchy (llm-proxy has one flat 300 s
per-route `TimeoutLayer` at `crates/llm-proxy-server/src/routes/mod.rs:64-67`
plus a single per-chunk idle watchdog applied identically to prefill wait
and reasoning pauses at `crates/llm-proxy-provider/src/transport.rs:71-81`);
idempotency-aware replay boundary; wall-clock retry ceiling;
active+passive health; per-backend concurrency caps; checkout-validated
connection pooling with POST replay; listener
admission control. (An "OS TCP keepalive on upstream sockets" gap was
removed 2026-11-08 per research audit: reqwest 0.12.28 defaults
tcp_keepalive to 15 s with interval 15 s and 3 retries, and llm-proxy's
pooled builder does not override it — upstream sockets already get
keepalive.)

### yolorouter/yolorouter

**Identity · Language · Form factor.** Go 1.25 single-binary gateway
(gleam SQLite or PostgreSQL) with an embedded Vue 3 console; distinctive
fact→decision kernel — request-path logic is a decision table guarded by
AST-based structural tests.

**Maintenance.** Fast and steady — ~11 releases 0.1.9 (Aug 2026) → 0.2.9
(2026-10-27); development lands as direct commits; PRs are an
outside-contribution channel with a 0% merge rate so far.

**Capabilities (code).** Key pool with per-provider rotation and per-key
benching — a plain 429 benches a key for its Retry-After window clamped to
[1 s, 10 min], benched keys demoted not excluded, soonest-to-recover first
(`research-repos/yolorouter__yolorouter/internal/gateway/keypool.go`);
three-state circuit breaker where admission/accounting travel as a
generation token so a pre-trip result cannot mis-book half-open
(`internal/gateway/circuit/circuit.go`); per-model scheduling modes with
per-caller-key sticky bindings that keep upstream prompt caches warm
(`internal/gateway/binding.go`); seven independent timeout phases (connect,
header, first-byte, body-idle, attempt, request, TLS) cross-validated
(`internal/config/config.go:42-79`); protocol-from-path ingress over five
surfaces incl. native Gemini (`internal/gateway/negotiate.go`);
framework-aware tool-output compression chains (`internal/compress/compress.go`);
vision fallback — images described by a vision model through a loopback
call, spliced back with surgical JSON patching
(`internal/capability/visionfallback/describe.go`); per-key access control
with model allowlist, RPM/TPM, concurrency, cumulative budget in micros,
and saturating token arithmetic (`internal/model/api_key.go`); learned
rate-limit evidence parsed from exactly three header dialects with a
corroboration-capped 24 h ceiling (`internal/gateway/rate_limit_evidence.go:1-27`);
config-time model-mapping verification against the real upstream
(`internal/service/modeladmin/probe_queue.go:15-27`); in-place self-update
with checksum and rollback (`internal/selfupdate/selfupdate.go`).

**Pain points (issues).** Egress proxying was impossible (hardcoded nil
Proxy) — shipped in v0.2.8 as global + per-provider overrides
(https://github.com/yolorouter/yolorouter/issues/8); no model capability
metadata — operators learned a claimed 128K context was fake by eating the
upstream error (https://github.com/yolorouter/yolorouter/issues/9);
upstream keys were write-only with no reveal/backup path
(https://github.com/yolorouter/yolorouter/issues/10); bulk import forced
pricing decisions before capability data existed
(https://github.com/yolorouter/yolorouter/issues/13); no model-centric
pricing comparison or bulk adjust (https://github.com/yolorouter/yolorouter/issues/14).

**Directions (pulls).** Windows one-click install reimplemented in-repo
after community PRs were declined (https://github.com/yolorouter/yolorouter/pull/1, /pull/2);
the 0.2.9 release is entirely pricing-surface work (per-model price
summaries, unpriced filter, transactional bulk adjust with dry-run —
`research-repos/yolorouter__yolorouter/CHANGELOG.md:27`).

**Gaps vs llm-proxy.** Vision fallback; tool-output compression;
per-key system-prompt injection (`internal/capability/systemprompt/inject.go:14-40`);
seven-phase timeouts; learned per-key rate-limit evidence; Gemini client
ingress (`internal/router/router.go:611-617`); OAuth2/OIDC console SSO
(`internal/service/oauth/oauth_login_service.go:30-40`); config-time
mapping probes; agent-session reconstruction over the request log
(`internal/agentturns/backfill.go:1-13`); SSRF-safe egress with redirects
disabled (`internal/gateway/upstream.go:12-13`).

### Chleba/ollamaMQ

**Identity · Language · Form factor.** Rust single-crate binary (edition
2024, axum 0.8 + tokio, ~11k LoC, v0.5.1) with a built-in ratatui dashboard
— a queueing, load-balancing proxy in front of multiple Ollama, LM Studio,
and OpenAI-compatible backends; everything runs through one central
scheduler.

**Maintenance.** Active — HEAD 2026-10-30; all four PRs in its history are
external, including a substantial Sept 2026 hardening pass; exactly one
user-filed issue ever.

**Capabilities (code).** Per-user fair-share queuing — FIFO per X-User-ID,
turn order by a usage score halving every 5 minutes, charged at dispatch
regardless of outcome so failing clients keep no priority
(`research-repos/Chleba__ollamaMQ/src/dispatcher.rs:735-786`);
model-residency-aware backend tiers (prefer loaded, then cold, evicting
last) (`src/dispatcher.rs:1122-1137`); bounded aggregate queue memory —
512 MiB default across all queues with an always-admit-when-empty rule
(`src/dispatcher.rs:2337-2410`); aggregated multi-backend `/v1/models`
merging live listings (`src/dispatcher.rs:360-535`); concurrent health
probing with negative capability memory — endpoints a backend rejects are
skipped until re-probe (`src/dispatcher.rs:1218-1282`); model load/unload
control plane (`src/control.rs:855-1060`); fail-fast satisfiability
analysis for unservable requests (`src/dispatcher.rs:1771-1860`); metadata
reads never queue (`src/dispatcher.rs:565-570`); non-blocking JSONL request
log on a bounded channel (`src/reqlog.rs:20-104`).

**Pain points (issues).** Docker deployment broken out of the box — the
entrypoint passed a flag the CLI didn't accept, and users were fully
blocked (https://github.com/Chleba/ollamaMQ/issues/3).

**Directions (pulls).** The merged hardening pass: admission control,
fair-share redesign, mutex-poisoning recovery, concurrent probes
(https://github.com/Chleba/ollamaMQ/pull/4); an optional embedded web
dashboard with durable blocklists open (https://github.com/Chleba/ollamaMQ/pull/5).

**Gaps vs llm-proxy.** Multi-user queueing with decayed fair-share;
interchangeable backend pools with residency awareness; aggregate in-flight
memory budgeting; fail-fast satisfiability; model lifecycle control;
aggregated cross-backend listings; negative capability memory; the
Ollama-native dialect on both sides (`src/main.rs:357-396`); per-model
concurrency caps keyed on normalized names (`src/config.rs:166-187`);
interactive live-ops surfaces (`src/tui.rs:455-640`).

### ghuntley/preflight

**Identity · Language · Form factor.** Rust single-crate loopback proxy
(edition 2024, axum 0.8, ~3.3k LoC) that inspects OpenAI-compatible requests
and attachments for secrets before forwarding to ghuntley/underclass;
chains a pair of proxies — preflight owns secret hygiene, underclass owns
routing.

**Maintenance.** Young and automation-heavy but quiet — single squashed
commit era from 2026-09-20, last master commit that date; the entire
tracker is three bot PRs (Renovate declined, two Dependabot bumps open);
zero human-filed issues.

**Capabilities (code).** Three-mode secret enforcement (redact / no-go /
advisory) where enforcing modes reject findings without a safe replacement
(`research-repos/ghuntley__preflight/src/main.rs:23-30,405-426`);
Gitleaks-derived rules embedded via `include_str!` behind an explicit
enable-profile with byte-safe regex rewrites (`src/scanner.rs:70-141`);
exact SHA-256 "carnet" allowlist — no regex allowlists, no path exclusions,
no inline suppression comments (`src/main.rs:149-163`); attachment
inspection in a Bubblewrap-sandboxed worker with rlimits and OCR, driving
Poppler/QPDF/ExifTool/ZBar/Tesseract through typed argv
(`src/attachments.rs:221-280`); verify-after-redact that fails closed if
any finding survives (`src/attachments.rs:139-194`); content-addressed
inspection cache keyed on credential scope + rule profile + toolchain
(`src/cache.rs:152-191`); SSRF-hardened resolver with DNS pinning and
refused redirects (`src/resolver.rs:61-104`); ingress-owned cross-proxy
request correlation (`src/correlation.rs:30-47`); SIGHUP reload that keeps
the previous runtime on failure and refuses restart-requiring changes
(`src/main.rs:120-127,296-305`); anti-smuggling JSON traversal rejecting
duplicate keys and reconstructing concatenated text across content
boundaries (`src/document.rs:11-73`).

**Pain points (from dependency state — no user issues exist).** Security-
relevant bumps (rustls 0.23.45→.46) sit unmerged on a 45-day-stale master
(https://github.com/ghuntley/preflight/pull/3); the PDF inspection path
pins lopdf four minors behind the line adding decryption
(https://github.com/ghuntley/preflight/pull/2).

**Directions (pulls).** Real updates bypass PRs: a weekly scheduled workflow
revalidates and commits a refreshed vendored Gitleaks database straight to
main (`.github/workflows/rules.yml:44-64`).

**Gaps vs llm-proxy.** Request-path credential DLP; attachment OCR redaction
with pixel-coordinate mapping; sandboxed out-of-process processing of
untrusted content; SSRF-hardened fetching of client-referenced content; a
request-derived verdict cache; SIGHUP reload with restart-required
refusals; automated provenance-pinned refresh of a vendored data dependency
(`docs/adr/0006-direct-validated-rules-updates.md`); hardened NixOS deploy
with a QEMU VM test (`nixos/tests/preflight.nix`); differential
detection-parity testing against the reference implementation
(`tests/differential.rs:14-60`).

### NoahStepheno/completion-to-response

**Identity · Language · Form factor.** Go stdlib-only single binary (~1.1k
LoC, zero dependencies) serving an OpenAI Responses API endpoint and
translating to any Chat Completions upstream — the reverse direction of
most proxies, so Codex CLI and Claude Code can drive
Chat-Completions-only backends (Zhipu, DeepSeek, Ollama, vLLM, Azure).

**Maintenance.** Terminal — one merged external PR (2026-05-17, also the
last push); archived read-only 2026-08-16; its single true issue (429
retry) has zero replies.

**Capabilities (code).** A stateful `StreamConverter` synthesizing the full
Responses streaming event ladder (created → in_progress → output_item →
content_part → deltas → done → completed) from chat deltas
(`research-repos/NoahStepheno__completion-to-response/internal/transform/transform.go:368-609`);
streaming tool-call arguments reassembled per tool-call index
(`internal/transform/transform.go:455-497`); consecutive `function_call`
items grouped into ONE assistant message with a `tool_calls` array, and
mid-conversation system/developer messages hoisted to the top so the
assistant(tool_calls)→tool sequence is never broken
(`internal/transform/transform.go:87-139,143-179`); DeepSeek
`reasoning_content` serialized WITHOUT `omitempty` so empty reasoning still
reaches the backend (`internal/types/completion.go:29`); `finish_reason` →
Responses status mapped identically on both paths
(`internal/transform/transform.go:250-302`); a companion Node gateway
adding model-name→backend routing (`codex-gateway.js:26-96`).

**Pain points (issues).** A single upstream 429 kills the entire Codex
session — no retry, and the repo is archived so it will never be fixed
(https://github.com/NoahStepheno/completion-to-response/issues/2);
multi-turn thinking-mode and parallel-tool-call correctness all arrived in
the one merged PR (https://github.com/NoahStepheno/completion-to-response/pull/1).

**Directions (pulls).** The single merged PR is the whole direction:
per-vendor extension fields must round-trip even when empty, grouping rules
for parallel tool calls, message-ordering repair, and a turnkey
Codex-on-DeepSeek kit (https://github.com/NoahStepheno/completion-to-response/pull/1).

**Gaps vs llm-proxy.** Codex-proven tool-call grouping and ordering
semantics for a Responses client surface; Responses SSE event synthesis
(llm-proxy's client-facing event machine is Anthropic-shaped);
`text.format.json_schema` → `response_format` structured-output translation
(a grep over llm-proxy `crates/` for `response_format|json_schema` returns
no non-test hits); model-field front-door routing at a single base URL.

### os-tack/ostk-cache

**Identity · Language · Form factor.** Rust (edition 2024) loopback-only
"L1.5" context-anchoring caching proxy for Anthropic `/v1/messages` plus an
OpenAI `/v1/responses` adapter; three binaries (proxy, hooks installer,
stats reporter) over a tokio-free core crate; depends on private
os-tack/haystack crates via authenticated git-deps.

**Maintenance.** Stale — v0.4.0 tagged 2026-06-12, last push that date
(~5 months before this wave); its only tracker entry ever is one
unmerged docs PR.

**Capabilities (code).** Strips Claude Code's per-turn volatile `cch=<hex>;`
billing token from every string so byte-for-byte prefix caching actually
hits — reported cache_read 0 → 963K tokens/turn, ~$2.50 → ~$0.29
(`research-repos/os-tack__ostk-cache/src/billing_strip.rs:1-57`);
progressive soft-cap reduction tiers (eject oversized tool_result bodies →
prune oldest tool pairs → drop unreferenced tool definitions → structured
413 payload) (`src/rebuild.rs:591-701`); overflow answered locally with
Anthropic's exact `prompt is too long: N tokens > M maximum` shape —
binary-verified to trigger Claude Code's reactive auto-compaction
(`src/usage_truth.rs:1-79`); an SSE usage rewriter scaling `input_tokens`
so the client's proactive auto-compact re-arms (`src/usage_truth.rs:103-160`);
WARM/DEAD cache-write lane state machine picking 5m vs 1h breakpoints from
observed wake cadence (`src/write_policy.rs:1-54`); config provenance —
every resolved value carries a Source and `--print-config` shows it
(`src/config.rs:86-118`); bounded async HTTP capture subsystem
(`crates/ostk-cache-core/src/capture.rs:104-178`); self-loop defenses —
own-address refusal plus an `x-ostk-cache-hop` header
(`crates/ostk-cache-core/src/capture.rs:29-54`); inline-content→file-handle
rewriting (`src/rewrite_middleware.rs:58-109`).

**Pain points (issues/pulls).** The sole tracker entry — a member's
CONTEXT_PROJECTION.md with Mermaid diagrams — has sat unmerged ~6 months
(https://github.com/os-tack/ostk-cache/pull/1); bus factor one with private
git-deps blocking outside builds.

**Directions (pulls).** Shipped direction is empirical validation of its
own premise: a `PASSTHROUGH=1` mode for A/B-measuring the amplification
ratio, with the docs openly questioning whether rewrites help
(`research-repos/os-tack__ostk-cache/docs/PASSTHROUGH.md`).

**Gaps vs llm-proxy.** Cache-busting byte normalization (llm-proxy does no
request-body normalization); soft-cap reduction before the 32 MiB wall
(llm-proxy rejects flat at `crates/llm-proxy-server/src/routes/mod.rs:37`);
client-compaction steering via dialect-exact errors and usage rewrites;
stateful cache-write policy; agent-lifecycle hook ingestion
(`src/main.rs:173-260`); cache-effectiveness analytics (amp ratios,
hit-rate per session — llm-proxy stores raw cache columns at
`crates/llm-proxy-storage/src/sqlite.rs:118-119` but computes no ratios);
opt-in wire capture with redaction; self-loop prevention; config
provenance.

### agiwhitelist/tokdiet

**Identity · Language · Form factor.** TypeScript/Node (≥20, MIT, v0.1.0)
loopback streaming reverse proxy + SQLite store + SSE dashboard that meters
tokens/USD and actively compacts agent context for Anthropic, OpenAI, and
Gemini traffic; ships as an npm bin plus a Claude Code metering-hook
plugin.

**Maintenance.** Dormant — created 2026-06-16, last push 2026-06-18 (a
3-day arc ending in launch marketing); zero PRs ever; its only issue is
unanswered.

**Capabilities (code).** Recoverable context compaction — old tool results
above a floor replaced by head+salient-lines+tail markers with the full
block persisted under a content-addressed id
(`research-repos/agiwhitelist__tokdiet/src/compactor/elision.ts:36-77`);
page-fault auto-recovery — if the answer echoes a paged-out id or complains,
the proxy restores the blobs and silently re-sends once
(`src/pagefault.ts:23-80`); a shadow-eval quality guard — sampled compacted
requests re-run uncompacted, scored heuristically or by a cheap LLM judge,
rolling degradation trips safe-mode per strategy with persisted per-repo
priors (`src/quality/guard.ts:83-136`); cache-prefix-aware mutation — every
strategy refuses to rewrite past the last Anthropic cache_control
breakpoint, savings re-derived from authoritative recounts
(`src/providers.ts:318-369`); thinking/signature blocks excluded from all
mutation (`src/providers.ts:46-63`); fail-open passthrough as the cardinal
rule (`src/proxy.ts:452-492`); tiered session/day/repo-month budgets whose
breach action can be force-compaction rather than refusal
(`src/budget.ts:53-105`); honest proxy-overhead accounting
(`src/proxy.ts:593-614`).

**Pain points (issues).** Models absent from pricing.json are silently
priced $0.00 with no flag — an unpriced month looks like a free month
(https://github.com/agiwhitelist/tokdiet/issues/1); longest-prefix fallback
prices next-generation ids at previous-generation rates — `claude-opus-4-5-*`
inherits the `claude-opus-4` key's rate (`src/pricing.ts:48-60`, per the
same report); the bundled catalog is frozen at the prior model generation
(the same issue: pricing.json dated 2026-06-16, models one generation
behind, refreshable only by hand).

**Directions (commits — no PRs exist).** The core bet is "context as
virtual memory" with measured compaction quality: a 66-task A/B suite and
honest two-model results shipped alongside the features
(`https://github.com/agiwhitelist/tokdiet/commits/main`).

**Gaps vs llm-proxy.** Request-path context compaction; shadow-eval
quality guard for any transformation llm-proxy performs; recoverable
elision; cache-prefix mutation safety; fail-open passthrough discipline;
Gemini client endpoint (`src/providers.ts:567-647`); session/repo/source
attribution on usage rows (`src/session.ts:20-61`); agent-side hooks;
budget-breach escalation to compaction.

---

## Part II — Issue-tracker sweeps (40 repositories)

Compact records: one identity line, then the strongest pain points and
directions with evidence.

### 0xzr/freellmpool

Python free-LLM aggregator proxy (chat/embeddings/transcription + MCP
toolbox + OpenCode plugins) pooling ~24 keyless providers behind multi-mode
routing. Routing evolved from naive `fast` to `fair` to latency-aware
`spread` after agentic loops caused 429 storms — each turn failing over
through ~25 providers (https://github.com/0xzr/freellmpool/pull/25); a
persistent, privacy-safe per-route/per-provider health store with circuit
breakers fed by Retry-After and rate-limit-reset headers, surviving
restarts (https://github.com/0xzr/freellmpool/pull/97); streaming hardening
— no fake `finish=stop` after mid-stream failure, per-chunk wall-clock
deadline against slow-drip upstreams (https://github.com/0xzr/freellmpool/pull/17);
omitting the OpenAI-contract `created` field broke Vercel AI SDK's date
parsing on every call (https://github.com/0xzr/freellmpool/issues/112);
free-provider catalogs churn — the repo's own sentinel auto-filed eight
drift issues until lifecycle-state detection and a single reusable sentinel
issue landed (https://github.com/0xzr/freellmpool/issues/141); an
async-client redirect bug let validated public URLs redirect requests and
credentials to internal targets (https://github.com/0xzr/freellmpool/issues/87).

### 1mcp-app/agent

TypeScript/Node MCP gateway aggregating stdio/SSE/HTTP servers behind one
client endpoint; 86 of 100 recent PRs merged. New Streamable HTTP sessions
timed out for hours while `/health` and `/health/ready` stayed green —
observational health that doesn't prove the server can accept sessions
(https://github.com/1mcp-app/agent/issues/670); a single backend exit
wedged the whole inbound stdio connection; the fix withdraws only the
failed backend's routes (https://github.com/1mcp-app/agent/issues/641);
upstream OAuth initiation failures around RFC 9728 `resource` matching and
incomplete issuer metadata (https://github.com/1mcp-app/agent/issues/653);
tool-result byte budgets with a recognizable `schema_budget_exceeded` error
and spill-to-file instead of hard failure (https://github.com/1mcp-app/agent/pull/726);
per-purpose capture budgets so an over-budget page is skipped without
silently dropping the whole capability list
(https://github.com/1mcp-app/agent/pull/645); a timeout taxonomy where an
execution timeout "never triggers fallback or replay" because the outcome
is uncertain (https://github.com/1mcp-app/agent/issues/709).

### AgentOps-AI/agentops

Python LLM-agent observability SDK + OTLP collector + dashboard; pushed to
fully open-source self-hosting in Sept 2025. Token-usage extraction is the
most-repeated defect — providers returning usage as a mapping get all-None
counts (https://github.com/AgentOps-AI/agentops/issues/1432); legacy
`prompt_tokens/completion_tokens` attribute names vs the new
`input_tokens/output_tokens` family silently undercount whichever a
collector sums (https://github.com/AgentOps-AI/agentops/issues/1447);
Anthropic paths drop cache_read/cache_creation entirely so cached sessions
under-report (https://github.com/AgentOps-AI/agentops/issues/1445); cost is
a bare float with no currency — the OTel GenAI proposal adds
amount/currency/source (https://github.com/AgentOps-AI/agentops/issues/1462);
model-price catalogs rot three ways at once
(https://github.com/AgentOps-AI/agentops/issues/1460); stream finalization
overwrites iteration errors to OK so failed streams export as successful
(https://github.com/AgentOps-AI/agentops/issues/1434); a hardcoded fallback
signing secret allowed JWT forgery (https://github.com/AgentOps-AI/agentops/issues/1446);
cost computed at query time from a versioned dictionary in ClickHouse UDFs
(https://github.com/AgentOps-AI/agentops/pull/1249).

### AmoyLab/Unla

Go + React self-hosted MCP gateway wrapping MCP servers and OpenAPI/REST
APIs. `initialize` hardcodes protocolVersion 2024-11-05 and never reads the
client's requested version (https://github.com/AmoyLab/Unla/issues/341);
compressed upstream bodies read as text garbled payloads for seven months
until a decode-and-normalize pass landed
(https://github.com/AmoyLab/Unla/issues/234,
https://github.com/AmoyLab/Unla/pull/302); SSRF hardening with DNS pinning
and per-redirect revalidation — and the matching escape hatch after blanket
internal bans broke in-cluster deployments (https://github.com/AmoyLab/Unla/pull/305,
https://github.com/AmoyLab/Unla/issues/299); OTel tracing with optional
body capture (https://github.com/AmoyLab/Unla/pull/252) and a configurable
Prometheus endpoint (https://github.com/AmoyLab/Unla/pull/289); MCP 2026-07-28
stateless revision support with `_meta` negotiation
(https://github.com/AmoyLab/Unla/pull/324); a gin middleware-order bug
silently stripped admin checks from tenant routes — registration order
matters (https://github.com/AmoyLab/Unla/issues/339).

### ENTERPILOT/GoModel

Go self-hostable multi-protocol gateway with virtual models, a dashboard,
MCP serving, and a licensed "Pro" billing tier; 76 of its last 100 PRs
merged within twelve days. Virtual-model `params`/`default_params` presets
(override + fill-if-omitted) answered a LiteLLM-migrant's ask
(https://github.com/ENTERPILOT/GoModel/issues/1333,
https://github.com/ENTERPILOT/GoModel/pull/1353); updating a provider's
credentials unregistered it and wiped the discovered inventory across every
replica within ~2 s — the fix is an atomic in-place adapter swap inheriting
the old inventory (https://github.com/ENTERPILOT/GoModel/issues/1308,
https://github.com/ENTERPILOT/GoModel/pull/1338); per-request steering
headers behind an allowlist because per-client control is an abuse vector
(https://github.com/ENTERPILOT/GoModel/issues/1347); native TLS/mTLS with
5 s-stat cert reload for cert-manager symlink swaps
(https://github.com/ENTERPILOT/GoModel/issues/1285,
https://github.com/ENTERPILOT/GoModel/pull/1326); DB-only cluster sync of
admin writes with per-replica writer ids
(https://github.com/ENTERPILOT/GoModel/pull/1295,
https://github.com/ENTERPILOT/GoModel/pull/1296); reseller billing —
per-customer markups, invoice export, `billable: false` BYOK skip
(https://github.com/ENTERPILOT/GoModel/pull/1291,
https://github.com/ENTERPILOT/GoModel/pull/1293); workload identity
federation exchanging platform tokens instead of static keys
(https://github.com/ENTERPILOT/GoModel/pull/1276), plus auth before body
decompression (https://github.com/ENTERPILOT/GoModel/pull/1259).

### EinStack/glide

Go open-source LLM gateway; development stalled after v0.1.0-rc.1 (June
2024) with roadmap epics still open. Its merged core was request-path
resilience — token-bucket health/rate tracking with priority fallback
(https://github.com/EinStack/glide/pull/68), least-latency routing
normalized per output token (https://github.com/EinStack/glide/pull/111),
permanent unavailability on unauthorized keys
(https://github.com/EinStack/glide/pull/180); streaming error semantics —
typed error codes with finish_reason on the error schema
(https://github.com/EinStack/glide/pull/203); a whole OTel subsystem built
and defaulted off (https://github.com/EinStack/glide/pull/237,
https://github.com/EinStack/glide/pull/254); unshipped demands: BYO keys
with validation (https://github.com/EinStack/glide/issues/257), authn/authz
epics (https://github.com/EinStack/glide/issues/232), message-logging
opt-out for compliance (https://github.com/EinStack/glide/issues/261), and
the still-open semantic-caching epic
(https://github.com/EinStack/glide/issues/230); a released image panicked
on a missing external spec file until the spec was embedded
(https://github.com/EinStack/glide/issues/209,
https://github.com/EinStack/glide/pull/211).

### Inebrio/Routerly

TypeScript/Fastify gateway + React dashboard centered on semantic-intent
routing, OAuth subscription providers, and transparent pass-through.
Usage persisted by whole-file JSON rewrite per request grew unbounded until
the Docker VM was OOM-killed — the acknowledged fix is append-only NDJSON
(https://github.com/Inebrio/Routerly/issues/124); missing cache prices
defaulted cached reads to the full input rate, overstating spend and
tripping budgets early (https://github.com/Inebrio/Routerly/issues/144);
operators want in-flight requests visible in the log — routing decisions
appear only after completion (https://github.com/Inebrio/Routerly/issues/148);
horizontal scaling impossible with in-memory routing state and file-based
config writes (https://github.com/Inebrio/Routerly/issues/86); transparent
pass-through forwards any unhandled path to the upstream with only the
credential swapped — a drop-in base-URL replacement for endpoints the proxy
never modeled (https://github.com/Inebrio/Routerly/pull/106); OAuth
subscription providers (Claude Pro/Max, ChatGPT/Codex) forwarded verbatim
(https://github.com/Inebrio/Routerly/pull/107, later hardened by
https://github.com/Inebrio/Routerly/pull/136); semantic intent routing with
per-intent examples — and the immediate follow-on demand for decision
explainability (https://github.com/Inebrio/Routerly/pull/49,
https://github.com/Inebrio/Routerly/issues/113).

### LMCache/LMCache

KV/prefix-cache layer for vLLM/SGLang over ZMQ or gRPC with tiered storage
and fleet coordination; mid-extraction into separate client/server
distributions. Cache identity omitted the model revision so two servers
running one model ID at different `--revision` values shared cache entries
computed by older weights (https://github.com/LMCache/LMCache/issues/6253);
ambient `http_proxy` env vars silently hijacked localhost gRPC for the full
300 s timeout — explicit per-endpoint proxy config preferred
(https://github.com/LMCache/LMCache/issues/6277); one blocked shared-socket
client froze every client in the process until NOBLOCK sends landed
(https://github.com/LMCache/LMCache/issues/6255); the server survived
SIGTERM and ran 45–90 s after failed startup until lifecycle CI
(kill/restart/same-port) was added (https://github.com/LMCache/LMCache/issues/6192);
internal clients built without credentials failed 401 every cycle and
silently degraded two subsystems (https://github.com/LMCache/LMCache/pull/6247);
request-scoped cache policy landing as a tagged identity with
operator-gated defaults (https://github.com/LMCache/LMCache/pull/6199).

### LostRuins/koboldcpp

llama.cpp-derived single-binary local inference server with OpenAI/Anthropic
APIs and a vendored stable-diffusion fork. Generation errors returned as
HTTP 200 with `finish_reason:"error"` drove unbounded agent retry storms,
and custom stop-string hits reported as `length` instead of `stop`
(https://github.com/LostRuins/koboldcpp/issues/2481); streamed tool requests
took a fake-stream path emitting no final usage chunk, so agentic clients
saw 0 tokens and auto-compaction never fired
(https://github.com/LostRuins/koboldcpp/issues/2480); Anthropic conformance
closed one field at a time — `message_start` content/stop_reason (without
which the official SDK's `get_final_message()` crashed), stop_sequences,
tool_choice (https://github.com/LostRuins/koboldcpp/pull/2567, /pull/2568,
/pull/2571); a non-function entry in the tools array crashed the handler
(https://github.com/LostRuins/koboldcpp/issues/2516); MCP tool-result images
were rendered into the prompt as ~120k tokens of text on one path
(https://github.com/LostRuins/koboldcpp/pull/2376); a malformed tool call
spun the Python layer at 100% CPU with no abort
(https://github.com/LostRuins/koboldcpp/issues/2500); disconnect-driven
cancellation and a progress endpoint for long generation jobs
(https://github.com/LostRuins/koboldcpp/pull/2318, /pull/2316).

### MartialBE/one-hub

Go + React gateway (the fork done-hub derives from); maintenance stalled
after 2026-01-25, with an open "is this still updating?" issue. Users
demanded failover retry on ANY non-success status and error-rate-driven
weight downgrades (https://github.com/MartialBE/one-hub/issues/852,
/issues/921); a client-facing `/v1/responses` is needed because the hub
only converts upstream Responses→Chat — and Responses SSE clients
hard-require wire-exact usage fields, a zero-value `cached_tokens` omitted
via `omitempty` broke Codex (https://github.com/MartialBE/one-hub/issues/868,
https://github.com/MartialBE/one-hub/pull/910); OpenAI→Gemini tool-calling
translation is a live bug farm — `strict` must be stripped, `functionCall`
parts must map back to `tool_calls` (https://github.com/MartialBE/one-hub/issues/918);
Claude streaming lost cache-token usage fields
(https://github.com/MartialBE/one-hub/issues/884); resale operators asked
for per-API-key rate limits because group-level limits throttle every user
sharing the group (https://github.com/MartialBE/one-hub/issues/835);
merged: a GroupManager
primary/backup failover ladder (https://github.com/MartialBE/one-hub/pull/823),
per-token IP whitelists (https://github.com/MartialBE/one-hub/pull/824) and
billing-tag metadata (https://github.com/MartialBE/one-hub/pull/891); a
per-channel request-mutation pipeline is a durable theme — `extra_body`
passthrough, custom-param modes, an open `__delete__` sentinel
(https://github.com/MartialBE/one-hub/pull/900, /pull/873,
/issues/927).

### NVIDIA-NeMo/Guardrails

NVIDIA's Python guardrails toolkit pivoting into infrastructure: an
experimental typed proxy launched from YAML manifests with
validate/inspect/schema/serve commands (https://github.com/NVIDIA-NeMo/Guardrails/pull/2468);
LiteLLM Chat and Responses dialects preserving gateway-added extension
fields and encrypted reasoning items (https://github.com/NVIDIA-NeMo/Guardrails/pull/2529);
an HTTP hooks sidecar exposing `/v1/hooks/claude-code` and `/v1/hooks/codex`
PreToolUse endpoints plus a Rust Relay plugin on the same toolchain as
llm-proxy (https://github.com/NVIDIA-NeMo/Guardrails/pull/2473, /pull/2489);
self-check rails sent `temperature`/`max_tokens` that reasoning models
reject — per-model-family parameter normalization must apply to every
internal caller (https://github.com/NVIDIA-NeMo/Guardrails/issues/2539);
per-tool rails need rewrite semantics, not just allow/block
(https://github.com/NVIDIA-NeMo/Guardrails/issues/2497); request-scoped
correlation IDs don't reach rails (https://github.com/NVIDIA-NeMo/Guardrails/issues/2495);
PII masking leaked raw values on overlapping spans and needs stable
numbered placeholders (https://github.com/NVIDIA-NeMo/Guardrails/issues/2469,
/issues/2546); built-in rails failed open on `text=None` while OTEL spans
logged untransformed messages (https://github.com/NVIDIA-NeMo/Guardrails/issues/2478).

### Nya-Foundation/NyaProxy

Python gateway with Vue dashboard and embedded config UI pooling provider
keys. A CVSS-9.8 report: with the default empty api_key the web config UI
and dashboard were fully unauthenticated, enabling endpoint rewrites for
SSRF, with redirect-following to private IPs and credentialed wildcard CORS
(https://github.com/Nya-Foundation/NyaProxy/pull/100); multi-window
per-key rate limits — every window enforced independently, key usable only
while all have room (https://github.com/Nya-Foundation/NyaProxy/pull/133);
dashboard write paths with schema validation, rollback, and a control flag
(https://github.com/Nya-Foundation/NyaProxy/pull/134); a held credential
lock was never released when stream teardown raised — finalizers must run
in `finally` with per-finalizer isolation (https://github.com/Nya-Foundation/NyaProxy/pull/117);
locks without expiry deadlocked an entire API for 30 minutes until
self-expiring deadlines landed (https://github.com/Nya-Foundation/NyaProxy/pull/121);
wheel and container smoke tests so a broken ENTRYPOINT can never ship
(https://github.com/Nya-Foundation/NyaProxy/pull/111).

### RouteWorks/RouterArena

Benchmark/leaderboard for LLM routers grading per-query model picks on
accuracy and cost. Two of the top four entries send >98% of queries to one
model — the audience demands routing reported as a per-query distribution
with single-model submissions labeled baselines
(https://github.com/RouteWorks/RouterArena/issues/272); models absent from
the price catalog were silently dropped before evaluation (0 of 5,273
evaluated) and new model slugs blocked submissions
(https://github.com/RouteWorks/RouterArena/issues/193, /issues/206); the
fix rebuilt pricing as a governed subsystem — alias spellings collapsed to
profiles, exact matching, unpriced charged the table maximum instead of $0,
and an offline re-pricing tool that re-costs stored results without
re-grading (https://github.com/RouteWorks/RouterArena/pull/218); a router's
own decision-call usage became a separately charged `router_cost` computed
by the same code as grading (https://github.com/RouteWorks/RouterArena/pull/251);
a wall-clock-only sandbox let one allocation loop OOM the whole runner
until RLIMIT_AS caps landed (https://github.com/RouteWorks/RouterArena/issues/238,
/pull/239); grading now runs base-branch code with write tokens away from
submission code (https://github.com/RouteWorks/RouterArena/pull/254);
stored scores are versioned artifacts of the evaluator and get re-graded
retroactively (https://github.com/RouteWorks/RouterArena/pull/244).

### StackOneHQ/defender

TypeScript prompt-injection defense library (`defendToolResult()`) with a
three-tier pipeline and a Python port held to shared fixtures. Default
config was exploitable via `__proto__` mutation and pre-formed fake
`[UD-TRUSTED]` boundary tags (https://github.com/StackOneHQ/defender/issues/46);
tier 3 became a bring-your-own-LLM judge where unusable replies BLOCK
because an attacker can cause them, with fail-closed selectable
(https://github.com/StackOneHQ/defender/pull/106); thresholds are governed —
deciding on P(block) instead of argmax after a corpus change silently moved
recall 96%→61% at a fixed cut (https://github.com/StackOneHQ/defender/pull/74);
two silent-coverage bypasses fixed: deterministic oversize sampling let an
attacker precompute the dropped slot (fixed with boundary-aware chunking),
and index-free drop tracking stripped a field from every array element
(https://github.com/StackOneHQ/defender/pull/96, /pull/118); flattening
structured data to bare values turned `permissionLevel: "create"` into a
block at p≈1.00 until record-oriented review input landed
(https://github.com/StackOneHQ/defender/pull/94); cross-implementation
behavior pinned by public conformance fixtures with mutation-testing
verification (https://github.com/StackOneHQ/defender/pull/108, /pull/110).

### abetlen/llama-cpp-python

Official Python bindings + OpenAI-compatible server for llama.cpp, on rapid
automated upstream tracking. macOS Metal wheels above 0.3.30 corrupt when
two runners build the same wheel filename into one dist directory
(https://github.com/abetlen/llama-cpp-python/issues/2352); a `**kwargs` sink
silently swallows documented flags (https://github.com/abetlen/llama-cpp-python/pull/2364);
`max_completion_tokens` requests pass `max_tokens=None`, silently dropping
the limit (https://github.com/abetlen/llama-cpp-python/pull/2386); a shared
instance is not thread-safe and aborts the process uncatchably — motivating
out-of-process local runtimes behind a proxy
(https://github.com/abetlen/llama-cpp-python/issues/2392); native-resource
teardown on reload fixed after a null-pointer crash on second load
(https://github.com/abetlen/llama-cpp-python/issues/2342); the disk state
cache ignores its own capacity knob — knobs must demonstrably drive
eviction (https://github.com/abetlen/llama-cpp-python/pull/2400).

### aws-samples/bedrock-access-gateway

AWS's Python/FastAPI OpenAI-compatible proxy in front of Bedrock Converse —
officially deprecated 2026-09-04 because Bedrock now natively serves
Anthropic Messages and OpenAI-compatible Chat/Responses endpoints, so a
Converse adapter is largely replaced by pointing existing OpenAI/Anthropic
adapters at the new surfaces (https://github.com/aws-samples/bedrock-access-gateway/pull/264);
per-generation thinking-API churn — Opus 4.7 requires
`thinking.type=adaptive` + `output_config.effort`, with community fix open
since May (https://github.com/aws-samples/bedrock-access-gateway/issues/248);
per-model sampling-parameter denylists grown release over release
(https://github.com/aws-samples/bedrock-access-gateway/pull/238);
tool-use content dropped when truncation coincides (issue #197);
OpenAI-SDK `cache_control` hints never become Bedrock `cachePoint` blocks
(issue #229); catalog correctness — `/v1/models` doubling entries where
only the inference-profile-routed ID is invocable (issue #256); SSRF fix
for server-side fetching of caller media URLs at deprecation time
(https://github.com/aws-samples/bedrock-access-gateway/pull/262).

### blue-pen5805/llm-proxy-on-cloudflare-workers

TypeScript proxy on Cloudflare Workers with per-provider key rotation and
AI Gateway integration. Caller-side proxy auth headers were forwarded
upstream and could override provider credentials because caller headers
spread after provider headers — fixed by stripping and making provider
credentials non-overridable (https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/issues/350);
discovery rotates provider keys only on 429, shares one 60 s timeout
across attempts, and logs named credential profiles
(https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/pull/349);
provider adapter headers are rebuilt per fallback slot so a later key
cannot keep the first slot's auth scheme
(https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/pull/346);
`Math.random()` key selection replaced with rejection-sampled crypto
randomness after a code-scanning alert
(https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/pull/315);
selective URL-logging redaction preserving non-sensitive params
(https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/pull/305);
continuous JS dependency-vulnerability pressure as a standing ops cost
(https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/issues/344).

### cita-777/metapi

TypeScript relay-site aggregator with pooled OAuth accounts, weighted
routing, and site probing; bursty maintenance. Gemini upstreams reject
replayed OpenAI-style tool-call history without `thought_signature` — fixed
on the OpenAI-compat path after an earlier native-path fix
(https://github.com/cita-777/metapi/issues/580); multi-turn `/v1/responses`
proxying breaks on reasoning items with `encrypted_content` but no
`content` (https://github.com/cita-777/metapi/issues/538); credential-health
classification marking keys expired on any 401 or "token"+"invalid"
substring silently drops healthy keys (https://github.com/cita-777/metapi/issues/582);
`ADMIN_IP_ALLOWLIST` reads client IP from `X-Forwarded-For` with no
trusted-proxy check — spoofable (https://github.com/cita-777/metapi/issues/631);
cost reconciliation by latency windows pollutes channel stats under
retries — match on request identity, not latency
(https://github.com/cita-777/metapi/issues/566); per-site model probing
with auto-disable past a latency threshold
(https://github.com/cita-777/metapi/pull/510); proactive OAuth/session
refresh before expiry (https://github.com/cita-777/metapi/pull/430);
per-site concurrency caps, explicit priorities, and channel isolation
during retries landed in one community PR
(https://github.com/cita-777/metapi/pull/609).

### cuihuan/awesome-ai-gateway

Curated bilingual gateway list with an evidence methodology — canary checks
against first-party references, dated data snapshots, a CI freshness gate.
Curated data rots faster than re-verification: the freshness gate failed
with seven datasets over the 30-day maximum
(https://github.com/cuihuan/awesome-ai-gateway/issues/146); relay commercial
terms (minimum top-up, credit expiry) became a dataset after user demand
(https://github.com/cuihuan/awesome-ai-gateway/issues/88); the canary
script detects silent model substitution via tokenizer-fingerprint skew and
downgrade flags (https://github.com/cuihuan/awesome-ai-gateway/issues/133);
two independent October submissions are pure failover companions — weighted
round-robin + cooldown + `/health` (https://github.com/cuihuan/awesome-ai-gateway/pull/156);
agentgateway CEL policies failing open on OR'ed matchExpressions
(https://github.com/cuihuan/awesome-ai-gateway/pull/110); routing
explainability (an `x-or-route-reason` header on every response) and
savings dashboards as emerging differentiators
(https://github.com/cuihuan/awesome-ai-gateway/pull/170); a reproducible
translation-overhead benchmark — Bifrost 0.63 ms vs LiteLLM 5.34 ms at c=1
(https://github.com/cuihuan/awesome-ai-gateway/pull/96); a key-liveness
monitor because balance pages stay green while keys are disabled upstream
(https://github.com/cuihuan/awesome-ai-gateway/pull/101).

### data-privacy-stack/presidio

Python PII detection/anonymization framework. Pattern detection is
systematically bypassable by Unicode variants — CJK word boundaries,
no-break-space-grouped card numbers, Arabic-Indic digits
(https://github.com/data-privacy-stack/presidio/issues/2421, /issues/2438, /issues/2377);
the fix pattern is an opt-in engine-level NFKC normalization pass rather
than touching 44 recognizer patterns
(https://github.com/data-privacy-stack/presidio/pull/2397); quadratic span
processing in the redaction hot path fixed as one sorted sweep
(https://github.com/data-privacy-stack/presidio/issues/2370); the HF NER
recognizer fails silent/open — a guardrail must fail closed
(https://github.com/data-privacy-stack/presidio/issues/2441);
REMOVE_INTERSECTIONS conflict resolution left flagged text in the clear
until results were sorted by (start, end)
(https://github.com/data-privacy-stack/presidio/pull/2331); analyzer pods
OOM on large inputs — bounded processing is config, not luck
(https://github.com/data-privacy-stack/presidio/issues/2446); a privacy
tool making outbound calls breaks air-gapped enterprises
(https://github.com/data-privacy-stack/presidio/issues/2447).

### eclipse-lmos/lmos-router

Kotlin/Spring routing library of the Eclipse LMOS agent platform (LLM,
embedding, and MVEL rule routers). Multi-provider support was the first ask
(https://github.com/eclipse-lmos/lmos-router/issues/30); LLM-authored
routing output breaking a strict schema resolved queries to no agent —
treat model decisions as untrusted input with tolerant parsing and a
deterministic fallback (https://github.com/eclipse-lmos/lmos-router/issues/45);
an optional routing flavor crashed at runtime because optional backends
weren't classpath-isolated (https://github.com/eclipse-lmos/lmos-router/issues/42);
routing runs on embeddings against agent specs in Qdrant
(https://github.com/eclipse-lmos/lmos-router/pull/254); a deterministic MVEL
rule engine coexists with statistical routing
(https://github.com/eclipse-lmos/lmos-router/pull/303); monthly merged
azure-identity bumps indicate sustained Entra credential support
(https://github.com/eclipse-lmos/lmos-router/pull/307); the 2026 tracker is
99% dependabot — no visible feature work
(https://github.com/eclipse-lmos/lmos-router/pull/211).

### ericc-ch/copilot-api

TypeScript/Bun proxy re-exposing GitHub Copilot subscriptions as
OpenAI/Anthropic endpoints for agent CLIs; stalled with a growing pile of
unmerged community fixes. Copilot now tags models with
`supported_endpoints`, and Responses-only models hard-fail on
`/chat/completions` — per-model endpoint-capability routing requested
(https://github.com/ericc-ch/copilot-api/issues/232); reasoning workloads
die at a ~60 s upstream idle timeout before the first token — retry,
keepalives, and abort-on-disconnect requested
(https://github.com/ericc-ch/copilot-api/issues/223; open PR
https://github.com/ericc-ch/copilot-api/pull/278 aborts the upstream fetch
when the client disconnects, to stop burning quota); client beta fields
outpace the upstream and `max_tokens` is rejected on newer models
(https://github.com/ericc-ch/copilot-api/issues/217); users cannot see or
control Copilot "premium interaction" quota burn
(https://github.com/ericc-ch/copilot-api/issues/201); multi-account token
pooling is a top ask (https://github.com/ericc-ch/copilot-api/issues/255);
hiding the real context window in `/v1/models` made Claude Code over-send
and compact too late (https://github.com/ericc-ch/copilot-api/issues/210);
usage passthrough so ccusage-class tools work
(https://github.com/ericc-ch/copilot-api/pull/86).

### ferro-labs/ai-gateway

Go LLM gateway ("ferrogw") with strategy routing across eight surfaces, a
plugin guardrail framework, an MCP tool loop, an embedded dashboard, and
weekly minor releases. The built-in response cache is in-memory only, so
multi-replica deployments get cold misses per pod
(https://github.com/ferro-labs/ai-gateway/issues/399); a crashed stdio MCP
server is detected but never respawned (blocked on the official Go MCP SDK)
(https://github.com/ferro-labs/ai-gateway/issues/361); MCP tool-call error
classification requested as a NEW Prometheus series rather than a relabel,
because adding a label to a live metric resets it
(https://github.com/ferro-labs/ai-gateway/issues/362); the dashboard keeps
its admin token in sessionStorage — HttpOnly cookie + CSRF requested
(https://github.com/ferro-labs/ai-gateway/issues/427); cost keyed on a
routing alias recorded $0 until canonical identity was resolved at provider
selection and carried to both cost paths
(https://github.com/ferro-labs/ai-gateway/issues/396); guardrail config
mistakes loaded as silent no-ops until validation began failing the load
with the offending key named (https://github.com/ferro-labs/ai-gateway/issues/447);
one strategy ranker with per-attempt timeouts, 429 cooldowns honoring
Retry-After, typed context-length failover, and pool modes that advance
only on failover-safe failures (https://github.com/ferro-labs/ai-gateway/pull/425);
a guardrail suite with one observable lifecycle event per match that never
carries the matched text (https://github.com/ferro-labs/ai-gateway/pull/439).

### guardrails-ai/guardrails

Python LLM-output validation framework (validators, structured-output
schemas, reask loops, OTel, a server mode). Truthiness coercion silently
corrupts structured output — legitimate 0/False/""/[] dropped, five fix PRs
unmerged (https://github.com/guardrails-ai/guardrails/issues/1603); two
parallel validator runtimes (sync vs async) return different outputs with
no warning (https://github.com/guardrails-ai/guardrails/issues/1633);
streaming-validation fidelity is a three-cluster pain — per-chunk instead
of per-line parsing, a literal validationPassed per chunk, and wire fields
hand-copied on the server path
(https://github.com/guardrails-ai/guardrails/pull/1617, /pull/1667, /pull/1606);
failure-policy defaults disagree per surface and degrade permissively —
one policy default plus hard errors on unknown names is the fix shape
(https://github.com/guardrails-ai/guardrails/issues/1656); OTel attributes
that are Python reprs break backends and leak prompts
(https://github.com/guardrails-ai/guardrails/issues/1631); SQLite trace
writes ran a retention DELETE per write until throttled
(https://github.com/guardrails-ai/guardrails/pull/1628); the
validation-result contract moved to a separately versioned types package —
the same bet llm-proxy makes with its protocol crate
(https://github.com/guardrails-ai/guardrails/pull/1571).

### inference-gateway/inference-gateway

Very active Go gateway multiplexing chat/responses/image/TTS/decisions plus
MCP client+server and A2A across dozens of providers, with OPA guardrails,
OIDC/mTLS, OTel, and a Kubernetes operator. The gateway itself is an MCP
server on `POST /mcp` speaking the stateless 2026-07-28 revision, tools
namespaced and run through the existing middleware
(https://github.com/inference-gateway/inference-gateway/pull/709); A2A as a
pure stateless relay with trace context propagated
(https://github.com/inference-gateway/inference-gateway/pull/756); new
OpenAI endpoint families shipped in days as byte-for-byte relays behind env
flags (https://github.com/inference-gateway/inference-gateway/pull/827);
mutual TLS in both directions, with a self-loop short-circuit for calls that
return through the gateway's own listener
(https://github.com/inference-gateway/inference-gateway/issues/822);
deprecated models hidden from `/v1/models` by syncing models.dev status
(https://github.com/inference-gateway/inference-gateway/issues/793); the
provider's Retry-After forwarded on upstream errors so clients learn the
real backoff signal (https://github.com/inference-gateway/inference-gateway/pull/752);
config/doc drift was the most frequent issue class (~12 of 100)
(https://github.com/inference-gateway/inference-gateway/issues/735).

### janhq/jan

Tauri/Rust + TypeScript local-first desktop assistant with an OpenAI-compatible
Local API and an agent runtime. The Anthropic→OpenAI converter mapped
context-window truncation to `finish_reason: "stop"` — truncation reported
as completion (https://github.com/janhq/jan/pull/9515); loopback Ollama
requests 403'd after Local-API hardening (https://github.com/janhq/jan/issues/9585);
MCP tool results with native image content pass through but are never
rendered (https://github.com/janhq/jan/issues/9547); security hardening
refuses non-loopback binds without an API key and stops `*` trusted-hosts
from accepting every Host header — DNS-rebinding defense
(https://github.com/janhq/jan/pull/9505, /pull/9507); tool dispatch became
consent-scoped with revalidated identities and a consent mirror that can
only narrow (https://github.com/janhq/jan/pull/9548); agent memory
snapshotted once per session so the system-prompt prefix stays
byte-identical — a changed prompt had been appending a ~10K-token copy per
run and destroying the provider cache
(https://github.com/janhq/jan/pull/9497).

### kvcache-ai/Mooncake

KVCache-centric disaggregated-serving data plane behind vLLM/SGLang. Async
transfer completion is poll-only although the engine already knows — an
eventfd-first completion RFC (https://github.com/kvcache-ai/Mooncake/issues/5164);
free-form "session" strings let a stale writer corrupt state — fenced
engine-owned sessions with abort propagation proposed
(https://github.com/kvcache-ai/Mooncake/issues/5182); under RelWithDebInfo,
struct_pack silently drops compatible fields between client and Master — a
release-only wire-compatibility drop (https://github.com/kvcache-ai/Mooncake/issues/5230);
lease expiry enforced inconsistently across replica types
(https://github.com/kvcache-ai/Mooncake/issues/5196); one week of merged PRs
fixed boolean env vars silently ignored, one env name meaning opposite
things in two transports, and a zero-thread RPC pool that hung every pooled
call while health checks stayed green
(https://github.com/kvcache-ai/Mooncake/pull/5125, /pull/5127, /pull/5189).

### lidge-jun/opencodex

TypeScript/Bun LLM proxy + Tauri dashboard fronting Codex and Claude Code
clients onto dozens of providers with account pools and quota metering;
releases 2.96.0→2.98.0 within four days. A pre-output 503
`server_is_overloaded` was returned to the client instead of hopping to the
next target — a decline before any output must be classified safe-to-replay
(https://github.com/lidge-jun/opencodex/issues/8107); upstreams that refuse
oversized input without the `context_length_exceeded` code got a generic
retryable 502, so Claude Code retried for minutes instead of compacting —
error classification now reads message text, not provider codes alone
(https://github.com/lidge-jun/opencodex/pull/8128); all client-facing
errors rewritten to proxy-owned constants that may name the provider id but
never provider text (https://github.com/lidge-jun/opencodex/pull/8108);
an Ollama-compatible inbound API reusing the existing admission chain
rather than new code (https://github.com/lidge-jun/opencodex/pull/8099);
Ollama's `prompt_eval_cached_count` mapped so cache-read stops reporting 0
(https://github.com/lidge-jun/opencodex/pull/8124); discovery timeouts
sized per source, single-flighted, with an `ok: true, discovering: true`
state distinct from failure (https://github.com/lidge-jun/opencodex/pull/8125);
headroom-aware account placement and HMAC-keyed durable conversation
affinity that discards all rows after an unclean shutdown
(https://github.com/lidge-jun/opencodex/pull/8092, /pull/8098); two open
classification bugs: an explicit body-stall 504 re-classified as 502, and
consumer backpressure recorded as upstream inactivity
(https://github.com/lidge-jun/opencodex/issues/8157, /issues/8159).

### meta-llama/PurpleLlama

Meta's umbrella for LLM cybersecurity tooling (CyberSecEval, CodeShield,
LlamaFirewall). A benchmark judging empty replies and "FAIL TO QUERY"
sentinels as accept gave a failing model the best refusal rate — bucket
transport failures separately from real responses
(https://github.com/meta-llama/PurpleLlama/issues/267); judge output valid
but off-schema crashes the run — schema validation with a neutral fallback,
not just syntax handling (https://github.com/meta-llama/PurpleLlama/pull/252);
frozen embeddings separate OOD injections at AUC≈0.999 while the native
head recalls ~23% — calibration is the lever, not the encoder
(https://github.com/meta-llama/PurpleLlama/issues/257); LlamaFirewall's
async wrapper hardcoded ALLOW/score-0 and discarded every scanner's real
verdict — sync and async paths must share one aggregation
(https://github.com/meta-llama/PurpleLlama/pull/185); the harness smuggles
protocol selection through model-name suffixes because the client is
hardwired to Chat Completions — per-model protocol choice is the expressed
need (https://github.com/meta-llama/PurpleLlama/pull/189); tamper-evident
signed, hash-chained results requested (https://github.com/meta-llama/PurpleLlama/issues/211).

### metatool-ai/metamcp

TypeScript MCP gateway aggregating servers into namespaced endpoints with
OAuth, API keys, and an admin UI. Pooled upstream sessions are the dominant
pain: stale session IDs after idle expiry, per-namespace staleness
multiplied, a leaked idle session per refresh cycle, and pool exhaustion
surfacing as a misleading "Unknown tool"
(https://github.com/metatool-ai/metamcp/issues/294, /issues/324, /issues/304, /issues/318);
the proxy injected schema-declared optional parameters with empty defaults
so strict servers rejected correctly-formed calls
(https://github.com/metatool-ai/metamcp/issues/365); StreamableHTTP never
enabled JSON responses, and `tools.list_changed` is never advertised so
clients hold stale tool lists (https://github.com/metatool-ai/metamcp/issues/331, /issues/314);
OAuth token exchange read `client_id` from the body only, failing every
RFC 6749 `client_secret_basic` client 100% of the time
(https://github.com/metatool-ai/metamcp/issues/342); multi-user trust
boundary wounds: a tool gate that fails open, unscoped admin queries, and
CVE-2026-79538 (CVSS 9.8) command spawn + SSRF from request input
(https://github.com/metatool-ai/metamcp/issues/364, /issues/375, /issues/381);
a June batch built pooled-session resilience worth copying — re-init on 404
per the spec, disjoint "session lost" vs "transport lost" envelopes feeding
one recovery path, and TOCTOU fixes for leaking stdio children
(https://github.com/metatool-ai/metamcp/pull/283, /pull/293, /pull/273).

### mostlygeek/llama-swap

Go model-swapping reverse proxy for local backends (llama-server, vLLM,
ComfyUI, audio), growing into a GPU-capacity-aware serving platform with
SQLite activity/capture stores. A pooled connection raced llama-server's
post-SSE close (`Keep-Alive` advertised but connection dying) — fixed with
a per-model `disableKeepAlives` option
(https://github.com/mostlygeek/llama-swap/issues/1205); forwarding the
browser's `Origin` header on the server-to-server hop made Anthropic 401 —
strip browser-only headers when the proxy terminates CORS
(https://github.com/mostlygeek/llama-swap/issues/1214); concurrency-limit
rejections returned as 429 so clients armed hour-long backoff ladders for a
90-second wait — capacity shedding must be 503
(https://github.com/mostlygeek/llama-swap/issues/1256); `/v1/models`
omitted `context_length` for never-loaded models, forcing clients to guess
— probe parsing made tolerant of upstream schema drift via field-wise reads
(https://github.com/mostlygeek/llama-swap/issues/1281); a hard-coded 250 MB
body limit silently broke multi-GB uploads between releases
(https://github.com/mostlygeek/llama-swap/issues/1245); per-backend token
semantics differ (llama.cpp excludes cached tokens from prompt counts,
vLLM includes them) — normalize per upstream
(https://github.com/mostlygeek/llama-swap/issues/1218); usage parsing
extracted into pluggable per-upstream extractors dispatched off an upstream
hint (https://github.com/mostlygeek/llama-swap/pull/1292); request/response
captures moved into a separate SQLite DB pruned asynchronously by total
compressed size, preserving request totals across pruning
(https://github.com/mostlygeek/llama-swap/pull/1294).

### mozilla-ai/llamafile

Single-file cosmopolitan llama.cpp distribution with an OpenAI/Anthropic-compatible
server. GPU/backend detection is the dominant unresolved bucket — AMD iGPUs
ignored, pre-Pascal laptops failing both backends
(https://github.com/mozilla-ai/llamafile/issues/965, /issues/1048); new
architectures wait on vendored-upstream bumps
(https://github.com/mozilla-ai/llamafile/issues/1075); no build provenance
ties release assets to a controlled build
(https://github.com/mozilla-ai/llamafile/issues/1059); HTTPS rides a frozen
2021 Mbed TLS fork (https://github.com/mozilla-ai/llamafile/issues/1012);
the default combined mode skips the sandbox entirely
(https://github.com/mozilla-ai/llamafile/issues/1017); Unix-socket listening
silently failed for 5.5 months — the fix binds the socket before installing
the sandbox (https://github.com/mozilla-ai/llamafile/issues/971); agentic
layering: documented built-in tools plus an Agentfile CLI with opt-in
web tools and guardrail callbacks (https://github.com/mozilla-ai/llamafile/pull/1060).

### mudler/LocalAI

Mature Go self-hosted OpenAI-compatible inference server turning into an
agentic platform. The prompt cache works but the API never returns
`usage.prompt_tokens_details.cached_tokens`, so gateways in front see every
request as uncached (https://github.com/mudler/LocalAI/issues/13006);
Responses clients send MCP tools as a single `{"type":"namespace"}` entry
that the converter drops — flattened to `ns_{n}___{name}` and split back
(https://github.com/mudler/LocalAI/pull/13098); TTS rejecting
`response_format=mp3` forced every mp3 client to run its own ffmpeg proxy —
format negotiation or an explicit 415 is the drop-in difference
(https://github.com/mudler/LocalAI/issues/13094); an in-app gallery
simulator asked so contributors can validate catalog definitions offline
(https://github.com/mudler/LocalAI/issues/13040); frontends sharing one
metadata DB can silently diverge to different blob backends — a stored
backend-identity fingerprint checked loudly at startup
(https://github.com/mudler/LocalAI/issues/13047); local signup trusted a
configured admin email without verification
(https://github.com/mudler/LocalAI/pull/12993); a capability-gated add-on
platform where a designated owner holds revisioned authority over egress
allowlists and settings, failing closed, re-authorizing every request
(https://github.com/mudler/LocalAI/pull/13033); durable, capped agent memory
with propose/confirm settlement (https://github.com/mudler/LocalAI/pull/13064).

### oobabooga/textgen

Python/Gradio local-LLM web UI with OpenAI/Anthropic-compatible APIs. The
API returns a non-standard `reasoning_content` field that mainstream
clients silently drop, yielding empty content — strict spec conformity
requested (https://github.com/oobabooga/textgen/issues/7601); premature
`LengthFinishReasonError` and mid-session `RemoteProtocolError` deaths
(https://github.com/oobabooga/textgen/issues/7585); bare tool-call parsing
resolves names by order-dependent prefix match, so `search_web` silently
executes `search` — longest-match resolution proposed twice
(https://github.com/oobabooga/textgen/issues/7666); image-heavy
conversations die at the transport's payload limit — opt-in image
downscaling before forwarding proposed
(https://github.com/oobabooga/textgen/issues/7603); per-request thinking
toggles and prior-turn reasoning stripping unresolved
(https://github.com/oobabooga/textgen/issues/7650); a loader endpoint
accepted-and-ignored an `args` dict and returned 200 — never
accept-and-ignore (https://github.com/oobabooga/textgen/issues/7577);
path-traversal fixes for API-supplied template names
(https://github.com/oobabooga/textgen/pull/7562); tool-call argument
parsing hardened for quotes and newlines — the recurring failure mode of
streamed tool args (https://github.com/oobabooga/textgen/pull/7477).

### open-webui/mcpo

Python/FastAPI "MCP-to-OpenAPI" proxy; hard maintenance stall — zero merges
in ~9 months while 15 later PRs (incl. security fixes) queue. A transient
502 left a broken async teardown pinning one CPU core at 100% for ~14
hours — containing upstream session failure per-connection is the hard part
(https://github.com/open-webui/mcpo/issues/302); MCP JSON-Schemas are not
valid OpenAPI schemas and the converter keeps crashing — enums dropped,
$ref broken, tuple items crashing startup
(https://github.com/open-webui/mcpo/issues/296, /issues/245, /issues/244);
static OAuth client metadata unusable because DCR is always performed —
Google has no DCR (https://github.com/open-webui/mcpo/issues/272); embedded
resources unhandled (https://github.com/open-webui/mcpo/issues/270); a
shipped option bricks startup — config surface needs round-trip validation
(https://github.com/open-webui/mcpo/issues/267); no per-server health
visibility (https://github.com/open-webui/mcpo/issues/225); the merged arc
is upstream-session resilience — reconnect/retry on ClosedResourceError,
tolerate some down upstreams, symlink-safe hot reload
(https://github.com/open-webui/mcpo/pull/266, /pull/194, /pull/212); a
constant-time auth comparison fix sits unmerged
(https://github.com/open-webui/mcpo/pull/317).

### protectai/llm-guard

Python composable LLM security scanners; archived 2026-07-08. The Secrets
scanner LEAKS plaintext when redact_mode=all and a prompt contains more
than one secret — offset drift; the fix stayed unmerged
(https://github.com/protectai/llm-guard/issues/305); deployment weight —
8 GB of model downloads when one scanner (~750 MB) is wanted
(https://github.com/protectai/llm-guard/issues/306); offline/air-gapped
operation broken by scanner model loading
(https://github.com/protectai/llm-guard/issues/347); false-positive
management never solved — no allowlist shipped
(https://github.com/protectai/llm-guard/issues/331); rigid dependency pins
froze downstream security patches
(https://github.com/protectai/llm-guard/issues/342); a known config
path-traversal in the API service carried unpatched to archival
(https://github.com/protectai/llm-guard/pull/323); unmerged community PRs
show demand had moved to agent-era scanning — tool-call audit, malicious
URLs, agent escape vectors — an entire niche left unowned at archival
(https://github.com/protectai/llm-guard/pull/327, /pull/350).

### redis/redis-vl-python

Redis's official Python client for vector search and agentic RAG, with a
semantic cache and an MCP server. Index-existence detection string-matches
error wording and keeps regressing per server flavor — classify by reply
type (https://github.com/redis/redis-vl-python/issues/773); `NullHandler`
on library loggers opts out of the stderr fallback so data-loss warnings
are never shown (https://github.com/redis/redis-vl-python/issues/786);
`to_dict()` lower-cases an enum that `from_dict()` rejects — self-produced
config must round-trip (https://github.com/redis/redis-vl-python/issues/808);
raw `execute_command` bypassed protocol response callbacks so the fallback
never ran (https://github.com/redis/redis-vl-python/issues/714); MCP
security arc: tool profiles AND-scoped to a tenant claim from the caller's
verified token, failing closed when the scope gate can't reach its config,
with the Host/Origin guard extended to every HTTP transport
(https://github.com/redis/redis-vl-python/pull/745, /pull/716, /pull/746);
TTL-fused cache round trips cut a 10-hit mget from 4,166 µs to 1,173 µs
(https://github.com/redis/redis-vl-python/pull/764); an `otel` extra with
per-operation spans and hit/miss counters — "every deployment writes its
own counters" (https://github.com/redis/redis-vl-python/pull/814).

### samanhappy/mcphub

TypeScript MCP gateway (hono + better-auth + dashboard), extremely fast
maintenance — 95 of 100 PRs merged within three weeks, issues closed in
0–4 days. Config-edit blast radius was the worst production pain — every
admin action re-spawned every non-connected stdio upstream; the fix scopes
invalidation to the changed server
(https://github.com/samanhappy/mcphub/issues/1448); re-tokenizing static
tool definitions on every dashboard poll saturated the event loop — cache
counts by digest (https://github.com/samanhappy/mcphub/issues/1408);
forward-auth JWTs via JWKS requested for identity-aware-proxy fronting
(https://github.com/samanhappy/mcphub/issues/1456); allow-list-only scoping
broke silently as servers were added — "all-except" exclude scoping
shipped, with exclusions filtered from listings
(https://github.com/samanhappy/mcphub/issues/1402); 99-tool governance
needed bulk selection and one batch endpoint
(https://github.com/samanhappy/mcphub/issues/1371); keys hashed at rest
with prefix-only display and per-user upstream credential slots injected
into env/args/URLs (https://github.com/samanhappy/mcphub/pull/1387, /pull/1313);
protocol negotiation per upstream with change propagation to both session
eras (https://github.com/samanhappy/mcphub/pull/1416); PostgreSQL replica
mode with config revisions and If-Match concurrency
(https://github.com/samanhappy/mcphub/pull/1431).

### smart-mcp-proxy/mcpproxy-go

Go local MCP gateway with agent tokens, authz profiles, and an offline
security scanner; 96 merged PRs in two weeks. Secret references (`${file:}`,
`${env:}`) expand in every config string with no allow-list — an
arbitrary-file-read primitive via a header
(https://github.com/smart-mcp-proxy/mcpproxy-go/issues/1721); Docker stdio
upstreams put resolved secrets on `-e KEY=VALUE` argv, exposing them in
`/proc` (https://github.com/smart-mcp-proxy/mcpproxy-go/issues/1809); a
bbolt read transaction held across a channel send hung every authenticated
request 40+ minutes while `/healthz` stayed 1 ms
(https://github.com/smart-mcp-proxy/mcpproxy-go/issues/1742); per-server
init timeouts silently capped at 30 s by a parent context
(https://github.com/smart-mcp-proxy/mcpproxy-go/issues/1772); real-client
interop tested against shipped binaries — Codex and Claude Desktop user
agents matched with a strict trailing-tag grammar
(https://github.com/smart-mcp-proxy/mcpproxy-go/pull/1799); scanning became
the default trust mode, backed by signed Ed25519 sequence-versioned rule
bundles with downgrade protection
(https://github.com/smart-mcp-proxy/mcpproxy-go/pull/1740, /pull/1701);
authz profiles with refusal attribution and `access suggest` proposing
reviewed patches that never auto-widen
(https://github.com/smart-mcp-proxy/mcpproxy-go/pull/1718); per-token
call-rate limits at one admission seam
(https://github.com/smart-mcp-proxy/mcpproxy-go/pull/1746).

### sparfenyuk/mcp-proxy

Python MCP transport bridge (stdio/SSE/Streamable HTTP, multi-server
configs), actively hardened by a solo maintainer. Multi-key scoped metadata
requested — inbound auth is a single shared token
(https://github.com/sparfenyuk/mcp-proxy/issues/201); a per-call
authorization hook requested because putting a stdio server on the network
is all-or-nothing (https://github.com/sparfenyuk/mcp-proxy/issues/270);
per-server circuit breakers requested (https://github.com/sparfenyuk/mcp-proxy/issues/202);
upstream-aware `/health` requested — `/status` always returns 200
(https://github.com/sparfenyuk/mcp-proxy/issues/197); one failing server
crashed the whole proxy until failures were isolated per server
(https://github.com/sparfenyuk/mcp-proxy/issues/198); the proxy kept
answering 200 after the wrapped stdio child died, so supervisors never
restarted it — exit non-zero when the child exits
(https://github.com/sparfenyuk/mcp-proxy/issues/247); the dominant merged
theme is radical protocol transparency — forward every optional field both
ways (https://github.com/sparfenyuk/mcp-proxy/pull/254, /pull/226, /pull/244, /pull/189, /pull/242, /pull/272, /pull/298);
and bounded lifecycle waits that never replay side effects
(https://github.com/sparfenyuk/mcp-proxy/pull/285, /pull/277, /pull/241).

---

## Pain-point index — theme → repos and issues

Cross-repo user pain, strongest first. Each theme lists the repos and the
representative issue/PR evidence.

**1. Retry/failover misclassification takes healthy capacity offline.**
sub2api — a request-level 403 disables every Gemini account in sequence
(https://github.com/Wei-Shaw/sub2api/issues/8442); family-wide cooldowns and
ignored Retry-Afters (https://github.com/Wei-Shaw/sub2api/issues/8439);
CLIProxyAPI — a model-scoped 429 benches a whole credential ~6 days
(https://github.com/router-for-me/CLIProxyAPI/issues/7197); metapi — any 401
or "token"+"invalid" substring marks keys expired
(https://github.com/cita-777/metapi/issues/582); one-hub — no retry on
504/524 (https://github.com/MartialBE/one-hub/issues/852); opencodex —
pre-output 503 returned instead of hopping
(https://github.com/lidge-jun/opencodex/issues/8107); mcp-gateway — limiter
keyed by display name (https://github.com/MikkoParkkola/mcp-gateway/issues/3950).

**2. Streaming failure semantics lie to clients.** sub2api — abnormal
termination masked as `end_turn` + `[DONE]`
(https://github.com/Wei-Shaw/sub2api/issues/8505); koboldcpp — HTTP 200 +
`finish_reason:"error"` drove retry storms
(https://github.com/LostRuins/koboldcpp/issues/2481); freellmpool — fake
`finish=stop` after mid-stream failure
(https://github.com/0xzr/freellmpool/pull/17); CCR — truncated stream as a
bare socket destroy (https://github.com/musistudio/claude-code-router/issues/1917);
glide — body-validation failure closes the stream channel
(https://github.com/EinStack/glide/issues/272); jan — truncation mapped to
`stop` (https://github.com/janhq/jan/pull/9515); opencodex — a stall 504
re-classified as 502 (https://github.com/lidge-jun/opencodex/issues/8157).

**3. Usage/token accounting has blind spots — especially cached tokens.**
done-hub — cache tokens logged 0 on OpenAI-compatible channels
(https://github.com/deanxv/done-hub/issues/51); one-hub — Claude streaming
loses cache fields (https://github.com/MartialBE/one-hub/issues/884);
AgentOps — Anthropic cache tokens dropped
(https://github.com/AgentOps-AI/agentops/issues/1445); LocalAI —
`cached_tokens` never returned (https://github.com/mudler/LocalAI/issues/13006);
llama-swap — per-backend cached-token semantics differ
(https://github.com/mostlygeek/llama-swap/issues/1218); opencodex — Ollama
cached count dropped (https://github.com/lidge-jun/opencodex/issues/8101);
aisix — secondary endpoints emit no usage events at all
(https://github.com/api7/aisix/issues/748); koboldcpp — streamed tool
requests emit no final usage chunk
(https://github.com/LostRuins/koboldcpp/issues/2480); otari — abandoned
streams billed upstream but free in the ledger
(https://github.com/mozilla-ai/otari/pull/2869).

**4. Protocol translation drops or mangles client fields into hard 400s.**
CCR — `max_tokens` and `thinking` stripped
(https://github.com/musistudio/claude-code-router/issues/1843); CLIProxyAPI —
Gemini roles, parallel tool batches, Devin reasoning
(https://github.com/router-for-me/CLIProxyAPI/issues/7186); done-hub —
object-vs-string fields break typed unmarshalling
(https://github.com/deanxv/done-hub/issues/58); metapi — Gemini
`thought_signature` replay rejected
(https://github.com/cita-777/metapi/issues/580); one-hub — OpenAI→Gemini
tool translation (https://github.com/MartialBE/one-hub/issues/918);
completion-to-response — empty `reasoning_content` omitted
(https://github.com/NoahStepheno/completion-to-response/pull/1);
bedrock-gateway — per-generation thinking-API churn
(https://github.com/aws-samples/bedrock-access-gateway/issues/248);
oobabooga — non-standard `reasoning_content` breaks clients
(https://github.com/oobabooga/textgen/issues/7601).

**5. Credential/OAuth lifecycle fragility.** CCR — Codex OAuth import
structurally broken (https://github.com/musistudio/claude-code-router/issues/1941);
CLIProxyAPI — refresh on a read-only store invalidates every credential
after restart (https://github.com/router-for-me/CLIProxyAPI/issues/7214);
sub2api — plan renames and quota-shape changes break account management
(https://github.com/Wei-Shaw/sub2api/issues/8456); 1mcp — OAuth initiation
failures around RFC 9728 matching
(https://github.com/1mcp-app/agent/issues/653); mcpo — static OAuth clients
unusable (https://github.com/open-webui/mcpo/issues/272); metamcp —
`client_secret_basic` clients fail 100%
(https://github.com/metatool-ai/metamcp/issues/342); copilot-api — token
swapping breaks provider analytics
(https://github.com/ericc-ch/copilot-api/issues/228).

**6. Hardcoded model catalogs lag provider launches.** claude-code-proxy —
every new model needs a release (https://github.com/raine/claude-code-proxy/issues/198);
inference-gateway —
deprecation dimension added via models.dev
(https://github.com/inference-gateway/inference-gateway/issues/793);
RouterArena — unknown slugs block submissions
(https://github.com/RouteWorks/RouterArena/issues/206); freellmpool —
sentinel auto-filed eight drift issues
(https://github.com/0xzr/freellmpool/issues/141); PurpleLlama — static
vendor lists reject custom endpoints
(https://github.com/meta-llama/PurpleLlama/pull/253). A "sub2api — new
models missing from sync" item cited to
https://github.com/Wei-Shaw/sub2api/issues/8508 was removed 2026-11-08 per
research audit: that issue reports Claude Ultra quota acquisition, not
model sync.

**7. Pricing data rots; unpriced traffic looks free.** tokdiet — absent
models silently $0.00 (https://github.com/agiwhitelist/tokdiet/issues/1);
RouterArena — unpriced models silently skipped at evaluation
(https://github.com/RouteWorks/RouterArena/issues/193); AgentOps — invalid
price rows abort collector config
(https://github.com/AgentOps-AI/agentops/issues/1460); Routerly — missing
cache prices default to full input rate
(https://github.com/Inebrio/Routerly/issues/144); awesome-ai-gateway —
snapshot freshness gates failing
(https://github.com/cuihuan/awesome-ai-gateway/issues/146); CCR — 24.6 MB
generated catalog needed to make pricing work out of the box
(`research-repos/musistudio__claude-code-router/packages/core/models.json`).

**8. Config surface silently no-ops or bricks startup.** mcphub — every
admin action re-spawned every upstream
(https://github.com/samanhappy/mcphub/issues/1448); oobabooga — loader
accepted-and-ignored args with a 200
(https://github.com/oobabooga/textgen/issues/7577); mcpo — `disabled_tools`
bricks startup (https://github.com/open-webui/mcpo/issues/267); Mooncake —
boolean env vars silently ignored
(https://github.com/kvcache-ai/Mooncake/pull/5125); ferro-labs — guardrail
config mistakes enforced nothing
(https://github.com/ferro-labs/ai-gateway/issues/447); GoModel — credential
updates wipe discovered inventory
(https://github.com/ENTERPILOT/GoModel/issues/1308); CLIProxyAPI — runtime
auth-dir changes leave the watcher behind
(https://github.com/router-for-me/CLIProxyAPI/issues/7160); mcpproxy-go —
per-server timeouts silently capped at 30 s
(https://github.com/smart-mcp-proxy/mcpproxy-go/issues/1772).

**9. Security boundary failures at the loopback/edge.** metapi — admin IP
allowlist spoofable via X-Forwarded-For
(https://github.com/cita-777/metapi/issues/631); NyaProxy — unauthenticated
dashboard + redirect SSRF + credentialed wildcard CORS
(https://github.com/Nya-Foundation/NyaProxy/pull/100); Unla — middleware
order stripped admin checks (https://github.com/AmoyLab/Unla/issues/339);
metamcp — CVE-2026-79538 command spawn + SSRF
(https://github.com/metatool-ai/metamcp/issues/381); cloudflare-workers —
caller auth headers forwarded and overridable
(https://github.com/blue-pen5805/llm-proxy-on-cloudflare-workers/issues/350);
freellmpool — redirects leak credentials to internal targets
(https://github.com/0xzr/freellmpool/issues/87); mcpproxy-go — secret
references as arbitrary file read
(https://github.com/smart-mcp-proxy/mcpproxy-go/issues/1721); mistral.rs —
0.0.0.0 default with no auth (https://github.com/EricLBuehler/mistral.rs/issues/2490).

**10. Session/state stickiness lost across failover or restart.**
claude-code-proxy — subagents sharing a parent session id destroy state
(https://github.com/raine/claude-code-proxy/issues/163); sub2api — WS
failover replays the conversation's first request
(https://github.com/Wei-Shaw/sub2api/issues/8513); mcp-gateway — busy
sessions reaped by age not idleness
(https://github.com/MikkoParkkola/mcp-gateway/issues/2567); metamcp — stale
pooled sessions surface as "Unknown tool"
(https://github.com/metatool-ai/metamcp/issues/318); plano — session ids
read only from an internal header
(https://github.com/katanemo/plano/issues/979); CCR — fallback served-model
misattribution (https://github.com/musistudio/claude-code-router/issues/1870).

**11. Guardrails/evaluators fail open or silently skip coverage.** otari —
inconclusive verdict passes a block-mode guardrail
(https://github.com/mozilla-ai/otari/issues/2788); presidio — HF recognizer
returns empty on error (https://github.com/data-privacy-stack/presidio/issues/2441);
PurpleLlama — async wrapper discards scanner scores
(https://github.com/meta-llama/PurpleLlama/pull/185); guardrails-ai —
permissive defaults differ per surface
(https://github.com/guardrails-ai/guardrails/issues/1656); aisix —
scan-vs-forward asymmetry as the largest defect class
(https://github.com/api7/aisix/issues/1016); tokdiet — pricing failures
invisible in the dashboard (https://github.com/agiwhitelist/tokdiet/issues/1).

**12. Ops/packaging fragility.** ollamaMQ — Docker entrypoint flag mismatch
(https://github.com/Chleba/ollamaMQ/issues/3); claude-code-proxy — GLIBC
floor jump made releases unrunnable
(https://github.com/raine/claude-code-proxy/issues/144); sub2api — one-click
self-update broken by permissions
(https://github.com/Wei-Shaw/sub2api/issues/8467); llama-cpp-python —
corrupted wheels from dual runners
(https://github.com/abetlen/llama-cpp-python/issues/2352); inference-gateway
— generated example configs unreachable in Docker
(https://github.com/inference-gateway/inference-gateway/issues/735); glide —
released image panicking on a missing external file
(https://github.com/EinStack/glide/issues/209).

## Direction index — where the niche ecosystem is heading

**Coding-agent-first servers.** mistral.rs ships per-request context limits
"for coding agents" with a rewritten agent guide
(https://github.com/EricLBuehler/mistral.rs/pull/2538); claude-code-proxy
fingerprint's Claude Code's auxiliary requests for cheaper routing
(https://github.com/raine/claude-code-proxy/pull/72); CCR intercepts the
client's own compaction flow (https://github.com/musistudio/claude-code-router/pull/1824);
plano replays routing decisions on tool-result turns
(https://github.com/katanemo/plano/pull/1015); opencodex sanitizes every
client-facing error to proxy-owned constants
(https://github.com/lidge-jun/opencodex/pull/8108).

**Quota-aware multi-account scheduling.** CLIProxyAPI's `soonest-reset`
strategy from passively observed quota headers
(https://github.com/router-for-me/CLIProxyAPI/pull/7246) and final
usage-limit errors with reset times
(https://github.com/router-for-me/CLIProxyAPI/pull/7239); sub2api's
quota-window account pausing
(`research-repos/Wei-Shaw__sub2api/backend/internal/service/account_scheduling_threshold_eval.go`);
opencodex's headroom-aware placement
(https://github.com/lidge-jun/opencodex/pull/8092); yolorouter's learned
reset evidence (`research-repos/yolorouter__yolorouter/internal/gateway/rate_limit_evidence.go`).

**MCP as a universal attach point.** aisix per-user OAuth to upstream MCP
servers (https://github.com/api7/aisix/pull/1394); inference-gateway as a
stateless 2026-07-28 MCP server
(https://github.com/inference-gateway/inference-gateway/pull/709); GoModel
serving provider-hosted MCP with per-key scoping
(https://github.com/ENTERPILOT/GoModel/pull/1272); SMG's MCP tool execution
inside the Responses path (https://github.com/smg-project/smg/pull/5045);
NVIDIA Guardrails' hooks sidecar for Claude Code/Codex
(https://github.com/NVIDIA-NeMo/Guardrails/pull/2473); mcp-gateway's
meta-tool surface with TF-IDF discovery
(`research-repos/MikkoParkkola__mcp-gateway/src/semantic_search/mod.rs`).

**models.dev as the canonical catalog/pricing source.** otari persisting
the full v2 catalog (https://github.com/mozilla-ai/otari/pull/2771); plano's
pluggable pricing feeds (https://github.com/katanemo/plano/pull/971);
inference-gateway syncing deprecation status
(https://github.com/inference-gateway/inference-gateway/issues/793);
done-hub's scheduled price sync
(`research-repos/deanxv__done-hub/model/pricing_modelsdev.go:18-108`); CCR's
generated merged catalog
(`research-repos/musistudio__claude-code-router/scripts/generate-models-json.mjs:17-32`).

**Policy/config hot reload under pinned generations.** mcp-gateway's
per-request pinned config generation (ADR-029,
https://github.com/MikkoParkkola/mcp-gateway/pull/4508); CLIProxyAPI's
debounced watcher with request re-dispatch
(`research-repos/router-for-me__CLIProxyAPI/internal/watcher/config_reload.go`);
GoModel's DB-only clustersync in ~2 s
(https://github.com/ENTERPILOT/GoModel/pull/1295); ThinkWatch's ArcSwap +
Redis pub/sub (`research-repos/ThinkWatchProject__ThinkWatch/crates/common/src/dynamic_config.rs`);
mcphub's surgical reload (https://github.com/samanhappy/mcphub/issues/1448).

**Loss-aware, diagnostics-carrying translation.** Switchyard's
`TranslationPolicy` with structured per-field diagnostics
(`research-repos/NVIDIA-NeMo__Switchyard/crates/switchyard-translation/README.md:91-123`,
issue https://github.com/NVIDIA-NeMo/Switchyard/issues/1215); NVIDIA
Guardrails preserving LiteLLM dialect fields and encrypted reasoning
(https://github.com/NVIDIA-NeMo/Guardrails/pull/2529); sparfenyuk's radical
optional-field transparency (https://github.com/sparfenyuk/mcp-proxy/pull/254);
aisix choosing native protocol fidelity over hub-and-spoke where
translation proved lossy (https://github.com/api7/aisix/pull/1318).

**Cost governance moving onto the gateway.** otari's tools as billable
units with budget holds (https://github.com/mozilla-ai/otari/pull/2831,
/pull/2839); GoModel's reseller markups and invoice export
(https://github.com/ENTERPILOT/GoModel/pull/1293); mcp-gateway's <0.1 ms
pre-invoke budget enforcer
(`research-repos/MikkoParkkola__mcp-gateway/src/cost_accounting/enforcer.rs`);
Switchyard's dual-source cost with provider-reported precedence
(`research-repos/NVIDIA-NeMo__Switchyard/crates/switchyard-costs/src/lib.rs:46-123`).

**LLM-judged correctness for caches and transformations.** GPTCache's JEV
five-question reuse gate (https://github.com/zilliztech/GPTCache/pull/701)
plus its frozen reuse benchmark (https://github.com/zilliztech/GPTCache/pull/702);
prompt-cache's gray-zone judge
(`research-repos/messkan__prompt-cache/internal/semantic/semantic.go:256-336`);
tokdiet's shadow-eval compaction guard
(`research-repos/agiwhitelist__tokdiet/src/quality/guard.ts:83-136`).

**Hardened, attested releases and supply chain.** antirion's SBOM + Sigstore
+ SLSA L3 releases (`research-repos/eliasbiondo__antirion/.github/workflows/release.yml`);
preflight's provenance-pinned weekly rule refresh
(`research-repos/ghuntley__preflight/.github/workflows/rules.yml`);
mistral.rs's verify-release job deriving the expected artifact set after a
release silently published 42 of 143 artifacts
(https://github.com/EricLBuehler/mistral.rs/pull/2407); SMG's resumable
release script (https://github.com/smg-project/smg/pull/4977); NyaProxy's
wheel/container smoke tests (https://github.com/Nya-Foundation/NyaProxy/pull/111).

**Multi-replica state sharing.** otari's Redis rate-limit store with
clock-skew-safe Lua (`research-repos/mozilla-ai__otari/src/gateway/adapters/rate_limit_store_adapter.py`);
antirion's Valkey cluster mode; SMG's gossip mesh
(`research-repos/smg-project__smg/crates/mesh/src/lib.rs`); GoModel's
per-replica writer ids (https://github.com/ENTERPILOT/GoModel/pull/1296);
plano's Redis session-affinity store
(`research-repos/katanemo__plano/crates/brightstaff/src/session_cache/redis.rs`).

**Per-key least privilege.** CLIProxyAPI's per-key model policies
(https://github.com/router-for-me/CLIProxyAPI/pull/7117); sub2api's ~60
selectable admin permissions
(https://github.com/Wei-Shaw/sub2api/pull/8476); mcphub's per-user
credential slots (https://github.com/samanhappy/mcphub/pull/1313);
mcpproxy-go's authz profiles with `access suggest`
(https://github.com/smart-mcp-proxy/mcpproxy-go/pull/1718); redis-vl's
tenant-claim-scoped tool profiles
(https://github.com/redis/redis-vl-python/pull/745).

## Recurring unmet needs — patterns in 3+ repositories

Patterns observed in three or more harvested repositories, each a
direct backlog input (see `compensation-backlog.json`). "Observed", not
"demanded by users": R4, R6, R7, R8, R10 and R11 rest on 0–1 user-filed
issues (corrected 2026-11-08 per research audit).

### How strong the demand evidence is

Caveats added 2026-11-08 per the research audit. Read this record's
pain-point index, R-sections and the backlog they feed as a catalogue of
possibilities, not a ranking of demand:

- Of 261 cited issues, 114 have no comments and 238 have no reactions; only
  5 have 3 or more reactions.
- 26 of the 40 sweep repositories are not LLM gateways (engines, guardrail
  libraries, MCP servers, a benchmark, a list). This was not disclosed
  before: R9 is 4 of 6 guardrail libraries; R12 is 4 of 4 MCP gateways.
- One account (`roy-tong`, 94 items across repositories in an
  "AgentMeasure" campaign) filed AgentOps #1445, Routerly #144 and tokdiet
  #1; R2 and R7 count these as independent demand.
- Some citations are vendor pitches (mistral.rs #2490, ThinkWatch #113,
  antirion #41) or self-described AI-written reports (koboldcpp
  #2480/#2481, raine #163).
- "Most bug-dense" and "dominant" rest on the latest 100 tracker items per
  repository; R6's "fastest-growing upstream segment" had no source and is
  withdrawn above.
- Several "researched in full" repositories are very young and much of
  their code is AI-written: xLB (created 2026-10-14, 38 commits), Antirion
  (created 2026-10-17), preflight (2026-09-20, 5 commits), tokdiet
  (32 stars), ostk-cache (10 stars, not buildable), completion-to-response
  (6 commits, archived). The per-repo sections flag most of these;
  yolorouter and mcp-gateway were not flagged there (see the wave-3 survey).
- Vendor numbers are stated as fact in places, such as mcp-gateway's
  "~8 ms hot reload".

**R1. Retry/failover that classifies before it reacts (13 repos — 12
implementations plus metapi in the failure examples; count corrected
2026-11-08 per research audit).**
ThinkWatch PR #122, Switchyard `client.rs:400-528`, xlb `config.rs:253-298`,
done-hub `relay/main.go:74-176`, CLIProxyAPI conductor, sub2api failover
loop, freellmpool PR #97, glide PR #68, one-hub issue #852, opencodex issue
#8107, smg `retry.rs`, antirion `budget.rs`. The recurring failure is not
absence but misclassification (sub2api #8442, CLIProxyAPI #7197, metapi
#582, opencodex #8107) — cooldowns keyed to model+credential scope and real
reset times, not blanket account bans.

**R2. Cached-token usage accounting end to end (12 repos).** done-hub PRs
#39/#52, one-hub #884, opencodex #8101, AgentOps #1445, LocalAI #13006,
llama-swap #1218, Routerly #144, otari #2869, Switchyard cache-write split,
aisix `token_estimate.rs`, CLIProxyAPI PR #7074, plano PR #995. llm-proxy's
core usage already carries cache fields fed only from Anthropic dialects
(`crates/llm-proxy-protocol/src/core.rs:683-685`, cite corrected 2026-11-08
per research audit — `:639-641` is `Cost`; no
`prompt_tokens_details` parsing anywhere — verified 2026-11-04).

**R3. Context-window signaling and preflight (7 repos).** mistral.rs PR
#2538, opencodex PR #8128, copilot-api #210, llama-swap #1281, ostk-cache
`usage_truth.rs`, koboldcpp #2481 (truncation semantics), yolorouter #9
(capability metadata). Clients branch on the mapped overflow code to decide
compact-vs-retry; llm-proxy already has the offline counter and
`context_length` on `/v1/models` but no preflight.

**R4. Session attribution and sticky routing (9 repos).** sub2api
`balancer.go:194-309`, CLIProxyAPI `session_cache.go:31-45`, done-hub sticky
sessions, otari session stats (#2792 is a budget-name feature request, not
session attribution — cite corrected 2026-11-08 per research audit),
Switchyard `routing_log.rs`, tokdiet
`session.ts:20-61`, plano session affinity, ThinkWatch sealed pinning
(`sealed.rs:1-92`), SMG cache-aware stickiness. llm-proxy's ledger records
only key id — no session, repo, or client-tool attribution.

**R5. Streaming idle tolerance: keepalives, phase timeouts, terminal usage
(7 repos).** done-hub `heartbeat.go:14-175`, Switchyard #1168, xlb PR #2,
yolorouter `config.go:42-79`, copilot-api #223, 1mcp #709, koboldcpp #2480.
A single flat timeout cannot express prefill wait vs reasoning pause vs
total stream; silent generation windows also trip client and proxy
timeouts that heartbeat comments prevent.

**R6. Subscription/OAuth upstream credentials (7 repos).** claude-code-proxy
`auth/manager.rs:18-70`, CLIProxyAPI `internal/auth/`, done-hub OAuth
channels, sub2api token providers, Routerly PR #107, metapi PR #430,
copilot-api #255. This R-section rests on 0–1 user-filed issues (audit
caveat, 2026-11-08); "fastest-growing upstream segment" had no source and
is withdrawn (corrected 2026-11-08 per research audit).

**R7. Externally synced price catalogs with explicit unpriced state
(6 repos).** otari `pricing_refresh_service.py`, plano PR #971, done-hub
`pricing_modelsdev.go`, CCR `generate-models-json.mjs`, RouterArena PR #218,
tokdiet #1. llm-proxy's hand-typed fixed-point TOML has no refresh story.
Unpriced is distinguishable row by row — cost stays NULL against a 0
"unmetered" price and the row API returns an Option — and only the usage
summary folds unpriced rows to 0 (corrected 2026-11-08 per research
audit).

**R8. Per-key scoping, windows, and budgets (8 repos).** yolorouter
`api_key.go`, otari `budgets.py`, mcp-gateway `enforcer.rs`, GoModel PR
#1293, NyaProxy PR #133, one-hub #835 (resale operators asking for
per-API-key rate limits because group-level limits throttle every user
sharing the group, https://github.com/MartialBE/one-hub/issues/835),
ThinkWatch PR #81, aisix conditional policies. llm-proxy rate-limits per
identity — caller key id when authenticated, client IP otherwise
(`crates/llm-proxy-server/src/routes/core_pipeline.rs:173-176`, per-key RPM
already exists per the compendium's narrowed gap #3) — but has no token
(TPM) windows, per-key spend caps, or per-key model scoping, and its keys
grant all-or-nothing model access.

**R9. Guardrails/inspection that fail closed (6 repos).** otari #2788,
presidio #2441, PurpleLlama PR #185, guardrails-ai #1656, aisix #1016,
defender PR #106. Inconclusive verdicts must land on the closed side, and
sync/async paths must share one aggregation.

**R10. An admin/config management plane beyond read-only (5 repos).**
CLIProxyAPI management v8, NyaProxy PR #134, mcphub batch endpoints, GoModel
clustersync, prompt-cache `PATCH /v1/config` + provider hot-swap. Operators
expect to change providers/keys/thresholds without a restart; llm-proxy
rejects `server.hot_reload` outright
(`crates/llm-proxy-core/src/provider_config.rs:1531-1538` per the wave-3
register) and its admin API is read-only.

**R11. Prompt-cache-aware request handling (6 repos).** ostk-cache
`billing_strip.rs` (cache-busting token strip), sub2api PR #8464 (fingerprint
convergence for cache hits), CCR PR #1866 (cache-key headers), plano
`prompt_caching.rs` (marker injection), tokdiet `providers.ts:318-369`
(mutation guard), jan PR #9497 (byte-stable memory snapshot). None of this
exists in llm-proxy: no normalization, no marker injection, no
mutation-safety boundary.

**R12. In-flight/observable failure isolation (4 repos).** 1mcp #641 (one
backend's exit must not wedge the gateway), sparfenyuk #198 (a single
failing server crashed the whole proxy until failures were isolated per
server, https://github.com/sparfenyuk/mcp-proxy/issues/198), mcphub #1448,
metamcp pool exhaustion #318. Whatever fails per-upstream must degrade only
that upstream.
