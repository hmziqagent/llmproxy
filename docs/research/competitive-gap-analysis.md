# Competitive gap analysis — llm-proxy vs. LLM gateways (October 2026)

The llm-proxy baseline and near-term candidates below describe the tree at
the [research audit](audit-2026-11-05/REPORT.md) (commit `79cf406`,
2026-11-05). Claims the audit found false are corrected in place and marked
"(corrected 2026-11-08 per research audit)". Plans 01–10 have since rebuilt
most of the described subsystems; current behaviour is in
[docs/goal.md](../goal.md) and [docs/completed/](../completed/).

Products surveyed in October 2026 across two research waves: LiteLLM, Portkey
(rebranded Prisma AIRS AI Gateway in its docs, https://portkey.ai/docs/product/ai-gateway/fallbacks),
Kong AI Gateway, Cloudflare AI Gateway, Agent Router (renamed from Envoy AI
Gateway, https://theagentrouter.ai/), Higress, agentgateway (split out of
kgateway, https://kgateway.dev/blog/kgateway-v2.2-release-blog), Helicone,
Langfuse/LangSmith, OpenRouter, Vercel AI Gateway, Glama, one-api/new-api,
gproxy, openproxy, LLMG, rust-llm-proxy, BricksLLM, Bifrost, vLLM Production
Stack, IBM ContextForge, Azure APIM AI gateway, AWS Bedrock AgentCore
Gateway, and — added in the second wave — Tyk AI Gateway/AI Studio, F5 AI
Gateway, uni-api, GPT-Load, Respan (formerly Keywords AI), Martian, Not
Diamond, Apache APISIX, and Traefik Hub.

Status notes: TensorZero's repo is archived (2026-06-12,
https://github.com/tensorzero/tensorzero); BricksLLM's last commit and
release are 2025-01-05 (https://api.github.com/repos/bricks-cloud/BricksLLM);
one-api is effectively dormant (last release v0.6.10, Feb 2025,
https://github.com/songquanpeng/one-api/releases); OpenPipe is sunset —
platform migrated to Weights & Biases/CoreWeave and docs.openpipe.ai no
longer resolves (https://openpipe.ai/security); the Portkey OSS gateway has
been dormant since 2026-05-25 after Palo Alto Networks completed its
acquisition while the enterprise line continues (https://github.com/Portkey-AI/gateway/releases);
F5's original gateway is in transition — docs redirect, Python SDK archived
2026-03-18, successor in Early Access (https://www.f5.com/company/news/press-releases/f5-unleashes-next-generation-agentic-ready-ai-gateway-to-optimize-the-economics-and-governance-of-enterprise-ai-costs);
Martian's gateway was live as of 2026-10-27 (https://api.withmartian.com/v1/models);
an "Upstash AI Gateway" named in one brief was verified not to exist — 11
live checks incl. sitemap, llms.txt, GitHub, npm, and DNS (https://upstash.com/docs/llms.txt).

Method: llm-proxy claims are verified against this working tree (file:line
cited); wave-1 grep checks ran 2026-10-14, and the wave-2 code claims
(hot_reload rejection, CacheControl shape, discovery field capture,
ModelCard, SSE keep-alive, cancellation handling, body limit) were
re-verified against the tree on 2026-10-27. Competitor claims come from
official docs and release notes current to October 2026 and from vendored
source pins read directly (`ref/litellm` at b84f7f82, `ref/oc-go-cc` at
99d756c, `ref/llm-api-key-proxy` at af40b91). No competitor code was built
or run. Entries first added in the second wave are tagged `new:`.

## Baseline: llm-proxy today

- Client surface: `POST /providers/{p}/v1/chat/completions`,
  `POST /providers/{p}/v1/messages`,
  `POST /providers/{p}/v1/messages/count_tokens`,
  `GET /providers/{p}/v1/models` — all four API routes carry the mandatory
  `/providers/{provider}` prefix — plus `/health`, `/ready`, `/version`
  (`crates/llm-proxy-server/src/routes/mod.rs:50-52,69-79`;
  `docs/protocol.md:25-30`) (corrected 2026-10-28 after code audit);
  32 MiB body limit (`crates/llm-proxy-server/src/routes/mod.rs:37,62`);
  provider API routes share a configurable `TimeoutLayer` returning 408
  while health/version routes are exempt
  (`crates/llm-proxy-server/src/routes/mod.rs:44,54-67`).
- Translation: hub-and-spoke `CoreRequest`/`CoreResponse`/`CoreEvent`
  (`docs/protocol.md:7-9`), with 10 content kinds Text/Image/Document/Audio/
  Video/ToolUse/ToolResult/Thinking/RedactedThinking/Refusal
  (`docs/protocol.md:80-91`). Client adapters: OpenAI Chat + Anthropic
  Messages. Upstream adapters: OpenAI Chat, Anthropic Messages, OpenAI
  Responses, Gemini GenerateContent (`crates/llm-proxy-provider/src/adapter/`).
- Streaming: normalized event state machine; in-band errors after visible
  output (`docs/protocol.md:120-133,139-151`); 3-second SSE keep-alive
  comments on idle streams
  (`crates/llm-proxy-server/src/routes/core_pipeline.rs:701,883`).
- Auth: `mode=none` (loopback-only, enforced) or `client-keys` with argon2id
  PHC hashes (`config.toml.example:42-54`); admin API behind a constant-time
  bearer token (`crates/llm-proxy-api/src/auth.rs`). The upstream `api_key`
  is a `SecretString` — redacted in `Debug`, skipped by serialization,
  zeroized on drop (`crates/llm-proxy-core/src/provider_config.rs:357-375`) —
  but stored as plaintext in the operator's TOML on disk.
- Routing: provider named in URL; static model-alias map
  (`crates/llm-proxy-core/src/provider_config.rs:392-394`). No request-path
  retry/fallback; only catalog discovery has bounded transient retry
  (`crates/llm-proxy-provider/src/discovery.rs:160-170`).
- Catalogs: static, live (OpenAI-compatible `/v1/models`, Anthropic, Gemini),
  or hybrid, with single-flight refresh and atomic disk cache
  (`crates/llm-proxy-server/src/catalog_service.rs`). Discovery captures only
  id/display_name/context_length — no price or lifecycle fields
  (`crates/llm-proxy-provider/src/discovery.rs:272-288`, verified 2026-10-27),
  and the `/v1/models` `ModelCard` hardcodes `created: 0` with a null
  `created_at` (`crates/llm-proxy-server/src/routes/models.rs:24-36`).
- Persistence: SQLite usage ledger, background writer, retention controls;
  the only request-body-derived column is `body_hash`
  (`crates/llm-proxy-storage/src/sqlite.rs:105-131`;
  `config.toml.example:56-68`) — full token/cost columns, no payloads.
- Admin API: read-only `/usage/summary|requests|keys` with window/group_by
  filters and cursor paging (`crates/llm-proxy-api/src/routes.rs:53-55, 116-117`).
- Metrics: lock-free atomic counters incl. per-provider-model counts
  (`crates/llm-proxy-core/src/metrics.rs`); request-id injection; graceful
  shutdown (`crates/llm-proxy-server/src/shutdown.rs`).
- Middleware: token-bucket RPM rate limit, burst-tolerant with continuous
  refill (`crates/llm-proxy-server/src/middleware.rs:117-129`), keyed by
  the caller's key id when authenticated and by client IP otherwise, with
  loopback exemption (`crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176`;
  per-key RPM since 2026-10-13, commit 499b22a;
  `config.toml.example:14-21`) (corrected 2026-10-28 after code audit), an
  opt-in dedup window — off by default, the shipped default is 0 ms — that
  409-rejects an identical path+body seen within a fixed window of the
  first sighting. It is not in-flight tracking and not caller-scoped, and
  bodies over 64 KiB bypass it
  (`crates/llm-proxy-server/src/routes/core_pipeline.rs:182-185`) —
  rejection, not coalescing — plus opt-in forwarded-header trust, CORS, and
  panic catching (dedup corrected 2026-11-08 per research audit).
- Extras: offline tiktoken counting with chars/4 fallback
  (`crates/llm-proxy-core/src/token/tiktoken.rs:50,150`, cite corrected
  2026-11-08 per research audit); per-provider fixed-point
  USD per-token prices, estimation only
  (`crates/llm-proxy-core/src/provider_config.rs:472,500`); per-provider
  egress forward proxy socks5h/socks5/http/https
  (`crates/llm-proxy-provider/src/transport.rs:124-125,293`); env-var
  interpolation. A musl single-binary release workflow exists
  (`dist-workspace.toml`), but no release had shipped and the
  then-active native-tls pulled an unvendored `openssl-sys`, so the musl
  build was unproven (corrected 2026-11-08 per research audit; plan 03
  later made rustls the only TLS stack, removing the OpenSSL blocker).
  Config changes require a
  restart — `server.hot_reload=true` is rejected at load
  (`crates/llm-proxy-core/src/provider_config.rs:1531-1538`, verified
  2026-10-27).

## Auth and keys

llm-proxy ships two modes: loopback-only no-auth, or static client keys stored
as argon2id hashes in config (`config.toml.example:42-54`). Keys carry no
budgets, model/provider scoping, expiry, or issuance/rotation workflow;
changes require a config edit and restart.

Competitors treat caller-facing keys as managed records: LiteLLM issues virtual keys via `POST /key/generate` carrying max_budget, budget_duration, tpm/rpm limits, max_parallel_requests, model allowlists and aliases, layered under org→team→user→key budgets (https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/access_control); Bifrost `sk-bf-*` keys carry expiry, an instant `is_active` kill switch, and deny-by-default provider/model scoping (https://docs.getbifrost.ai/features/governance/virtual-keys); OpenRouter provisions credit-capped inference keys via a separate management-key API (https://openrouter.ai/docs/guides/overview/auth/management-api-keys.md); Vercel adds 12-hour OIDC project tokens and an unauthenticated compromised-secret revocation endpoint (https://vercel.com/docs/ai-gateway/authentication-and-byok/oidc); gproxy scopes keys by ordered allow/deny rules over provider, model glob, and operation (https://gproxy.leenhawk.com/guides/permissions/); one-api tokens support expiry, quota, IP ranges, and model allowlists (https://github.com/songquanpeng/one-api/blob/main/README.md); even small rust-llm-proxy ships issue/rotate/revoke endpoints (https://raw.githubusercontent.com/habibi-dev/rust-llm-proxy/master/README.md).
new: agentgateway swaps client virtual keys (`vk-...` Bearer, strict/optional/permissive modes) for provider keys, stores only key hashes, enforces per-key model allowlists (403 `model_not_allowed`), and its v1.7 UI auto-generates rotated values (https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/virtual-keys.md). new: Portkey ships dual-secret rotation — `POST /v2/api-keys/:id/rotate` keeps the old secret valid for `key_transition_period_ms` (min/default 30 min, key id unchanged) plus scheduled rotation policies (https://portkey.ai/docs/product/enterprise-offering/org-management/api-key-rotation). new: Respan (ex-Keywords AI) keys carry Read/Write/Gateway/Admin scopes, Production/Test environments, spend caps, expiry, and an admin-gated `/api/temporary-keys/` endpoint for short-lived minting (https://www.respan.ai/docs/documentation/admin/respan-api-keys). new: GPT-Load AccessKeys are scoped to groups/models/protocols with IP-CIDR restrictions and rotation that preserves id, permissions, and consumed quota (https://www.gpt-load.com/docs/access-keys). new: Tyk App API keys are created per App and must be activated by an administrator before they work — an approve/reject issuance workflow (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/apps.md). new: uni-api exposes `/v1/generate-api-key` behind admin-role keys (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). new: IBM ContextForge hashes gateway tokens (SHA-256) with jti-based revocation/replay detection, encrypts tool credentials AES-GCM, hashes passwords with Argon2id, and optionally resolves secrets from HashiCorp Vault (https://ibm.github.io/mcp-context-forge/latest/architecture/security-features/). new: Tyk stores vendor keys as `$SECRET/Name` references encrypted under `TYK_AI_SECRET_KEY` with env-reference allowlists (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/secrets.md).
Weaker comparators: the Portkey OSS gateway's only auth middleware is the admin-UI token (its middleware directory contains just adminAuth, cache, hooks, log, requestValidator — https://github.com/Portkey-AI/gateway/tree/main/src/middlewares), Higress key-auth stores plaintext credentials in YAML (https://higress.ai/en/docs/latest/user/plugins/authentication/key-auth), and Cloudflare gateway tokens are account-scoped, not restrictable to one gateway (https://developers.cloudflare.com/ai-gateway/configuration/bring-your-own-keys/). Upstream-credential encryption at rest ships at gproxy (AES-256-GCM under `GPROXY_MASTER_KEY`, https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md), LiteLLM (NaCl SecretBox keyed from `LITELLM_SALT_KEY` — spend logs and cached prompts stay plaintext — https://docs.litellm.ai/docs/proxy/security_encryption_faq.md, plus enterprise secret-manager backends https://docs.litellm.ai/docs/secret_managers/overview.md), Kong (Vault entity over Konnect Config Store/AWS/GCP/Azure/HashiCorp, https://developer.konghq.com/gateway/entities/vault/), new: Portkey (enterprise secret references to AWS Secrets Manager/Azure Key Vault/HashiCorp Vault resolved at runtime, plus AWS KMS envelope encryption over configs and integration keys — https://portkey.ai/docs/product/enterprise-offering/secret-references.md, https://portkey.ai/docs/product/enterprise-offering/kms.md), and new-api/GPT-Load (`CRYPTO_SECRET`/`ENCRYPTION_KEY`, https://github.com/QuantumNous/new-api, https://raw.githubusercontent.com/tbphp/gpt-load/main/README.md); llm-proxy has no equivalent (plaintext TOML, per the baseline above).

## Routing and resilience

llm-proxy routes by URL-path provider plus static alias resolution
(`crates/llm-proxy-core/src/provider_config.rs:392-394`) and performs no
retries, fallbacks, or cooldowns on the request path
(`crates/llm-proxy-provider/src/discovery.rs:160-170` is discovery-only).

Mature gateways ship retry + fallback + cooldown as core machinery. LiteLLM: six routing strategies (simple-shuffle weighted, least-busy, usage-, latency-, and cost-based; https://docs.litellm.ai/docs/routing), per-exception RetryPolicy, TTL cooldown cache tripped on 429 or >50% failure rate, typed fallback chains (general / context-window / content-policy, max_fallbacks default 5 and cooldown default 5s, `ref/litellm/litellm/constants.py:13,37-43`; https://docs.litellm.ai/docs/proxy/reliability); new: opt-in mid-stream fallback continuation on prefill-capable deployments via assistant prefill, constrained to plain-text partial output (https://docs.litellm.ai/docs/proxy/config_settings). Portkey: JSON config objects compose fallback/loadbalance/conditional strategies recursively; retries cap cumulative wait at 60s and honor Retry-After (https://portkey.ai/docs/product/ai-gateway/fallbacks, https://portkey.ai/docs/product/ai-gateway/automatic-retries); new: a `cb_config` circuit breaker per strategy with cooldown interval ≥30s (https://portkey.ai/docs/product/ai-gateway/circuit-breaker). Bifrost: a FailureClass taxonomy (credential/quota/model_access/rate_limit/caller_fault/transient/timeout…) drives retry-same-key vs rotate-key vs stop; per-request fallback arrays each run as a fresh request with their own retry budget; TTFT timeouts fall back, post-output errors do not (https://docs.getbifrost.ai/features/fallbacks, https://github.com/maximhq/bifrost/blob/dev/core/schemas/failureclass.go); new: plus CEL routing rules with scope precedence, a session-affinity provider pin, and an LLM complexity router exposing `complexity_tier` to rules (https://docs.getbifrost.ai/providers/routing-rules.md, https://docs.getbifrost.ai/providers/session-affinity.md, https://docs.getbifrost.ai/features/complexity-router.md). Cloudflare: up to 5 attempts with per-request backoff control plus ordered fallbacks and versioned dynamic routing (https://developers.cloudflare.com/ai-gateway/configuration/request-handling/, https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/). Agent Router: priority-ordered backends where retries walk the priority chain (https://theagentrouter.ai/docs/capabilities/traffic/provider-fallback). Kong: seven load-balancing algorithms and a circuit breaker (https://developer.konghq.com/plugins/ai-proxy-advanced/). Higress: API-key pool failover with health checks and cooldown restore (https://github.com/higress-group/higress/blob/main/plugins/wasm-go/extensions/ai-proxy/README_EN.md). Vendored oc-go-cc ships three-state per-model circuit breakers and fallback chains (`ref/oc-go-cc/internal/router/fallback.go:15-115`). gproxy routes gateway-defined model names over member providers with operator-set weights and fallback tiers (https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md).
Also shipping retry/fallback: OpenRouter model-array fallbacks and price-weighted provider balancing (https://openrouter.ai/docs/guides/routing/model-fallbacks.md), Vercel health-aware ordering plus model fallback chains with per-provider BYOK timeouts (https://vercel.com/docs/ai-gateway/models-and-providers/model-fallbacks, https://vercel.com/docs/ai-gateway/models-and-providers/provider-timeouts), TensorZero three-level retry/fallback hierarchy (https://github.com/tensorzero/tensorzero/blob/main/docs/gateway/guides/retries-fallbacks.mdx), one-api automatic retry and new-api configurable cross-channel retry, off by default (https://github.com/songquanpeng/one-api/blob/main/README.md, https://github.com/QuantumNous/new-api/blob/main/model/request_policy.go), gproxy retries within one provider's credentials before moving on (https://gproxy.leenhawk.com/introduction/architecture/), and llm-api-key-proxy same-key server-error retries under a global deadline (`ref/llm-api-key-proxy/src/rotator_library/README.md`).
new: Second-wave additions — AWS Bedrock AgentCore Gateway: fallback blocks declaring 1–2 ordered targets, triggers on status-code classes, CONNECTION_ERROR/FIRST_BYTE_TIMEOUT/UNROUTABLE, a default-on circuit breaker (open state skips and does not bill), no fallback after a stream starts (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-rules-fallback.html). It also offers weighted (2-way) route/config-bundle traffic splits in its rule engine (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-rules.html). Tyk: per-provider failover waterfalls (max 10 rungs; triggers 408/429/5xx/timeout/connect, never 401/403; `X-Tyk-Served-LLM`/`X-Tyk-Failover` headers; pre-first-token only) (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/llm-management.md); its Model Router load-balances across vendor pools by glob model patterns, weighted or round-robin (Enterprise, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/model-router.md). F5: per-service retries (default codes 429/500/502/503/504, Retry-After priority over backoff, 1.5x growth capped 60s) plus a single named `fallbackServiceName` (https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/configuration/configure-services.html). agentgateway: per-route retries that always stay on the same backend — cross-model failover is delegated to virtual-model eviction groups (https://agentgateway.dev/docs/standalone/latest/documentation/configuration/resiliency/retries.md, https://agentgateway.dev/docs/standalone/latest/documentation/llm/virtual-models.md). uni-api: `AUTO_RETRY` up to 100 attempts with tiered channel/key cooldowns plus six per-key channel scheduling algorithms incl. success-rate-aware `smart_round_robin` and key rotation/cooldown knobs (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). GPT-Load: tier-priority weighted scheduling (group weight × credential weight) under a shared global retry budget that only moves down tiers, no switch after committed stream output (https://www.gpt-load.com/docs/internals/scheduling). Respan: retries on by default (≤3 attempts) with ordered `fallback_models`; validation/context/credential failures never retried (https://www.respan.ai/docs/documentation/features/gateway/retries).

## Rate limiting and quotas

llm-proxy rate limits authenticated callers per key id and unauthenticated
callers per client IP — a burst-tolerant token bucket with continuous
refill (`crates/llm-proxy-server/src/middleware.rs:117-129`) keyed at
`crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176` (per-key RPM
since 2026-10-13, commit 499b22a) — exempting loopback
(`config.toml.example:14-21`). Token-based (TPM) limits and concurrency
caps are what remain absent (corrected 2026-10-28 after code audit).

Competitors scope limits to identity, not IP. LiteLLM enforces max_parallel_requests/rpm/tpm per key, user, and team in one batched pre-call check, with a reserve-then-reconcile TPM path (estimate prompt tokens + max_tokens up front, settle actuals after, refund on failure) (https://docs.litellm.ai/docs/proxy/users). Kong limits on token kinds or cost with policy-matched ceilings at consumer, IP, header, path, model, and provider level (https://developer.konghq.com/plugins/ai-rate-limiting-advanced/). Agent Router charges token budgets (typed token kinds or CEL expressions) into Redis-backed windows and never interrupts an admitted stream (https://theagentrouter.ai/docs/capabilities/traffic/usage-based-ratelimiting). Higress counts tokens per consumer/header/IP over Redis (https://higress.ai/en/docs/latest/user/plugins/ai/api-consumer/ai-token-ratelimit). Bifrost applies request and token limits at provider-config, key, team, and customer level with typed 401/402/403/429 errors (https://docs.getbifrost.ai/features/governance/budget-and-limits). gproxy rate limits per user/key including concurrency (https://gproxy.leenhawk.com/guides/permissions/). new-api adds per-IP, per-group, and model-level limits (https://docs.newapi.ai/en/docs/guide/feature-guide/admin/system-setting-advanced).
new: Second-wave additions — agentgateway: request- and token-type limits keyed by CEL expressions (`jwt.sub`, `apiKey.user`, `llm.requestModel`), local in-memory LRU buckets (≤65,536/rule) or a remote Envoy Rate Limit gRPC service; token limits enforce in two phases when the backend sets `tokenize: true` (upfront estimate blocks, otherwise post-hoc) (https://agentgateway.dev/docs/standalone/latest/documentation/configuration/resiliency/rate-limits.md). AWS AgentCore: dimension-keyed limits (targetName, toolName, qualifiedModelId, JWT claims, IAM principal; immutable dimension keys, most-specific-wins) over requests/sec, tokens/min with pre-estimate plus provider-reported actuals, and concurrent connections — fail-open by default, explicitly "not a security boundary" (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-rate-limits.html). Traefik Hub: `ai-rate-limit` token bucket (burst-tolerant) vs `ai-quota` sliding window (no bursts), input/output/total-token or USD-cost limits, per-consumer `sourceCriterion`, pre-request body-length/4 estimate rejection, Redis-backed fail-open (https://doc.traefik.io/traefik-hub/ai-gateway/middlewares/token-rate-limit). Azure APIM: one `llm-token-limit` policy combining tokens-per-minute (429) and long-period token quotas over hourly-to-yearly fixed UTC windows (403), keyed on any counter-key expression, with optional `estimate-prompt-tokens` pre-call rejection (https://learn.microsoft.com/en-us/azure/api-management/llm-token-limit-policy). GPT-Load: per-AccessKey RPM plus a concurrency cap (https://www.gpt-load.com/docs/access-keys). uni-api: stacked windows like `15/min,10/day` per key, per-model substring limits, and per-upstream-key knobs (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). Respan: layered per-key cost/request/token limits plus customer (402) and org-plan (429) tiers (https://www.respan.ai/docs/documentation/features/gateway/limits). TensorZero shipped scoped rate-limit rules over inferences/tokens/cost with priority resolution and Valkey/Postgres-persisted state before archival (https://raw.githubusercontent.com/tensorzero/tensorzero/main/docs/operations/enforce-custom-rate-limits.mdx).
Weaker comparators: Cloudflare limits per gateway only (https://developers.cloudflare.com/ai-gateway/features/rate-limiting/), one-api's global IP limit excludes the relay data path (https://github.com/songquanpeng/one-api/blob/main/README.en.md), Vercel does not rate-limit paid-tier requests (https://vercel.com/docs/ai-gateway/rate-limits), and Tyk ships no core RPM/TPM limiter on the LLM path (rate limiting arrives as plugins, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/plugins-overview.md).

## Cost tracking

llm-proxy computes estimated USD cost from operator-configured per-token
prices (`crates/llm-proxy-core/src/provider_config.rs:472`) and records usage
in the SQLite ledger with body digests
(`crates/llm-proxy-storage/src/sqlite.rs:105-131`). Prices are neither
shipped nor synced — a grep for pricing code across `crates/` and `apps/`
(run 2026-10-14) finds per-token prices only in
`crates/llm-proxy-core/src/provider_config.rs` and their consumer
`crates/llm-proxy-server/src/routes/core_pipeline.rs`, and discovery fetches
model lists, not prices (`crates/llm-proxy-provider/src/discovery.rs`) — and
no code path enforces a spend limit (those same two files are the only price
consumers).

Competitors maintain pricing databases wired into budgets: LiteLLM computes per-request cost against a 2,752-model price map, writes full spend logs, exports spend to Prometheus, and supports pre-call budget reservation (https://docs.litellm.ai/docs/proxy/cost_tracking); Portkey attributes cost from the open Portkey-AI/models database, 2,300+ models pushed 2026-10 (https://github.com/Portkey-AI/models); Helicone prices from a 300+ model repository (https://docs.helicone.ai/references/how-we-calculate-cost.md); agentgateway prices by token kind (input/output/cacheRead/cacheWrite/reasoning/audio) with context-length tiers and models.dev catalog import (https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/costs.md); Kong models provider billing including TTL-tiered cache-write rates (https://developer.konghq.com/ai-gateway/model-cost-management/). Budget enforcement wired to prices ships at OpenRouter, Vercel, Cloudflare, and LangSmith (https://openrouter.ai/docs/guides/best-practices/spend-controls.md, https://vercel.com/docs/ai-gateway/observability-and-spend/budgets, https://developers.cloudflare.com/ai-gateway/features/spend-limits/, https://docs.langchain.com/langsmith/llm-gateway-spend-policies), and BricksLLM pioneered per-key USD caps in 2023 (https://raw.githubusercontent.com/bricks-cloud/BricksLLM/main/README.md). gproxy enforces USD budgets on users, keys, orgs, and teams, settled after the upstream call (https://gproxy.leenhawk.com/guides/permissions/).
new: Budget shape and semantics (second wave) — Bifrost: budget arrays `{max_limit USD, reset_duration 1m–1Y}` with optional UTC calendar alignment, fiscal-quarter starts, temporary overrides, and reset-usage actions; enforced cumulatively at customer→team→key→provider-config scope (https://docs.getbifrost.ai/features/governance/budget-and-limits.md). LiteLLM: stackable multi-window `budget_limits` (e.g. $10/24h + $100/30d), calendar-aligned resets with IANA timezone and `budget_reset_time`, soft budgets, temporary raises, and tag-based budgets attached per request header or W3C baggage (https://docs.litellm.ai/docs/proxy/users, https://docs.litellm.ai/docs/proxy/budget_reset_and_tz, https://docs.litellm.ai/docs/proxy/tag_budgets). agentgateway: per-key USD-or-token budgets with rolling windows, Block (429 `budget_exceeded`) or Audit modes, multiple independent windows per key, requiring its SQLite/Postgres config DB (https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/budget-limits/per-key.md). Portkey enterprise: usage policies with AND-conditions and `group_by` counters (per-user, per-model, per-key, even per-MCP-server) with weekly/monthly resets — breach returns 412 (https://portkey.ai/docs/product/enterprise-offering/budget-policies). Cloudflare: spend-limit rules scoped by provider/model/metadata with split-by-value buckets (per-user budgets via `cf.user_id`) and a budget-exceeded fallback to a cheaper dynamic-route model; LiteLLM likewise routes within provider budgets — neither is the only budget→fallback action (corrected 2026-11-08 per research audit) (https://developers.cloudflare.com/ai-gateway/features/spend-limits/, https://docs.litellm.ai/docs/routing). OpenRouter: layered guardrail/workspace/key budgets with in-flight holds that pre-reject oversized requests (`weight_exceeds_budget` vs retryable `in_flight_budget_exhausted`, both 402) (https://openrouter.ai/docs/api-reference/limits). new-api: subscription quota packages plus a prepaid wallet — a monetization loop, not just a cap (https://github.com/QuantumNous/new-api). Tyk: monthly budgets at LLM/App/team level with a ~30s budget pulse pushing block-lists to edge gateways; enforcement and team pools are Enterprise-only — Community tracks but never blocks (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/budgeting.md). Respan: per-key spend caps with alert/Block thresholds and customer-level budgets returning 402 (https://www.respan.ai/docs/documentation/features/gateway/limits). GPT-Load: cost quotas as never-resetting totals or periodic windows from one minute to one year — explicitly soft enforcement, charged after completion (https://www.gpt-load.com/docs/access-keys). uni-api: per-key dollar-credit quotas with configurable 30-day default windows, 0 blocks use (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). Traefik Hub: `costLimit` in USD alongside its token limits (https://doc.traefik.io/traefik-hub/ai-gateway/middlewares/token-rate-limit). new: Status-code taxonomy has not converged — Bifrost returns 402, Portkey 412, Respan 402/429, LiteLLM a documented 400/401/429 mix — so the choice must be deliberate (https://github.com/maximhq/bifrost/blob/dev/plugins/governance/main.go, https://portkey.ai/docs/api-reference/inference-api/error-codes, https://docs.litellm.ai/docs/proxy/users). new: agentgateway also attributes invoice-grade cost into provider bills — Bedrock per-request STS role assumption so session names land in CloudTrail/CUR, Vertex billing labels into Cloud Billing export (https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/attribution.md).

## Caching

llm-proxy has no response cache. Its opt-in dedup window — off by default —
409-rejects an identical path+body seen within a fixed window of the first
sighting, keyed without the caller (`crates/llm-proxy-server/src/routes/core_pipeline.rs:182-185`):
a duplicate past the window can reach the origin while the first request is
still running, and a sequential repeat after completion is refused — pressure
control, neither a cache nor coalescing (corrected 2026-11-08 per research
audit).

Exact-match caching is near-universal: LiteLLM (8+ backends; SHA-256 over recognized params; streams reassembled then replayed synthetically; https://docs.litellm.ai/docs/proxy/caching), Portkey (simple + semantic, TTL bounds 60s–90d, namespacing; https://portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic), Cloudflare (SHA-256 over provider+endpoint+model+auth+body, TTL 60s–1mo; https://developers.cloudflare.com/ai-gateway/features/caching/), Helicone (edge cache, plus an OSS Rust proxy hashing seed+path+body with RFC 7234 freshness over Moka or Redis, buffered bodies, 2xx-only; https://docs.helicone.ai/features/advanced-usage/caching.md, https://github.com/helicone/ai-gateway), and OpenRouter opt-in with free hits (https://openrouter.ai/docs/guides/features/response-caching.md). Higress ai-cache's base mode caches chat completions in Redis keyed by GJSON paths over the request body (exact string match, streaming replay included); new: a semantic mode adds a DashVector collection plus an embedding provider through the console (https://higress.ai/en/docs/latest/user/plugins/ai/api-provider/ai-cache, https://higress.ai/en/docs/ai/scene-guide/semantic-cache). new: Respan caches exact "messages only" matches with per-customer/per-model scoping and a hit/savings dashboard (https://www.respan.ai/docs/documentation/features/gateway/caching). new: Tyk delivers caching as plugins — a community exact-match plugin and an enterprise semantic plugin (cosine ≥0.95 over Redis/memory vector stores) (https://tyk.io/docs/ai-management/ai-studio/semantic-caching). new: the vLLM Production Stack ships an alpha, feature-gated semantic cache (sentence-transformer, threshold 0.95) (https://raw.githubusercontent.com/vllm-project/production-stack/main/src/vllm_router/experimental/README.md). Semantic caching also ships at Bifrost (hash + cosine similarity over vector stores; https://docs.getbifrost.ai/features/semantic-caching), Kong (https://developer.konghq.com/plugins/ai-semantic-cache/), and Azure APIM via RediSearch (https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities).
Single-flight is rarer than it looks: Cloudflare documents the duplicate-origin race and OpenRouter documents that simultaneous identical requests are not coalesced (URLs above); the Portkey OSS, Helicone OSS, and LiteLLM cache implementations examined at source level contain no in-flight dedup; coalescing was not verified either way for the remaining products. llm-proxy's dedup window (off by default; a fixed window from first sighting, caller-blind, bodies over 64 KiB exempt) rejects some duplicates outright rather than serving them, so the second caller fails when it fires — and duplicates outside the window always reach the origin (corrected 2026-11-08 per research audit). new: Verified wave-2 absences — agentgateway ships no LLM response cache (homepage advertises "semantic caching" but the Kubernetes API reference contains no such resource; https://agentgateway.dev/), and AWS AgentCore, Tyk core, GPT-Load, and uni-api document none. The Portkey cache counted here is the hosted product; its OSS gateway cache is exact-match and in-memory only (corrected 2026-11-08 per research audit).

## Guardrails

llm-proxy ships no content inspection, redaction, or moderation.

LiteLLM integrates ~50 guardrail services across seven hook phases including pre-call, post-call, and MCP phases (https://docs.litellm.ai/docs/proxy/guardrails/quick_start, https://docs.litellm.ai/release_notes); new: it also ships reversible Presidio PII masking with `output_parse_pii` re-insertion (https://docs.litellm.ai/docs/tutorials/presidio_pii_masking.md). Portkey runs 20+ deterministic checks plus LLM checks (PII, moderation, language, gibberish) with deny=HTTP 446 / flag=246 semantics that can drive fallback orchestration (https://portkey.ai/docs/product/guardrails); new: its pre-LLM PII redaction is irreversible by design (https://portkey.ai/docs/product/guardrails/pii-redaction). Cloudflare ships a fixed taxonomy (S1–S14 hazards via Llama Guard 3 8B plus P1 prompt injection via Prompt Guard 2) with per-category flag/block (https://developers.cloudflare.com/ai-gateway/features/guardrails/) plus DLP scans shared with Cloudflare One (https://developers.cloudflare.com/ai-gateway/features/dlp/). Guardrails, DLP, Auto Router and spend limits are Cloudflare Beta features (corrected 2026-11-08 per research audit). Kong's ai-sanitizer redacts 20+ PII entity types bidirectionally (https://developer.konghq.com/plugins/ai-sanitizer/). Bifrost (enterprise) enforces CEL rules, Gitleaks secret detection, and LLM-as-judge policies (https://docs.getbifrost.ai/enterprise/guardrails). LangSmith's gateway redacts PII/secrets in-path with placeholder restoration, including streams (https://docs.langchain.com/langsmith/llm-gateway-data-policy). Helicone's LLM Security screens prompt injection via Meta Prompt Guard (advanced: Llama Guard), currently OpenAI models only (https://docs.helicone.ai/features/advanced-usage/llm-security.md). new-api screens sensitive words on prompts and/or responses (https://github.com/QuantumNous/new-api/blob/main/model/request_policy.go); new: that screening became off-by-default in rc.45 (https://github.com/QuantumNous/new-api/releases). Higress delegates moderation to Aliyun's paid service (https://higress.ai/en/docs/latest/user/plugins/ai/api-provider/ai-security-guard). Cloud vendors: Azure AI Content Safety (https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities), AWS Bedrock Guardrails (https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
new: Second-wave additions — Tyk: Tengo-script filters with vendor-agnostic redaction/quarantine helpers over normalized messages (OpenAI/Anthropic/Google/Bedrock shapes) calling Lakera, Azure Content Safety, Presidio, Bedrock Guardrails, or HTTP classifiers, with streaming re-evaluation and tool-result injection screening from 2.3 (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/filters.md, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/guardrails.md). F5: a processor chain (prompt-injection ML detector, deterministic Data Security PII matcher incl. an `mcp` matcher, language-id, system-prompt injection) with modify/reject/annotate outcomes (https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/processors/processors-overview.html). agentgateway: ordered prompt guards — regex with built-in PII detectors (mask/reject/audit), OpenAI moderation, Bedrock/Azure/Google vendor guards, generic webhooks, failClosed default; masking is reject-only on streams (https://agentgateway.dev/docs/standalone/latest/documentation/llm/prompt-guards/overview). GPT-Load: experimental JEV guardrails via an external decision service plus global RE2 redaction with reversible per-AccessKey encryption restored in responses and streams (https://www.gpt-load.com/docs/advanced/guardrails, https://www.gpt-load.com/docs/advanced/redaction). uni-api: optional omni-moderation screening of user messages (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). Respan: free 8-category request blocking via its span-01-security model, org-wide only (https://www.respan.ai/docs/documentation/features/gateway/guardrails). AWS AgentCore: deny-by-default Cedar policies on MCP `tools/call` with Bedrock Guardrails confidence-score conditions (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/use-gateway-with-policy.html).
Agent Router documents no guardrails as of October 2026 — its security surface is transport/auth (https://theagentrouter.ai/docs/capabilities/security/); that absence is per its docs, not a code audit.

## Observability

llm-proxy exposes atomic counters in-process, injects request ids, logs
plain/JSON, and shuts down gracefully; it collects up to 1000 latency
samples (`crates/llm-proxy-core/src/metrics.rs:34-35`) and computes
percentiles in-process (`:210-239` — `Snapshot.latencies` plus
`calculate_percentile`/`calculate_p95`/`calculate_p99`) — but has no
`/metrics` scrape endpoint, no TTFT measurement, and no trace export, and
the percentiles are never exposed (corrected 2026-10-28 after code audit).

Competitors standardize on Prometheus + OTel GenAI conventions. LiteLLM exports request/latency/TTFT/queue/overhead histograms, token and spend counters with identity labels, budget-remaining gauges (https://docs.litellm.ai/docs/proxy/prometheus), and OTel trace export with `traceparent` propagation via its callback pipeline (https://docs.litellm.ai/docs/proxy/logging). agentgateway — also Rust/tokio — implements the OTel GenAI semconv metric names directly (`gen_ai.server.request.duration`, `time_to_first_token`, `time_per_output_token`, token-usage histograms) on :15020 plus OTLP tracing with per-request custom labels and metric removal via CEL (https://agentgateway.dev/docs/standalone/latest/documentation/observability/metrics/reference/, https://agentgateway.dev/docs/standalone/latest/documentation/observability/metrics/overview). Bifrost exposes per-provider/model/virtual-key cost, token, latency-histogram, TTFT, and key-health metrics with SQLite/Postgres-backed logs and a web UI (https://docs.getbifrost.ai/features/observability/prometheus, https://docs.getbifrost.ai/features/observability/default), and an `llm.call` OTel span per request with per-attempt spans exported over OTLP HTTP or gRPC (https://docs.getbifrost.ai/features/otel.md). Kong's Prometheus `ai_metrics` exposes request/cost/token counters (token_type labels incl. cache-write TTL splits), LLM latency histograms, and TTFT under the gen_ai conventions (https://developer.konghq.com/ai-gateway/monitor-ai-llm-metrics/). Cloudflare adds OTLP export under GenAI semconv and per-user spend anomaly detection (https://developers.cloudflare.com/ai-gateway/observability/otel-integration/). Vercel ships per-attempt routing logs and OTel trace drains (https://vercel.com/docs/ai-gateway/observability-and-spend/logs, https://vercel.com/docs/ai-gateway/observability-and-spend/trace-drains). OpenRouter broadcasts OTLP traces to ~20 destinations (https://openrouter.ai/docs/guides/features/broadcast/webhook.md). TensorZero shipped three Prometheus metrics plus OTLP traces before archival (https://raw.githubusercontent.com/tensorzero/tensorzero/main/docs/operations/export-prometheus-metrics.mdx); the vLLM Production Stack ships Prometheus/Grafana/OTel in-chart (https://github.com/vllm-project/production-stack); Higress ai-statistics emits per-route/cluster/model/consumer token and latency counters (https://higress.ai/en/docs/latest/user/plugins/ai/api-o11y/ai-statistics); vendored oc-go-cc computes p95/p99 at /health (`ref/oc-go-cc/internal/metrics/metrics.go:132-154`). The OTel client-inference span shape is standardized at https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/client-inference.md.
new: Second-wave additions — Tyk serves a secure-by-default Prometheus `/metrics` exporting OTel GenAI semconv metrics (`gen_ai_server_request_duration_seconds`, time-to-first-token, time-per-output-token) plus `aistudio_*` governance counters, a Server-Timing header splitting gateway overhead from upstream latency, and opt-in OTel span export with always-on W3C traceparent propagation (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/observability.md). IBM ContextForge ships OTLP tracing to eight backends plus a JWT-protected Prometheus endpoint (https://ibm.github.io/mcp-context-forge/latest/manage/observability/). AWS AgentCore publishes CloudWatch metrics (Latency measured to first response token, p50/p90/p99, throttles, per-target execution time) and vended logs including request/response bodies with OTel-compatible spans (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-gateway-metrics.html). Helicone computes TTFT and p50/p95/p99 latency in its analytics dashboards rather than exporting histograms (docs mirror: https://helicone-helicone-7.mintlify.app/observability/metrics).
new: Streaming SLO mechanics (second wave) — LiteLLM sends SSE `: ping` keep-alives on silent streams (`keepalive_seconds` 1–300), cancels upstream on client disconnect (`cancel_on_disconnect`, logged 499 — documented for non-streaming calls, so it is not a streaming-cancel mechanism; corrected 2026-11-08 per research audit), and can embed usage in every chunk (`always_include_stream_usage`) for billing when the final chunk never arrives (https://docs.litellm.ai/docs/proxy/config_settings). OpenRouter documents per-provider stream-cancel billing semantics (abort propagates at ~24 providers; Bedrock/Groq/Google/Mistral keep processing and bill) and a final usage-chunk accounting frame with in-band mid-stream error events (https://openrouter.ai/docs/api_reference/streaming). oc-go-cc emits a `:keepalive` comment every 3s during streaming (`ref/oc-go-cc/internal/handlers/messages.go:239-266`); uni-api ships an SSE heartbeat interval (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). Keep-alive itself is parity: llm-proxy already emits 3s SSE keep-alive comments (`crates/llm-proxy-server/src/routes/core_pipeline.rs:701,883`). None of the products whose pages are cited above documents SLO machinery (burn-rate alerts, error budgets) — percentiles are left to PromQL, and TTFT measurement points differ across products.
Payload capture is a separate axis llm-proxy declines: its ledger keeps digests only (`crates/llm-proxy-storage/src/sqlite.rs:130-131`), while Helicone records full bodies with per-request omit headers (https://docs.helicone.ai/features/advanced-usage/omit-logs.md), LiteLLM stores messages with redaction controls (https://docs.litellm.ai/docs/proxy/logging), Cloudflare logs payloads with per-request opt-out (https://developers.cloudflare.com/ai-gateway/observability/logging/). new: Cloudflare Logpush additionally exports bodies customer-key-encrypted — a fresh AES-256-GCM key per log wrapped with the customer's RSA-4096 public key (https://developers.cloudflare.com/ai-gateway/observability/logging/logpush/). OpenRouter retains prompt/completion content only behind explicit opt-in (https://openrouter.ai/docs/guides/features/logs.md). new: Per-request metadata/tag attribution — Helicone `Helicone-Property-*` headers, LiteLLM body `metadata.tags`/`x-litellm-tags` plus User-Agent auto-tagging, Portkey `x-portkey-metadata`, Cloudflare `cf-aig-metadata` (deliberately excluded from the cache key), Vercel `providerOptions.gateway.user`/`tags`, Glama `x-glama-metadata` — is the dashboard grouping key everywhere; llm-proxy records only the key id in usage events.

## MCP

llm-proxy has no MCP client, server, or gateway functionality.

The MCP wire protocol standardized Streamable HTTP + OAuth 2.1 resource-server auth in the 2025-11-25 spec (https://modelcontextprotocol.io/specification/2025-11-25/basic/transports, https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization); the 2026-07-28 revision — final, not a release candidate (corrected 2026-11-08 per research audit) — removes sessions and adds gateway-friendly per-operation routing headers (https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate).
Shipped gateways: LiteLLM routes all MCP tool traffic through one endpoint with six-level permissions and per-tool allowlists (https://docs.litellm.ai/docs/mcp, https://docs.litellm.ai/docs/mcp_control); Cloudflare MCP portals aggregate up to 80 HTTP MCP servers behind Access policies with managed upstream OAuth (https://developers.cloudflare.com/cloudflare-one/access-controls/ai-controls/mcp-portals/); agentgateway federates MCP servers behind one endpoint with CEL authorization per MCP method (https://agentgateway.dev/docs/standalone/latest/documentation/mcp/mcp-authz.md); new: v1.7 adds an ExtMCP guardrail processor chain and an inline CEL processor (https://agentgateway.dev/docs/standalone/latest/documentation/mcp/guardrails/about.md); Bifrost re-exposes federated tools at /mcp with virtual-key-scoped filtering (https://docs.getbifrost.ai/mcp/overview); new: tool execution there is explicit by default (a separate `/v1/mcp/tool/execute` call) with opt-in agent auto-approval and a sandboxed code mode across 3+ servers (https://docs.getbifrost.ai/mcp/overview.md). Kong bridges REST↔MCP in four modes (https://developer.konghq.com/plugins/ai-mcp-proxy/); new: Kong 2.1.0 added the stateless 2026-07-28 revision with `ttlMs`/`cacheScope` cache hints and W3C trace-context to tool calls (https://developer.konghq.com/ai-gateway/changelog/). Azure APIM exports managed REST APIs as MCP servers (https://learn.microsoft.com/en-us/azure/api-management/mcp-server-overview); AWS Bedrock AgentCore Gateway converts APIs/Lambda into MCP tools with per-tool credential injection (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html); new: it also serves stateful MCP sessions with elicitation/sampling and semantic tool search over the catalog (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-features.html). IBM ContextForge federates MCP/A2A/REST/gRPC (https://github.com/IBM/mcp-context-forge); Glama runs a hosted MCP gateway with per-tool ACLs (https://glama.ai/mcp/gateway); Higress hosts MCP servers in-gateway (https://higress.cn/en/docs/ai/mcp-quick-start/).
new: Second-wave additions — Tyk ships MCP twice: Tyk Gateway v5.13+ proxies remote MCP servers with OAuth 2.1, per-key tool allowlists, per-tool rate limits and circuit breakers (https://tyk.io/docs/ai-management/mcp-gateway/overview.md), and AI Studio exposes its own tools as MCP servers at `/tools/{slug}/mcp` (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/tools.md). Respan runs a remote MCP server exposing logs, traces, prompts and monitors to AI tools (https://www.respan.ai/docs/documentation/mcp). OpenRouter hosts an MCP server at mcp.openrouter.ai whose OAuth flow issues spend-capped 7-day keys (https://openrouter.ai/docs/guides/overview/mcp-server.md).

## Protocol breadth

llm-proxy speaks two client formats (OpenAI Chat, Anthropic Messages) and
four upstream formats, plus count_tokens; embeddings/audio/images/batch/
realtime are scoped future work (docs/research/endpoints.md).

Breadth leaders: LiteLLM proxies chat, Responses, batches, files, assistants, images, audio, rerank, RAG, realtime, OCR, containers, fine-tuning, A2A, and search families (https://docs.litellm.ai/release_notes); Portkey OSS ships handlers for completions, embeddings, batches, files, fine-tunes, images, speech, transcriptions, translations, realtime WebSocket, Responses, and Anthropic messages + count_tokens (https://github.com/Portkey-AI/gateway/tree/main/src/handlers); OpenRouter serves chat, Responses, and Anthropic Messages plus embeddings, rerank, image/video generation, TTS/STT, batch, files, containers (https://openrouter.ai/docs/_llms/api-reference.txt); Vercel serves the same three client formats plus embeddings, rerank, and Decisions (https://vercel.com/docs/ai-gateway/sdks-and-apis); new: Vercel organizes the surface as nine modalities including video generation and a typed Decision API (https://vercel.com/docs/ai-gateway/modalities). new: Cloudflare's hosted REST API exposes `/ai/v1/chat/completions`, `/ai/v1/responses` (OpenAI Responses schema), and `/ai/v1/messages` (https://developers.cloudflare.com/ai-gateway/usage/rest-api/). new: the LangSmith LLM Gateway (beta) serves `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` through one API key (https://docs.langchain.com/langsmith/llm-gateway). new: Helicone's gateway accepts the OpenAI Responses format at `/v1/responses` for both OpenAI and Anthropic models (https://docs.helicone.ai/gateway/concepts/responses-api.md). new: agentgateway's client-facing route types include OpenAI Chat Completions, Anthropic Messages, OpenAI Responses, and Gemini generateContent, translated between format families in both directions (https://agentgateway.dev/docs/standalone/latest/documentation/llm/about.md). new: AWS AgentCore inference accepts chat/completions, messages, responses, and Bedrock converse/InvokeModel wire formats with cross-dialect translation (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-targets-inference.html). new-api accepts four client protocols (OpenAI, Responses, Anthropic, Gemini) with cross-conversion and realtime WebSocket (https://github.com/QuantumNous/new-api); new: it also ships gateway-provided `/v1/tools/search` and `/v1/tools/fetch` billed per call (https://github.com/QuantumNous/new-api/releases). Agent Router accepts four client formats under configurable prefixes and translates to ~20 providers (https://theagentrouter.ai/docs/capabilities/llm-integrations/supported-endpoints); Higress ai-proxy auto-detects OpenAI vs Claude clients and targets ~30 provider backends (https://higress.ai/en/docs/latest/plugins/ai/api-provider/ai-proxy); LLMG auto-registers 70+ providers from environment variables (https://modpotatodotdev.github.io/LLMG/providers/all/); Kong passes requests upstream in native provider format across 17 providers (https://developer.konghq.com/plugins/ai-proxy/). new: Tyk fronts four surfaces (vendor-native passthrough, OpenAI-format translation, an Anthropic→Bedrock bridge, legacy), with governed batches/Files/Responses arriving only in the unreleased 2.3.0 line (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/proxy.md, https://github.com/TykTechnologies/ai-studio/blob/main/CHANGELOG-v2.3.0.md). new: uni-api proxies chat, responses (+compact), messages, images, embeddings, audio speech/transcriptions, moderations, and search across 16+ upstream engines (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). new: GPT-Load accepts OpenAI Chat/Responses/Images/Embeddings/Rerank, Mistral-native OCR/audio, Anthropic Messages, and Gemini incl. `:embedContent` across 35 channels (https://raw.githubusercontent.com/tbphp/gpt-load/main/README.md). new: Respan serves OpenAI chat, OpenAI Responses, and Anthropic-native surfaces over 1000+ models (https://www.respan.ai/docs/documentation/features/gateway/gateway-quickstart). new: F5's client schema is OpenAI-only (a chat-completions subset plus /v1/models) with Anthropic upstream-executor only (https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/configuration/configure-services.html). new: Martian's hosted gateway (live as of 2026-10-27) fronts 312 models from 49 providers at `/v1/chat/completions`, `/v1/responses`, and `/v1/messages` (https://api.withmartian.com/v1/models, https://docs.withmartian.com/api-reference/endpoints); Not Diamond remains a routing-decision API, not a proxy (https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post.md). Deliberately narrow: Langfuse's Rust ai-gateway relays only OpenAI + Anthropic, byte-preserving (https://github.com/langfuse/langfuse/tree/main/ai-gateway).

Translation fidelity is only partially comparable from public docs. llm-proxy normalizes 10 content kinds (`docs/protocol.md:80-91`) and records normalized usage including cache/reasoning token splits (`crates/llm-proxy-storage/src/sqlite.rs:116-121`). agentgateway normalizes cached-prompt-token accounting across providers (Anthropic/Bedrock exclude, OpenAI/Azure/Gemini include them; https://agentgateway.dev/docs/standalone/latest/documentation/configuration/resiliency/rate-limits.md) and reports both normalized and provider-raw token counts for billing parity (https://agentgateway.dev/docs/standalone/latest/documentation/llm/observability.md). TensorZero normalizes structured outputs to native constrained decoding or forced tool calls (https://github.com/tensorzero/tensorzero/blob/main/docs/gateway/generate-structured-outputs.mdx). Per-field coverage of provider-specific options (reasoning controls, tool-choice variants, response formats) was not audited for any product here.

## Client BYOK, data residency, saturation, and SDKs

new: Axes added in this revision that a gateway comparison expects.

- Client-supplied upstream keys (BYOK passthrough): Cloudflare accepts
  provider keys per request with aliased Secrets-Store selection
  (https://developers.cloudflare.com/ai-gateway/configuration/bring-your-own-keys/);
  Vercel takes ordered per-provider credentials via
  `providerOptions.gateway.byok`, falling back to system credentials
  (https://vercel.com/docs/ai-gateway/authentication-and-byok/byok);
  OpenRouter attaches prioritized or fallback BYOK credentials with
  per-key model/user filters (https://openrouter.ai/docs/guides/overview/auth/byok.md);
  Agent Router injects per-request credentials from ext-authz metadata or
  the `x-aigw-api-key` header (https://theagentrouter.ai/docs/capabilities/security/upstream-auth);
  Bifrost accepts direct keys via `x-bf-direct-key` behind an opt-in
  (https://docs.getbifrost.ai/features/keys-management.md); the Portkey OSS
  gateway takes provider keys per request by design
  (https://github.com/Portkey-AI/gateway). llm-proxy: opt-in BYOK
  passthrough exists — `passthrough_auth: bool` per provider
  (`crates/llm-proxy-core/src/provider_config.rs:378-385`), the inbound
  client's bearer/x-api-key token swapped in as the upstream credential at
  `crates/llm-proxy-server/src/routes/core_pipeline.rs:294-304` (the flag
  read at `:727-730`; cite corrected 2026-11-08 per research audit); a
  missing client token fails 401
  "missing client auth token for passthrough provider"
  (`crates/llm-proxy-server/src/routes/error_response.rs:54`); shipped
  2026-06-20 (commit 201e29e) (corrected 2026-10-28 after code audit).
- Data residency/compliance posture: Vercel routes only to providers with
  signed ZDR agreements on request and pins inference regions
  (https://vercel.com/docs/ai-gateway/security-and-compliance); OpenRouter
  offers eu/us regional endpoints and per-key `allowed_data_regions`
  guardrails (https://openrouter.ai/docs/guides/features/in-region-routing.md);
  LiteLLM filters fallbacks by `region_name`
  (https://docs.litellm.ai/docs/proxy/reliability). llm-proxy has no
  data-posture-aware routing.
- Saturation behavior: LiteLLM fails over-capacity requests fast with 429
  instead of queueing (https://docs.litellm.ai/docs/routing); uni-api
  admits or queues requests using cgroup memory awareness
  (`RESOURCE_ADMISSION_MODE`, https://raw.githubusercontent.com/yym68686/uni-api/main/README.md);
  the vLLM Production Stack routes by request priority header
  (https://raw.githubusercontent.com/vllm-project/production-stack/main/src/vllm_router/routers/routing_logic.py);
  Helicone documents no queueing on rate limits
  (https://docs.helicone.ai/features/advanced-usage/custom-rate-limits.md).
  llm-proxy over-limit requests get 429
  (`crates/llm-proxy-server/src/routes/core_pipeline.rs:177-180`, rendered
  by `crates/llm-proxy-server/src/routes/error_response.rs:220-224`; cite
  corrected 2026-11-08 per research audit); no queueing or
  prioritization exists.
- Client SDK coverage: Vercel ships a native AI SDK integration
  (https://vercel.com/docs/ai-gateway/sdks-and-apis); OpenRouter generates
  clients in three languages (https://openrouter.ai/docs/_llms/api-reference.txt);
  the Portkey SDK is a superset of the OpenAI SDK
  (https://portkey.ai/docs/product/ai-gateway/universal-api); Bifrost
  documents zero-code SDK drop-ins for six SDKs
  (https://raw.githubusercontent.com/maximhq/bifrost/main/README.md).
  llm-proxy targets stock OpenAI/Anthropic SDKs via base-URL swap; no
  dedicated SDK.

## Upstream cloud authentication

new: llm-proxy authenticates upstreams with a single static bearer `api_key`
(`SecretString`, `crates/llm-proxy-core/src/provider_config.rs:374-385`); a
wave-2 workspace grep for `sigv4|bedrock|vertex|azure|entra|workload-identity`
matched only docs/research design notes and zero Rust/TOML files, so AWS
SigV4, Azure Entra, and GCP ADC upstreams are unreachable today.

The converged competitor pattern is a per-provider auth object separating credential acquisition (ambient cloud identity chains vs explicit secrets) from request attachment (bearer header vs SigV4 signature), plus an expiry-driven credential cache. agentgateway — Rust, like llm-proxy — builds all of it in-process on official SDK crates (aws-config/aws-sigv4, azure_identity, google-cloud-auth; verified in its Cargo.toml): per-backend mutually-exclusive auth methods (static/aws/azure/gcp/copilot/jwtSign/oauthTokenExchange), SigV4 computed last after every other request mutation, AssumeRole with per-request CEL session tags cached and refreshed before expiry, Azure Entra via the full DefaultAzureCredential chain with per-policy credential-source caches, and GCP ADC access/id tokens incl. impersonated and GDCH service accounts (https://agentgateway.dev/docs/standalone/latest/documentation/configuration/security/backend-authn/, https://raw.githubusercontent.com/agentgateway/agentgateway/main/crates/agentgateway/Cargo.toml). LiteLLM: boto3 credential chains, SigV4 over a filtered header set so forwarded client headers cannot break the canonical request, a DualCache IAM policy (static ~59 min, ambient 600s, STS TTL minus 60s; AssumeRole deliberately uncached), and a Vertex token manager with single-flight proactive refresh (FRESH/STALE/INVALID) plus one re-auth retry on 401 — source-verified at `ref/litellm/litellm/llms/bedrock/base_aws_llm.py:74-148,1400-1439` and `ref/litellm/litellm/llms/vertex_ai/vertex_llm_base.py:480-603` (https://docs.litellm.ai/docs/providers/bedrock, https://docs.litellm.ai/docs/providers/azure, https://docs.litellm.ai/docs/providers/vertex). Kong declares per-type provider auth (aws assume-role incl. batch_role_arn, azure managed identity/client secret, gcp service-account JSON/Workload Identity Federation) on its AI Model Provider entity (https://developer.konghq.com/ai-gateway/entities/ai-model-provider/). Agent Router's BackendSecurityPolicy CRD covers APIKey/AWSCredentials/AzureCredentials/GCPCredentials with controller-side OIDC token exchange writing rotated secrets at expiry minus a 5-minute window, and its extproc SigV4 self-corrects the signing region from the resolved Bedrock host (https://theagentrouter.ai/docs/capabilities/security/upstream-auth, source-verified https://raw.githubusercontent.com/theagentrouter/agent-router/main/internal/backendauth/aws.go). Azure APIM authenticates to Azure AI via managed identities (https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities); AWS AgentCore vaults outbound credentials per target (token-vault ARNs, OAuth flows, SigV4 via the gateway role; https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-outbound-auth.html).
new: Upstream mTLS client certificates ship at agentgateway (file cert/key/root in its backendTLS policy; SPIFFE SVID rotation is Enterprise); LiteLLM documents a client certificate for upstreams, and Kong's Service entity carries `client_certificate` (not tested through its AI plugins) — Portkey and Agent Router document none (corrected 2026-11-08 per research audit) (https://agentgateway.dev/docs/standalone/latest/documentation/configuration/security/backend-tls/).

## Prompt caching and cache locality

new: llm-proxy's protocol carries a cache marker — `CoreContent::Text`
holds `cache: Option<CacheControl>` — but `CacheControl` has only a
`type` field (`Ephemeral`/`Other`) with no TTL, so Anthropic's 1-hour
lifetime is unrepresentable. At the audited commit a `ttl` on a marker also
dropped the whole block from the upstream request while returning HTTP 200
(audit BUG-2; fixed by plan 01). The client adapters do translate system-block
markers (`crates/llm-proxy-protocol/src/client/anthropic.rs:117-119`,
`client/openai_chat.rs:84-95`), but message-level markers are decoded and
then dropped by the upstream Anthropic adapter
(`crates/llm-proxy-provider/src/adapter/anthropic.rs:640,825`), nothing
injects markers, and nothing routes for cache locality
(`crates/llm-proxy-protocol/src/core.rs:142,261-286`; marker translation
corrected 2026-11-08 per research audit).
Cache read/write usage columns already land in the ledger
(`crates/llm-proxy-storage/src/sqlite.rs:116-121`).

Competitors treat prompt caching as a routing and translation concern. OpenRouter translates block markers bidirectionally (Anthropic `cache_control` ↔ OpenAI `prompt_cache_breakpoint`, TTLs dropped in translation), auto-applies a top-level `cache_control` to the last cacheable block, sticky-routes follow-ups to the cache-holding provider for 10 minutes only when cache-read price beats prompt price, and derives sticky keys from `session_id`/`x-session-id`/`prompt_cache_key`/a message hash (https://openrouter.ai/docs/features/prompt-caching). LiteLLM injects markers at configured points (role/index/`tool_config`, mapped to Bedrock `cachePoint` and OpenAI `prompt_cache_breakpoint`), caps injection at Anthropic's 4 blocks (v1.84.9), ships a one-flag Anthropic auto-caching mode (v1.94.0+), translates Anthropic markers to Gemini `cachedContents`, and — closest to a routing feature — pins deployments by SHA-256 of the cacheable prefix plus tools for 300s in its `prompt_caching` pre-call check (https://docs.litellm.ai/docs/tutorials/prompt_caching, https://docs.litellm.ai/docs/completion/prompt_caching, source-verified `ref/litellm/litellm/router_utils/prompt_caching_cache.py:56-197`). Vercel's `caching: 'auto'` classifies providers into explicit-marker vs implicit and places two markers (last message; message before the last user message, system fallback) plus advisory `cache_anchor_items`/`cache_ttl` that degrade with warnings instead of errors; its session affinity pins provider and credential with a pin lifetime matched per provider (5m Anthropic/Bedrock/Google, 10m OpenAI/Azure, 1h DeepSeek, minus 15s, capped 1h), a session-ID hierarchy ending in a conversation fingerprint, minimum-prompt-size gating, and a reported affinity outcome per request (https://vercel.com/docs/ai-gateway/models-and-providers/automatic-caching, https://vercel.com/docs/ai-gateway/models-and-providers/session-affinity). Bifrost pins session IDs (incl. auto-adopted Claude Code/Codex/OpenCode headers) to the last-serving provider+key for 1h (https://docs.getbifrost.ai/providers/session-affinity.md). GPT-Load derives soft affinity from AccessKey/protocol/first-input prefix with LRU capacity 10,000 (https://www.gpt-load.com/docs/internals/scheduling); new-api tracks per-model channel affinity (https://github.com/QuantumNous/new-api). Helicone passes Anthropic `cache_control` through and documents the pricing multipliers (https://docs.helicone.ai/gateway/concepts/prompt-caching).

## Model lifecycle metadata

new: llm-proxy discovery discards lifecycle data: `parse_models` keeps only
id/display_name/context_length (`crates/llm-proxy-provider/src/discovery.rs:272-288`,
verified 2026-10-27), and `/v1/models` hardcodes `created: 0` with null
`created_at` (`crates/llm-proxy-server/src/routes/models.rs:24-36`). Nothing
warns or blocks on a deprecated model.

Providers publish machine-readable lifecycle fields: OpenAI's Model object carries optional `shutdown_date` (notice: ≥6 months GA, 3 months specialized, ~2 weeks preview; https://developers.openai.com/api/docs/api-reference/models, https://developers.openai.com/api/docs/deprecations); Anthropic ships a tri-state `lifecycle` (active/deprecated/retired) plus `deprecated_at`/`retires_at`, keeping retired models listed while inference fails (https://platform.claude.com/docs/en/api/models-list); Bedrock's `modelLifecycle` (ACTIVE/LEGACY + timestamps) drops EOL models from listings, with a 6-month or 45-day Legacy window (https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html, https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html); the Gemini Model resource has no lifecycle fields (https://ai.google.dev/api/models).
Gateways consuming it: OpenRouter's catalog carries `expiration_date` (live check: 14 of 464 entries dated, expired entries delisted) plus `canonical_slug` permanent aliases and a provider-feed deprecation pipeline (https://openrouter.ai/docs/guides/overview/models, https://openrouter.ai/docs/guides/providers/integration); LiteLLM's cost map carries `deprecation_date` on 874 entries, `/model/deprecations` buckets configured models by urgency, and Slack alerts fire on a daily lock — but LiteLLM never blocks or rewrites (https://github.com/BerriAI/litellm/blob/main/litellm/proxy/common_utils/model_deprecation.py, https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json). new: Of the gateways cited in this section, none emits a request-time signal (sunset-style header, response field, or staged rejection) from these dates — LiteLLM stops at dashboards and alerts, and OpenRouter delists. LiteLLM's cited cost map does define `successor_model` entries (feeding its deprecation guidance, not request rewrites); no other gateway cited maps to a successor (corrected 2026-11-08 per research audit).

## Admin and tooling

llm-proxy ships a CLI (serve/stop/status/validate/init/models/autostart,
`apps/llm-proxy/src/cli.rs:23`), a read-only usage admin API
(`crates/llm-proxy-api/src/routes.rs:53-55, 116-117`), and per-provider
egress forward proxying (`crates/llm-proxy-provider/src/transport.rs:293`).
A musl single-binary release workflow exists (`dist-workspace.toml`), but no
release had shipped at the audited commit and the then-active native-tls left
the musl build unproven (corrected 2026-11-08 per research audit; see Extras
above for the full corrected statement).

Competitor admin surfaces: LiteLLM mutates gateway behavior at runtime via a DB-backed control plane and ships a Next.js admin UI (https://docs.litellm.ai/docs/proxy/virtual_keys); gproxy embeds a web console for providers, routes, users, and usage (https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md); Bifrost serves a web UI with live log streaming (https://docs.getbifrost.ai/features/observability/default); agentgateway has a UI with key management and a CEL playground plus the experimental agctl CLI (https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/costs.md); new-api/one-api ship multi-user consoles with payments (https://github.com/QuantumNous/new-api); Vercel exposes CLI + REST + MCP control-plane tools (https://vercel.com/docs/ai-gateway/sdks-and-apis). new: Second wave adds the Higress console (https://higress.cn/en/docs/latest/user/quickstart/), Tyk's AI Studio console with edge microgateways (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/edge-gateways.md), IBM ContextForge's admin UI with LLM provider CRUD and model testing (https://ibm.github.io/mcp-context-forge/latest/api/llm-proxy/), GPT-Load's embedded health/log/route-check/usage UI (https://www.gpt-load.com/docs/monitor), and Respan's dashboard (https://www.respan.ai/docs/).
Prompt management is a control-plane feature llm-proxy lacks entirely: TensorZero binds prompts as config variants (https://github.com/tensorzero/tensorzero/blob/main/docs/experimentation/run-adaptive-ab-tests.mdx), Langfuse serves versioned prompts with stale-while-revalidate client caching (https://langfuse.com/docs/prompt-management/features/caching.md), Portkey versions prompts immutably and A/B-tests them through loadbalance configs (https://portkey.ai/docs/product/prompt-engineering-studio/prompt-versioning). new: Admin audit trails ship at LiteLLM (entity lifecycle audit with before/after values and S3 export — Enterprise edition, corrected 2026-11-08 per research audit, https://docs.litellm.ai/docs/proxy/audit_logs), Kong (RSA-signed admin audit logs with TTL purge, https://developer.konghq.com/gateway/audit-logs), Cloudflare (gateway config-change audit on all plans, https://developers.cloudflare.com/ai-gateway/reference/audit-logs/), Portkey (audit logs API, https://portkey.ai/docs/product/enterprise-offering/audit-logs), and new-api (security audit log with step-up verification, https://github.com/QuantumNous/new-api/releases).
new: Extensibility/hook systems (second wave): LiteLLM exposes in-process Python CustomLogger callbacks plus full request-lifecycle hooks (modify/reject/stream/header) and custom auth/key-policy hooks (https://docs.litellm.ai/docs/proxy/call_hooks, https://docs.litellm.ai/docs/observability/custom_callback); Bifrost loads Go `.so` plugins at runtime with hot reload across the full request lifecycle incl. MCP hooks (https://docs.getbifrost.ai/plugins); Kong ships Lua/PDK plugins, serverless Lua snippets, and external runners (https://developer.konghq.com/custom-plugins/reference, https://developer.konghq.com/plugins/pre-function/); Higress and APISIX run proxy-wasm plugins in Go/Rust/C++ (https://github.com/alibaba/higress/tree/main/plugins, https://apisix.apache.org/docs/apisix/plugins/ai-proxy/); Portkey compiles TypeScript plugin manifests into the gateway (https://github.com/Portkey-AI/gateway/blob/main/plugins/README.md); Tyk unifies gateway/UI/analytics plugins behind one gRPC-isolated SDK (https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/plugins-overview.md). Hook names, registration, and lifecycle are bespoke in each system cited here — no common API emerged among them — and Agent Router ships no custom extension point at all (its ext_proc request was closed stale; https://theagentrouter.ai/).
Config lifecycle: llm-proxy rejects `server.hot_reload=true` at startup (`crates/llm-proxy-core/src/provider_config.rs:1531-1538`; that rejection was lifted by plan 07's snapshot swap, which hot-reloads provider files only). new: agentgateway hot-reloads everything except an explicit startup-only list (adminAddr/storage/database/logging/tracing) via a debounced file watch (re-registering on atomic replace) with validate-then-swap, last-known-good fallback, a `config_synchronized` metric, `--validate-only`, and file/hybrid/readOnly storage modes (https://agentgateway.dev/docs/standalone/latest/documentation/setup/update, https://github.com/agentgateway/agentgateway/blob/main/crates/agentgateway/src/state_manager.rs). new: uni-api (now Rust) compiles validated snapshots atomically and swaps an all-`Arc` immutable snapshot under one write, with revision-CAS admin overlays (https://github.com/yym68686/uni-api/blob/main/src/runtime/state.rs). LiteLLM converges pods by 30s DB polling with YAML-bootstrap-then-DB-overlay precedence and incremental router updates whose errors are caught and swallowed — source-verified at `ref/litellm/litellm/proxy/proxy_server.py:5027-5099` (https://docs.litellm.ai/docs/proxy/configs, https://docs.litellm.ai/docs/proxy/prod); Higress pushes runtime config over xDS in milliseconds (https://github.com/higress-group/higress/blob/main/docs/architecture.md); new: Kong data planes receive configuration dynamically from the Konnect control plane (https://developer.konghq.com/ai-gateway/); new: APISIX applies Admin-API changes to running nodes dynamically from etcd (https://apisix.apache.org/docs/apisix/admin-api/). oc-go-cc reloads via fsnotify+SIGHUP (`ref/oc-go-cc/internal/config/watcher.go:19-101`) and openproxy adds SIGUSR2 zero-downtime upgrade (https://raw.githubusercontent.com/x5iu/openproxy/master/README.md). new: Versioned config with in-gateway history and rollback does ship somewhere: Cloudflare's dynamic routes version configurations with instant rollback (https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/) and Azure APIM exposes API revisions (https://learn.microsoft.com/en-us/azure/api-management/api-management-revisions); among the self-hosted products cited, none does — Kong's decK (https://raw.githubusercontent.com/Kong/deck/main/README.md) is the closest, keeping history in git (corrected 2026-11-08 per research audit). On Rust performance: LiteLLM is migrating translation to a bundled Rust core (beta, off by default; https://docs.litellm.ai/docs/proxy/rust_gateway), Portkey's self-hosted gateway runs a Rust front-end process from v2.30.0 (https://portkey.ai/docs/self-hosting/major-performance-improvements.md), and TensorZero's Rust gateway claimed 0.94 ms p99 overhead at 10k QPS — self-published (https://github.com/tensorzero/tensorzero/blob/main/docs/gateway/benchmarks.mdx) and archived; llm-proxy's own overhead was not measured for this report, so no performance comparison is asserted.

## Transport, state, and deployment topology

llm-proxy's listener is plain HTTP: TLS exists only on the upstream client
— and at the audited commit the active stack was native-tls/OpenSSL, not
rustls: reqwest's default features were on, so `default-tls` and
`rustls-tls` were both compiled and reqwest selects native-tls
(`crates/llm-proxy-provider/Cargo.toml:23`, corrected 2026-11-08 per
research audit; plan 03 later disabled the defaults, making rustls the
single stack in `crates/llm-proxy-transport/Cargo.toml`). A grep
for server-side TLS (rustls/axum-server/certificate) across workspace
Cargo.toml files, run 2026-10-14, returns no listener TLS dependency; HTTPS
fronting is delegated to a reverse proxy. openproxy shows the in-proxy
alternative (rustls TLS 1.3 listener,
https://raw.githubusercontent.com/x5iu/openproxy/master/README.md).

All llm-proxy state is single-node: the usage ledger is a local SQLite file
(`config.toml.example:56-68`), and the rate limiter, dedup window, metrics,
and catalog refresh are in-process — nothing shares state across replicas,
so N instances mean N independent limits and N ledgers. LiteLLM keeps
limits, cooldowns, and spend in Redis/Postgres for multi-proxy deployments
(https://docs.litellm.ai/docs/proxy/load_balancing); gproxy shares
rate-limit counters via a cache backend
(https://gproxy.leenhawk.com/guides/permissions/); Bifrost's OSS governance
is explicitly single-node ("running multiple OSS Bifrost nodes with a
Postgres backend is not supported",
https://docs.getbifrost.ai/features/governance/budget-and-limits) — the same
shape llm-proxy has; new-api's per-node Redis limits count independently
(https://github.com/QuantumNous/new-api). new: Second-wave notes — AWS
AgentCore reaches VPC-internal targets and identity providers through VPC
Lattice private endpoints (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-vpc-egress.html); Cloudflare supports custom gateway domains and a custom-provider registry for arbitrary HTTPS providers (https://developers.cloudflare.com/ai-gateway/configuration/custom-domains/, https://developers.cloudflare.com/ai-gateway/configuration/custom-providers/); Kong 2.2.0 ships FIPS 140-3 and distroless variants (https://developer.konghq.com/ai-gateway/changelog/).

## What llm-proxy lacks

Ranked by impact on a self-hosted proxy serving coding-agent workloads — request-path reliability and spend/rate control first, ecosystem interop second, operator convenience last. Prevalence is the count of surveyed products named per item across both waves; every named product is sourced in the section cited for that item or inline in the entry. Items marked `new:` were first identified in the second wave.

Prevalence counts mean "someone sells this", not self-hostable parity (corrected 2026-11-08 per research audit). Where a named product ships the feature only in a paid edition or a beta, the affected rows restate the count without it. The paid/beta set: Kong's `ai-proxy-advanced`, `ai-rate-limiting-advanced`, `ai-semantic-cache`, `ai-sanitizer` and `ai-mcp-proxy` require an AI license; Portkey budgets, key rotation, secret references and KMS are enterprise or hosted (the OSS cache is exact-match, in memory only); LiteLLM audit logs are Enterprise; Tyk upstream OAuth is Enterprise; Cloudflare Guardrails, DLP, Auto Router and spend limits are Beta.

1. **Request-path retries, fallback chains, and provider cooldowns** — 22 products (LiteLLM, Portkey, Bifrost, Cloudflare, Agent Router, Higress, Kong, OpenRouter, Vercel, TensorZero, one-api, new-api, gproxy, oc-go-cc, agentgateway, AWS AgentCore, Tyk, F5, uni-api, GPT-Load, Respan, llm-api-key-proxy; sources in Routing and resilience); 21 without a paid license (Kong's is `ai-proxy-advanced`). The llm-proxy request path has none; retry/fallback is scoped future work (`docs/research/core-protocol.md:170-186`).
2. **Per-key spend budgets with enforcement and periodic resets** — 15 ship USD budgets (LiteLLM, Bifrost, OpenRouter, Vercel, Cloudflare, LangSmith, gproxy, BricksLLM, agentgateway, Portkey, Tyk, Respan, GPT-Load, uni-api, Traefik Hub; sources in Cost tracking); 12 self-hostable without a paid edition or beta (Portkey budgets are enterprise, Cloudflare spend limits are Beta, Tyk enforcement is Enterprise-only); one-api/new-api/Azure APIM enforce token-quota budgets instead (sources in Cost tracking and Rate limiting). llm-proxy pricing is estimation-only with no enforcement (`crates/llm-proxy-core/src/provider_config.rs:472`).
3. **Identity-scoped rate limits: per-key RPM/TPM and concurrency caps** — 14 products (LiteLLM, Kong, Agent Router, Higress, Bifrost, gproxy, new-api, agentgateway, AWS AgentCore, Traefik Hub, Respan, GPT-Load, uni-api, TensorZero; sources in Rate limiting and quotas); 13 without a paid license (Kong's is `ai-rate-limiting-advanced`). Per-key RPM exists for authenticated callers (token bucket keyed by key id, `crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176`, since 2026-10-13); TPM limits and concurrency caps do not (corrected 2026-10-28 after code audit).
4. **new: Cloud-native upstream authentication (AWS SigV4/STS, Azure Entra, GCP ADC/workload identity)** — 6 products (agentgateway, LiteLLM, Kong, Agent Router, Azure APIM, AWS AgentCore; sources in Upstream cloud authentication). llm-proxy supports only static bearer keys (`crates/llm-proxy-core/src/provider_config.rs:374-385`).
5. **Upstream credential pools with rotation, cooldowns, and quota windows** — 8 products (llm-api-key-proxy `ref/llm-api-key-proxy/src/rotator_library/config/defaults.py:139-158`, gproxy, Bifrost, Higress, new-api — sources in Routing and resilience — plus openproxy weighted upstream key pools https://raw.githubusercontent.com/x5iu/openproxy/master/README.md, GPT-Load credential cooldown/blacklist isolation https://www.gpt-load.com/docs/internals/scheduling, and uni-api key rotation with per-key cooldown knobs https://raw.githubusercontent.com/yym68686/uni-api/main/README.md). llm-proxy takes one static key per provider (`crates/llm-proxy-core/src/provider_config.rs:375`).
6. **OpenAI Responses client endpoint (`/v1/responses`)** — 15 products (LiteLLM, Portkey OSS, OpenRouter, Vercel, LangSmith, Helicone, new-api, Agent Router, agentgateway, AWS AgentCore, Cloudflare, Respan, uni-api, GPT-Load, Martian; sources in Protocol breadth; Tyk ships one only in its unreleased 2.3.0 line). The upstream Responses adapter exists; the client surface does not (`crates/llm-proxy-server/src/routes/mod.rs:69-79`) (corrected 2026-10-28 after code audit).
7. **Guardrails: PII redaction, moderation, prompt-injection detection** — 17 products (LiteLLM, Portkey, Cloudflare, Kong, Bifrost, LangSmith, Helicone, new-api, Higress, Azure, AWS, Tyk, F5, agentgateway, GPT-Load, uni-api, Respan; sources in Guardrails); 15 without a paid license or beta (Kong's is `ai-sanitizer`, Cloudflare Guardrails and DLP are Beta). llm-proxy ships none.
8. **MCP gateway (client, server, or federation)** — 11 products (LiteLLM, Cloudflare, agentgateway, Bifrost, Kong, Azure APIM, AWS AgentCore, IBM ContextForge, Higress, Glama, Tyk; sources in MCP); 10 without a paid license (Kong's is `ai-mcp-proxy`; Tyk's MCP proxying with upstream OAuth 2.1 is Enterprise). llm-proxy has no MCP functionality.
9. **Key lifecycle API (create/list/update/rotate/revoke) without restart** — 12 products (LiteLLM, Bifrost, OpenRouter, Vercel, rust-llm-proxy, one-api, Portkey, agentgateway, GPT-Load, Respan, Tyk, uni-api; sources in Auth and keys); 11 without a paid edition (Portkey key rotation is enterprise). llm-proxy keys are static config (`config.toml.example:42-54`).
10. **Exact-match response caching** — 8 products ship exact-match (LiteLLM, Portkey, Cloudflare, Helicone, OpenRouter, Higress, Respan, Tyk; Portkey's OSS cache is exact-match in memory only); semantic variants at Bifrost, Kong, Azure APIM, LiteLLM's Qdrant backend, and the vLLM stack's alpha (sources in Caching; Kong's semantic cache is `ai-semantic-cache`, paid). llm-proxy has no cache.
11. **OTel trace export under GenAI semantic conventions** — 9 products (Cloudflare, Vercel, OpenRouter, agentgateway, LiteLLM, Bifrost, Tyk, IBM ContextForge, vLLM Production Stack; sources in Observability). llm-proxy tracing exports no OTLP spans.
12. **Prometheus `/metrics` endpoint** — 9 products (LiteLLM, agentgateway, Bifrost, Kong, TensorZero, vLLM Production Stack, Higress, Tyk, IBM ContextForge; sources in Observability). llm-proxy counters are in-process only (`crates/llm-proxy-core/src/metrics.rs`).
13. **Weighted/multi-strategy load balancing per model group** — 12 products (LiteLLM, Portkey, Kong, gproxy, Bifrost, OpenRouter, Cloudflare Auto Router, Higress, uni-api, GPT-Load, AWS AgentCore, Tyk; sources in Routing and resilience, with Cloudflare and Higress sourced inline in the first edition's gap 11 at https://developers.cloudflare.com/ai-gateway/features/auto-router/ and https://github.com/higress-group/higress/blob/main/plugins/wasm-go/extensions/ai-load-balancer/README_EN.md); 10 without a paid license or beta (Kong's is `ai-proxy-advanced`, Cloudflare Auto Router is Beta). An llm-proxy alias resolves to exactly one provider (`crates/llm-proxy-core/src/provider_config.rs:392-394`).
14. **new: Cross-provider prompt-cache plumbing and cache-locality routing** — 7 products (OpenRouter, LiteLLM, Vercel, Helicone, Bifrost, GPT-Load, new-api; sources in Prompt caching and cache locality). llm-proxy's `CacheControl` cannot express TTLs; system-block markers translate but message-level markers are dropped, and nothing injects or pins (`crates/llm-proxy-protocol/src/core.rs:261-286`, corrected 2026-11-08 per research audit).
15. **new: Upstream credential encryption at rest / secret-manager references** — 8 products (gproxy, LiteLLM, Kong, Tyk, IBM ContextForge, new-api, GPT-Load, Portkey; sources in Auth and keys); 7 without a paid edition (Portkey secret references and KMS are enterprise/hosted). llm-proxy stores keys in plaintext TOML.
16. **new: Config hot reload without restart** — 8 products (agentgateway, uni-api, Higress, Kong, APISIX, LiteLLM, oc-go-cc, openproxy; sources in Admin and tooling). llm-proxy rejects `server.hot_reload` (`crates/llm-proxy-core/src/provider_config.rs:1531-1538`).
17. **Latency histograms and TTFT metrics** — 9 products (LiteLLM, agentgateway, Bifrost, oc-go-cc, vLLM Production Stack, Kong, Tyk, Helicone, AWS AgentCore; sources in Observability). llm-proxy computes latency percentiles in-process (`crates/llm-proxy-core/src/metrics.rs:210-239`) but never exports them (corrected 2026-10-28 after code audit).
18. **new: Model lifecycle metadata (deprecation dates, successor aliases) surfaced to clients** — 2 gateways consume it (LiteLLM, OpenRouter) and three providers publish machine-readable fields (OpenAI, Anthropic, Bedrock; sources in Model lifecycle metadata). llm-proxy discovery discards it (`crates/llm-proxy-provider/src/discovery.rs:272-288`).
19. **Web admin UI** — 11 products (LiteLLM, gproxy, Bifrost, agentgateway, new-api, one-api, Higress, Tyk, IBM ContextForge, GPT-Load, Respan; sources in Admin and tooling). llm-proxy admin is read-only JSON (`crates/llm-proxy-api/src/routes.rs:53-55, 116-117`).
20. **new: Stream-cancellation usage settlement** — 2 products document accounting for cancelled requests: LiteLLM embeds usage in every chunk (`always_include_stream_usage`) and cancels upstream on disconnect, though `cancel_on_disconnect` is documented for non-streaming calls (https://docs.litellm.ai/docs/proxy/config_settings); OpenRouter documents per-provider cancel-billing semantics plus a final usage chunk (https://openrouter.ai/docs/api_reference/streaming) (corrected 2026-11-08 per research audit). llm-proxy persists a `ClientCancelled` failure event with no usage payload when a client disconnects — tokens delivered before cancellation are never settled into the ledger (`crates/llm-proxy-server/src/routes/core_pipeline.rs:952-985`, `crates/llm-proxy-storage/src/lib.rs:80`); cancellation propagation itself already works upstream once streaming. SSE keep-alive is already parity (`crates/llm-proxy-server/src/routes/core_pipeline.rs:701,883`).

Already-scoped future work (docs/research/endpoints.md) — embeddings, rerank,
realtime, transcription/TTS, image/video generation, files, OCR, batch,
fine-tuning, search bridges — is not re-listed above. Axes surveyed in
"Client BYOK, data residency, saturation, and SDKs" are comparison context,
not ranked gaps: llm-proxy ships opt-in BYOK passthrough (`passthrough_auth`,
since 2026-06-20) but lacks data-posture routing, queueing/prioritization,
and a dedicated SDK there, each sourced in that section (corrected
2026-10-28 after code audit).

## Near-term candidates

Each candidate names its code touchpoint and a done-condition. Gaps without a candidate below (OTel traces, MCP, key lifecycle, guardrails, load-balancing strategies, the Responses client endpoint, admin UI, cloud-native upstream auth, hot reload, credential encryption at rest, stream-cancellation usage settlement) are deferred: each needs a dependency, config/schema, or control-plane decision the candidates avoid.

- Pre-first-event retry/fallback: the first-byte window cannot host this as
  built. A 429, 5xx or connect error returns at `send_stream`
  (`crates/llm-proxy-server/src/routes/core_pipeline.rs:792-808`), before
  `FirstByteResult` exists, and the window closes on the first encoded
  event — which is a prelude frame (`message_start`, a role-only chat
  chunk, `response.created`, the first Gemini chunk), before any visible
  output — so an `overloaded` arriving after the prelude already has HTTP
  200 committed. Fallback needs the retries to wrap the send itself plus a
  bounded hold before commit (corrected 2026-11-08 per research audit);
  after the first event, keep the in-band error invariant
  (`docs/protocol.md:139-151`). Done when a 429/5xx/connect error from the
  primary advances to the configured fallback before commit and the ledger
  records both attempts — this requires extending the usage-row schema with
  an attempt ordinal and provider per upstream attempt (or an attempts-count
  column), a storage migration scoped into this candidate
  (`crates/llm-proxy-storage/src/sqlite.rs:105-131`). Plan 04 built exactly
  that: classified retries around the send, a per-route hold budget, and
  attempt-level ledger rows.
- Client-key budgets, check-then-accrue: reject when the in-memory counter
  is over budget, accrue actual cost after completion against the existing
  SQLite ledger (`crates/llm-proxy-storage/src/sqlite.rs:105`). Reset
  windows are a rolling duration (optionally calendar-aligned later — the
  two resets competitors converge on, Cost tracking above); accrued spend
  must be rehydrated from the ledger on startup so a restart does not zero
  an in-flight window, and the window start persisted with the key record.
  Deliberately not reservation-based: bounded overshoot from in-flight
  calls is accepted, and the cited LiteLLM behaviour — cost reservation
  silently skipping when cost is unestimatable — comes from
  https://github.com/BerriAI/litellm/issues/35524, which closed 2026-10-08
  with a linked fix, so it is a historical example rather than a current
  one (corrected 2026-11-08 per research audit);
  Bifrost's lazy window reset needs no scheduler
  (https://github.com/maximhq/bifrost/blob/dev/plugins/governance/store.go).
  Done when an over-budget key gets a defined status (402) pre-dispatch,
  settled cost lands in the ledger, and accrued spend survives a restart.
- TPM (per-key RPM already exists, keyed by CallerId at
  `crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176`): reuse the
  offline token counter (`crates/llm-proxy-core/src/token/tiktoken.rs:50,150`,
  cite corrected 2026-11-08 per research audit)
  for reserve-then-reconcile TPM — reservation suits tokens because
  estimates are cheap and refundable (corrected 2026-10-28 after code
  audit). Done when over-limit requests get 429 with retry-after and
  rate-limit headers, reconciled to provider-reported usage after
  completion.
- Prometheus `/metrics` with latency/TTFT histograms: hand-rendered text
  exposition walking the metrics snapshot
  (`crates/llm-proxy-core/src/metrics.rs`), served on the admin listener.
  Done when the endpoint scrapes counters plus a latency histogram and TTFT
  for streams.
- Exact-match response cache keyed on a canonical `CoreRequest`
  serialization, in-memory v1 with a bounded TTL (floor/default/max — the
  60s-floor convention Cloudflare and Portkey use, Caching above), an entry
  cap, and LRU eviction under the cap. Done when a second identical request
  is served from the store with a HIT header and zero recorded usage,
  entries expire at the TTL and evict under the cap. The dedup window
  cannot stay ahead of this as previously claimed: it is off by default,
  keys raw path+body (not `CoreRequest`) from first sighting, and runs
  before any cache lookup, so with it on, a second identical request inside
  the window gets the dedup 409 where this condition requires a HIT —
  the done-condition was self-contradictory (corrected 2026-11-08 per
  research audit). Coalescing (serving the second caller) remains unbuilt.
- Credential pools: optional `[[provider.credentials]]` tables extending the
  single `api_key` field (`crates/llm-proxy-core/src/provider_config.rs:375`)
  with per-credential cooldown state and quota windows persisted via the
  existing storage layer. Done when a credential that 401s is locked out,
  its cooldown survives restart, and the credential id appears in ledger
  rows.
- new: Prompt-cache TTL and marker passthrough: this is more than one core
  field (corrected 2026-11-08 per research audit). A `ttl` in the marker
  makes the containing block unrepresentable on the wire today — the
  Anthropic request decoder's strict `cache_control` deletes the whole
  block with only a WARN, HTTP 200 (`crates/llm-proxy-protocol/src/anthropic.rs:112,174-186`),
  so the fix spans the wire `CacheControl` in both dialects
  (`crates/llm-proxy-protocol/src/anthropic.rs:113`, `openai.rs:11`), a
  `cache` slot on `ToolResult`/`Image`/`ToolUse`/`Thinking` blocks and
  tools (the core `Text.cache` at `crates/llm-proxy-protocol/src/core.rs:261-266`
  is text-only), and the message-level marker drop in the upstream adapter
  (`crates/llm-proxy-provider/src/adapter/anthropic.rs:640,825`). The
  ledger half additionally needs the stream decoder fixed: `message_start`
  usage is ignored and a cumulative `message_delta` carrying only
  `output_tokens` fails to parse, so cache columns stay empty on Anthropic
  streams (`crates/llm-proxy-protocol/src/anthropic.rs:631`,
  `crates/llm-proxy-provider/src/adapter/anthropic.rs:337-361`). Done when a
  request carrying a 1h marker on `/v1/messages` produces the same marker
  upstream — on any block kind — and the ledger records cache-write tokens
  from streams. Plan 01 shipped the wire TTL, block-level markers and the
  stream usage merge; usage columns already existed
  (`crates/llm-proxy-storage/src/sqlite.rs:116-121`).
- new: Model lifecycle capture: extend `StaticModelCatalogEntry`
  (`crates/llm-proxy-core/src/provider_config.rs:735-747`, currently
  `deny_unknown_fields`, so new fields need `#[serde(default)]`) and
  discovery's `parse_models` to capture per-kind lifecycle fields
  (OpenAI `shutdown_date`, Anthropic `lifecycle`/`retires_at`, Bedrock-style
  `modelLifecycle`), surfaced per client shape in `/v1/models`
  (`crates/llm-proxy-server/src/routes/models.rs:24-36`) with retired
  entries kept listed. Done when both client protocols list deprecation
  dates and a warn-level log fires on deprecated-model use; blocking stays
  opt-in and out of scope.
