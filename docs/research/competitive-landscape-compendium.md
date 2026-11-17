# Competitive landscape compendium — llm-proxy research record (October–November 2026)

Consolidated record of the three-wave competitive research summarized in
[`competitive-gap-analysis.md`](competitive-gap-analysis.md), reorganized
for deciding what to build next. Competitor claims and their URLs originate
in the audited analysis (competitive-gap-analysis.md) and the research
runs' recorded findings; task counts and severity tiers come from the runs'
final results; llm-proxy-side claims were re-verified against the working
tree (code audit, 2026-10-28).

The llm-proxy baseline below describes the tree at the
[research audit](audit-2026-11-05/REPORT.md) (commit `79cf406`,
2026-11-05). Claims the audit found false are corrected in place and
marked "(corrected 2026-11-08 per research audit)". Plans 01–10 have since
rebuilt most of the described subsystems; current behaviour is in
[docs/goal.md](../goal.md) and [docs/completed/](../completed/).

## Scope and method

- Two research waves in October 2026 (plus the November waves recorded
  below): 57 research tasks (product scouts, theme
  studies, doc sweeps, and 10 deep dives — 6 from wave 1, 4 from wave 2)
  plus 10 frontier analyses — 67 inventory items total, all accounted for
  in the coverage appendix. Wave-1 grep checks ran 2026-10-14; wave-2 code
  claims (hot_reload rejection, `CacheControl` shape, discovery field
  capture, `ModelCard`, SSE keep-alive, cancellation handling, body limit)
  were re-verified on 2026-10-27.
- Third wave, November 2026: niche-repository hunts over the
  Rust/Go/TypeScript/Python long tail produced a deduplicated find list of
  121 candidate repositories, of which 23 were researched in full — 90
  inventory items across the three waves. Wave-3 records live modularly in
  [`niche-repos.md`](niche-repos.md) with the machine-readable registry
  [`repos-registry.json`](repos-registry.json) covering all three waves;
  the most significant wave-3 finds are summarized in the market map below
  and all 23 are accounted for in the coverage appendix.
- Competitor claims from official docs, release notes, and changelogs
  current to October 2026, plus vendored source pins read directly:
  `ref/litellm` at b84f7f82, `ref/oc-go-cc` at 99d756c, `ref/llm-api-key-proxy`
  at af40b91. `ref/REFS` also pins `ref/llm-proxy` at fb588f6 (2024-05-24),
  uncited by the analysis but researched directly for this compendium on
  2026-10-28 (Reference repositories); a 2026-10-28 follow-up likewise
  researched kgateway from its official docs. No competitor code was built
  or run.
- llm-proxy claims verified against this working tree, cited `file:line`;
  wave-2 additions are tagged `new:` where the distinction matters.

## Market map

One factual line per researched product with its source URL; status notes
are from the analysis's survey header. kgateway and the `ref/llm-proxy` pin
were researched directly on 2026-10-28 (Scope and method).

### Open-source gateways

| Product | Line | Source |
|---------|------|--------|
| LiteLLM | Python gateway, DB-backed control plane, virtual keys, six routing strategies, one of several breadth leaders | https://docs.litellm.ai/release_notes |
| Portkey OSS gateway | Node gateway, dormant since 2026-05-25 after the Palo Alto Networks acquisition; enterprise line (rebranded Prisma AIRS AI Gateway) continues | https://github.com/Portkey-AI/gateway/releases, https://portkey.ai/docs/product/ai-gateway/fallbacks |
| Kong AI Gateway | `ai-proxy-advanced`: seven load-balancing algorithms, circuit breaker; FIPS 140-3 and distroless variants since 2.2.0 | https://developer.konghq.com/plugins/ai-proxy-advanced/, https://developer.konghq.com/ai-gateway/changelog/ |
| agentgateway | Rust/tokio, split out of kgateway; virtual keys, CEL rate limits, cloud upstream auth, OTel GenAI metrics, MCP federation; 23 native providers incl. Bedrock, Vertex, GitHub Copilot; A2A proxying and REST-as-tools; MCP OAuth2 resource-server auth with CEL-based MCP authorization; promptCaching policy auto-inserting provider prompt-cache markers | https://agentgateway.dev/docs/standalone/latest/documentation/llm/about.md, https://agentgateway.dev/docs/standalone/latest/documentation/about/introduction.md, https://agentgateway.dev/docs/standalone/latest/documentation/mcp/about.md, https://agentgateway.dev/docs/standalone/latest/documentation/configuration/traffic-management/llm.md |
| kgateway | Solo.io's Envoy-based Gateway API control plane (ex-Gloo, CNCF sandbox, very active; v2.5.0-rc.1 on 2026-10-27). Its Envoy-based AI Gateway was deprecated in v2.1.x and removed in v2.2.0 — the AI line became agentgateway (Rust, Linux Foundation 2025, Agentic AI Foundation 2026). What remains: Gateway API edge features — JWT/OAuth2/OIDC + ext-proc auth, local + global rate limiting, retries with per-try timeouts and outlier detection, priority failover, transformations, mirroring. Absent now: any in-tree AI gateway, LLM cost tracking, prompt guardrails, token rate limiting, MCP | https://kgateway.dev/, https://kgateway.dev/blog/kgateway-v2.2-release-blog, https://kgateway.dev/docs/envoy/2.1.x/ai/about.md, https://kgateway.dev/docs/envoy/2.2.x/reference/release-notes.md, https://kgateway.dev/docs/envoy/latest/about/overview.md, https://kgateway.dev/docs/envoy/latest/llms.txt |
| Agent Router (ex-Envoy AI Gateway) | Priority-ordered backend fallback, Redis-backed usage rate limits, BackendSecurityPolicy CRD; no guardrails as of October 2026 | https://theagentrouter.ai/ |
| Higress | wasm-plugin gateway; ai-proxy auto-detects OpenAI vs Claude clients across ~30 backends, xDS config push, hosted MCP servers | https://higress.ai/en/docs/latest/plugins/ai/api-provider/ai-proxy |
| Bifrost (Maxim) | Go gateway; `sk-bf-*` virtual keys, FailureClass retry taxonomy, layered USD budgets, MCP federation, Go `.so` plugins | https://docs.getbifrost.ai/features/governance/budget-and-limits, https://github.com/maximhq/bifrost/blob/dev/core/schemas/failureclass.go |
| TensorZero | Rust gateway, archived 2026-06-12; three-level retry/fallback, scoped rate-limit rules, self-published 0.94 ms p99 benchmark at 10k QPS | https://github.com/tensorzero/tensorzero |
| one-api | Token expiry/quota/IP/model controls; dormant (v0.6.10, Feb 2025) | https://github.com/songquanpeng/one-api/blob/main/README.md, https://github.com/songquanpeng/one-api/releases |
| new-api | Four client protocols, subscription quotas plus prepaid wallet, console with payments | https://github.com/QuantumNous/new-api |
| uni-api | (now Rust) `AUTO_RETRY` to 100 attempts, six per-key scheduling algorithms, stacked rate-limit windows, cgroup-memory-aware admission | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md |
| GPT-Load | Tier-priority weighted scheduling (group × credential weight) under a global retry budget; AccessKeys with IP-CIDR restrictions | https://www.gpt-load.com/docs/internals/scheduling |
| gproxy | Weighted fallback tiers over member providers, allow/deny key scoping, AES-256-GCM credential encryption, embedded web console | https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md |
| openproxy | rustls TLS 1.3 listener, SIGUSR2 zero-downtime upgrade, weighted upstream key pools | https://raw.githubusercontent.com/x5iu/openproxy/master/README.md |
| LLMG | Auto-registers 70+ providers from environment variables | https://modpotatodotdev.github.io/LLMG/providers/all/ |
| rust-llm-proxy | Issue/rotate/revoke key endpoints | https://raw.githubusercontent.com/habibi-dev/rust-llm-proxy/master/README.md |
| llm-api-key-proxy | Key-pool rotator; same-key server-error retries under a global deadline | `ref/llm-api-key-proxy/src/rotator_library/README.md`, https://github.com/Mirrowel/LLM-API-Key-Proxy |
| IBM ContextForge (mcp-context-forge) | Federates MCP/A2A/REST/gRPC; SHA-256 tokens with jti revocation, AES-GCM tool credentials, Argon2id, optional Vault; OTLP to eight backends | https://github.com/IBM/mcp-context-forge, https://ibm.github.io/mcp-context-forge/latest/architecture/security-features/ |
| Tyk AI Gateway / AI Studio | Failover waterfalls, Tengo filters, `$SECRET/` references, edge microgateways | https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/llm-management.md |
| vLLM Production Stack | Request-priority routing, alpha feature-gated semantic cache, in-chart Prometheus/Grafana/OTel | https://github.com/vllm-project/production-stack |
| Apache APISIX | `new:` ai-proxy plugin; Admin-API changes applied to running nodes from etcd | https://apisix.apache.org/docs/apisix/plugins/ai-proxy/, https://apisix.apache.org/docs/apisix/admin-api/ |
| BricksLLM | Pioneered per-key USD caps in 2023; last commit/release 2025-01-05 | https://raw.githubusercontent.com/bricks-cloud/BricksLLM/main/README.md, https://api.github.com/repos/bricks-cloud/BricksLLM |

### Cloud and enterprise gateways

| Product | Line | Source |
|---------|------|--------|
| Cloudflare AI Gateway | Caching, guardrails (S1–S14 + P1), spend limits, OTLP under GenAI semconv, BYOK, encrypted-body Logpush, custom domains and providers — Guardrails, DLP, Auto Router and spend limits are Beta (corrected 2026-11-08 per research audit) | https://developers.cloudflare.com/ai-gateway/features/caching/ |
| Vercel AI Gateway | Three client formats plus nine modalities incl. Decisions; BYOK, OIDC project tokens, session affinity, automatic prompt caching, budgets, trace drains | https://vercel.com/docs/ai-gateway/sdks-and-apis |
| Azure API Management (GenAI Gateway) | `llm-token-limit` policy, managed identities, RediSearch semantic caching, REST-as-MCP export | https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities |
| AWS Bedrock AgentCore Gateway | Four inference wire formats, fallback rule blocks, dimension-keyed rate limits, Cedar-policy MCP authorization, per-target credential vaulting, VPC Lattice egress | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html |
| F5 AI Gateway | In transition (docs redirect, Python SDK archived 2026-03-18, successor Early Access); processor-chain guardrails, per-service retries, one named fallback service | https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/configuration/configure-services.html, https://www.f5.com/company/news/press-releases/f5-unleashes-next-generation-agentic-ready-ai-gateway-to-optimize-the-economics-and-governance-of-enterprise-ai-costs |
| Traefik Hub | `new:` `ai-rate-limit` token bucket vs `ai-quota` sliding window over tokens or USD, Redis-backed fail-open | https://doc.traefik.io/traefik-hub/ai-gateway/middlewares/token-rate-limit |

### Observability platforms

| Product | Line | Source |
|---------|------|--------|
| Helicone | TTFT and p50/p95/p99 dashboards, edge cache, OSS Rust proxy (RFC 7234 over Moka/Redis), LLM Security screening, full-body capture with omit headers, `Helicone-Property-*` metadata | https://docs.helicone.ai/features/advanced-usage/caching.md, https://github.com/helicone/ai-gateway |
| Langfuse | Rust ai-gateway relaying OpenAI + Anthropic byte-preserving; versioned prompts with stale-while-revalidate caching | https://github.com/langfuse/langfuse/tree/main/ai-gateway, https://langfuse.com/docs/prompt-management/features/caching.md |
| LangSmith (LLM Gateway, beta) | `/v1/chat/completions`, `/v1/messages`, `/v1/responses` behind one key; spend policies; in-path PII/secret redaction with restoration incl. streams | https://docs.langchain.com/langsmith/llm-gateway, https://docs.langchain.com/langsmith/llm-gateway-data-policy |

### Routers and SaaS

| Product | Line | Source |
|---------|------|--------|
| OpenRouter | Chat/Responses/Anthropic plus embeddings, rerank, image/video, TTS/STT, batch, files, containers; BYOK, fallbacks, prompt-cache marker translation, OTLP broadcast, spend-capped MCP server | https://openrouter.ai/docs/_llms/api-reference.txt |
| Martian | Hosted gateway live 2026-10-27; 312 models, 49 providers at `/v1/chat/completions`, `/v1/responses`, `/v1/messages` | https://api.withmartian.com/v1/models, https://docs.withmartian.com/api-reference/endpoints |
| Not Diamond | Routing-decision API, not a proxy | https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post.md |
| Respan (formerly Keywords AI) | OpenAI chat, Responses, and Anthropic surfaces over 1000+ models; scoped keys, budgets, retries, caching, guardrails, remote MCP server, dashboard | https://www.respan.ai/docs/documentation/features/gateway/gateway-quickstart |
| OpenPipe | Sunset; migrated to Weights & Biases/CoreWeave, docs.openpipe.ai dead | https://openpipe.ai/security |
| Glama | Hosted MCP gateway with per-tool ACLs; `x-glama-metadata` header | https://glama.ai/mcp/gateway |

Negative finding, kept out of the product rows (housekeeping correction,
2026-11-08): an "Upstash AI Gateway" named in one brief was verified not to
exist — 11 live checks incl. sitemap, llms.txt, GitHub, npm, and DNS
(https://upstash.com/docs/llms.txt).

### Chinese-ecosystem aggregators

| Product | Line | Source |
|---------|------|--------|
| one-api | Token quotas/expiry/IP/model controls, automatic retry; global IP limit excludes the relay data path; dormant | https://github.com/songquanpeng/one-api/blob/main/README.en.md, https://github.com/songquanpeng/one-api/releases |
| new-api | Cross-channel retry (off by default), per-IP/group/model limits, sensitive-word screening (off by default since rc.45), subscription + prepaid wallet, `/v1/tools/search` and `/v1/tools/fetch` | https://github.com/QuantumNous/new-api, https://github.com/QuantumNous/new-api/releases |
| Higress | Redis token rate limits per consumer/header/IP; ai-cache over Redis with DashVector semantic mode | https://higress.ai/en/docs/latest/user/plugins/ai/api-consumer/ai-token-ratelimit |
| uni-api | Per-key dollar-credit quotas, stacked windows, tiered channel/key cooldowns | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md |
| GPT-Load | 35 channels (OpenAI/Mistral/Anthropic/Gemini incl. `:embedContent`), credential cooldown/blacklist isolation | https://raw.githubusercontent.com/tbphp/gpt-load/main/README.md |

### Niche repositories (wave 3, November 2026)

One-line rows for the most significant third-wave finds; full modular
records (capabilities, evidence, absences) in
[`niche-repos.md`](niche-repos.md), machine-readable rows in
[`repos-registry.json`](repos-registry.json).

| Product | Line | Source |
|---------|------|--------|
| Switchyard (NVIDIA-NeMo) | Rust routing toolkit (libsy + Python/C/Node bindings): ordered per-candidate fallback with per-backend cooldowns and Retry-After, 429-aware fallback policy, triple-protocol demo server, judge/stage-router suite, OTel GenAI spans, theoretical cache-hit-rate tracking | https://github.com/NVIDIA-NeMo/Switchyard |
| SMG (Shepherd Model Gateway) | Rust engine-agnostic gateway: KV-cache-aware routing (four index modes), queue-depth/KV-usage overload vetoes, WASM plugin middleware, priority admission with preemption-before-first-byte, OIDC/RBAC, 150+ Prometheus metrics | https://lightseek.org/smg/v1.14.0/concepts/routing/cache-aware |
| CLIProxyAPI (router-for-me) | Go proxy (57.5k stars) turning CLI subscriptions into APIs: OAuth login flows, credential pools with three rotation strategies, retry/cooldown state machines, session-affinity sticky routing, v8 management API (ToS-gray; pattern reference) | https://github.com/router-for-me/CLIProxyAPI |
| sub2api (Wei-Shaw) | Go multi-tenant subscription-resale gateway: five-state account isolation, 401/403/429-keyed failover, cache-affinity sticky hashing, token-level cost ledger with multipliers, hash-verified remote price catalog (ToS-flagged; design reference) | https://github.com/Wei-Shaw/sub2api |
| AISIX (api7) | Rust OSS data-plane AI gateway from the APISIX creators: six routing strategies, Redis-cluster RPM/TPM/concurrency limits, exact+semantic cache, guardrail stack, MCP+A2A gateways, per-attempt usage events, SIGHUP/etcd hot config | https://github.com/api7/aisix |
| Antirion (eliasbiondo) | 18-day-old Rust gateway with embedded console: route trees with hedge races and circuit breakers, reservation-based limits, exact+semantic cache, SAML/SCIM/WebAuthn identity, OTLP+alerting, signed supply-chain releases | https://github.com/eliasbiondo/antirion |
| Plano (katanemo) | Envoy + proxy-wasm Rust data plane for agentic apps: NL-intent agent orchestration, three-mode LLM routing with fallback, MCP/HTTP guardrail filter chains, header-keyed token budgets, Agentic Signals quality detectors | https://docs.planoai.dev/guides/llm_router.html |
| ThinkWatch | Rust dual AI+MCP gateway (BSL 1.1): OIDC SSO with five-role RBAC, per-user MCP OAuth so tool calls run as the user, reversible PII redaction in streams, sliding-window limits + budgets, ClickHouse audit with SIEM export | https://github.com/ThinkWatchProject/ThinkWatch |
| Otari (mozilla-ai) | Python gateway you own and run: pre-dispatch budget holds reconciled after settlement, workspace-scoped virtual keys with model allowlists, budget-conditional routing, managed MCP tool loop, layered guardrails, idempotency replay + first-chunk hold | https://github.com/mozilla-ai/otari |
| Claude Code Router (musistudio) | TypeScript control plane for coding agents (38k stars): ordered conditional routing rules with request rewriting, sandboxed route scripts, credential pools, per-key local quotas, ToolHub MCP aggregation with lazy tool resolution | https://github.com/musistudio/claude-code-router |
| mistral.rs | Rust inference engine serving OpenAI /v1 and Anthropic /v1/messages side by side with full Anthropic SSE fidelity, agent permission gates, MCP client+server modes, Prometheus TTFT/ITL histograms | https://github.com/EricLBuehler/mistral.rs |
| claude-code-proxy (raine) | Rust Anthropic-compatible proxy over subscription upstreams (Codex/Kimi/Grok/Cursor): request-path retries, full OAuth credential lifecycle with keychain storage, model-ID routing with service tiers, monitor TUI | https://github.com/raine/claude-code-proxy |
| yolorouter | Go single-binary gateway for coding CLIs: key pool with Retry-After benching and auto-recovery, generation-guarded circuit breakers, seven-phase timeouts, request-path tool-output compression, community price catalog | https://github.com/yolorouter/yolorouter |
| xLB (EeroEternal) | Rust thin-L7/L4 streaming load balancer ("the HAProxy role"): phase-shaped timeouts (connect/first_byte/idle/total), wall-clock retry budget, non-idempotent retry split, TTFT-delta metric, client-abort propagation (dormant) | https://github.com/EeroEternal/xlb |
| GPTCache (zilliztech) | Seminal Python semantic-cache library (maintenance-mode): five-stage pipeline taxonomy, 11 similarity evaluators, LLM-verification gate, eviction subsystem, SSE hit-replay | https://github.com/zilliztech/GPTCache |

### Reference repositories (vendored pins and README references)

- `ref/litellm` at b84f7f82 — routing constants, Bedrock/Vertex auth
  internals, prompt-caching cache, proxy config convergence.
- `ref/oc-go-cc` at 99d756c — circuit breakers, p95/p99 health metrics, 3s
  SSE keep-alive, fsnotify+SIGHUP reload.
- `ref/llm-api-key-proxy` at af40b91 — same-key retries and rotator
  defaults (`src/rotator_library/config/defaults.py:139-158`).
- `ref/llm-proxy` at fb588f6 — the vendored pin is `proxyllm` v0.2.3, an
  in-process Python routing library (no HTTP server, no endpoints;
  `ref/llm-proxy/README.md`, `ref/llm-proxy/proxyllm/proxyllm.py`): each
  call routes to one of 6 provider SDK adapters by three strategies — cost
  (pre-flight worst-case estimate), static Elo, or zero-shot category
  classification via bart-large-mnli
  (`ref/llm-proxy/proxyllm/utils/cost.py`,
  `ref/llm-proxy/proxyllm/utils/categorization.py`) — with sequential
  fallback down a min-heap, no retries or health tracking, a hard-coded
  May-2024 price+Elo catalog
  (`ref/llm-proxy/proxyllm/config/internal_config.py`), the user's own keys
  via .env, and console logging only. Development stopped 2024-05-24 with
  guardrails, multimodal, and context injection on its unfinished roadmap.
  As contrast it validates gateway-style scope: it is a library, not a
  proxy.
- gproxy, LLMG, openproxy, rust-llm-proxy — README reference projects,
  sourced above under open-source gateways.

## Baseline: llm-proxy today

Citations verified against the working tree (wave-2 items re-verified
2026-10-27).

| Area | State | Citations |
|------|-------|-----------|
| Client surface | `POST /providers/{p}/v1/chat/completions`, `POST /providers/{p}/v1/messages`, `POST /providers/{p}/v1/messages/count_tokens`, `GET /providers/{p}/v1/models` — all four API routes carry the mandatory `/providers/{provider}` prefix — plus `/health`, `/ready`, `/version`; 32 MiB body limit; configurable `TimeoutLayer` returning 408 on provider routes, health/version exempt | `crates/llm-proxy-server/src/routes/mod.rs:50-52,69-79`, `docs/protocol.md:25-30`, `crates/llm-proxy-server/src/routes/mod.rs:37,62`, `:44,54-67` |
| Translation | Hub-and-spoke `CoreRequest`/`CoreResponse`/`CoreEvent`, 10 content kinds; client adapters OpenAI Chat + Anthropic Messages; upstream adapters OpenAI Chat, Anthropic Messages, OpenAI Responses, Gemini GenerateContent | `docs/protocol.md:7-9`, `docs/protocol.md:80-91`, `crates/llm-proxy-provider/src/adapter/` |
| Streaming | Normalized event state machine; in-band errors after visible output; 3s SSE keep-alive comments | `docs/protocol.md:120-133,139-151`, `crates/llm-proxy-server/src/routes/core_pipeline.rs:701,883` |
| Auth | `mode=none` (loopback-only, enforced) or `client-keys` with argon2id PHC hashes; admin API behind constant-time bearer token; upstream `api_key` is a `SecretString` (redacted in `Debug`, unserialized, zeroized on drop) but plaintext in the operator's TOML on disk | `config.toml.example:42-54`, `crates/llm-proxy-api/src/auth.rs`, `crates/llm-proxy-core/src/provider_config.rs:357-375` |
| Routing | URL-path provider plus static model-alias map; no request-path retry/fallback, only bounded transient retry in catalog discovery | `crates/llm-proxy-core/src/provider_config.rs:392-394`, `crates/llm-proxy-provider/src/discovery.rs:160-170` |
| Catalogs | Static/live/hybrid with single-flight refresh and atomic disk cache; discovery keeps only id/display_name/context_length (no price or lifecycle fields); `ModelCard` hardcodes `created: 0`, null `created_at` | `crates/llm-proxy-server/src/catalog_service.rs`, `crates/llm-proxy-provider/src/discovery.rs:272-288`, `crates/llm-proxy-server/src/routes/models.rs:24-36` |
| Persistence | SQLite usage ledger, background writer, retention controls; only body-derived column is `body_hash` — full token/cost columns, no payloads | `crates/llm-proxy-storage/src/sqlite.rs:105-131`, `config.toml.example:56-68` |
| Admin API | Read-only `/usage/summary|requests|keys` with window/group_by filters and cursor paging | `crates/llm-proxy-api/src/routes.rs:53-55,116-117` |
| Metrics | Lock-free atomic counters incl. per-provider-model counts; latency ring buffer up to 1000 samples with in-process percentile helpers (never exposed); request ids; graceful shutdown | `crates/llm-proxy-core/src/metrics.rs:34-35,210-239`, `crates/llm-proxy-server/src/shutdown.rs` |
| Middleware | Per-identity token-bucket RPM, burst-tolerant with continuous refill, keyed by key id when authenticated else client IP (since 2026-10-13), loopback exempt; opt-in dedup window (off by default, 0 ms shipped) 409-rejecting an identical path+body seen within a fixed window of the first sighting — not in-flight tracking, caller-blind, bodies over 64 KiB exempt (rejection, not coalescing; corrected 2026-11-08 per research audit); opt-in forwarded-header trust, CORS, panic catching | `crates/llm-proxy-server/src/middleware.rs:117-129`, `crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176`, `config.toml.example:14-21`, `crates/llm-proxy-server/src/routes/core_pipeline.rs:182-185` |
| Extras | Offline tiktoken counting, chars/4 fallback; per-provider fixed-point USD per-token prices, estimation only; per-provider egress forward proxy (socks5h/socks5/http/https); env-var interpolation; a musl single-binary release workflow exists but no release had shipped and unvendored `openssl-sys` behind the then-active native-tls left the musl build unproven (corrected 2026-11-08 per research audit; plan 03 later made rustls the only stack). Config changes require restart — `server.hot_reload=true` rejected at load (superseded: plan 07 hot-reloads provider and pool files on SIGHUP; `server.hot_reload = true` is accepted) | `crates/llm-proxy-core/src/token/tiktoken.rs` (HeuristicTokenizer; heuristic documented at `crates/llm-proxy-core/src/token/counter.rs:5`), `crates/llm-proxy-core/src/provider_config.rs:472,500`, `crates/llm-proxy-provider/src/transport.rs:124-125,293`, `crates/llm-proxy-core/src/provider_config.rs:1531-1538` |
| Cancellation | `ClientCancelled` failure event persists with no usage payload — tokens delivered before cancellation never settle | `crates/llm-proxy-server/src/routes/core_pipeline.rs:952-985`, `crates/llm-proxy-storage/src/lib.rs:80` |

## Capability areas

### Auth and keys

llm-proxy: loopback-only no-auth or static argon2id client keys
(`config.toml.example:42-54`); no budgets, scoping, expiry, or
issuance/rotation; config edit + restart to change. Upstream: one static
bearer per provider, plaintext on disk.

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | Virtual keys (`POST /key/generate`: max_budget, budget_duration, tpm/rpm, max_parallel_requests, model allowlists/aliases) under org→team→user→key budgets | https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/access_control |
| Bifrost | `sk-bf-*` keys: expiry, instant `is_active` kill switch, deny-by-default provider/model scoping | https://docs.getbifrost.ai/features/governance/virtual-keys |
| agentgateway `new:` | Client virtual keys (`vk-...`, strict/optional/permissive), hash-only storage, per-key model allowlists (403 `model_not_allowed`), v1.7 UI auto-rotated values | https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/virtual-keys.md |
| Portkey `new:` | Dual-secret rotation (`POST /v2/api-keys/:id/rotate`, old secret valid `key_transition_period_ms`, min/default 30 min, id unchanged) plus scheduled policies | https://portkey.ai/docs/product/enterprise-offering/org-management/api-key-rotation |
| OpenRouter | Credit-capped inference keys via management-key API | https://openrouter.ai/docs/guides/overview/auth/management-api-keys.md |
| Vercel | 12-hour OIDC project tokens; unauthenticated compromised-secret revocation | https://vercel.com/docs/ai-gateway/authentication-and-byok/oidc |
| gproxy | Ordered allow/deny rules over provider, model glob, operation | https://gproxy.leenhawk.com/guides/permissions/ |
| one-api | Token expiry, quota, IP ranges, model allowlists | https://github.com/songquanpeng/one-api/blob/main/README.md |
| rust-llm-proxy | Issue/rotate/revoke endpoints | https://raw.githubusercontent.com/habibi-dev/rust-llm-proxy/master/README.md |
| Respan `new:` | Read/Write/Gateway/Admin scopes, Production/Test environments, spend caps, expiry, admin-gated `/api/temporary-keys/` | https://www.respan.ai/docs/documentation/admin/respan-api-keys |
| GPT-Load `new:` | AccessKeys scoped to groups/models/protocols, IP-CIDR, rotation preserving id/permissions/quota | https://www.gpt-load.com/docs/access-keys |
| Tyk `new:` | App keys activated by an administrator (approve/reject issuance) | https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/apps.md |
| uni-api `new:` | `/v1/generate-api-key` behind admin-role keys | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md |
| IBM ContextForge `new:` | SHA-256 token hashes with jti revocation/replay detection, AES-GCM tool credentials, Argon2id, optional Vault | https://ibm.github.io/mcp-context-forge/latest/architecture/security-features/ |
| Weaker | Portkey OSS auth middleware is admin-UI-token only (its middleware directory contains just adminAuth, cache, hooks, log, requestValidator); Higress key-auth stores plaintext in YAML; Cloudflare gateway tokens account-scoped, not gateway-restrictable | https://github.com/Portkey-AI/gateway/tree/main/src/middlewares, https://higress.ai/en/docs/latest/user/plugins/authentication/key-auth, https://developers.cloudflare.com/ai-gateway/configuration/bring-your-own-keys/ |
| Encryption at rest | gproxy AES-256-GCM under `GPROXY_MASTER_KEY`; LiteLLM NaCl SecretBox from `LITELLM_SALT_KEY` (spend logs and cached prompts stay plaintext) plus enterprise secret managers; Kong Vault entity over Konnect/AWS/GCP/Azure/HashiCorp; `new:` Portkey secret references + AWS KMS envelope encryption; new-api and GPT-Load `CRYPTO_SECRET`/`ENCRYPTION_KEY`; `new:` Tyk `$SECRET/Name` under `TYK_AI_SECRET_KEY` with env-reference allowlists | https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md, https://docs.litellm.ai/docs/proxy/security_encryption_faq.md, https://docs.litellm.ai/docs/secret_managers/overview.md, https://developer.konghq.com/gateway/entities/vault/, https://portkey.ai/docs/product/enterprise-offering/secret-references.md, https://portkey.ai/docs/product/enterprise-offering/kms.md, https://github.com/QuantumNous/new-api, https://raw.githubusercontent.com/tbphp/gpt-load/main/README.md, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/secrets.md |

Verdict: gap #9 (medium, key lifecycle), gap #15 (medium, encryption at
rest).

### Routing and resilience

llm-proxy: URL-path provider + static alias
(`crates/llm-proxy-core/src/provider_config.rs:392-394`); no request-path
retry/fallback/cooldown
(`crates/llm-proxy-provider/src/discovery.rs:160-170` is discovery-only);
retry/fallback is scoped future work (`docs/research/core-protocol.md:170-186`).

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | Six strategies (weighted simple-shuffle, least-busy, usage/latency/cost-based); per-exception RetryPolicy; TTL cooldown cache on 429 or >50% failure; typed fallback chains (general / context-window / content-policy; max_fallbacks default 5, cooldown default 5s, `ref/litellm/litellm/constants.py:13,37-43`); `new:` mid-stream fallback via assistant prefill; `region_name` fallback filtering | https://docs.litellm.ai/docs/routing, https://docs.litellm.ai/docs/proxy/reliability, https://docs.litellm.ai/docs/proxy/config_settings |
| Portkey | Recursive fallback/loadbalance/conditional configs; retries cap cumulative wait at 60s, honor Retry-After; `new:` per-strategy `cb_config` circuit breaker (cooldown ≥30s) | https://portkey.ai/docs/product/ai-gateway/fallbacks, https://portkey.ai/docs/product/ai-gateway/automatic-retries, https://portkey.ai/docs/product/ai-gateway/circuit-breaker |
| Bifrost | FailureClass taxonomy (credential/quota/model_access/rate_limit/caller_fault/transient/timeout) driving retry-same-key vs rotate-key vs stop; per-fallback retry budgets; TTFT timeouts fall back, post-output errors do not; `new:` CEL routing rules, session-affinity pin, complexity router | https://docs.getbifrost.ai/features/fallbacks, https://github.com/maximhq/bifrost/blob/dev/core/schemas/failureclass.go, https://docs.getbifrost.ai/providers/routing-rules.md, https://docs.getbifrost.ai/providers/session-affinity.md, https://docs.getbifrost.ai/features/complexity-router.md |
| Cloudflare | Up to 5 attempts with per-request backoff, ordered fallbacks, versioned dynamic routing; Auto Router weighted balancing | https://developers.cloudflare.com/ai-gateway/configuration/request-handling/, https://developers.cloudflare.com/ai-gateway/features/dynamic-routing/, https://developers.cloudflare.com/ai-gateway/features/auto-router/ |
| Agent Router | Priority-ordered backends; retries walk the priority chain | https://theagentrouter.ai/docs/capabilities/traffic/provider-fallback |
| Kong | Seven load-balancing algorithms, circuit breaker | https://developer.konghq.com/plugins/ai-proxy-advanced/ |
| Higress | API-key pool failover with health checks and cooldown restore; ai-load-balancer extension | https://github.com/higress-group/higress/blob/main/plugins/wasm-go/extensions/ai-proxy/README_EN.md, https://github.com/higress-group/higress/blob/main/plugins/wasm-go/extensions/ai-load-balancer/README_EN.md |
| oc-go-cc | Three-state per-model circuit breakers, fallback chains (`ref/oc-go-cc/internal/router/fallback.go:15-115`) | `ref/oc-go-cc` |
| gproxy | Weights and fallback tiers over member providers; retries within one provider's credentials before moving on | https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md, https://gproxy.leenhawk.com/introduction/architecture/ |
| OpenRouter | Model-array fallbacks, price-weighted provider balancing | https://openrouter.ai/docs/guides/routing/model-fallbacks.md |
| Vercel | Health-aware ordering, model fallback chains, per-provider BYOK timeouts | https://vercel.com/docs/ai-gateway/models-and-providers/model-fallbacks, https://vercel.com/docs/ai-gateway/models-and-providers/provider-timeouts |
| TensorZero | Three-level retry/fallback hierarchy | https://github.com/tensorzero/tensorzero/blob/main/docs/gateway/guides/retries-fallbacks.mdx |
| one-api / new-api | Automatic retry; new-api configurable cross-channel retry, off by default | https://github.com/songquanpeng/one-api/blob/main/README.md, https://github.com/QuantumNous/new-api/blob/main/model/request_policy.go |
| llm-api-key-proxy | Same-key server-error retries under a global deadline | `ref/llm-api-key-proxy/src/rotator_library/README.md` |
| AWS AgentCore `new:` | Fallback blocks (1–2 ordered targets; status-code classes, CONNECTION_ERROR/FIRST_BYTE_TIMEOUT/UNROUTABLE; default-on circuit breaker whose open state skips and does not bill; no fallback after stream start); weighted 2-way traffic splits | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-rules-fallback.html, https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-rules.html |
| Tyk `new:` | Failover waterfalls (max 10 rungs; 408/429/5xx/timeout/connect triggers, never 401/403; `X-Tyk-Served-LLM`/`X-Tyk-Failover` headers; pre-first-token only); Enterprise glob Model Router weighted/round-robin | https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/llm-management.md, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/model-router.md |
| F5 `new:` | Per-service retries (429/500/502/503/504, Retry-After priority, 1.5x growth capped 60s) plus one `fallbackServiceName` | https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/configuration/configure-services.html |
| agentgateway `new:` | Per-route retries stay on the same backend; cross-model failover via virtual-model eviction groups | https://agentgateway.dev/docs/standalone/latest/documentation/configuration/resiliency/retries.md, https://agentgateway.dev/docs/standalone/latest/documentation/llm/virtual-models.md |
| uni-api `new:` | `AUTO_RETRY` to 100, tiered channel/key cooldowns, six scheduling algorithms incl. success-rate-aware `smart_round_robin` | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md |
| GPT-Load `new:` | Tier-priority weighted scheduling, down-tier-only global retry budget, no switch after committed stream output | https://www.gpt-load.com/docs/internals/scheduling |
| Respan `new:` | Retries on by default (≤3, ordered `fallback_models`); validation/context/credential failures never retried | https://www.respan.ai/docs/documentation/features/gateway/retries |

Verdict: gap #1 (high, 22 products), gap #13 (medium, 12 products).

### Rate limiting and quotas

llm-proxy: per-identity token-bucket RPM, burst-tolerant with continuous
refill (`crates/llm-proxy-server/src/middleware.rs:117-129`), keyed by key
id when authenticated, client IP otherwise, loopback exempt
(`crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176`, since
2026-10-13; `config.toml.example:14-21`); TPM (token-based) limits and
concurrency caps absent; 429 with no queueing or prioritization.

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | max_parallel_requests/rpm/tpm per key/user/team in one batched pre-call check; reserve-then-reconcile TPM (estimate prompt + max_tokens, settle actuals, refund on failure) | https://docs.litellm.ai/docs/proxy/users |
| Kong | Token-kind or cost ceilings at consumer/IP/header/path/model/provider level | https://developer.konghq.com/plugins/ai-rate-limiting-advanced/ |
| Agent Router | Token budgets (typed kinds or CEL) into Redis windows; never interrupts an admitted stream | https://theagentrouter.ai/docs/capabilities/traffic/usage-based-ratelimiting |
| Higress | Tokens per consumer/header/IP over Redis | https://higress.ai/en/docs/latest/user/plugins/ai/api-consumer/ai-token-ratelimit |
| Bifrost | Request/token limits at provider-config/key/team/customer level, typed 401/402/403/429 errors | https://docs.getbifrost.ai/features/governance/budget-and-limits |
| gproxy | Per user/key incl. concurrency | https://gproxy.leenhawk.com/guides/permissions/ |
| new-api | Per-IP, per-group, model-level limits | https://docs.newapi.ai/en/docs/guide/feature-guide/admin/system-setting-advanced |
| agentgateway `new:` | CEL-keyed limits (`jwt.sub`, `apiKey.user`, `llm.requestModel`); in-memory LRU buckets (≤65,536/rule) or Envoy Rate Limit gRPC; two-phase token enforcement when backend sets `tokenize: true` (upfront estimate blocks, otherwise post-hoc) | https://agentgateway.dev/docs/standalone/latest/documentation/configuration/resiliency/rate-limits.md |
| AWS AgentCore `new:` | Dimension-keyed limits (targetName, toolName, qualifiedModelId, JWT claims, IAM principal; immutable keys, most-specific-wins) over req/s, tokens/min (pre-estimate + actuals), concurrency; fail-open, "not a security boundary" | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-rate-limits.html |
| Traefik Hub `new:` | Token-bucket `ai-rate-limit` (burst-tolerant) vs sliding-window `ai-quota` (no bursts); input/output/total-token or USD; `sourceCriterion`; body-length/4 pre-rejection; Redis fail-open | https://doc.traefik.io/traefik-hub/ai-gateway/middlewares/token-rate-limit |
| Azure APIM `new:` | One `llm-token-limit` policy: tokens/min (429) + long-period quotas over hourly-to-yearly fixed UTC windows (403), any counter-key expression, optional `estimate-prompt-tokens` | https://learn.microsoft.com/en-us/azure/api-management/llm-token-limit-policy |
| GPT-Load `new:` | Per-AccessKey RPM plus concurrency cap | https://www.gpt-load.com/docs/access-keys |
| uni-api `new:` | Stacked windows (`15/min,10/day`) per key, per-model substring, per-upstream-key knobs | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md |
| Respan `new:` | Layered per-key cost/request/token limits plus customer (402) and org-plan (429) tiers | https://www.respan.ai/docs/documentation/features/gateway/limits |
| TensorZero `new:` | Scoped rules over inferences/tokens/cost with priority resolution, Valkey/Postgres state (pre-archival) | https://raw.githubusercontent.com/tensorzero/tensorzero/main/docs/operations/enforce-custom-rate-limits.mdx |
| Weaker | Cloudflare per gateway only; one-api global IP limit excludes the relay data path; Vercel does not rate-limit paid tier; Tyk no core RPM/TPM limiter on the LLM path (plugins only) | https://developers.cloudflare.com/ai-gateway/features/rate-limiting/, https://github.com/songquanpeng/one-api/blob/main/README.en.md, https://vercel.com/docs/ai-gateway/rate-limits, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/plugins-overview.md |

Verdict: gap #3 (medium, 14 products).

### Cost tracking

llm-proxy: estimated USD from operator-configured per-token prices into the
SQLite ledger (`crates/llm-proxy-storage/src/sqlite.rs:105-131`). Pricing
code exists in `crates/llm-proxy-core/src/provider_config.rs:472,500` and
`crates/llm-proxy-core/src/provider_registry.rs:387-398` (`pricing_for`),
re-exported at `crates/llm-proxy-core/src/lib.rs:36`, with commented
examples in `apps/llm-proxy/src/defaults.rs:52-60` — no shipped or synced
price table; estimation only, never enforced.

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | Per-request cost against a 2,752-model price map, spend logs, Prometheus spend export, pre-call reservation | https://docs.litellm.ai/docs/proxy/cost_tracking |
| Portkey | Attribution from Portkey-AI/models (2,300+ models, pushed 2026-10) | https://github.com/Portkey-AI/models |
| Helicone | 300+ model pricing repository | https://docs.helicone.ai/references/how-we-calculate-cost.md |
| agentgateway | Per-token-kind pricing (input/output/cacheRead/cacheWrite/reasoning/audio) with context tiers, models.dev import | https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/costs.md |
| Kong | Model cost management incl. TTL-tiered cache-write rates | https://developer.konghq.com/ai-gateway/model-cost-management/ |
| Enforcement wired to prices | OpenRouter, Vercel, Cloudflare, LangSmith budgets/policies; BricksLLM pioneered per-key USD caps (2023); gproxy USD budgets on users/keys/orgs/teams settled after the call | https://openrouter.ai/docs/guides/best-practices/spend-controls.md, https://vercel.com/docs/ai-gateway/observability-and-spend/budgets, https://developers.cloudflare.com/ai-gateway/features/spend-limits/, https://docs.langchain.com/langsmith/llm-gateway-spend-policies, https://raw.githubusercontent.com/bricks-cloud/BricksLLM/main/README.md, https://gproxy.leenhawk.com/guides/permissions/ |
| Bifrost `new:` | Arrays `{max_limit USD, reset_duration 1m–1Y}`, UTC calendar alignment, fiscal quarters, overrides, cumulative customer→team→key→provider-config enforcement | https://docs.getbifrost.ai/features/governance/budget-and-limits.md |
| LiteLLM budgets `new:` | Stackable multi-window `budget_limits` ($10/24h + $100/30d), IANA-tz calendar resets, `budget_reset_time`, soft budgets, temporary raises, tag budgets via header or W3C baggage | https://docs.litellm.ai/docs/proxy/users, https://docs.litellm.ai/docs/proxy/budget_reset_and_tz, https://docs.litellm.ai/docs/proxy/tag_budgets |
| agentgateway budgets `new:` | Per-key USD-or-token rolling windows, Block (429 `budget_exceeded`) / Audit modes, multiple windows per key, needs SQLite/Postgres config DB | https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/budget-limits/per-key.md |
| Portkey enterprise `new:` | AND-condition usage policies with `group_by` counters (per-user/model/key/MCP-server), weekly/monthly resets, breach 412 | https://portkey.ai/docs/product/enterprise-offering/budget-policies |
| Cloudflare `new:` | Spend rules by provider/model/metadata, split-by-value buckets (`cf.user_id`), budget-exceeded fallback to cheaper model (Beta); LiteLLM likewise routes within provider budgets — neither is the only budget→fallback action (corrected 2026-11-08 per research audit) | https://developers.cloudflare.com/ai-gateway/features/spend-limits/, https://docs.litellm.ai/docs/routing |
| OpenRouter `new:` | Layered guardrail/workspace/key budgets with in-flight holds pre-rejecting oversized requests (`weight_exceeds_budget` vs retryable `in_flight_budget_exhausted`, both 402) | https://openrouter.ai/docs/api-reference/limits |
| new-api / Tyk / Respan / GPT-Load / uni-api / Traefik Hub `new:` | new-api subscription + prepaid wallet; Tyk monthly budgets at LLM/App/team level with ~30s block-list pulse (enforcement Enterprise-only, Community never blocks); Respan alert/Block thresholds, customer 402; GPT-Load quotas as never-resetting totals or periodic windows from one minute to one year — soft enforcement, charged after completion; uni-api dollar-credit quotas (30-day default, 0 blocks); Traefik Hub `costLimit` | https://github.com/QuantumNous/new-api, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/budgeting.md, https://www.respan.ai/docs/documentation/features/gateway/limits, https://www.gpt-load.com/docs/access-keys, https://raw.githubusercontent.com/yym68686/uni-api/main/README.md, https://doc.traefik.io/traefik-hub/ai-gateway/middlewares/token-rate-limit |
| Status codes `new:` | Not converged — Bifrost 402, Portkey 412, Respan 402/429, LiteLLM 400/401/429 | https://github.com/maximhq/bifrost/blob/dev/plugins/governance/main.go, https://portkey.ai/docs/api-reference/inference-api/error-codes, https://docs.litellm.ai/docs/proxy/users |
| Attribution `new:` | agentgateway invoice-grade: Bedrock per-request STS session names into CloudTrail/CUR, Vertex billing labels into Cloud Billing export | https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/attribution.md |

Verdict: gap #2 (high, 15 products ship USD budgets; one-api/new-api/Azure
APIM enforce token quotas instead).

### Caching

llm-proxy: no response cache; the opt-in dedup window — off by default, a
fixed window from first sighting, caller-blind, bodies over 64 KiB exempt —
409-rejects repeats (`crates/llm-proxy-server/src/routes/core_pipeline.rs:182-185`)
— pressure control, neither cache nor coalescing (corrected 2026-11-08 per
research audit).

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | 8+ backends, SHA-256 over recognized params, streams reassembled then replayed synthetically | https://docs.litellm.ai/docs/proxy/caching |
| Portkey | Simple + semantic, TTL 60s–90d, namespacing (hosted product; the OSS gateway cache is exact-match, in memory only — corrected 2026-11-08 per research audit) | https://portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic |
| Cloudflare | SHA-256 over provider+endpoint+model+auth+body, TTL 60s–1mo | https://developers.cloudflare.com/ai-gateway/features/caching/ |
| Helicone | Edge cache; OSS Rust proxy hashing seed+path+body, RFC 7234 freshness over Moka/Redis, buffered bodies, 2xx-only | https://docs.helicone.ai/features/advanced-usage/caching.md, https://github.com/helicone/ai-gateway |
| OpenRouter | Opt-in exact-match with free hits | https://openrouter.ai/docs/guides/features/response-caching.md |
| Higress | Redis ai-cache keyed by GJSON paths (exact match, streaming replay); `new:` DashVector semantic mode | https://higress.ai/en/docs/latest/user/plugins/ai/api-provider/ai-cache, https://higress.ai/en/docs/ai/scene-guide/semantic-cache |
| Respan `new:` | Exact "messages only" matches, per-customer/model scoping, hit/savings dashboard | https://www.respan.ai/docs/documentation/features/gateway/caching |
| Tyk `new:` | Community exact-match plugin; enterprise semantic (cosine ≥0.95 over Redis/memory vector stores) | https://tyk.io/docs/ai-management/ai-studio/semantic-caching |
| vLLM stack `new:` | Alpha, feature-gated semantic cache (sentence-transformer, threshold 0.95) | https://raw.githubusercontent.com/vllm-project/production-stack/main/src/vllm_router/experimental/README.md |
| Semantic elsewhere | Bifrost (hash + cosine over vector stores), Kong ai-semantic-cache, Azure APIM via RediSearch | https://docs.getbifrost.ai/features/semantic-caching, https://developer.konghq.com/plugins/ai-semantic-cache/, https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities |
| Single-flight | Rare: Cloudflare documents the duplicate-origin race and OpenRouter non-coalescing; Portkey OSS, Helicone OSS, and LiteLLM cache code examined at source has no in-flight dedup; unverified for the rest | (synthesis of the sourced rows above) |
| Verified absences `new:` | agentgateway (advertised "semantic caching" absent from its Kubernetes API reference); AWS AgentCore, Tyk core, GPT-Load, uni-api document none | https://agentgateway.dev/ |

Verdict: gap #10 (medium, 8 exact-match products).

### Guardrails

llm-proxy: no content inspection, redaction, or moderation.

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | ~50 guardrail services across seven hook phases incl. pre-call, post-call, MCP; `new:` reversible Presidio PII masking with `output_parse_pii` re-insertion | https://docs.litellm.ai/docs/proxy/guardrails/quick_start, https://docs.litellm.ai/release_notes, https://docs.litellm.ai/docs/tutorials/presidio_pii_masking.md |
| Portkey | 20+ deterministic checks plus LLM checks (PII, moderation, language, gibberish), deny=446 / flag=246 driving fallback orchestration; `new:` pre-LLM PII redaction irreversible by design | https://portkey.ai/docs/product/guardrails, https://portkey.ai/docs/product/guardrails/pii-redaction |
| Cloudflare | S1–S14 hazards (Llama Guard 3 8B) + P1 injection (Prompt Guard 2), per-category flag/block; DLP shared with Cloudflare One — Guardrails and DLP are Beta (corrected 2026-11-08 per research audit) | https://developers.cloudflare.com/ai-gateway/features/guardrails/, https://developers.cloudflare.com/ai-gateway/features/dlp/ |
| Kong | ai-sanitizer redacts 20+ PII entity types bidirectionally | https://developer.konghq.com/plugins/ai-sanitizer/ |
| Bifrost (enterprise) | CEL rules, Gitleaks secret detection, LLM-as-judge | https://docs.getbifrost.ai/enterprise/guardrails |
| LangSmith | In-path PII/secret redaction with placeholder restoration, incl. streams | https://docs.langchain.com/langsmith/llm-gateway-data-policy |
| Helicone | Prompt-injection screening via Meta Prompt Guard (advanced: Llama Guard), OpenAI models only | https://docs.helicone.ai/features/advanced-usage/llm-security.md |
| new-api | Sensitive-word screening on prompts and/or responses; `new:` off by default since rc.45 | https://github.com/QuantumNous/new-api/blob/main/model/request_policy.go, https://github.com/QuantumNous/new-api/releases |
| Higress | Moderation delegated to Aliyun's paid service | https://higress.ai/en/docs/latest/user/plugins/ai/api-provider/ai-security-guard |
| Cloud vendors | Azure AI Content Safety; AWS Bedrock Guardrails | https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities, https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html |
| Tyk `new:` | Tengo filters with redaction/quarantine over normalized messages, calling Lakera/Azure/Presidio/Bedrock/HTTP classifiers; streaming re-evaluation; tool-result screening from 2.3 | https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/filters.md, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/guardrails.md |
| F5 `new:` | Processor chain (injection ML detector, PII matcher incl. `mcp`, language-id, system-prompt injection) with modify/reject/annotate | https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/processors/processors-overview.html |
| agentgateway `new:` | Ordered prompt guards: regex + built-in PII (mask/reject/audit), OpenAI moderation, vendor guards, webhooks, failClosed default; masking reject-only on streams | https://agentgateway.dev/docs/standalone/latest/documentation/llm/prompt-guards/overview |
| GPT-Load `new:` | Experimental JEV guardrails via external decision service; global RE2 redaction with reversible per-AccessKey encryption restored in responses and streams | https://www.gpt-load.com/docs/advanced/guardrails, https://www.gpt-load.com/docs/advanced/redaction |
| uni-api / Respan `new:` | Optional omni-moderation of user messages; free 8-category blocking via span-01-security, org-wide | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md, https://www.respan.ai/docs/documentation/features/gateway/guardrails |
| AWS AgentCore `new:` | Deny-by-default Cedar policies on MCP `tools/call` with Bedrock Guardrails confidence-score conditions | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/use-gateway-with-policy.html |
| Documented absence | Agent Router: no guardrails as of October 2026 — transport/auth only; per its docs, not a code audit | https://theagentrouter.ai/docs/capabilities/security/ |

Verdict: gap #7 (medium, 17 products).

### Observability

llm-proxy: in-process atomic counters, request ids, plain/JSON logs,
graceful shutdown; up to 1000 latency samples collected
(`crates/llm-proxy-core/src/metrics.rs:34-35`) with `Snapshot.latencies`
plus `calculate_percentile`/`calculate_p95`/`calculate_p99` computed
in-process (`crates/llm-proxy-core/src/metrics.rs:210-239`) — but no
exposure surface: no `/metrics`, percentiles never exported, no TTFT
measurement, no trace export.

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | Request/latency/TTFT/queue/overhead histograms, identity-labeled token/spend counters, budget-remaining gauges; OTel export with `traceparent` propagation | https://docs.litellm.ai/docs/proxy/prometheus, https://docs.litellm.ai/docs/proxy/logging |
| agentgateway | OTel GenAI semconv names directly (`gen_ai.server.request.duration`, `time_to_first_token`, `time_per_output_token`, token-usage histograms) on :15020; OTLP tracing with per-request labels and CEL metric removal | https://agentgateway.dev/docs/standalone/latest/documentation/observability/metrics/reference/, https://agentgateway.dev/docs/standalone/latest/documentation/observability/metrics/overview |
| Bifrost | Per-provider/model/virtual-key cost, token, latency-histogram, TTFT, key-health metrics with SQLite/Postgres logs and web UI; `llm.call` OTel span per request, per-attempt spans over OTLP HTTP/gRPC | https://docs.getbifrost.ai/features/observability/prometheus, https://docs.getbifrost.ai/features/observability/default, https://docs.getbifrost.ai/features/otel.md |
| Kong | `ai_metrics` request/cost/token counters (token_type labels incl. cache-write TTL splits), LLM latency histograms, TTFT under gen_ai conventions | https://developer.konghq.com/ai-gateway/monitor-ai-llm-metrics/ |
| Cloudflare | OTLP under GenAI semconv, per-user spend anomaly detection | https://developers.cloudflare.com/ai-gateway/observability/otel-integration/ |
| Vercel | Per-attempt routing logs, OTel trace drains | https://vercel.com/docs/ai-gateway/observability-and-spend/logs, https://vercel.com/docs/ai-gateway/observability-and-spend/trace-drains |
| OpenRouter | OTLP trace broadcast to ~20 destinations | https://openrouter.ai/docs/guides/features/broadcast/webhook.md |
| TensorZero | Three Prometheus metrics + OTLP traces (pre-archival) | https://raw.githubusercontent.com/tensorzero/tensorzero/main/docs/operations/export-prometheus-metrics.mdx |
| vLLM stack | Prometheus/Grafana/OTel in-chart | https://github.com/vllm-project/production-stack |
| Higress | ai-statistics per-route/cluster/model/consumer token and latency counters | https://higress.ai/en/docs/latest/user/plugins/ai/api-o11y/ai-statistics |
| oc-go-cc | p95/p99 at `/health` (`ref/oc-go-cc/internal/metrics/metrics.go:132-154`) | `ref/oc-go-cc` |
| Tyk `new:` | Secure-by-default Prometheus `/metrics` under GenAI semconv (`gen_ai_server_request_duration_seconds`, time-to-first-token, time-per-output-token) plus `aistudio_*` governance counters, Server-Timing header splitting gateway vs upstream latency, opt-in OTel with always-on W3C traceparent | https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/observability.md |
| IBM ContextForge `new:` | OTLP to eight backends + JWT-protected Prometheus | https://ibm.github.io/mcp-context-forge/latest/manage/observability/ |
| AWS AgentCore `new:` | CloudWatch metrics (latency to first token p50/p90/p99, throttles, per-target time), vended logs incl. bodies with OTel-compatible spans | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-gateway-metrics.html |
| Helicone `new:` | TTFT and p50/p95/p99 in dashboards rather than exported histograms | https://helicone-helicone-7.mintlify.app/observability/metrics |
| Semconv | OTel client-inference span shape standardized | https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/main/docs/gen-ai/client-inference.md |

Streaming SLO mechanics `new:` LiteLLM SSE `: ping` keep-alives
(`keepalive_seconds` 1–300), `cancel_on_disconnect` (logged 499 — documented
for non-streaming calls, so it is not a streaming-cancel mechanism; corrected
2026-11-08 per research audit), and
`always_include_stream_usage` for billing when the final chunk never arrives
(https://docs.litellm.ai/docs/proxy/config_settings); OpenRouter documents
per-provider cancel-billing (abort propagates at ~24 providers;
Bedrock/Groq/Google/Mistral keep processing and bill) and a final
usage-chunk accounting frame with in-band mid-stream errors
(https://openrouter.ai/docs/api_reference/streaming); oc-go-cc emits
`:keepalive` every 3s (`ref/oc-go-cc/internal/handlers/messages.go:239-266`);
uni-api ships an SSE heartbeat
(https://raw.githubusercontent.com/yym68686/uni-api/main/README.md).
Keep-alive is parity for llm-proxy
(`crates/llm-proxy-server/src/routes/core_pipeline.rs:701,883`); no cited
product documents SLO machinery (burn-rate alerts, error budgets) —
percentiles are left to PromQL and TTFT measurement points differ.

Payload and attribution: llm-proxy keeps digests only
(`crates/llm-proxy-storage/src/sqlite.rs:130-131`) and records only the key
id; Helicone records full bodies with per-request omit headers
(https://docs.helicone.ai/features/advanced-usage/omit-logs.md), LiteLLM
stores messages with redaction controls
(https://docs.litellm.ai/docs/proxy/logging), Cloudflare logs payloads with
per-request opt-out and `new:` Logpush exports bodies customer-key-encrypted
(a fresh AES-256-GCM key per log wrapped with the customer's RSA-4096 public
key — https://developers.cloudflare.com/ai-gateway/observability/logging/,
https://developers.cloudflare.com/ai-gateway/observability/logging/logpush/),
and OpenRouter retains content only behind explicit opt-in
(https://openrouter.ai/docs/guides/features/logs.md). Metadata/tag headers —
Helicone `Helicone-Property-*`, LiteLLM `metadata.tags`/`x-litellm-tags` +
User-Agent auto-tagging, Portkey `x-portkey-metadata`, Cloudflare
`cf-aig-metadata` (excluded from the cache key), Vercel
`providerOptions.gateway.user`/`tags`, Glama `x-glama-metadata` — are the
dashboard grouping key everywhere.

Verdict: gap #11 (medium, OTel, 9 products), gap #12 (medium, Prometheus,
9), gap #17 (low, percentile/TTFT exposure, 9), additional gap #20
(stream-cancellation settlement, 2).

### MCP

llm-proxy: no MCP client, server, or gateway functionality.

Protocol: Streamable HTTP + OAuth 2.1 resource-server auth standardized in
the 2025-11-25 spec (https://modelcontextprotocol.io/specification/2025-11-25/basic/transports,
https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization);
the 2026-07-28 revision — final, not a release candidate (corrected
2026-11-08 per research audit) — removes sessions and adds gateway-friendly
per-operation routing headers (https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate).

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | One endpoint for all MCP tool traffic, six-level permissions, per-tool allowlists | https://docs.litellm.ai/docs/mcp, https://docs.litellm.ai/docs/mcp_control |
| Cloudflare | MCP portals aggregating up to 80 HTTP servers behind Access with managed upstream OAuth | https://developers.cloudflare.com/cloudflare-one/access-controls/ai-controls/mcp-portals/ |
| agentgateway | Federation with CEL authorization per method; MCP OAuth2 resource-server auth; `new:` v1.7 ExtMCP guardrail chain and inline CEL processor | https://agentgateway.dev/docs/standalone/latest/documentation/mcp/mcp-authz.md, https://agentgateway.dev/docs/standalone/latest/documentation/mcp/about.md, https://agentgateway.dev/docs/standalone/latest/documentation/mcp/guardrails/about.md |
| Bifrost | Federated tools at /mcp with virtual-key-scoped filtering; `new:` execution explicit by default (separate `/v1/mcp/tool/execute`), opt-in auto-approval, sandboxed code mode across 3+ servers | https://docs.getbifrost.ai/mcp/overview, https://docs.getbifrost.ai/mcp/overview.md |
| Kong | REST↔MCP bridging in four modes; `new:` 2.1.0 stateless 2026-07-28 revision with `ttlMs`/`cacheScope` hints and W3C trace-context | https://developer.konghq.com/plugins/ai-mcp-proxy/, https://developer.konghq.com/ai-gateway/changelog/ |
| Azure APIM | Exports managed REST APIs as MCP servers | https://learn.microsoft.com/en-us/azure/api-management/mcp-server-overview |
| AWS AgentCore | Converts APIs/Lambda into MCP tools with per-tool credential injection; `new:` stateful sessions with elicitation/sampling, semantic tool search | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html, https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-features.html |
| IBM ContextForge | Federates MCP/A2A/REST/gRPC | https://github.com/IBM/mcp-context-forge |
| Glama | Hosted MCP gateway with per-tool ACLs | https://glama.ai/mcp/gateway |
| Higress | Hosts MCP servers in-gateway | https://higress.cn/en/docs/ai/mcp-quick-start/ |
| Tyk `new:` | Gateway v5.13+ proxies remote MCP servers (OAuth 2.1, per-key tool allowlists, per-tool rate limits and circuit breakers); AI Studio exposes tools at `/tools/{slug}/mcp` | https://tyk.io/docs/ai-management/mcp-gateway/overview.md, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/tools.md |
| Respan `new:` | Remote MCP server exposing logs, traces, prompts, monitors | https://www.respan.ai/docs/documentation/mcp |
| OpenRouter `new:` | MCP server at mcp.openrouter.ai issuing spend-capped 7-day keys via OAuth | https://openrouter.ai/docs/guides/overview/mcp-server.md |

Verdict: gap #8 (medium, 11 products).

### Protocol breadth

llm-proxy: two client formats (OpenAI Chat, Anthropic Messages), four
upstream formats, count_tokens. Embeddings/audio/images/batch/realtime are
scoped future work (docs/research/endpoints.md).

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | Chat, Responses, batches, files, assistants, images, audio, rerank, RAG, realtime, OCR, containers, fine-tuning, A2A, search | https://docs.litellm.ai/release_notes |
| Portkey OSS | Handlers: completions, embeddings, batches, files, fine-tunes, images, speech, transcriptions, translations, realtime WebSocket, Responses, Anthropic + count_tokens | https://github.com/Portkey-AI/gateway/tree/main/src/handlers |
| OpenRouter | Chat, Responses, Anthropic Messages plus embeddings, rerank, image/video, TTS/STT, batch, files, containers | https://openrouter.ai/docs/_llms/api-reference.txt |
| Vercel | Three client formats plus embeddings, rerank, Decisions; `new:` nine modalities incl. video and a typed Decision API | https://vercel.com/docs/ai-gateway/sdks-and-apis, https://vercel.com/docs/ai-gateway/modalities |
| Cloudflare `new:` | REST API `/ai/v1/chat/completions`, `/ai/v1/responses`, `/ai/v1/messages` | https://developers.cloudflare.com/ai-gateway/usage/rest-api/ |
| LangSmith `new:` | Beta gateway serves `/v1/chat/completions`, `/v1/messages`, `/v1/responses` behind one key | https://docs.langchain.com/langsmith/llm-gateway |
| Helicone `new:` | OpenAI Responses at `/v1/responses` for OpenAI and Anthropic models | https://docs.helicone.ai/gateway/concepts/responses-api.md |
| agentgateway `new:` | Client routes OpenAI Chat, Anthropic Messages, OpenAI Responses, Gemini generateContent, translated both directions; 23 native providers incl. Bedrock, Vertex, GitHub Copilot; A2A proxying and REST-as-tools | https://agentgateway.dev/docs/standalone/latest/documentation/llm/about.md, https://agentgateway.dev/docs/standalone/latest/documentation/about/introduction.md |
| AWS AgentCore `new:` | chat/completions, messages, responses, Bedrock converse/InvokeModel with cross-dialect translation | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-targets-inference.html |
| new-api | Four client protocols with cross-conversion and realtime WebSocket; `new:` `/v1/tools/search` and `/v1/tools/fetch` billed per call | https://github.com/QuantumNous/new-api, https://github.com/QuantumNous/new-api/releases |
| Agent Router | Four client formats under configurable prefixes, ~20 providers | https://theagentrouter.ai/docs/capabilities/llm-integrations/supported-endpoints |
| Higress | Auto-detects OpenAI vs Claude, ~30 backends | https://higress.ai/en/docs/latest/plugins/ai/api-provider/ai-proxy |
| LLMG | 70+ providers auto-registered from env | https://modpotatodotdev.github.io/LLMG/providers/all/ |
| Kong | Native provider passthrough across 17 providers | https://developer.konghq.com/plugins/ai-proxy/ |
| Tyk `new:` | Four surfaces (vendor-native, OpenAI translation, Anthropic→Bedrock bridge, legacy); governed batches/Files/Responses only in unreleased 2.3.0 | https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/proxy.md, https://github.com/TykTechnologies/ai-studio/blob/main/CHANGELOG-v2.3.0.md |
| uni-api `new:` | Chat, responses (+compact), messages, images, embeddings, audio, moderations, search across 16+ engines | https://raw.githubusercontent.com/yym68686/uni-api/main/README.md |
| GPT-Load `new:` | OpenAI Chat/Responses/Images/Embeddings/Rerank, Mistral-native OCR/audio, Anthropic, Gemini incl. `:embedContent`, 35 channels | https://raw.githubusercontent.com/tbphp/gpt-load/main/README.md |
| Respan `new:` | OpenAI chat, Responses, Anthropic-native over 1000+ models | https://www.respan.ai/docs/documentation/features/gateway/gateway-quickstart |
| F5 `new:` | OpenAI-only client schema (chat subset + /v1/models), Anthropic upstream executor only | https://web.archive.org/web/2025/https://aigateway.clouddocs.f5.com/configuration/configure-services.html |
| Martian / Not Diamond `new:` | Martian: 312 models / 49 providers at three endpoints; Not Diamond a routing-decision API, not a proxy | https://api.withmartian.com/v1/models, https://docs.withmartian.com/api-reference/endpoints, https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post.md |
| Deliberately narrow | Langfuse Rust ai-gateway relays OpenAI + Anthropic byte-preserving | https://github.com/langfuse/langfuse/tree/main/ai-gateway |

Translation fidelity is only partially comparable from public docs.
llm-proxy normalizes 10 content kinds (`docs/protocol.md:80-91`) and records
normalized usage incl. cache/reasoning splits
(`crates/llm-proxy-storage/src/sqlite.rs:116-121`); agentgateway normalizes
cached-prompt-token accounting (Anthropic/Bedrock exclude, OpenAI/Azure/
Gemini include) and reports normalized plus provider-raw counts for billing
parity (https://agentgateway.dev/docs/standalone/latest/documentation/configuration/resiliency/rate-limits.md,
https://agentgateway.dev/docs/standalone/latest/documentation/llm/observability.md);
TensorZero normalizes structured outputs to constrained decoding or forced
tool calls (https://github.com/tensorzero/tensorzero/blob/main/docs/gateway/generate-structured-outputs.mdx).
Per-field coverage of provider-specific options was not audited for any
product here.

Verdict: gap #6 (high, Responses client endpoint; 15 products, Tyk only in
unreleased 2.3.0; upstream adapter exists at
`crates/llm-proxy-provider/src/adapter/`, client surface does not at
`crates/llm-proxy-server/src/routes/mod.rs:69-79`).

### BYOK, data residency, saturation, and SDKs

`new:` comparison context, not ranked gaps.

| Axis | Market | llm-proxy |
|------|--------|-----------|
| BYOK passthrough | Cloudflare per-request keys with aliased Secrets-Store selection (https://developers.cloudflare.com/ai-gateway/configuration/bring-your-own-keys/); Vercel ordered per-provider credentials via `providerOptions.gateway.byok` with system fallback (https://vercel.com/docs/ai-gateway/authentication-and-byok/byok); OpenRouter prioritized/fallback BYOK with per-key model/user filters (https://openrouter.ai/docs/guides/overview/auth/byok.md); Agent Router ext-authz metadata or `x-aigw-api-key` header (https://theagentrouter.ai/docs/capabilities/security/upstream-auth); Bifrost `x-bf-direct-key` opt-in (https://docs.getbifrost.ai/features/keys-management.md); Portkey OSS by design (https://github.com/Portkey-AI/gateway) | Opt-in passthrough BYOK ships: `passthrough_auth: bool` per provider (`crates/llm-proxy-core/src/provider_config.rs:378-385`) makes the inbound client's bearer/x-api-key token the upstream credential — swapped at `crates/llm-proxy-server/src/routes/core_pipeline.rs:294-304`, flag read at `:727-730` (cite corrected 2026-11-08 per research audit); a missing client token fails 401 'missing client auth token for passthrough provider' (`crates/llm-proxy-server/src/routes/error_response.rs:54`); shipped 2026-06-20 (commit 201e29e) |
| Data residency | Vercel ZDR-signed providers only, pinned regions (https://vercel.com/docs/ai-gateway/security-and-compliance); OpenRouter eu/us endpoints, per-key `allowed_data_regions` (https://openrouter.ai/docs/guides/features/in-region-routing.md); LiteLLM `region_name` fallback filtering (https://docs.litellm.ai/docs/proxy/reliability) | No data-posture-aware routing |
| Saturation | LiteLLM fails over-capacity fast with 429 (https://docs.litellm.ai/docs/routing); uni-api cgroup-aware `RESOURCE_ADMISSION_MODE` admit-or-queue (https://raw.githubusercontent.com/yym68686/uni-api/main/README.md); vLLM stack priority-header routing (https://raw.githubusercontent.com/vllm-project/production-stack/main/src/vllm_router/routers/routing_logic.py); Helicone no queueing on rate limits (https://docs.helicone.ai/features/advanced-usage/custom-rate-limits.md) | 429 with no queueing or prioritization (`crates/llm-proxy-server/src/routes/core_pipeline.rs:177-180` via `error_response.rs:220-224`; cite corrected 2026-11-08 per research audit) |
| SDKs | Vercel native AI SDK integration (https://vercel.com/docs/ai-gateway/sdks-and-apis); OpenRouter generated clients in three languages (https://openrouter.ai/docs/_llms/api-reference.txt); Portkey SDK a superset of the OpenAI SDK (https://portkey.ai/docs/product/ai-gateway/universal-api); Bifrost zero-code drop-ins for six SDKs (https://raw.githubusercontent.com/maximhq/bifrost/main/README.md) | Stock OpenAI/Anthropic SDKs via base-URL swap; no dedicated SDK |

### Upstream cloud authentication

`new:` llm-proxy: single static bearer `api_key`
(`crates/llm-proxy-core/src/provider_config.rs:374-385`); wave-2 grep for
`sigv4|bedrock|vertex|azure|entra|workload-identity` matched only
docs/research design notes and zero Rust/TOML files — AWS SigV4, Azure
Entra, GCP ADC upstreams unreachable. Converged market pattern: a
per-provider auth object separating credential acquisition (ambient cloud
identity chains vs explicit secrets) from request attachment (bearer header
vs SigV4 signature), plus an expiry-driven credential cache.

| Product | What it ships | Source |
|---------|---------------|--------|
| agentgateway | Rust like llm-proxy; all in-process on official SDK crates (aws-config/aws-sigv4, azure_identity, google-cloud-auth, verified in its Cargo.toml): per-backend mutually-exclusive methods (static/aws/azure/gcp/copilot/jwtSign/oauthTokenExchange), SigV4 computed last after every other request mutation, AssumeRole with per-request CEL session tags cached and refreshed before expiry, Entra via full DefaultAzureCredential chain with per-policy caches, GCP ADC access/id tokens incl. impersonated and GDCH service accounts | https://agentgateway.dev/docs/standalone/latest/documentation/configuration/security/backend-authn/, https://raw.githubusercontent.com/agentgateway/agentgateway/main/crates/agentgateway/Cargo.toml |
| LiteLLM | boto3 chains; SigV4 over a filtered header set so forwarded client headers cannot break the canonical request; DualCache IAM policy (static ~59 min, ambient 600s, STS TTL minus 60s; AssumeRole uncached); Vertex token manager with single-flight proactive refresh (FRESH/STALE/INVALID) plus one 401 re-auth retry — source-verified `ref/litellm/litellm/llms/bedrock/base_aws_llm.py:74-148,1400-1439`, `ref/litellm/litellm/llms/vertex_ai/vertex_llm_base.py:480-603` | https://docs.litellm.ai/docs/providers/bedrock, https://docs.litellm.ai/docs/providers/azure, https://docs.litellm.ai/docs/providers/vertex |
| Kong | Per-type provider auth on the AI Model Provider entity (aws assume-role incl. batch_role_arn, azure managed identity/client secret, gcp service-account JSON/Workload Identity Federation) | https://developer.konghq.com/ai-gateway/entities/ai-model-provider/ |
| Agent Router | BackendSecurityPolicy CRD (APIKey/AWSCredentials/AzureCredentials/GCPCredentials) with controller-side OIDC token exchange writing rotated secrets at expiry minus 5 minutes; extproc SigV4 self-corrects the signing region from the resolved Bedrock host | https://theagentrouter.ai/docs/capabilities/security/upstream-auth, https://raw.githubusercontent.com/theagentrouter/agent-router/main/internal/backendauth/aws.go |
| Azure APIM | Managed identities to Azure AI | https://learn.microsoft.com/en-us/azure/api-management/genai-gateway-capabilities |
| AWS AgentCore | Per-target vaulting (token-vault ARNs, OAuth flows, SigV4 via the gateway role) | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-outbound-auth.html |
| Upstream mTLS | agentgateway (backendTLS file cert/key/root; SPIFFE SVID rotation Enterprise); LiteLLM documents a client certificate; Kong's Service entity carries `client_certificate` (untested through its AI plugins); Portkey and Agent Router document none (corrected 2026-11-08 per research audit) | https://agentgateway.dev/docs/standalone/latest/documentation/configuration/security/backend-tls/ |

Verdict: gap #4 (high, 6 products).

### Prompt caching and cache locality

`new:` llm-proxy: `CoreContent::Text` carries `cache:
Option<CacheControl>`, but `CacheControl` has only a `type` field
(`Ephemeral`/`Other`) — no TTL, so Anthropic's 1-hour lifetime is
unrepresentable. At the audited commit a `ttl` on a marker also dropped the
whole block from the upstream request while returning HTTP 200 (audit BUG-2;
fixed by plan 01). The client adapters do translate system-block markers
(`crates/llm-proxy-protocol/src/client/anthropic.rs:117-119`,
`client/openai_chat.rs:84-95`); message-level markers are decoded and then
dropped by the upstream Anthropic adapter
(`crates/llm-proxy-provider/src/adapter/anthropic.rs:640,825`), and nothing
injects markers or routes for cache locality (marker translation corrected
2026-11-08 per research audit;
`crates/llm-proxy-protocol/src/core.rs:142,261-286`). Cache read/write
usage columns already land in the
ledger (`crates/llm-proxy-storage/src/sqlite.rs:116-121`). Market treats
prompt caching as a routing and translation concern:

| Product | What it ships | Source |
|---------|---------------|--------|
| OpenRouter | Bidirectional marker translation (Anthropic `cache_control` ↔ OpenAI `prompt_cache_breakpoint`, TTLs dropped); auto top-level `cache_control` on the last cacheable block; 10-minute sticky routing only when cache-read price beats prompt price; sticky keys from `session_id`/`x-session-id`/`prompt_cache_key`/message hash | https://openrouter.ai/docs/features/prompt-caching |
| LiteLLM | Injection at configured points (role/index/`tool_config` → Bedrock `cachePoint`, OpenAI `prompt_cache_breakpoint`), 4-block cap (v1.84.9), one-flag Anthropic auto-caching (v1.94.0+), markers → Gemini `cachedContents`, 300s deployment pinning by SHA-256 of cacheable prefix + tools (`ref/litellm/litellm/router_utils/prompt_caching_cache.py:56-197`) | https://docs.litellm.ai/docs/tutorials/prompt_caching, https://docs.litellm.ai/docs/completion/prompt_caching |
| Vercel | `caching: 'auto'` classifies explicit-marker vs implicit providers, places two markers (last message; message before last user message, system fallback), advisory `cache_anchor_items`/`cache_ttl` degrading with warnings; session affinity pins provider+credential with per-provider lifetime (5m Anthropic/Bedrock/Google, 10m OpenAI/Azure, 1h DeepSeek, minus 15s, capped 1h), session-ID hierarchy ending in a conversation fingerprint, minimum-prompt-size gating, reported affinity outcome | https://vercel.com/docs/ai-gateway/models-and-providers/automatic-caching, https://vercel.com/docs/ai-gateway/models-and-providers/session-affinity |
| Bifrost | Pins session IDs (incl. auto-adopted Claude Code/Codex/OpenCode headers) to last-serving provider+key for 1h | https://docs.getbifrost.ai/providers/session-affinity.md |
| GPT-Load | Soft affinity from AccessKey/protocol/first-input prefix, LRU 10,000 | https://www.gpt-load.com/docs/internals/scheduling |
| new-api | Per-model channel affinity | https://github.com/QuantumNous/new-api |
| Helicone | Passes Anthropic `cache_control` through, documents pricing multipliers | https://docs.helicone.ai/gateway/concepts/prompt-caching |

Verdict: gap #14 (medium, 7 products).

### Model lifecycle metadata

`new:` llm-proxy: `parse_models` keeps only
id/display_name/context_length
(`crates/llm-proxy-provider/src/discovery.rs:272-288`, verified 2026-10-27);
`/v1/models` hardcodes `created: 0`, null `created_at`
(`crates/llm-proxy-server/src/routes/models.rs:24-36`). Nothing warns or
blocks on a deprecated model.

| Who | What it ships | Source |
|-----|---------------|--------|
| OpenAI | Model object optional `shutdown_date`; notice ≥6 months GA, 3 months specialized, ~2 weeks preview | https://developers.openai.com/api/docs/api-reference/models, https://developers.openai.com/api/docs/deprecations |
| Anthropic | Tri-state `lifecycle` (active/deprecated/retired) + `deprecated_at`/`retires_at`; retired stays listed while inference fails | https://platform.claude.com/docs/en/api/models-list |
| Bedrock | `modelLifecycle` (ACTIVE/LEGACY + timestamps) drops EOL models from listings; 6-month or 45-day Legacy window | https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html, https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html |
| Gemini | Model resource has no lifecycle fields | https://ai.google.dev/api/models |
| OpenRouter | Catalog `expiration_date` (live check: 14 of 464 entries dated, expired delisted), `canonical_slug` permanent aliases, provider-feed deprecation pipeline | https://openrouter.ai/docs/guides/overview/models, https://openrouter.ai/docs/guides/providers/integration |
| LiteLLM | Cost map `deprecation_date` on 874 entries, `/model/deprecations` urgency buckets, daily-lock Slack alerts — never blocks or rewrites | https://github.com/BerriAI/litellm/blob/main/litellm/proxy/common_utils/model_deprecation.py, https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json |
| `new:` all gateways cited | None emits a request-time signal (sunset-style header, response field, staged rejection); LiteLLM's cited cost map defines `successor_model` entries feeding its deprecation guidance, not request rewrites, and no other gateway cited maps to a successor (corrected 2026-11-08 per research audit) | (synthesis of the sourced rows above) |

Verdict: gap #18 (low, 2 gateways + 3 providers).

### Admin and tooling

llm-proxy: CLI serve/stop/status/validate/init/models/autostart
(`apps/llm-proxy/src/cli.rs:23`), read-only usage admin API
(`crates/llm-proxy-api/src/routes.rs:53-55,116-117`), per-provider egress
forward proxying (`crates/llm-proxy-provider/src/transport.rs:293`).
A musl single-binary release workflow exists (`dist-workspace.toml`), but no
release had shipped at the audited commit and the then-active native-tls left
the build unproven (corrected 2026-11-08 per research audit; see the Extras
row above for the full corrected statement).

| Surface | Market | Source |
|---------|--------|--------|
| Admin UIs | LiteLLM DB-backed control plane + Next.js UI; gproxy embedded console; Bifrost web UI with live log streaming; agentgateway UI with key management and CEL playground + experimental agctl CLI; new-api/one-api multi-user consoles with payments; Vercel CLI + REST + MCP control-plane tools; `new:` Higress console, Tyk AI Studio console with edge microgateways, IBM ContextForge admin UI with provider CRUD and model testing, GPT-Load health/log/route-check/usage UI, Respan dashboard | https://docs.litellm.ai/docs/proxy/virtual_keys, https://raw.githubusercontent.com/LeenHawk/gproxy/main/README.md, https://docs.getbifrost.ai/features/observability/default, https://agentgateway.dev/docs/standalone/latest/documentation/llm/cost-controls/costs.md, https://github.com/QuantumNous/new-api, https://vercel.com/docs/ai-gateway/sdks-and-apis, https://higress.cn/en/docs/latest/user/quickstart/, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/edge-gateways.md, https://ibm.github.io/mcp-context-forge/latest/api/llm-proxy/, https://www.gpt-load.com/docs/monitor, https://www.respan.ai/docs/ |
| Prompt management | TensorZero prompts as config variants; Langfuse versioned prompts with stale-while-revalidate caching; Portkey immutable versioning A/B-tested via loadbalance configs | https://github.com/tensorzero/tensorzero/blob/main/docs/experimentation/run-adaptive-ab-tests.mdx, https://langfuse.com/docs/prompt-management/features/caching.md, https://portkey.ai/docs/product/prompt-engineering-studio/prompt-versioning |
| Audit trails `new:` | LiteLLM entity lifecycle audit with before/after values and S3 export (Enterprise edition — corrected 2026-11-08 per research audit); Kong RSA-signed audit logs with TTL purge; Cloudflare config-change audit on all plans; Portkey audit logs API; new-api security audit log with step-up verification | https://docs.litellm.ai/docs/proxy/audit_logs, https://developer.konghq.com/gateway/audit-logs, https://developers.cloudflare.com/ai-gateway/reference/audit-logs/, https://portkey.ai/docs/product/enterprise-offering/audit-logs, https://github.com/QuantumNous/new-api/releases |
| Extensibility `new:` | LiteLLM CustomLogger callbacks + request-lifecycle hooks (modify/reject/stream/header) and custom auth/key-policy hooks; Bifrost Go `.so` plugins with hot reload incl. MCP hooks; Kong Lua/PDK plugins, serverless snippets, external runners; Higress and APISIX proxy-wasm in Go/Rust/C++; Portkey compiled TypeScript plugin manifests; Tyk one gRPC-isolated SDK for gateway/UI/analytics plugins. Hook names, registration, and lifecycle are bespoke — no common API emerged. Agent Router ships no custom extension point (ext_proc request closed stale) | https://docs.litellm.ai/docs/proxy/call_hooks, https://docs.litellm.ai/docs/observability/custom_callback, https://docs.getbifrost.ai/plugins, https://developer.konghq.com/custom-plugins/reference, https://developer.konghq.com/plugins/pre-function/, https://github.com/alibaba/higress/tree/main/plugins, https://apisix.apache.org/docs/apisix/plugins/ai-proxy/, https://github.com/Portkey-AI/gateway/blob/main/plugins/README.md, https://raw.githubusercontent.com/TykTechnologies/ai-studio/main/docs/site/docs/plugins-overview.md, https://theagentrouter.ai/ |
| Config lifecycle `new:` | agentgateway hot-reloads everything except a startup-only list (adminAddr/storage/database/logging/tracing) via debounced file watch (re-registering on atomic replace), validate-then-swap, last-known-good fallback, `config_synchronized` metric, `--validate-only`, file/hybrid/readOnly modes; uni-api compiles validated snapshots atomically and swaps an all-`Arc` immutable snapshot under one write with revision-CAS overlays; LiteLLM converges pods by 30s DB polling, YAML-bootstrap-then-DB-overlay, router-update errors caught and swallowed (`ref/litellm/litellm/proxy/proxy_server.py:5027-5099`); Higress xDS push in milliseconds; Kong data planes fed from Konnect; APISIX dynamic from etcd; oc-go-cc fsnotify+SIGHUP (`ref/oc-go-cc/internal/config/watcher.go:19-101`); openproxy SIGUSR2 zero-downtime upgrade. In-gateway config history with rollback does exist at Cloudflare (dynamic-route versions, instant rollback) and Azure APIM (revisions); among the self-hosted products cited, none ships it — Kong decK keeps history in git (corrected 2026-11-08 per research audit) | https://agentgateway.dev/docs/standalone/latest/documentation/setup/update, https://github.com/agentgateway/agentgateway/blob/main/crates/agentgateway/src/state_manager.rs, https://github.com/yym68686/uni-api/blob/main/src/runtime/state.rs, https://docs.litellm.ai/docs/proxy/configs, https://docs.litellm.ai/docs/proxy/prod, https://github.com/higress-group/higress/blob/main/docs/architecture.md, https://developer.konghq.com/ai-gateway/, https://apisix.apache.org/docs/apisix/admin-api/, https://raw.githubusercontent.com/x5iu/openproxy/master/README.md, https://raw.githubusercontent.com/Kong/deck/main/README.md |
| Rust performance | LiteLLM migrating translation to a bundled Rust core (beta, off by default); Portkey self-hosted gateway runs a Rust front-end from v2.30.0; TensorZero claimed 0.94 ms p99 at 10k QPS — self-published and archived. llm-proxy overhead was not measured; no performance comparison asserted | https://docs.litellm.ai/docs/proxy/rust_gateway, https://portkey.ai/docs/self-hosting/major-performance-improvements.md, https://github.com/tensorzero/tensorzero/blob/main/docs/gateway/benchmarks.mdx |

Verdict: gap #16 (low, hot reload, 8 products — llm-proxy rejects
`server.hot_reload` at `crates/llm-proxy-core/src/provider_config.rs:1531-1538`)
(superseded: plan 07 hot-reloads provider and pool files on SIGHUP;
`server.hot_reload = true` is accepted);
additional gap #19 (web admin UI, 11 products). Prompt management, audit
trails, and hook systems have no llm-proxy counterpart but were surveyed as
context.

### Transport, state, and deployment topology

llm-proxy: plain-HTTP listener; TLS only on the upstream client — and at
the audited commit the active stack was native-tls/OpenSSL, not rustls:
reqwest's default features were on, compiling `default-tls` and
`rustls-tls` together, and reqwest selects native-tls
(`crates/llm-proxy-provider/Cargo.toml:23`, corrected 2026-11-08 per
research audit; plan 03 later disabled the defaults, making rustls the
single stack in `crates/llm-proxy-transport/Cargo.toml`); a 2026-10-14 grep
for server-side TLS (rustls/axum-server/certificate) across workspace
Cargo.toml files returns no listener TLS dependency — HTTPS fronting is
delegated to a reverse proxy; openproxy shows the in-proxy alternative
(rustls TLS 1.3 listener,
https://raw.githubusercontent.com/x5iu/openproxy/master/README.md). All
state is single-node: local SQLite ledger (`config.toml.example:56-68`);
rate limiter, dedup window, metrics, and catalog refresh in-process — N
instances mean N independent limits and N ledgers.

| Product | What it ships | Source |
|---------|---------------|--------|
| LiteLLM | Limits, cooldowns, and spend in Redis/Postgres for multi-proxy deployments | https://docs.litellm.ai/docs/proxy/load_balancing |
| gproxy | Rate-limit counters shared via a cache backend | https://gproxy.leenhawk.com/guides/permissions/ |
| Bifrost | OSS governance explicitly single-node ("running multiple OSS Bifrost nodes with a Postgres backend is not supported") — the same shape llm-proxy has | https://docs.getbifrost.ai/features/governance/budget-and-limits |
| new-api | Per-node Redis limits count independently | https://github.com/QuantumNous/new-api |
| AWS AgentCore `new:` | VPC-internal targets and identity providers through VPC Lattice private endpoints | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-vpc-egress.html |
| Cloudflare `new:` | Custom gateway domains; custom-provider registry for arbitrary HTTPS providers | https://developers.cloudflare.com/ai-gateway/configuration/custom-domains/, https://developers.cloudflare.com/ai-gateway/configuration/custom-providers/ |
| Kong `new:` | 2.2.0 ships FIPS 140-3 and distroless variants | https://developer.konghq.com/ai-gateway/changelog/ |

Verdict: single-node state matches the OSS Bifrost posture; listener TLS
delegated to a reverse proxy.

## Complete gap register

Severities are the research runs' final finding tiers; the audited analysis
orders the same gaps by impact — ranked by impact on a self-hosted proxy
serving coding-agent workloads: request-path reliability and spend/rate
control first, ecosystem interop second, operator convenience last. "Ships
at" counts products named across both waves.

| # | Severity | Gap | Ships at | Evidence | Detail |
|---|----------|-----|----------|----------|--------|
| 1 | high | Request-path retries, fallback chains, provider cooldowns | 22 products (LiteLLM, Portkey, Bifrost, Cloudflare, Agent Router, Higress, Kong, OpenRouter, Vercel, TensorZero, one-api, new-api, gproxy, oc-go-cc, agentgateway, AWS AgentCore, Tyk, F5, uni-api, GPT-Load, Respan, llm-api-key-proxy); 21 without a paid license — Kong's is `ai-proxy-advanced` | https://docs.litellm.ai/docs/proxy/reliability | No request-path retry/fallback in llm-proxy; scoped future work (`docs/research/core-protocol.md:170-186`) |
| 2 | high | Per-key spend budgets with enforcement and periodic resets | 15 USD budgets (LiteLLM, Bifrost, OpenRouter, Vercel, Cloudflare, LangSmith, gproxy, BricksLLM, agentgateway, Portkey, Tyk, Respan, GPT-Load, uni-api, Traefik Hub); 12 self-hostable without a paid edition or beta (Portkey budgets enterprise, Cloudflare spend limits Beta, Tyk enforcement Enterprise); one-api/new-api/Azure APIM enforce token quotas instead | https://docs.getbifrost.ai/features/governance/budget-and-limits.md | Pricing estimation-only, no enforcement (`crates/llm-proxy-core/src/provider_config.rs:472`) |
| 3 | medium | Token-based (TPM) and concurrency rate limits — per-key RPM already exists for authenticated callers (token bucket keyed by key id, core_pipeline.rs:172-176, since 2026-10-13); unauthenticated callers fall back to per-IP (narrowed 2026-10-28: per-key RPM now exists) | 14 (LiteLLM, Kong, Agent Router, Higress, Bifrost, gproxy, new-api, agentgateway, AWS AgentCore, Traefik Hub, Respan, GPT-Load, uni-api, TensorZero); 13 without a paid license — Kong's is `ai-rate-limiting-advanced` | https://docs.litellm.ai/docs/proxy/users | TPM and concurrency caps absent; limiter is a burst-tolerant token bucket keyed by key id when authenticated else client IP (`crates/llm-proxy-server/src/middleware.rs:117-129`) |
| 4 | high | Cloud-native upstream auth (SigV4/STS, Entra refresh, ADC) | 6 (agentgateway, LiteLLM, Kong, Agent Router, Azure APIM, AWS AgentCore) | https://agentgateway.dev/docs/standalone/latest/documentation/configuration/security/backend-authn/ | Static bearer only (`crates/llm-proxy-core/src/provider_config.rs:374-385`) |
| 5 | high | Upstream credential pools with rotation, cooldowns, quota windows | 8 (llm-api-key-proxy, gproxy, Bifrost, Higress, new-api, openproxy, GPT-Load, uni-api) | https://github.com/Mirrowel/LLM-API-Key-Proxy | One static key per provider (`crates/llm-proxy-core/src/provider_config.rs:375`); rotator defaults `ref/llm-api-key-proxy/src/rotator_library/config/defaults.py:139-158` |
| 6 | high | OpenAI Responses client endpoint (`/v1/responses`) | 15 (LiteLLM, Portkey OSS, OpenRouter, Vercel, LangSmith, Helicone, new-api, Agent Router, agentgateway, AWS AgentCore, Cloudflare, Respan, uni-api, GPT-Load, Martian); Tyk only in unreleased 2.3.0 | https://vercel.com/docs/ai-gateway/sdks-and-apis | Upstream Responses adapter exists, client surface does not (`crates/llm-proxy-server/src/routes/mod.rs:69-79`) |
| 7 | medium | Guardrails (PII redaction, moderation, injection detection) | 17 products; 15 without a paid license or beta (Kong's is `ai-sanitizer`; Cloudflare Guardrails and DLP are Beta) | https://docs.litellm.ai/docs/proxy/guardrails/quick_start | llm-proxy ships none |
| 8 | medium | MCP gateway functionality | 11 (LiteLLM, Cloudflare, agentgateway, Bifrost, Kong, Azure APIM, AWS AgentCore, IBM ContextForge, Higress, Glama, Tyk); 10 without a paid license (Kong's is `ai-mcp-proxy`; Tyk's MCP proxying with upstream OAuth 2.1 is Enterprise) | https://docs.litellm.ai/docs/mcp | No MCP client, server, or federation |
| 9 | medium | Client-key lifecycle API without restart | 12 (LiteLLM, Bifrost, OpenRouter, Vercel, rust-llm-proxy, one-api, Portkey, agentgateway, GPT-Load, Respan, Tyk, uni-api); 11 without a paid edition (Portkey key rotation is enterprise) | https://portkey.ai/docs/product/enterprise-offering/org-management/api-key-rotation | Keys are static config (`config.toml.example:42-54`) |
| 10 | medium | Exact-match response caching | 8 exact-match (LiteLLM, Portkey, Cloudflare, Helicone, OpenRouter, Higress, Respan, Tyk — Portkey's OSS cache is exact-match in memory only); semantic at Bifrost, Kong (paid `ai-semantic-cache`), Azure APIM, LiteLLM Qdrant, vLLM alpha | https://portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic | No cache; the opt-in dedup window (off by default) rejects the second caller (`crates/llm-proxy-server/src/routes/core_pipeline.rs:182-185`) |
| 11 | medium | OTel GenAI trace export | 9 (Cloudflare, Vercel, OpenRouter, agentgateway, LiteLLM, Bifrost, Tyk, IBM ContextForge, vLLM stack) | https://developers.cloudflare.com/ai-gateway/observability/otel-integration/ | No OTLP span export |
| 12 | medium | Prometheus /metrics scrape endpoint | 9 (LiteLLM, agentgateway, Bifrost, Kong, TensorZero, vLLM stack, Higress, Tyk, IBM ContextForge) | https://docs.litellm.ai/docs/proxy/prometheus | Counters in-process only (`crates/llm-proxy-core/src/metrics.rs`) |
| 13 | medium | Weighted/multi-strategy load balancing per model group | 12 (LiteLLM, Portkey, Kong, gproxy, Bifrost, OpenRouter, Cloudflare Auto Router, Higress, uni-api, GPT-Load, AWS AgentCore, Tyk); 10 without a paid license or beta (Kong's is `ai-proxy-advanced`; Cloudflare Auto Router is Beta) | https://developer.konghq.com/plugins/ai-proxy-advanced/ | Alias resolves to exactly one provider (`crates/llm-proxy-core/src/provider_config.rs:392-394`); inline: https://developers.cloudflare.com/ai-gateway/features/auto-router/, https://github.com/higress-group/higress/blob/main/plugins/wasm-go/extensions/ai-load-balancer/README_EN.md |
| 14 | medium | Cross-provider prompt-cache plumbing and cache-locality routing | 7 (OpenRouter, LiteLLM, Vercel, Helicone, Bifrost, GPT-Load, new-api) | https://vercel.com/docs/ai-gateway/models-and-providers/automatic-caching | `CacheControl` cannot express TTLs; system-block markers translate but message-level markers are dropped, and nothing injects or pins (`crates/llm-proxy-protocol/src/core.rs:261-286`, corrected 2026-11-08 per research audit) |
| 15 | medium | Upstream credential encryption at rest / secret-manager references | 8 (gproxy, LiteLLM, Kong, Tyk, IBM ContextForge, new-api, GPT-Load, Portkey); 7 without a paid edition (Portkey secret references and KMS are enterprise/hosted) | https://docs.litellm.ai/docs/secret_managers/overview.md | Plaintext TOML |
| 16 | low | Config hot reload without restart | 8 (agentgateway, uni-api, Higress, Kong, APISIX, LiteLLM, oc-go-cc, openproxy) | https://agentgateway.dev/docs/standalone/latest/documentation/setup/update | `server.hot_reload` rejected (`crates/llm-proxy-core/src/provider_config.rs:1531-1538`) (superseded: plan 07 hot-reloads provider and pool files on SIGHUP; `server.hot_reload = true` is accepted) |
| 17 | low | No exposure surface for latency percentiles (no /metrics; snapshots and p95/p99 helpers exist in-process only, metrics.rs:210-239) | 9 (LiteLLM, agentgateway, Bifrost, oc-go-cc, vLLM stack, Kong, Tyk, Helicone, AWS AgentCore) | https://agentgateway.dev/docs/standalone/latest/documentation/observability/metrics/reference/ | Latency samples and percentile helpers in-process only (`crates/llm-proxy-core/src/metrics.rs:34-35,210-239`); TTFT unmeasured |
| 18 | low | Model lifecycle metadata (deprecation dates, successor aliases) | 2 gateways (LiteLLM, OpenRouter); 3 providers publish fields (OpenAI, Anthropic, Bedrock) | https://openrouter.ai/docs/guides/overview/models | Discovery discards it (`crates/llm-proxy-provider/src/discovery.rs:272-288`) |
| 19 | — (report rank 19) | Web admin UI | 11 (LiteLLM, gproxy, Bifrost, agentgateway, new-api, one-api, Higress, Tyk, IBM ContextForge, GPT-Load, Respan) | https://docs.litellm.ai/docs/proxy/virtual_keys | Admin is read-only JSON (`crates/llm-proxy-api/src/routes.rs:53-55,116-117`) |
| 20 | — (report rank 20) | Stream-cancellation usage settlement | 2 document accounting for cancelled requests (LiteLLM: per-chunk usage; its `cancel_on_disconnect` is documented for non-streaming calls — OpenRouter: per-provider cancel-billing + final usage chunk; corrected 2026-11-08 per research audit) | https://docs.litellm.ai/docs/proxy/config_settings | `ClientCancelled` persists with no usage payload — delivered tokens never settle (`crates/llm-proxy-server/src/routes/core_pipeline.rs:952-985`, `crates/llm-proxy-storage/src/lib.rs:80`); cancellation propagation itself already works upstream once streaming; OpenRouter: https://openrouter.ai/docs/api_reference/streaming |

Prevalence counts include the audited analysis's `new:` additions (Apache
APISIX, Traefik Hub), which sit outside the 67-item inventory. They mean
"someone sells this", not self-hostable parity: rows affected by paid-tier
or beta features restate the count without them (corrected 2026-11-08 per
research audit — Kong `ai-proxy-advanced`/`ai-rate-limiting-advanced`/
`ai-semantic-cache`/`ai-sanitizer`/`ai-mcp-proxy` need an AI license;
Portkey budgets, key rotation, secret references and KMS are enterprise or
hosted; LiteLLM audit logs are Enterprise; Tyk upstream OAuth is
Enterprise; Cloudflare Guardrails, DLP, Auto Router and spend limits are
Beta).

Third wave (November 2026): the niche survey
([`niche-repos.md`](niche-repos.md)) surfaces nine candidate gaps that fall
outside this register — timeout hierarchies, request-payload cost
transforms, shadow-eval quality guards, OAuth credential lifecycle for
non-cloud upstreams, engine-state-aware
routing, admission queueing, caller SSO/RBAC, hash-verified price catalogs,
and hedged request races. Upstream cancellation propagation was removed
from this list: llm-proxy already cancels the upstream request when the
client disconnects once streaming, in every phase (audit probe E6); what is
missing there is usage settlement and a settled row for disconnects before
the first event (corrected 2026-11-08 per research audit). Each remaining
gap is defined and evidenced there and tagged per
capability in [`repos-registry.json`](repos-registry.json); they are
candidates for register entries, not yet ranked.

Already-scoped future work (docs/research/endpoints.md) — embeddings,
rerank, realtime, transcription/TTS, image/video generation, files, OCR,
batch, fine-tuning, search bridges — is not re-listed. BYOK, data
residency, queueing/prioritization, and SDKs are comparison context, not
ranked gaps.

## Implementation blueprints

One subsection per deep dive (6 wave-1, 4 wave-2); market mechanisms are
summarized, with full sourcing in the capability tables above. Where a deep
dive was deferred from near-term candidates, the deferral and its reason
are stated as the analysis gives them.

### 1. Request-path retries, fallback chains, cooldowns (wave 1)

Market: per-exception retry policies, typed fallback chains, cooldowns tripped on 429 or >50% failure rate; no fallback after stream start; default-on circuit breakers (Routing table). llm-proxy: the first-byte window cannot host this as built — 429/5xx/connect
errors return at `send_stream`
(`crates/llm-proxy-server/src/routes/core_pipeline.rs:792-808`) before
`FirstByteResult` exists, and the window closes on the first encoded
event, a prelude frame (`message_start`, role-only chat chunk,
`response.created`, first Gemini chunk) before any visible output
(corrected 2026-11-08 per research audit); fallback needs retries wrapped
around the send plus a bounded hold before commit, and after the first
event the in-band error invariant
(`docs/protocol.md:139-151`). Done when a 429/5xx/connect error from the
primary advances to the configured fallback before commit and the ledger
records both attempts — requires extending the usage-row schema with an
attempt ordinal and provider per upstream attempt (or an attempts-count
column), a storage migration scoped into this candidate
(`crates/llm-proxy-storage/src/sqlite.rs:105-131`). Plan 04 shipped this:
classified retries around the send, per-route hold budgets, pools and
attempt-level ledger rows.

### 2. Client-key budgets and virtual-key lifecycle (wave 1)

Market: layered USD budgets with periodic resets — stackable multi-window limits, calendar-aligned resets, reservation; lazy window reset (https://github.com/maximhq/bifrost/blob/dev/plugins/governance/store.go); Block/Audit modes; status codes not converged (Cost table). llm-proxy: check-then-accrue — reject when the in-memory counter is over
budget, accrue actual cost after completion against the existing SQLite
ledger (`crates/llm-proxy-storage/src/sqlite.rs:105`). Two preconditions
the candidate originally glossed (corrected 2026-11-08 per research
audit): streaming cost is often NULL — the default chat stream sent
`include_usage: false` and the documented Anthropic delta yielded no usage
— so "accrue settled cost" needs the stream usage fixes first; and no key
record exists to persist a window start with (clients are a TOML map of
id→PHC hash), nor an API to rehydrate spend from the write-only ledger.
Deliberately not reservation-based: bounded overshoot from in-flight calls
is accepted, and the cited LiteLLM reservation-skips-when-unestimatable
behaviour comes from
https://github.com/BerriAI/litellm/issues/35524, closed 2026-10-08 with a
linked fix — a historical example, not a current one. Done when
an over-budget key gets 402 pre-dispatch, settled cost lands in the ledger,
and accrued spend survives a restart. Virtual-key lifecycle
(issuance/rotation/revocation without restart) is deferred: it needs a
control-plane decision the candidate avoids; current shape
`config.toml.example:42-54`. Plan 05 resolved the admission-model
contradiction with one reserve-then-settle `Admission` step and shipped
budgets, windows and key records.

### 3. TPM and per-key/concurrency rate limits (wave 1)

Market: batched pre-call checks per key/user/team with reserve-then-reconcile TPM; token-bucket vs sliding-window split; TPM (429) combined with long-period quotas (403) (Rate-limiting table). llm-proxy: per-key RPM already exists for authenticated callers (`crates/llm-proxy-server/src/routes/core_pipeline.rs:172-176`); TPM and concurrency caps would reuse the offline token counter
(`crates/llm-proxy-core/src/token/tiktoken.rs`, heuristic documented at
`crates/llm-proxy-core/src/token/counter.rs:5`) for reserve-then-reconcile
TPM — reservation suits tokens because estimates are cheap and refundable —
keyed by CallerId as the RPM limiter already is. Done when over-limit
requests get 429 with retry-after and rate-limit headers, reconciled to
provider-reported usage after completion.

### 4. Exact-match response caching (wave 1)

Market: SHA-256 keys over recognized parameters; 60s TTL floors; single-flight absent from the Portkey OSS, Helicone OSS, and LiteLLM cache code examined at source (Caching table). llm-proxy: cache keyed on a canonical `CoreRequest` serialization,
in-memory v1 with a bounded TTL (floor/default/max per the 60s-floor
convention), an entry cap, and LRU eviction under the cap. Done when a
second identical request is served from the store with a HIT header and
zero recorded usage, entries expire at the TTL and evict under the cap.
The dedup window cannot coexist in front of this: it is off by default,
keys raw path+body (not `CoreRequest`) from first sighting, and runs
before any cache lookup, so with it on the second identical request gets
its 409 where a HIT is required — the original done-condition was
self-contradictory (corrected 2026-11-08 per research audit). Coalescing
(serving the second caller) remains unbuilt.

### 5. Upstream credential pools (wave 1)

Market: same-key server-error retries under a global deadline with rotator defaults (`ref/llm-api-key-proxy/src/rotator_library/config/defaults.py:139-158`); key-pool failover with cooldown restore; tier-priority scheduling with credential isolation (Routing table). llm-proxy: optional `[[provider.credentials]]` tables extending the single
`api_key` field (`crates/llm-proxy-core/src/provider_config.rs:375`) with
per-credential cooldown state and quota windows. This is not "without
changing request handling" as the build-order list claimed, and the
existing storage layer is a write-only usage ledger with no state store,
so cooldown persistence needs new state storage (both corrected
2026-11-08 per research audit). Done when a credential that 401s is locked
out, its cooldown
survives restart, and the credential id appears in ledger rows. Plan 04
shipped credential selection and failure feedback in the attempt loop with
the key id in ledger rows.

### 6. Prometheus /metrics and OTel GenAI telemetry export (wave 1)

Market: OTel GenAI semconv metric names (`gen_ai.server.request.duration`, `time_to_first_token`, `time_per_output_token`); latency histograms with identity-labeled counters and budget-remaining gauges; secure-by-default `/metrics`; standardized client-inference span shape (Observability table). llm-proxy: hand-rendered text exposition walking the metrics snapshot
(`crates/llm-proxy-core/src/metrics.rs`), served on the admin listener.
Done when the endpoint scrapes counters plus a latency histogram and TTFT
for streams. OTel GenAI trace export is deferred (needs a dependency
decision the candidate avoids).

### 7. Cloud-native upstream auth: SigV4/STS, Entra, ADC, mTLS (wave 2)

Market: per-provider auth object separating credential acquisition from request attachment plus an expiry-driven credential cache; SigV4 computed last after every other request mutation; DualCache IAM policy (static ~59 min, ambient 600s, STS TTL minus 60s) and single-flight proactive token refresh with one 401 re-auth retry (`ref/litellm/litellm/llms/bedrock/base_aws_llm.py:74-148,1400-1439`, `ref/litellm/litellm/llms/vertex_ai/vertex_llm_base.py:480-603`); OIDC token exchange writing rotated secrets at expiry minus 5 minutes; upstream mTLS ships at agentgateway (file cert/key/root; SPIFFE rotation Enterprise), LiteLLM documents a client certificate, and Kong's Service entity carries `client_certificate` (untested through its AI plugins) — not agentgateway-only (corrected 2026-11-08 per research audit; Upstream-auth table). llm-proxy: static bearer only
(`crates/llm-proxy-core/src/provider_config.rs:374-385`); the wave-2 grep
matched zero Rust/TOML files. Deferred from near-term candidates: needs a
config/schema and dependency decision.

### 8. Cross-provider prompt-cache plumbing (wave 2)

Market: bidirectional marker translation (`cache_control` ↔ `prompt_cache_breakpoint`, TTLs dropped; Bedrock `cachePoint`, Gemini `cachedContents`, 4-block cap); auto-injection with advisory anchors that degrade with warnings; sticky cache-locality routing — 10-min pin only when cache-read price beats prompt price, 300s pin by SHA-256 of cacheable prefix + tools, per-provider pin lifetimes capped 1h, LRU-10,000 soft affinity (Prompt-caching table). llm-proxy: more than a core field (corrected 2026-11-08 per research
audit): a `ttl` on the wire marker makes the containing block unparseable
today (the Anthropic request decoder deletes the whole block, HTTP 200,
WARN only), `cache` exists only on `Text`, message-level markers are
dropped by the upstream adapter, and the "ledger records cache-write
tokens" half needs the Anthropic stream usage decode fixed. Fix spans the
wire `CacheControl` (`crates/llm-proxy-protocol/src/anthropic.rs:113`,
`openai.rs:11`, core at `:261-266`), `cache` slots on the other content
kinds, the adapter drop at `crates/llm-proxy-provider/src/adapter/anthropic.rs:640,825`
and the stream decoder at `:337-361`; cache read/write usage columns
already exist
(`crates/llm-proxy-storage/src/sqlite.rs:116-121`). Done when a request
carrying a 1h marker on `/v1/messages` produces the same marker upstream —
on any block kind — and stream runs record cache-write tokens. Plan 01
shipped the wire TTL, block-level markers and the stream usage merge.

### 9. Hot-reloadable configuration (wave 2)

Market: debounced file watch with validate-then-swap, last-known-good fallback, `config_synchronized` metric, `--validate-only`, startup-only exclusion list; immutable-snapshot swap with revision-CAS overlays; 30s DB polling with swallowed router-update errors (`ref/litellm/litellm/proxy/proxy_server.py:5027-5099`); fsnotify+SIGHUP; SIGUSR2 zero-downtime upgrade; no versioned config with in-gateway history/diff/rollback (Admin table). llm-proxy: `server.hot_reload=true` rejected at load
(`crates/llm-proxy-core/src/provider_config.rs:1531-1538`) (superseded: plan
07 hot-reloads provider and pool files on SIGHUP; `server.hot_reload = true`
is accepted). Deferred from near-term candidates: needs a config/schema
decision.

### 10. Model lifecycle metadata in catalogs and /v1/models (wave 2)

Market: providers publish `shutdown_date`, tri-state `lifecycle` + `deprecated_at`/`retires_at` (retired stays listed, inference fails), `modelLifecycle` (6-month or 45-day window); Gemini none. Gateways: catalog `expiration_date` (14 of 464 dated, expired delisted) + `canonical_slug` aliases; cost-map `deprecation_date` on 874 entries, urgency buckets, daily-lock Slack alerts, never blocks; no request-time signal anywhere cited, and LiteLLM's cost map defines `successor_model` entries feeding deprecation guidance (not request rewrites) while no other gateway cited maps to a successor (corrected 2026-11-08 per research audit; Model-lifecycle table). llm-proxy: extend `StaticModelCatalogEntry`
(`crates/llm-proxy-core/src/provider_config.rs:735-747`, currently
`deny_unknown_fields`, so new fields need `#[serde(default)]`) and
discovery's `parse_models` to capture per-kind lifecycle fields (OpenAI
`shutdown_date`, Anthropic `lifecycle`/`retires_at`, Bedrock-style
`modelLifecycle`), surfaced per client shape in `/v1/models`
(`crates/llm-proxy-server/src/routes/models.rs:24-36`) with retired entries
kept listed. Done when both client protocols list deprecation dates and a
warn-level log fires on deprecated-model use; blocking stays opt-in and out
of scope.

## Build-order candidates

From the analysis's near-term list; each names its code touchpoint and
done-condition. Gaps without a candidate (OTel traces, MCP, key lifecycle,
guardrails, load-balancing strategies, the Responses client endpoint, admin
UI, cloud-native upstream auth, hot reload (hot reload since done: plan 07),
credential encryption at rest,
stream-cancellation usage settlement) are deferred: each needs a
dependency, config/schema, or control-plane decision the candidates avoid.

1. Pre-first-event retry/fallback — the pre-commit window does not admit
   this as built: 429/5xx/connect errors return at `send_stream`
   (`crates/llm-proxy-server/src/routes/core_pipeline.rs:792-808`) before
   `FirstByteResult` exists, and the window closes on the first encoded
   event, a prelude frame before any visible output
   (`core_pipeline.rs:702-705`; corrected 2026-11-08 per research audit);
   fallback needs retries around the send plus a bounded hold before
   commit; done includes the attempts migration
   (`crates/llm-proxy-storage/src/sqlite.rs:105-131`).
2. Client-key budgets, check-then-accrue — the ledger does not already
   record settled streaming cost (the default chat stream sent
   `include_usage: false` and the documented Anthropic delta yielded no
   usage, so streaming spend was often NULL), and it is optional and
   write-only from the server, so enforcement is not just a counter over
   it (corrected 2026-11-08 per research audit); the cited
   reservation-skips example
   (https://github.com/BerriAI/litellm/issues/35524) closed 2026-10-08
   with a linked fix; done: 402 pre-dispatch,
   settled cost in ledger, spend survives restart.
3. TPM (per-key RPM already exists, `core_pipeline.rs:172-176`) — the
   offline token counter exists and token estimates are cheap and
   refundable; done: 429 with retry-after and rate-limit headers,
   reconciled to provider-reported usage.
4. Prometheus `/metrics` with latency/TTFT histograms — counters and
   latency samples already exist (`crates/llm-proxy-core/src/metrics.rs`);
   done: admin-listener endpoint scraping counters plus latency histogram
   and stream TTFT.
5. Exact-match response cache — canonical `CoreRequest` serialization gives
   a stable key; the dedup window keys raw path+body from first sighting
   and runs before any cache lookup, so it cannot front the cache (with it
   on, the second identical request gets 409 where a HIT is required;
   corrected 2026-11-08 per research audit); done: HIT header, zero usage
   on hit, TTL expiry, LRU eviction under cap.
6. Credential pools — `[[provider.credentials]]` extends the single
   `api_key` field, which requires credential selection and failure
   feedback in the pipeline plus a ledger column for the key used — not
   "without changing request handling" (corrected 2026-11-08 per research
   audit); done: 401 lockout,
   cooldown survives restart, credential id in ledger rows.
7. Prompt-cache TTL and marker passthrough — `CacheControl` is not the
   only missing piece: the wire marker, `cache` slots on non-text content
   kinds, the adapter's message-level marker drop and the Anthropic stream
   usage decode all need work (corrected 2026-11-08 per research audit;
   the ledger columns already exist); done: 1h marker on
   `/v1/messages` round-trips upstream, cache-write tokens land in the
   ledger.
8. Model lifecycle capture — discovery already parses model lists, so this
   is field work plus `#[serde(default)]` on `StaticModelCatalogEntry`;
   done: both client protocols list deprecation dates, warn-level log on
   deprecated-model use, blocking opt-in and out of scope.

## Research coverage appendix

Every inventory item from both waves, with outcome. "Not detailed in the
analysis" marks items the audited report carried no usable findings for.

### Wave 1 — products (19)

| Item | Outcome | Where |
|------|---------|-------|
| LiteLLM | Findings noted | Auth, Routing, Rate limiting, Cost, Caching, Guardrails, Observability, MCP, Protocol, Upstream auth, Prompt caching, Model lifecycle, Admin; `ref/litellm` pins |
| Portkey | Findings noted | Auth, Routing, Cost, Caching, Guardrails, Admin, Protocol |
| Kong AI Gateway | Findings noted | Auth, Routing, Rate limiting, Caching, Guardrails, Observability, MCP, Protocol, Upstream auth, Admin, Transport |
| Cloudflare AI Gateway | Findings noted | Auth, Routing, Rate limiting, Cost, Caching, Guardrails, Observability, MCP, Protocol, BYOK, Transport |
| Envoy AI Gateway (renamed Agent Router) | Findings noted | Market map status; Routing, Rate limiting, Guardrails (documented absence), Protocol, BYOK, Upstream auth |
| Higress | Findings noted | Auth, Routing, Rate limiting, Caching, Guardrails, Observability, MCP, Protocol, Admin and tooling (xDS config push) |
| kgateway | Findings noted — Gateway API edge features (JWT/OAuth2/OIDC + ext-proc auth, local + global rate limiting, retries with per-try timeouts and outlier detection, priority failover, transformations, mirroring); Envoy-based AI Gateway deprecated in v2.1.x, removed in v2.2.0, became agentgateway; no in-tree AI gateway, LLM cost tracking, prompt guardrails, token rate limiting, or MCP | Market map |
| TensorZero (archived 2026-06-12) | Findings noted | Market map status; Routing, Rate limiting, Protocol, Observability, Admin (benchmarks) |
| Helicone | Findings noted | Cost, Caching, Guardrails, Observability (TTFT, payload, metadata), Protocol, BYOK |
| Langfuse and LangSmith | Findings noted | Langfuse: Protocol (narrow relay), Admin (prompt caching); LangSmith: Protocol, Cost, Guardrails |
| OpenRouter | Findings noted | Auth, Routing, Cost, Caching, Observability (incl. streaming SLO), Protocol, BYOK, Prompt caching, Model lifecycle |
| Vercel AI Gateway | Findings noted | Auth, Routing, Cost, Observability, Protocol, BYOK, Prompt caching |
| Glama | Findings noted (thin: MCP gateway per-tool ACLs; `x-glama-metadata`) | MCP, Observability |
| one-api and new-api | Findings noted | Auth, Routing, Rate limiting, Cost, Guardrails, Protocol, Prompt caching, Admin, Transport; dormancy status |
| Vendored reference checkouts (oc-go-cc, llm-proxy, llm-api-key-proxy under ref/) | Findings noted for oc-go-cc, llm-api-key-proxy, and `ref/litellm`; the `ref/llm-proxy` checkout (fb588f6 per `ref/REFS`) — an in-process Python routing library (proxyllm v0.2.3), uncited by the analysis — is researched under Reference repositories | Routing, Observability, Admin, config lifecycle, blueprints; Reference repositories |
| README reference projects (gproxy, LLMG, openproxy, rust-llm-proxy) | Findings noted | Auth, Routing, Rate limiting, Cost, Protocol, Admin, Transport, blueprints |
| BricksLLM and Maxim | Findings noted — BricksLLM per-key USD-cap pioneer, dormant since 2025-01-05; Maxim covered through Bifrost (maximhq) | Market map, Cost tracking |
| vLLM production stack | Findings noted | Caching (alpha semantic), Observability, BYOK (priority routing) |
| Cloud provider LLM gateways (Azure APIM, AWS) | Findings noted — Azure APIM: Rate limiting, Caching, MCP, Upstream auth; AWS Bedrock AgentCore: Routing, Rate limiting, Guardrails, Observability, MCP, Protocol, Upstream auth, Transport | respective sections |

### Wave 1 — themes (8)

| Theme | Outcome | Where |
|-------|---------|-------|
| Virtual keys and multi-tenant auth | Findings noted | Auth and keys |
| Usage metering and cost tracking | Findings noted | Cost tracking; Observability (payload, metadata) |
| Routing resilience and load balancing | Findings noted | Routing and resilience |
| Caching | Findings noted | Caching |
| Guardrails and content safety | Findings noted | Guardrails |
| Observability and tracing | Findings noted | Observability |
| MCP gatewaying | Findings noted | MCP |
| Prompt management and smart routing | Findings noted | Admin (prompt management); Routing (complexity router, CEL rules) |

### Wave 1 — deep dives (6)

| Deep dive | Outcome | Where |
|-----------|---------|-------|
| Request-path retries/fallback/cooldowns | Findings noted | Routing; blueprint 1 |
| Client-key budgets and virtual-key lifecycle | Findings noted (budgets); lifecycle deferred with reason | Cost tracking; blueprint 2; gaps 2 and 9 |
| TPM and per-key/concurrency rate limits | Findings noted | Rate limiting; blueprint 3 |
| Exact-match response caching | Findings noted | Caching; blueprint 4 |
| Upstream credential pools | Findings noted | Routing; blueprint 5; gap 5 |
| Prometheus /metrics and OTel GenAI export | Findings noted (Prometheus carried; OTel deferred) | Observability; blueprint 6; gaps 11–12 |

### Wave 2 — scouts (10)

| Scout | Outcome | Where |
|-------|---------|-------|
| Bifrost (Maxim) | Findings noted | Auth, Routing, Rate limiting, Cost, Caching, Guardrails, Observability, MCP, Protocol, Prompt caching, Admin |
| agentgateway | Findings noted | Auth, Routing, Rate limiting, Cost, Observability, MCP, Protocol, Upstream auth, Admin; cache and prompt-cache absences |
| AWS Bedrock AgentCore Gateway | Findings noted | Routing, Rate limiting, Guardrails, Observability, MCP, Protocol, Upstream auth, Transport |
| IBM ContextForge | Findings noted | Auth, Observability, MCP, Admin; gap 15 |
| Tyk AI Gateway | Findings noted | Auth, Routing, Rate limiting, Cost, Caching, Guardrails, Observability, MCP, Protocol, Admin; gap 15 |
| F5 AI Gateway | Findings noted | Market map status (transition); Routing, Guardrails, Protocol |
| uni-api and GPT-Load | Findings noted | Market map; Auth, Routing and resilience, Rate limiting, Cost, Caching (absence), Guardrails, Observability, Protocol, Prompt caching, Admin; gap 5 |
| Martian and Not Diamond | Findings noted | Market map status (Martian live 2026-10-27); Protocol |
| OpenPipe and Keywords AI (now Respan) | Findings noted — OpenPipe: sunset status only; Respan: detailed | Market map status (OpenPipe); Auth, Routing, Rate limiting, Cost, Caching, Guardrails, MCP, Protocol, Admin (Respan) |
| Upstash AI Gateway | Negative finding — verified not to exist (11 live checks incl. sitemap, llms.txt, GitHub, npm, DNS) | Market map |

### Wave 2 — doc sweeps (10)

| Sweep | Outcome | Where |
|-------|---------|-------|
| LiteLLM | Findings noted (folded into `new:` entries) | mid-stream fallback, Presidio masking, tag budgets, rust_gateway beta |
| Portkey | Findings noted | key rotation, circuit breaker, secret references/KMS, usage policies |
| Kong AI Gateway | Findings noted | Vault entity, changelog 2.1.0/2.2.0 |
| Cloudflare AI Gateway | Findings noted | REST API surfaces, Logpush encryption, spend-limit fallback |
| one-api and new-api | Findings noted | rc.45 screening default, `/v1/tools/*` |
| Higress | Findings noted | semantic cache mode, console |
| OpenRouter | Findings noted | limits-API 402 taxonomy, streaming accounting, MCP server, prompt caching |
| vLLM production stack | Findings noted | alpha semantic cache, priority routing |
| Vercel AI Gateway | Findings noted | nine modalities, OIDC tokens, BYOK, budgets |
| TensorZero | Findings noted (pre-archival docs) | scoped rate-limit rules, benchmarks |

### Wave 2 — frontiers (10)

| Frontier | Outcome | Where |
|----------|---------|-------|
| Protocol surface beyond chat | Findings noted | Protocol breadth (`new:` entries) |
| Upstream auth complexity | Findings noted | Upstream cloud authentication |
| Config and operations | Findings noted | Admin and tooling (config lifecycle) |
| Extensibility and hooks | Findings noted | Admin and tooling (hook systems) |
| Advanced quota shapes | Findings noted | Cost tracking (budget shapes), Rate limiting |
| Model lifecycle management | Findings noted | Model lifecycle metadata |
| Security and compliance | Findings noted | Auth (encryption, secret refs), Observability (Logpush encryption), BYOK (residency), Transport (FIPS) |
| Client experience | Findings noted | BYOK/SDKs; Protocol (`/v1/responses`) |
| Context and prompt aids | Findings noted | Prompt caching and cache locality |
| Streaming SLO features | Findings noted | Observability (streaming SLO mechanics) |

### Wave 2 — deep dives (4)

| Deep dive | Outcome | Where |
|-----------|---------|-------|
| Cloud-native upstream auth (SigV4/STS, Entra OIDC refresh, GCP ADC/workload identity, mTLS) | Findings noted; deferred from candidates with reason | Upstream cloud authentication; blueprint 7 |
| Cross-provider prompt-cache plumbing (marker translation, auto-injection, sticky cache-locality routing) | Findings noted | Prompt caching and cache locality; blueprint 8 |
| Hot-reloadable configuration | Findings noted; deferred from candidates with reason | Admin and tooling; blueprint 9 |
| Model lifecycle metadata in catalogs and /v1/models | Findings noted | Model lifecycle metadata; blueprint 10 |

### Wave 3 — niche-repository hunts (November 2026)

Niche hunts across the Rust/Go/TypeScript/Python long tail —
agent-facing local proxies, semantic and prompt caches, MCP gateways,
guardrail/DLP toolkits, queue dispatchers and load balancers for local
inference, subscription-to-API adapters, research routers,
local-inference front doors, and curated meta-lists — produced a
deduplicated find list of 121 candidate repositories; 23 were researched in
full and are recorded modularly (identity, maintenance, capabilities with
evidence URLs, verified absences, gap-register mapping) in
[`niche-repos.md`](niche-repos.md), which also tables the full 121-find
coverage appendix. All 23 appear in [`repos-registry.json`](repos-registry.json).

| Repo | Outcome | Where |
|------|---------|-------|
| NVIDIA-NeMo/Switchyard | Findings noted | niche-repos.md (retries/cooldowns #1, chain deadline, #6, #13, #11/#12, redaction #7) |
| EeroEternal/xlb | Findings noted; dormant six-day blaze | niche-repos.md (timeout hierarchy, #1, #3, #12/#17, cancel propagation) |
| os-tack/ostk-cache | Findings noted; dormant, private deps | niche-repos.md (#14, payload cost transforms, usage rewriting) |
| raine/claude-code-proxy | Findings noted | niche-repos.md (#1, OAuth credential lifecycle, #13, #6, #19) |
| EricLBuehler/mistral.rs | Findings noted (serving-side cousin) | niche-repos.md (#8 client+server modes, #12/#17, protocol fidelity) |
| smg-project/smg | Findings noted | niche-repos.md (engine-state routing, admission queueing, #1, #2/#3, SSO/RBAC, #8, #11/#12) |
| eliasbiondo/antirion | Findings noted; 18 days old, pre-1.0 | niche-repos.md (#1/#13 hedging, #2/#3, #7, #10, SSO/RBAC, #8, #19) |
| katanemo/plano | Findings noted | niche-repos.md (#1/#13, #3, #7 filter chains, #6, #8, #11/#12) |
| api7/aisix | Findings noted | niche-repos.md (#1/#13, #3, #9, OIDC auth, #7, #10/#14, #8/#A2A, #16) |
| MikkoParkkola/mcp-gateway | Findings noted; PolyForm NC — ideas only | niche-repos.md (#8 tool-poisoning/pinning, #1, #2, #3, #10, #16 8 ms reload) |
| ThinkWatchProject/ThinkWatch | Findings noted; BSL 1.1 | niche-repos.md (SSO/RBAC, #9, #8 per-user OAuth, #7, #2/#3, #1/#13) |
| ghuntley/preflight | Findings noted; dev dormant, CI green | niche-repos.md (#7 Gitleaks DLP + OCR, #12, #16, backpressure) |
| agiwhitelist/tokdiet | Findings noted; dormant PoC | niche-repos.md (compaction, shadow-eval guard, #2, #14 mutation guards) |
| router-for-me/CLIProxyAPI | Findings noted; ToS-gray pattern reference | niche-repos.md (OAuth lifecycle, #5, #1, #14, #6, #19/#16) |
| Wei-Shaw/sub2api | Findings noted; ToS-flagged design reference | niche-repos.md (#5, #14, #1, #2, price catalog, #6) |
| musistudio/claude-code-router | Findings noted | niche-repos.md (#13 rules, #1, #5, #9/#3, #6, #8 ToolHub, #19) |
| mozilla-ai/otari | Findings noted; pre-1.0 | niche-repos.md (#2 holds/settlement, #9, #1/#13, #3, #8, #7, #11/#12) |
| deanxv/done-hub | Findings noted | niche-repos.md (#2 cache-token ledger, OAuth channels, #1, #3, #12, #8, #7) |
| yolorouter | Findings noted | niche-repos.md (#5, #1, #14, compaction, #2/#3/#9, timeout hierarchy, price catalog) |
| NoahStepheno/completion-to-response | Findings noted; archived | niche-repos.md (#6 blueprint: Responses synthesis + event-contract tests) |
| zilliztech/GPTCache | Findings noted; maintenance-mode | niche-repos.md (#10 semantic taxonomy, verification gate, SSE replay) |
| messkan/prompt-cache | Findings noted; quiet | niche-repos.md (#10 two-threshold + gray-zone judge, #1, #12) |
| Chleba/ollamaMQ | Findings noted | niche-repos.md (admission queueing, engine-state routing, #3/#13, #16/#19) |



- Documentation research only: no competitor code was built or run; no
  hands-on benchmarking. llm-proxy's own overhead was not measured, so no
  performance comparison is asserted (TensorZero's 0.94 ms p99 is
  self-published and archived).
- Per the research runs' recorded notCovered list: no pricing, SLA, or
  market-share comparisons; closed betas and undocumented capabilities not
  covered; per-field translation coverage of provider-specific options
  (reasoning controls, tool-choice variants, response formats) was not
  audited for any product.
- Cache coalescing unverified either way beyond Cloudflare, OpenRouter,
  Portkey OSS, Helicone OSS, and LiteLLM; Agent Router's guardrail absence
  is per its docs, not a code audit. Gap-register prevalence counts are
  counts of surveyed products named per item across both waves, not a
  market census.
