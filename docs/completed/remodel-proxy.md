# Provider-Based Routing Remodel

## Goal

Build provider-based routing as the only routing surface.

Provider choice is explicit in the request path. If a user hits a Fireworks
route, the provider choice is already known; the request model should be sent
to Fireworks as requested, without any global model-to-provider lookup.

The remodel keeps the normalized core protocol and provider adapters, and uses
this target resolution:

```text
URL provider -> provider config -> inbound route kind -> adapter
request model -> upstream model
```

## Implementation Status

Implemented and audited as of June 11, 2026. The provider-scoped routes,
configuration schema, registry resolution, protocol adapters, model discovery,
catalog cache and enforcement, CLI commands, and protocol-aware errors described
below are present in the workspace.

The implementation is verified with the final formatting, workspace test,
Clippy, and removed-symbol scans listed in [Final Verification](#phase-7-final-verification).
The phase sections remain as design and implementation history.

## Design Principles

- Provider selection is static and explicit in the URL.
- Model selection comes from the client request body by default.
- Config describes providers and protocol endpoints, not every model a provider
  might serve.
- Model catalogs are discovery metadata, not routing rules.
- Existing provider adapters keep receiving `requested_model` and
  `upstream_model`; adapters continue to own request body encoding and URL
  template expansion.
- Transport remains protocol-neutral.
- This is greenfield. Every API request names a provider in the path.

## Locked Decisions

These decisions are part of the implementation plan. Decisions 1 and 2 are
amended by `docs/goal.md`, which adds pools and an opt-in bare `/v1/*`.

1. Provider selection is URL-only. Do not add `X-LLM-Provider`.
2. Bare `/v1/*` API routes are not registered. Provider choice must be present
   in the path.
3. `GET /providers/{provider}/v1/models` is the canonical model-list route.
4. Discovery does not run automatically during server startup.
5. Normal `/models` reads use static plus cached catalog data. A
   `refresh=live` query refreshes only when the cache TTL has expired, while
   CLI `--live` always performs an upstream fetch.
6. Cross-protocol mappings such as `messages = "chat"` are allowed because the
   normalized core protocol exists specifically to translate between client and
   provider protocols. The operator must declare the mapping explicitly.
7. Provider-local aliases are supported, but global model-to-provider aliases
   are removed.
8. Catalog enforcement is disabled by default.

## Target Routes

Canonical provider routes:

```text
POST /providers/{provider}/v1/chat/completions
POST /providers/{provider}/v1/messages
POST /providers/{provider}/v1/messages/count_tokens
GET  /providers/{provider}/v1/models
```

Bare `/v1/*` routes are intentionally not part of the API.

Example OpenAI SDK base URL:

```text
http://127.0.0.1:3456/providers/fireworks/v1
```

Example Anthropic-style base URL:

```text
http://127.0.0.1:3456/providers/fireworks
```

## Model Source Rules

For OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Fireworks
OpenAI-compatible APIs, and similar providers, the model is in the JSON request
body:

```json
{
  "model": "accounts/fireworks/models/deepseek-v3p1",
  "messages": []
}
```

For Gemini GenerateContent, the selected model still comes from the decoded
`CoreRequest.model.requested`, but the Gemini adapter expands it into the
configured endpoint template:

```toml
endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
```

Header or query-string model selection is not needed for the first
implementation. If a provider later requires model selection outside the body,
add it as adapter-specific config, not global routing logic.

## New Config Shape

Main config keeps server settings only:

```toml
[server]
bind = "127.0.0.1:3456"
request_timeout = "300s"
log_level = "info"
hot_reload = false
rate_limit_rpm = 100
trust_forwarded_headers = false
server_name = "llm-proxy"
```

`hot_reload = true` is rejected because hot reload is not implemented.
`rate_limit_rpm = 0` disables rate limiting; direct loopback clients are
exempt. Forwarding headers are ignored unless `trust_forwarded_headers = true`.
Identical requests are never rejected; see plan 06 for single-flight, `Idempotency-Key` and the opt-in response cache.

Provider config owns provider details, adapters, inbound route mapping, and
optional discovery:

```toml
[provider]
name = "fireworks"
api_key = "${FIREWORKS_API_KEY}"
auth_style = "bearer"

[provider.adapters.chat]
protocol = "openai_chat_completions"
endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"

[provider.adapters.responses]
protocol = "openai_responses"
endpoint = "https://api.fireworks.ai/inference/v1/responses"

[provider.routes]
chat_completions = "chat"
messages = "chat"

[provider.model_aliases]
# Optional provider-local aliases. These do not choose a provider; they only
# rewrite the model after the provider has already been selected by URL.
kimi = "accounts/fireworks/models/kimi-k2.6"
deepseek = "accounts/fireworks/models/deepseek-v3p1"

[provider.discovery]
kind = "fireworks_account_models"
endpoint = "https://api.fireworks.ai/v1/accounts/fireworks/models"
max_pages = 100
max_models = 20000
max_response_bytes = 4194304

[provider.catalog]
# Optional static model metadata. These entries are useful when a provider does
# not expose a model-list endpoint, when the operator wants stable display
# names/capabilities, or when a provider endpoint is too noisy.
mode = "hybrid" # static | discovered | hybrid
enforce = false
cache_ttl = "24h"
allow = ["*"]
deny = ["*-preview"]

[[provider.catalog.models]]
id = "accounts/fireworks/models/deepseek-v3p1"
display_name = "DeepSeek V3.1"
supports = ["chat_completions"]
context_length = 131072
```

For Anthropic:

```toml
[provider]
name = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"
auth_style = "x-api-key"

[provider.adapters.messages]
protocol = "anthropic_messages"
endpoint = "https://api.anthropic.com/v1/messages"
headers = { "anthropic-version" = "2023-06-01" }

[provider.routes]
messages = "messages"
chat_completions = "messages"

[provider.discovery]
kind = "anthropic_models"
endpoint = "https://api.anthropic.com/v1/models"
```

The Anthropic adapter and discovery client both default `anthropic-version` to
`2023-06-01`; configured adapter or discovery headers override the default.

For Gemini:

```toml
[provider]
name = "gemini"
api_key = "${GEMINI_API_KEY}"
auth_style = "x-google-api-key"

[provider.adapters.generate_content]
protocol = "gemini_generate_content"
endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

[provider.routes]
chat_completions = "generate_content"
messages = "generate_content"

[provider.discovery]
kind = "gemini_models"
endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
```

## Core Type Changes

Do not include any global model-routing config in the new schema. The core
schema should contain provider settings, provider route mappings, aliases, and
catalog metadata only.

Add:

```rust
pub enum ProviderRouteKind {
    ChatCompletions,
    Messages,
}

pub struct ProviderRoutesConfig {
    pub chat_completions: Option<String>,
    pub messages: Option<String>,
}

pub struct ProviderDiscoveryConfig {
    pub kind: ProviderDiscoveryKind,
    pub endpoint: String,
    pub headers: HashMap<String, String>,
    pub max_pages: usize,
    pub max_models: usize,
    pub max_response_bytes: usize,
}

pub struct ProviderCatalogConfig {
    pub mode: ProviderCatalogMode,
    pub enforce: bool,
    pub cache_ttl: Duration,
    pub allow: Vec<String>,
    pub deny: Vec<String>,
    pub models: Vec<StaticModelCatalogEntry>,
}

pub enum ProviderCatalogMode {
    Static,
    Discovered,
    Hybrid,
}

pub struct StaticModelCatalogEntry {
    pub id: String,
    pub display_name: Option<String>,
    pub supports: Vec<ProviderRouteKind>,
    pub context_length: Option<u32>,
}

pub enum ProviderDiscoveryKind {
    OpenAiCompatibleModels,
    OpenAiModels,
    AnthropicModels,
    GeminiModels,
    FireworksAccountModels,
}
```

`ProviderAdapterConfig` also needs optional static request headers:

```rust
pub struct ProviderAdapterConfig {
    pub protocol: String,
    pub endpoint: String,
    pub headers: HashMap<String, String>,
}
```

This is required for provider APIs whose protocol needs non-auth headers such
as `anthropic-version`. Header names and values must be validated against CRLF
injection, and forbidden transport headers such as `Host`, `Content-Length`,
and `Transfer-Encoding` must be rejected.

Auth configuration covers the real provider requirements used by both
inference and discovery. TOML values are `bearer`, `x-api-key`,
`x-google-api-key`, and `both`. The Google style sends the actual
`x-goog-api-key` header; `both` sends Bearer and `x-api-key`. Secret values are
redacted from Debug/logs.

The provider route kind describes the inbound client endpoint, not the upstream
protocol. For example, either of these is valid:

```toml
[provider.routes]
chat_completions = "responses"
messages = "responses"
```

That maps OpenAI Chat or Anthropic Messages input through the normalized core
protocol to an OpenAI Responses upstream adapter. There is no inbound
`/v1/responses` client endpoint in the current scope.

Change `ProviderConfig` to include:

```rust
pub routes: ProviderRoutesConfig,
pub model_aliases: HashMap<String, String>,
pub discovery: Option<ProviderDiscoveryConfig>,
pub catalog: Option<ProviderCatalogConfig>,
```

`ServerConfig` contains server operational settings only.

## Registry Changes

Registry resolution:

```text
provider_name
provider.routes[route_kind]
provider.adapters[adapter_name]
provider.model_aliases[requested_model] -> upstream_model, if present
```

New API:

```rust
pub fn resolve_provider_route(
    &self,
    provider_name: &str,
    route_kind: ProviderRouteKind,
    requested_model: &str,
) -> Result<ProviderAdapterTargetConfig, CoreError>
```

The returned target keeps:

```rust
requested_model = requested_model
upstream_model = provider.model_aliases[requested_model].unwrap_or(requested_model)
```

This preserves current provider adapter behavior:

- OpenAI Chat and Responses put `target.upstream_model` into the request body.
- Anthropic puts `target.upstream_model` into the request body.
- Gemini expands `{model}` from `target.upstream_model`.
- Client response encoders preserve `target.requested_model`.
- Provider-local aliases remain local to the selected provider and do not
  reintroduce global model routing.

## Server Pipeline Changes

`core_pipeline::resolve_target` should resolve by provider route:

```rust
fn resolve_target(
    state: &AppState,
    provider_name: &str,
    route_kind: ProviderRouteKind,
    core: &CoreRequest,
) -> Result<(ProviderAdapterTarget, ProviderAdapter), RouteError>
```

Handlers extract `Path(provider)` for canonical routes and pass it into the
pipeline.

Add explicit route errors instead of mapping provider resolution failures to a
generic 500:

```text
UnknownProvider       -> 404
UnsupportedRoute      -> 404 or 400 with actionable message
ModelNotAllowed       -> 400
InvalidProviderName   -> 400
```

The error envelope remains selected by inbound client protocol. Internal
provider config errors remain generic 500 responses and must not expose API
keys, endpoints containing secrets, or discovery response bodies.

`/providers/{provider}/v1/messages/count_tokens` is a local token-estimation
route in the initial remodel. It should not resolve an upstream adapter unless
a future native provider token-count route is explicitly added.
Provider-scoped count_tokens is useful for SDK/base-URL consistency and future
provider-specific tokenizers, but Phase 1 should keep current heuristic
counting behavior.

Provider names used in paths should be validated as stable URL-safe slugs,
for example ASCII lowercase letters, digits, hyphens, and underscores. Provider
TOML names and route path names must use the same canonical form; do not
silently lowercase or normalize mismatches.

The 404 classifier should classify `/providers/{provider}/v1/chat/*` as
OpenAI-shaped and `/providers/{provider}/v1/messages*` as Anthropic-shaped.
`/v1/models` has no single client protocol (the listing body is a
superset), so its layer errors are OpenAI-shaped to match the handler.

## CLI Changes

`llm-proxy init`:

- Generate `config.toml` with server settings only.
- Generate provider files with `[provider.routes]`.
- Include commented discovery blocks.

`llm-proxy validate`:

- Load main config and provider files.
- Validate every route adapter exists.
- Validate every adapter protocol is registered.
- Print provider route table:

```text
=== Provider Routes ===
fireworks:
  chat_completions -> chat/openai_chat_completions
  messages         -> chat/openai_chat_completions
anthropic:
  messages         -> messages/anthropic_messages
```

`llm-proxy models`:

- New default behavior should list cached catalogs if present.
- `--provider <name> --live` should fetch the provider model list using
  `[provider.discovery]`, regardless of cache freshness.
- `--write-catalog` should persist the discovered catalog to a separate catalog
  cache file, not rewrite the provider TOML that may contain secret references.

Suggested commands:

```text
llm-proxy models --provider fireworks --live
llm-proxy models --provider fireworks --live --write-catalog
llm-proxy models --provider fireworks
```

Catalog cache path:

```text
providers/.catalog/{provider}.toml
```

The cache directory should use the same symlink/regular-file safety posture as
provider config loading. Catalog files contain no API keys, but they can reveal
which providers/models the operator uses.

Catalog writes must use a temporary file plus atomic rename. A failed refresh
must never truncate the last known-good cache.

Catalog state should live in a dedicated `ModelCatalogService` in `AppState`,
not in `ProviderRegistry`. The provider registry remains immutable routing
configuration; the catalog service owns mutable cached discovery data,
timestamps, refresh state, and filtering.

Use single-flight refreshes per provider so concurrent `/models` refresh
requests do not create an upstream request stampede. Normal completion requests
must never acquire catalog refresh locks when enforcement is disabled.

## Model Discovery

Discovery should feed a provider model catalog. It helps users inspect models
and lets `GET /providers/{provider}/v1/models` return useful data, but it
should not block arbitrary model IDs unless `provider.catalog.enforce = true`.
If `[provider.catalog]` is omitted, default to advisory hybrid behavior:
`mode = "hybrid"`, `enforce = false`, `allow = ["*"]`, `deny = []`, and a
reasonable `cache_ttl`.

This is the "best of both worlds" policy:

- Static config is always allowed.
- Live discovery is used only when configured and supported by the provider.
- Hybrid catalogs merge static entries and discovered entries.
- Request routing does not depend on live discovery.
- Operators can allow/deny models with glob patterns.
- Enforcement is opt-in because many providers accept model IDs before their
  model-list endpoint catches up.

Supported discovery backends:

```text
openai_compatible_models
openai_models
anthropic_models
gemini_models
fireworks_account_models
```

`openai_compatible_models` is the default reusable backend for providers whose
model endpoint returns OpenAI-style `{ "data": [{ "id": ... }] }`. Do not add
a new hardcoded discovery enum variant for every OpenAI-compatible provider.
Provider-specific discovery kinds are only justified when authentication,
pagination, or response shape differs materially.

Catalog modes:

```text
static      Only use [[provider.catalog.models]].
discovered  Only use live/cached discovery results.
hybrid      Merge static and discovered models; static metadata wins on ID conflicts.
```

Runtime behavior:

```text
/providers/{provider}/... request:
  provider comes from URL
  model comes from request body/path
  provider-local alias maps requested_model -> upstream_model, if configured
  if catalog.enforce = false:
    do not check catalog; send model upstream
  if catalog.enforce = true:
    require upstream_model to match merged catalog after allow/deny filtering

/providers/{provider}/v1/models:
  return merged catalog if available
  refresh from discovery when cache is stale and live refresh is requested
  otherwise return cached/static data
```

Model filter rules:

- `allow` and `deny` support glob patterns.
- `allow` wins over `deny`.
- With no `allow`, all models are allowed except denied models.
- With `allow = ["*"]`, all models are allowed except denied models.
- With a narrow allow list, only matching models are shown and, if enforcement
  is enabled, accepted.

Provider behavior from official docs:

- OpenAI exposes `GET /v1/models`.
- Anthropic exposes `GET /v1/models`.
- Gemini exposes model listing through the Models API and uses
  `models/{model}:generateContent` for GenerateContent.
- Fireworks exposes OpenAI-compatible inference and model listing APIs.

Discovery auth/HTTP behavior:

- Discovery uses a separate GET-capable client, not the protocol-neutral POST
  transport used for completions.
- Bearer-style providers use `Authorization: Bearer <key>`.
- Anthropic discovery sends `x-api-key` and defaults `anthropic-version` to
  `2023-06-01`; an explicit discovery header overrides that default.
- Gemini discovery must support the API-key style required by the configured
  endpoint.
- Discovery failures should not stop the server from starting unless the
  command explicitly requested `--live --require-success`.
- Upstream discovery response bodies must be sanitized before logging, same as
  completion errors.
- Discovery page count, model count, and per-page response size use configurable
  nonzero limits with defaults of 100 pages, 20,000 models, and 4 MiB.
- Exceeding any configured discovery limit fails the refresh rather than
  returning a truncated catalog. A last known-good cache remains available as
  stale fallback unless the caller requires success.
- Follow provider pagination with a configured maximum page count and maximum
  model count.
- De-duplicate model IDs deterministically and produce stable sorted output.
- Set independent connect/request timeouts for discovery; do not reuse the
  potentially long completion timeout.
- Reject redirects to non-HTTP(S) destinations and rely on the configured
  endpoint trust boundary rather than accepting model-provided URLs.

Discovery parser behavior:

- OpenAI-compatible: consume `data[].id`.
- Anthropic: consume `data[]`, following `has_more`/cursor fields.
- Gemini: consume `models[]`, following `nextPageToken`.
- Fireworks: parse its account model response and pagination separately.
- Ignore malformed individual records with a warning; fail the refresh if the
  top-level response shape is invalid.

Catalog file shape:

```toml
[catalog]
provider = "fireworks"
source = "live"
generated_at = "2026-06-08T00:00:00Z"

[[catalog.models]]
id = "accounts/fireworks/models/deepseek-v3p1"
display_name = "DeepSeek V3.1"
supports = ["chat_completions"]
context_length = 131072
```

Catalog normalization rules:

- Preserve the exact upstream model ID in `id`.
- Add display fields when the provider returns them.
- Keep unknown provider metadata in a raw JSON field only if needed later.
- For Gemini, strip a leading `models/` prefix only for display; preserve the
  exact value required by the adapter path.
- Gemini catalog enforcement treats `gemini-x` and `models/gemini-x` as the
  same model ID, matching endpoint-template expansion behavior.
- Filter Gemini models to ones whose `supportedActions` include
  `generateContent` for generation routes.
- Do not require provider prefixes inside provider-scoped routes. A request to
  `/providers/fireworks/...` can use `model = "accounts/fireworks/models/foo"`.
- Static model entries and provider-local aliases must be validated for empty
  IDs and unsafe Gemini URL-template characters when they are used as
  `upstream_model`.
- When static and discovered entries share the same ID, static metadata wins
  because it is operator-authored.

Models endpoint response:

- Return a normalized superset model card containing OpenAI fields (`id`,
  `object`, `created`, `owned_by`) and Anthropic fields (`type`,
  `display_name`, `created_at`) so common SDKs can ignore fields they do not
  use.
- Keep output deterministic. The endpoint currently returns the complete
  merged catalog in one response rather than implementing proxy-side paging.
- Include Anthropic pagination fields (`first_id`, `last_id`, `has_more`) while
  retaining OpenAI-compatible root fields; `has_more` is currently always
  `false` because the full merged catalog is returned.
- Never return provider credentials, raw discovery headers, or unfiltered raw
  provider metadata.
- Model IDs are provider-local because `/providers/{provider}/v1/models` has no
  cross-provider ID collision problem.

## Reference Project Findings

LiteLLM:

- Uses static `model_list` entries as the main routing surface.
- Supports wildcard model routes like `anthropic/*`.
- `/v1/models` lists models available through config and access rules.
- It does not rely on live upstream discovery for normal routing.

llm-api-key-proxy:

- Implements real provider discovery through provider plugins with
  `get_models`.
- OpenAI, Groq, Mistral, OpenRouter, Gemini, and OpenAI-compatible providers
  fetch upstream model-list endpoints.
- Dynamic OpenAI-compatible providers can be discovered from environment
  variables like `<NAME>_API_BASE`.
- It applies allow/deny filtering and caches fetched model lists.
- `/v1/models` returns OpenAI-style model cards built from available models.

oc-go-cc:

- Uses static model defaults and request-shape rules.
- Has `respect_requested_model`, which can pass through a requested model, but
  provider choice still comes from config defaults rather than URL route.
- It has no real model discovery.

Design decision from refs:

- Keep LiteLLM's static/wildcard configurability.
- Keep llm-api-key-proxy's optional live model discovery and allow/deny filters.
- Do not adopt oc-go-cc's request-shape model router.
- Avoid making discovery a request-time routing dependency.

## Module Ownership

`llm-proxy-core`:

- Provider, route, alias, discovery, and catalog config types.
- Pure catalog merge and allow/deny filtering.
- Catalog entry types and validation.
- No HTTP calls.

`llm-proxy-provider`:

- GET-capable discovery HTTP client.
- Discovery protocol parsers and pagination.
- Authentication/header construction.
- Response sanitization.

`llm-proxy-server`:

- Provider-scoped Axum routes.
- `ModelCatalogService` runtime cache and single-flight refresh coordination.
- Model-list response encoding.
- Catalog enforcement before upstream dispatch.

`apps/llm-proxy`:

- CLI orchestration for init, validate, models, and catalog refresh/write.
- No duplicate provider discovery parsing.

## Current Code Impact Map

The remodel touches every place that currently treats model names as the
provider selection key.

`crates/llm-proxy-core/src/model_route.rs`:

- Delete this module, or replace it with provider-route resolution only if the
  name is changed.
- Remove `ProviderTarget`, `ModelRouteError`, and `resolve_model_route`.

`crates/llm-proxy-core/src/provider_config.rs`:

- Remove `AppConfig.models`, `ModelRoute`, `ProviderModelConfig`, and
  provider-local model tables.
- Add `[provider.routes]`, `[provider.model_aliases]`, `[provider.discovery]`,
  and `[provider.catalog]`.
- Add adapter static headers and new auth styles.
- Keep `deny_unknown_fields`; removed model-routing fields should fail as
  unknown configuration.

`crates/llm-proxy-core/src/provider_registry.rs`:

- Replace `resolve_adapter_target(ProviderTarget)` with
  `resolve_provider_route(provider, route_kind, requested_model)`.
- Route adapter selection comes from `[provider.routes]`.
- Model aliasing happens after provider selection.

`crates/llm-proxy-core/src/error.rs` and server route errors:

- Replace `UnknownModel` routing semantics with provider-route errors:
  unknown provider, unsupported route, invalid provider name, and optional
  model-not-allowed.

`crates/llm-proxy-server/src/routes/mod.rs`:

- Add provider-prefixed Axum routes.
- Update 404 classification for provider-prefixed OpenAI and Anthropic
  paths.

`crates/llm-proxy-server/src/routes/chat.rs` and `messages.rs`:

- Accept the provider name from the canonical route path.
- Pass `ProviderRouteKind` into the core pipeline.
- Use the provider-prefixed request path in request preparation and tracing.

`crates/llm-proxy-server/src/routes/core_pipeline.rs`:

- Remove `resolve_model_route`.
- Resolve through `ProviderRegistry::resolve_provider_route`.
- Preserve streaming first-byte behavior and protocol-aware error envelopes.
- Add provider/upstream-model tracing fields.

`crates/llm-proxy-server/src/routes/token_count.rs`:

- Add provider-prefixed route support without forcing upstream adapter
  resolution.

`crates/llm-proxy-server/src/state.rs`:

- Add `ModelCatalogService` to runtime state once catalog endpoints are
  implemented.
- Keep `ProviderRegistry` immutable.

`crates/llm-proxy-provider/src/transport.rs`:

- Apply validated adapter static headers in addition to auth headers for
  streaming and non-streaming requests.
- Keep auth/debug redaction guarantees.

`crates/llm-proxy-core/src/metrics.rs` and
`crates/llm-proxy-server/src/routes/health.rs`:

- Change model-only counters to provider plus model dimensions.
- Keep unauthenticated health output aggregate-only; do not expose
  provider/model counters.

`apps/llm-proxy/src/main.rs`:

- Rewrite default TOML constants.
- Update `init`, `validate`, and `models`.
- Convert live model discovery command paths to async orchestration.
- Remove tests that expect global model routes to resolve.

Examples and docs:

- Rewrite `config.toml.example`.
- Rewrite provider examples under `providers/`.
- Update architecture docs and source guard tests for provider routing.

## Implementation Phases

### Phase 1: Config Types

- Add `ProviderRoutesConfig`, `ProviderRouteKind`, provider-local aliases,
  catalog config, and discovery config types.
- Add serde defaults so `[provider.catalog]`, `[provider.discovery]`, and
  `[provider.model_aliases]` can be omitted.
- Add optional adapter headers and the required auth styles.
- Update TOML parsing and validation.
- Update config tests.

### Phase 2: Provider Registry

- Replace `resolve_adapter_target` with `resolve_provider_route`.
- Keep `ProviderAdapterTargetConfig` fields unchanged.
- Add errors for unknown provider, unsupported route kind, and route adapter
  pointing to an unknown adapter.
- Apply provider-local model aliases during target construction.
- Update registry tests.

### Phase 3: Server Routes

- Add canonical `/providers/{provider}/v1/...` routes.
- Thread `provider_name` and `ProviderRouteKind` through handlers into
  `core_pipeline`.
- Preserve current non-streaming and streaming first-byte behavior.
- Keep count_tokens local and avoid requiring a provider adapter for it in the
  first implementation.
- Add provider-specific route errors and update protocol-aware 404
  classification.
- Record provider and upstream model in structured tracing fields.

### Phase 4: CLI and Examples

- Rewrite default config constants.
- Rewrite `init`, `validate`, and `models`.
- Update `config.toml.example` and provider examples.
- Update CLI tests for provider route generation and validation.
- Make the `models` command async so it can reuse the provider discovery client.

### Phase 5: Model Catalogs

- Add a discovery adapter registry keyed by discovery response protocol.
- Add GET-capable discovery HTTP logic with auth, pagination, limits, and
  independent timeouts.
- Add catalog read/write helpers.
- Add `ModelCatalogService` with per-provider single-flight refreshes.
- Implement `GET /providers/{provider}/v1/models`.
- Implement `llm-proxy models --provider ... --live`.
- Implement static, discovered, and hybrid catalog modes.
- Add allow/deny glob filtering.
- Add opt-in catalog enforcement with `provider.catalog.enforce`.
- Keep enforcement disabled by default.

### Phase 6: Cleanup

- Delete `model_route.rs` or repurpose it only if the name still makes sense.
- Update `llm-proxy-core/src/lib.rs` exports and docs.
- Update source guard tests to fail on `resolve_model_route` usage.
- Update architecture docs from "model routing" to "provider routing".
- Update metrics labels from model-only to provider plus model so identical
  model IDs served by different providers do not collapse into one counter.
- Remove provider/model counters from unauthenticated `/health`; retain only
  aggregate operational counters.

### Phase 7: Final Verification

- Run formatting, full workspace tests, Clippy with warnings denied, and source
  guard scans.

## Test Plan

Core config tests:

- App config parses with server settings only.
- Unknown top-level routing tables fail because `deny_unknown_fields` protects
  the schema.
- Provider config parses with `[provider.routes]`.
- Route adapter references must point to existing adapters.
- Empty provider route adapter names fail.
- Unknown protocols fail.
- Provider config parses when discovery, catalog, and model aliases are omitted.
- Empty alias names or empty alias targets fail.
- Catalog entries with empty IDs fail.
- Provider names reject path separators, dots, control characters, and
  whitespace.
- Adapter headers reject CRLF and forbidden transport headers.
- Auth styles parse and serialize correctly.

Registry tests:

- Provider route resolves to the correct adapter and protocol.
- Unknown provider fails.
- Unsupported route for a provider fails.
- Resolved target uses request model as both requested and upstream.
- Provider-local alias preserves requested model and rewrites upstream model.
- API keys stay redacted in Debug.

Server tests:

- `/providers/fireworks/v1/chat/completions` routes to Fireworks.
- Same request model can be sent to different providers by changing only the
  URL.
- Bare `/v1/chat/completions` is not registered and returns 404.
- Provider-scoped count_tokens works without an upstream adapter.
- Streaming first-byte behavior remains unchanged.
- Unknown provider and unsupported provider route return client errors, not
  generic 500 responses.
- Provider-prefixed unknown chat paths return OpenAI-shaped 404 responses.
- Provider-prefixed unknown messages paths return Anthropic-shaped 404
  responses.
- Auth rejection, wrong method, and unknown provider on `/v1/models` all
  answer the same top-level OpenAI-shaped envelope.

CLI tests:

- Generated config has server settings only.
- Generated provider files have `[provider.routes]`.
- `validate` prints provider routes.
- `models --provider X --live` handles provider discovery responses.
- Missing discovery config gives a clear error.
- `models --provider X` reads static and cached catalogs without network.
- Discovery failure does not start the server failure path.

Catalog tests:

- Static mode returns only static models.
- Discovered mode returns only cached/live discovered models.
- Hybrid mode merges static and discovered models with static metadata winning.
- Allow/deny filtering applies consistently to listings and enforcement.
- Enforcement disabled allows unknown model IDs through.
- Enforcement enabled rejects models absent from the filtered catalog.
- Catalog cache loader rejects symlinks and non-TOML files.
- Concurrent refresh requests perform one upstream discovery call.
- Pagination stops at configured page/model limits.
- Duplicate IDs are removed with deterministic output ordering.
- Stale cache remains usable when a live refresh fails.
- Model endpoint output contains no credentials or raw sensitive metadata.

Transport tests:

- Adapter static headers are sent for streaming and non-streaming requests.
- `anthropic-version` is present for Anthropic requests.
- Gemini authentication uses the configured supported API-key mechanism.
- Debug output redacts API keys and any configured sensitive header values.

Final verification commands:

```text
cargo fmt --all -- --check
cargo test --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
rg "resolve_model_route|ModelRoute|ProviderModelConfig|provider\\.models" crates apps
```

## Deferred Features

These are intentionally outside the initial implementation:

- Header-based provider selection.
- Global model routing or weighted provider selection.
- Automatic discovery during server startup.
- Background scheduled catalog refresh.
- Native upstream token-count APIs.
- Database-backed catalogs.
- Per-client model access control, since the proxy currently has no inbound
  authentication.

## Research Sources

- OpenAI API reference: https://developers.openai.com/api/reference/resources/models/methods/list
- OpenAI Chat Completions API reference: https://developers.openai.com/api/reference/resources/chat
- Anthropic Models API reference: https://anthropic.mintlify.app/en/api/models-list
- Anthropic Messages API reference: https://docs.claude.com/en/api/messages
- Gemini Models API reference: https://ai.google.dev/api/models
- Gemini GenerateContent API reference: https://ai.google.dev/api/generate-content
- Fireworks OpenAI compatibility docs: https://docs.fireworks.ai/tools-sdks/openai-compatibility
- Fireworks list models API reference: https://docs.fireworks.ai/api-reference/list-models
