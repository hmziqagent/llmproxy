# 02. Request fidelity

- **Status:** Completed
- **Depends on:** 01
- **Audit rows:** 8, 9, 12
- **Code-audit bugs:** BUG-18, BUG-19, BUG-20
- **Backlog:** 10, 13, 16
- **Conflicts:** 10

Real clients must work through the proxy unchanged. Today it drops 11 of 17
tested OpenAI fields with a 200 and rejects common Anthropic fields with a 400.

## To do

1. **Contract tests first.** Capture traffic from Claude Code, Codex, the OpenAI
   SDK and the Anthropic SDK. Remove secrets, store it as fixtures and replay it
   against a fake upstream in CI.
2. **Same-format passthrough (conflict 10).** When client and provider use the same format, keep
   the raw body and apply only the edits the proxy needs, such as the model name.
   Unknown fields survive. Remove `deny_unknown_fields` from the request path in
   `MessageRequest`, `Tool` and `SystemContentBlock`. Read usage from the response
   without rewriting it.
3. **Cross-format field policy.** Keep one table per format pair in which each
   field is carried, mapped or rejected, and reject unknown fields by naming them.
   A per-adapter `lossy` opt-in turns a rejection into a drop with a logged warning.
4. **Missing fields.** Carry `response_format`, `parallel_tool_calls`, `seed`,
   `image_url`, `stop_sequences`, `top_k`, `output_config` where the provider
   supports them. Server tools reach same-format upstreams verbatim and are
   rejected by name cross-format.
5. **Reasoning translation (BUG-20, backlog 13).** Convert between `reasoning_effort`,
   `thinking.budget_tokens` and Gemini's `thinkingBudget` with per-model limits, at
   the provider edge rather than in `llm-proxy-protocol`. Stop forwarding these
   fields unfiltered.
6. **Tool-state replay (backlog 16).** Preserve Gemini thought signatures and
   OpenAI `reasoning_content` across turns.
7. **Header allowlist (row 12, backlog 10).** Add a per-provider list of inbound
   headers to forward, with `anthropic-beta` and session headers on by default for
   Anthropic. Client auth, `Origin` and browser headers are never forwarded.
   Client-supplied request ids are never adopted; logs use the proxy's own
   ids. `ProxyRequest` lives in `crates/llm-proxy-adapters/src/request.rs`
   and the upstream call in `crates/llm-proxy-transport/src/client.rs`; the
   former has no inbound headers today.

## Done

- **Contract fixtures and replay (step 1).**
  `crates/llm-proxy-server/tests/fixtures/contract/` holds two fixture families,
  both replayed by `tests/contract_replay.rs` through the real router against a
  loopback fake upstream, asserting the upstream request equals the fixture
  modulo the listed edits:
  - **Live captures** (`fixtures/contract/captured/`, captured 2026-11-05):
    the real clients ran against a loopback capture upstream — Claude Code CLI
    2.1.251 (a full streaming tool session: user → `tool_use` → `tool_result`
    with `cache_control`, `thinking` incl. the auxiliary `display` subfield,
    top-level `context_management`, the five-beta `anthropic-beta` list,
    `x-api-key` auth), `@anthropic-ai/sdk` (JS) 1.5.0 and `openai` (JS) 7.51.0
    (full knob-set conversations, streaming variants). Capture provenance and
    the exact curation steps are in the fixture README. The captures are
    replayed with their real header sets so the forward-allowlist is exercised
    against `x-stainless-*`, `x-app`, `sec-fetch-mode` and the client
    credential — none of which reach the upstream.
  - **Synthesized fixtures** from the official API/SDK documentation (kept for
    field sets the short capture scripts did not send; provenance per fixture
    in the fixture README).
  The Claude Code capture surfaced one real fidelity gap, now fixed: the
  reasoning rewrite dropped auxiliary `thinking` subfields (Claude Code's
  `display`), so `crates/llm-proxy-adapters/src/reasoning.rs` now rewrites
  only the reasoning knobs (`type`, `budget_tokens`) and lets subfields it has
  no opinion about ride along.
  Codex CLI traffic replays through the Responses route from the live
  `captured/codex/` fixtures (`first-request.json`, `session-request.json`)
  and the synthesized `codex/session-request.json`.
- **Same-format passthrough (step 2, conflict 10).** The routes keep the raw
  client body (`ClientWireRequest` in
  `crates/llm-proxy-adapters/src/client_wire.rs`); when the client dialect
  matches the provider protocol the adapters forward it with only surgical
  edits — the model alias everywhere, plus per-adapter edits: Anthropic
  re-derives the reasoning knobs and strips the proxy-emitted
  `tool_use.signature`; Chat removes a foreign `thinking` object, re-derives
  `reasoning_effort` and forces `stream_options.include_usage` on streams;
  Gemini clamps `thinkingConfig.thinkingBudget` to the model's limits; every
  adapter applies operator `prompt_cache` hygiene when configured.
  `deny_unknown_fields` is gone from `MessageRequest`, `Tool` and
  `SystemContentBlock`; unknown fields decode and survive. Image and document
  `source` objects decode as opaque JSON, so `url`/`file` sources and
  `document` blocks pass through too, and block types the core does not model
  (replayed `server_tool_use`/`web_search_tool_result`, MCP blocks) decode as
  `Opaque` blocks and reach the upstream verbatim. Non-streaming
  same-format responses are returned verbatim — usage and the stop reason are
  read for the ledger without rewriting the body.
- **Cross-format field policy (step 3).** One disposition table keyed per
  client format and provider protocol
  (`crates/llm-proxy-adapters/src/client_wire.rs`): each leftover client
  field is **carried**
  verbatim (`service_tier`, `top_k`, `output_config` to Anthropic;
  `service_tier`, `parallel_tool_calls`, `store`, `prompt_cache_key` to
  Responses and Chat), **mapped**
  (`parallel_tool_calls` → `tool_choice.disable_parallel_tool_use` on
  Anthropic; `output_config.effort` → reasoning on Chat; `response_format` →
  Responses `text.format` and Gemini `responseMimeType`/`responseJsonSchema`;
  `seed` → Gemini `generationConfig.seed` from Chat clients only; `top_k` →
  Gemini `generationConfig.topK`; images between neutral, OpenAI and
  Anthropic shapes), or **rejected with a 400 naming every field**
  (`response_format` stays rejected on Anthropic; `seed` on Responses and
  Anthropic; server tools are rejected by name cross-format, dropped under
  lossy, never mis-encoded as functions). The per-adapter `lossy` TOML flag
  turns rejections into drops with logged warnings. `tool_choice` maps onto
  Gemini `toolConfig.functionCallingConfig` (auto/any/none and named tools;
  emitted only with tools present). `Opaque` blocks, document blocks and
  images without a `base64`/`url` source are handled the same way. A failed
  tool result (`tool_result.is_error`) crosses formats marked as failed: its
  text is prefixed `[tool error] ` for Chat and Responses upstreams and sent
  under `functionResponse.response.error` for Gemini.
- **Missing fields (step 4).** `stop_sequences`/`top_k` decode into core
  sampling and re-encode on Anthropic and Gemini (`topK`); `image_url` parts
  decode into a neutral image source that maps onto Anthropic `base64`/`url`
  image blocks, Gemini `inlineData`/`fileData` and back to OpenAI parts; the
  OpenAI knobs (`response_format`, `seed`, `n`, `logprobs`, `logit_bias`,
  penalties, `service_tier`, `prompt_cache_key`, `store`,
  `parallel_tool_calls`) reach same-format upstreams verbatim via passthrough
  (audit row 8's eleven dropped fields: none dropped any more).
  Cross-format, `response_format` maps onto Responses `text.format` and Gemini
  `generationConfig.responseMimeType` (+ `responseJsonSchema` for a schema);
  unknown types are rejected naming `response_format.type`, and Anthropic
  keeps rejecting the field by name. Cross-format, `seed` maps onto Gemini
  `generationConfig.seed` from Chat-dialect clients only (a non-integer seed
  is rejected naming the field, dropped under lossy); other client dialects
  and non-Gemini targets keep rejecting it by name. Cross-format
  `stop_sequences` have no Responses equivalent: strict routes reject naming
  the field, `lossy` routes drop with a warning. Non-text blocks inside a
  `tool_result` are rejected by name the same way (dropped under lossy).
  `metadata.user_id` carries onto the Responses `user` field; where a drop is
  only attribution or a cache hint (Gemini `user_id`, non-Anthropic
  cache-control markers) it is an exempt debug-level log.
- **Reasoning translation (step 5, BUG-20, backlog 13).**
  `crates/llm-proxy-adapters/src/reasoning.rs` converts between
  `reasoning_effort`, `thinking.budget_tokens` and Gemini's
  `thinkingConfig.thinkingBudget` with a per-model limit table and clamps
  (including Anthropic's budget < max_tokens rule), entirely at the provider
  edge. `reasoning_effort` never reaches Anthropic, `thinking` never reaches a
  chat API, unknown effort values are 400s naming the value, a requested
  thinking budget that cannot fit under `max_tokens` is a 400 naming the
  limit (dropped with a warning under `lossy`), and same-format passthrough
  re-derives the fields through the same filter on Anthropic, Chat and Gemini;
  Responses same-format forwards the client's `reasoning` object untouched,
  because its effort vocabulary is the provider's own.
- **Tool-state replay (step 6, backlog 16).** `CoreContent::ToolUse` carries an
  opaque signature: Gemini `thoughtSignature` values are captured on
  `functionCall` parts, streaming included, and re-emitted on both the
  replayed `functionCall` part and the matching `functionResponse` part; they
  ride the Anthropic client
  wire on `tool_use.signature`, for streamed tool calls too
  (`CoreEvent::ToolCallStart.signature` → the streamed `content_block_start`
  `tool_use` block). An explicitly empty `reasoning_content`
  round-trips as an empty thinking block instead of vanishing.
- **Header allowlist (step 7, row 12, backlog 10).** `ProxyRequest` gained
  `forwarded_headers`; the pipeline selects inbound headers through the
  per-provider `forward_inbound_headers` config (defaulting to
  `anthropic-beta` + `x-claude-code-session-id` for Anthropic, nothing
  elsewhere) minus a hard never-forwarded set (client auth, `Origin`,
  browser/hop-by-hop headers, `x-request-id`), enforced again at config
  validation. Client-supplied request ids are never adopted: the proxy keys
  logs, ledger and `x-request-id` responses on its own generated ids.
- CI equivalents of the audit probes (the probes print; these assert):
  `contract_replay.rs` cases — `openai_sdk_fields_all_reach_the_upstream`
  (row 8's eleven fields all reach the upstream),
  `cross_format_rejection_names_the_fields_and_lossy_drops_them` (row 9:
  rejections name every field, `service_tier` carries verbatim),
  `claude_code_session_replays_byte_identical_modulo_model` (row 12:
  `anthropic-beta` and the session header reach the upstream while the client
  credential, `Origin`, `user-agent` and `x-request-id` do not;
  `stop_sequences` and `top_k` survive),
  `web_search_replay_reaches_a_same_format_upstream_unchanged` and
  `cross_format_server_tool_replay_blocks_are_rejected_by_name_never_stripped`
  (server tools); `cross_format_output_config_is_carried_to_anthropic` and
  `message_level_and_tool_result_cache_markers_reach_upstream` (adapter
  tests) cover the remaining cells.

## Done when

- Every captured client conversation replays with the same request reaching the
  upstream, apart from the listed edits.
- A cross-format request either carries each field or fails with an error naming it.
- A request can enable an Anthropic beta feature through the proxy.
