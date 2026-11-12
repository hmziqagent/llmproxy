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
   A per-route `lossy` opt-in turns a rejection into a drop with a logged warning.
4. **Missing fields.** Carry `response_format`, `parallel_tool_calls`, `seed`,
   `image_url`, `stop_sequences`, `top_k`, `output_config` and server tools wherever
   the provider supports them.
5. **Reasoning translation (BUG-20, backlog 13).** Convert between `reasoning_effort`,
   `thinking.budget_tokens` and Gemini's `thinkingBudget` with per-model limits, at
   the provider edge rather than in `llm-proxy-protocol`. Stop forwarding these
   fields unfiltered.
6. **Tool-state replay (backlog 16).** Preserve Gemini thought signatures and
   OpenAI `reasoning_content` across turns.
7. **Header allowlist (row 12, backlog 10).** Add a per-provider list of inbound
   headers to forward, with `anthropic-beta` and session headers on by default for
   Anthropic. Client auth, `Origin` and browser headers are never forwarded. Clean
   client-supplied request ids before logging them. `ProxyRequest` in
   `crates/llm-proxy-provider/src/transport.rs` has no inbound headers today.

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
  `display`), so `adapter/reasoning.rs` now rewrites only the reasoning knobs
  (`type`, `budget_tokens`) and lets subfields it has no opinion about ride
  along.
  Codex traffic needs the Responses client route (plan 08) and is not
  replayable yet.
- **Same-format passthrough (step 2, conflict 10).** The routes keep the raw
  client body (`ClientWireRequest` in `crates/llm-proxy-provider`); when the
  client dialect matches the provider protocol the adapters forward it with only
  surgical edits: the model alias, reasoning translation, usage-on-streams
  (`stream_options.include_usage`) and stripping foreign block fields.
  `deny_unknown_fields` is gone from `MessageRequest`, `Tool` and
  `SystemContentBlock`; unknown fields decode and survive. Image and document
  `source` objects decode as opaque JSON, so `url`/`file` sources and
  `document` blocks pass through too, and block types the core does not model
  (replayed `server_tool_use`/`web_search_tool_result`, MCP blocks) decode as
  `Opaque` blocks and reach the upstream verbatim. Non-streaming
  same-format responses are returned verbatim — usage and the stop reason are
  read for the ledger without rewriting the body.
- **Cross-format field policy (step 3).** One disposition table per provider
  protocol (`adapter/client_wire.rs`): each leftover client field is **carried**
  verbatim (e.g. `service_tier`, `top_k` to Anthropic; `store`,
  `prompt_cache_key`, `parallel_tool_calls` to Responses and Chat), **mapped**
  (`parallel_tool_calls` → `tool_choice.disable_parallel_tool_use`;
  `output_config.effort` → reasoning; images between neutral, OpenAI and
  Anthropic shapes), or **rejected with a 400 naming every field**. The
  per-adapter `lossy` TOML flag turns rejections into drops with logged
  warnings. Server tools are rejected by name cross-format (dropped under
  lossy), never mis-encoded as functions. `Opaque` blocks, document blocks and
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
  `generationConfig.seed` (a non-integer seed is rejected naming the field,
  dropped under lossy); Responses and Anthropic keep rejecting it by name.
- **Reasoning translation (step 5, BUG-20, backlog 13).**
  `adapter/reasoning.rs` converts between `reasoning_effort`,
  `thinking.budget_tokens` and Gemini's `thinkingConfig.thinkingBudget` with a
  per-model limit table and clamps (including Anthropic's budget <
  max_tokens rule), entirely at the provider edge. `reasoning_effort` never
  reaches Anthropic, `thinking` never reaches a chat API, unknown effort
  values are 400s naming the value, and same-format passthrough re-derives
  the fields through the same filter.
- **Tool-state replay (step 6, backlog 16).** `CoreContent::ToolUse` carries an
  opaque signature: Gemini `thoughtSignature` values are captured on
  `functionCall` parts and re-emitted on both the replayed `functionCall` part
  and the matching `functionResponse` part; they ride the Anthropic client
  wire on `tool_use.signature`. An explicitly empty `reasoning_content`
  round-trips as an empty thinking block instead of vanishing.
- **Header allowlist (step 7, row 12, backlog 10).** `ProxyRequest` gained
  `forwarded_headers`; the pipeline selects inbound headers through the
  per-provider `forward_inbound_headers` config (defaulting to
  `anthropic-beta` + `x-claude-code-session-id` for Anthropic, nothing
  elsewhere) minus a hard never-forwarded set (client auth, `Origin`,
  browser/hop-by-hop headers, `x-request-id`), enforced again at config
  validation. Client-supplied request ids are never adopted: the proxy keys
  logs, ledger and `x-request-id` responses on its own generated ids.
- Verified with the audit probes
  (`docs/research/audit-2026-11-05/evidence/probes/run.sh`): rows 8, 9 and 12
  now behave as specified (eleven-field drop list empty; `stop_sequences`,
  `top_k`, `service_tier`, `output_config`, tool `cache_control` and server
  tools all forwarded with 200; `anthropic-beta` and the session header reach
  the upstream while the client credential, `Origin` and `user-agent` do not).

## Done when

- Every captured client conversation replays with the same request reaching the
  upstream, apart from the listed edits.
- A cross-format request either carries each field or fails with an error naming it.
- A request can enable an Anthropic beta feature through the proxy.
