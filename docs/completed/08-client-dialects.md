# 08. Client dialects

- **Status:** Completed (2026-11-07)
- **Depends on:** 03
- **Audit rows:** None
- **Code-audit bugs:** None
- **Backlog:** 9, 19, 36
- **Conflicts:** None

Accept more client formats. Each one is a codec in `llm-proxy-protocol` plus a route
through the same engine, so it gets limits, routing and accounting without new
pipeline code.

## To do

1. **OpenAI Responses (backlog 9).** Serve `/providers/{provider}/v1/responses` with
   streaming event synthesis, exact usage fields and Codex-compatible tool-call
   ordering. Comes first because Codex-class clients can't be pointed elsewhere.
2. **Gemini native (backlog 19).** Serve `generateContent` and `streamGenerateContent`
   under `/providers/{provider}/v1beta/models/{model}`.
3. **Ollama native (backlog 36).** Serve `/api/chat`, `/api/generate` and `/api/tags`
   as a provider-scoped route, reusing the same admission path.

## Done

- **OpenAI Responses (step 1, backlog 9).**
  `llm-proxy-protocol/src/client/responses.rs` decodes the documented request
  shape (`instructions`, typed `input` items including replayed `reasoning`
  items with `encrypted_content`, function tools (a tool-level `strict: true`
  is carried same-format and rejected by name cross-format),
  `tool_choice`, sampling, `reasoning.effort`) and encodes responses and
  synthesized SSE events. Tool-call ordering is the Codex contract:
  `response.output_item.added` (with the call id and empty arguments) before
  every `response.function_call_arguments.delta`, then
  `response.function_call_arguments.done` and `response.output_item.done`
  with the complete arguments, one terminal
  `response.completed`/`incomplete`/`failed` event. Usage is wire-exact:
  `input_tokens` includes the cached subset (the core's `input_tokens`
  excludes cache reads — Anthropic wire semantics — so the cache read is
  added back at the client encode; a same-format Responses hop round-trips
  the upstream's exact numbers), and `input_tokens_details.cached_tokens` /
  `output_tokens_details.reasoning_tokens` are always present as explicit
  numbers — the omitempty omission of a zero `cached_tokens` is the exact
  bug that broke Codex against one-hub. The route is
  `POST /providers/{provider}/v1/responses` under the `responses` endpoint
  family; the provider resolves it through `routes.responses`. Same-format
  passthrough forwards the raw Responses body with only the model rewrite,
  so `store`, `include`, `previous_response_id`, `prompt_cache_key`,
  `reasoning.summary` and replayed items survive byte-for-byte; a
  Responses client field with no equivalent on another provider is rejected
  by name through the cross-format policy.
- **Gemini native (step 2, backlog 19).**
  `llm-proxy-protocol/src/client/gemini.rs` decodes `contents` /
  `systemInstruction` / `generationConfig` (incl. `thinkingConfig` onto the
  core thinking hint) / `tools.functionDeclarations` / `toolConfig`, takes
  the model from the URL and streaming from the method, and encodes
  candidates/parts/finishReason/usageMetadata (with `thoughtsTokenCount`).
  Function calls and results correlate by name (the shared
  name-correlation pass), and `thoughtSignature` round-trips on
  `functionCall` parts. Routes:
  `POST /providers/{provider}/v1beta/models/{model}:generateContent` and
  `:streamGenerateContent` — SSE with `?alt=sse`, and the documented
  JSON-array-of-chunks framing without it. The provider resolves the
  dialect through `routes.generate_content`; same-format passthrough
  forwards the body unchanged (the model lives in the URL, streaming in
  the method; the adapter rewrites both). Errors use the Google
  `{"error": {code, message, status}}` shape.
- **Ollama native (step 3, backlog 36).**
  `llm-proxy-protocol/src/client/ollama.rs` decodes `/api/chat`
  (messages, `images`, `tool_calls`, `tools`) and `/api/generate`
  (`prompt` + `system`); `options.{temperature, top_p, top_k,
  num_predict, stop}` map onto core sampling and any other option key is
  rejected by name (an Ollama engine knob has no cross-format equivalent);
  `think` levels map onto `reasoning_effort`. Streaming is NDJSON (one JSON
  object per line, no SSE), tool calls emit as complete `tool_calls`
  function objects, and the terminal object carries `done_reason` plus
  `prompt_eval_count`/`eval_count`. `/api/tags` lists the provider's (or
  pool's) catalog in the Ollama shape. Both generation routes resolve
  through `routes.chat_completions` — they are chat endpoints reached in a
  different dialect — under the `ollama` endpoint family, reusing the same
  admission middleware chain.
- **Engine plumbing.** New `ProviderRouteKind::Responses` and
  `GenerateContent` (TOML keys `routes.responses` /
  `routes.generate_content`), new `ClientDialect` variants recorded on
  usage rows, new `ClientWireFormat` variants wired into same-format
  detection and the cross-format field policy (the Responses and Gemini
  server-tool checks reject `web_search`/`googleSearch`-style tools by
  name cross-format), per-dialect error bodies, and a stream-framing layer
  that serves SSE with named events (Anthropic, Responses), SSE data-only
  (Chat, Gemini `alt=sse`), NDJSON (Ollama) or a JSON array (Gemini
  default) from the same core-event pipeline.
- **Done-when verification.** The shared dialect scenarios — streaming
  text, tool calls, exact usage, upstream errors, same-format passthrough —
  run for every dialect against a loopback fake upstream in
  `crates/llm-proxy-server/tests/client_dialects.rs` (21 tests), including
  the Codex tool-call ordering assertion and the wire-exact usage fields.
  Captured Codex traffic replays through the Responses route in the
  plan-02 contract harness (`tests/contract_replay.rs`): the real Codex
  CLI 0.167.0 was installed from the `@openai/codex` npm package and run
  against a loopback capture upstream (fake credential, canned replies —
  one reply drove a real `exec_command` tool turn the CLI executed in its
  read-only sandbox), and both requests of that session
  (`fixtures/contract/captured/codex/`) must reach the fake Responses
  upstream byte-for-byte apart from the model alias, with none of the
  Codex client headers (`originator`, `session-id`, `thread-id`,
  `x-client-request-id`, `x-codex-*`, user agent) forwarded and the SSE
  response returning with exact usage. The capture documents the real
  0.167.0 wire shape — `developer`-role system-prompt/skills/permissions
  messages, `environment_context`, replayed `reasoning` with
  `encrypted_content`, a real `function_call`/`function_call_output` pair,
  `namespace` and `web_search` tool shapes, `client_metadata` — which the
  synthesized `codex/` fixture (kept for extra documented field sets)
  only approximated. All three dialects decode,
  run through the engine and re-encode from the SDK with no
  `llm-proxy-server` item (`crates/llm-proxy-sdk/tests/dialects_without_server.rs`,
  passing under `--no-default-features`, where the server crate is not
  compiled).

## Done when

- Captured Codex traffic replays through the contract tests from plan 02.
- Each dialect passes the shared streaming, tool-call, usage and error tests.
- A dialect can be used from the SDK without the server crate.
