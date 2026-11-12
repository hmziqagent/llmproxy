# 01. Bug fixes

- **Status:** Completed
- **Depends on:** Nothing
- **Audit rows:** 1, 2, 3, 4, 5, 6, 7, 10, 11, 13
- **Code-audit bugs:** BUG-1, BUG-2, BUG-3, BUG-4, BUG-5, BUG-6, BUG-7, BUG-8,
  BUG-9, BUG-10, BUG-11, BUG-12, BUG-15, BUG-16, BUG-17, BUG-21, BUG-22, BUG-26
- **Backlog:** 3, 5, 7, 28
- **Conflicts:** None

Make what already exists correct and truthful before anything is built on it.

## Done

- **Server exit (row 1, BUG-1):** the shutdown deadline now starts at the
  shutdown signal, not at startup.
- **Prompt deletion (row 2, BUG-2):** `ttl` is carried on cache markers, and
  content the proxy can't parse gets a 400 instead of disappearing.
- **Redirect leak (row 3, BUG-8):** upstream redirects are followed only within
  the same origin.
- **Client key sent upstream (row F, BUG-9):** `passthrough_auth` combined with
  `auth.mode = "client-keys"` is refused at config validation
  (`validate_passthrough_auth_combination` in
  `crates/llm-proxy-core/src/provider_config.rs`, checked by `validate` and at
  startup) until a separate upstream-key header exists.
- **Anthropic stream usage and stop reason (row 4, BUG-3):** `message_delta`
  usage objects carry only `output_tokens` on the wire and parse now; the
  decoder merges the input-side usage captured at `message_start`, so usage
  and the real stop reason (`max_tokens` and friends) reach the client and the
  ledger. The `anthropic/stream_usage` fixture asserts the merged numbers.
- **Usage on chat streams (row 5, BUG-7, backlog 7):** the chat adapter always
  sends `stream_options.include_usage = true` upstream; the client encoders
  emit a usage chunk only for OpenAI clients that asked for one. Absent usage
  stays labelled by `UsageProvenance`, not a new flag. `Accept-Encoding` was
  not touched — the proxy sends no such header.
- **Client-visible usage (row 6, BUG-6):** both client encoders hold the
  terminal pair back until `finish()`, so usage arriving after the finish
  event lands in it instead of being dropped.
- **Cut-off streams (row 7, BUG-4, backlog 28):** the provider decoders expose
  `saw_terminal()`; a stream the upstream cut off, or ended with an in-band
  `{"error": ...}` frame (now decoded instead of skipped), is marked with an
  in-band error event for the client while the synthetic terminal keeps the
  framing valid, and the ledger records `ResponseFailed` instead of success.
- **Cache markers (rows 10, 11, BUG-5):** message-level text markers (with
  `ttl`) and `tool_result` markers — the `CoreContent::ToolResult` block grew
  a `cache` field — survive Anthropic encode; the OpenAI `cache_control` shape
  carries `ttl` too.
- **Usage mapping (row 13, BUG-11, BUG-12, backlog 3):** DeepSeek miss tokens
  count as plain input, never cache creation; `prompt_tokens_details.
  cached_tokens` (chat and Responses), `completion_tokens_details.
  reasoning_tokens` and Anthropic's 5m/1h `cache_creation` split map into the
  core fields; absent usage is labelled missing (`SyntheticZero`), not
  provider-reported zeros. OpenAI clients receive `prompt_tokens` including
  cached tokens.
- **Ledger accuracy (BUG-10, BUG-21, BUG-26, backlog 5):** streaming rows keep
  `model_upstream` under aliases; a client disconnect before the first event
  settles its row; events lost to a failed ledger batch or a full channel are
  counted in `dropped_events()` and logged with the count.
- **Upstream codecs (BUG-15, BUG-16, BUG-17, BUG-22):** the Responses decoder
  reads the documented `item` / `response.usage` SSE shapes and the encoder
  emits typed `function_call` input items plus temperature/top_p/
  max_output_tokens; Gemini reads camelCase `functionCall` parts, streams via
  `:streamGenerateContent?alt=sse` with no `stream` body field;
  `tool_choice: "auto"` is encoded as a string; deltas no longer carry
  `"role": ""`. Each fix has a regression test against the documented shape.

## Done when

- [`run.sh`](../research/audit-2026-11-05/evidence/probes/run.sh) shows correct
  results for rows 4 to 7, 10, 11 and 13.
- Each fix has a regression test that fails without it.
