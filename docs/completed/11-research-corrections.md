# 11. Research corrections

- **Status:** Completed (2026-11-08)
- **Depends on:** Nothing
- **Audit rows:** None
- **Code-audit bugs:** None
- **Backlog:** All 38, plus the ten items the audit found missing
- **Conflicts:** None

Correct the research so it can be used without reading the audit alongside it. The line
references are in the [audit report](../research/audit-2026-11-05/REPORT.md).

## To do

1. **Claims about llm-proxy.** Fix `competitive-gap-analysis.md` and
   `competitive-landscape-compendium.md` from "False or misleading claims". Line numbers
   in those docs were recorded at the audited commit, so re-check them first.
2. **Claims about other products.** Correct the "nobody ships X" claims and stale
   statuses from the same section.
3. **Paid and beta features.** Separate them from self-hostable ones, and restate the
   prevalence counts on that basis.
4. **Citations and demand.** Fix the wave 3 and 4 citations in `repo-intelligence.md` and
   `niche-repos.md`, and add the caveats from "How strong the demand evidence is".
5. **Provider notes.** Update `providers.md`, `core-protocol.md` and `endpoints.md`, add
   the missing current behaviour and date each section.
6. **Backlog.** In `compensation-backlog.json`, apply the corrections from "Backlog
   review", add the missing items and record each candidate's plan number.
7. **Housekeeping.** Work through "Doc housekeeping", including the root `README.md`
   description of `llm-proxy-api` and the rate-limit comment in `config.toml.example`.
8. **Remove the warning** in `docs/research/README.md` once the steps above are done.

## Done when

- Re-checking the audit's false claims finds each one corrected or removed.
- Every backlog candidate names a plan, or says why it's dropped.

## Done

- Every claim in the audit's "False or misleading claims" section (both the
  llm-proxy and other-products subsections) is corrected in place, marked
  "(corrected 2026-11-08 per research audit)" and cross-checked against the
  same audit line references; GA and CO carry a dated banner scoping their
  baseline to the audited commit `79cf406`, and `repo-intelligence.md`
  carries the same scoping after its intro paragraph.
- The paid-edition and beta features (Kong, Portkey, LiteLLM, Tyk,
  Cloudflare) are separated from self-hostable ones, and the affected
  prevalence counts are restated on that basis in both gap lists.
- The seven wave-3/4 citation errors are fixed (each verified against the
  live GitHub item), R1's count is corrected, and the audit's
  "How strong the demand evidence is" caveats are a section of the
  repository-intelligence record, with the young-repo and vendor-number
  caveats added where they apply in the niche survey.
- `providers.md`, `core-protocol.md` and `endpoints.md` are dated per
  section, carry the audit's corrections (OpenRouter `usage.include`,
  `guardContent`, Anthropic Files beta header, forced-`tool_choice` 400) and
  the missing current behaviour (Anthropic 4-breakpoint limit and `ttl`,
  Anthropic's model-gated 400s on temperature/prefill/thinking, OpenAI
  `prompt_cache_key`, the Azure v1 path) — each verified against the
  provider's current documentation.
- `compensation-backlog.json` carries the audit's backlog-review
  corrections (rescopes, restated rationales, corrected crates, #28 raised
  to high), records each candidate's consuming plan or drop reason, and
  appends the ten missing items.
- Housekeeping done: compendium title/first paragraph say three waves, the
  research README no longer says "three October waves" and the audit
  warning is replaced by a pointer to the completed corrections, the
  registry scope says 72 rows, the Upstash negative finding is out of the
  products table, the harvest manifest is linked with its 120-vs-63
  disclosure, clone-path recheckability is disclosed, `config.toml.example`
  describes key-scoped rate limiting. The root `README.md` description of
  `llm-proxy-api` was already correct. The `upstash.com/docs/llms.txt`
  note in the audit's housekeeping list is an audit self-disclosure and
  stays recorded in the audit report itself.

