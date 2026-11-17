# Documentation

Start with the [project goal](goal.md). It holds the goal, the rules every
change follows and the decisions already made.

## Plans

Each plan is one stage of work, with its scope, steps and finish line.
Finished plans are linked under Completed below.

| # | Plan | Status | Depends on |
|---|---|---|---|

How to use them:

- Take the lowest-numbered plan whose dependencies are done.
- Update the status here and at the top of the plan in the same change as the
  work.
- When a plan is finished, move it to `completed/` and link it below.
- New work goes into an existing plan, or gets a new plan listed here.

Plan headers point to the [research audit](research/audit-2026-11-05/REPORT.md):

- **Audit rows** are the numbered bugs in its bug table.
- **Code-audit bugs** (`BUG-n`) are in
  [section F of the claim tables](research/audit-2026-11-05/evidence/llm-proxy-claims.md).
- **Conflicts** are the numbered rows in its feature conflicts table.
- **Backlog** numbers are positions in
  [`compensation-backlog.json`](research/compensation-backlog.json).

## Not planned

- Subscription OAuth and client fingerprinting (backlog 12, part of backlog 15,
  conflict 18). See the [credentials decision](goal.md#credentials).
- Reseller billing. See the [features decision](goal.md#features).
- Streaming tool-call fragment repair (backlog 29). The proxy doesn't reassemble
  fragments, so there's nothing to repair.
- Ledger retention work (backlog 33). Retention already works, and there's no
  sidecar data to orphan.

## Reference

- [Protocol architecture](protocol.md): the normalized core and streaming rules.
- [Data-class policy](data-class-policy.md): what every opt-in stores, where
  and for how long; prompts and responses are stored nowhere by default.
- [Lint guide](rust-anti-slop-lint-guide.md),
  [lint setup](SETUP-RUST-LINTS.agent.md) and
  [lint backlog](lint-backlog.md).

## Research

- [Research index](research/README.md): competitor surveys, provider notes and
  the feature backlog. Use it as a catalogue of options, not a ranking of demand.
- [Research audit, 2026-11-05](research/audit-2026-11-05/REPORT.md): what was
  checked and what turned out to be wrong. Its false and misleading claims
  were corrected in the research docs on 2026-11-08
  ([plan 11](completed/11-research-corrections.md)); where a doc still
  describes llm-proxy's own code, it describes the audited commit.

## Completed

- [11. Research corrections](completed/11-research-corrections.md)
- [10. Language bindings](completed/10-bindings.md)
- [09. Optional crates](completed/09-optional-crates.md)
- [08. Client dialects](completed/08-client-dialects.md)
- [07. Hardening and operations](completed/07-hardening-and-operations.md)
- [06. Cache and dedup](completed/06-cache-and-dedup.md)
- [05. Admission and budgets](completed/05-admission-and-budgets.md)
- [04. Routing and failover](completed/04-routing-and-failover.md)
- [03. Engine and crate split](completed/03-engine.md)
- [02. Request fidelity](completed/02-request-fidelity.md)
- [01. Bug fixes](completed/01-bug-fixes.md)
- [Provider-scoped remodel](completed/remodel-proxy.md)
