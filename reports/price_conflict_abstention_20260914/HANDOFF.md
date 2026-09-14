# Price-conflict abstention — 2026-09-14

## Current phase

`BLOCKED_PROTECTED_IDENTITY_COVERAGE`: required identity evidence is incomplete. This work package
stops before new data construction, training or inference. No new GPU job, final, deployment or
monitoring task has been started. The semantic price regression is NOT claimed fixed.

- Exclusive workspace: Codex-managed `91ac/Trip_Project`.
- Exclusive branch: `codex/trip-scorer-audit-20260914`.
- Starting tip: `6efd95aa39851f271033440be210ab2fc12bb0f5`, clean and pushed at inspection.
- Accepted scorer source: `e2cc1f06608ed48de101eeb015c54e52e2b825e8`; previous scorer package is
  accepted and immutable. This is a separate development experiment, not another scorer repair.
- Formal release and old `b8bb` branch remain read-only.

## Scope and stop conditions

One versioned synthetic training/development experiment only: fixed v9 adapter baseline,
continuation candidate with added price-conflict counterexamples, repaired v10 renderer, and
style/facility/category/dialogue retention checks. No final split or Fresh Test 120 access.
New input identities must be checked using available protected registries; insufficient exclusion
evidence stops inference. Protocol, source/dependency hashes and data locks must be committed and
pushed before training. At most one single-GPU job, three-hour limit, existing Iris environment
only, with account-wide queue inspection before submission. Failure is retained; no automatic retry.

## Completed changes and fixed commits

- P2 documentation-only commit: `c5d59556f3bbdf658f9dacae74fde3369e4b8f4e`. Exactly the six
  authorized Markdown files distinguish current scorer v2 from historical gates and test counts;
  clarify that promotion cannot change annotation provenance; limit old Spartan cleanup statements
  to their historical point; remove the unnecessary public account name from README. Old scoring
  evidence JSON/reports and accepted scorer code remain unchanged.
- Prerequisite audit/protocol/tests: `f2eccec376b12fbb72c93e68e071803d8af19e39`.
- The new comparison is explicitly price-conflict-enriched continuation versus fixed v9, not
  causally isolated data-composition improvement: additional optimization steps remain confounded.
- Eighteen focused tests passed (18 run, 0 failed/error/skip). No historical 1013-suite rerun.
- Ten identity-only files hash-verified, with 96,788 registry rows (NOT distinct or new samples).
  Independent hashes and field counts agree; post-commit inventory reproduction is byte-identical.
- Complete required coverage is 0/12 scopes. Query/source-content hashes are absent in available
  registries; Week 6 lacks source/image/query fields too; v4 image registry excludes dialogue;
  v5 complete identity registry was not located. The remaining protected-pool coverage is uncertified.
- Python inventory CLI correctly returns 2. New data overlap counts are null/NOT_RUN, never zero.

Fresh Test 120 raw samples, labels and predictions were not opened. Only existing identity/lock
metadata were read. Its stale historical lock status is not treated as unconsumed. No sealed final
was regenerated or opened to fill missing metadata. No new input cards were created, so visual QA
and input/data locks are NOT_RUN, not passed. No model metric, quality gain or negative model run.

See [review](../development/reviews/price_conflict_abstention_20260914.md),
[identity inventory](identity_inventory.json), [verification](verification.json), and
[machine evidence/index](../../experiments/price_conflict_abstention_20260914.json).

## Required external evidence

The data custodian must supply a source-bound identity-only export covering missing query/content
SHA and all protected pools, with canonicalization and manifest/lock provenance. This writer must
not read or rebuild sealed final to obtain it. Once available, the coordinator can separately
authorize continuation and resource scheduling. No automatic resubmission or further tuning here.

## Resources

CPU verification slot was allocated by the current cross-project coordinator, NOT by the original
Project Control task (which is waiting for serialized integration). It is now **RELEASED**.
No GPU budget consumed. Account-wide Iris queue was empty at 2026-09-14T04:54:32Z; historical
jobs 30044630 and 30211170 are COMPLETED 0:0 and do not need resubmission. No extra writer,
active helper, monitor, scheduled task, model download or dependency installation.

All changes stay on the existing exclusive branch. The closeout response supplies the exact final
pushed tip; code and documentation commits above are fixed independently of the evidence commit.
No main/stg integration, formal-package mutation, default-adapter switch or resume edit.
