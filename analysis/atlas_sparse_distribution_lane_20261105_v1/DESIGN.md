# Concern #8 sparse-distribution lane (PACE vs HCCL, 17 shapes, dual clock; prepared source, not yet reviewed or run)

Status: **prepared source spec**. Pure local CPU output; no device, remote or
SSH operation has been performed. Before any device execution this source
needs an independent `review.json` (status `APPROVED`, scope
`sparse_distribution_pace_vs_hccl_p16`) binding the source-manifest digest,
the frozen `cases.json` digest and the exact campaign structure — validated
against the ONE shared contract constant in `plan.py`
(`plan.validate_review`, consumed by `launch.verify` through the alias
`launch.verify_review_binding`). This exists specifically so the v3-class
BLOCKER (launcher expecting a review schema no reviewer can issue) cannot
recur; `test_cpu.py` proves a synthetic APPROVED fixture is accepted and that
REJECTED status / wrong manifest SHA / wrong scope / missing reviewed_files /
wrong groups are all refused.

Implements the delivered plan
`analysis/atlas_sparse_distribution_plan_20261105_v1/PLAN.md` (bound by
SHA-256: PLAN.md, gen_cases.py, cases.json — the case list is **referenced,
never copied**; every guard is recomputed from the embedded 16×16 matrices
and the recomputation is authoritative over the declared statistics).

## Measurement boundary (read first)

- **Dual timing口径 from the same batch of samples.** Every measured
  iteration records per-rank `t0/t1` (`perf_counter_ns`, one container ⇒ one
  CLOCK_MONOTONIC domain; the口径 is void if ranks ever run in separate
  containers — declared fragile in PLAN §11 and enforced here only by the
  single-container launch structure):
  - `rank_local_max = max_i(t1_i − t0_i)` — the historical boundary (host API
    entry → own `npu.synchronize()` return), directly comparable in structure
    to `three_groups_report.md`;
  - `makespan = max_i(t1_i) − min_i(t0_i)` — common-start口径 (earliest actual
    entry → latest completion);
  - `arrival_skew = max_i(t0_i) − min_i(t0_i)` (gate-quality diagnostic,
    >5 ms flagged per iteration) and
    `wait_gap = makespan − rank_local_max ≥ 0` (arrival skew + in-collective
    straggler wait), reported per (case, arm, group).
- **The common start gate is an HCCL collective** (rank-0 flag +
  `dist.all_reduce(MAX)` before `t0`, measured iterations only; warmups keep
  the historical ungated structure). It sits outside the timed interval but
  **is not neutral**: it may warm HCCL context in the HCCL arm. Both arms
  share the identical gate, and this non-neutrality is a declared limitation
  of every HCCL-arm number in this campaign, including the rank-local口径.
- Comparability with the historical rank-local data is structural (same call
  boundary, same APIs) but not controlled: the gate adds前置 state and
  cross-campaign drift is uncontrolled. R1/R2 are therefore re-measured
  within this campaign; history (3 forward-order + 3 reverse-order groups,
  embedded as constants with binding notes) is shown only as descriptive
  drift, never pooled.
- Synchronous complete-call scope: includes Python enqueue, host wait and
  peer wait; no kernel-only time; no bandwidth model.

## Correctness argument (per case, per group, fail-closed)

- Inputs/outputs are the lane's deterministic shake_256 BF16 payloads (bit0
  and sign of every high byte cleared ⇒ exponent-bounded, finite), generated
  per segment on CPU; the CPU oracle is compared **bit-exactly** (uint8 views)
  against PACE output and HCCL output.
- Initial full check (1 PACE call + 1 HCCL call, PACE first in every group —
  order influence is a recorded covariate via `arm_first`): output bit-exact ×
  2, input unchanged, send/recv splits unchanged, five tail sentinels
  (input/pace/hccl/send/recv canaries). Final full check repeats the set and
  adds `all_reduce(SUM, bad)`; any nonzero global count fails the whole group
  with no partial results published. Intermediate repetitions are not
  individually checked (validation_scope recorded per case, lane precedent).
- Generation labels `11 × case_index + 1` generalize the lane formula to
  17 cases; runner-embedded self-check rebuilds each rank's send/recv from
  the embedded matrix and asserts the per-rank output rows.
- 11 calls per case per arm (1 + 2 + 8), identical to the historical lane;
  the audit re-derives the exact iteration sequence per group and rejects any
  order/count/timestamp/gate deviation.

## Campaign structure (PLAN §4, all pre-registered)

- 6 fresh process groups (job0–job5), each its own container/ports; arm order
  3/3 (job0/2/4 PACE-first); within-group measured blocks are
  (base, reverse) × 4 with 2 calls per arm per block; case order rotates by
  `3g mod len` (6 distinct offsets over 17 cases; anchors leave the head).
- **Trimming ladder** (explicit downgrade, never silent): S8 → R3 → S5 → R5,
  one case at a time, floor 13 cases. `group_start --drops N` may only raise
  the level by one and must cite a prior group whose observed container
  duration exceeded 14 min (`plan.trim_gate`); the trigger and the dropped
  case are recorded in `preflight.json` and `completion.json`. Reduced case
  coverage can never be classified "stable" (6/6 rule unreachable) and the
  aggregate marks it.
- **Plan B** (only after an observed SQ fail-closed rejection): 17 cases
  split into disjoint subgroups a (10 cases, contains the R1/R2 anchor pair)
  and b (7 cases), balanced by the PLAN's device-time ladder; 12 groups; per
  subgroup aggregates only; plan_b results are never pooled with plan_a. The
  launcher arms `sq_gate_fail_closed.json` only when all ranks rejected the
  SQ budget before any payload.
- **Ladder coupling across plan_b subgroups is intentional**: the trim
  admission (`require_prior_groups`/`plan.trim_gate`) reads every earlier
  receipt of the *variant*, so pbb0's permitted level inherits all pba*
  receipts' drops and subgroup b can never run at a lower level than
  subgroup a ended on — one single monotone ladder per variant, never
  restartable in subgroup b. Rationale: subgroup b runs strictly after the
  a-campaign; allowing the level to drop again at the hand-off would
  reintroduce exactly the mid-campaign "un-dropping" the pre-registered
  ladder forbids, and the monotonicity invariant (drops never decrease
  within a campaign) is what makes the recorded trim timeline auditable.

## SQ budget gate (PLAN §6)

After SHMEM init and before any payload, each rank captures quiescent queue
metadata with the frozen helper (read-only `g_state` export +
`aclrtMemcpy` D2H) and `plan.plan_call_budgets` recomputes the cumulative SQE
bound per channel over the **exact planned call sequence** (Σ over cases of
per-case `submission_budget × 11`). Any channel that could wrap the SQ ring
(**depth measured at runtime**; 2048 is the reviewed historical observation,
cross-checked against the cases.json planning block, never trusted as a
gate) or the 16-bit task counter (65536) fails closed before payload; the
rejected run persists `sq_gate_reject.json` per rank. The audit re-parses the
raw workspace bytes and recomputes the identical budget; large-payload SDMA
chunking is not modeled (PLAN §11.1).

**True margin (helper-based, review round 1):** recomputing the gate with
the frozen helper over all 17 cases × 11 calls gives a worst-channel
cumulative of **1969 SQEs at rank 1 vs the observed depth 2048 — a 3.9%
margin**, not the 54%/62% that PLAN §6's linear planning models suggest. The
cases.json estimates (model A 935 = 5.0/call × 187; model B 785 =
0.156/MiB × 5024 MiB) are the frozen 2-case-audit extrapolation; they
materially under-count because burst/zipf/dense shapes emit
`ceil(peer_bytes/1 MiB)` payload SQEs per peer (syn-perm-burst 31 SQE/call
on one channel, syn-burst 25, 163-combine 24, zipf 16). They are recomputed
and cross-checked for accounting integrity only — **the helper-based bound
is authoritative**. Plan_b subgroups recompute to 671 (a) / 1364 (b) and the
fully-trimmed plan_a floor (13 cases) to 1705: plan_a at full scope sits one
depth reduction away from fail-closed, so the pre-registered Plan B path has
material probability, and a smaller runtime-observed depth legitimately
forces it.

## Launcher discipline

`launch.py` actions: `precheck_arm`/`precheck_audit` (no-device,
no-network container: imports, RTLD_NOLOAD/dladdr provider identity, PACE op
schema, cases/guards/plan identity for both variants, helper/layout digests,
ACL provider identity, review-contract constant echo), then per group
`group_start`/`group_audit`, then `aggregate`. Every group gets **two
complete 16-die admission preflights** (guard lane: docker-empty, davinci
nodes, npu-smi all-empty/healthy, fuser, per-group port bind) re-validated at
audit; the free-memory admission is the reverse-lane formula
(heap 4 GiB + max peak tensors + 1 GiB) agreed by all-reduce before SHMEM
init; HCCL liveness all-reduce before and after the measured region;
container contract audited (image, host net/ipc, exact device map, read-only
mounts + rw payload, env, natural zero exits, no OOM); exact-nanosecond
RFC3339 ordering preflight ≤ preflight ≤ start < finish; postflight npu-smi
observation. Strict group order, unique ports, no retries, no overwrites, no
signals. `aggregate` requires every group receipt, **rehashes all evidence
including the precheck completion and both preflight snapshots** (v3
advisory lesson) and re-parses both snapshots. PLAN §9.3 (device-split
admission semantics source verification) is carried by the review contract
field `device_split_guard_semantics ∈ {per_peer, whole_input, union}` — the
lane sees only the built .so, so `unverified` never opens the gate.

## What this campaign cannot establish (must ship with results)

- Synthetic density/skew families are **not** production routing
  distributions; the captured 512 calls are one fixed prompt, two forwards;
  no model/serving benefit is claimed anywhere.
- Single node, P16; no extrapolation to 32/64 ranks or multi-node (concern
  #14 unchanged). Synchronous complete calls on host clocks; the start gate
  is an HCCL collective with possible HCCL-arm warm-up bias (above).
- All shapes keep per-peer ≤ 30 MiB ≪ 96 MiB slot (union-guard safe: whole
  input ≤ 96 MiB also enforced); overflow/fallback behavior is untouched
  (concern #10 unchanged).
- The group is the independent unit (n=6); within-group repetitions are
  repeated measures. Median/IQR/min/max of group medians only; no confidence
  intervals, no inferential tests; the pre-registered direction rule
  (6/6 same sign beyond the tie threshold `max(1% × HCCL group median,
  2 × arm MAD)` — union of both arms' MADs, the conservative reading of
  PLAN §8's "该 arm" under symmetric two-sided use) classifies
  stable-faster/stable-slower/unstable, and unstable outcomes are reported
  with flip counts and both directions' medians. All samples are kept,
  including any HCCL early-iteration elevation, if it reappears.
- Cross-shape "direction flips" are descriptive contrasts only
  (rotation + AB/BA mitigate but same-day drift is uncontrolled); historical
  anchor tables are cross-campaign drift only.
- Fragile assumptions carried from PLAN §11: SQE extrapolation (the linear
  planning models under-count; the helper-based truth is 1969/2048 = 3.9%
  margin at full scope, see the SQ budget section — a smaller
  runtime-observed depth legitimately forces the pre-registered Plan B),
  per-case duration extrapolation for the ≈89 MiB combines (ladder
  pre-registered), gate non-neutrality, single-container monotonic clock.

## Source set and fixtures

`FILES`/`REVIEWED_FILES` live in `plan.py` (DESIGN.md, plan.py, worker.py,
supervisor.py, precheck.py, launch.py, test_cpu.py, make_manifest.py,
observed_npu_info_fixture.json + source_manifest.json). The npu-smi fixture
is the observed all-idle 8-card/16-die snapshot shared verbatim with the
barrier-v3 lane (identical SHA-256) and is used only by host-side parser
tests. Regenerate the manifest with `make_manifest.py` after any intentional
edit; the launcher pins its digest.
