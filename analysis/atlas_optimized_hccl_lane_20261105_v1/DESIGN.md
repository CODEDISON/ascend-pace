# Concern #4 four-arm optimized-HCCL campaign lane (prepared source, not yet reviewed or run)

Status: **prepared source spec**. Pure local CPU output; no device, remote or
SSH operation has been performed. Implements the delivered plan
`analysis/atlas_optimized_hccl_plan_20261105_v1/` (PLAN.md + arms.json, both
bound by SHA-256 and re-derived here). Any execution requires an independent
`review.json` validated against the ONE shared contract constant
(`plan.REVIEW_CONTRACT` / `plan.validate_review`, consumed by
`launch.verify` through the alias `launch.verify_review_binding`), plus every
PLAN §13 prerequisite; `test_cpu.py` proves a synthetic APPROVED fixture is
accepted and REJECTED / wrong manifest / wrong scope / missing reviewed_files
/ wrong groups variants are all refused (v3-class BLOCKER prevention).

## Arms and what each timer number means

| Arm | Flow | Timer |
|---|---|---|
| H | frozen uninstrumented-timing driver, `REV2_NO_RESIZE_ARM=native` | driver `perf_counter_ns` immediately around `llm.generate` |
| HmR | same frozen driver, `no-resize` (frozen patch `64f1da23…`) | same |
| HOPT | P0: identical config to HmR (H4 null control); P1: + the one probe-verified, pilot-frozen `HCCL_BUFFSIZE` value | same |
| F | frozen sync-model v1 session (PACE hook + 4 GiB/96 MiB SHMEM) + this lane's recorder removal; hook retained through timing | same |

Every number includes the full synchronous `llm.generate` endpoint (Python
enqueue, model execution, real waits, sampling, metadata) measured in the
driver process; no per-call profiling, no events, no synchronize, no
allocator query inside the measured path; all validation (token equality,
census, request IDs) sits outside the timer. 24 requests per fresh group =
1 gate + 3 warmups + 20 measured; group estimator = median of the 20.

## Why H/HmR/HOPT delegate instead of reimplementing

The frozen timing lane (`1bec8680…`) already implements the reviewed
uninstrumented-timing flow for native/no-resize: captured gate with the full
64-call census, native ten-binding capture removal with identity/fingerprint
verification, W3/I20 and every frozen validator. The campaign reuses it
byte-frozen by SHA-checked import (`worker.run_hccl_arm`) and only
normalizes its persisted evidence into the lane record (census per rank,
request rows). Only the F flow is new code — necessarily, because no prior
lane timed the PACE hook with instrumentation removed.

## F-arm restoration gate (PLAN §12.1, §13.2 — hard launch gate)

1. **CPU fake-import proof** (`restoration.fake_import_proof`, run in
   `test_cpu.py` AND echoed by the no-device precheck): the ten-binding state
   machine over synthetic modules proves save → hook-layer adoption (full
   `__wrapped__` chains, code-fingerprint bound) → exactly-one-layer
   (recorder) removal → hook retained, plus double-restore refusal and
   forged-chain rejection before any mutation.
2. **One-request restoration proof** (`launch.f_proof_start/f_proof_audit`,
   device, before g01, `REV4_MODE=f_proof`): full F flow truncated after the
   proof request — gate census, recorder removal, one reference-equal
   request, recorder digest unchanged, natural finalize — emitting
   `F_PROOF_PASS`. The audit turns a failure into a FAIL receipt; the
   campaign then degrades to three arms ONLY explicitly
   (`aggregate --degraded-3arm`, and the review must say
   `degraded_three_arm`). `group_start` for any F group hard-requires both
   the review's `f_restoration_gate: PROVEN` and the PASS receipt.
3. Through the timing region, every group re-verifies per rank: restored
   binding chains unchanged and the recorder evidence byte-identical
   (`revision4_pace_timing_snapshot` after the last request).

## O2 honorability probe (concrete criterion, P0 default)

`launch.probe_start/probe_audit` run one ≤3-minute container: HCCL init +
one all-reduce with `HCCL_BUFFSIZE=333` (a value no plausible default would
echo), then the container's own HCCL log files are collected (payload
`probe_logs.txt`, digest-bound). The pre-registered judge
(`plan.judge_probe_receipt`) declares P1 **only** if the raw log itself
contains a line matching `HCCL_BUFFSIZE … 333` (knob name AND probe value),
with the receipt's matched lines required to equal an independent rescan;
any mismatch rejects. No such echo evidence exists anywhere in the local
materials (the hcomm v9.0.0 docs document the knob's existence and priority
`hcclBufferSize > HCCL_BUFFSIZE > default`, not an init-time echo), so the
lane plans for **P0**: HOPT ≡ HmR configuration, reported as the
"maximal implementable optimized-HCCL" finding with the H4 same-config
control; a P1 review without an audited P1 probe receipt is refused.
K2 (one extra knob evidenced by the pinned image's own log/doc strings in
the probe receipt) is admitted by the plan text but unimplemented in this
lane's judge — only K1 is pre-registered here (declared deviation).

## Census (per rank, fail closed; frozen-predecessor REAL formats)

The census adapters (`plan.normalize_hccl_census` /
`plan.normalize_pace_census`) consume the exact evidence formats the pinned
predecessors persist, verified against their real stored artifacts and
frozen source line numbers (cited in `frozen_census_fixture.json`
provenance):

- **H/HmR/HOPT**: `validated_gate_trace['alltoallv_global_calls'] == 64`
  **distinct collectives** (each executed by all 16 ranks; the frozen
  `run_timing.py:60` asserts the same — never 16×64). Each rank's gate
  capture must carry `counts == {collective: 64, forward: 2, method: 64,
  runner: 3}`, 64 `method_enter` events all under `backend_path
  'All2AllV'` (32 dispatch + 32 combine), 64 `alltoallv_begin` events
  inside those methods, and 2 `ALLTOALL` selectors on 2 `execute_model`
  forwards (`capture_contract.validate_worker` L107–186).
- **F**: the session snapshot's `counts['python_pace_calls'] == 62` /
  `['python_hccl_calls'] == 2` / `['python_logical_calls'] == 64`,
  cross-checked against the 64 call records keyed by `row['backend']`
  (`'PACE'`/`'HCCL'`; `model_state.py:88`, snapshot L108–113; the qualified
  sync-model v2 raw-audit counts).
- **Recorded finding**: the arms.json census parenthetical "(2 All2AllV +
  62 MC2)" does not match any of the six persisted 2026-10-28 groups — all
  64 collectives ride All2AllV methods across 2 execute_model forwards
  (the 2+62 split belongs to the F arm's logical-call accounting). The
  lane binds the empirical invariant; `test_cpu` records the discrepancy
  (`test_arms_json_census_parenthetical_is_empirically_wrong`).
- CPU fixtures: `frozen_census_fixture.json` carries verbatim subsets of
  the real persisted evidence (six-group campaign `pair2-no-resize`
  `request_raw.json` after_gate + `validated_gate_trace.json`; sync-model
  v2 `request_raw.json` after rows), with source paths + SHA-256 provenance
  re-verified by the tests, plus a full unexcerpted 11 MB end-to-end
  adapter test and mutation rejections (global 1024, broken counts,
  non-All2AllV backend, wrong selector, broken 32+32 split, orphan
  collective, rank drop, contradicting backends).

## F-arm SQ gate (PLAN §7)

After SHMEM init and before any payload the frozen install captures
quiescent queue metadata and budgets the 64-call sequence (frozen helper
`0d7fde8e…`, read-only `g_state` + one D2H). This lane adds the
request-boundary check: one capture per rank after gate/w1/w2/w3;
`plan.validate_sq_boundaries` fails closed if any (rank, channel) occupancy
(tail−head) series is strictly increasing across the four boundaries
(monotone accumulation). Constant or decreasing occupancy passes — drain is
evidenced, not proven (declared fragile; large-payload SDMA chunking not
modeled, prior-lane statement retained).

## Campaign discipline

- **Williams schedule** (arms.json, re-derived + tested): 16 groups, 4
  blocks, each arm leads once, each within-block position once, all 12
  ordered adjacent pairs exactly once; the 3 cross-block adjacencies are
  declared extras. Group k starts only after group k−1 audited PASS
  (`plan.GROUPS` order, `require_prior_groups`).
- **Trimming ladder** (downgrade only, never silent): L0 16 → L1 12 (drop
  rep4) → L2 8 (drop rep3); each increment must cite prior receipts (one
  group > 8 min, two consecutive > 6 min, or window extrapolation beyond
  4 h) via `plan.ladder_step`; L2 reports descriptively only — no stable
  classification below 3 groups/arm; floor 8. **Per-level aggregation**
  (`aggregate --level L1|L2`, sparse-lane per-subgroup precedent): the
  rollup selects the receipts of the requested level's fixed group set
  (`plan.select_level_receipts` — complete coverage required; receipts
  recorded at an earlier level label are admissible after a mid-campaign
  trim, later-than-requested labels refused; the per-group recorded labels
  ride along in the output as `receipt_levels_by_group`). Whole-campaign
  mode (no `--level`) still requires every receipt at one level.
- **Exclusivity window**: the two complete 16-die admission preflights in
  front of every group double as the inter-group device re-check
  (docker-empty, davinci nodes, npu-smi all-empty/healthy, fuser, per-group
  port bind); a non-empty preflight refuses the start and is recorded, not
  sampled. Audits re-parse both persisted snapshots, enforce RFC3339
  preflight ≤ preflight ≤ start < finish, and rehash every artifact
  (container log/inspect, both preflights, postflight, payload files, and
  the no-device precheck receipt) — in `group_audit` and again in
  `aggregate`. `probe_audit` inspects its container with the full contract
  (guard + world + the rw payload mount at `/workspace/results`), the same
  as every other device container. The source tree walk rejects symlinks
  unconditionally — including inside `__pycache__` (only regular files are
  permitted there).
- **Ports**: 20 disjoint (shmem, master) pairs (16 groups + probe + f_proof
  + 2 pilot slots); HCCL arms carry no SHMEM env; per-group Gloo-free nonce
  derived from the campaign name + group id.
- **Prior qualification**: corrected-parity v2 (native + no-resize) and
  sync-model v2 PASS receipts revalidated by exact hash in `verify()`;
  frozen-lane source manifests (candidate `64f1da23…`, runner `365019d6…`,
  timing `1bec8680…`, sync-model `c755584d…`, qualification `fa29505c…`)
  pinned and FrozenPin-tested against the local artifacts.
- **F-arm module-family wiring**: `model_extension.py` imports its runtime
  family (contract/controller/frontend/padded_buffers/runtime/dependencies/
  queue_metadata + dataset.json/layout.json) by bare module name; the
  family's home is the frozen sync-qualification v2 lane, mounted read-only
  at `/workspace/qualification` (manifest-pinned) and placed on PYTHONPATH
  **before** `/workspace/model_hook` and `/workspace/queue_metadata` so the
  family-internal `queue_metadata` sibling wins any same-name resolution
  (the lane's own helper use is by absolute SHA-checked path, unaffected).
  The no-device precheck proves every family module resolves to
  `/workspace/qualification` inside the container, and `verify()` refuses
  to act if any family member is missing (f_proof device failure 54167a66
  was exactly this gap; the CPU fake-import proof had synthesized the
  family, which is why it could not catch the wiring hole). Round-5
  correction: the qualification mount is the **v3 harness**
  (`revision2_olmoe_o1_sync_qualification_20261028_v3`, manifest
  `f3b36ae0…`) — `model_contract.QUALIFICATION_SHA` binds exactly this
  manifest and `sources()` verifies per-module digests against it; the v2
  dir mounted at round-3/4 carried the modules and satisfied imports, but
  `sources()` would have rejected it at the F gate.
- **Delegated-chain DATA-FILE audit (round-5 root fix)**: an AST scan of
  the frozen chain sources for path constants and file reads (the import
  closure cannot see path+SHA accesses; device failure d4e26ece was the
  legacy prompt helper) produced `plan.DELEGATED_DATA_FILES` — nine
  entries covering the legacy prompt helper
  (`/workspace/legacy/benchmark_olmoe_fixed_batch.py`, pin
  `1c0d4073…`), the model directory (`/workspace/model`, digest-checked
  by the delegated flow via the capture lane's `model_manifest.json`),
  the capture lane's `model_manifest.json`/`runtime_pins.json`, the prior
  mixed-qualification receipt (`/workspace/mixed_qualification/completion.json`,
  pin `263b2b59…`, env-bound for `model_contract.sources()` as
  `REV2_QUALIFICATION_COMPLETION_SHA256` on F groups), the fixed
  reference `request_raw.json`, the qualification `dataset.json`/
  `layout.json` and the cashflex wrapper. Three new read-only mounts
  (LEGACY, MODEL, MIXED_RESULTS) plus digest hard gates on the HOST in
  `verify()` and an in-container existence/digest proof in the precheck
  (record `delegated_data_files`, re-validated by `precheck_audit`).
- **Delegated-chain module resolution (root fix for the wiring class)**:
  the HCCL delegation (`run_timing.py` → `run_o1.py` → `capture_contract`/
  `run_capture`/`no_resize_worker` → `patch_source`/`worker_extension` …)
  and the F-arm composition close over SEVEN module-home directories, all
  now mounted and on PYTHONPATH in `plan.DELEGATED_MODULE_ROOTS` order
  (adding the capture lane at `/workspace/capture`, manifest-pinned — the
  home of `capture_contract`/`capture_core`/`run_capture`/
  `worker_extension`; and `/workspace/no_resize_candidate`, home of
  `patch_source` — both closure findings after device failure 216a2400).
  The no-device precheck now runs a **recursive AST transitive-closure
  proof** over the three delegated entry files: every bare-name import of
  the closure must `find_spec`-resolve to a file inside its expected
  mounted root (stdlib and third-party excluded by name set); any miss
  fails the precheck — the class of defect that reached the device twice
  can no longer pass the gate. The CPU suite runs the identical closure
  over the local mirrors (edge-table, home-coverage and mirror-existence
  assertions included).
- **Selection pilot (P1 only)**: at most 2 candidate groups (pilot0/pilot1
  port slots exist); the fastest group median freezes the HOPT tuning for
  the whole campaign via the review's `hopt_env`; pilot groups never pool
  with the campaign. (Pilot launch actions are not implemented in this
  revision — P0 is the planned path; declared deviation.)

## Plan-document digest corrections (found by programmatic transcription)

FrozenPinTests compare every arms.json pin against the local truth. The
original arms.json carried two transcription typos —
`queue_metadata_helper` `…6ee6ce5f5df…` (should be `…6ee6f5df…`) and
`reference_request_raw_sha256` `…4e70d8d0…` (should be `…4e7048d0…`) —
which this lane's tests recorded as defects on first delivery. The plan
owner corrected both on 2026-11-05; the lane now binds the corrected
arms.json file digest (`dbdb1aef…`) and asserts both fields strictly equal
the actual frozen files (queue-metadata helper and the split-capture
hccl-o1 reference), verified character-for-character in
`test_pins_match_arms_json`.

## Deviations from PLAN.md (explicit)

1. **Pilot launcher actions not implemented** (P1-only path): the port slots
   and the review `hopt_env` binding exist, but `pilot_start/pilot_audit`
   are left for a follow-up revision because the planned default is P0 and
   the probe materials contain no adoption evidence. A P1 review without a
   pilot would have nothing to freeze.
2. **K2 not judge-implemented**: only K1 (`HCCL_BUFFSIZE`) is
   pre-registered in `plan.judge_probe_receipt`; adding K2 requires a new
   lane revision with its own evidence form.
3. **Probe container shape**: single global `WORLD_SIZE=16` HCCL group with
   rank 0 on device 0 and `ASCEND_RT_VISIBLE_DEVICES=0..15` (torch HCCL
   init + one all-reduce), matching PLAN §8.4's "HCCL init + a single
   all-reduce" without spawning 16 processes.
4. **P2-style staging not applicable** (OLMoE campaign, not SHMEM-barrier);
   the delegation reuses the six-group campaign's proven flow instead.

## What this campaign cannot claim (PLAN §14, verbatim binding)

- No component-level attribution of framework vs transport vs completion vs
  compaction (concerns #5/#2); the F-vs-HOPT residual is a bundle that
  inherently includes split residency (O4 is not portable to HCCL).
- No causal share of resize removal beyond the H vs HmR/HOPT contrast; the
  legacy 1.081×/1.087× L− local ratios are not additive with this campaign.
- No generalization to Qwen3/GPT-OSS, O32/decode, serving
  throughput/TTFT/TPOT/tail latency, multi-node or other SKUs.
- No memory conclusions (allocator reset/query forbidden inside the latency
  loop).
- No claim that HOPT is theoretically optimal HCCL — only maximal
  implementable under frozen-source constraints (P1 additionally with a
  one-group, non-optimal tuning search).
- No pooling with any historical campaign (the six-group 2026-10-28
  medians and the 199.326/194.527 endpoint table ride along as
  descriptive-only drift context in `aggregate`), and no post-hoc upgrade
  of an `unstable` classification. `unstable` is a valid, preregistered
  outcome; ~4% group-median spread vs the 2.4–3.1% historical effect means
  every contrast may legitimately classify unstable.

## Fragile assumptions (ranked, from PLAN §12, inherited)

1. F-arm uninstrumented timing via the restoration proof (this lane's
   hard gate exists precisely for it).
2. Resolution vs between-group noise (mitigated by same-day squares,
   exclusivity, serialization; accepted unstable outcomes).
3. P0 likelihood (explicit P0 path; "maximal implementable" framing).
4. SQ request-boundary drain (evidenced by the four warmup-boundary
   captures, not proven device-side).
5. Probe/pilot selection bias (P1 only; winner's curse; exploratory only).
6. Cross-block adjacency and single-day drift.
7. OLMoE-only conclusions.
