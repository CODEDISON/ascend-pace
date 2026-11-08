# Joint memory campaign lane (WP-A #11 + WP-B #12; prepared source, not yet reviewed or run)

Status: **prepared source spec**. Pure local CPU output; no device, remote or
SSH operation. Before any device execution this source needs an independent
`review.json` (status `APPROVED`, scope `memory_campaign_wp_a_wp_b_p16`)
binding the source-manifest digest, the frozen `arms.json` digest, the full
45-job campaign order and the conditional gates — validated against the ONE
shared contract constant in `plan.py` (`plan.validate_review`, consumed by
`launch.verify` through the `launch.verify_review_binding` alias; the five
refusal variants are tested). Every prior transcription-defect class is
prevented by construction: pins were extracted programmatically from the
barrier-v3/#8 sources and `FrozenPinTests` freezes them verbatim (IMAGE
71-char, four provider digests, GUARD_*, L2 memory_snapshot identity).

Implements `analysis/atlas_memory_campaign_plan_20261105_v1/` (PLAN.md +
arms.json bound by SHA-256; referenced, never copied; every recomputable
arithmetic fact is recomputed at bind time).

## Measurement boundaries (read first)

- **Three lenses, each stated with every number; peaks are always "sampled
  maximum (lower bound)"; lenses overlap and are never summed.**
  - **L1 host poller** (launch-side, 1 Hz `npu-smi` per die, never inside the
    container — both saved lanes log `can not use command: npu-smi info`):
    physical HBM watermark including every process on the die. The poller
    thread samples until the job container stops; sampler failure ends the
    stream with a recorded gap, never the job (FA2 fallback = fenceposts +
    L1-with-gaps).
  - **L2 fenceposts** (role-aware, R2-B2): T4 is steady-phase-complete,
    **WP-A/fill/pilot only** (PLAN §1.4); capacity/throughput jobs audit
    exactly the sequence they emit (T0/T1/T2[a|b]/T3/T5). Frozen
    `memory_snapshot.capture` (digest-pinned,
  pinned API-source hashes) at T0/T1/T2(a/b)/T3/T4/T5 plus per-request
    captures; device-scope free/total + torch allocator current/peak. Torch
    allocator **excludes** SHMEM/ACL/CANN direct allocations. No
    peak-counter reset exists anywhere in this path (a reset would destroy
    the full-run peak; reset-window numbers stay a separate published
    observation).
  - **L3 background sampler** (0.5–2 Hz per-rank `mem_get_info` thread):
    adopted only if the A0 pilot shows median request-time delta ≤ 2% AND
    fencepost free deltas ≤ 64 MiB (`plan.l3_adoption`, both directions
    reported). Pilot data is quarantined from formal aggregates regardless.
- **Reconciliation**: L1 vs L2 paired at shared timestamps (±2 s), bias
  reported per lens pair per die (min/median/max), never averaged away.
- **Arms differ in diagnostic instrumentation**: native-vs-PACE deltas are
  descriptive, never a causal transport-only memory cost.
- One fresh process group per job (45 preregistered slots, strict order
  gating, no cross-job watermark pooling, no retries/signals; duration
  budgets recorded per job with a >125%-of-budget warning in every
  completion).

## WP-A (concern #11)

Full-run init→steady→teardown watermark timeline per model×arm: A0 pilot
(OLMoE A/P, L3 on/off passes inside each pilot job — see deviations),
A1/A2 + A1r/A2r (OLMoE repeats, also WP-B anchors), A3/A4 (Qwen3),
A5/A6 (GPT-OSS DP2×TP8). Steady phase = 8 further sequential B16/L1024/O1
requests after the frozen first request. **A6 KV floor rule** (infeasibility is a preregistered legitimate outcome,
  PLAN/FA6): the worker's infeasible early-exit (fence_T0 + a6_infeasible
  only) audits on exactly that evidence subset and earns a PASS receipt
  carrying the infeasible floor; the C7 conditional gate then refuses every
  C7 job start on `kv_floor.feasible=False` (they are trim-droppable at
  level ≥1), and the aggregate lists the outcome — never a deadlock. A6
  requests
`floor(A5_free_T1 − 16.5 heap − 2 guard)` GiB; < 2 GiB ⇒ the arm is reported
**infeasible at Table 2 heap on 64 GB/die** — a reported outcome that also
gates every C7 job (`conditional_gate` refuses C7* without an A6-feasible
receipt). Headline rows carry the SHMEM-init free delta (T2a→T2b) and the
native-vs-PACE sampled-max-used delta, both labeled as above.

## WP-B (concern #12) and the C0 probes

- **C0-1** (cgroup probe): a job container started with `--memory 2g` (far
  below any planned NPU allocation) performs a deliberate 16 GiB device
  over-allocation (a single bf16 tensor; the record states exactly that). **Fail-closed acceptance**: a cap counts ONLY if the
  over-allocation actually fails AND the error is recorded
  (`plan.c0_outcome`); expected outcome is no enforcement (cgroup memory
  governs host RAM; NPU HBM is device memory outside it). **C0-2**: any
  documented NPU-side per-process cap knob must fail closed under deliberate
  over-allocation; none is documented in the pinned local docs (Ascend KG
  unreachable, `[未核实]`). **The aggregate's equal-memory wording is decided
  by the C0 evidence alone** (`plan.equal_memory_wording`): default Tier-1
  "matched configured budget with measured-peak/headroom alignment", never
  "equal physical HBM ceiling" unless a probe demonstrated enforcement.
- **Tier-1 legs**: (i) config alignment native KV = PACE KV + heap (16 GiB
  nominal sum, enforced at arms-bind time for B-A/B-B/B-C/B-D); (ii)
  measured-peak alignment within ε = max(1 GiB, 2%·64 GiB), one
  preregistered adjustment round, else wording degrades to "aligned
  configuration, measured peaks reported separately" (`plan.peak_alignment`);
  (iii) fill-probe headroom equality at floor F = 2 GiB in 256 MiB steps,
  filler is not KV (`plan.fill_probe_result`).
- **Capacity frontier**: ladder L ∈ {1024, 4096}, C doubling from 16 capped
  2048/512, then binary refinement depth ≤ 3 (4 doubling + 3 binary probes
  per (arm, L), `plan.next_capacity_cell`); exhaustion = insufficient-KV
  logs on ≥100 consecutive scheduler iterations AND completed < submitted
  after a 3× median-completion wall cap; censoring rule fixed
  (`plan.classify_probe`: served/exhausted/fault/censored — failures
  preserved, never retried). Predicted C* (512 vs 384 at L=4096) is a
  cross-check only; the measurement tests whether PACE reaches its
  configured blocks under the heap's real footprint and the boundary
  behavior.
- **Throughput frontier**: one fresh container per arm sweeping
  C ∈ {16,64,256} × L ∈ {1024,4096} × O ∈ {1,32} + one fresh-group confirm
  at each arm's best admitted C; metrics (request/token throughput, TTFT
  distribution, per-request latency) paired across arms at identical cells
  only.
- **B-D conditional**: valid only under per-peer admission semantics; the
  #9 whole-input minimum (88.95 MiB) exceeds its 16 MiB slot, so under the
  reviewed `whole_input`/`union` semantics the arm is a projection-only
  table row — `conditional_gate` refuses C3BD* unless the review carries
  `device_split_guard_semantics == 'per_peer'` (the deployed-guard source
  check is review-carried exactly like the #8 lane, because this lane sees
  only the built .so).
- **#9 linkage stated in every output**: at B16/L1024 the per-rank KV
  residency ≈ 129 MiB ≈ 3% of the 4 GiB heap; the measurable tradeoff is the
  capacity ceiling, not per-request footprint.

## Launcher discipline

`launch.py`: `precheck_arm`/`precheck_audit` (no-device container: imports,
RTLD_NOLOAD + dladdr fname-only provider identity, namespaced PACE schema,
L2 module identity, arms/campaign binding, review-contract echo), then per
job `job_start`/`job_audit` (two complete 16-die admission preflights
re-validated at audit; container contract incl. the C0-1 `--memory` variant;
L1 host poller stream persisted per job; strict prior-job gating; trim
monotonicity; exact-ns preflight ≤ preflight ≤ start < finish; postflight),
then `aggregate` (all receipts rehashed — container log/inspect, both
preflight snapshots, postflight, payload files, and the precheck receipt;
wording decided by C0). Recursive source-tree coverage + launcher
self-digest (barrier-v3 round-2 form). The trimming ladder
(`--trim N`, monotone, one job per step, 19 steps: drop C7 → C6 → L=1024
confirms → A1r/A2r → B-C) is an explicit recorded downgrade.

## Interpretations/deviations from PLAN (round-2: declared AND implemented)

1. **A0 pilot reading (re-implemented per review)**: two pilot jobs (A0a arm
   A, A0p arm P) each run the request sequence in two real passes inside the
   one fresh group — pass 1 with the L3 sampler instantiated and running,
   pass 2 with it off — then call `plan.l3_adoption` on the measured passes
   (`pilot.json`); non-pilot jobs run the sampler only under `MEMCAMP_L3=1`.
   Pilot data is quarantined in every aggregate regardless.
2. **Capacity job slots (derivation now enforced)**: the 15 C2 jobs are
   preregistered slots (4 search + 1 confirm per arm); `job_start` DERIVES
   each search cell from the same arm's prior probe receipts via
   `plan.next_capacity_cell` (free CLI cells are refused outside
   confirm/sweep slots), the confirm slot runs at L=1024 with the
   L=4096-derived C* (capped 2048), and `job_audit` re-derives from the
   receipts. Adaptive cells, deterministic gating, never re-run.
3. **B-C block note**: arms.json's "~13,055" is prose; the authoritative
   arithmetic is `floor(12.75 GiB / 1 MiB) = 13,056` blocks.
4. **WP-A budget cross-check tolerance**: PLAN §2.4's "≈106 min" covers
   pilot 16 + formal 88 = 104 with rounding; the arms-bind check tolerates
   0–4 min of that declared slack.
5. **Guard-semantics source check review-carried** (#8-lane mechanism).
   Under the reviewed `whole_input` semantics the C3BD jobs are recorded as
   **PROJECTION_ONLY receipts** (skip-and-record) that satisfy the ordering
   gate and appear in the aggregate as projection rows — never run, never
   blocking, never poisoning.
6. **Model engine construction** follows the frozen budget-matched/
   first-memory templates (TP16/EP16, seed 2027, block 128, explicit
   `kv_cache_memory_bytes`, `num_gpu_blocks_override=None`; GPT-OSS
   DP2×TP8 via `data_parallel_size=2/tensor_parallel_size=8`); only the
   preregistered switches vary, validated by `plan.llm_config` before any
   engine is constructed. PACE heap/slot are plumbed through
   `HPA_SHMEM_HEAP_BYTES`/`HPA_SHMEM_SLOT_BYTES` (the only sizing mechanism
   in this stack, model-memory-pilot precedent) and echoed into
   `engine_runtime.json` as actually-configured bytes.
7. **SQ gate conditional inapplicability** (review-accepted): model-driven
   call sequences have no static exact list to budget; the optional post-hoc
   quiescent capture (A8) is declined and documented here.
8. **FA4 gate**: every engine job's first request passes the token gate —
   reference match when `/workspace/reference/<model>_tokens.json` exists,
   else intra-job determinism (two identical temperature-0 generates must
   match token-for-token); a mismatch fails the job closed with evidence
   preserved. The gate record (`token_gate.json`) is required audit
   evidence.
9. **C0-2 is a desk determination, not a device probe** (A3): no NPU-side
   per-process HBM cap is documented in the pinned local docs (Ascend KG
   unreachable, `[未核实]`); any future documented knob requires a
   fail-closed over-allocation job to count. C0-1 records the true single
   16 GiB over-allocation (A2).
10. **Launcher usage order**: `job_start` (docker -d) → `job_l1` (BLOCKING
    host npu-smi poller until container exit; stream persisted on the main
    thread) → `job_audit` (requires and validates the L1 stream, the
    role-aware fencepost sequence, the token gate, the engine runtime
    record, the A6 floor lift and the log-lifted KV streak; re-derives every
    capacity cell from the same receipts the start used). Projection-only
    job dirs are a graceful no-op in both `job_l1` and `job_audit`.
11. **Free-cell policy (R2-A7)**: capacity search/confirm slots REJECT free
    CLI cells (never silently override); throughput-sweep cells are the
    frozen grid; C5 throughput-confirm cells stay operator-supplied from the
    arm's sweep results (recorded as such) — the only free cells in the
    campaign, by design.
12. **Model-aware WP-B splits (R2-A8)**: the Qwen3 extension arms use the
    coherent A4 split (PACE 7.5 GiB KV + 8.5 GiB heap, slot 256 MiB — the
    OLMoE B-B 96 MiB slot is below the #9 Qwen whole-input minimum
    112.32 MiB and would fault at admission under the reviewed semantics);
    the GPT-OSS PACE arms use the A6 feasible floor (16.5 GiB heap, slot
    512 MiB) resolved from A5's measured T1 free.
13. **Warm pass (R2-A9, declared)**: every non-pilot engine job (including
    capacity/throughput probes) runs an 8-request B16/L1024/O1 pass (the
    steady-phase machinery) before its role probe — undeclared device time
    no more; it warms the engine identically across arms and its fencepost
    marks are recorded per request.
14. **Extension capacity jobs (R3-B1)**: the four role-`capacity` jobs
    (C6QA/C6QB/C7GA/C7GB) run the full probe path (capacity_probe.json with
    the actual KV-exhaustion behavior and the engine block count is
    hard-gated) at a DECLARED preregistered cell — L=4096/C=16, the ladder
    floor, per model (`plan.EXTENSION_CAPACITY_CELLS`) — because an
    extension arm's cell is not derivable from the OLMoE frontier
    (cross-model incomparability) and free CLI cells remain rejected.
    Frontier histories, c* and confirm inputs are grouped by (model, arm);
    the aggregate computes the Tier-1 leg-(ii) peak alignment per model.
    A capacity arm whose boundary settles before a search slot (first-probe
    failure below the ladder, or the whole ladder served) records a
    `SEARCH_COMPLETE` not-run receipt that satisfies the ordering gate and
    is single-listed in the aggregate (R3-A1). The A6-infeasible audit
    branch requires the L1 stream for evidence uniformity (R3-A4).
15. **Pilot pairing (R2-A5)**: the A0 adoption rule pairs the two POST-PASS
    fencepost captures (P1END vs P2END, one shared boundary one pass apart);
    the earlier pre-pass/post-pass mix was conservative but not the
    preregistered pairing. Capacity arms additionally require a non-None
    engine `num_gpu_blocks` at audit (R2-A6); other roles record absence
    honestly.

## What this campaign cannot support (preregistered)

No true instantaneous physical peak; no causal transport-only memory cost;
no "25% usable KV loss" (configured-block differences, measured C* and
watermark deltas only); no "equal-memory" claim without proven enforcement;
no async-path/multi-node/steady-state-safety/failure-recovery conclusions;
no GPT-OSS per-peer L validation (no captured matrices); no general
serving-distribution statements (fixed prompts, one node, O ∈ {1,32}).

## Source set

`FILES`/`REVIEWED_FILES` in `plan.py` (DESIGN.md, plan.py, worker.py,
supervisor.py, precheck.py, launch.py, test_cpu.py, make_manifest.py,
wp_a_profile.py, wp_b_frontier.py, observed_npu_info_fixture.json +
source_manifest.json) — the two analysis scripts are frozen with the
manifest before the first formal job (PLAN §1.6/§7, A7). The npu-smi fixture
is the observed all-idle 8-card/16-die snapshot shared verbatim with the
barrier-v3/#8 lanes (identical SHA-256), used only by host-side parser
tests. Regenerate the manifest with `make_manifest.py` after any intentional
edit; the launcher pins its digest.
