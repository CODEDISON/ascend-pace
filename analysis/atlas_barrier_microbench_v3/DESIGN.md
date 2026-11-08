# Barrier/control-cost v3 (three arms, P16; prepared source, not yet reviewed or run)

Goal: close the three evidence gaps reviewer concern #13 left open after the v2
single-group empty-barrier pilot (P16 median max-rank 89.860 µs, 50 counts, one
process group): (1) multi-group stable estimation, (2) barrier increment under a
matched producer/consumer control, and (3) the isolated incremental barrier cost
inside a complete PACE call. All arms run on one Atlas 800T A3 node, P=16 ranks,
synchronous mode, fresh containerized process groups.

## Frozen identities (unchanged from prior lanes)

| Artifact | Identity |
|---|---|
| Runtime image | `sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f` |
| libshmem.so | `c778fb46fd6d4acdbba4f2138f1bfca6d85273d6608855753cb290b202fadb71` |
| libcashflex.so | `24963ee7a28bb8f397299f1842bd69bcea6f170c7cdc7f855dfe9a8c4e7753d9` |
| Queue-metadata helper | `0d7fde8e4eb1f0c8c312412cfc2bb6ee6f5df116ea48e2e27aae2feadef962be` |
| Device layout.json | `e844a5175ca3de3e4a54ef54723b76df6665607c4268ad29cc3ed541d7bf6154` |
| Admission guard (v3 group lane) | launch.py `b4d96c96…`, manifest `288fb691…` |

The worker reaches SHMEM only through `RTLD_NOLOAD` on the provider already
loaded by cashflex, exactly like v2. The measured barrier symbol is the same
`aclshmemx_barrier_all_on_stream` on each rank's current stream. Every group
performs the 2 GiB-class free-memory all-reduce admission (arm A/B: v2's fixed
2 GiB margin; arm C: heap + peak tensors + 1 GiB, the reverse-lane formula),
a scalar HCCL liveness all-reduce before and after the measured region, and
finalizes only when the clean flag is set; any fault leaves SHMEM unfinalized
with no retry, signal, reset or cleanup.

## Arm A — multi-group empty barrier (5 fresh groups)

Worker logic is v2's loop verbatim: 10 untimed warmups, 50 measured consecutive
`barrier(stream)` + same-stream host synchronize calls, host `CLOCK_MONOTONIC`
nanoseconds, per-call `[start_ns, end_ns]`. Each group is its own container,
process group and supervisor run; the launcher refuses to start group *k* until
group *k−1* (in the fixed order A1..A5, B1..B2, C1..C3) audited PASS, and every
group gets two complete 16-die admission preflights with its own port pair.
Per group: median of max-rank latency and shared-clock group envelope; across
groups: median/IQR/min/max of the five per-group medians plus a pooled 250-call
summary. Arm A1 is directly comparable with the v2 single-group P16 value
because the measured code, timing boundary and heap/slot layout are identical.

## Arm B — matched producer/consumer control (2 fresh groups)

There is **no host-side per-peer put in the verified ABI**: every
`aclshmem_put*` is a device-side kernel helper, and calling one would require a
newly written, unreviewed kernel plus raw peer-pointer arithmetic. Arm B
therefore uses the mandated fallback: the producer traffic is one small **real
PACE call** per round (`torch.ops.cashflex.hpa_alltoallv_device_splits`,
engine 1 forced SDMA, threshold 0, profile −1, compact 0 — the exact
reverse-order-lane call), which writes a deterministic 1024-byte int8 payload
to every peer's receive slot through the SHMEM write path. The task's 4 KiB
example cannot be realized under this ABI with v2's 32 KiB slots: the
device-split contract requires the *whole* input to fit one slot, and
16×4096 B = 64 KiB > 32 KiB; 1024 B/peer (16 KiB total, half a slot) is the
largest comfortable power-of-two payload, keeping the same 512 MiB heap /
32 KiB 2·P slot layout as v2.

Each round: producer PACE call → same-stream quiet (synchronize) → measured
region. Sub-arm **B1** times barrier enqueue + same-stream synchronize;
sub-arm **B2** times synchronize only. Five 10-round blocks per sub-arm with
alternating lead sub-arm (and per-group alternating first sub-arm) cancel
order and thermal drift; both sub-arms see identical generation parity mixes.
After the measured region of *every* round the rank verifies bit-exactly that
the peers' payloads arrived, its input is untouched, the split vectors are
unchanged and both PACE-side tail canaries are intact (fail-closed on any
mismatch). The PACE input/output live in explicit backing allocations with
at least 1024 tail bytes (row-width-converted tail rows, reverse-order-lane
precedent; the frozen operator rejects bare allocations — B1 device failure
42fe5bd9); the tails are canary-filled (−83 int8 / 173 BF16) and the payload
semantics of the 16×1024 B live rows are unchanged. The four tensor
arguments to the frozen operator (input, output and BOTH int32 split
vectors) each live in an explicit backing with ≥ 1024 tail bytes: payload
tensors by row-width-converted tail rows, split vectors as int32 backings
with 256 canary tail elements (reverse-order-lane verified form; the
operator rejects bare split allocations the same way — B1 device failure
15a6e1f33a5b); every tail is canary-filled (−83 int8 / 173 int32+BF16) and
verified untouched each round. 3 warmup rounds per sub-arm precede the blocks; 100 measured rounds per group. Per group:
per-sub-arm medians of max-rank time and the B1−B2 increment; per-block
medians expose residual drift.

Because the producer is quieted before the measured region, Arm B measures the
barrier increment *immediately after* real SHMEM write traffic — arrival-skew
and cache/queue state included — not a barrier concurrent with in-flight
traffic. The PACE call's internal ready-flag handshake is part of the producer
context and is identically present in both sub-arms.

## Arm C — incremental barrier inside a complete PACE call (3 fresh groups)

Shape: the frozen reverse-order-lane case `qwen-o1-call-038-dispatch`
(BF16, 4096-byte rows, per-rank 512 send rows, exact 16×16 split matrix copied
from the lane's dataset.json, file digest
`3954f89a…`), with the lane's 96 MiB slots / 4 GiB heap. Rationale: k=0 is
then directly comparable with the existing reverse-order PACE measurements on
the same shape, and the slope isolates only the appended barriers. Case 038
(dispatch) is chosen over 039 to match the lane's *first* case position and
halve runtime; the slope is a within-call comparison and does not need both
positions.

Each iteration times host API entry → PACE device-split call → *k* appended
`aclshmemx_barrier_all_on_stream` enqueues on the same stream →
`torch.npu.synchronize()`, i.e. the reverse-order lane's complete-call
boundary. The PACE input AND output live in explicit canary-filled backing
allocations with at least 1024 tail bytes (same frozen-operator contract and
reverse-lane precedent as arm B; tails verified untouched at every
checkpoint), and BOTH int32 split vectors use the same explicit-backing
form (256 canary tail elements). k ∈ {0, 1, 2}, 2 warmups + 30 measured
iterations per k per group
(96 PACE calls), scheduled as three 10-round blocks per k with alternating
k-order to cancel drift. **Correctness argument for the appended barriers:**
they are enqueued after the PACE call's compact kernel on the same stream, add
only synchronization, touch no data the call produced, and the two receive
planes are reused only at generation g+2 (source-verified handshake), so
appending barriers cannot change data semantics or plane safety. Bit-exact
output/input/split checks run at the first and last iteration of each k
(reverse-lane precedent of one initial and one final complete check).
Per group: medians per k, slopes (k1−k0) and (k2−k0)/2; across groups:
median/IQR of slopes. The per-barrier incremental cost inside a pipelined
complete call is reported as the k2 slope unless the two slopes disagree,
in which case both are reported with the discrepancy.

**Submission-queue budget (arms B and C):** the arms issue 106 and 96 PACE
calls per process. After SHMEM init and before any payload, each rank captures
quiescent queue metadata with the frozen helper (read-only `g_state` export +
`aclrtMemcpy` D2H of the ordinary workspace) and computes the cumulative SQE
bound for its exact planned call sequence, failing closed if any channel could
wrap its SQ ring or the 16-bit task counter. The ACL provider path is
discovered read-only from `/proc/self/maps` (already loaded by torch_npu),
hash-bound, cross-checked against the no-device precheck's identity, and the
captured metadata plus budget are persisted per rank and re-verified by the
launcher audit (channels re-parsed, budget recomputed, evidence rehashed).

## Launcher discipline

`launch.py` actions: `precheck_arm`/`precheck_audit` (no-device, no-network
container proving imports, RTLD_NOLOAD/dladdr provider identity, stream
ctypes ABI, PACE operator schema, case/plan identity, helper/layout digests
and the ACL provider identity), then per group `group_start`/`group_audit`,
then `aggregate`. Two complete 16-die admission preflights (guard lane:
docker-empty, all davinci nodes char devices, npu-smi all-empty/healthy,
fuser, per-group port bind) precede every container; the audit re-checks the
container contract (image, host network/ipc, exact device map, read-only
mounts plus the rw payload mount, env, natural zero exits), validates every
record against the frozen plan (counts, order, timers, fail-closed flags),
recomputes all metrics from raw nanosecond timestamps, re-parses both
persisted preflight snapshots, rehashes every evidence file including the
no-device precheck receipt, takes a read-only postflight snapshot, and only
then writes a completion receipt binding the manifest/review digests and
every evidence hash. Strict group order, no retries, no overwriting, unique
per-group ports, `aggregate` requires all ten receipts and re-verifies all
evidence symmetrically.

## Independent review binding

The review contract has a single source of truth: `plan.REVIEW_CONTRACT`
(status `APPROVED`, scope `barrier_v3_arms_abc_p16`,
`source_manifest_sha256` equal to the SHA-256 of `source_manifest.json`
itself, `reviewed_files` covering the full frozen source set plus
`source_manifest.json`, and `groups` exactly
`{"arm_a":["A1".."A5"],"arm_b":["B1","B2"],"arm_c":["C1".."C3"]}`).
`launch.verify()` consumes that same constant via
`plan.check_review_binding`, and the local test suite exercises the full
`verify()` path (synthetic tree injection) against one conforming review and
five non-conforming variants, so the launcher schema and the review schema
cannot drift apart. `review.json` is deliberately outside the source
manifest: a fresh review binding the regenerated manifest digest opens the
gate without any chicken-and-egg cycle.

## Intentional deviations, documented

- **P2-before-P16 staging is intentionally omitted.** v2 staged a P2 group
  before its P16 group because that lane placed the first-ever P16 SHMEM
  barrier cohort on the stage. v3 runs P16-only: the v2 P16 group already
  completed on this stage with the same libraries and image, and every v3
  group is independently gated by two complete 16-die admission preflights
  plus its own no-device precheck receipt, so a P2 repetition would add no
  protective power this lane lacks. The barrier call itself has already been
  exercised at P2 and P16 by v2.
- The ACL provider discovery accepts only mappings ending exactly
  `/libascendcl.so`. A versioned soname (e.g. `libascendcl.so.9`) or a
  `(deleted)` mapping yields zero providers and a hard fail. This
  conservatism is intentional fail-closed behavior tied to the pinned
  image's exact
  `/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libascendcl.so`
  mapping; if the image ever changes, precheck fails loudly instead of
  silently binding a different ACL build.
- Arm C's iteration plan does not vary per group (unlike arm B's per-group
  first-sub-arm alternation): block k-order alternation by block position
  already gives every k an identical even/odd generation mix, and one frozen
  sequence shared by all groups and validators is simpler to audit.

## What these arms cannot establish

- No barrier-algorithm latency independent of arrival skew, and no device-only
  kernel time: every number includes Python/ctypes enqueue, host
  synchronization and peer wait, on the host clock.
- Arm B is a *matched control after quieted* small real traffic, not a barrier
  overlapped with in-flight transfers, and not a producer/consumer visibility
  protocol proof beyond the per-round bit-exact checks.
- Arm C's slope assumes the per-barrier increment is locally linear in k for
  k ≤ 2 and measures *appended post-call* barriers, not barriers interleaved
  inside the operator's data plane; k=0 comparability with the reverse-order
  lane holds only for the same shape/config, and cross-lane timing drift
  (different day, fabric state) is not controlled.
- Five/two/three fresh groups are still descriptive replicates, not a
  confidence interval; IQRs are reported, not inferential statistics.
- Snapshots do not reserve the node; concurrent host/fabric contention by
  other users cannot be excluded, only observed and recorded.
- Payloads are synthetic (shake_256, BF16 exponent-safe like the original
  lane), not model data; no serving/model-time subtraction is justified.
- No failed P4 Alltoallv matrix is executed anywhere in this lane.

Before any device run this source needs an independent review
(`review.json`, status APPROVED, scope `barrier_v3_arms_abc_p16`, binding the
manifest digest and the exact arm/group plan), exactly like v2.
