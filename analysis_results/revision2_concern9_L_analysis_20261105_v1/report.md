# Revision2 Concern #9: Minimum Safe Per-Peer Slot Capacity L

Generated 2026-11-05T09:39:18+00:00 by `analysis.py` (pure local CPU; stdlib only; no device, remote or SSH operation). All numbers below are recomputed from the raw capture files listed in `inputs.sha256.md`; prior published numbers are used only as cross-check assertions inside the script (a mismatch aborts the run).

## 1. Data inventory (what is actually analyzed)

The "512 matrices" are the **512 global Alltoallv calls** stored in the frozen four-arm workload file `atlas_fair_trace_workloads_20261028_v1/workloads.json` (verified against its manifest): one fixed native B16/L1024 request per arm, P16/EP16, BF16 rows of 4,096 B; each call carries a 16x16 send-split matrix (bytes = rows x 4,096):

| arm | calls | forwards with Alltoallv | capture run |
|---|---|---|---|
| olmoe-o1 | 64 | 2 (prompt-only) | `atlas_model_split_capture_lane_20261028_v1/hccl-o1` |
| olmoe-o32 | 64 | first 2 of 33 (later 31 use MC2) | `atlas_model_split_capture_lane_20261028_v1/hccl-o32` |
| qwen-o1 | 192 | 2 (prompt-only) | `atlas_qwen_split_capture_20261028_v1/hccl-o1` |
| qwen-o32 | 192 | first 2 of 33 (later 31 use MC2) | `atlas_qwen_split_capture_20261028_v1/hccl-o32` |

So the **O1-only subset is 256 calls (OLMoE 64 + Qwen 192)**; the O32 arms add the first two Alltoallv forwards of the separate O32 runs, giving 512 total. The script rebuilds every matrix independently from the four raw `request_raw.json` worker event streams (rank coverage, per-rank call counts, send/receive transpose, layout consistency), then asserts equality with the frozen arm and the anchor maxima; every check passed.

Two supplementary native captures (`atlas_native_repeat_capture_20261028_v3`, same fixed OLMoE B16/L1024 batch) are replayed as **repeatability** evidence only: request-00 (64 calls) matches the frozen olmoe-o1 matrices 64/64; request-01 (128 calls) repeats the same batch twice and its forwards 2-3 equal forwards 0-1 64/64. Deterministic routing under one fixed batch is therefore established; this adds no new routing distribution.

## 2. Guard semantics (per paper main.tex and supplement)

- **Per-peer guard (host-list interface):** a call is rejected if any send or receive segment exceeds L. Because receive[j][i] = send[i][j], this equals: any matrix entry > L. The host lists include the self segment, so the binding max includes the diagonal (both variants reported).
- **Whole-input guard (historical device-split binding):** additionally requires every rank's whole input (row sum x 4,096 B) <= L. This dominates the per-peer bound because a row sum is at least as large as any of its entries.
- **Historical deployed semantics = union** of both checks (device-split required segment AND whole-input bounds); admission curves are reported separately for all three.

## 3. Recomputed maxima and cross-validation

| arm | calls | max peer segment incl self | max peer excl self | max rank whole input | anchor match |
|---|---|---|---|---|---|
| olmoe-o1 | 64 | 7,118,848 B (6.7891 MiB) | 7,118,848 B (6.7891 MiB) | 93,265,920 B (88.9453 MiB) | exact (OLMoE anchors) |
| olmoe-o32 | 64 | 7,135,232 B (6.8047 MiB) | 7,135,232 B (6.8047 MiB) | 93,290,496 B (88.9688 MiB) | exact (OLMoE anchors) |
| qwen-o1 | 192 | 10,297,344 B (9.8203 MiB) | 10,153,984 B (9.6836 MiB) | 117,772,288 B (112.3164 MiB) | exact (Qwen anchors) |
| qwen-o32 | 192 | 10,416,128 B (9.9336 MiB) | 10,141,696 B (9.6719 MiB) | 117,813,248 B (112.3555 MiB) | exact (Qwen anchors) |

All four arms reproduce the previously published O1 anchors exactly (OLMoE 7,118,848 / 93,265,920 B; Qwen 10,297,344 / 117,772,288 B) and the O32 initial-forward numbers (7,135,232 / 93,290,496 B and 10,416,128 / 117,813,248 B). Note the Qwen per-peer maximum is a **self segment** (10,297,344 B incl vs 10,153,984 B excl): under a slots-only-for-remote-peers reading the remote bound would be the smaller value, but the host-list guard checks all segments including self, so the larger value is the guard-relevant minimum.

## 4. Minimum safe L for the captured distributions, vs historical L

| model/arm | min safe L, per-peer guard | min safe L, whole-input guard (= union) | historical L | utilization at historical L (peer / whole) | headroom ratio (hist L / whole min) |
|---|---|---|---|---|---|
| olmoe-o1 | 6.7891 MiB | 88.9453 MiB | 96 MiB | 7.07% / 92.65% | 1.079x |
| olmoe-o32 | 6.8047 MiB | 88.9688 MiB | 96 MiB | 7.09% / 92.68% | 1.079x |
| qwen-o1 | 9.8203 MiB | 112.3164 MiB | 256 MiB | 3.84% / 43.87% | 2.279x |
| qwen-o32 | 9.9336 MiB | 112.3555 MiB | 256 MiB | 3.88% / 43.89% | 2.278x |

Key reading (captured B16/L1024/O1 requests):

- Under the **per-peer guard** alone, the observed minimum is only ~6.79 MiB (OLMoE) and ~9.82 MiB (Qwen): per-peer staging in PACE is nowhere near binding for these traces; the historical 96/256 MiB slots are 14.1x / 26.1x above the per-peer minimum.
- Under the **historical whole-input guard**, the minimums are 88.95 MiB (OLMoE) and 112.32 MiB (Qwen). OLMoE's historical 96 MiB is only 1.079x above this observed minimum (92.65% utilized), i.e., nearly tight; Qwen's 256 MiB is 2.28x (43.87% utilized).
- These are minima **for these captures only**. The whole-input requirement grows with scheduled tokens: the paper's B16/L1536 run rejected a 103,866,368 B whole input at 96 MiB, which is outside (and consistent with the length-scaling of) this capture set.

## 5. Demand distributions (bytes; nearest-rank percentiles)

Population = every (call, source rank, peer) segment for per-peer demand (incl self), and every (call, rank) row for whole input.

**olmoe-o1** (n segments = 16,384; n rank-inputs = 1,024)

| metric | P50 | P90 | P99 | P99.9 | max |
|---|---|---|---|---|---|
| per-peer segment (MiB) | 0.3281 | 2.7070 | 4.7305 | 6.2969 | 6.7891 |
| rank whole input (MiB) | 6.8086 | 30.2383 | 59.7695 | 86.0391 | 88.9453 |

**olmoe-o32** (n segments = 16,384; n rank-inputs = 1,024)

| metric | P50 | P90 | P99 | P99.9 | max |
|---|---|---|---|---|---|
| per-peer segment (MiB) | 0.3242 | 2.7109 | 4.7227 | 6.2148 | 6.8047 |
| rank whole input (MiB) | 6.8086 | 30.2500 | 59.7070 | 86.0156 | 88.9688 |

**qwen-o1** (n segments = 49,152; n rank-inputs = 3,072)

| metric | P50 | P90 | P99 | P99.9 | max |
|---|---|---|---|---|---|
| per-peer segment (MiB) | 0.2773 | 3.1016 | 5.6133 | 7.3555 | 9.8203 |
| rank whole input (MiB) | 4.7383 | 30.4336 | 75.0898 | 106.5195 | 112.3164 |

**qwen-o32** (n segments = 49,152; n rank-inputs = 3,072)

| metric | P50 | P90 | P99 | P99.9 | max |
|---|---|---|---|---|---|
| per-peer segment (MiB) | 0.2773 | 3.1133 | 5.6016 | 7.4023 | 9.9336 |
| rank whole input (MiB) | 4.7383 | 30.4570 | 75.1953 | 106.4336 | 112.3555 |

Per-forward structure (all four arms): forward 0 schedules 1,024 tokens (first chunk of request 1) and its maxima are ~11-14x smaller (peer ratios 13.7-14.4x, whole-input ratios 11.0-13.1x; e.g., olmoe-o1: 0.473 MiB peer / 6.809 MiB whole; qwen-o1: 0.719 / 10.250); forward 1 schedules the remaining 15,360 prompt tokens and produces every arm maximum. Demand tracks scheduled tokens roughly linearly (exact per-forward numbers in `summary.json` `forward_breakdown`).

## 6. Admission curves: fraction of captured calls rejected at capacity L

Counts are counterfactual guard rejections over the captured calls, not observed PACE events. Peer cell exceedance counts (call, rank, peer) segments above L. The whole-input guard rejection counts at 64/96 MiB exactly reproduce the supplement's published values (olmoe 9->0, qwen 40->9 rejected calls); every rejected call is a second-forward combine call, so the per-call sets here are identical to the published per-forward counts. `analysis.py` asserts this equality and aborts on any mismatch.

**olmoe-o1** (64 calls; 16,384 peer segments; 1,024 rank inputs)

| L (MiB) | calls rejected, peer guard | calls rejected, whole-input guard | calls rejected, union (historical) | peer segments > L | rank inputs > L |
|---|---|---|---|---|---|
| 8 | 0 (0.00%) | 32 (50.00%) | 32 (50.00%) | 0 (0.000%) | 507 (49.51%) |
| 16 | 0 (0.00%) | 32 (50.00%) | 32 (50.00%) | 0 (0.000%) | 472 (46.09%) |
| 32 | 0 (0.00%) | 16 (25.00%) | 16 (25.00%) | 0 (0.000%) | 93 (9.08%) |
| 48 | 0 (0.00%) | 14 (21.88%) | 14 (21.88%) | 0 (0.000%) | 31 (3.03%) |
| 64 | 0 (0.00%) | 9 (14.06%) | 9 (14.06%) | 0 (0.000%) | 9 (0.88%) |
| 88 | 0 (0.00%) | 1 (1.56%) | 1 (1.56%) | 0 (0.000%) | 1 (0.10%) |
| 96 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 100 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 104 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 112 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 116 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 120 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 128 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 160 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 192 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 256 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 320 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 360 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 384 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 448 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 512 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |

**olmoe-o32** (64 calls; 16,384 peer segments; 1,024 rank inputs)

| L (MiB) | calls rejected, peer guard | calls rejected, whole-input guard | calls rejected, union (historical) | peer segments > L | rank inputs > L |
|---|---|---|---|---|---|
| 8 | 0 (0.00%) | 32 (50.00%) | 32 (50.00%) | 0 (0.000%) | 507 (49.51%) |
| 16 | 0 (0.00%) | 32 (50.00%) | 32 (50.00%) | 0 (0.000%) | 472 (46.09%) |
| 32 | 0 (0.00%) | 16 (25.00%) | 16 (25.00%) | 0 (0.000%) | 93 (9.08%) |
| 48 | 0 (0.00%) | 14 (21.88%) | 14 (21.88%) | 0 (0.000%) | 31 (3.03%) |
| 64 | 0 (0.00%) | 9 (14.06%) | 9 (14.06%) | 0 (0.000%) | 9 (0.88%) |
| 88 | 0 (0.00%) | 1 (1.56%) | 1 (1.56%) | 0 (0.000%) | 1 (0.10%) |
| 96 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 100 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 104 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 112 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 116 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 120 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 128 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 160 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 192 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 256 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 320 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 360 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 384 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 448 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 512 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |

**qwen-o1** (192 calls; 49,152 peer segments; 3,072 rank inputs)

| L (MiB) | calls rejected, peer guard | calls rejected, whole-input guard | calls rejected, union (historical) | peer segments > L | rank inputs > L |
|---|---|---|---|---|---|
| 8 | 6 (3.12%) | 102 (53.12%) | 102 (53.12%) | 14 (0.028%) | 1,430 (46.55%) |
| 16 | 0 (0.00%) | 96 (50.00%) | 96 (50.00%) | 0 (0.000%) | 1,299 (42.29%) |
| 32 | 0 (0.00%) | 48 (25.00%) | 48 (25.00%) | 0 (0.000%) | 294 (9.57%) |
| 48 | 0 (0.00%) | 48 (25.00%) | 48 (25.00%) | 0 (0.000%) | 157 (5.11%) |
| 64 | 0 (0.00%) | 40 (20.83%) | 40 (20.83%) | 0 (0.000%) | 60 (1.95%) |
| 88 | 0 (0.00%) | 12 (6.25%) | 12 (6.25%) | 0 (0.000%) | 13 (0.42%) |
| 96 | 0 (0.00%) | 9 (4.69%) | 9 (4.69%) | 0 (0.000%) | 10 (0.33%) |
| 100 | 0 (0.00%) | 7 (3.65%) | 7 (3.65%) | 0 (0.000%) | 7 (0.23%) |
| 104 | 0 (0.00%) | 5 (2.60%) | 5 (2.60%) | 0 (0.000%) | 5 (0.16%) |
| 112 | 0 (0.00%) | 1 (0.52%) | 1 (0.52%) | 0 (0.000%) | 1 (0.03%) |
| 116 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 120 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 128 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 160 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 192 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 256 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 320 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 360 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 384 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 448 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 512 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |

**qwen-o32** (192 calls; 49,152 peer segments; 3,072 rank inputs)

| L (MiB) | calls rejected, peer guard | calls rejected, whole-input guard | calls rejected, union (historical) | peer segments > L | rank inputs > L |
|---|---|---|---|---|---|
| 8 | 6 (3.12%) | 102 (53.12%) | 102 (53.12%) | 14 (0.028%) | 1,431 (46.58%) |
| 16 | 0 (0.00%) | 96 (50.00%) | 96 (50.00%) | 0 (0.000%) | 1,300 (42.32%) |
| 32 | 0 (0.00%) | 48 (25.00%) | 48 (25.00%) | 0 (0.000%) | 294 (9.57%) |
| 48 | 0 (0.00%) | 48 (25.00%) | 48 (25.00%) | 0 (0.000%) | 158 (5.14%) |
| 64 | 0 (0.00%) | 40 (20.83%) | 40 (20.83%) | 0 (0.000%) | 60 (1.95%) |
| 88 | 0 (0.00%) | 12 (6.25%) | 12 (6.25%) | 0 (0.000%) | 13 (0.42%) |
| 96 | 0 (0.00%) | 9 (4.69%) | 9 (4.69%) | 0 (0.000%) | 10 (0.33%) |
| 100 | 0 (0.00%) | 7 (3.65%) | 7 (3.65%) | 0 (0.000%) | 7 (0.23%) |
| 104 | 0 (0.00%) | 5 (2.60%) | 5 (2.60%) | 0 (0.000%) | 5 (0.16%) |
| 112 | 0 (0.00%) | 1 (0.52%) | 1 (0.52%) | 0 (0.000%) | 1 (0.03%) |
| 116 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 120 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 128 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 160 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 192 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 256 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 320 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 360 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 384 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 448 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |
| 512 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.000%) | 0 (0.00%) |

Curve highlights: the per-peer guard admits everything down to ~7-10 MiB; the whole-input guard is the binding constraint and rejects OLMoE calls below 89 MiB and Qwen calls below 113 MiB; the historical 96/256 MiB configurations sit above every captured whole input (0 rejections). The full 1..512 MiB scan at 1 MiB resolution is in `summary.json` `arm_results[arm].admission_scan`.

## 7. Confidence boundaries (order statistics only)

| arm | rejections observed at historical L (whole / peer) | CP95 upper bound on per-call rejection prob. IF calls were exchangeable | CP95 upper bound on per-segment exceedance prob. |
|---|---|---|---|
| olmoe-o1 | 0 / 0 | 4.57% | 0.0183% |
| olmoe-o32 | 0 / 0 | 4.57% | 0.0183% |
| qwen-o1 | 0 / 0 | 1.55% | 0.0061% |
| qwen-o32 | 0 / 0 | 1.55% | 0.0061% |

These are one-sided Clopper-Pearson upper bounds given zero observed exceedances at L >= the arm minimum. **They are not production statements**: the captured calls come from fixed prompts, are not exchangeable random draws from a serving distribution, and the bounds condition on that unverified assumption. With n = 64/192 calls per arm the call-level bounds (4.57% / 1.55%) are weak; the segment-level bounds are small only because n is large, not because tail risk is measured.

## 8. Limitations (scope of what this analysis can claim)

- captured distributions are fixed-prompt B16/L1024 captures, not production workloads.
- O32 arms cover only the first two Alltoallv forwards; the later 31 forwards use MC2 and have no Alltoallv matrices.
- later captures cannot retro-justify the historical L choices (96 MiB was a prior config; 256 MiB followed a 96 MiB whole-input rejection; 512 MiB a prospective bound).
- GPT-OSS has no capture; its peak per-peer demand is unverifiable here.
- percentiles are descriptive of the sample; Clopper-Pearson bounds assume exchangeability, which fixed prompts violate.
- the whole-input guard requirement scales with scheduled tokens; the paper L1536 example (103,866,368 B rejected at 96 MiB) is outside this capture set.
- Zero-total calls (OLMoE O1: 2, Qwen O1: 4; O32 initial forwards: 2 and 4) are scheduler artifacts of the fixed batch, not capacity events; they still pass both guards.
- No admission curve here measures runtime benefit, fallback frequency or fallback cost; the deployed system has no validated transparent fallback, and rejection remains the documented behavior.
- GPT-OSS has no captured matrices; nothing in this analysis validates or falsifies its 512 MiB slot or the 360 MiB prospective bound.

## 9. Files

- `analysis.py` - this analysis (rerun: `python3 analysis.py` from any directory; writes the three other files).
- `summary.json` - machine-readable results: per-arm metrics, distributions, full 1..512 MiB admission scan, confidence bounds, input manifest, limitations.
- `inputs.sha256.md` - path/SHA-256/size/role of every input file read.
- `report.md` - this report.
