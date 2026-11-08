# Revision2 Concern #9 (v2): Capacity Demand of Routing-Distribution Families

Generated 2026-11-05T14:34:35+00:00 by `analysis.py` (pure local CPU; stdlib only; no device, remote or SSH operation). This extends v1 (`revision2_concern9_L_analysis_20261105_v1`, untouched) which answered the same question **for the captured distributions only**; v2 fills the gap "how much capacity would *different* routing distributions need" with a deterministic synthetic family grid anchored to the captured token scale, and positions the captured distributions inside that spectrum. All input data is re-verified through the same hash chain and raw-event reconstruction as v1, and the v1 anchor maxima are asserted (mismatch aborts).

## 1. Question and headline answer

**Q:** how does the required per-peer slot capacity L (and the binding whole-input requirement) change across routing distributions at the captured scale?

**A:** the requirement decomposes into two nearly independent factors:

- **Whole-input (binding, historical union guard): depends only on per-rank totals** – i.e., token scale x rank imbalance – and is *exactly family-invariant* here by construction (families redistribute each rank's rows across peers; row sums are preserved). At 1x fwd1 scale it is 88.95 MiB (OLMoE base) / 112.32 MiB (Qwen base), reproducing v1's captured minima exactly.
- **Per-peer guard: depends on routing shape**, spanning a 16x range from the uniform-dense floor (max rank rows / 16) to the single-peer worst case (the full max rank rows). Captured routing sits at the benign end: its fwd1 per-peer maximum is only 1.22x (OLMoE) / 1.40x (Qwen) above the uniform-dense floor.

Because any matrix entry is bounded by its row sum, the per-peer requirement can **never exceed** the whole-input requirement: routing shape can never justify an L larger than the whole-input need; it only determines how much *smaller* than that L could be if the whole-input guard were dropped.

## 2. Data, verification and the token->row conversion

Same inventory as v1: four frozen native arms (olmoe/qwen x o1/o32, P16, BF16, row 4,096 B) re-verified end-to-end (manifest -> workloads.json digest -> raw `request_raw.json` digests -> independent matrix reconstruction -> equality with the frozen arms -> v1 anchor maxima). Every check passed; files in `inputs.sha256.md`.

**Conversion (asserted against every captured call):** each Alltoallv call carries `scheduled_tokens x 8` rows (top-8 expert expansion) of 4,096 B. Forward 1 schedules 15,360 tokens (B16/L1024 minus the 1,024-token first chunk), so every fwd1 call moves 122,880 rows = **480 MiB**, mean 7,680 rows (30 MiB) per rank. Dispatch calls distribute these perfectly evenly (every rank exactly 7,680 rows in the O1 arms); combine calls are rank-imbalanced, and the whole-input maxima always come from combine calls.

**Reference rank-total profiles** (the worst fwd1 combine call per O1 arm; this is the call that sets v1's whole-input maximum, asserted):

| base arm | worst call | rank totals (rows) | total | max rank | max-rank share | whole-input min @1x |
|---|---|---|---|---|---|---|
| olmoe-o1 | fwd1 combine | 22770 max of 16 rows (full profile in `summary.json`) | 122,880 rows (480 MiB) | 22,770 (88.9453 MiB) | 18.53% | 88.9453 MiB |
| qwen-o1 | fwd1 combine | 28753 max of 16 rows (full profile in `summary.json`) | 122,880 rows (480 MiB) | 28,753 (112.3164 MiB) | 23.40% | 112.3164 MiB |

O32 robustness: the O32 worst fwd1 whole-input maxima are 88.9688 / 112.3555 MiB (<0.04% above the O1 bases), so the O1 profiles are used as the canonical bases.

**Scales:** each family is generated at {0.5x, 1x, 2x} of the reference call: every rank total is scaled by m with the grand total preserved exactly (`round(T x m)`) by largest-remainder rounding; cells are integer row counts, row sums exact.

## 3. Family definitions (all deterministic; no RNG)

| family | construction (per source rank i) | analytic peakedness (16 x max share) | HHI |
|---|---|---|---|
| `uniform_dense` | uniform dense (all 16 peers equal) | 1.000 | 0.0625 |
| `sparse_p1_2` | uniform sparse p=1/2 (8 peers) | 2.000 | 0.1250 |
| `zipf_s0.5` | Zipf s=0.5 (16 peers, tilted) | 2.401 | 0.0761 |
| `sparse_p1_4` | uniform sparse p=1/4 (4 peers) | 4.000 | 0.2500 |
| `zipf_s1.0` | Zipf s=1.0 (16 peers, tilted) | 4.733 | 0.1386 |
| `zipf_s1.5` | Zipf s=1.5 (16 peers, strongly tilted) | 7.547 | 0.2670 |
| `diag_q0.5` | diagonal-dominant q=50% self | 8.000 | 0.2667 |
| `diag_q0.9` | diagonal-dominant q=90% self | 14.400 | 0.8107 |
| `sparse_p1_16` | uniform sparse p=1/16 (1 peer, rotated permutation) | 16.000 | 1.0000 |
| `single_burst` | single-peer burst (every rank -> peer 0 hotspot) | 16.000 | 1.0000 |

Uniform = uniform over the rank's *own* total (rows keep the captured imbalance); sparse supports are strided and off-self (rank i's peers are (i+1), (i+1+16/k), ...); Zipf weights w_t = (t+1)^-s / normalizer are assigned circulant with the hottest peer at i+1 (no artificial self bias); diagonal-dominant puts share q on the self cell and spreads 1-q uniformly over the other 15; single-peer burst sends every rank's whole input to peer 0 (hotspot). `sparse_p1_16` (rotated single peer, a permutation) and `single_burst` (all to the same peer) have identical send-side demand and differ only in the receive-side column maximum.

## 4. Distribution family -> capacity demand

### Base: qwen-o1 reference profile

**Scale 0.5x** - call total 240.00 MiB; whole-input min L (= historical union guard, family-invariant) **56.1602 MiB** = 240.00 MiB x 23.40% max-rank share; admits at historical L: 96: yes, 256: yes, 512: yes.

| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |
|---|---|---|---|---|
| `uniform_dense` | 3.5117 | 1.00x | 15.0234 | Y/Y/Y |
| `sparse_p1_2` | 7.0234 | 2.00x | 18.4297 | Y/Y/Y |
| `zipf_s0.5` | 8.4258 | 2.40x | 18.3711 | Y/Y/Y |
| `sparse_p1_4` | 14.0430 | 4.00x | 22.4219 | Y/Y/Y |
| `zipf_s1.0` | 16.6133 | 4.73x | 24.6094 | Y/Y/Y |
| `zipf_s1.5` | 26.4883 | 7.54x | 32.4727 | Y/Y/Y |
| `diag_q0.5` | 28.0820 | 8.00x | 34.2344 | Y/Y/Y |
| `diag_q0.9` | 50.5430 | 14.39x | 51.7930 | Y/Y/Y |
| `sparse_p1_16` | 56.1602 | 15.99x | 56.1602 | Y/Y/Y |
| `single_burst` | 56.1602 | 15.99x | 240.0000 | Y/Y/Y |

**Scale 1.0x** - call total 480.00 MiB; whole-input min L (= historical union guard, family-invariant) **112.3164 MiB** = 480.00 MiB x 23.40% max-rank share; admits at historical L: 96: NO, 256: yes, 512: yes.

| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |
|---|---|---|---|---|
| `uniform_dense` | 7.0234 | 1.00x | 30.0312 | Y/Y/Y |
| `sparse_p1_2` | 14.0430 | 2.00x | 36.8477 | Y/Y/Y |
| `zipf_s0.5` | 16.8555 | 2.40x | 36.7734 | Y/Y/Y |
| `sparse_p1_4` | 28.0820 | 4.00x | 44.8320 | Y/Y/Y |
| `zipf_s1.0` | 33.2227 | 4.73x | 49.2305 | Y/Y/Y |
| `zipf_s1.5` | 52.9766 | 7.54x | 64.9336 | Y/Y/Y |
| `diag_q0.5` | 56.1602 | 8.00x | 68.4414 | Y/Y/Y |
| `diag_q0.9` | 101.0859 | 14.39x | 103.5625 | N/Y/Y |
| `sparse_p1_16` | 112.3164 | 15.99x | 112.3164 | N/Y/Y |
| `single_burst` | 112.3164 | 15.99x | 480.0000 | N/Y/Y |

**Scale 2.0x** - call total 960.00 MiB; whole-input min L (= historical union guard, family-invariant) **224.6328 MiB** = 960.00 MiB x 23.40% max-rank share; admits at historical L: 96: NO, 256: yes, 512: yes.

| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |
|---|---|---|---|---|
| `uniform_dense` | 14.0430 | 1.00x | 60.0273 | Y/Y/Y |
| `sparse_p1_2` | 28.0820 | 2.00x | 73.6797 | Y/Y/Y |
| `zipf_s0.5` | 33.7070 | 2.40x | 73.5195 | Y/Y/Y |
| `sparse_p1_4` | 56.1602 | 4.00x | 89.6523 | Y/Y/Y |
| `zipf_s1.0` | 66.4453 | 4.73x | 98.4453 | Y/Y/Y |
| `zipf_s1.5` | 105.9570 | 7.55x | 129.8945 | N/Y/Y |
| `diag_q0.5` | 112.3164 | 8.00x | 136.8633 | N/Y/Y |
| `diag_q0.9` | 202.1719 | 14.40x | 207.1016 | N/Y/Y |
| `sparse_p1_16` | 224.6328 | 16.00x | 224.6328 | N/Y/Y |
| `single_burst` | 224.6328 | 16.00x | 960.0000 | N/Y/Y |

### Base: olmoe-o1 reference profile

**Scale 0.5x** - call total 240.00 MiB; whole-input min L (= historical union guard, family-invariant) **44.4727 MiB** = 240.00 MiB x 18.53% max-rank share; admits at historical L: 96: yes, 256: yes, 512: yes.

| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |
|---|---|---|---|---|
| `uniform_dense` | 2.7812 | 1.00x | 15.0273 | Y/Y/Y |
| `sparse_p1_2` | 5.5625 | 2.00x | 18.3398 | Y/Y/Y |
| `zipf_s0.5` | 6.6719 | 2.40x | 17.4609 | Y/Y/Y |
| `sparse_p1_4` | 11.1211 | 4.00x | 21.4219 | Y/Y/Y |
| `zipf_s1.0` | 13.1562 | 4.73x | 21.9102 | Y/Y/Y |
| `zipf_s1.5` | 20.9766 | 7.54x | 27.5352 | Y/Y/Y |
| `diag_q0.5` | 22.2383 | 8.00x | 28.7578 | Y/Y/Y |
| `diag_q0.9` | 40.0234 | 14.39x | 41.3359 | Y/Y/Y |
| `sparse_p1_16` | 44.4727 | 15.99x | 44.4727 | Y/Y/Y |
| `single_burst` | 44.4727 | 15.99x | 240.0000 | Y/Y/Y |

**Scale 1.0x** - call total 480.00 MiB; whole-input min L (= historical union guard, family-invariant) **88.9453 MiB** = 480.00 MiB x 18.53% max-rank share; admits at historical L: 96: yes, 256: yes, 512: yes.

| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |
|---|---|---|---|---|
| `uniform_dense` | 5.5625 | 1.00x | 30.0312 | Y/Y/Y |
| `sparse_p1_2` | 11.1211 | 2.00x | 36.6602 | Y/Y/Y |
| `zipf_s0.5` | 13.3477 | 2.40x | 34.9102 | Y/Y/Y |
| `sparse_p1_4` | 22.2383 | 4.00x | 42.8320 | Y/Y/Y |
| `zipf_s1.0` | 26.3086 | 4.73x | 43.8242 | Y/Y/Y |
| `zipf_s1.5` | 41.9531 | 7.54x | 55.0586 | Y/Y/Y |
| `diag_q0.5` | 44.4727 | 8.00x | 57.5156 | Y/Y/Y |
| `diag_q0.9` | 80.0508 | 14.39x | 82.6680 | Y/Y/Y |
| `sparse_p1_16` | 88.9453 | 15.99x | 88.9453 | Y/Y/Y |
| `single_burst` | 88.9453 | 15.99x | 480.0000 | Y/Y/Y |

**Scale 2.0x** - call total 960.00 MiB; whole-input min L (= historical union guard, family-invariant) **177.8906 MiB** = 960.00 MiB x 18.53% max-rank share; admits at historical L: 96: NO, 256: yes, 512: yes.

| family | per-peer min L (MiB) | ratio vs uniform floor | recv-side col max (MiB) | peer guard admits @96/256/512 |
|---|---|---|---|---|
| `uniform_dense` | 11.1211 | 1.00x | 60.0273 | Y/Y/Y |
| `sparse_p1_2` | 22.2383 | 2.00x | 73.3008 | Y/Y/Y |
| `zipf_s0.5` | 26.6953 | 2.40x | 69.8281 | Y/Y/Y |
| `sparse_p1_4` | 44.4727 | 4.00x | 85.6523 | Y/Y/Y |
| `zipf_s1.0` | 52.6172 | 4.73x | 87.6367 | Y/Y/Y |
| `zipf_s1.5` | 83.9062 | 7.54x | 110.1172 | Y/Y/Y |
| `diag_q0.5` | 88.9453 | 8.00x | 115.0273 | Y/Y/Y |
| `diag_q0.9` | 160.1016 | 14.40x | 165.3203 | N/Y/Y |
| `sparse_p1_16` | 177.8906 | 16.00x | 177.8906 | N/Y/Y |
| `single_burst` | 177.8906 | 16.00x | 960.0000 | N/Y/Y |

Reading the grid (Qwen base, 1x): per-peer min L spans **7.02 MiB** (uniform dense) to **112.32 MiB** (single-peer = the whole-input value), with Zipf s=1.0 at 33.2 MiB and 90% self-dominance at 101.1 MiB; OLMoE base spans 5.56-88.95 MiB. The per-peer requirement is capped by the whole-input requirement in every family. The receive-side column maximum is within 0.1% of call total/16 (30.03 MiB; largest-remainder tie bias favors low column indices) for uniform dense, moderately above it for the rotated sparse/Zipf/diagonal families (36.8-103.6 MiB at 1x Qwen), and equals the **full call total (480 MiB at 1x)** for the single-peer hotspot - relevant only under a receive-side whole-staging reading, which the historical (send-side) whole-input guard did not check.

## 5. Admission behavior on the L grid (family-shaped call populations)

Each family shape was also applied to **every captured call** of the base arm (64 OLMoE / 192 Qwen calls, fwd0+fwd1) at each scale. Because row sums are preserved, the whole-input/union rejection counts equal the captured ones at 1x - the replay asserts the v1/supplement published counts (olmoe 64 MiB: 9, 96 MiB: 0; qwen: 40, 9) and aborts on mismatch. Only the peer-guard counts move with family shape; per-call max-cell percentiles (1x scale):

| base | family | per-call max cell P50 / P90 / max (MiB) |
|---|---|---|
| qwen-o1 | `uniform_dense` | 0.6406 / 5.1133 / 7.0234 |
| qwen-o1 | `zipf_s0.5` | 1.5391 / 12.2773 / 16.8555 |
| qwen-o1 | `zipf_s1.0` | 3.0312 / 24.1992 / 33.2227 |
| qwen-o1 | `diag_q0.9` | 9.2266 / 73.6250 / 101.0859 |
| qwen-o1 | `single_burst` | 10.2500 / 81.8047 / 112.3164 |
| olmoe-o1 | `uniform_dense` | 0.4258 / 4.1992 / 5.5625 |
| olmoe-o1 | `zipf_s0.5` | 1.0234 / 10.0820 / 13.3477 |
| olmoe-o1 | `zipf_s1.0` | 2.0156 / 19.8750 / 26.3086 |
| olmoe-o1 | `diag_q0.9` | 6.1289 / 60.4688 / 80.0508 |
| olmoe-o1 | `single_burst` | 6.8086 / 67.1875 / 88.9453 |

Set-level per-peer minima equal the worst-call values of section 4 (asserted in the script). Full grids are in `summary.json` `family_replay_percentiles`.

**Break-even vs historical L (union guard, any family):** since the union requirement is family-invariant, the historical configurations break at a pure scale multiple:

| base | fwd1 tokens | breaks 96 MiB at | breaks 256 MiB at | breaks 512 MiB at |
|---|---|---|---|---|
| olmoe-o1 | 15,360 | 1.08x (16,578 tok) | 2.88x (44,209 tok) | 5.76x (88,417 tok) |
| qwen-o1 | 15,360 | 0.85x (13,129 tok) | 2.28x (35,010 tok) | 4.56x (70,019 tok) |

At 2x fwd1 scale (~30.7k scheduled tokens) every family still admits under 256/512 MiB; 512 MiB first breaks at ~4.6x (Qwen base, ~70k tokens) / ~5.8x (OLMoE base) regardless of routing shape. This bounds what the synthetic families can say about GPT-OSS's 512 MiB slot: at captured-like token scale no family - not even the single-peer hotspot - needs more than 512 MiB; only larger token scales do, and then for every family alike.

## 6. Where the captured distributions sit in the family spectrum

Row-share metrics over all captured fwd1 rows (peakedness = 16 x max share; HHI with uniform = 0.0625):

| arm x fwd1 | peakedness P50 / P90 / P99 / max | HHI P50 / max | support (of 16) | rank imbalance (max/mean) | global max cell |
|---|---|---|---|---|---|
| olmoe-o1 dispatch | 2.30 / 2.97 / 3.51 / 3.62 | 0.0825 / 0.1105 | 15-16 | 1.00-1.00 | 1,738 rows (remote cell, row peak 3.62) |
| olmoe-o1 combine | 1.56 / 2.33 / 3.14 / 3.36 | 0.0675 / 0.1154 | 15-16 | 1.50-2.96 | 1,738 rows (remote cell, row peak 1.26) |
| qwen-o1 dispatch | 2.67 / 3.58 / 4.27 / 5.24 | 0.0989 / 0.1573 | 14-16 | 1.00-1.00 | 2,514 rows (self cell, row peak 5.24) |
| qwen-o1 combine | 1.56 / 2.70 / 5.95 / 10.08 | 0.0678 / 0.4228 | 13-16 | 1.65-3.74 | 2,514 rows (self cell, row peak 1.61) |

Family ladder for comparison (analytic peakedness): uniform_dense 1.00 < sparse_p1_2 2.00 < zipf_s0.5 2.40 < sparse_p1_4 4.00 < zipf_s1.0 4.73 < zipf_s1.5 7.55 < diag_q0.5 8.00 < diag_q0.9 14.40 < sparse_p1_16 16.00 < single_burst 16.00.

**Positioning conclusions:**

- **Support is dense**: captured rows send to 13-16 of 16 peers - far from every sparse family; the routing is an "all-peers, mildly tilted" point.
- **Typical concentration is between uniform and Zipf(0.5)**: P50 peakedness 1.56 (combine) / 2.30-2.67 (dispatch), HHI 0.068-0.099 vs uniform 0.0625.
- **Tail rows reach Zipf(1.0)-equivalent**: P99 peakedness 3.1-5.9, max 3.4-3.6 (OLMoE) and 5.2-10.2 (Qwen, whose global max cell is a *self* cell in a balanced dispatch row - the only echo of the diagonal-dominant family in the captures).
- **But the per-peer maxima land in *large* rows with mild internal concentration** (OLMoE combine row peak 1.26; Qwen combine row peak 1.61), so the realized per-peer requirement is only 1.22x (OLMoE) / 1.40x (Qwen) the uniform-dense floor - i.e., captured routing sits at the *benign* end of the per-peer spectrum, much closer to uniform-dense than to any skewed family.
- **The binding whole-input requirement is set by rank imbalance, not peer skew**: captured rank totals are perfectly balanced in dispatch (1.00) but 1.50-2.96 / 1.65-3.74x imbalanced in combine (OLMoE / Qwen), which alone lifts the whole-input need from the rank-balanced 30 MiB floor to 88.95/112.32 MiB at 1x. Rank imbalance, not routing shape, is what makes the historical 96/256 MiB slots nearly-tight/moderately-utilized in v1.

## 7. Cross-reference to v1

| quantity | v1 (captured) | v2 re-derivation |
|---|---|---|
| OLMoE per-peer max (incl self) | 7,118,848 B = 6.7891 MiB | same anchor, asserted; = 1.22x uniform-dense floor 5.5625 MiB |
| Qwen per-peer max (incl self) | 10,297,344 B = 9.8203 MiB | same anchor, asserted; = 1.40x uniform-dense floor 7.0234 MiB (ceil-row caliber, as in the tables) |
| OLMoE whole-input max | 93,265,920 B = 88.9453 MiB | = uniform-dense 1x whole min (family-invariant) = 480 MiB x 18.53% max-rank share |
| Qwen whole-input max | 117,772,288 B = 112.3164 MiB | = uniform-dense 1x whole min (family-invariant) = 480 MiB x 23.40% max-rank share |
| v1 admission counts at 64/96 MiB | olmoe 9/0, qwen 40/9 | reproduced exactly by the family replay at 1x (asserted) |
| v1 utilization at historical L | olmoe 92.65%, qwen 43.87% (whole) | identical at 1x for every family (whole guard family-invariant) |

## 8. Limitations

- synthetic families are parameterized constructs anchored to the captured token scale; they are not production routing distributions and no claim is made that any of them occurs in serving.
- rank-total profiles are held at the captured worst fwd1 combine call per O1 arm (row sums preserved by construction), so whole-input results are family-invariant by design; only per-peer and receive-side results differ across families.
- this analysis cannot retro-justify the historical L choices (96 MiB was a prior config; 256 MiB followed a 96 MiB whole-input rejection; 512 MiB a prospective bound).
- single-node fixed-prompt B16/L1024 token scale only: fwd1 = 15,360 scheduled tokens; the paper L1536 example (103,866,368 B whole input) is outside this set.
- GPT-OSS still has no capture; nothing here validates its 512 MiB slot or the 360 MiB prospective bound.
- largest-remainder rounding moves individual cells by <1 row (4 KiB); all totals are preserved exactly.
- the hotspot (single-peer burst) receive-side column max is a staging upper bound under a receive-whole-input reading, which the historical guard (send-side whole-input) did not check.
- No admission result here measures runtime benefit, fallback frequency or fallback cost; rejection remains the documented deployed behavior.
- fwd0 rows include tiny single-peer artifacts (support 1, peakedness 16) of the 1,024-token first chunk; they are reported in `summary.json` but excluded from the positioning conclusions above, which use fwd1.

## 9. Files

- `analysis.py` - this analysis (rerun: `python3 analysis.py`; writes the three other files; deterministic apart from the timestamp).
- `summary.json` - machine-readable results: conversion, reference profiles, full family x scale grid, replay percentiles, captured positioning, break-evens, limitations, input manifest.
- `inputs.sha256.md` - path/SHA-256/size/role of every input file read.
- `report.md` - this report.
