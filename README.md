<div align="center">

# Ascend-PACE

**Capacity-Bounded Alltoallv and MoE Integration on Ascend NPUs**

Experiment harnesses, device measurements, and reproduction drivers for the
Ascend-PACE project on a single Atlas 800T A3 node.

![Python](https://img.shields.io/badge/Python-3.11.15-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c)
![torch_npu](https://img.shields.io/badge/torch__npu-2.10.0-red)
![CANN](https://img.shields.io/badge/CANN-9.0.0-orange)
![vLLM](https://img.shields.io/badge/vLLM%2DAscend-0.20.2rc1-green)
![Hardware](https://img.shields.io/badge/Hardware-Atlas%20800T%20A3-purple)

</div>

## 📣 Highlights

- **Operator**: PACE — a capacity-bounded split-vector Alltoallv with fixed
  symmetric slots, double buffering, engine selection, and source-order
  compaction on Ascend NPUs.
- **Official-baseline audit (new)**: across **3 models × 2 MC2 algorithms ×
  batch sizes 1–32 × decode lengths 16–256**, the official Ascend MC2 MoE
  communication optimization **never outperforms native HCCL** (+4.2% to
  +7.9% slower, equal memory). Native HCCL is the strongest available
  baseline on this stack.
- **Barrier microbenchmark**: pipeline barrier cost saturates non-linearly
  (57.7 µs → 14.4 µs per barrier as group size grows).
- **Capacity analysis**: independent verification of minimum safe slot
  capacity L for captured OLMoE/Qwen routing shapes.

## 📊 Key Results

All device measurements on one Atlas 800T A3 node (16 dies, TP=16 unless
noted), inside `hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0`.

### MC2 official communication vs native HCCL (12-prompt batches, 5 runs/arm)

| Model | Arm | Median (ms) | Δ vs native |
|---|---|---:|---:|
| OLMoE-1B-7B | native HCCL | 741.2 | — |
| OLMoE-1B-7B | MC2 fullmesh | 783.8 | +5.7% |
| OLMoE-1B-7B | MC2 hierarchy | 799.9 | +7.9% |
| Qwen3-30B-A3B | native HCCL | 1,816.9 | — |
| Qwen3-30B-A3B | MC2 fullmesh | 1,905.8 | +4.9% |
| GPT-OSS-120B (TP=8¹) | native HCCL | 1,422.8 | — |
| GPT-OSS-120B (TP=8¹) | MC2 fullmesh | 1,483.2 | +4.2% |

¹ GPT-OSS has 8 KV heads; TP=16 cannot shard them, so both arms run at TP=8.

Memory after init is within 25 MB across all seven arms (~57.7–57.8 GB per
device). Full logs: [`analysis_results/mc2_baseline_v3/`](analysis_results/mc2_baseline_v3/),
[`analysis_results/mc2_v4/`](analysis_results/mc2_v4/),
[`analysis_results/mc2_gptoss8/`](analysis_results/mc2_gptoss8/).

### Batch-size sweep (OLMoE, min of 3 runs/config)

| Batch | 1 (short) | 1 (long) | 4 (short) | 4 (medium) | 12 | 20 | 32 |
|---|---|---|---|---|---|---|---|
| MC2 vs native | +18.4% | +5.0% | +3.7% | +4.0% | +5.0% | +5.5% | −0.8% |

No batch size shows an MC2 speedup; at best parity at 32.
Logs: [`analysis_results/mc2_sweep2/`](analysis_results/mc2_sweep2/).

### Sustained decoding (OLMoE, 12 prompts × {16, 64, 256} decode steps)

| Decode steps | 16 | 64 | 256 |
|---|---|---|---|
| MC2 vs native (min-of-3) | +1.2% | −0.3% | **+4.3%** |

At 256 steps every MC2 run is slower than every native run.
Logs: [`analysis_results/mc2_long/`](analysis_results/mc2_long/).

## 🛠️ Environment

| Component | Version |
|---|---|
| NPU | Atlas 800T A3, single node, 8 cards / 16 dies |
| Container image | `hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0` |
| vLLM | 0.20.2 (from `/vllm-workspace` source in image) |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| CANN | 9.0.0 |
| transformers | 5.9.0 |
| Python | 3.11.15 |

Third-party repositories and model checkpoints are pinned in
[THIRD_PARTY.md](THIRD_PARTY.md).

## 🚀 Quickstart (MC2 baseline reproduction)

```bash
# 1. Copy experiment scripts to your Ascend host workspace
scp -r experiments/mc2/ Ascend:/home/<user>/ascend-pace/

# 2. Place an OLMoE-1B-7B-0125-Instruct checkpoint under ./models/OLMoE

# 3. Launch the A/B driver (native vs MC2 fullmesh, TP=16)
ssh Ascend 'cd ~/ascend-pace && nohup bash mc2_v4_driver.sh > mc2.log 2>&1 &'

# 4. Watch for the completion marker and parse results
grep -E '"event": "(measured|arm_complete)"' results/mc2_v4/*/container.log
```

Expected output per arm (JSON lines on stdout):

```json
{"event": "measured", "run": 0, "duration_ms": 738.1, "tokens": 192, "tok_per_s": 260.1}
{"event": "arm_complete", "arm": "A", "init_consumed_mb": 57711, "steady_consumed_mb": 57835}
```

Scripts are model-generic — set `MODEL_PATH` / `WORLD_SIZE` / `MC2_ARM=A|B|C`
to reproduce the Qwen3 or GPT-OSS arms.

## 📁 Repository Structure

```
ascend-pace/
├── experiments/          # Ready-to-run vLLM/MC2 driver scripts (hpa image)
│   └── mc2/              #   baseline, batch sweep, sustained decoding, cross-model
├── analysis/             # Benchmark harness lanes (pre-device source review)
│   └── atlas_barrier_microbench_v3/       # multi-group barrier cost
├── analysis_results/     # Device measurement outputs (JSON event logs)
│   ├── mc2_baseline_v3/  # OLMoE native vs fullmesh
│   ├── mc2_v4/           # OLMoE hierarchy + Qwen3-30B-A3B
│   ├── mc2_gptoss8/      # GPT-OSS-120B at TP=8
│   ├── mc2_sweep2/       # batch-size sweep
│   ├── mc2_long/         # sustained decoding
│   └── ...               # barrier aggregate, L-capacity analysis, findings
├── requirements.txt      # Python dependencies (runtime is the pinned image)
└── THIRD_PARTY.md        # Pinned third-party repos, images, and models
```

## 🔬 Experiment Index

| Directory | What it measures | Status |
|---|---|---|
| `experiments/mc2/mc2_v3.py` | MC2 baseline A/B, OLMoE TP16 | ✅ complete |
| `experiments/mc2/mc2_v4.py` | Generic A/B/C arms (fullmesh/hierarchy), any model | ✅ complete |
| `experiments/mc2/mc2_sweep2.py` | Batch-size sweep bs1–bs32 | ✅ complete |
| `experiments/mc2/mc2_long.py` | Sustained decoding 16/64/256 steps | ✅ complete |
| `analysis/atlas_barrier_microbench_v3/` | Multi-group empty-barrier cost | ✅ complete |

## 🙏 Third-Party & Acknowledgments

Pinned versions, container digests, and model checkpoints are listed in
[THIRD_PARTY.md](THIRD_PARTY.md). Key upstream projects: **vLLM**,
**vLLM-Ascend**, **PyTorch / torch_npu**, **Ascend CANN**, and the
OLMoE / Qwen3 / GPT-OSS model releases.

## 📜 Citation

The accompanying manuscript is under review. If you use these harnesses or
measurements, cite the Ascend-PACE paper and this repository.

## ⚠️ Notes

- Device findings are specific to the pinned image/stack; absolute times are
  **not** comparable across different torch/CANN builds.
- All experiment arms use fresh engines per arm, one untimed warmup, and
  report descriptive medians or minima (see per-directory logs).
