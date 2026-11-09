# Third-Party Dependencies, Images, and Models

Everything device-side runs inside pinned container images on one Atlas
800T A3 node. This file lists the exact upstream artifacts so results are
reproducible.

## Container images

| Role | Image | Key contents |
|---|---|---|
| MC2 / vLLM experiments | `hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0` | vLLM 0.20.2 (`/vllm-workspace` source), torch 2.10.0+cpu, torch_npu 2.10.0, CANN 9.0.0, transformers 5.9.0, Python 3.11.15 |
| Original PACE operator measurements (frozen) | internal frozen image (see paper supplement) | PyTorch 2.10.0, torch-npu 2.10.0, CANN 9.1.0 |

Device passthrough used by every driver: `/dev/davinci0..15`,
`/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`, plus read-only
mounts of `/usr/local/Ascend/driver` and `/etc/ascend_install.info`.

## Upstream repositories

| Project | Version / commit | Role |
|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 0.20.2 | Serving engine (v1 engine in image) |
| [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | v0.20.2rc1 | Ascend platform plugin; `additional_config={"mc2_comm_alg": ...}` enters here |
| [Ascend/pytorch](https://gitee.com/ascend/pytorch) (torch_npu) | 2.10.0 | NPU backend; fused MoE + MC2 communication algorithms |
| [Ascend CANN Toolkit](https://www.hiascend.com/software/cann) | 9.0.0 (image) / 9.1.0 (frozen) | Runtime, HCCL, SDMA/MTE engines |

Notes:
- MC2 `fullmesh` / `hierarchy` are the two communication algorithms the
  pinned deployment exposes via `mc2_comm_alg`; no other algorithm was
  measured.
- The PACE operator itself (fixed symmetric slots, double buffering,
  engine selection, source-order compaction) is exercised by the analysis
  lanes under `analysis/`, built against the frozen image stack.

## Model checkpoints

| Model | Revision used | Parallelism |
|---|---|---|
| [allenai/OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) | 0125-Instruct | TP=16 |
| [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | hf snapshot `ad44e…` (2026-10-27) | TP=16 |
| [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | BF16 snapshot `ms82f41` (2026-08-14) | TP=8 (8 KV heads cannot divide TP=16) |

## Hardware

- One Atlas 800T A3 node: 8 cards / 16 dies, 64 GB HBM per device,
  host-side HCCL/SDMA interconnect.
- Other users share the node; drivers tolerate and disclose shared-device
  noise (e.g., occasional ~2.5 s stalls observed on both arms).

## Reproduction entry points

| Experiment | Script | Output dir |
|---|---|---|
| MC2 baseline A/B | `experiments/mc2/mc2_v3.py` | `results/mc2_v3/` |
| MC2 A/B/C, any model | `experiments/mc2/mc2_v4.py` (`MODEL_PATH`, `MC2_ARM`) | `results/mc2_v4/` |
| GPT-OSS at TP=8 | `experiments/mc2/` drivers with `WORLD_SIZE=8`, dies 0–7 | `results/mc2_gptoss8/` |
| Batch sweep bs1–32 | `experiments/mc2/mc2_sweep2.py` | `results/mc2_sweep2/` |
| Sustained decoding | `experiments/mc2/mc2_long.py` | `results/mc2_long/` |
