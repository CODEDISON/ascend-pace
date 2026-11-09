# experiments/mc2 — vLLM-Ascend MC2 comparison drivers

Ready-to-run scripts comparing **native HCCL** against the official
**MC2 MoE communication** (`mc2_comm_alg = fullmesh | hierarchy`) inside
`hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0`.

## Scripts

| Script | Purpose | Arms |
|---|---|---|
| `mc2_v3.py` | Baseline: OLMoE TP=16, 12 prompts × 5 measured runs | A native / B fullmesh |
| `mc2_v4.py` | Model-generic baseline (`MODEL_PATH`, `WORLD_SIZE` env) | A native / B fullmesh / C hierarchy |
| `mc2_sweep2.py` | Batch-size sweep bs1–bs32, 3 runs per config | A / B |
| `mc2_long.py` | Sustained decoding, 16/64/256 decode steps, 3 runs per config | A / B |
| `mc2_v4_driver.sh` | Sequenced A→C→(Qwen3 A,B) driver with device passthrough | — |
| `mc2_sweep2_driver.sh`, `mc2_long_driver.sh`, `mc2_gptoss8_driver.sh` | Single-campaign drivers | — |

## Usage

Scripts run **inside** the pinned image (all deps preinstalled):

```bash
docker run ... hpa/vllm-ascend:v0.20.2rc1-a3-transformers5.9.0 \
    python3 -u /experiment.py          # env: MC2_ARM, WORLD_SIZE, MODEL_PATH
```

Use the provided `*_driver.sh` wrappers — they mount all 16 dies, the model
directory, and wait for completion while archiving `container.log`.

## Output format

One JSON event per line on stdout:

```json
{"event": "arm_start", "arm": "A", "model": "/workspace/model", "world": 16}
{"event": "engine_init_ms", "value": 41903.4}
{"event": "measured", "run": 0, "duration_ms": 738.1, "tokens": 192, "tok_per_s": 260.1}
{"event": "arm_complete", "arm": "A", "init_consumed_mb": 57711, "steady_consumed_mb": 57835}
```

## Gotchas learned on this stack

- Do **not** pass `additional_config=None` (pydantic rejects it); omit the
  key or pass `{}`.
- GPT-OSS-120B cannot run TP=16 (8 KV heads); use TP=8 on dies 0–7.
- Occasional shared-device stalls (~2.5 s, and one 8.4 s outlier) hit both
  arms — prefer min-of-N or median and keep raw runs.
- First run after engine init is slowest in both arms; scripts do an
  untimed warmup per arm/config before timing.
