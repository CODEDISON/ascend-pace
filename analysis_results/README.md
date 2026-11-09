# analysis_results/ — device measurement outputs

Pulled `container.log` / aggregate files from device campaigns. Each JSON
event line is emitted by the corresponding harness under `experiments/` or
`analysis/`.

## MC2 communication campaigns (hpa image, 2026-11-08)

| Directory | Campaign | Key numbers |
|---|---|---|
| `mc2_baseline_v3/` | OLMoE TP16: native vs MC2 fullmesh | 741.2 vs 783.8 ms median (+5.7%) |
| `mc2_v4/` | OLMoE hierarchy arm + Qwen3-30B-A3B A/B | hierarchy 799.9 ms (+7.9%); Qwen3 1816.9 vs 1905.8 ms (+4.9%) |
| `mc2_gptoss8/` | GPT-OSS-120B at TP=8 | 1422.8 vs 1483.2 ms (+4.2%) |
| `mc2_sweep2/` | Batch sweep bs1–bs32 (min-of-3) | no MC2 win; parity at bs32 (−0.8%) |
| `mc2_long/` | Sustained decoding 16/64/256 steps | +4.3% at 256 steps, strict run separation |

## Earlier campaigns (frozen image stack)

| Directory | Content |
|---|---|
| `revision2_barrier_v3_campaign_20261105_v1/final_20261107/` | 10-group barrier aggregate (57.7 µs → 14.4 µs pipeline saturation) |
| `revision2_concern9_L_analysis_*` | Minimum safe slot capacity L for captured routing shapes (independent v1+v2) |
| `sparse_lane_campaign_20261105_v1/` | Device finding: aicpu 507018 on syn-diag-512 (job3 limitation) |
| `vllm_npu_incompatibility_finding.md` | Frozen-image vLLM+NPU engine-init failure diagnosis |

## Parsing

```bash
grep -E '"event": "(measured|config_result|arm_complete)"' <dir>/*/container.log
```

Statistics convention across reports: medians for 5-run baselines,
min-of-3 for sweep configs (shared-device outliers hit both arms).
