# analysis/ — benchmark harness lanes

Each subdirectory is a self-contained experiment lane: design doc
(`DESIGN.md`), launcher (`launch.py`), supervisor, per-rank worker,
prechecks (`precheck.py`), CPU-side tests (`test_cpu.py`), and a frozen
`source_manifest.json` of the exact sources that ran on device.

| Lane | Measures | Status |
|---|---|---|
| `atlas_barrier_microbench_v2/`, `v3/` | Empty SHMEM barrier cost across group sizes; pipeline saturation | ✅ complete (aggregate in `analysis_results/revision2_barrier_v3_campaign_*/`) |
| `atlas_sparse_distribution_lane_20261105_v1/` | 17 captured-shape PACE vs native HCCL, forced SDMA | ✅ 3/6 groups; job3 hit a device-level aicpu limitation (see findings) |
| `atlas_optimized_hccl_lane_20261105_v1/` | 4-arm Williams design vs optimized HCCL | 📋 design + prechecks (frozen image vLLM broken; superseded by MC2 device data) |
| `atlas_memory_campaign_lane_20261105_v1/` | 45-job 3-lens memory profiling | 📋 design + prechecks |

Operating discipline (applies to every lane):

1. Source review and CPU tests pass before any device submission.
2. Two fresh prechecks (device/port/process) immediately before launch.
3. All failure evidence is preserved; no kills, no blind retries.
4. Post-run independent replay audits saved outputs against manifests.

To run a lane: `python3 launch.py` after filling its plan/manifest inputs —
see each lane's `DESIGN.md` for the exact invocation contract.
