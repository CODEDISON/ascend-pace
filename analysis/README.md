# analysis/ — benchmark harness lanes

Each subdirectory is a self-contained experiment lane: design doc
(`DESIGN.md`), launcher (`launch.py`), supervisor, per-rank worker,
prechecks (`precheck.py`), CPU-side tests (`test_cpu.py`), and a frozen
`source_manifest.json` of the exact sources that ran on device.

| Lane | Measures | Status |
|---|---|---|
| `atlas_barrier_microbench_v2/`, `v3/` | Empty SHMEM barrier cost across group sizes; pipeline saturation | ✅ complete (aggregate in `analysis_results/revision2_barrier_v3_campaign_*/`) |

Operating discipline (applies to every lane):

1. Source review and CPU tests pass before any device submission.
2. Two fresh prechecks (device/port/process) immediately before launch.
3. All failure evidence is preserved; no kills, no blind retries.
4. Post-run independent replay audits saved outputs against manifests.

To run a lane: `python3 launch.py` after filling its plan/manifest inputs —
see each lane's `DESIGN.md` for the exact invocation contract.
