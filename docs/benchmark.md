# Benchmark protocol

The paper states that benchmark protocols follow AMB3R and that SE3R uses the same benchmark scripts. The model in this repository exposes `run_amb3r_benchmark(frames)` and returns the standard prediction keys used by that interface: `world_points`, `depth`, `pose`, and `pts3d_by_unprojection`.

Use the upstream [AMB3R benchmark documentation](https://github.com/HengyiWang/amb3r/tree/main/benchmark) for dataset preparation, split definitions, metrics, and evaluation commands. Cite AMB3R as the benchmark implementation and cite each underlying dataset separately.

This repository includes a SE3R model adapter, not a copy of the full AMB3R benchmark. The upstream benchmark currently needs its model loader to construct SE3R and load the SE3R checkpoint. See [`/eval/README.md`](../eval/README.md).
