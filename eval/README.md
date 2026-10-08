# SE3R benchmark adapter

`se3r_adapter.py` loads the trained `AMB3R_SE_Combined` model and exposes the AMB3R benchmark forward call. The model's `run_amb3r_benchmark(frames)` method returns world points, depth, pose, and unprojected point maps.

The full evaluation datasets, preprocessing, metrics, and benchmark scripts remain in the upstream [AMB3R benchmark repository](https://github.com/HengyiWang/amb3r/tree/main/benchmark). Clone a specific AMB3R revision and use its evaluator. Its model registry currently does not know the `se3r` model name, so add a loader entry that calls:

```python
from eval.se3r_adapter import load_se3r

model = load_se3r(checkpoint_path)
```