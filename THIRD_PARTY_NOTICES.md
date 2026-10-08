# Third-party code and attribution

This repository contains copied or modified third-party code. The table records the source and license information found during the initial repository assembly. Confirm the exact upstream revisions and preserve the matching license/notice files before public release.

| Component | Source | License / release note |
|---|---|---|
| AMB3R-derived model and training code | [HengyiWang/amb3r](https://github.com/HengyiWang/amb3r) | No root license was present in the inspected AMB3R checkout. Citation alone does not establish permission to redistribute the copied/modified code. Confirm terms with the AMB3R authors before publication. |
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | The supplementary snapshot modifies `models/aggregator.py` and includes `models/aggregator_se.py`. The snapshot has no license file or pinned source commit. Add the exact license agreement corresponding to the used VGGT revision before redistribution. |
| CroCo | [naver/croco](https://github.com/naver/croco) | The copied tree includes its `LICENSE` and `NOTICE`. The upstream license is CC BY-NC-SA 4.0; review its non-commercial and share-alike terms. |
| MoGe | [microsoft/MoGe](https://github.com/microsoft/MoGe) | The copied tree includes its `LICENSE`. Preserve it and confirm the revision-specific terms. |
| DUSt3R utilities | [naver/dust3r](https://github.com/naver/dust3r) | Copied from the local AMB3R third-party tree because SE3R dataset loaders import DUSt3R image and dataset utilities. The local copy did not include its upstream license/notice files; add the matching files and pin the revision before redistribution. Upstream currently describes its code as CC BY-NC-SA 4.0. |

SE3R also depends on third-party datasets and pretrained checkpoints. Their access and use terms are separate from these source-code notices. See [`docs/datasets.md`](docs/datasets.md) and [`docs/training.md`](docs/training.md).

No project-wide license is declared in this draft. Do not treat the repository as granting a license to components whose terms are missing or separately restricted.
