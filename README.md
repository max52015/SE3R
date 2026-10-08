# SE3R: Multi-View-Aware Squeeze-and-Excitation for Parameter-Efficient 3D Reconstruction
### I-Han Cho, Shih-Yao Su, [Ming-Ching Chang](https://www.albany.edu/faculty/mchang2/), and [Jhih-Ciang Wu](https://jhih-ciang.github.io/)

SE3R adapts the frozen AMB3R front end with Multi-View-Aware Squeeze-and-Excitation (MVSE) modules. The released training configuration follows the implementation described in the SE3R supplementary material.

> Under review


## Installation

Create the Conda environment and install the pinned dependencies:

```bash
conda create -n se3r python=3.9 cmake=3.14.0 -y
conda activate se3r
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation
pip install flash-attn==2.7.3 --no-build-isolation
pip install -r requirements.txt
```

The training script also needs the datasets and an AMB3R checkpoint at the paths set in `configs/se3r.yaml`. Dataset sources and reported sampling quotas are listed in [`docs/datasets.md`](docs/datasets.md).

## Training

From this directory, set the paths in `configs/se3r.yaml` for your local AMB3R checkpoint and datasets, then run:

```bash
python train.py --config configs/se3r.yaml
```

The configuration uses the AMB3R checkpoint to initialize the model. Only the newly inserted MVSE modules are trainable. The training script saves timestamped checkpoints and experiment metadata below the configured output directory.

## Evaluation

The model implements `run_amb3r_benchmark(frames)`. See [`eval/README.md`](eval/README.md) and [`docs/benchmark.md`](docs/benchmark.md) for connecting it to the upstream AMB3R benchmark and for the benchmark citation.

## Research data

This repository does not contain copies of the source datasets. Dataset access links and sampling information are listed in [`docs/datasets.md`](docs/datasets.md).
## Acknowledgements

SE3R builds on [AMB3R](https://github.com/HengyiWang/amb3r) and uses modified VGGT code together with components from DUSt3R, CroCo, and MoGe. Their source links and known license information are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). We thank the authors and maintainers of these projects and of the datasets used in this work.
