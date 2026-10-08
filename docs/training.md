# Training

## Environment

The training implementation depends on PyTorch with CUDA, PyTorch3D, `flash-attn`, and the packages in [`../requirements.txt`](../requirements.txt). The AMB3R setup used as a starting point documents Python 3.9, PyTorch 2.5.0 with CUDA 11.8, PyTorch3D 0.7.8, and `flash-attn` 2.7.3. Follow the installation instructions for those packages that match your CUDA and GPU environment; several are compiled extensions.

```bash
conda create -n se3r python=3.9 cmake=3.14.0 -y
conda activate se3r
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation
pip install flash-attn==2.7.3 --no-build-isolation
pip install -r requirements.txt
```

These are the inherited AMB3R environment pins and have not been freshly validated as a clean SE3R install. Record the exact tested versions and GPU/CUDA details before the public release.

## Checkpoints

Set `training.pretrained` in `configs/se3r.yaml` to the AMB3R checkpoint used by the experiment. The model loads the AMB3R weights with non-strict matching for the newly introduced MVSE modules. Do not publish a checkpoint until the AMB3R and VGGT weight terms have been checked for redistribution.

## Data

The paper reports an 11-dataset mixture totaling 2,000 samples per epoch, with 5–10 views per sample. The exact dataset quota is listed in [`datasets.md`](datasets.md). Obtain each dataset from its original source and follow its access and use terms. Do not copy source dataset files into this repository.

The released configuration uses a ScanNet validation split with 20 frames at resolution `(518, 392)`. The paper reports choosing the best checkpoint by `loss_refine_avg` on this validation set.

## Run

From the repository root:

```bash
python train.py --config configs/se3r.yaml
```

The current config sets seed 42, batch size 1, gradient accumulation 6, one warmup epoch, `bf16`, learning rate `1e-4`, minimum learning rate `1e-6`, weight decay `0.05`, and 30 epochs. The supplementary material also reports weight decay 0.0 for biases, AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, gradient clipping 1.0, and one training worker; those values are fixed in the training implementation or parser defaults. The supplementary config had a `se_init` field, but the final `AMB3R_SE_Combined` constructor and YAML loader do not consume it, so it is omitted here. Confirm all settings against the exact run that produced the paper checkpoint before claiming full reproduction.

The script records experiment metadata and writes timestamped output directories under the `output_dir` configured in `configs/se3r.yaml`. Update local absolute paths and ensure the output directory has sufficient disk space before starting.

## AMB3R training documentation

The upstream [AMB3R training guide](https://github.com/HengyiWang/amb3r/blob/main/docs/train.md) describes AMB3R's general data setup. It reports 5–16 frames per sample; SE3R's supplementary material reports 5–10 views. Use this document for upstream context, and use the SE3R-specific recipe and configuration here for the SE3R experiment.
