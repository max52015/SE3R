# Datasets

SE3R uses third-party datasets obtained under their respective access and license terms. This repository does not redistribute the original data.

## Training mixture

The supplementary material reports the following training mixture: 2,000 samples per epoch and 5–10 views per sample.

| Group | Dataset | Samples per epoch | Link |
|---|---|---:|---|
| Indoor | ScanNet | 175 | [GitHub](https://github.com/ScanNet/ScanNet) |
| Indoor | ScanNet++ | 175 | [GitHub](https://github.com/scannetpp/scannetpp) |
| Indoor | Project Aria Digital Twin | 175 | [Dataset page](https://www.projectaria.com/datasets/adt/) |
| Indoor | Hypersim | 175 | [GitHub](https://github.com/apple/ml-hypersim) |
| Object-centric | WildRGBD | 150 | [GitHub](https://github.com/wildrgbd/wildrgbd) |
| Object-centric | OmniObject3D | 150 | [GitHub](https://github.com/omniobject3d/OmniObject3D) |
| Driving | Waymo Open Dataset | 300 | [GitHub](https://github.com/waymo-research/waymo-open-dataset) |
| Driving | Virtual KITTI 2 | 300 | [Dataset page](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) |
| Outdoor | MapFree | 133 | [GitHub](https://github.com/nianticlabs/map-free-reloc) |
| Outdoor | GTASfM / Flow-Motion-Depth | 133 | [GitHub](https://github.com/HKUST-Aerial-Robotics/Flow-Motion-Depth) |
| Outdoor | MVS-Synth | 134 | [Dataset page](https://phuang17.github.io/DeepMVS/mvs-synth.html) |
| **Total** |  | **2,000** |  |

The reported validation set is the ScanNet test split, using 20 frames at `(518, 392)` resolution. Confirm the exact split and dataset release used by the experiment and avoid exposing any restricted test data or labels.

## Evaluation datasets

The supplementary material reports these evaluation datasets and tasks:

- Multi-view 3D reconstruction: ETH3D, DTU, and 7-Scenes. ETH3D and DTU use RMVD image tuples.
- Multi-view depth and metric-scale multi-view depth: KITTI, ScanNet, ETH3D, DTU, and Tanks and Temples.
- Monocular depth: NYUv2, KITTI, ETH3D, ScanNet, and DIODE.
- Camera pose: RealEstate10K and CO3Dv2.

The links above point to the dataset pages supplied for this release. Add the exact protocol, split, resolution, and sample/frame count for each evaluation dataset. Cite the dataset publications in the article's references separately from these access links. The AMB3R benchmark guide documents benchmark protocols and should also be cited separately.
