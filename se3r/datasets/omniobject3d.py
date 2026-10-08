import os
import json
import cv2
import re
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from ..tools.utils import threshold_depth_map
from ..tools.pose_dist import compute_ranking


def extract_number(filename):
    """Extract the leading integer from a filename for sorting."""
    match = re.search(r"\d+", filename)
    if match:
        return int(match.group())
    return 0


class OmniObject3D(BaseManyViewDataset):
    """
    OmniObject3D dataset loader for AMB3R training.

    Expects the preprocessed directory structure:
        ROOT/
          scale.json                (top-level, per-scene scale factors)
          {scene_id}/
            rgb/{frame}.png
            depth/{frame}.npy
            cam/{frame}.npz         (keys: intrinsics, pose)
    """

    def __init__(
        self,
        num_seq=1,
        test_id=None,
        full_video=False,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every

        # Load scale.json
        scale_path = osp.join(ROOT, "scale.json")
        assert osp.exists(scale_path), f"Missing {scale_path}"
        with open(scale_path, "r") as f:
            self.scales = json.load(f)

        # load all scenes
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is None:
            self.scene_list = sorted(
                [
                    d
                    for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                    and os.path.isdir(os.path.join(base_dir, d, "rgb"))
                ]
            )
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
            print(f"Test_id: {self.test_id}")

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, scene_id)

        rgb_dir = osp.join(scene_dir, "rgb")
        depth_dir = osp.join(scene_dir, "depth")
        cam_dir = osp.join(scene_dir, "cam")

        # List all available frames (rgb is .png for OmniObject3D)
        basenames = sorted(
            [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")],
            key=extract_number,
        )
        num_files = len(basenames)

        # Get scale factor for this scene
        scale = self.scales.get(scene_id, 1.0)

        # Load intrinsics from the first frame
        cam0 = np.load(osp.join(cam_dir, basenames[0] + ".npz"))
        intri = cam0["intrinsics"].astype(np.float32)

        # Load all poses (cache as npy for speed)
        pose_path = osp.join(scene_dir, "poses.npy")
        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)
        else:
            pose_all = []
            for basename in basenames:
                cam_data = np.load(osp.join(cam_dir, basename + ".npz"))
                pose = cam_data["pose"].astype(np.float32)
                # Normalize translation by scale (consistent with CUT3R)
                pose[:3, 3] = pose[:3, 3] / scale / 1000.0
                pose_all.append(pose)
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)

        # Load or compute ranking (cache for speed)
        rank_path = osp.join(scene_dir, "ranking.npy")
        if os.path.exists(rank_path):
            ranking = np.load(rank_path)
        else:
            ranking, dist = compute_ranking(
                pose_all, lambda_t=1.0, normalize=True, batched=True
            )
            print(ranking.shape)
            np.save(rank_path, ranking[:, :512])
            print(f"Ranking saved to {rank_path}")

        # Sample frames using ranking-based selection
        while True:
            ref_idx = np.random.randint(0, num_files)
            k = 256
            topk = ranking[ref_idx][1 : k + 1]
            replace = False if len(topk) >= num_frames - 1 else True
            support_ids = np.random.choice(
                topk, size=num_frames - 1, replace=replace
            )

            imgs_idxs = [ref_idx] + support_ids.tolist()
            poses = pose_all[imgs_idxs]

            if np.isfinite(poses).all():
                break

        imgs_idxs = deque(imgs_idxs)

        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            basename = basenames[im_idx]

            # Load image (.png for OmniObject3D)
            impath = osp.join(rgb_dir, basename + ".png")
            rgb_image = imread_cv2(impath)

            # Load depth and normalize by scale
            depthpath = osp.join(depth_dir, basename + ".npy")
            depthmap = np.load(depthpath).astype(np.float32)
            depthmap = depthmap / scale / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            # Load camera pose (already normalized if cached)
            camera_pose = pose_all[im_idx].astype(np.float32)

            # Keep originals for the view dict
            rgb_ori = rgb_image.copy()
            depth_ori = depthmap.copy()
            intri_ori = intri.copy()

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath
            )

            # Validate
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__() - 1)
                        return self._get_views(
                            new_idx, resolution, num_frames, rng
                        )
                    return self._get_views(
                        idx, resolution, num_frames, rng, attempts + 1
                    )

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset="omniobject3d",
                label=osp.join(scene_id, basename),
                instance=osp.split(impath)[1],
                is_metric=False,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori,
            )

            views.append(dict_info)

        return views


if __name__ == "__main__":
    num_frames = 5
    print("loading dataset")

    dataset = OmniObject3D(
        split="train",
        ROOT="./data/omniobject3d/",
        resolution=224,
        num_seq=1,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scene_list)}")
