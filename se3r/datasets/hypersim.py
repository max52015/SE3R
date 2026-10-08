import os
import cv2
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from ..tools.utils import threshold_depth_map
from ..tools.pose_dist import compute_ranking


class Hypersim(BaseManyViewDataset):
    """
    Hypersim dataset loader for AMB3R training.

    Expects the preprocessed directory structure:
        ROOT/
          {scene}/
            {cam_folder}/
              {frame}_rgb.png
              {frame}_depth.npy    (float32, meters)
              {frame}_cam.npz      (keys: intrinsics, pose)
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

        # load all scenes (each cam_folder is a sequence)
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is None:
            self.scene_list = []
            all_scenes = sorted(
                [
                    d
                    for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                ]
            )
            for scene in all_scenes:
                scene_path = os.path.join(base_dir, scene)
                cam_folders = sorted(
                    [
                        d
                        for d in os.listdir(scene_path)
                        if os.path.isdir(os.path.join(scene_path, d))
                        and len(os.listdir(os.path.join(scene_path, d))) > 0
                    ]
                )
                for cam_folder in cam_folders:
                    # Store as "scene/cam_folder" path
                    self.scene_list.append(osp.join(scene, cam_folder))
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
            print(f"Test_id: {self.test_id}")

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, scene_id)

        # List all available frames by finding _rgb.png files
        rgb_files = sorted(
            [f for f in os.listdir(scene_dir) if f.endswith("_rgb.png")]
        )
        basenames = [f.replace("_rgb.png", "") for f in rgb_files]
        num_files = len(basenames)

        if num_files < 2:
            # Not enough frames, try another scene
            new_idx = rng.integers(0, self.__len__() - 1)
            return self._get_views(new_idx, resolution, num_frames, rng)

        # Load all poses (cache as npy for speed)
        pose_path = osp.join(scene_dir, "poses.npy")
        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)
        else:
            pose_all = []
            for basename in basenames:
                cam_data = np.load(
                    osp.join(scene_dir, basename + "_cam.npz")
                )
                pose_all.append(cam_data["pose"].astype(np.float32))
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)

        # Load intrinsics from the first frame (constant across sequence)
        cam0 = np.load(osp.join(scene_dir, basenames[0] + "_cam.npz"))
        intri = cam0["intrinsics"].astype(np.float32)

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

            # Load image
            impath = osp.join(scene_dir, basename + "_rgb.png")
            rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)

            # Load depth (float32 npy, already in meters)
            depthpath = osp.join(scene_dir, basename + "_depth.npy")
            depthmap = np.load(depthpath).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            # Load camera pose
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
                dataset="hypersim",
                label=osp.join(scene_id, basename),
                instance=osp.split(impath)[1],
                is_metric=True,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori,
            )

            views.append(dict_info)

        return views


if __name__ == "__main__":
    num_frames = 5
    print("loading dataset")

    dataset = Hypersim(
        split="train",
        ROOT="./data/hypersim/",
        resolution=224,
        num_seq=1,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scene_list)}")
