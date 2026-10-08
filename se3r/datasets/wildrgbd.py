import os
import json
import cv2
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from ..tools.utils import threshold_depth_map
from ..tools.pose_dist import compute_ranking


class WildRGBD(BaseManyViewDataset):
    """
    WildRGBD dataset loader for AMB3R training.

    Expects the preprocessed directory structure:
        ROOT/
          selected_seqs_train.json   (top-level)
          selected_seqs_test.json    (top-level)
          {category}/
            scenes/
              {scene_id}/
                rgb/{frame:05d}.jpg
                depth/{frame:05d}.png         (uint16, mm -> / 1000 = meters)
                masks/{frame:05d}.png
                metadata/{frame:05d}.npz      (keys: camera_intrinsics, camera_pose)
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

        # load all scenes
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is not None:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
            print(f"Test_id: {self.test_id}")
            return

        # Load selected sequences from JSON
        split_file = f"selected_seqs_{self.split}.json"
        json_path = osp.join(base_dir, split_file)
        assert osp.exists(json_path), f"Missing {json_path}"

        with open(json_path, "r") as f:
            selected = json.load(f)

        # Format: {category: {scenes/scene_XXX: [frame_ids]}}
        # The key already contains the "scenes/" prefix
        self.scene_list = []
        self.scene_frames = {}
        for category, instances in selected.items():
            if not isinstance(instances, dict) or len(instances) == 0:
                continue
            for scene_rel_path, frame_ids in instances.items():
                if not frame_ids:
                    continue
                # scene_rel_path is e.g. "scenes/scene_000"
                scene_key = (category, scene_rel_path)
                scene_dir = osp.join(base_dir, category, scene_rel_path)
                if osp.isdir(scene_dir):
                    self.scene_list.append(scene_key)
                    self.scene_frames[scene_key] = frame_ids

        # Remove problematic scene if present (following CUT3R)
        bad_key = ("box", "scenes/scene_257")
        if bad_key in self.scene_frames:
            self.scene_list.remove(bad_key)
            del self.scene_frames[bad_key]

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_key = self.scene_list[idx // self.num_seq]
        category, scene_rel_path = scene_key
        scene_dir = osp.join(self.ROOT, category, scene_rel_path)

        frame_ids = self.scene_frames[scene_key]
        num_files = len(frame_ids)

        # Load all poses (cache as npy for speed)
        pose_path = osp.join(scene_dir, "poses.npy")
        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)
        else:
            pose_all = []
            for fid in frame_ids:
                meta = np.load(
                    osp.join(scene_dir, "metadata", f"{fid:05d}.npz")
                )
                pose_all.append(
                    meta["camera_pose"].astype(np.float32)
                )
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)

        # Load intrinsics from the first frame
        meta0 = np.load(
            osp.join(scene_dir, "metadata", f"{frame_ids[0]:05d}.npz")
        )
        intri = meta0["camera_intrinsics"].astype(np.float32)

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
            fid = frame_ids[im_idx]

            # Load image
            impath = osp.join(scene_dir, "rgb", f"{fid:05d}.jpg")
            rgb_image = imread_cv2(impath)

            # Load depth (png, uint16, mm)
            depthpath = osp.join(scene_dir, "depth", f"{fid:05d}.png")
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000.0
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
                dataset="wildrgbd",
                label=osp.join(category, scene_rel_path, str(fid)),
                instance=f"{fid:05d}.jpg",
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

    dataset = WildRGBD(
        split="train",
        ROOT="./data/wildrgbd/",
        resolution=224,
        num_seq=1,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scene_list)}")
