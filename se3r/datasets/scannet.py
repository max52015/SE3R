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


class Scannet(BaseManyViewDataset):
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

    # RMVD benchmark evaluation scenes (ScanNet val split subset).
    # These must be excluded from training to prevent data leakage.
    RMVD_EVAL_SCENES = {
        'scene0697_02', 'scene0671_00', 'scene0666_00', 'scene0672_00',
        'scene0699_00', 'scene0685_01', 'scene0673_01', 'scene0686_00',
        'scene0673_05', 'scene0667_00', 'scene0694_01', 'scene0694_00',
        'scene0700_01', 'scene0693_00', 'scene0681_00', 'scene0679_01',
        'scene0664_01', 'scene0665_01', 'scene0706_00', 'scene0664_02',
        'scene0696_02', 'scene0693_01', 'scene0701_02', 'scene0704_01',
        'scene0674_00', 'scene0678_01', 'scene0670_00', 'scene0701_00',
        'scene0667_01', 'scene0664_00', 'scene0678_00', 'scene0697_00',
        'scene0683_00', 'scene0688_00', 'scene0698_00', 'scene0705_00',
        'scene0691_00', 'scene0702_02', 'scene0673_00', 'scene0677_01',
        'scene0676_01', 'scene0673_04', 'scene0687_00', 'scene0678_02',
        'scene0696_01', 'scene0689_00', 'scene0697_01', 'scene0673_02',
        'scene0672_01', 'scene0685_02', 'scene0700_02', 'scene0677_00',
        'scene0671_01', 'scene0696_00', 'scene0697_03', 'scene0693_02',
        'scene0676_00', 'scene0685_00', 'scene0700_00', 'scene0705_01',
        'scene0670_01', 'scene0679_00',
    }

    def _load_split_file(self, base_dir, split_name):
        """Load scene list from an official ScanNet split file (e.g. scannetv2_train.txt)."""
        split_file = osp.join(base_dir, f"scannetv2_{split_name}.txt")
        if not osp.exists(split_file):
            return None
        with open(split_file, "r") as f:
            scenes = {line.strip() for line in f if line.strip()}
        return scenes

    def load_all_scenes(self, base_dir):

        self.folder = {"train": "scans", "val": "scans", "test": "scans_test"}[
            self.split
        ]

        if self.test_id is None:

            all_scenes = [
                d
                for d in os.listdir(osp.join(base_dir, self.folder))
                if os.path.isdir(os.path.join(base_dir, self.folder, d))
            ]

            if self.split == "train":
                # Filter by official train split file
                train_scenes = self._load_split_file(base_dir, "train")
                if train_scenes is not None:
                    before = len(all_scenes)
                    all_scenes = [s for s in all_scenes if s in train_scenes]
                    print(f"[Scannet] Filtered by scannetv2_train.txt: "
                          f"{before} -> {len(all_scenes)} scenes")

                # Exclude RMVD benchmark evaluation scenes
                before = len(all_scenes)
                all_scenes = [s for s in all_scenes if s not in self.RMVD_EVAL_SCENES]
                excluded = before - len(all_scenes)
                if excluded > 0:
                    print(f"[Scannet] Excluded {excluded} RMVD eval scenes, "
                          f"remaining: {len(all_scenes)} scenes")

            self.scene_list = all_scenes
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]

            print(f"Test_id: {self.test_id}")

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):

        scene_id = self.scene_list[idx // self.num_seq]

        # Load metadata
        intri_path = osp.join(
            self.ROOT, self.folder, scene_id, "intrinsic/intrinsic_depth.txt"
        )
        intri = np.loadtxt(intri_path).astype(np.float32)[:3, :3]

        # Load image data
        image_path = osp.join(self.ROOT, self.folder, scene_id, "color")
        num_files = len(os.listdir(image_path))

        pose_path = osp.join(self.ROOT, self.folder, scene_id, "poses.npy")

        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)

        else:
            pose_all = []

            for i in range(num_files):
                posepath = osp.join(
                    self.ROOT, self.folder, scene_id, "pose", f"{i}.txt"
                )
                camera_pose = np.loadtxt(posepath).astype(np.float32)
                pose_all.append(camera_pose)
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)

        # ranking: NxN
        rank_path = osp.join(self.ROOT, self.folder, scene_id, "ranking.npy")
        if os.path.exists(rank_path):
            ranking = np.load(rank_path)
        else:
            ranking, dist = compute_ranking(
                pose_all, lambda_t=1.0, normalize=True, batched=True
            )

            print(ranking.shape)
            np.save(rank_path, ranking[:, :512])
            print(f"Ranking saved to {rank_path}")

        while True:
            ref_idx = np.random.randint(0, num_files)
            k = 256
            topk = ranking[ref_idx][1 : k + 1]
            replace = False if len(topk) >= num_frames - 1 else True
            support_ids = np.random.choice(topk, size=num_frames - 1, replace=replace)

            imgs_idxs = [ref_idx] + support_ids.tolist()
            poses = pose_all[imgs_idxs]

            if np.isfinite(poses).all():
                break

        imgs_idxs = deque(imgs_idxs)

        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            # Load image data
            impath = osp.join(
                self.ROOT, self.folder, scene_id, "color", f"{im_idx}.jpg"
            )
            depthpath = osp.join(
                self.ROOT, self.folder, scene_id, "depth", f"{im_idx}.png"
            )

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            # depthmap[depthmap>5.] = 0.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            # camera_pose = np.loadtxt(posepath).astype(np.float32)
            camera_pose = pose_all[im_idx].astype(np.float32)

            rgb_ori = rgb_image.copy()
            depth_ori = depthmap.copy()
            intri_ori = intri.copy()

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath
            )

            # Check if the image is valid
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__() - 1)
                        return self._get_views(new_idx, resolution, num_frames, rng)
                    return self._get_views(
                        idx, resolution, num_frames, rng, attempts + 1
                    )

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset="scannet",
                label=osp.join(scene_id, str(im_idx)),
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

    dataset = Scannet(split="train", ROOT="./data/scannet", resolution=224, num_seq=1)
