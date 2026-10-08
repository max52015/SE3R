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


class ScanNetpp(BaseManyViewDataset):
    """
    ScanNet++ dataset loader for AMB3R training.

    Expects the preprocessed directory structure:
        ROOT/
          {scene_id}/
            images/{basename}.jpg
            depth/{basename}.png       (uint16, mm -> / 1000 = meters)
            new_scene_metadata.npz     (keys: images, trajectories, intrinsics)
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
        if self.test_id is None:
            self.scene_list = sorted(
                [
                    d
                    for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))
                    and os.path.isfile(
                        os.path.join(base_dir, d, "new_scene_metadata.npz")
                    )
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

        # Load metadata
        metadata = np.load(
            osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
        )
        basenames = metadata["images"]  # array of basename strings
        trajectories = metadata["trajectories"]  # (N, 4, 4)
        intrinsics_all = metadata["intrinsics"]  # (N, 3, 3)
        num_files = len(basenames)

        # Collect all poses for ranking
        pose_all = trajectories.astype(np.float32)

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
            basename = str(basenames[im_idx])

            # Load image
            impath = osp.join(scene_dir, "images", basename + ".jpg")
            rgb_image = imread_cv2(impath)

            # Load depth (png, uint16, mm)
            depthpath = osp.join(scene_dir, "depth", basename + ".png")
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000.0
            depthmap[~np.isfinite(depthmap)] = 0.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            # Camera pose and intrinsics from metadata
            camera_pose = pose_all[im_idx].astype(np.float32)
            intri = intrinsics_all[im_idx].astype(np.float32)

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
                dataset="scannetpp",
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

    dataset = ScanNetpp(
        split="train",
        ROOT="./data/scannetpp/",
        resolution=224,
        num_seq=1,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scene_list)}")
