import os
import cv2
import numpy as np
import os.path as osp
import h5py
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from ..tools.utils import threshold_depth_map
from ..tools.pose_dist import compute_ranking


class Waymo(BaseManyViewDataset):
    """
    Waymo Open Dataset loader for AMB3R training.

    Expects the preprocessed directory structure:
        ROOT/
          invalid_files.h5
          {segment_name}/
            {frame_id}_{seq_id}.jpg
            {frame_id}_{seq_id}.exr      (depth, EXR format)
            {frame_id}_{seq_id}.npz      (keys: intrinsics, cam2world, distortion)

    Note: seq_id=5 (rear camera) is excluded per CUT3R convention.
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

    def _load_invalid_dict(self, h5_file_path):
        """Load invalid file pairs from the h5 file."""
        invalid_dict = {}
        if not os.path.exists(h5_file_path):
            return invalid_dict
        with h5py.File(h5_file_path, "r") as h5f:
            for scene in h5f:
                data = h5f[scene]["invalid_pairs"][:]
                invalid_pairs = set(
                    tuple(pair.decode("utf-8").split("_")) for pair in data
                )
                invalid_dict[scene] = invalid_pairs
        return invalid_dict

    def load_all_scenes(self, base_dir):
        if self.test_id is not None:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
            print(f"Test_id: {self.test_id}")
            return

        # Load invalid pairs
        invalid_dict = self._load_invalid_dict(
            os.path.join(base_dir, "invalid_files.h5")
        )

        # Find all segment directories
        segment_dirs = sorted(
            [
                d
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
                and d.startswith("segment")
            ]
        )

        self.scene_list = []
        self.scene_data = {}  # (segment, seq_id) -> list of frame_ids

        for segment in segment_dirs:
            segment_dir = osp.join(base_dir, segment)
            invalid_pairs = invalid_dict.get(segment, set())

            seq2frames = {}
            for f in os.listdir(segment_dir):
                if not f.endswith(".jpg"):
                    continue
                basename = f[:-4]
                parts = basename.split("_")
                frame_id = parts[0]
                seq_id = parts[1]

                # Skip rear camera (seq_id=5)
                if seq_id == "5":
                    continue
                # Skip invalid pairs
                if (seq_id, frame_id) in invalid_pairs:
                    continue

                if seq_id not in seq2frames:
                    seq2frames[seq_id] = []
                seq2frames[seq_id].append(frame_id)

            for seq_id, frame_ids in seq2frames.items():
                frame_ids = sorted(frame_ids)
                if len(frame_ids) < 2:
                    continue
                scene_key = (segment, seq_id)
                self.scene_list.append(scene_key)
                self.scene_data[scene_key] = frame_ids

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_key = self.scene_list[idx // self.num_seq]
        segment, seq_id = scene_key
        segment_dir = osp.join(self.ROOT, segment)
        frame_ids = self.scene_data[scene_key]
        num_files = len(frame_ids)

        # Load all poses (cache as npy per sequence)
        pose_path = osp.join(segment_dir, f"poses_seq{seq_id}.npy")
        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)
        else:
            pose_all = []
            for fid in frame_ids:
                npz_path = osp.join(segment_dir, f"{fid}_{seq_id}.npz")
                cam_data = np.load(npz_path)
                pose_all.append(cam_data["cam2world"].astype(np.float32))
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)

        # Load or compute ranking (cache for speed)
        rank_path = osp.join(segment_dir, f"ranking_seq{seq_id}.npy")
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
            k = 128
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
            impath_base = f"{fid}_{seq_id}"

            # Load image
            impath = osp.join(segment_dir, impath_base + ".jpg")
            rgb_image = imread_cv2(impath)

            # Load depth (EXR format)
            depthpath = osp.join(segment_dir, impath_base + ".exr")
            depthmap = imread_cv2(depthpath)
            if depthmap is not None and len(depthmap.shape) == 3:
                depthmap = depthmap[:, :, 0]  # take first channel
            depthmap = depthmap.astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0.0
            depthmap = threshold_depth_map(
                depthmap, min_percentile=-1, max_percentile=98
            ).astype(np.float32)

            # Load intrinsics and pose
            cam_data = np.load(osp.join(segment_dir, impath_base + ".npz"))
            intri = cam_data["intrinsics"].astype(np.float32)
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
                dataset="waymo",
                label=osp.join(segment, seq_id, fid),
                instance=impath_base + ".jpg",
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

    dataset = Waymo(
        split="train",
        ROOT="./data/waymo/",
        resolution=224,
        num_seq=1,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scene_list)}")
