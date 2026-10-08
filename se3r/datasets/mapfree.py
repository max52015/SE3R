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


class MapFree(BaseManyViewDataset):
    """
    MapFree dataset loader for AMB3R training.

    Expects the preprocessed directory structure:
        ROOT/
          {scene}/
            dense{N}/
              rgb/frame_{frame:05d}.jpg
              depth/frame_{frame:05d}.npy
              cam/frame_{frame:05d}.npz      (keys: intrinsic, pose)
              sky_mask/frame_{frame:05d}.jpg

    Note: The cam npz key is 'intrinsic' (singular), not 'intrinsics'.
    Note: Frame pool is filtered by intersection of all modalities to handle
          missing depth/sky_mask files in some dense sequences.
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

        # Build scene_list as (scene, dense_seq) tuples
        self.scene_list = []
        self.scene_frames = {}  # (scene, dense_seq) -> list of valid basenames

        scene_dirs = sorted(
            [
                d
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
        )

        for scene in scene_dirs:
            scene_path = os.path.join(base_dir, scene)
            dense_seqs = sorted(
                [
                    d
                    for d in os.listdir(scene_path)
                    if d.startswith("dense")
                    and os.path.isdir(os.path.join(scene_path, d))
                ]
            )

            for dense_seq in dense_seqs:
                seq_path = os.path.join(scene_path, dense_seq)
                rgb_dir = os.path.join(seq_path, "rgb")
                depth_dir = os.path.join(seq_path, "depth")
                cam_dir = os.path.join(seq_path, "cam")
                sky_dir = os.path.join(seq_path, "sky_mask")

                if not all(
                    os.path.isdir(d)
                    for d in [rgb_dir, depth_dir, cam_dir, sky_dir]
                ):
                    continue

                # Filter frame pool by intersection of all modalities
                rgb_stems = {
                    f[:-4]
                    for f in os.listdir(rgb_dir)
                    if f.endswith(".jpg")
                }
                depth_stems = {
                    f[:-4]
                    for f in os.listdir(depth_dir)
                    if f.endswith(".npy")
                }
                cam_stems = {
                    f[:-4]
                    for f in os.listdir(cam_dir)
                    if f.endswith(".npz")
                }
                sky_stems = {
                    f[:-4]
                    for f in os.listdir(sky_dir)
                    if f.endswith(".jpg")
                }

                valid_frames = sorted(
                    rgb_stems & depth_stems & cam_stems & sky_stems
                )

                if len(valid_frames) < 2:
                    continue

                scene_key = (scene, dense_seq)
                self.scene_list.append(scene_key)
                self.scene_frames[scene_key] = valid_frames

    def _get_views(self, idx, resolution, num_frames, rng, attempts=0):
        scene_key = self.scene_list[idx // self.num_seq]
        scene, dense_seq = scene_key
        seq_dir = osp.join(self.ROOT, scene, dense_seq)

        basenames = self.scene_frames[scene_key]
        num_files = len(basenames)

        # Load intrinsics from the first frame
        cam0 = np.load(
            osp.join(seq_dir, "cam", basenames[0] + ".npz")
        )
        intri = cam0["intrinsic"].astype(np.float32)

        # Load all poses (cache as npy for speed)
        pose_path = osp.join(seq_dir, "poses.npy")
        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)
            # Check if cached poses match current valid frame count
            if pose_all.shape[0] != num_files:
                os.remove(pose_path)
                pose_all = None
            else:
                pose_all = pose_all
        else:
            pose_all = None

        if pose_all is None:
            pose_all = []
            for basename in basenames:
                cam_data = np.load(
                    osp.join(seq_dir, "cam", basename + ".npz")
                )
                pose_all.append(cam_data["pose"].astype(np.float32))
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)

        # Load or compute ranking (cache for speed)
        rank_path = osp.join(seq_dir, "ranking.npy")
        if os.path.exists(rank_path):
            ranking = np.load(rank_path)
            # Validate ranking matches current frame count
            if ranking.shape[0] != num_files:
                os.remove(rank_path)
                ranking = None
            else:
                ranking = ranking
        else:
            ranking = None

        if ranking is None:
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
            impath = osp.join(seq_dir, "rgb", basename + ".jpg")
            rgb_image = imread_cv2(impath)

            # Load depth
            depthpath = osp.join(seq_dir, "depth", basename + ".npy")
            depthmap = np.load(depthpath).astype(np.float32)

            # Load sky mask and apply
            sky_mask_path = osp.join(seq_dir, "sky_mask", basename + ".jpg")
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_UNCHANGED)
            if sky_mask is not None:
                sky_mask = sky_mask >= 127
                depthmap[sky_mask] = 0.0

            # Depth processing (following CUT3R mapfree)
            depthmap[depthmap > 400.0] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
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
                dataset="mapfree",
                label=osp.join(scene, dense_seq, basename),
                instance=basename + ".jpg",
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

    dataset = MapFree(
        split="train",
        ROOT="./data/mapfree/",
        resolution=224,
        num_seq=1,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of scenes: {len(dataset.scene_list)}")
