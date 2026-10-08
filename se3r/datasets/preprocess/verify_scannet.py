#!/usr/bin/env python3
"""
Verification script for extracted ScanNet data.

Checks that the extracted data from .sens files matches the format
expected by AMB3R's Scannet dataset loader. Performs the following checks:

1. Directory structure: color/, depth/, pose/, intrinsic/ exist
2. File count consistency: same number of files across color/depth/pose
3. Depth map validity: non-zero uint16 values, reasonable range
4. Pose validity: finite values, valid SE(3) matrices (det(R) ~ 1)
5. Intrinsic validity: positive fx, fy, cx, cy values
6. (Optional) AMB3R dataset loader smoke test

Usage:
    python verify_scannet.py \
        --scannet_dir /path/to/scannet_subset

    # Test AMB3R dataset loader on a single scene:
    python verify_scannet.py \
        --scannet_dir ~/Datasets/scannet_subset \
        --scene_id scene0000_01 \
        --test_loader
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path


class VerificationResult:
    """Container for verification results of a single scene."""

    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.checks = []
        self.warnings = []
        self.errors = []

    def add_check(self, name, passed, message=""):
        self.checks.append((name, passed, message))
        if not passed:
            self.errors.append(f"[{name}] {message}")

    def add_warning(self, message):
        self.warnings.append(message)

    @property
    def passed(self):
        return len(self.errors) == 0

    def summary(self):
        status = "PASS" if self.passed else "FAIL"
        n_pass = sum(1 for _, p, _ in self.checks if p)
        n_total = len(self.checks)
        return f"[{status}] {self.scene_id}: {n_pass}/{n_total} checks passed"


def verify_directory_structure(scene_dir, result):
    """Check that all required subdirectories exist and have files."""
    required_dirs = ["color", "depth", "pose", "intrinsic"]

    for d in required_dirs:
        dir_path = os.path.join(scene_dir, d)
        exists = os.path.isdir(dir_path)
        has_files = exists and len(os.listdir(dir_path)) > 0
        result.add_check(
            f"dir_{d}",
            has_files,
            (
                f"{'Missing' if not exists else 'Empty'}: {dir_path}"
                if not has_files
                else ""
            ),
        )


def verify_file_counts(scene_dir, result):
    """Check that color, depth, pose have the same number of files."""
    dirs = ["color", "depth", "pose"]
    counts = {}
    for d in dirs:
        dir_path = os.path.join(scene_dir, d)
        if os.path.isdir(dir_path):
            counts[d] = len(os.listdir(dir_path))
        else:
            counts[d] = 0

    all_equal = len(set(counts.values())) == 1
    result.add_check(
        "file_counts",
        all_equal,
        f"Mismatched file counts: {counts}" if not all_equal else "",
    )

    if all_equal and counts.get("color", 0) > 0:
        result.add_check("num_frames", True, f"{counts['color']} frames")
    return counts


def verify_depth_samples(scene_dir, result, num_samples=5):
    """Verify a few depth map samples for validity."""
    depth_dir = os.path.join(scene_dir, "depth")
    if not os.path.isdir(depth_dir):
        return

    depth_files = sorted(os.listdir(depth_dir))
    if len(depth_files) == 0:
        return

    # Sample a few depth files evenly
    indices = np.linspace(0, len(depth_files) - 1, min(num_samples, len(depth_files)))
    indices = [int(i) for i in indices]

    valid_count = 0
    total_checked = 0
    for idx in indices:
        depth_path = os.path.join(depth_dir, depth_files[idx])
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        total_checked += 1
        if depth is None:
            result.add_warning(f"Cannot read depth file: {depth_files[idx]}")
            continue

        # Check dtype
        if depth.dtype != np.uint16:
            result.add_warning(
                f"Depth {depth_files[idx]} has dtype {depth.dtype}, expected uint16"
            )

        # Check if there are valid (non-zero) pixels
        nonzero_ratio = (depth > 0).sum() / depth.size
        if nonzero_ratio < 0.01:
            result.add_warning(
                f"Depth {depth_files[idx]} has very few valid pixels: {nonzero_ratio:.1%}"
            )
        else:
            valid_count += 1

        # Check value range (ScanNet uses depth_shift=1000, so 1000 = 1 meter)
        if depth.max() > 0:
            max_depth_m = depth.max() / 1000.0
            if max_depth_m > 20.0:
                result.add_warning(
                    f"Depth {depth_files[idx]} has max depth {max_depth_m:.1f}m (unusually large)"
                )

    result.add_check(
        "depth_validity",
        valid_count > 0,
        f"{valid_count}/{total_checked} sampled depth maps are valid",
    )


def verify_pose_samples(scene_dir, result, num_samples=5):
    """Verify a few camera pose samples for validity."""
    pose_dir = os.path.join(scene_dir, "pose")
    if not os.path.isdir(pose_dir):
        return

    pose_files = sorted(os.listdir(pose_dir))
    if len(pose_files) == 0:
        return

    indices = np.linspace(0, len(pose_files) - 1, min(num_samples, len(pose_files)))
    indices = [int(i) for i in indices]

    valid_count = 0
    total_checked = 0
    invalid_pose_count = 0

    for idx in indices:
        pose_path = os.path.join(pose_dir, pose_files[idx])
        try:
            pose = np.loadtxt(pose_path, dtype=np.float32)
        except Exception as e:
            result.add_warning(f"Cannot read pose file {pose_files[idx]}: {e}")
            total_checked += 1
            continue

        total_checked += 1

        # Check shape
        if pose.shape != (4, 4):
            result.add_warning(
                f"Pose {pose_files[idx]} has shape {pose.shape}, expected (4, 4)"
            )
            continue

        # Check for NaN/Inf
        if not np.isfinite(pose).all():
            invalid_pose_count += 1
            continue

        # Check rotation matrix det(R) ~ 1 (valid SO(3))
        R = pose[:3, :3]
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=0.1):
            result.add_warning(
                f"Pose {pose_files[idx]} has det(R)={det:.3f}, expected ~1.0"
            )
            continue

        valid_count += 1

    result.add_check(
        "pose_validity",
        valid_count > 0,
        f"{valid_count}/{total_checked} sampled poses are valid "
        f"({invalid_pose_count} have NaN/Inf)",
    )

    if invalid_pose_count > 0:
        result.add_warning(
            f"{invalid_pose_count}/{total_checked} sampled poses contain NaN/Inf "
            "(this is normal for ScanNet - the dataset loader handles this)"
        )


def verify_intrinsics(scene_dir, result):
    """Verify camera intrinsic matrices."""
    intrinsic_dir = os.path.join(scene_dir, "intrinsic")
    if not os.path.isdir(intrinsic_dir):
        return

    # The AMB3R Scannet loader reads intrinsic_depth.txt
    depth_intri_path = os.path.join(intrinsic_dir, "intrinsic_depth.txt")
    if not os.path.isfile(depth_intri_path):
        result.add_check("intrinsic_depth", False, f"Missing: intrinsic_depth.txt")
        return

    try:
        intri = np.loadtxt(depth_intri_path, dtype=np.float32)
    except Exception as e:
        result.add_check("intrinsic_depth", False, f"Cannot parse: {e}")
        return

    # Check shape: the AMB3R loader reads it as [:3, :3]
    if intri.shape not in [(3, 3), (4, 4)]:
        result.add_check(
            "intrinsic_depth",
            False,
            f"Shape {intri.shape}, expected (4,4) or (3,3)",
        )
        return

    # Check reasonable values for fx, fy, cx, cy
    fx = intri[0, 0]
    fy = intri[1, 1]
    cx = intri[0, 2]
    cy = intri[1, 2]

    valid = fx > 0 and fy > 0 and cx > 0 and cy > 0
    result.add_check(
        "intrinsic_depth",
        valid,
        f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
        + (" (some <= 0!)" if not valid else ""),
    )


def verify_scene(scene_dir):
    """Run all verification checks on a single scene directory."""
    scene_id = os.path.basename(scene_dir)
    result = VerificationResult(scene_id)

    verify_directory_structure(scene_dir, result)
    counts = verify_file_counts(scene_dir, result)
    verify_depth_samples(scene_dir, result)
    verify_pose_samples(scene_dir, result)
    verify_intrinsics(scene_dir, result)

    return result


def test_amb3r_loader(scannet_dir, scene_id):
    """
    Smoke test: try loading data through AMB3R's Scannet dataset class.
    """
    print(f"\nTesting AMB3R Scannet loader for scene: {scene_id}")
    print("-" * 50)

    try:
        # Add amb3r and its thirdparty dependencies to path
        amb3r_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        thirdparty_dir = os.path.join(amb3r_root, "thirdparty")
        sys.path.insert(0, amb3r_root)
        if os.path.isdir(thirdparty_dir) and thirdparty_dir not in sys.path:
            sys.path.insert(0, thirdparty_dir)
        for subdir in ["dust3r", "croco", "vggt", "moge"]:
            p = os.path.join(thirdparty_dir, subdir)
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        from ..scannet import Scannet

        dataset = Scannet(
            split="train",
            ROOT=scannet_dir,
            resolution=224,
            num_seq=1,
            num_frames=3,
            test_id=scene_id,
        )

        print(f"  Dataset length: {len(dataset)}")
        print(f"  Loading sample...")

        views, views_all = dataset[0]

        print(f"  Number of views: {len(views)}")
        print(f"  views_all keys: {list(views_all.keys())}")
        for key, val in views_all.items():
            if hasattr(val, "shape"):
                print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"    {key}: {type(val)}")

        # Basic sanity checks
        assert "pts3d" in views_all, "pts3d missing from views_all"
        assert "depthmap" in views_all, "depthmap missing from views_all"
        assert "valid_mask" in views_all, "valid_mask missing from views_all"
        assert "extrinsics" in views_all, "extrinsics missing from views_all"
        assert "camera_intrinsics" in views_all, "camera_intrinsics missing"

        valid_ratio = views_all["valid_mask"].float().mean().item()
        print(f"  Valid mask ratio: {valid_ratio:.2%}")
        print(
            f"  Depth range: [{views_all['depthmap'].min():.2f}, {views_all['depthmap'].max():.2f}]"
        )
        print(
            f"  pts3d range: [{views_all['pts3d'].min():.2f}, {views_all['pts3d'].max():.2f}]"
        )

        print("\n  AMB3R loader test: PASSED")
        return True

    except Exception as e:
        print(f"\n  AMB3R loader test: FAILED")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify extracted ScanNet data for AMB3R training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scannet_dir",
        type=str,
        required=True,
        help="Root directory containing scans/ subdirectory",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Verify only a specific scene",
    )
    parser.add_argument(
        "--test_loader",
        action="store_true",
        help="Also test AMB3R's Scannet dataset loader",
    )
    args = parser.parse_args()

    scans_dir = os.path.join(args.scannet_dir, "scans")
    if not os.path.isdir(scans_dir):
        print(f"Error: Scans directory not found: {scans_dir}")
        sys.exit(1)

    # Collect scenes to verify
    if args.scene_id:
        scene_dirs = [os.path.join(scans_dir, args.scene_id)]
    else:
        scene_dirs = sorted(
            [
                os.path.join(scans_dir, d)
                for d in os.listdir(scans_dir)
                if os.path.isdir(os.path.join(scans_dir, d))
            ]
        )

    print("=" * 60)
    print("ScanNet Extraction Verification")
    print("=" * 60)
    print(f"  Checking {len(scene_dirs)} scenes in {scans_dir}")
    print()

    # Verify each scene
    pass_count = 0
    fail_count = 0
    not_extracted = 0

    for scene_dir in scene_dirs:
        if not os.path.isdir(scene_dir):
            print(f"  Scene directory not found: {scene_dir}")
            fail_count += 1
            continue

        # Skip scenes that have not been extracted yet
        has_any_extract = any(
            os.path.isdir(os.path.join(scene_dir, d))
            for d in ["color", "depth", "pose", "intrinsic"]
        )
        if not has_any_extract:
            not_extracted += 1
            continue

        result = verify_scene(scene_dir)

        if result.passed:
            pass_count += 1
            if args.scene_id:
                # Print detailed output for single scene
                print(result.summary())
                for name, passed, msg in result.checks:
                    status = "OK" if passed else "FAIL"
                    print(f"  [{status}] {name}: {msg}")
                for warning in result.warnings:
                    print(f"  [WARN] {warning}")
        else:
            fail_count += 1
            print(result.summary())
            for error in result.errors:
                print(f"  {error}")

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"  Passed: {pass_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Not extracted: {not_extracted}")
    print(f"  Total: {len(scene_dirs)}")

    # Optional AMB3R loader test
    if args.test_loader:
        target_scene = args.scene_id
        if target_scene is None:
            # Pick the first extracted scene
            for scene_dir in scene_dirs:
                has_extract = all(
                    os.path.isdir(os.path.join(scene_dir, d))
                    for d in ["color", "depth", "pose", "intrinsic"]
                )
                if has_extract:
                    target_scene = os.path.basename(scene_dir)
                    break

        if target_scene:
            test_amb3r_loader(args.scannet_dir, target_scene)
        else:
            print("\nNo extracted scenes found to test AMB3R loader.")


if __name__ == "__main__":
    main()
