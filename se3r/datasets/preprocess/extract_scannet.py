#!/usr/bin/env python3
"""
Batch extraction script for ScanNet .sens files.

Extracts color images, depth maps, camera poses, and intrinsics from
raw .sens files into the directory structure expected by AMB3R's Scannet
dataset loader:

    ROOT/scans/<scene_id>/
        color/      -> {idx}.jpg  (RGB images)
        depth/      -> {idx}.png  (16-bit depth, depth_shift=1000, unit: mm)
        pose/       -> {idx}.txt  (4x4 camera-to-world matrix)
        intrinsic/  -> intrinsic_depth.txt, intrinsic_color.txt, etc.

Usage:
    python extract_scannet.py \
        --scannet_dir /path/to/scannet_subset \
        --frame_skip 1 \
        --num_workers 4

    # Extract a single scene for testing:
    python extract_scannet.py \
        --scannet_dir /path/to/scannet_subset \
        --scene_id scene0000_01

Notes:
    - The original SensorData.py from ScanNet is written in Python 2.
      This script uses a Python 3 compatible version (sensor_data.py).
    - Supports resuming: scenes with existing extraction are skipped.
    - Color images are resized to depth resolution (typically 640x480)
      to match what the AMB3R Scannet dataset loader expects.
"""

import os
import sys
import glob
import time
import shutil
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent paths so we can import sensor_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sensor_data import SensorData


# Files/patterns that are NOT needed for AMB3R training
# and can be safely deleted after extraction (originals on NAS)
UNNEEDED_PATTERNS = [
    "*.sens",  # Raw sensor stream (~3.7 GB each)
    "*_vh_clean*.ply",  # Mesh reconstructions
    "*_2d-instance*.zip",  # 2D instance annotation projections
    "*_2d-label*.zip",  # 2D label annotation projections
    "*_vh_clean*.segs.json",  # Mesh segmentation
    "*.aggregation.json",  # Semantic annotations
    "*.txt",  # Scene metadata (NOT pose/*.txt)
]

# Temp directories to remove
TEMP_DIR_PREFIXES = ["tmp"]


def cleanup_scene(scene_dir, scene_id):
    """
    Remove unnecessary files from a scene directory after extraction.
    Only removes files in the scene root directory, not in subdirectories
    like color/, depth/, pose/, intrinsic/.

    Returns:
        Total bytes freed.
    """
    freed = 0

    # Remove matching files in the scene root directory only
    for pattern in UNNEEDED_PATTERNS:
        for fpath in glob.glob(os.path.join(scene_dir, pattern)):
            # Safety: only delete files directly in scene_dir, not subdirs
            if os.path.dirname(fpath) == scene_dir and os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                os.remove(fpath)
                freed += size

    # Remove temp files and directories (e.g., tmpd0f6j6hs)
    for item in os.listdir(scene_dir):
        item_path = os.path.join(scene_dir, item)
        if not any(item.startswith(prefix) for prefix in TEMP_DIR_PREFIXES):
            continue
        if os.path.isdir(item_path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fnames in os.walk(item_path)
                for f in fnames
            )
            shutil.rmtree(item_path)
            freed += size
        elif os.path.isfile(item_path):
            freed += os.path.getsize(item_path)
            os.remove(item_path)

    return freed


def extract_single_scene(args_tuple):
    """
    Extract a single scene from its .sens file.

    Args:
        args_tuple: (sens_path, output_dir, frame_skip, resize_color, do_cleanup)

    Returns:
        (scene_id, success, message)
    """
    sens_path, output_dir, frame_skip, resize_color, do_cleanup = args_tuple
    scene_id = os.path.basename(output_dir)

    try:
        # Check if already extracted by looking for all required directories
        required_dirs = ["color", "depth", "pose", "intrinsic"]
        all_exist = all(
            os.path.isdir(os.path.join(output_dir, d)) for d in required_dirs
        )

        if all_exist:
            # Verify at least some files exist in each directory
            has_files = all(
                len(os.listdir(os.path.join(output_dir, d))) > 0 for d in required_dirs
            )
            if has_files:
                # Still do cleanup even if extraction was skipped
                if do_cleanup:
                    freed = cleanup_scene(output_dir, scene_id)
                    freed_mb = freed / (1024 * 1024)
                    if freed_mb > 0.1:
                        print(
                            f"[{scene_id}] Already extracted, cleaned up {freed_mb:.0f} MB"
                        )
                        return (
                            scene_id,
                            True,
                            f"Already extracted, cleaned up {freed_mb:.0f} MB",
                        )
                return (scene_id, True, "Already extracted, skipping")

        # Load the .sens file
        print(f"\n[{scene_id}] Loading .sens file...")
        start_time = time.time()
        sd = SensorData(sens_path)
        load_time = time.time() - start_time
        print(
            f"[{scene_id}] Loaded {sd.num_frames} frames in {load_time:.1f}s "
            f"(color: {sd.color_width}x{sd.color_height}, "
            f"depth: {sd.depth_width}x{sd.depth_height})"
        )

        # Determine target size for color images
        # Resize color to depth resolution for alignment
        if resize_color:
            color_target_size = (sd.depth_height, sd.depth_width)
        else:
            color_target_size = None

        # Export intrinsics (only need to do once, not affected by frame_skip)
        sd.export_intrinsics(os.path.join(output_dir, "intrinsic"))

        # Export color images
        sd.export_color_images(
            os.path.join(output_dir, "color"),
            image_size=color_target_size,
            frame_skip=frame_skip,
        )

        # Export depth images (native resolution)
        sd.export_depth_images(
            os.path.join(output_dir, "depth"),
            frame_skip=frame_skip,
        )

        # Export camera poses
        sd.export_poses(
            os.path.join(output_dir, "pose"),
            frame_skip=frame_skip,
        )

        # Free memory explicitly before cleanup
        num_frames_exported = len(range(0, sd.num_frames, frame_skip))
        del sd

        # Cleanup unnecessary files if requested
        cleanup_msg = ""
        if do_cleanup:
            freed = cleanup_scene(output_dir, scene_id)
            freed_mb = freed / (1024 * 1024)
            cleanup_msg = f", cleaned up {freed_mb:.0f} MB"
            print(f"[{scene_id}] Cleaned up {freed_mb:.0f} MB")

        total_time = time.time() - start_time
        msg = (
            f"Extracted {num_frames_exported} frames in {total_time:.1f}s{cleanup_msg}"
        )
        print(f"[{scene_id}] {msg}")

        return (scene_id, True, msg)

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{scene_id}] {error_msg}")
        return (scene_id, False, error_msg)


def find_sens_files(scannet_dir):
    """
    Find all .sens files in the scannet directory.

    Looks for files matching: scannet_dir/scans/<scene_id>/<scene_id>.sens

    Returns:
        List of (sens_path, scene_dir) tuples
    """
    scans_dir = os.path.join(scannet_dir, "scans")
    if not os.path.isdir(scans_dir):
        raise FileNotFoundError(
            f"Scans directory not found: {scans_dir}\n"
            f"Expected structure: {scannet_dir}/scans/<scene_id>/<scene_id>.sens"
        )

    sens_files = []
    for scene_id in sorted(os.listdir(scans_dir)):
        scene_dir = os.path.join(scans_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        sens_path = os.path.join(scene_dir, f"{scene_id}.sens")
        if os.path.isfile(sens_path):
            sens_files.append((sens_path, scene_dir))

    return sens_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract ScanNet .sens files for AMB3R training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scannet_dir",
        type=str,
        required=True,
        help="Root directory containing scans/ subdirectory with .sens files",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Extract only a specific scene (e.g., scene0000_01)",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Extract every N-th frame. Default: 1 (all frames)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for extraction. Default: 4",
    )
    parser.add_argument(
        "--no_resize_color",
        action="store_true",
        help="Do NOT resize color images to depth resolution",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete unnecessary files (.sens, meshes, annotations) after extraction "
        "to reclaim disk space. Only use if originals are backed up (e.g., on NAS)",
    )
    args = parser.parse_args()

    scannet_dir = args.scannet_dir
    resize_color = not args.no_resize_color

    print("=" * 60)
    print("ScanNet .sens Extraction for AMB3R Training")
    print("=" * 60)
    print(f"  ScanNet directory: {scannet_dir}")
    print(f"  Frame skip: {args.frame_skip}")
    print(f"  Resize color to depth resolution: {resize_color}")
    print(f"  Cleanup after extraction: {args.cleanup}")
    print(f"  Num workers: {args.num_workers}")

    # Find .sens files
    if args.scene_id:
        # Extract only a specific scene
        scene_dir = os.path.join(scannet_dir, "scans", args.scene_id)
        sens_path = os.path.join(scene_dir, f"{args.scene_id}.sens")
        if not os.path.isfile(sens_path):
            if not os.path.isdir(scene_dir):
                print(f"Error: scene directory not found: {scene_dir}")
                sys.exit(1)
            # .sens already deleted (e.g., by prior cleanup), still allow cleanup
            sens_path = None
        sens_files = [(sens_path, scene_dir)]
    else:
        sens_files = find_sens_files(scannet_dir)

    print(f"  Found {len(sens_files)} scenes with .sens files")
    print("=" * 60)

    if len(sens_files) == 0:
        print("No .sens files found. Nothing to do.")
        return

    # Prepare task arguments
    tasks = [
        (sens_path, scene_dir, args.frame_skip, resize_color, args.cleanup)
        for sens_path, scene_dir in sens_files
    ]

    # Execute extraction
    results = {"success": [], "skipped": [], "failed": []}
    start_time = time.time()

    if args.num_workers <= 1:
        # Sequential execution (easier to debug)
        for task in tasks:
            scene_id, success, msg = extract_single_scene(task)
            if success:
                if "skipping" in msg.lower():
                    results["skipped"].append(scene_id)
                else:
                    results["success"].append(scene_id)
            else:
                results["failed"].append((scene_id, msg))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(extract_single_scene, task): task for task in tasks
            }
            for future in as_completed(futures):
                scene_id, success, msg = future.result()
                if success:
                    if "skipping" in msg.lower():
                        results["skipped"].append(scene_id)
                    else:
                        results["success"].append(scene_id)
                else:
                    results["failed"].append((scene_id, msg))

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Newly extracted: {len(results['success'])} scenes")
    print(f"  Skipped (already done): {len(results['skipped'])} scenes")
    print(f"  Failed: {len(results['failed'])} scenes")

    if results["failed"]:
        print("\nFailed scenes:")
        for scene_id, error_msg in results["failed"]:
            print(f"  - {scene_id}: {error_msg[:200]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
