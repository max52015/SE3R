import os
import sys
import math
import time
import json
import torch
import trimesh
import argparse
import datetime
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from typing import Sized
from pathlib import Path
from skimage import measure
from shutil import copyfile
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from se3r.model import (
    AMB3R,
    AMB3R_SE_Combined,
)

SE_PARAM_GROUPS = (
    "se_dpt_depth_refinenet",
    "se_dpt_depth_post_fusion",
    "se_dpt_point_refinenet",
    "se_dpt_point_post_fusion",
    "se_encoder",
    "se_frame",
    "se_global",
    "se_pair",
    "se_dpt",
)


def _match_se_param_group(name):
    for group in SE_PARAM_GROUPS:
        if group in name:
            return group
    return None


def _should_train_param(name, freeze_config):
    group = _match_se_param_group(name)
    if group is None:
        return False

    if not isinstance(freeze_config, dict) or not freeze_config:
        return True

    if group in freeze_config:
        return not freeze_config[group]

    if group.startswith("se_dpt") and "se_dpt" in freeze_config:
        return not freeze_config["se_dpt"]

    return False


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "thirdparty"
    )
)

import croco.utils.misc as misc

from se3r.datasets import *
from se3r.loss import MultitaskLoss, get_depth_loss, get_point_loss
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from vggt.train_utils.general import check_and_fix_inf_nan

from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from moge.moge.train.losses import scale_invariant_alignment


torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True  # Also good to enable for convolutions


def get_resolution_by_gpu():
    gpu_name = torch.cuda.get_device_name(0).lower()

    res_str = "[(518, 392), (518, 336), (518, 294), (518, 266), (518, 210), (518, 154)]"
    test_res_str = "(518, 392)"
    batch_test = 1

    # debug branch (single GPU, smaller frames)
    if "4090" in gpu_name or "3090" in gpu_name:
        # num_frames = list(range(5, 10)) # 5 to 10 frames
        num_frames = list(range(5, 10))
        # num_frames = [9]
        num_frames_test = 20
    else:
        num_frames = list(range(5, 16))
        num_frames_test = 20

    # --- Training datasets (11 datasets, 2000 samples per epoch) ---
    # Indoor (700): ScanNetpp 175, Scannet 175, ADT 175, Hypersim 175
    # Object (300): WildRGBD 150, OmniObject3D 150
    # Driving (600): Waymo 300, VKitti2 300
    # Outdoor (400): MapFree 133, GTASfM 133, MVSSynth 134
    train_datasets = [
        # Indoor (700 samples)
        f"175 @ ScanNetpp(split='train', ROOT='./data/scannetpp/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"175 @ Scannet(split='train', ROOT='./data/scannet/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"175 @ ADT(split='train', ROOT='./data/adt/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"175 @ Hypersim(split='train', ROOT='./data/hypersim/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        # Object-Centric (300 samples)
        f"150 @ WildRGBD(split='train', ROOT='./data/wildrgbd/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"150 @ OmniObject3D(split='train', ROOT='./data/omniobject3d/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        # Autonomous Driving (600 samples)
        f"300 @ Waymo(split='train', ROOT='./data/waymo/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"300 @ VKitti2(split='train', ROOT='./data/vkitti2/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        # Outdoor (400 samples)
        f"133 @ MapFree(split='train', ROOT='./data/mapfree/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"133 @ GTASfM(split='train', ROOT='./data/gtasfm/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
        f"134 @ MVSSynth(split='train', ROOT='./data/mvs_synth/', resolution={res_str}, num_seq=1, num_frames={num_frames})",
    ]
    trainset = " + ".join(train_datasets)

    # --- Test dataset ---
    testset = f"Scannet(split='test', ROOT='./data/scannet/', resolution={test_res_str}, num_seq=1, num_frames={num_frames_test})"

    return res_str, num_frames, trainset, testset, batch_test


def get_args_parser():
    resolution, num_frames, trainset, testset, batch_size_test = get_resolution_by_gpu()

    parser = argparse.ArgumentParser("SE3R training", add_help=False)
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="path to the SE3R experiment config YAML",
    )
    parser.add_argument(
        "--model",
        default="AMB3R_SE_Combined(metric_scale=False)",
        type=str,
        help="model constructor to build",
    )
    parser.add_argument(
        "--se_reduction",
        default=16,
        type=int,
        help="SE reduction ratio used by YAML model constructors.",
    )
    parser.add_argument(
        "--interp_v2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use improved voxel interpolation (v2). Use --no-interp_v2 to disable.",
    )

    parser.add_argument(
        "--pretrained", default=None, help="path of a starting checkpoint"
    )

    parser.add_argument(
        "--train_dataset",
        default=trainset,
        required=False,
        type=str,
        help="training set",
    )

    parser.add_argument(
        "--test_dataset", default=testset, required=False, type=str, help="training set"
    )

    parser.add_argument(
        "--train_criterion",
        default="MultitaskLoss()",
    )
    parser.add_argument(
        "--test_criterion",
        default="MultitaskLoss()",
    )

    # Exp
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    # Training
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--batch_size_test",
        default=batch_size_test,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--accum_iter",
        default=6,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--epochs",
        default=40,
        type=int,
        help="Maximum number of epochs for the scheduler",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1.5e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-06,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=1, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--amp",
        choices=[False, "bf16", "fp16"],
        default="bf16",
        help="Use Automatic Mixed Precision for pretraining",
    )

    # others
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--num_workers_test", default=0, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--eval_freq", type=int, default=1, help="Test loss evaluation frequency"
    )
    parser.add_argument(
        "--save_freq",
        default=1,
        type=int,
        help="frequence (number of epochs) to save checkpoint in checkpoint-last.pth",
    )
    parser.add_argument(
        "--keep_freq",
        default=10,
        type=int,
        help="frequence (number of epochs) to save checkpoint in checkpoint-%d.pth",
    )
    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="frequence (number of iterations) to print infos while training",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="path where to save the output (required, set in YAML or CLI)",
    )

    return parser


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    save_path = os.path.join(args.output_dir, f"eval_{epoch}")

    os.makedirs(save_path, exist_ok=True)

    dtype = get_dtype(args)

    for i, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):

        views, views_all = batch

        for key in views_all.keys():
            views_all[key] = views_all[key].to(device)

        with torch.autocast("cuda", dtype=dtype):
            pred_all = model.forward(views_all)

        (
            views_all["extrinsics"],
            _,
            views_all["pts3d"],
            views_all["depthmap"],
            views_all["scale"],
        ) = normalize_camera_extrinsics_and_points_batch(
            extrinsics=views_all["extrinsics"],
            cam_points=None,
            world_points=views_all["pts3d"],
            depths=views_all["depthmap"],
            scale_by_points=True,
            point_masks=views_all["valid_mask"],
            pred_points=None,
        )

        for iter in range(len(pred_all)):

            bs, t, h, w, _ = views_all["pts3d"].shape
            pred_all[iter]["world_points"], scale = scale_invariant_alignment(
                pred_all[iter]["world_points"].view(bs, t * h, w, 3),
                views_all["pts3d"].view(bs, t * h, w, 3),
                views_all["valid_mask"].view(bs, t * h, w),
                trunc=None,
                detach=False,
            )

            pred_all[iter]["world_points"] = pred_all[iter]["world_points"].view(
                bs, t, h, w, 3
            )

            pred_all[iter]["depth"], scale_depth = scale_invariant_alignment(
                pred_all[iter]["depth"].repeat(1, 1, 1, 1, 3),
                views_all["depthmap"][..., None].repeat(1, 1, 1, 1, 3),
                views_all["valid_mask"],
                trunc=None,
                detach=False,
            )  # take the first channel only
            pred_all[iter]["depth"] = pred_all[iter]["depth"].view(bs, t, h, w, 3)[
                ..., :1
            ]

            for pose_stage in range(len(pred_all[iter]["pose_enc_list"])):
                pred_all[iter]["pose_enc_list"][pose_stage][..., :3] *= scale_depth[
                    ..., None
                ]

        if i < 5 and misc.is_main_process():
            #  check if it is main process
            pcd_gt = views_all["pts3d"]

            color = (
                views_all["images"].permute(0, 1, 3, 4, 2).reshape(-1, 3) + 1.0
            ) / 2.0

            pcd_gt_np = pcd_gt.detach().cpu().numpy().reshape(-1, 3)
            color_np = color.cpu().numpy().reshape(-1, 3)

            # Save ground truth point cloud
            pc_gt = trimesh.points.PointCloud(
                pcd_gt_np, colors=(color_np * 255).astype(np.uint8)
            )
            pc_gt.export(
                os.path.join(save_path, f"{views[0]['dataset'][0]}_gt_{i}.ply")
            )

            pcd_pred_np = (
                pred_all[0]["world_points"].detach().cpu().numpy().reshape(-1, 3)
            )
            pc_pred = trimesh.points.PointCloud(
                pcd_pred_np, colors=(color_np * 255).astype(np.uint8)
            )
            pc_pred.export(
                os.path.join(save_path, f'{views[0]["dataset"][0]}_front_{i}.ply')
            )

            # Save old prediction point cloud
            pcd_pred1_np = (
                pred_all[-1]["world_points"].detach().cpu().numpy().reshape(-1, 3)
            )
            pc_pred1 = trimesh.points.PointCloud(
                pcd_pred1_np, colors=(color_np * 255).astype(np.uint8)
            )
            # pc_pred1.export(f'pred_old_{i}.ply')
            pc_pred1.export(
                os.path.join(save_path, f'{views[0]["dataset"][0]}_back_{i}.ply')
            )

        loss_all = []
        for iter in range(len(pred_all)):
            pred = pred_all[iter]
            loss_all.append(criterion(pred, views_all))

        loss = 0.0

        loss_all_depth0 = 0
        loss_all_pts0 = 0
        loss_all_cam0 = 0
        loss_all_depth1 = 0
        loss_all_pts1 = 0
        loss_all_cam1 = 0

        for iter in range(len(pred_all)):
            if iter == 0:
                pass
            else:
                loss += loss_all[iter]["objective"]

                loss_depth0 = get_depth_loss(
                    pred_all[iter - 1]["depth"],
                    views_all["depthmap"][..., None],
                    views_all["valid_mask"],
                )
                loss_depth1 = get_depth_loss(
                    pred_all[iter]["depth"],
                    views_all["depthmap"][..., None],
                    views_all["valid_mask"],
                )

                loss_depth0 = check_and_fix_inf_nan(loss_depth0, "loss_depth0")
                loss_depth1 = check_and_fix_inf_nan(loss_depth1, "loss_depth1")

                loss_diff_depth = (
                    loss_depth1 - loss_depth0
                ) / loss_depth0  # if loss_depth0 > 1e-8 else torch.tensor(0.0, device=device)
                # clip nan to zero
                loss_diff_depth = torch.nan_to_num(loss_diff_depth, nan=0.0)

                loss_pts0 = get_point_loss(
                    pred_all[iter - 1]["world_points"],
                    views_all["pts3d"],
                    views_all["valid_mask"],
                )
                loss_pts1 = get_point_loss(
                    pred_all[iter]["world_points"],
                    views_all["pts3d"],
                    views_all["valid_mask"],
                )
                loss_diff_pts = (
                    loss_pts1 - loss_pts0
                ) / loss_pts0  # if loss_pts0 > 1e-8 else torch.tensor(0.0, device=device)
                loss_diff_pts = torch.nan_to_num(loss_diff_pts, nan=0.0)

                loss_pts0 = check_and_fix_inf_nan(loss_pts0, "loss_pts0")
                loss_pts1 = check_and_fix_inf_nan(loss_pts1, "loss_pts1")

                loss_diff_cam = (
                    loss_all[iter]["loss_camera"] - loss_all[iter - 1]["loss_camera"]
                )

                metric_logger.update(
                    **{f"loss_diff_depth_{iter}": float(loss_diff_depth.item())}
                )
                metric_logger.update(
                    **{f"loss_diff_pts_{iter}": float(loss_diff_pts.item())}
                )
                metric_logger.update(
                    **{f"loss_diff_cam_{iter}": float(loss_diff_cam.item())}
                )
                metric_logger.update(**{f"loss_pts": float(loss_pts1.item())})
                metric_logger.update(**{f"loss_depth": float(loss_depth1.item())})
                metric_logger.update(
                    **{f"loss_camera": float(loss_all[iter]["loss_camera"].item())}
                )
                metric_logger.update(
                    **{
                        f"loss_refine": float(
                            loss_pts1.item()
                            + loss_depth1.item()
                            + loss_all[iter]["loss_camera"].item()
                        )
                    }
                )

                loss_all_depth0 += loss_depth0.item()
                loss_all_depth1 += loss_depth1.item()
                loss_all_pts0 += loss_pts0.item()
                loss_all_pts1 += loss_pts1.item()
                loss_all_cam0 += loss_all[iter - 1]["loss_camera"].item()
                loss_all_cam1 += loss_all[iter]["loss_camera"].item()

        loss /= (
            len(pred_all) - 1
        )  # average loss over all iterations except the first one
        loss_value = loss.item()

        loss_avg_diff_depth = (
            (loss_all_depth1 - loss_all_depth0) / loss_all_depth0
            if loss_all_depth0 > 1e-8
            else torch.tensor(0.0, device=device)
        )
        loss_avg_diff_pts = (
            (loss_all_pts1 - loss_all_pts0) / loss_all_pts0
            if loss_all_pts0 > 1e-8
            else torch.tensor(0.0, device=device)
        )
        loss_avg_diff_cam = (
            (loss_all_cam1 - loss_all_cam0) / loss_all_cam0
            if loss_all_cam0 > 1e-8
            else torch.tensor(0.0, device=device)
        )

        metric_logger.update(loss=float(loss_value))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    results["loss_relative_avg"] = float(
        (loss_avg_diff_depth + loss_avg_diff_pts + loss_avg_diff_cam) / 3.0
    )

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

        log_writer.add_scalar(
            prefix + "_loss_diff_depth_avg", loss_avg_diff_depth, 1000 * epoch
        )
        log_writer.add_scalar(
            prefix + "_loss_diff_pts_avg", loss_avg_diff_pts, 1000 * epoch
        )
        log_writer.add_scalar(
            prefix + "_loss_diff_cam_avg", loss_avg_diff_cam, 1000 * epoch
        )

    return results


def get_dtype(args):
    if args.amp:
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    else:
        dtype = torch.float32
    return dtype


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    dtype = get_dtype(args)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        epoch_f = epoch + data_iter_step / len(data_loader)

        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        views, views_all = batch

        for key in views_all.keys():
            views_all[key] = views_all[key].to(device)

        with torch.autocast("cuda", dtype=dtype):
            pred_all = model.forward(views_all)

        (
            views_all["extrinsics"],
            _,
            views_all["pts3d"],
            views_all["depthmap"],
            views_all["scale"],
        ) = normalize_camera_extrinsics_and_points_batch(
            extrinsics=views_all["extrinsics"],
            cam_points=None,
            world_points=views_all["pts3d"],
            depths=views_all["depthmap"],
            scale_by_points=True,
            point_masks=views_all["valid_mask"],
            pred_points=None,
        )

        for iter in range(len(pred_all)):

            bs, t, h, w, _ = views_all["pts3d"].shape
            pred_all[iter]["world_points"], scale_point = scale_invariant_alignment(
                pred_all[iter]["world_points"].view(bs, t * h, w, 3),
                views_all["pts3d"].view(bs, t * h, w, 3),
                views_all["valid_mask"].view(bs, t * h, w),
                trunc=None,
                detach=False,
            )

            pred_all[iter]["world_points"] = pred_all[iter]["world_points"].view(
                bs, t, h, w, 3
            )

            pred_all[iter]["depth"], scale_depth = scale_invariant_alignment(
                pred_all[iter]["depth"].repeat(1, 1, 1, 1, 3),
                views_all["depthmap"][..., None].repeat(1, 1, 1, 1, 3),
                views_all["valid_mask"],
                trunc=None,
                detach=False,
            )  # take the first channel only
            pred_all[iter]["depth"] = pred_all[iter]["depth"].view(bs, t, h, w, 3)[
                ..., :1
            ]
            for pose_stage in range(len(pred_all[iter]["pose_enc_list"])):
                pred_all[iter]["pose_enc_list"][pose_stage][..., :3] *= scale_depth[
                    ..., None
                ]

        loss_all = []
        for iter in range(len(pred_all)):
            pred = pred_all[iter]
            loss_all.append(criterion(pred, views_all))

        loss = 0.0

        for iter in range(len(pred_all)):
            if iter == 0:
                pass
            else:
                loss += loss_all[iter]["objective"]
                loss_diff = (
                    loss_all[iter]["objective"] - loss_all[iter - 1]["objective"]
                )

                loss_depth0 = get_depth_loss(
                    pred_all[iter - 1]["depth"],
                    views_all["depthmap"][..., None],
                    views_all["valid_mask"],
                )
                loss_depth0 = check_and_fix_inf_nan(loss_depth0, "loss_depth0")
                loss_depth1 = get_depth_loss(
                    pred_all[iter]["depth"],
                    views_all["depthmap"][..., None],
                    views_all["valid_mask"],
                )
                loss_depth1 = check_and_fix_inf_nan(loss_depth1, "loss_depth1")
                loss_diff_depth = (loss_depth1 - loss_depth0) / loss_depth0

                loss_diff_depth = check_and_fix_inf_nan(
                    loss_diff_depth, "loss_diff_depth"
                )

                loss_pts0 = get_point_loss(
                    pred_all[iter - 1]["world_points"],
                    views_all["pts3d"],
                    views_all["valid_mask"],
                )
                loss_pts0 = check_and_fix_inf_nan(loss_pts0, "loss_pts0")
                loss_pts1 = get_point_loss(
                    pred_all[iter]["world_points"],
                    views_all["pts3d"],
                    views_all["valid_mask"],
                )
                loss_pts1 = check_and_fix_inf_nan(loss_pts1, "loss_pts1")
                loss_diff_pts = (loss_pts1 - loss_pts0) / loss_pts0
                loss_diff_pts = check_and_fix_inf_nan(loss_diff_pts, "loss_diff_pts")

                loss_diff_cam = (
                    loss_all[iter]["loss_camera"] - loss_all[iter - 1]["loss_camera"]
                )
                loss_diff_cam = check_and_fix_inf_nan(loss_diff_cam, "loss_diff_cam")

                epoch_1000x = int(epoch_f * 1000)
                metric_logger.update(**{f"loss_diff_{iter}": float(loss_diff.item())})
                metric_logger.update(
                    **{f"loss_diff_depth_{iter}": float(loss_diff_depth.item())}
                )
                metric_logger.update(
                    **{f"loss_diff_pts_{iter}": float(loss_diff_pts.item())}
                )
                metric_logger.update(
                    **{f"loss_diff_cam_{iter}": float(loss_diff_cam.item())}
                )

                metric_logger.update(**{f"loss_pts": float(loss_pts1.item())})
                metric_logger.update(**{f"loss_depth": float(loss_depth1.item())})
                metric_logger.update(
                    **{f"loss_camera": float(loss_all[iter]["loss_camera"].item())}
                )
                metric_logger.update(
                    **{
                        f"loss_refine": float(
                            loss_pts1.item()
                            + loss_depth1.item()
                            + loss_all[iter]["loss_camera"].item()
                        )
                    }
                )
                if log_writer is None:
                    continue

                log_writer.add_scalar(
                    f"train_loss_diff_depth_{iter}", loss_diff_depth.item(), epoch_1000x
                )
                log_writer.add_scalar(
                    f"train_loss_diff_pts_{iter}", loss_diff_pts.item(), epoch_1000x
                )
                log_writer.add_scalar(
                    f"train_loss_diff_cam_{iter}", loss_diff_cam.item(), epoch_1000x
                )

                log_writer.add_scalar(f"train_loss_pts", loss_pts1.item(), epoch_1000x)
                log_writer.add_scalar(
                    f"train_loss_depth", loss_depth1.item(), epoch_1000x
                )
                log_writer.add_scalar(
                    f"train_loss_camera",
                    loss_all[iter]["loss_camera"].item(),
                    epoch_1000x,
                )
                log_writer.add_scalar(
                    f"train_loss_refine",
                    loss_pts1.item()
                    + loss_depth1.item()
                    + loss_all[iter]["loss_camera"].item(),
                    epoch_1000x,
                )

        loss /= len(pred_all) - 1
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            clip_grad=1.0,
        )  #

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss, pred, batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.print_freq)
        ) == 0:
            loss_value_reduce = misc.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue

            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("train_lr", lr, epoch_1000x)
            log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class TeeLogger:
    """Duplicate writes to both a stream (terminal) and a log file.

    All print() and warning outputs will appear on screen and be
    saved to train_output.log in the output directory.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def save_experiment_meta(args, model, output_dir):
    """Save experiment metadata for reproducibility.

    Automatically records model info, trainable parameter counts and names,
    and all training arguments to experiment_meta.json.
    """
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_class": model.__class__.__name__,
        "model_name": getattr(model, "name", "unknown"),
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_param_names": [
            n for n, p in model.named_parameters() if p.requires_grad
        ],
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    meta_path = os.path.join(output_dir, "experiment_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Experiment metadata saved to {meta_path}")


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    # auto resume logic: find the latest timestamped folder
    args.resume = None
    if os.path.exists(args.output_dir):
        subdirs = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if os.path.isdir(os.path.join(args.output_dir, d)) and not d == "recording"
        ]
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            last_ckpt_fname = os.path.join(latest_subdir, "checkpoint-last.pth")
            if os.path.isfile(last_ckpt_fname):
                args.resume = last_ckpt_fname

        # fallback for old structure
        if not args.resume:
            last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
            if os.path.isfile(last_ckpt_fname):
                args.resume = last_ckpt_fname

    # Update output_dir to be a timestamped subdirectory for this new run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, run_timestamp)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup terminal output logging to file
    # All stdout/stderr will be written to both terminal and train_output.log
    if misc.is_main_process():
        log_file = open(os.path.join(args.output_dir, "train_output.log"), "a")
        sys.stdout = TeeLogger(sys.stdout, log_file)
        sys.stderr = TeeLogger(sys.stderr, log_file)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print("Building train dataset {:s}".format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset, args.batch_size, args.num_workers, test=False
    )

    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset, args.batch_size_test, args.num_workers_test, test=True
        )
        for dataset in args.test_dataset.split("+")
    }

    vggt_ckpt = "./checkpoints/VGGT.pt"
    assert os.path.isfile(vggt_ckpt), (
        f"VGGT checkpoint not found at {vggt_ckpt}. "
        "Please download the pretrained VGGT weights and place them at ./checkpoints/VGGT.pt"
    )

    model_str = args.model
    if args.interp_v2 and model_str.startswith("AMB3R(") and "interp_v2" not in model_str:
        model_str = model_str.rstrip(")") + ", interp_v2=True)"
    print("Loading model: {:s}".format(model_str))
    model = eval(model_str)

    print(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    test_criterion = eval(args.test_criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print("Loading pretrained: ", args.pretrained)
        if hasattr(model, "load_weights"):
            # SE models have custom load_weights that handles key filtering
            model.load_weights(args.pretrained)
        else:
            ckpt = torch.load(args.pretrained, map_location=device)
            print(model.load_state_dict(ckpt["model"], strict=False))
            del ckpt

    # Convert aggregator to mixed precision
    if hasattr(model, "prepare"):
        model.prepare(args.amp if args.amp else "fp32")

    freeze_config = getattr(args, "freeze_config", None)
    if isinstance(freeze_config, dict) and freeze_config:
        print("Applying YAML freeze config for SE modules:")
        for key in sorted(freeze_config):
            if key.startswith("se_"):
                state = "frozen" if freeze_config[key] else "trainable"
                print(f"  {key}: {state}")

    # Freeze everything except trainable SE modules.
    for name, param in model.named_parameters():
        param.requires_grad = _should_train_param(name, freeze_config)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True
        )
        model_without_ddp = model.module

    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    param_names_to_train = [
        name
        for name, param in model_without_ddp.named_parameters()
        if param.requires_grad
    ]

    # Now you can inspect the names
    print("Parameters being trained:")
    for name in param_names_to_train:
        print(name)

    use_loss_scaler = args.amp != "bf16"
    if not use_loss_scaler:
        print("Disabling GradScaler for bf16 AMP")
    loss_scaler = NativeScaler(enabled=use_loss_scaler)

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    best_so_far, _ = misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    # Always reset best_so_far because we changed the best-checkpoint metric
    # from relative improvement (loss_relative_avg, negative) to absolute
    # quality (loss_refine_avg, positive). Old stored values are incompatible.
    best_so_far = float("inf")

    # Top-3 best checkpoint tracker: list of (loss, epoch, filename)
    # Sorted ascending by loss (index 0 = best)
    top3_best = []

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    file_path_all = ["./", "se3r"]

    os.makedirs(os.path.join(args.output_dir, "recording"), exist_ok=True)

    for file_path in file_path_all:
        cur_dir = os.path.join(args.output_dir, "recording", file_path)
        os.makedirs(cur_dir, exist_ok=True)

        files = os.listdir(file_path)
        for f_name in files:
            if f_name[-3:] == ".py":
                copyfile(os.path.join(file_path, f_name), os.path.join(cur_dir, f_name))

    # Save experiment metadata for reproducibility
    if misc.is_main_process():
        save_experiment_meta(args, model_without_ddp, args.output_dir)

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):
        torch.cuda.empty_cache()

        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            test_loss = 0.0
            test_relative_loss = 0.0
            for test_name, testset in data_loader_test.items():

                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

                test_loss += stats["loss_refine_avg"]
                test_relative_loss += stats["loss_relative_avg"]

            test_loss /= len(data_loader_test)
            test_relative_loss /= len(data_loader_test)

            if test_loss < best_so_far:
                best_so_far = test_loss
                new_best = True
                print(f"New best loss (refine_avg): {best_so_far:.4f} at epoch {epoch}")

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > 0:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                # Update top-3 best checkpoints
                new_entry = (test_loss, epoch, f"epoch_{epoch}_no1")

                # Shift existing entries down by 1 rank
                updated_top3 = []
                for rank_idx, (old_loss, old_epoch, old_fname) in enumerate(top3_best):
                    new_rank = rank_idx + 2  # current no1 -> no2, no2 -> no3, etc.
                    new_fname = f"epoch_{old_epoch}_no{new_rank}"
                    # Rename existing checkpoint file
                    old_ckpt = os.path.join(
                        args.output_dir, f"checkpoint-{old_fname}.pth"
                    )
                    new_ckpt = os.path.join(
                        args.output_dir, f"checkpoint-{new_fname}.pth"
                    )
                    if os.path.exists(old_ckpt):
                        os.rename(old_ckpt, new_ckpt)
                    updated_top3.append((old_loss, old_epoch, new_fname))

                top3_best = [new_entry] + updated_top3

                # Keep only top 3, delete the 4th if it exists
                if len(top3_best) > 3:
                    _, _, evicted_fname = top3_best.pop()
                    evicted_path = os.path.join(
                        args.output_dir, f"checkpoint-{evicted_fname}.pth"
                    )
                    if os.path.exists(evicted_path):
                        os.remove(evicted_path)
                        print(f"Removed evicted checkpoint: {evicted_fname}")

                # Save the new best
                save_model(epoch - 1, new_entry[2], best_so_far)
                print(f"Top-3 best: {[(f, f'{l:.4f}') for l, e, f in top3_best]}")

                # Also save as 'best' symlink/copy for backward compatibility
                save_model(epoch - 1, "best", best_so_far)

        if epoch >= args.epochs:
            break

        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
