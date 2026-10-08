"""Adapter for loading SE3R through the AMB3R benchmark model interface."""

from se3r.model import AMB3R_SE_Combined


def load_se3r(checkpoint_path, device="cuda"):
    """Load a trained SE3R checkpoint and return an evaluation-ready model."""
    model = AMB3R_SE_Combined(metric_scale=False, se_reduction=16)
    model.load_weights(str(checkpoint_path), data_type="bf16")
    return model.to(device).eval()


def run_amb3r_benchmark(model, frames):
    """Run the standard AMB3R benchmark forward interface."""
    return model.run_amb3r_benchmark(frames)
