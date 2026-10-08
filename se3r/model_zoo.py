from .model import AMB3R, AMB3R_SE_Combined


def load_model(model_name, ckpt_path=None):
    """Load SE3R or its AMB3R base model for benchmark use."""
    if model_name == "se3r":
        model = AMB3R_SE_Combined(metric_scale=False, se_reduction=16)
    elif model_name == "amb3r":
        model = AMB3R()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if ckpt_path is not None:
        model.load_weights(ckpt_path)

    return model
