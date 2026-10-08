import yaml
import sys
from se3r.training import get_args_parser, train


def format_model_kwarg(value):
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def load_config_yaml(config_path):
    """Load experiment config YAML and return a flat dict of overrides.

    Maps YAML nested structure to argparse-compatible flat keys:
        training.lr -> lr
        training.epochs -> epochs
        model.class -> model
        output_dir -> output_dir
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    overrides = {}

    # model section -> --model arg
    if "model" in cfg:
        model_cfg = cfg["model"]
        model_class = model_cfg.get("class", "AMB3R_SE_Combined")
        model_kwargs = {
            "metric_scale": model_cfg.get("metric_scale", False),
        }
        for key in ("se_reduction", "se_scale_mode"):
            if key in model_cfg:
                model_kwargs[key] = model_cfg[key]
                overrides[key] = model_cfg[key]

        kwargs_str = ", ".join(
            f"{key}={format_model_kwarg(value)}"
            for key, value in model_kwargs.items()
        )
        overrides["model"] = f"{model_class}({kwargs_str})"

    # training section -> flat training args
    if "training" in cfg:
        train_cfg = cfg["training"]
        key_map = {
            "lr": "lr",
            "weight_decay": "weight_decay",
            "epochs": "epochs",
            "batch_size": "batch_size",
            "accum_iter": "accum_iter",
            "warmup_epochs": "warmup_epochs",
            "amp": "amp",
            "pretrained": "pretrained",
            "seed": "seed",
            "min_lr": "min_lr",
            "num_workers": "num_workers",
            "num_workers_test": "num_workers_test",
        }
        for yaml_key, arg_key in key_map.items():
            if yaml_key in train_cfg:
                overrides[arg_key] = train_cfg[yaml_key]

    # output_dir (top-level)
    if "output_dir" in cfg:
        overrides["output_dir"] = cfg["output_dir"]

    if "freeze" in cfg:
        overrides["freeze_config"] = cfg["freeze"]

    return overrides, cfg


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # Load YAML config and apply overrides (YAML < CLI)
    if args.config:
        overrides, raw_cfg = load_config_yaml(args.config)
        for key, value in overrides.items():
            if not hasattr(args, key):
                setattr(args, key, value)
                continue

            # Only apply YAML value if CLI did not explicitly set it
            # (i.e., the arg still has its default value)
            default_val = parser.get_default(key)
            current_val = getattr(args, key)
            if current_val == default_val:
                setattr(args, key, value)

        # Store raw YAML config for metadata recording
        args.experiment_config = args.config
        args.experiment_name = raw_cfg.get("experiment", {}).get("name", "unknown")
        print(f"Loaded config from: {args.config}")
        print(f"Experiment: {args.experiment_name}")

    # Validate required args
    if args.output_dir is None:
        print("Error: --output_dir is required. Set it in YAML or via CLI.")
        print("Example: python train.py --config configs/se3r.yaml")
        sys.exit(1)

    train(args)
