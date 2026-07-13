"""Centralized configuration for DRIDNet."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
WEIGHT_ROOT = PROJECT_ROOT / "weights"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATASET_NAMES = ("Houston2013", "Houston2018", "Trento", "Muufl")

# -------------------------- Palettes / class names --------------------------
def _hex_to_rgb(h: str):
    h = h.strip().lstrip('#')
    if len(h) != 6:
        raise ValueError(f"Bad hex color: {h}")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]

def _hex_list_to_rgb_list(hex_list):
    return [_hex_to_rgb(h) for h in hex_list]

PALETTES: Dict[str, Dict[str, Any]] = {
    "Houston2013": dict(
        class_names=[
            "Healthy Grass", "Stressed Grass", "Synthetic Grass", "Trees", "Soil",
            "Water", "Residential", "Commercial", "Road", "Highway",
            "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track",
        ],
        hex_colors=[
            "#4DAF4A", "#A6D854", "#2C7A3F", "#166C2B", "#8C4B2D",
            "#71C6D4", "#8DA0CB", "#C8B4D9", "#D73027", "#8B1A1A",
            "#1F3D99", "#F0E442", "#F39C12", "#5E3AA8", "#D95A48",
        ],
        num_classes=15,
    ),
    "Houston2018": dict(
        class_names=[
            "Healthy grass", "Stressed grass", "Artificial turf", "Evergreen trees", "Deciduous trees",
            "Bare earth", "Water", "Residential buildings", "Non-residential bldgs", "Roads",
            "Sidewalks", "Crosswalks", "Major thoroughfares", "Highways", "Railways",
            "Paved parking lots", "Unpaved parking lots", "Cars", "Trains", "Stadium seats",
        ],
        hex_colors=[
            "#0B3C5D", "#114E7A", "#16639A", "#1B74B5", "#2586CC",
            "#3998D5", "#5AAAD8", "#7DBBE0", "#A6CEE3", "#CFE3F2",
            "#FFF7BC", "#FEE391", "#FEC44F", "#FE9929", "#EC7014",
            "#CC4C02", "#B36B00", "#D9A400", "#F2C84B", "#FFE680",
        ],
        num_classes=20,
    ),
    "Muufl": dict(
        class_names=[
            "Trees", "Grass ground", "Mixed ground surface", "Dirt and sand", "Road",
            "Water", "Buildings", "Shadow", "Sidewalk", "Yellow curb", "Cloth panels",
        ],
        hex_colors=[
            "#2C2E83", "#4EA3D9", "#1FA2A7", "#2DB6A3", "#6FBF73",
            "#1CB5E0", "#F6B26B", "#8E7CC3", "#FFD95A", "#F4B942", "#FFE680",
        ],
        num_classes=11,
    ),
    "Trento": dict(
        class_names=["Apple trees", "Buildings", "Ground", "Woods", "Vineyard", "Roads"],
        hex_colors=["#2C7BB6", "#22B5C1", "#1FA8A0", "#97B66A", "#FFE680", "#2C2E83"],
        num_classes=6,
    ),
}

CLASS_NAMES = {k: v["class_names"] for k, v in PALETTES.items()}
COLOR_MAPS = {k: _hex_list_to_rgb_list(v["hex_colors"]) for k, v in PALETTES.items()}

# -------------------------- Dataset paths / run paths --------------------------
DATASETS: Dict[str, Dict[str, Any]] = {
    "Houston2013": {
        "image_size": (349, 1905),
        "num_classes": 15,
        "paths": {
            "hsi": DATA_ROOT / "Houston2013" / "Houston2013_hsi.mat",
            "lidar": DATA_ROOT / "Houston2013" / "Houston2013_lidar.mat",
            "gt": DATA_ROOT / "Houston2013" / "Houston2013_gt.mat",
            "index": DATA_ROOT / "Houston2013" / "Houston2013_index.mat",
        },
        "keys": ["Houston2013_hsi", "Houston2013_lidar", "Houston2013_gt", "Houston2013_train", "Houston2013_test", "Houston2013_all"],
        "normalize_lidar": True,
    },
    "Houston2018": {
        "image_size": (1202, 4768),
        "num_classes": 20,
        "paths": {
            "hsi": DATA_ROOT / "Houston2018" / "Houston2018_hsi.mat",
            "lidar": DATA_ROOT / "Houston2018" / "Houston2018_lidar.mat",
            "gt": DATA_ROOT / "Houston2018" / "Houston2018_gt.mat",
            "index": DATA_ROOT / "Houston2018" / "Houston2018_index.mat",
        },
        "keys": ["Houston2018_hsi", "Houston2018_lidar", "Houston2018_gt", "Houston2018_train", "Houston2018_test", "Houston2018_all"],
        "normalize_lidar": False,
    },
    "Muufl": {
        "image_size": (325, 220),
        "num_classes": 11,
        "paths": {
            "hsi": DATA_ROOT / "Muufl" / "Muufl_hsi.mat",
            "lidar": DATA_ROOT / "Muufl" / "Muufl_lidar.mat",
            "gt": DATA_ROOT / "Muufl" / "Muufl_gt.mat",
            "index": DATA_ROOT / "Muufl" / "Muufl_index.mat",
        },
        "keys": ["Muufl_hsi", "Muufl_lidar", "Muufl_gt", "Muufl_train", "Muufl_test", "Muufl_all"],
        "normalize_lidar": True,
    },
    "Trento": {
        "image_size": (166, 600),
        "num_classes": 6,
        "paths": {
            "hsi": DATA_ROOT / "Trento" / "Trento_hsi.mat",
            "lidar": DATA_ROOT / "Trento" / "Trento_lidar.mat",
            "gt": DATA_ROOT / "Trento" / "Trento_gt.mat",
            "index": DATA_ROOT / "Trento" / "Trento_index.mat",
        },
        "keys": ["Trento_hsi", "Trento_lidar", "Trento_gt", "Trento_train", "Trento_test", "Trento_all"],
        "normalize_lidar": True,
    },
}

DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Houston2013": dict(channels=21, hsi_windowSize=11, embed_size=64, uni_dimension=64, size_SA=25, lidar_channel=64, hsi_channel=56),
    "Houston2018": dict(channels=18, hsi_windowSize=23, embed_size=64, uni_dimension=64, size_SA=25, lidar_channel=64, hsi_channel=56),
    "Muufl":       dict(channels=9,  hsi_windowSize=11, embed_size=64, uni_dimension=64, size_SA=25, lidar_channel=64, hsi_channel=56),
    "Trento":      dict(channels=18, hsi_windowSize=19, embed_size=32, uni_dimension=64, size_SA=25, lidar_channel=64, hsi_channel=56),
}

RUN_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Houston2013": {
        "default_mode": "train",
        "result_root": OUTPUT_ROOT / "Houston2013",
        "intrinsic_ckpt": WEIGHT_ROOT / "Houston2013" / "intrinsic_decomposition.pth",
        "ddpm_ckpt": WEIGHT_ROOT / "Houston2013" / "spectral_prior_ddpm.pt",
        "default_test_ckpt": OUTPUT_ROOT / "Houston2013" / "best.pth",
        "prefer_metric": "OA",
    },
    "Houston2018": {
        "default_mode": "train",
        "result_root": OUTPUT_ROOT / "Houston2018",
        "intrinsic_ckpt": WEIGHT_ROOT / "Houston2018" / "intrinsic_decomposition.pth",
        "ddpm_ckpt": WEIGHT_ROOT / "Houston2018" / "spectral_prior_ddpm.pt",
        "default_test_ckpt": OUTPUT_ROOT / "Houston2018" / "best.pth",
        "prefer_metric": "OA",
    },
    "Muufl": {
        "default_mode": "train",
        "result_root": OUTPUT_ROOT / "Muufl",
        "intrinsic_ckpt": WEIGHT_ROOT / "Muufl" / "intrinsic_decomposition.pth",
        "ddpm_ckpt": WEIGHT_ROOT / "Muufl" / "spectral_prior_ddpm.pt",
        "default_test_ckpt": OUTPUT_ROOT / "Muufl" / "best.pth",
        "prefer_metric": "OA",
    },
    "Trento": {
        "default_mode": "train",
        "result_root": OUTPUT_ROOT / "Trento",
        "intrinsic_ckpt": WEIGHT_ROOT / "Trento" / "intrinsic_decomposition.pth",
        "ddpm_ckpt": WEIGHT_ROOT / "Trento" / "spectral_prior_ddpm.pt",
        "default_test_ckpt": OUTPUT_ROOT / "Trento" / "best.pth",
        "prefer_metric": "OA",
    },
}

# -------------------------- CLI / defaults --------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DRIDNet")
    parser.add_argument('--mode', choices=['train', 'test', 'full_pre'], default=None,
                        help='train, test, or full_pre. The default comes from RUN_CONFIGS.')
    parser.add_argument('--dataset', choices=list(DATASET_NAMES), default='Houston2013')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=1)

    # Dataset/model shape parameters. None means DATASET_DEFAULTS is used.
    parser.add_argument('--channels', type=int, default=None)
    parser.add_argument('--hsi_windowSize', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--uni_dimension', type=int, default=None)
    parser.add_argument('--embed_size', type=int, default=None)
    parser.add_argument('--hid_router', type=int, default=256)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--size_SA', type=int, default=None)
    parser.add_argument('--lidar_channel', type=int, default=None)
    parser.add_argument('--hsi_channel', type=int, default=None)
    parser.add_argument('--router_eps', type=float, default=1000)
    parser.add_argument('--lambda_gen', type=float, default=0.0)
    parser.add_argument('--intrinsic_gate_logit', type=float, default=-1.0)
    parser.add_argument('--trento_spec_groups', type=int, default=-1)

    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=100)

    # Decomposition / prior losses
    parser.add_argument('--lambda_recon', type=float, default=0.0)
    parser.add_argument('--lambda_spec', type=float, default=0.0)
    parser.add_argument('--lambda_spec2', type=float, default=0.0)
    parser.add_argument('--lambda_local', type=float, default=0.0)
    parser.add_argument('--lambda_dsm', type=float, default=0.0)
    parser.add_argument('--lambda_anchor', type=float, default=0.0)
    parser.add_argument('--prior_refine_steps', type=int, default=0)
    parser.add_argument('--prior_loss_pixels', type=int, default=2048)
    parser.add_argument('--intrinsic_mode', choices=['blend', 'none'], default='blend')
    parser.add_argument('--balanced_sampling', action='store_true', default=True)

    # Runtime paths. None means RUN_CONFIGS is used.
    parser.add_argument('--result_root', type=str, default=None)
    parser.add_argument('--intrinsic_ckpt', type=str, default=None)
    parser.add_argument('--ddpm_ckpt', type=str, default=None)
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    return parser

def apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    ds = args.dataset
    if ds not in DATASETS:
        raise ValueError(f"Unsupported dataset={ds}; supported={list(DATASETS)}")

    # Fill dataset-specific default model/data parameters.
    preset = DATASET_DEFAULTS[ds]
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.num_classes is None:
        args.num_classes = int(DATASETS[ds]["num_classes"])

    run_cfg = RUN_CONFIGS[ds]
    if args.mode is None:
        args.mode = run_cfg.get("default_mode", "full_pre")
    if args.result_root is None:
        args.result_root = str(run_cfg["result_root"])
    if args.intrinsic_ckpt is None:
        p = run_cfg.get("intrinsic_ckpt", None)
        args.intrinsic_ckpt = str(p) if p is not None and Path(p).exists() else None
    if args.ddpm_ckpt is None:
        p = run_cfg.get("ddpm_ckpt", None)
        args.ddpm_ckpt = str(p) if p is not None and Path(p).exists() else None
    if args.test_ckpt is None:
        p = run_cfg.get("default_test_ckpt", None)
        args.test_ckpt = str(p) if p is not None else None
    args.prefer_metric = run_cfg.get("prefer_metric", "OA")
    return args

def parse_args(argv=None) -> argparse.Namespace:
    args, _ = build_argparser().parse_known_args(argv)
    return apply_dataset_defaults(args)

def get_dataset_spec(name: str) -> Dict[str, Any]:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASETS[name]

def get_color_map(name: str):
    return COLOR_MAPS[name]
