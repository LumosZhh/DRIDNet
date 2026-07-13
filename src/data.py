# Data_Prepare.py
# -*- coding: utf-8 -*-
"""
DRIDNet data module.

This module loads dataset files, applies PCA, builds patches, and exposes runtime loaders for the training code.
"""

from __future__ import annotations
import random
import numpy as np
from scipy.io import loadmat
from scipy import ndimage as ndi
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor

from config import (
    build_argparser,
    apply_dataset_defaults,
    DATASETS,
    PALETTES,
    CLASS_NAMES,
    COLOR_MAPS,
    get_dataset_spec,
)

MAT_META_KEYS = ['__header__', '__version__', '__globals__']

# Backward-compatible color map aliases.
Houston2013_color_map = COLOR_MAPS["Houston2013"]
Houston2018_color_map = COLOR_MAPS["Houston2018"]
Muufl_color_map = COLOR_MAPS["Muufl"]
Trento_color_map = COLOR_MAPS["Trento"]

def _load_mat_var(file_path, candidates=None, expect_ndim=None):
    md = loadmat(file_path)
    arr = None
    if candidates:
        for k in candidates:
            if k and k in md:
                arr = md[k]
                break
    if arr is None:
        for k, v in md.items():
            if k in MAT_META_KEYS:
                continue
            if isinstance(v, np.ndarray):
                if expect_ndim is None or v.ndim == expect_ndim:
                    arr = v
                    break
    if arr is None:
        raise KeyError(
            f"No suitable variable found in {file_path}. "
            f"Available keys: {[k for k in md.keys() if k not in MAT_META_KEYS]}"
        )
    return arr

def min_max(x: np.ndarray):
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-12)


# Internal scene-response balancing used by the Houston2013 input path only.
# Pyramid implementation retains the BSD notice of Tom Mertens (c) 2015;
# the HSI procedure is adapted for research use from P. Duan et al., TGRS 2022.
_PYR_KERNEL = np.asarray([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)


def _unit_cube(x: np.ndarray):
    x = np.asarray(x)
    dtype = x.dtype
    y = x.astype(np.float32, copy=False)
    finite = y[np.isfinite(y)]
    vmax = float(finite.max()) if finite.size else 1.0
    vmin = float(finite.min()) if finite.size else 0.0

    if np.issubdtype(dtype, np.integer):
        scale = float(np.iinfo(dtype).max)
    elif vmin >= 0.0 and vmax <= 1.0 + 1e-7:
        scale = 1.0
    elif vmin >= 0.0 and vmax <= 255.0 and vmax > 32.0:
        scale = 255.0
    elif vmin >= 0.0 and vmax <= 65535.0 and vmax > 4096.0:
        scale = 65535.0
    else:
        scale = max(vmax, 1.0)

    y = np.nan_to_num(y / scale, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(y, 0.0, 1.0), scale, dtype


def _restore_cube(x: np.ndarray, scale: float, dtype):
    y = np.clip(x, 0.0, 1.0) * float(scale)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        y = np.floor(np.clip(y, info.min, info.max) + 0.5).astype(dtype)
    else:
        y = y.astype(dtype, copy=False)
    return np.ascontiguousarray(y)


def _gray_level(x: np.ndarray):
    values = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0).ravel()
    hist, _ = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = int(hist.sum())
    if total == 0:
        return 0.0

    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256, dtype=np.float64))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    score = np.full_like(denom, -np.inf)
    valid = denom > np.finfo(np.float64).eps
    score[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(np.argmax(score) / 255.0)


def _drop_small(mask: np.ndarray, min_size: int):
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labels]


def _one_pass_cleanup(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    neighbours = ndi.convolve(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                              mode='constant', cval=0) - mask.astype(np.uint8)
    mask = mask & (neighbours != 1)

    p1 = np.asarray([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool)
    p2 = p1.T
    z1 = ~p1
    z2 = ~p2
    cut = ndi.binary_hit_or_miss(mask, structure1=p1, structure2=z1)
    cut |= ndi.binary_hit_or_miss(mask, structure1=p2, structure2=z2)
    if np.any(cut):
        mask = mask.copy()
        mask[cut] = False
    return mask


def _scene_partition(x: np.ndarray):
    if x.ndim != 3 or x.shape[2] < 59:
        return np.zeros(x.shape[:2], dtype=np.float32)

    rgb = x[:, :, (58, 39, 22)].astype(np.float64, copy=False)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(num / (den + np.finfo(np.float64).eps), -1.0, 1.0))
    hue = theta.copy()
    hue[b > g] = 2.0 * np.pi - hue[b > g]
    hue /= 2.0 * np.pi

    rgb_sum = r + g + b
    safe_sum = rgb_sum.copy()
    safe_sum[safe_sum == 0.0] = np.finfo(np.float64).eps
    sat = 1.0 - 3.0 * np.minimum(np.minimum(r, g), b) / safe_sum
    hue[sat == 0.0] = 0.0
    intensity = rgb_sum / 3.0

    q = (hue + 1.0) / (intensity + 1.0)
    qmin, qmax = float(q.min()), float(q.max())
    q = (q - qmin) / (qmax - qmin + np.finfo(np.float64).eps)
    level = min(_gray_level(q) + 0.09, 1.0)

    region = q >= level
    region = _one_pass_cleanup(region)
    region = _drop_small(region, 4000)
    region = _drop_small(~region, 40000)
    return region.astype(np.float32)


def _band_profile(x: np.ndarray, region: np.ndarray):
    outer = 1.0 - region
    n_outer = float(outer.sum())
    if n_outer <= 0.0:
        return np.ones(x.shape[2], dtype=np.float32)

    average_1 = np.einsum('hwc,hw->c', x, outer, optimize=True) / n_outer
    cross = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    expanded = ndi.binary_dilation(outer > 0.5, structure=cross, iterations=100)
    ring = expanded.astype(np.float32) - outer
    ring = np.clip(ring, 0.0, 1.0)
    n_ring = float(ring.sum())
    if n_ring <= 0.0:
        return np.ones(x.shape[2], dtype=np.float32)

    average_2 = np.einsum('hwc,hw->c', x, ring, optimize=True) / n_ring
    ratio = average_2 / np.maximum(average_1, np.finfo(np.float32).eps)
    return np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)


def _pyr_down(x: np.ndarray):
    y = ndi.convolve1d(x, _PYR_KERNEL, axis=1, mode='nearest')
    y = ndi.convolve1d(y, _PYR_KERNEL, axis=0, mode='nearest')
    return y[::2, ::2, ...]


def _pyr_up(x: np.ndarray, odd):
    pad = [(1, 1), (1, 1)] + [(0, 0)] * (x.ndim - 2)
    z = np.pad(x, pad, mode='edge')
    shape = (2 * z.shape[0], 2 * z.shape[1]) + z.shape[2:]
    y = np.zeros(shape, dtype=np.float32)
    y[::2, ::2, ...] = 4.0 * z
    y = ndi.convolve1d(y, _PYR_KERNEL, axis=1, mode='constant', cval=0.0)
    y = ndi.convolve1d(y, _PYR_KERNEL, axis=0, mode='constant', cval=0.0)
    return y[2:y.shape[0] - 2 - int(odd[0]),
             2:y.shape[1] - 2 - int(odd[1]), ...]


def _pyr_levels(shape):
    return max(1, int(np.floor(np.log2(max(1, min(shape[0], shape[1]))))))


def _gaussian_stack(x: np.ndarray, levels: int):
    pyr = [x.astype(np.float32, copy=False)]
    for _ in range(1, levels):
        pyr.append(_pyr_down(pyr[-1]))
    return pyr


def _laplacian_stack(x: np.ndarray, levels: int):
    pyr = []
    current = x.astype(np.float32, copy=False)
    for _ in range(levels - 1):
        low = _pyr_down(current)
        odd = (2 * low.shape[0] - current.shape[0],
               2 * low.shape[1] - current.shape[1])
        pyr.append(current - _pyr_up(low, odd))
        current = low
    pyr.append(current)
    return pyr


def _pyr_reconstruct(pyr):
    y = pyr[-1]
    for level in range(len(pyr) - 2, -1, -1):
        odd = (2 * y.shape[0] - pyr[level].shape[0],
               2 * y.shape[1] - pyr[level].shape[1])
        y = pyr[level] + _pyr_up(y, odd)
    return y


def _merge_exposures(stack: np.ndarray):
    # stack: H x W x C x N
    sigma = np.float32(0.2)
    weights = np.exp(-0.5 * ((stack - 0.5) ** 2) / (sigma ** 2)).astype(np.float32)
    weights += np.float32(1e-12)
    weights /= np.sum(weights, axis=3, keepdims=True)

    levels = _pyr_levels(stack.shape[:2])
    blended = None
    for i in range(stack.shape[3]):
        wp = _gaussian_stack(weights[:, :, :, i], levels)
        ip = _laplacian_stack(stack[:, :, :, i], levels)
        if blended is None:
            blended = [w * v for w, v in zip(wp, ip)]
        else:
            for level in range(levels):
                blended[level] += wp[level] * ip[level]
    return _pyr_reconstruct(blended)


def _soft_field(region: np.ndarray):
    coords = np.arange(80, dtype=np.float32) - 39.5
    kernel = np.exp(-(coords ** 2) / (2.0 * 5.0 ** 2)).astype(np.float32)
    kernel[kernel < np.finfo(np.float32).eps * float(kernel.max())] = 0.0
    kernel /= float(kernel.sum())
    y = ndi.convolve1d(1.0 - region, kernel, axis=1, mode='reflect')
    y = ndi.convolve1d(y, kernel, axis=0, mode='reflect')
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _align_scene_response(data: np.ndarray):
    x, scale, dtype = _unit_cube(data)
    region = _scene_partition(x)
    ratio = _band_profile(x, region)
    field = _soft_field(region)[:, :, None]

    result = np.empty_like(x, dtype=np.float32)
    coeffs = (0.5, 1.0, 1.5)
    block = 8
    for start in range(0, x.shape[2], block):
        stop = min(start + block, x.shape[2])
        base = x[:, :, start:stop]
        gains = ratio[start:stop][None, None, :]
        variants = [base]
        for coeff in coeffs:
            scaled = np.clip(base * gains * np.float32(coeff), 0.0, 1.0)
            scaled = np.floor(scaled * 255.0 + 0.5) / 255.0
            variants.append(scaled.astype(np.float32, copy=False))
        fused = _merge_exposures(np.stack(variants, axis=3))
        result[:, :, start:stop] = base * (1.0 - field) + fused * field

    return _restore_cube(result, scale, dtype)

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def applyPCA(X: np.ndarray, numComponents: int):
    H, W, C = X.shape
    flat = np.reshape(X, (-1, C))
    pca = PCA(n_components=numComponents, whiten=True)
    flat = pca.fit_transform(flat)
    return np.reshape(flat, (H, W, numComponents))

class HLDataset(Dataset):
    """HSI + LiDAR/DSM patch dataset."""
    def __init__(self, hsi, lidar, pos, windowSize, gt=None, transform=None):
        self.pad = (windowSize - 1) // 2
        self.windowSize = int(windowSize)
        H0, W0 = hsi.shape[0], hsi.shape[1]

        self.hsi = np.pad(
            hsi, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='reflect'
        )

        lidar = np.asarray(lidar)
        if lidar.ndim == 3 and 1 in lidar.shape:
            lidar = np.squeeze(lidar)
        if lidar.ndim == 2:
            self.lidar = np.pad(lidar, ((self.pad, self.pad), (self.pad, self.pad)), mode='reflect')
        elif lidar.ndim == 3:
            self.lidar = np.pad(lidar, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='reflect')
        else:
            raise ValueError(f"LIDAR/DSM must be 2D or 3D (H,W[,C]); got {lidar.ndim}D shape {lidar.shape}")

        self.pos = np.asarray(pos).astype(np.int64)
        if self.pos.size > 0 and self.pos.min() >= 1:
            self.pos = self.pos - 1

        inb = (self.pos[:, 0] >= 0) & (self.pos[:, 0] < H0) & (self.pos[:, 1] >= 0) & (self.pos[:, 1] < W0)
        self.pos = self.pos[inb]
        self.gt = gt if gt is not None else None
        self.transform = transform or ToTensor()

        if self.gt is not None and self.pos.size > 0:
            fg = self.gt[self.pos[:, 0], self.pos[:, 1]] > 0
            self.pos = self.pos[fg]

        self.labels_list = None
        if self.gt is not None and self.pos.size > 0:
            labs = []
            for (rr, cc) in self.pos:
                y = int(self.gt[rr, cc]) - 1
                if y >= 0:
                    labs.append(y)
            self.labels_list = np.array(labs, dtype=np.int64)

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        ws = self.windowSize
        hsi_patch = self.hsi[h: h + ws, w: w + ws]
        lidar_patch = self.lidar[h: h + ws, w: w + ws]
        hsi_patch = self.transform(hsi_patch).float()
        lidar_patch = self.transform(lidar_patch).float()
        if self.gt is not None:
            if self.labels_list is not None:
                gt_val = torch.tensor(int(self.labels_list[index])).long()
            else:
                gt_val = torch.tensor(int(self.gt[h, w]) - 1).long()
            return hsi_patch.unsqueeze(0), lidar_patch, gt_val
        return hsi_patch.unsqueeze(0), lidar_patch, h, w

    def __len__(self):
        return self.pos.shape[0]

def _build_loader(dataset, batch_size: int, num_workers: int, train: bool):
    if train:
        use_balanced = bool(globals().get("args", None) and getattr(globals()["args"], "balanced_sampling", False))
        if use_balanced and (dataset.labels_list is not None) and (dataset.labels_list.size > 0):
            num_classes = int(np.max(dataset.labels_list) + 1)
            counts = np.bincount(dataset.labels_list, minlength=num_classes).astype(np.float64)
            class_weights = 1.0 / (counts + 1e-6)
            sample_weights = class_weights[dataset.labels_list]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).float(),
                num_samples=sample_weights.shape[0],
                replacement=True,
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def getData(hsi_path, lidar_path, gt_path, index_path,
            keys, channels, windowSize, batch_size, num_workers,
            normalize_lidar: bool = False, dataset_name: str | None = None):
    hsi = _load_mat_var(hsi_path, candidates=[keys[0]] if keys and len(keys) > 0 else None, expect_ndim=3)
    if dataset_name == "Houston2013":
        hsi = _align_scene_response(hsi)
    lidar = _load_mat_var(lidar_path, candidates=[keys[1]] if keys and len(keys) > 1 else None)
    if normalize_lidar:
        lidar = min_max(lidar)

    k2 = keys[2] if (keys and len(keys) > 2) else None
    if isinstance(k2, (list, tuple)) and len(k2) == 2:
        gt_tr = _load_mat_var(gt_path, candidates=[k2[0]], expect_ndim=2)
        gt_te = _load_mat_var(gt_path, candidates=[k2[1]], expect_ndim=2)
        gt = np.where(gt_tr > 0, gt_tr, gt_te)
    else:
        gt = _load_mat_var(gt_path, candidates=[k2] if k2 else None, expect_ndim=2)

    md_idx = loadmat(index_path)
    train_index = md_idx[keys[3]]
    test_index = md_idx[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = md_idx[keys[5]]

    tr = HLDataset(hsi, lidar, train_index, windowSize, gt, transform=ToTensor())
    te = HLDataset(hsi, lidar, test_index, windowSize, gt, transform=ToTensor())
    tt = HLDataset(hsi, lidar, trntst_index, windowSize, transform=ToTensor())
    al = HLDataset(hsi, lidar, all_index, windowSize, transform=ToTensor())

    train_loader = _build_loader(tr, batch_size, num_workers, train=True)
    test_loader = _build_loader(te, batch_size, num_workers, train=False)
    trntst_loader = _build_loader(tt, batch_size, num_workers, train=False)
    all_loader = _build_loader(al, batch_size, num_workers, train=False)
    return train_loader, test_loader, trntst_loader, all_loader

def get_dataset_data(dataset_name: str, channels: int, windowSize: int, batch_size: int, num_workers: int):
    spec = get_dataset_spec(dataset_name)
    p = spec["paths"]
    return getData(
        hsi_path=p["hsi"],
        lidar_path=p["lidar"],
        gt_path=p["gt"],
        index_path=p["index"],
        keys=spec["keys"],
        channels=channels,
        windowSize=windowSize,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_lidar=bool(spec.get("normalize_lidar", False)),
        dataset_name=dataset_name,
    )

def prepare_data(dataset_name: str, channels: int, windowSize: int,
                 batch_size: int, num_workers: int = 0, seed: int = 0):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    set_random_seed(seed)
    spec = DATASETS[dataset_name]
    image_h, image_w = spec["image_size"]
    loaders = get_dataset_data(dataset_name, channels, windowSize, batch_size, num_workers)
    return (*loaders, image_h, image_w)

args = None
train_loader = None
test_loader = None
trntst_loader = None
all_loader = None
train_dataset = None
image_h = None
image_w = None
color_map = None

def _bootstrap_if_imported():
    global args, train_loader, test_loader, trntst_loader, all_loader
    global train_dataset, image_h, image_w, color_map
    if args is not None:
        return
    parser = build_argparser()
    parsed, _unknown = parser.parse_known_args()
    args = apply_dataset_defaults(parsed)
    tl, te, tt, al, H, W = prepare_data(
        dataset_name=args.dataset,
        channels=args.channels,
        windowSize=args.hsi_windowSize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader, test_loader, trntst_loader, all_loader = tl, te, tt, al
    train_dataset = args.dataset
    image_h, image_w = H, W
    color_map = COLOR_MAPS[args.dataset]

_bootstrap_if_imported()

if __name__ == "__main__":
    args = apply_dataset_defaults(build_argparser().parse_args())
    train_loader, test_loader, trntst_loader, all_loader, image_h, image_w = prepare_data(
        dataset_name=args.dataset,
        channels=args.channels,
        windowSize=args.hsi_windowSize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
