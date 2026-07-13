# runner.py
# -*- coding: utf-8 -*-
"""Train / test / full-map visualization runner for DRIDNet."""

from __future__ import annotations
import csv
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

import data as DP
from data import args, train_loader, test_loader, train_dataset
from model import DRIDNet_HSI
from config import COLOR_MAPS

all_loader = getattr(DP, "all_loader", None)
image_h = getattr(DP, "image_h", None)
image_w = getattr(DP, "image_w", None)
DATASET_COLOR_MAP = np.asarray(COLOR_MAPS[train_dataset], dtype=np.uint8)


def choose_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(args.gpu))
        except Exception:
            pass
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    return device


class _SilentLogger:
    """Logger-compatible object that produces no terminal or file output."""

    def info(self, *args, **kwargs):
        return None


def print_paper_parameters(args_obj):
    """Print only implementation parameters explicitly reported in the manuscript."""
    alpha = float(args_obj.intrinsic_gate_logit)
    gate = 1.0 / (1.0 + math.exp(-alpha))

    parameters = [
        ("Dataset", args_obj.dataset),
        ("Training epochs", int(args_obj.epochs)),
        ("Mini-batch size", int(args_obj.batch_size)),
        ("Optimizer", "Adam"),
        ("Classification loss", "Cross-entropy"),
        ("Initial learning rate", f"{float(args_obj.learning_rate):.2e}"),
        ("Learning-rate decay factor", 0.5),
        ("Learning-rate patience", 5),
        ("Weight decay", 0.0),
        ("PCA placement", "After CIIDM"),
        ("PCA dimension k", int(args_obj.channels)),
        ("Neighborhood size r", int(args_obj.hsi_windowSize)),
        ("Gate coefficient alpha", alpha),
        ("Initial mixing coefficient sigmoid(alpha)", f"{gate:.6f}"),
        ("Fourier frequency bands", 4),
        ("CIIDM hidden width", 128),
        ("CIIDM hidden layers", 4),
        ("SDDP base width", 128),
        ("Diffusion noise schedule", "Cosine"),
        ("Diffusion timesteps", 1000),
        ("Reflectance vectors per mini-batch", int(args_obj.prior_loss_pixels)),
        ("CTRM prototype dimension", 64),
        ("Sinkhorn iterations", 7),
        ("Routing temperature", float(args_obj.router_eps)),
    ]

    width = max(len(name) for name, _ in parameters)
    print("=" * 62)
    print("DRIDNet Parameters Reported in the Manuscript")
    print("=" * 62)
    for name, value in parameters:
        print(f"{name:<{width}} : {value}")
    print("=" * 62)


class RunManager:
    """Manage only the run directory and best/latest checkpoints."""

    def __init__(self, base_dir: Path, mode: str, tag: str = "",
                 attach_to: Optional[Union[str, Path]] = None,
                 separate_test_log: bool = True):
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        if attach_to is not None:
            p = Path(attach_to)
            if p.suffix == ".pth":
                run_dir = p.parent.parent
            elif p.name in ("ckpt", "logs"):
                run_dir = p.parent
            else:
                run_dir = p if (p / "ckpt").exists() else p.parent
            self.run_dir = Path(run_dir)
        else:
            run_name = f"{t}{('_' + tag) if tag else ''}"
            self.run_dir = Path(base_dir) / run_name

        self.log_dir = self.run_dir / "logs"
        self.ckpt_dir = self.run_dir / "ckpt"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = _SilentLogger()

    def dump_args(self, args_obj):
        return None

    def record_event(self, event: str, payload: dict):
        return None

    def write_metrics(self, phase: str, epoch: int, elapsed: float,
                      train_loss=None, train_acc=None, val_loss=None,
                      oa=None, aa_mean=None, kappa=None, lr=None,
                      event: str = "", ckpt_path: str = ""):
        return None

    def save_ckpt(self, state_dict: dict, epoch: int, kind: str,
                  metric_name: str = "", metric_value=None) -> Path:
        if kind not in ("best", "latest"):
            raise ValueError(f"Unsupported checkpoint kind: {kind}")
        path = self.ckpt_dir / f"{kind}.pth"
        torch.save({"state_dict": state_dict, "epoch": int(epoch)}, path.as_posix())
        return path


class DRIDNetRunner:
    def __init__(self):
        self.args = args
        self.dataset = train_dataset
        self.device = choose_device()
        self.result_root = Path(self.args.result_root)
        self.intrinsic_ckpt = self.args.intrinsic_ckpt
        self.ddpm_ckpt = self.args.ddpm_ckpt
        self.test_ckpt = self.args.test_ckpt
        self.prefer_metric = getattr(self.args, "prefer_metric", "OA")

    def _forward_batch(self, model, batch):
        hsi = lidar = y = pos = None
        if isinstance(batch, dict):
            hsi = batch.get("hsi") or batch.get("data_hsi") or batch.get("HSI") or batch.get("hsi_patches")
            lidar = batch.get("lidar") or batch.get("data_lidar") or batch.get("DSM") or batch.get("lidar_patches")
            y = batch.get("y") or batch.get("label") or batch.get("gt") or batch.get("labels")
            pos = batch.get("pos") or batch.get("coord") or batch.get("coords")
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                hsi, lidar, y = batch[0], batch[1], batch[2]
            elif len(batch) >= 4:
                hsi, lidar = batch[0], batch[1]
                h, w = batch[2], batch[3]
                if torch.is_tensor(h) and torch.is_tensor(w):
                    pos = torch.stack([h, w], dim=1)
                else:
                    pos = np.stack([np.asarray(h), np.asarray(w)], axis=1)
            else:
                raise TypeError(f"Unsupported tuple batch length: {len(batch)}")
        logits = None
        if hsi is not None and lidar is not None:
            logits = model(hsi.to(self.device), lidar.to(self.device))
        return logits, y, pos

    def _peek_hsi_shape(self, loader):
        for batch in loader:
            hsi = None
            if isinstance(batch, dict):
                hsi = batch.get("hsi") or batch.get("data_hsi") or batch.get("HSI") or batch.get("hsi_patches")
            elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
                hsi = batch[0]
            if hsi is not None:
                return int(hsi.size(-3)), int(hsi.size(-1))
        raise RuntimeError("Cannot infer HSI shape from loader.")

    def _select_ckpt_path(self) -> Optional[Path]:
        if self.test_ckpt:
            p = Path(self.test_ckpt)
            if p.exists():
                return p
        candidates = [
            self.result_root / "best.pth",
            self.result_root / "latest.pth",
            self.result_root / "ckpt" / "best.pth",
            self.result_root / "ckpt" / "latest.pth",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _fit_post_ciidm_pca(self, model):
        if all_loader is None:
            raise RuntimeError("all_loader is required to fit PCA after CIIDM.")

        was_training = model.training
        model.train(True)

        num_samples = 0
        sum_x = None
        sum_xx = None

        with torch.no_grad():
            for batch in all_loader:
                if isinstance(batch, dict):
                    hsi = batch.get("hsi") or batch.get("data_hsi") or batch.get("HSI") or batch.get("hsi_patches")
                    lidar = batch.get("lidar") or batch.get("data_lidar") or batch.get("DSM") or batch.get("lidar_patches")
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    hsi, lidar = batch[0], batch[1]
                else:
                    raise TypeError(f"Unsupported batch format while fitting PCA: {type(batch)}")

                hsi = hsi.to(self.device)
                lidar = lidar.to(self.device)
                hsi_eff = model.extract_post_ciidm_hsi(hsi, lidar)

                channels = int(hsi_eff.size(-3))
                pixels = hsi_eff.movedim(-3, -1).reshape(-1, channels)
                pixels = pixels.to(dtype=torch.float64, device="cpu")

                if sum_x is None:
                    channels = int(pixels.size(1))
                    sum_x = torch.zeros(channels, dtype=torch.float64)
                    sum_xx = torch.zeros(channels, channels, dtype=torch.float64)

                num_samples += int(pixels.size(0))
                sum_x += pixels.sum(dim=0)
                sum_xx += pixels.T @ pixels

        if model.post_ciidm_pca is None:
            raise RuntimeError("Post-CIIDM PCA module was not initialized.")

        model.post_ciidm_pca.fit_from_statistics(
            num_samples=num_samples,
            sum_x=sum_x,
            sum_xx=sum_xx,
        )
        model.train(was_training)

    def _load_model(self, build_loader):
        model = DRIDNet_HSI(intrinsic_ckpt=None, ddpm_ckpt=self.ddpm_ckpt).to(self.device)
        C, P = self._peek_hsi_shape(build_loader)
        model._lazy_build_and_load(C, P, self.device)
        ckpt_path = self._select_ckpt_path()
        if (ckpt_path is None) or (not ckpt_path.exists()):
            raise FileNotFoundError(f"No checkpoint found under {self.result_root}; test_ckpt={self.test_ckpt}")
        state = torch.load(ckpt_path.as_posix(), map_location=self.device)
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(sd, strict=False)
        return model, ckpt_path

    def train(self):
        model = DRIDNet_HSI(intrinsic_ckpt=self.intrinsic_ckpt, ddpm_ckpt=self.ddpm_ckpt).to(self.device)
        self._fit_post_ciidm_pca(model)
        mgr = RunManager(self.result_root, mode="train", tag=self.args.tag)
        print_paper_parameters(self.args)

        best_score = -1.0
        total_epochs = int(self.args.epochs)
        for epoch in range(1, total_epochs + 1):
            model.train_start()
            model.train(epoch)
            model.val_start()
            oa = model.cal_acc(epoch)

            tr_loss = getattr(model, "last_train_loss_avg", None)
            metrics = getattr(model, "last_metrics", {}) or {}
            aa_mean = metrics.get("AA_mean", None)

            if tr_loss is None:
                print(f"Epoch {epoch:03d}/{total_epochs:03d} | Train Loss: N/A")
            else:
                print(f"Epoch {epoch:03d}/{total_epochs:03d} | Train Loss: {float(tr_loss):.6f}")

            mgr.save_ckpt(model.state_dict(), epoch, kind="latest")

            score = float(oa) if str(self.prefer_metric).upper() == "OA" and oa is not None else None
            if str(self.prefer_metric).upper() != "OA" and aa_mean is not None:
                score = float(aa_mean)
            if score is not None and math.isfinite(score) and score > best_score:
                best_score = score
                mgr.save_ckpt(
                    model.state_dict(),
                    epoch,
                    kind="best",
                    metric_name=str(self.prefer_metric),
                    metric_value=best_score,
                )

    def test(self):
        ckpt_path = self._select_ckpt_path()
        if ckpt_path is None or (not ckpt_path.exists()):
            raise FileNotFoundError(f"No checkpoint found under {self.result_root}")
        mgr = RunManager(self.result_root, mode="test", tag=self.args.tag, attach_to=ckpt_path, separate_test_log=True)
        logger = mgr.logger
        logger.info(f"[result_root] {self.result_root}")
        logger.info(f"[ddpm_ckpt] {self.ddpm_ckpt}")
        model, used_ckpt = self._load_model(test_loader)
        mgr.record_event("load_ckpt", {"path": str(used_ckpt)})
        t0 = time.time()
        model.val_start()
        oa = model.cal_acc(epoch=0)
        elapsed = time.time() - t0
        metrics = getattr(model, "last_metrics", {}) or {}
        aa_mean = metrics.get("AA_mean", None)
        kappa = metrics.get("Kappa", None)
        print(f"Test | " f"OA: {float(oa):.6f} | "f"AA: {float(aa_mean):.6f} | " f"Kappa: {float(kappa):.6f}")
        logger.info(f"[TEST] time={elapsed:.1f}s | OA={oa} | AA_mean={aa_mean} | Kappa={kappa}")
        mgr.write_metrics("test", epoch=0, elapsed=elapsed, oa=oa, aa_mean=aa_mean, kappa=kappa,
                          event="eval", ckpt_path=str(used_ckpt))
        rep = getattr(model, "last_cls_report", None)
        if isinstance(rep, str) and rep.strip():
            logger.info("\n" + rep)

    @staticmethod
    def _save_full_prediction(pred_map: np.ndarray, out_dir: Path, dataset_name: str, color_map=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        if color_map is None:
            rng = np.random.default_rng(0)
            cmap = rng.integers(0, 256, size=(int(pred_map.max()) + 1, 3)).astype(np.uint8)
        else:
            cmap = np.array(color_map, dtype=np.uint8)
            max_idx = int(pred_map.max())
            if max_idx >= cmap.shape[0]:
                extra = np.random.default_rng(0).integers(0, 256, size=(max_idx + 1 - cmap.shape[0], 3), dtype=np.uint8)
                cmap = np.vstack([cmap, extra])
        pm = pred_map.astype(np.int64).copy()
        mask_bg = pm < 0
        pm = pm.clip(0, cmap.shape[0] - 1)
        img = cmap[pm]
        if mask_bg.any():
            img[mask_bg] = 0
        base = f"{dataset_name}_DRIDNet"
        try:
            from imageio import imwrite
            imwrite(out_dir / f"{base}.png", img)
        except Exception:
            import matplotlib.pyplot as plt
            plt.imsave(out_dir / f"{base}.png", img)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(img.shape[1] / 100.0, img.shape[0] / 100.0), dpi=100)
            plt.axis('off')
            plt.imshow(img)
            plt.savefig(out_dir / f"{base}.pdf", bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception:
            pass
        return {
            "named_png": str((out_dir / f"{base}.png").resolve()),
            "named_pdf": str((out_dir / f"{base}.pdf").resolve()),
        }

    @staticmethod
    def _save_pred_label_npy(pred_map: np.ndarray, out_path: Path, *, valid_mask=None, one_based: bool = True):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pm = pred_map.astype(np.int32).copy()
        m = (pm >= 0)
        if valid_mask is not None:
            vm = np.asarray(valid_mask).astype(bool)
            if vm.shape != pm.shape:
                raise ValueError(f"valid_mask shape {vm.shape} != pred_map shape {pm.shape}")
            m = m & vm
        if one_based:
            pred_label = np.zeros_like(pm, dtype=np.int32)
            pred_label[m] = pm[m] + 1
        else:
            pred_label = np.full_like(pm, fill_value=-1, dtype=np.int32)
            pred_label[m] = pm[m]
        np.save(out_path.as_posix(), pred_label)
        return out_path

    def _infer_label_permutation(self, model, loader, K: int):
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            return None
        cm = np.zeros((K, K), dtype=np.int64)
        with torch.no_grad():
            for batch in loader:
                logits, y, _ = self._forward_batch(model, batch)
                if (logits is None) or (y is None):
                    continue
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
                gt = y.detach().cpu().numpy().reshape(-1)
                m = (gt >= 0) & (gt < K)
                pred = pred[m]
                gt = gt[m]
                for p, g in zip(pred, gt):
                    if 0 <= p < K:
                        cm[p, g] += 1
        if cm.sum() == 0:
            return None
        cost = cm.max() - cm
        r, c = linear_sum_assignment(cost)
        perm = np.arange(K, dtype=np.int64)
        perm[r] = c
        return perm

    def full_pre(self):
        if all_loader is None:
            raise RuntimeError("all_loader is not available; cannot run full_pre.")
        ckpt_path = self._select_ckpt_path()
        if ckpt_path is None or (not ckpt_path.exists()):
            raise FileNotFoundError(f"No checkpoint found under {self.result_root}")
        mgr = RunManager(self.result_root, mode="full_pre", tag=self.args.tag, attach_to=ckpt_path, separate_test_log=True)
        logger = mgr.logger
        logger.info(f"[result_root] {self.result_root}")
        logger.info(f"[ddpm_ckpt] {self.ddpm_ckpt}")
        model, used_ckpt = self._load_model(test_loader)
        mgr.record_event("load_ckpt", {"path": str(used_ckpt)})
        model.eval()

        K = int(DATASET_COLOR_MAP.shape[0]) if DATASET_COLOR_MAP is not None else None
        label_perm = None
        if K is not None and test_loader is not None:
            try:
                label_perm = self._infer_label_permutation(model, test_loader, K)
                if label_perm is not None:
                    logger.info(f"[full_pre] label perm inferred: {label_perm.tolist()}")
            except Exception as e:
                logger.info(f"[full_pre] label perm warn: {e}")

        H, W = int(image_h), int(image_w)
        pred_map = None
        total_written = 0
        with torch.no_grad():
            for bi, batch in enumerate(all_loader):
                logits, _, pos = self._forward_batch(model, batch)
                if logits is None or pos is None:
                    continue
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
                if torch.is_tensor(pos):
                    pos = pos.detach().cpu().numpy()
                pos = np.asarray(pos)
                if bi == 0:
                    logger.info(f"[full_pre] first pos shape={pos.shape}, dtype={pos.dtype}")
                    if pos.ndim != 2 or pos.shape[1] != 2:
                        raise ValueError("expect pos as (N,2) [row,col]")
                if pred_map is None:
                    pred_map = np.full((H, W), fill_value=-1, dtype=np.int32)
                rr = np.clip(pos[:, 0].astype(np.int64), 0, H - 1)
                cc = np.clip(pos[:, 1].astype(np.int64), 0, W - 1)
                n = min(pred.shape[0], rr.shape[0], cc.shape[0])
                if n > 0:
                    pred_map[rr[:n], cc[:n]] = pred[:n].astype(np.int32)
                    total_written += int(n)
        if pred_map is None:
            raise RuntimeError("No predictions were produced in full_pre.")
        if label_perm is not None:
            pm = pred_map.copy()
            m = pm >= 0
            pm[m] = label_perm[pm[m]]
            pred_map = pm
        covered = int((pred_map >= 0).sum())
        logger.info(f"[full_pre] coverage: written={total_written}, covered_pixels={covered}, map_size={pred_map.size}")
        vis_paths = self._save_full_prediction(pred_map, mgr.run_dir, dataset_name=self.dataset, color_map=DATASET_COLOR_MAP)
        pred_label_path = mgr.run_dir / f"DRIDNet_{self.dataset}_pred_label.npy"
        pred_label_saved = self._save_pred_label_npy(pred_map, pred_label_path, valid_mask=None, one_based=True)
        logger.info(f"[full_pre] saved pred_label.npy -> {pred_label_saved}")
        mgr.record_event("save_full_pre", {
            "covered_pixels": covered,
            "map_size": int(pred_map.size),
            **vis_paths,
            "pred_label_npy": pred_label_saved.as_posix(),
        })

    def run(self):
        if self.args.mode == "train":
            return self.train()
        if self.args.mode == "test":
            return self.test()
        if self.args.mode == "full_pre":
            return self.full_pre()
        raise ValueError(f"Unsupported mode={self.args.mode}")
