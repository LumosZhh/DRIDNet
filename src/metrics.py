from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report

_EPS = 1e-12


def cal_results(cm: np.ndarray):
    """
     OA / AA_mean / Kappa / AA.
    Args:
        cm: ndarray [C, C], ,
    Returns:
        OA (float), AA_mean (float), Kappa (float), AA_per_class (ndarray[C])
    """
    cm = np.asarray(cm, dtype=np.int64)
    total = cm.sum()
    # OA
    oa = float(np.trace(cm) / (total + _EPS))
    # Per-class AA
    row_sums = cm.sum(axis=1)  # 
    with np.errstate(divide='ignore', invalid='ignore'):
        aa_per_class = np.divide(cm.diagonal(), row_sums, where=row_sums > 0)
    valid = row_sums > 0
    aa_mean = float(aa_per_class[valid].mean()) if np.any(valid) else 0.0
    # Kappa
    pe = float((cm.sum(axis=0) * row_sums).sum()) / float((total * total) + _EPS)
    kappa = float((oa - pe) / (1.0 - pe + _EPS))
    return oa, aa_mean, kappa, aa_per_class


def confusion_matrix_from_labels(y_true, y_pred, labels: list[int] | None = None):
    """
    .; labels( 0..C-1  1..C).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if labels is None:
        return confusion_matrix(y_true, y_pred)
    return confusion_matrix(y_true, y_pred, labels=labels)


def output_metric(y_true, y_pred, labels: list[int] | None = None, return_cm: bool = False):
    """
     y_true / y_pred .
     4 ():
        OA, AA_mean, Kappa, AA_per_class
     return_cm=True , 5 :
        OA, AA_mean, Kappa, AA_per_class, cm
    """
    cm = confusion_matrix_from_labels(y_true, y_pred, labels=labels)
    oa, aa_mean, kappa, aa = cal_results(cm)
    if return_cm:
        return oa, aa_mean, kappa, aa, cm
    else:
        return oa, aa_mean, kappa, aa



def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Top-k (%). (list[float], target, pred.squeeze()) .
    : maxk>1 ,pred.squeeze()  [maxk, B].
    """
    maxk = int(max(topk))
    batch_size = target.size(0)

    with torch.no_grad():
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
    return res, target, pred.squeeze()


class AvgrageMeter(object):
    """DRIDNet module."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n: int = 1):
        self.sum += float(val) * n
        self.cnt += int(n)
        self.avg = self.sum / max(self.cnt, 1)


def classification_report_text(y_true, y_pred, num_classes: int | None = None, labels: list[int] | None = None):
    """
     precision/recall/f1/support .
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if labels is None and num_classes is not None:
        labels = list(range(num_classes))
    try:
        return classification_report(
            y_true, y_pred, labels=labels, target_names=[str(x) for x in labels] if labels else None,
            digits=4, zero_division=0
        )
    except Exception as e:
        return f"[warn] classification_report failed: {e}"
