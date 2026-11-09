"""Métricas de verificação e identificação. Funções puras, sem dependência de PyTorch."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Retorna (EER, threshold) no ponto em que FAR e FRR mais se aproximam."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    frr = 1 - tpr
    index = int(np.nanargmin(np.abs(fpr - frr)))
    return float((fpr[index] + frr[index]) / 2), float(thresholds[index])


def threshold_at_far(impostor_scores: np.ndarray, target_far: float) -> float:
    """Menor threshold cuja taxa de falsa aceitação não excede `target_far`."""
    ordered = np.sort(impostor_scores)[::-1]
    allowed = int(np.floor(target_far * len(ordered)))
    if allowed >= len(ordered):
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], np.inf))


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Métricas no threshold dado, sendo 1 = mesma pessoa e score >= threshold = aceito."""
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": fp / (fp + tn) if fp + tn else 0.0,
        "frr": fn / (fn + tp) if fn + tp else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float,
    iterations: int = 300, seed: int = 0,
) -> dict[str, list[float]]:
    """Intervalo de 95% reamostrando grupos, pois pares do mesmo grupo não são independentes."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    accuracies, eers = [], []
    for _ in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[group] for group in chosen])
        sample_labels = labels[index]
        if sample_labels.min() == sample_labels.max():
            continue
        accuracies.append(float(np.mean((scores[index] >= threshold) == (sample_labels == 1))))
        eers.append(equal_error_rate(scores[index], sample_labels)[0])
    return {
        "accuracy": [float(v) for v in np.percentile(accuracies, [2.5, 97.5])],
        "eer": [float(v) for v in np.percentile(eers, [2.5, 97.5])],
    }


def open_set_rates(
    predicted: np.ndarray, scores: np.ndarray, truth: np.ndarray, threshold: float, unknown_label: int = -1
) -> dict[str, float]:
    """Taxas de identificação aberta. `truth` usa `unknown_label` para não cadastrados."""
    known = truth != unknown_label
    accepted = scores >= threshold
    return {
        "dir": float(np.mean(accepted[known] & (predicted[known] == truth[known]))),
        "far": float(np.mean(accepted[~known])),
        "wrong_identity_rate": float(np.mean(accepted[known] & (predicted[known] != truth[known]))),
    }
