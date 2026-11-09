"""Treino e avaliação das configurações experimentais."""
from __future__ import annotations

import copy
import platform
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from . import metrics
from .data import OpenSetProtocol, Pairs, make_open_set, make_pairs
from .losses import AngularMarginHead, BatchHardTriplet

PARTIAL_MODULES = ("repeat_3", "avgpool_1a", "last_linear", "last_bn")
TARGET_FAR = 0.01


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment() -> dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def to_batch(crops: torch.Tensor) -> torch.Tensor:
    return (crops.float() - 127.5) / 128.0


@torch.inference_mode()
def embed_all(model, crops: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(crops), batch_size):
        with torch.amp.autocast("cuda"):
            output = model(to_batch(crops[start:start + batch_size]))
        parts.append(F.normalize(output.float(), dim=1).cpu().numpy())
    return np.concatenate(parts)


def degrade(crops: torch.Tensor, seed: int) -> torch.Tensor:
    """Simula captura ruim, de forma determinística: baixa resolução e brilho variável."""
    generator = torch.Generator(device=crops.device).manual_seed(seed)
    low = F.interpolate(crops.float(), size=40, mode="bilinear", antialias=True)
    restored = F.interpolate(low, size=crops.shape[-1], mode="bilinear")
    gain = 0.6 + 0.8 * torch.rand(len(crops), 1, 1, 1, generator=generator, device=crops.device)
    return (restored * gain).clamp(0, 255).round().to(torch.uint8)


def augment(batch: torch.Tensor, strong: bool) -> torch.Tensor:
    """Aumento de dados apenas no treino. Flip sempre; degradações se `strong`."""
    flip = torch.rand(len(batch), device=batch.device) < 0.5
    batch = batch.float()
    batch[flip] = batch[flip].flip(-1)
    if strong:
        size = int(torch.randint(48, batch.shape[-1] + 1, (1,)))
        batch = F.interpolate(batch, size=size, mode="bilinear", antialias=True)
        batch = F.interpolate(batch, size=160, mode="bilinear")
        gain = 0.7 + 0.6 * torch.rand(len(batch), 1, 1, 1, device=batch.device)
        batch = (batch * gain).clamp(0, 255)
    return batch


def pk_batches(labels: np.ndarray, p: int, k: int, rng: np.random.Generator):
    """Lotes de P identidades com K imagens, equilibrando pessoas muito fotografadas."""
    identities = rng.permutation(np.unique(labels))
    members = {label: np.flatnonzero(labels == label) for label in identities}
    for start in range(0, len(identities) - p + 1, p):
        chosen = identities[start:start + p]
        yield np.concatenate(
            [rng.choice(members[label], size=k, replace=len(members[label]) < k) for label in chosen]
        )


def apply_freezing(model, strategy: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = strategy in ("full", "scratch") or name.startswith(PARTIAL_MODULES)


def set_train_mode(model, strategy: str) -> None:
    """Mantém camadas congeladas em eval, para o BatchNorm não mudar suas estatísticas."""
    if strategy in ("full", "scratch"):
        model.train()
        return
    model.eval()
    for name, module in model.named_children():
        if name in PARTIAL_MODULES:
            module.train()


@dataclass
class TrainResult:
    best_val_eer: float
    epochs_run: int
    best_epoch: int
    seconds: float
    history: list[float] = field(default_factory=list)


def train(model, cfg: dict, crops: torch.Tensor, labels: np.ndarray, val_crops: torch.Tensor, val_pairs: Pairs) -> TrainResult:
    """Treina restaurando o melhor estado, medido só na validação. Lotes PK iguais para toda perda."""
    strategy, loss_name = cfg["strategy"], cfg["loss"]
    device = crops.device
    apply_freezing(model, strategy)
    remap = {label: index for index, label in enumerate(np.unique(labels))}
    mapped = np.array([remap[label] for label in labels])

    if loss_name == "triplet":
        criterion = BatchHardTriplet(cfg.get("margin", 0.2)).to(device)
    else:
        margin = cfg.get("margin", 0.3) if loss_name == "arcface" else 0.0
        criterion = AngularMarginHead(512, len(remap), cfg.get("scale", 30.0), margin).to(device)

    backbone = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": backbone, "lr": cfg["backbone_lr"]}]
    head = list(criterion.parameters())
    if head:
        groups.append({"params": head, "lr": cfg["head_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(cfg["seed"])

    best_eer, best_epoch, best_state, history, epoch = 1.0, -1, None, [], 0
    started = time.time()
    for epoch in range(cfg["epochs"]):
        set_train_mode(model, strategy)
        for batch_index in pk_batches(mapped, cfg["p"], cfg["k"], rng):
            index = torch.from_numpy(batch_index).to(device)
            batch = to_batch(augment(crops[index], cfg.get("strong_augment", False)))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(batch), index_labels(mapped, batch_index, device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        embeddings = embed_all(model, val_crops)
        scores = np.sum(embeddings[val_pairs.a] * embeddings[val_pairs.b], axis=1)
        eer = metrics.equal_error_rate(scores, val_pairs.same)[0]
        history.append(eer)
        print(f"  [{cfg['id']}] época {epoch + 1}/{cfg['epochs']} EER_val={eer:.4f}", flush=True)
        if eer < best_eer:
            best_eer, best_epoch = eer, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return TrainResult(best_eer, epoch + 1, best_epoch + 1, time.time() - started, history)


def index_labels(mapped: np.ndarray, batch_index: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(mapped[batch_index]).long().to(device)


def scores_for(embeddings: np.ndarray, pairs: Pairs) -> np.ndarray:
    return np.sum(embeddings[pairs.a] * embeddings[pairs.b], axis=1)


def identify(train_e: np.ndarray, train_y: np.ndarray, probes: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (rótulo previsto, escore) para cada sonda, conforme o comparador."""
    classes = np.unique(train_y)
    if method == "cosine_max":
        similarity = probes @ train_e.T
        index = similarity.argmax(axis=1)
        return train_y[index], similarity[np.arange(len(probes)), index]
    if method == "euclidean":
        distance = np.linalg.norm(probes[:, None] - train_e[None], axis=2)
        index = distance.argmin(axis=1)
        return train_y[index], -distance[np.arange(len(probes)), index]
    if method == "prototype":
        prototypes = np.stack([train_e[train_y == c].mean(axis=0) for c in classes])
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
        similarity = probes @ prototypes.T
        index = similarity.argmax(axis=1)
        return classes[index], similarity[np.arange(len(probes)), index]
    if method == "knn3":
        similarity = probes @ train_e.T
        top = np.argsort(-similarity, axis=1)[:, :3]
        predicted, score = [], []
        for row, neighbors in enumerate(top):
            votes: dict[int, float] = {}
            for neighbor in neighbors:
                votes[int(train_y[neighbor])] = votes.get(int(train_y[neighbor]), 0.0) + float(similarity[row, neighbor])
            label = max(votes, key=votes.get)
            predicted.append(label)
            score.append(votes[label] / 3)
        return np.array(predicted), np.array(score)
    if method == "svm":
        from sklearn.svm import LinearSVC

        classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=0).fit(train_e, train_y)
        decision = classifier.decision_function(probes)
        index = decision.argmax(axis=1)
        return classifier.classes_[index], decision[np.arange(len(probes)), index]
    raise ValueError(method)


def _roc_points(scores: np.ndarray, same: np.ndarray, points: int = 200) -> dict[str, list[float]]:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(same, scores)
    grid = np.logspace(-4, 0, points)
    return {"fpr": grid.round(6).tolist(), "tpr": np.interp(grid, fpr, tpr).round(5).tolist()}


def _histogram(scores: np.ndarray, same: np.ndarray) -> dict[str, list[float]]:
    edges = np.linspace(-0.5, 1.0, 76)
    return {
        "edges": edges.round(4).tolist(),
        "same": np.histogram(scores[same == 1], bins=edges, density=True)[0].round(4).tolist(),
        "different": np.histogram(scores[same == 0], bins=edges, density=True)[0].round(4).tolist(),
    }


def _open_set_threshold(embeddings: np.ndarray, protocol: OpenSetProtocol, method: str) -> float:
    predicted, scores = identify(embeddings[protocol.gallery], protocol.gallery_labels, embeddings[protocol.probes], method)
    return metrics.threshold_at_far(scores[protocol.probe_truth == -1], TARGET_FAR)


def evaluate(
    model, val_crops: torch.Tensor, val_labels: np.ndarray,
    test_crops: torch.Tensor, test_labels: np.ndarray, seed: int,
) -> dict:
    """Thresholds são obtidos apenas da validação e depois aplicados ao teste (CA-007.3)."""
    val_e = embed_all(model, val_crops)
    val_pairs = make_pairs(val_labels, seed)
    val_scores = scores_for(val_e, val_pairs)
    val_eer, th_eer = metrics.equal_error_rate(val_scores, val_pairs.same)
    th_far = metrics.threshold_at_far(val_scores[val_pairs.same == 0], TARGET_FAR)
    val_open = make_open_set(val_labels, seed)
    th_open = _open_set_threshold(val_e, val_open, "cosine_max")

    test_e = embed_all(model, test_crops)
    test_pairs = make_pairs(test_labels, seed)
    test_scores = scores_for(test_e, test_pairs)
    degraded_e = embed_all(model, degrade(test_crops, seed))
    degraded_scores = scores_for(degraded_e, test_pairs)

    test_open = make_open_set(test_labels, seed)
    predicted, scores = identify(
        test_e[test_open.gallery], test_open.gallery_labels, test_e[test_open.probes], "cosine_max"
    )
    return {
        "val": {
            "eer": val_eer, "auc": metrics.auc(val_scores, val_pairs.same),
            "threshold_eer": th_eer, "threshold_far1": th_far, "threshold_open_far1": th_open,
            "num_pairs": len(val_pairs.same),
        },
        "test": {
            "auc": metrics.auc(test_scores, test_pairs.same),
            "eer": metrics.equal_error_rate(test_scores, test_pairs.same)[0],
            "at_val_eer_threshold": metrics.binary_metrics(test_scores, test_pairs.same, th_eer),
            "at_val_far1_threshold": metrics.binary_metrics(test_scores, test_pairs.same, th_far),
            "ci95": metrics.bootstrap_ci(test_scores, test_pairs.same, test_pairs.group, th_eer, seed=seed),
            "open_set_at_val_threshold": metrics.open_set_rates(predicted, scores, test_open.probe_truth, th_open),
            "num_pairs": len(test_pairs.same),
        },
        "test_degraded": {
            "auc": metrics.auc(degraded_scores, test_pairs.same),
            "eer": metrics.equal_error_rate(degraded_scores, test_pairs.same)[0],
            "at_val_eer_threshold": metrics.binary_metrics(degraded_scores, test_pairs.same, th_eer),
        },
        "curves": {
            "roc": _roc_points(test_scores, test_pairs.same),
            "histogram": _histogram(test_scores, test_pairs.same),
        },
    }


def compare_matchers(val_e, val_labels, test_e, test_labels, seed: int) -> dict[str, dict[str, float]]:
    """Compara métodos de identificação, com threshold de cada um definido na validação."""
    val_open, test_open = make_open_set(val_labels, seed), make_open_set(test_labels, seed)
    results = {}
    for method in ("cosine_max", "euclidean", "prototype", "knn3", "svm"):
        started = time.time()
        threshold = _open_set_threshold(val_e, val_open, method)
        val_predicted, val_scores = identify(
            val_e[val_open.gallery], val_open.gallery_labels, val_e[val_open.probes], method
        )
        val_rates = metrics.open_set_rates(val_predicted, val_scores, val_open.probe_truth, threshold)
        predicted, scores = identify(
            test_e[test_open.gallery], test_open.gallery_labels, test_e[test_open.probes], method
        )
        rates = metrics.open_set_rates(predicted, scores, test_open.probe_truth, threshold)
        results[method] = {
            **rates, "val_dir": val_rates["dir"], "threshold": float(threshold), "seconds": time.time() - started,
        }
    return results


def _pooled_open_set(embeddings, protocol: OpenSetProtocol, method: str, size: int, draws: int, rng):
    """Simula galerias de `size` pessoas, retornando escores de desconhecidos e de conhecidos."""
    identities = np.unique(protocol.gallery_labels)
    unknown_scores, known_scores, known_correct = [], [], []
    for _ in range(draws):
        chosen = rng.choice(identities, size=min(size, len(identities)), replace=False)
        in_gallery = np.isin(protocol.gallery_labels, chosen)
        predicted, scores = identify(
            embeddings[protocol.gallery[in_gallery]], protocol.gallery_labels[in_gallery],
            embeddings[protocol.probes], method,
        )
        unknown_scores.append(scores[protocol.probe_truth == -1])
        known = np.isin(protocol.probe_truth, chosen)
        known_scores.append(scores[known])
        known_correct.append(predicted[known] == protocol.probe_truth[known])
    return np.concatenate(unknown_scores), np.concatenate(known_scores), np.concatenate(known_correct)


def gallery_size_study(
    val_e, val_labels, test_e, test_labels, method: str, seed: int,
    sizes=(5, 20, 50), far_targets=(0.01, 0.001), draws: int = 30,
) -> list[dict[str, float]]:
    """Calibra o threshold na validação para cada tamanho de galeria e o aplica ao teste."""
    rng = np.random.default_rng(seed)
    val_open, test_open = make_open_set(val_labels, seed), make_open_set(test_labels, seed)
    rows = []
    for size in sizes:
        val_unknown, _, _ = _pooled_open_set(val_e, val_open, method, size, draws, rng)
        unknown, known, correct = _pooled_open_set(test_e, test_open, method, size, draws, rng)
        for target in far_targets:
            threshold = metrics.threshold_at_far(val_unknown, target)
            rows.append({
                "gallery_size": size, "target_far": target, "threshold": threshold,
                "test_far": float(np.mean(unknown >= threshold)),
                "test_dir": float(np.mean((known >= threshold) & correct)),
            })
    return rows
