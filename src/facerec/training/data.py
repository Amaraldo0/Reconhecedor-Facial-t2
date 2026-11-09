"""Divisão por identidade, controle de vazamento e geração de pares (SPEC-007)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_by_identity(labels: np.ndarray, seed: int, ratios: tuple[float, float] = (0.6, 0.2)) -> Split:
    """Divide índices de imagens de modo que nenhuma identidade cruze os conjuntos.

    Identidades com uma imagem e com várias são divididas separadamente, mantendo a
    proporção de pessoas úteis para pares positivos em cada conjunto.
    """
    rng = np.random.default_rng(seed)
    identities, counts = np.unique(labels, return_counts=True)
    buckets: list[list[np.ndarray]] = [[], [], []]
    for group in (identities[counts == 1], identities[counts > 1]):
        shuffled = rng.permutation(group)
        cut_a = int(round(len(shuffled) * ratios[0]))
        cut_b = cut_a + int(round(len(shuffled) * ratios[1]))
        for bucket, part in zip(buckets, np.split(shuffled, [cut_a, cut_b])):
            bucket.append(part)
    train, val, test = (np.flatnonzero(np.isin(labels, np.concatenate(b))) for b in buckets)
    return Split(train, val, test)


def dhash(crop: np.ndarray) -> int:
    """Hash perceptual de 64 bits, baseado em gradiente horizontal."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).tobytes().hex(), 16)


def _near_matches(query: np.ndarray, reference: np.ndarray, max_distance: int) -> np.ndarray:
    """Máscara das consultas que têm alguma referência a até `max_distance` bits."""
    found = np.zeros(len(query), dtype=bool)
    if len(reference) == 0:
        return found
    for start in range(0, len(query), 512):
        distances = np.bitwise_count(query[start:start + 512, None] ^ reference[None, :])
        found[start:start + 512] = distances.min(axis=1) <= max_distance
    return found


def remove_cross_split_duplicates(
    hashes: np.ndarray, split: Split, max_distance: int = 2
) -> tuple[Split, dict[str, int]]:
    """Remove de treino e validação imagens iguais ou quase iguais às de outros conjuntos.

    O conjunto de teste nunca é alterado, para não ser moldado pelos demais.
    """
    val_mask = ~_near_matches(hashes[split.val], hashes[split.test], max_distance)
    val = split.val[val_mask]
    reference = np.concatenate([hashes[val], hashes[split.test]])
    train_mask = ~_near_matches(hashes[split.train], reference, max_distance)
    report = {"removed_from_val": int((~val_mask).sum()), "removed_from_train": int((~train_mask).sum())}
    return Split(split.train[train_mask], val, split.test), report


def has_cross_split_duplicates(hashes: np.ndarray, split: Split, max_distance: int = 2) -> bool:
    groups = (split.train, split.val, split.test)
    return any(
        _near_matches(hashes[groups[a]], hashes[groups[b]], max_distance).any()
        for a in range(3) for b in range(a + 1, 3)
    )


@dataclass(frozen=True)
class Pairs:
    a: np.ndarray
    b: np.ndarray
    same: np.ndarray
    group: np.ndarray


def make_pairs(labels: np.ndarray, seed: int, max_positive_per_identity: int = 20) -> Pairs:
    """Gera pares balanceados entre mesma pessoa e pessoas diferentes.

    Limita os positivos por identidade para que pessoas muito fotografadas não dominem
    as métricas. `labels` é indexado pela posição dentro do conjunto.
    """
    rng = np.random.default_rng(seed)
    by_identity = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    first, second, group = [], [], []
    for label, members in by_identity.items():
        if len(members) < 2:
            continue
        i, j = np.triu_indices(len(members), k=1)
        chosen = rng.permutation(len(i))[:max_positive_per_identity]
        first.append(members[i[chosen]])
        second.append(members[j[chosen]])
        group.append(np.full(len(chosen), label))
    pos_a, pos_b, pos_group = np.concatenate(first), np.concatenate(second), np.concatenate(group)

    needed = len(pos_a)
    neg_a = rng.integers(0, len(labels), size=needed * 3)
    neg_b = rng.integers(0, len(labels), size=needed * 3)
    keep = np.flatnonzero(labels[neg_a] != labels[neg_b])[:needed]
    neg_a, neg_b = neg_a[keep], neg_b[keep]
    return Pairs(
        a=np.concatenate([pos_a, neg_a]),
        b=np.concatenate([pos_b, neg_b]),
        same=np.concatenate([np.ones(needed, dtype=int), np.zeros(needed, dtype=int)]),
        group=np.concatenate([pos_group, labels[neg_a]]),
    )


@dataclass(frozen=True)
class OpenSetProtocol:
    gallery: np.ndarray
    gallery_labels: np.ndarray
    probes: np.ndarray
    probe_truth: np.ndarray


def make_open_set(labels: np.ndarray, seed: int, max_gallery_images: int = 3) -> OpenSetProtocol:
    """Sorteia metade das identidades como cadastradas; as outras são desconhecidas."""
    rng = np.random.default_rng(seed)
    identities, counts = np.unique(labels, return_counts=True)
    enrollable = rng.permutation(identities[counts >= 2])
    known = set(enrollable[: len(enrollable) // 2].tolist())
    gallery, gallery_labels, probes, truth = [], [], [], []
    for label in identities:
        members = rng.permutation(np.flatnonzero(labels == label))
        if label in known:
            take = min(max_gallery_images, len(members) - 1)
            gallery.extend(members[:take])
            gallery_labels.extend([label] * take)
            probes.extend(members[take:])
            truth.extend([label] * (len(members) - take))
        else:
            probes.extend(members)
            truth.extend([-1] * len(members))
    return OpenSetProtocol(np.array(gallery), np.array(gallery_labels), np.array(probes), np.array(truth))
