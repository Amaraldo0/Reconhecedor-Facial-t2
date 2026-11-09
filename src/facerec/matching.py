"""Comparação de embeddings e regra de decisão (SPEC-003 e SPEC-004)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    person_id: str
    similarity: float


def is_recognized(similarity: float, threshold: float) -> bool:
    return similarity >= threshold


def best_match(
    query: np.ndarray, gallery: np.ndarray, person_ids: list[str], aggregation: str = "max"
) -> Match | None:
    """Encontra a pessoa mais similar. Embeddings devem estar normalizados (L2)."""
    if gallery.shape[0] == 0:
        return None
    scores = gallery @ query
    owners = np.asarray(person_ids)
    best_person, best_score = None, -np.inf
    for person in np.unique(owners):
        mask = owners == person
        if aggregation == "max":
            score = float(scores[mask].max())
        elif aggregation == "prototype":
            prototype = gallery[mask].mean(axis=0)
            prototype /= np.linalg.norm(prototype) + 1e-12
            score = float(prototype @ query)
        else:
            raise ValueError(f"Agregação desconhecida: {aggregation}")
        if score > best_score:
            best_person, best_score = person, score
    return Match(str(best_person), best_score)
