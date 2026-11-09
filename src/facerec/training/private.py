"""Avaliação agregada do dataset privado (CA-007.6). Nunca retorna dado por pessoa."""
from __future__ import annotations

import numpy as np

from . import metrics
from .data import make_pairs
from .engine_numpy import identify_cosine

MIN_PERSONS = 3


def summarize_private(embeddings: np.ndarray, labels: np.ndarray, threshold: float, seed: int = 42) -> dict:
    """Calcula métricas agregadas de verificação e de conjunto aberto (leave-one-person-out)."""
    identities = np.unique(labels)
    if len(identities) < MIN_PERSONS:
        raise ValueError(f"São necessárias ao menos {MIN_PERSONS} pessoas para agregar sem identificar ninguém.")

    pairs = make_pairs(labels, seed)
    scores = np.sum(embeddings[pairs.a] * embeddings[pairs.b], axis=1)
    verification = metrics.binary_metrics(scores, pairs.same, threshold)

    accepted_unknown, unknown_total = 0, 0
    for person in identities:
        unknown = labels == person
        _, score = identify_cosine(embeddings[~unknown], labels[~unknown], embeddings[unknown])
        accepted_unknown += int(np.sum(score >= threshold))
        unknown_total += int(unknown.sum())

    return {
        "num_persons": int(len(identities)),
        "num_images": int(len(labels)),
        "num_pairs": int(len(pairs.same)),
        "threshold": float(threshold),
        "auc": metrics.auc(scores, pairs.same),
        "eer": metrics.equal_error_rate(scores, pairs.same)[0],
        "accuracy": verification["accuracy"],
        "f1": verification["f1"],
        "far": verification["far"],
        "frr": verification["frr"],
        "open_set_far_leave_one_person_out": accepted_unknown / unknown_total,
    }
