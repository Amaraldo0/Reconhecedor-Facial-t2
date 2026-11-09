"""Comparação sem dependência de PyTorch, reutilizada na avaliação privada."""
from __future__ import annotations

import numpy as np


def identify_cosine(gallery: np.ndarray, gallery_labels: np.ndarray, probes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarity = probes @ gallery.T
    index = similarity.argmax(axis=1)
    return gallery_labels[index], similarity[np.arange(len(probes)), index]
