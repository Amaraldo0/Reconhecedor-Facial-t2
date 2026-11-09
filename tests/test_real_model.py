"""Integração com os pesos públicos e o LFW local: pytest -m slow."""
from __future__ import annotations

import numpy as np
import pytest

from facerec.config import ModelConfig, default_home
from facerec.training.data import make_pairs
from facerec.vision import FaceNetEmbedder

pytestmark = pytest.mark.slow
LFW = default_home() / "public" / "lfw"


@pytest.mark.skipif(not (LFW / "aligned.npy").exists(), reason="LFW não preparado")
def test_embedding_normalizado_e_separa_identidades():
    embedder = FaceNetEmbedder(ModelConfig(checkpoint=None), default_home())
    labels = np.load(LFW / "labels.npy")[:1500]
    crops = np.load(LFW / "aligned.npy", mmap_mode="r")[:1500]
    embeddings = np.concatenate([embedder.embed(np.array(crops[i:i + 128])) for i in range(0, 1500, 128)])
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)
    pairs = make_pairs(labels, 0)
    scores = np.sum(embeddings[pairs.a] * embeddings[pairs.b], axis=1)
    assert scores[pairs.same == 1].mean() > scores[pairs.same == 0].mean() + 0.4
