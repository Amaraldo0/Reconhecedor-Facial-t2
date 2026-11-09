"""Critérios da SPEC-007, com identidades e escores sintéticos."""
from __future__ import annotations

import numpy as np
import pytest

from facerec.training import metrics
from facerec.training.data import (
    Split,
    assert_partitions_disjoint,
    cap_per_identity,
    central_face_index,
    dhash,
    has_cross_split_duplicates,
    make_open_set,
    make_pairs,
    overlapping_identities,
    remove_cross_split_duplicates,
    split_by_identity,
)
from facerec.training.private import summarize_private


@pytest.fixture
def labels():
    rng = np.random.default_rng(1)
    return np.concatenate([np.full(rng.integers(1, 9), identity) for identity in range(300)])


def test_ca_007_1_identidades_disjuntas(labels):
    split = split_by_identity(labels, 42)
    sets = [set(labels[part]) for part in (split.train, split.val, split.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(split.train) + len(split.val) + len(split.test) == len(labels)


def test_ca_007_2_mesma_semente_mesma_divisao(labels):
    a, b = split_by_identity(labels, 42), split_by_identity(labels, 42)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.train, b.train)
    assert not np.array_equal(a.test, split_by_identity(labels, 7).test)


def test_ca_007_3_threshold_nao_depende_do_teste():
    impostor_val = np.array([0.1, 0.2, 0.3, 0.4])
    assert metrics.threshold_at_far(impostor_val, 0.0) > 0.4
    assert metrics.threshold_at_far(impostor_val, 0.25) == pytest.approx(0.3, abs=1e-6)


def test_ca_007_4_remove_duplicatas_entre_conjuntos():
    hashes = np.array([1, 2, 4, 2**40, 2**41, 2**42, 2**60, 2**61], dtype=np.uint64)
    hashes[0] = hashes[6]
    hashes[3] = hashes[7]
    split = Split(np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7]))
    assert has_cross_split_duplicates(hashes, split, 0)
    clean, report = remove_cross_split_duplicates(hashes, split, 0)
    assert not has_cross_split_duplicates(hashes, clean, 0)
    assert report == {"removed_from_val": 1, "removed_from_train": 1}
    assert np.array_equal(clean.test, split.test)


def test_dhash_detecta_copia_e_distingue_imagem_diferente():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    assert dhash(image) == dhash(image.copy()) and dhash(image) != dhash(other)


def test_pares_balanceados_e_corretos(labels):
    pairs = make_pairs(labels, 0)
    assert pairs.same.sum() == len(pairs.same) - pairs.same.sum()
    assert np.array_equal(labels[pairs.a] == labels[pairs.b], pairs.same == 1)


def test_open_set_nao_mistura_cadastrados_e_desconhecidos(labels):
    protocol = make_open_set(labels, 0)
    known_truth = set(protocol.probe_truth[protocol.probe_truth != -1])
    unknown_ids = set(labels[protocol.probes[protocol.probe_truth == -1]])
    assert known_truth <= set(protocol.gallery_labels)
    assert not unknown_ids & set(protocol.gallery_labels)
    assert not set(protocol.gallery) & set(protocol.probes)


def test_ca_007_5_metricas_calculadas_a_mao():
    scores = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    same = np.array([1, 1, 1, 0, 0, 0])
    result = metrics.binary_metrics(scores, same, 0.5)
    assert result["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["far"] == pytest.approx(1 / 3)
    assert result["frr"] == pytest.approx(1 / 3)


def test_ca_007_5_eer_separacao_perfeita():
    eer, threshold = metrics.equal_error_rate(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
    assert eer == 0.0 and 0.2 < threshold <= 0.8


def test_open_set_rates_a_mao():
    predicted = np.array([1, 2, 3, 1])
    scores = np.array([0.9, 0.9, 0.1, 0.9])
    truth = np.array([1, 1, 3, -1])
    rates = metrics.open_set_rates(predicted, scores, truth, 0.5)
    assert rates["dir"] == pytest.approx(1 / 3)
    assert rates["wrong_identity_rate"] == pytest.approx(1 / 3)
    assert rates["far"] == 1.0


def _private_embeddings():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 5)
    centers = np.eye(512)[labels]
    vectors = centers + rng.normal(0, 0.01, centers.shape)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True), labels


def test_ca_007_6_avaliacao_privada_so_tem_agregados():
    embeddings, labels = _private_embeddings()
    summary = summarize_private(embeddings, labels, 0.5)
    assert all(isinstance(value, (int, float)) for value in summary.values())
    assert summary["num_persons"] == 4
    assert summary["open_set_far_leave_one_person_out"] == 0.0


def test_ca_007_6_recusa_grupo_pequeno_demais():
    embeddings, labels = _private_embeddings()
    keep = labels < 2
    with pytest.raises(ValueError):
        summarize_private(embeddings[keep], labels[keep], 0.5)


def test_ca_008_2_identidade_nao_pode_cruzar_particoes():
    assert_partitions_disjoint(np.array([1, 1, 2]), np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        assert_partitions_disjoint(np.array([1, 1]), np.array([0, 1]))


def test_ca_008_3_limite_por_identidade(labels):
    first, second = cap_per_identity(labels, 3, 5), cap_per_identity(labels, 3, 5)
    assert np.array_equal(first, second)
    assert np.unique(labels[first], return_counts=True)[1].max() <= 3
    assert set(labels[first]) == set(labels)


def test_ca_008_4_remove_so_identidades_parecidas():
    embeddings = np.eye(4, dtype=np.float32)[[0, 0, 1, 1]]
    reference = np.eye(4, dtype=np.float32)[[1, 2]]
    removed = overlapping_identities(embeddings, np.array([10, 10, 20, 20]), reference, np.array([7, 8]), 0.6)
    assert removed.tolist() == [20]


def test_face_central_e_escolhida():
    boxes = np.array([[0.0, 0.0, 20.0, 20.0], [80.0, 90.0, 120.0, 130.0]])
    assert central_face_index(boxes, 100.0, 110.0) == 1
