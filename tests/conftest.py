"""Dublês determinísticos: os testes não usam rostos reais (§23 do enunciado).

O pixel (0,0) da imagem sintética comanda o dublê: R = identidade, G = número de
faces, B = problema de qualidade (0 nenhum, 1 borrada, 2 escura, 3 pequena).
"""
from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from facerec.app.server import create_app
from facerec.config import AppConfig, RecognitionConfig
from facerec.service import FaceService
from facerec.storage import FaceStore
from facerec.vision import DetectedFace

BLURRY, DARK, SMALL = 1, 2, 3


def make_image(identity: int = 1, faces: int = 1, problem: int = 0) -> bytes:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[0, 0] = (identity, faces, problem)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "PNG")
    return buffer.getvalue()


class FakeDetector:
    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        identity, count, problem = (int(v) for v in image[0, 0])
        rng = np.random.default_rng(identity)
        crop = rng.integers(40, 200, (160, 160, 3), dtype=np.uint8)
        if problem == BLURRY:
            crop = np.full((160, 160, 3), 120, dtype=np.uint8)
        elif problem == DARK:
            crop = (crop / 10).astype(np.uint8)
        crop[0, 0, 0] = identity
        size = 30.0 if problem == SMALL else 120.0
        return [DetectedFace((10.0 * i, 10.0, 10.0 * i + size, 10.0 + size), 0.99, crop) for i in range(count)]


class FakeEmbedder:
    model_id = "fake-model"

    def embed(self, crops: np.ndarray) -> np.ndarray:
        vectors = np.zeros((len(crops), 512), dtype=np.float32)
        vectors[np.arange(len(crops)), crops[:, 0, 0, 0].astype(int)] = 1.0
        return vectors


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return replace(AppConfig(home=tmp_path), recognition=RecognitionConfig(threshold=0.5, aggregation="max"))


@pytest.fixture
def store(config):
    instance = FaceStore(config.db_path, config.photos_dir)
    yield instance
    instance.close()


@pytest.fixture
def service(config, store) -> FaceService:
    return FaceService(config, FakeDetector(), FakeEmbedder(), store)


@pytest.fixture
def client(config, service) -> TestClient:
    return TestClient(create_app(service, config, extra_hosts={"testserver"}))
