"""Adaptadores do detector MTCNN e da rede FaceNet (dependem de PyTorch)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DetectionConfig, ModelConfig
from .imaging import align_and_crop


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[float, float, float, float]
    confidence: float
    crop: np.ndarray  # RGB uint8, image_size x image_size x 3


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[DetectedFace]: ...


class Embedder(Protocol):
    model_id: str

    def embed(self, crops: np.ndarray) -> np.ndarray: ...


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class MtcnnDetector:
    def __init__(self, cfg: DetectionConfig, device_name: str = "auto"):
        from facenet_pytorch import MTCNN

        self.cfg = cfg
        self.mtcnn = MTCNN(keep_all=True, device=resolve_device(device_name), post_process=False)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        if boxes is None:
            return []
        faces = []
        for box, prob, landmarks in zip(boxes, probs, points):
            if prob is None or prob < self.cfg.min_confidence:
                continue
            crop = align_and_crop(
                image, box,
                landmarks[0] if self.cfg.align else None,
                landmarks[1] if self.cfg.align else None,
                self.cfg.image_size, self.cfg.margin,
            )
            faces.append(DetectedFace(tuple(float(v) for v in box), float(prob), crop))
        return faces


def build_backbone(pretrained: str | None, device):
    from facenet_pytorch import InceptionResnetV1

    return InceptionResnetV1(pretrained=pretrained, classify=False).to(device)


def standardize(crops: np.ndarray):
    """Converte NHWC uint8 em NCHW float na escala usada pelos pesos FaceNet."""
    import torch

    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float()
    return (tensor - 127.5) / 128.0


class FaceNetEmbedder:
    def __init__(self, cfg: ModelConfig, home: Path):
        import torch

        self.device = resolve_device(cfg.device)
        self.model = build_backbone(cfg.pretrained, self.device).eval()
        self.model_id = cfg.id
        if cfg.checkpoint:
            path = home / cfg.checkpoint
            self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            self.model_id = f"{cfg.id}+{digest}"

    def embed(self, crops: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            output = self.model(standardize(crops).to(self.device))
            output = torch.nn.functional.normalize(output, dim=1)
        return output.cpu().numpy()
