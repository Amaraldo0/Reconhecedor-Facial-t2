"""Regras de adequação da imagem para cadastro (CA-002.4)."""
from __future__ import annotations

import cv2

from .config import QualityConfig
from .vision import DetectedFace


def inadequacy_reason(face: DetectedFace, cfg: QualityConfig) -> str | None:
    """Retorna uma orientação em português, ou None se a face é adequada."""
    x1, y1, x2, y2 = face.box
    if min(x2 - x1, y2 - y1) < cfg.min_face_px:
        return "O rosto está muito pequeno. Aproxime-se da câmera ou use uma foto maior."
    if face.confidence < cfg.min_confidence:
        return "O rosto não está nítido o suficiente para o detector. Use uma foto de frente."
    gray = cv2.cvtColor(face.crop, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    if brightness < cfg.min_brightness:
        return "A imagem está muito escura. Melhore a iluminação."
    if brightness > cfg.max_brightness:
        return "A imagem está muito clara. Evite luz direta sobre o rosto."
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < cfg.min_sharpness:
        return "A imagem está borrada. Mantenha a câmera parada."
    return None
