"""Decodificação segura de imagens e alinhamento de faces (SPEC-001)."""
from __future__ import annotations

import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import InvalidImageError


def decode_image(data: bytes, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Converte bytes em matriz RGB uint8, aplicando EXIF e os limites configurados."""
    if not data:
        raise InvalidImageError("O arquivo enviado está vazio.")
    if len(data) > max_bytes:
        raise InvalidImageError(f"Arquivo maior que o limite de {max_bytes // 1_000_000} MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                if img.width * img.height > max_pixels:
                    raise InvalidImageError("A imagem tem resolução acima do limite permitido.")
                return np.asarray(ImageOps.exif_transpose(img).convert("RGB"), dtype=np.uint8).copy()
    except InvalidImageError:
        raise
    except Exception as exc:  # Pillow lança vários tipos para arquivos corrompidos
        raise InvalidImageError() from exc


def align_and_crop(
    image: np.ndarray,
    box: np.ndarray,
    left_eye: np.ndarray | None,
    right_eye: np.ndarray | None,
    size: int = 160,
    margin: int = 14,
) -> np.ndarray:
    """Nivela os olhos girando em torno do centro da caixa e recorta com margem.

    A margem é medida em pixels da imagem final, a convenção usada no treino dos
    pesos FaceNet. A escala é preservada para o recorte continuar comparável ao
    usado no treino. Áreas fora da imagem são preenchidas com a borda.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    width, height = x2 - x1, y2 - y1
    margin_x = margin * width / (size - margin)
    margin_y = margin * height / (size - margin)
    x1, x2, y1, y2 = x1 - margin_x / 2, x2 + margin_x / 2, y1 - margin_y / 2, y2 + margin_y / 2

    if left_eye is not None and right_eye is not None:
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))
    else:
        angle = 0.0

    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    scale_x, scale_y = size / (x2 - x1), size / (y2 - y1)
    matrix[0] = matrix[0] * scale_x
    matrix[1] = matrix[1] * scale_y
    matrix[0, 2] += size / 2 - center[0] * scale_x
    matrix[1, 2] += size / 2 - center[1] * scale_y
    interpolation = cv2.INTER_AREA if max(scale_x, scale_y) < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(image, matrix, (size, size), flags=interpolation, borderMode=cv2.BORDER_REPLICATE)
