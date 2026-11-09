"""Carrega a configuração externa e define onde ficam os dados privados (SPEC-006)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import UnsafeDataDirError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "app.yaml"
_SYNCED_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")


def default_home() -> Path:
    if env := os.environ.get("FACEREC_HOME"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "facerec-ufms"
    return Path.home() / ".local" / "share" / "facerec-ufms"


def assert_safe_data_dir(path: Path) -> None:
    """Impede que dados biométricos fiquem em repositório git ou pasta sincronizada."""
    resolved = path.resolve()
    for part in resolved.parts:
        if any(marker in part.lower() for marker in _SYNCED_MARKERS):
            raise UnsafeDataDirError(
                f"FACEREC_HOME está em uma pasta sincronizada com a nuvem ({part}). "
                "Escolha um diretório local privado."
            )
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            raise UnsafeDataDirError(
                f"FACEREC_HOME está dentro de um repositório git ({candidate}). "
                "Escolha um diretório fora do repositório."
            )


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class ModelConfig:
    id: str = "facenet-vggface2"
    pretrained: str = "vggface2"
    checkpoint: str | None = None
    device: str = "auto"


@dataclass(frozen=True)
class DetectionConfig:
    min_confidence: float = 0.90
    image_size: int = 160
    margin: int = 14
    align: bool = True


@dataclass(frozen=True)
class QualityConfig:
    min_face_px: int = 60
    min_confidence: float = 0.95
    min_sharpness: float = 15.0
    min_brightness: float = 40.0
    max_brightness: float = 225.0


@dataclass(frozen=True)
class RecognitionConfig:
    threshold: float = 0.5
    aggregation: str = "max"


@dataclass(frozen=True)
class EnrollmentConfig:
    max_images: int = 10
    consistency_threshold: float = 0.35


@dataclass(frozen=True)
class LimitsConfig:
    max_upload_bytes: int = 10_000_000
    max_pixels: int = 25_000_000


@dataclass(frozen=True)
class StorageConfig:
    store_photos: bool = False


@dataclass(frozen=True)
class AppConfig:
    home: Path
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    detection: DetectionConfig = DetectionConfig()
    quality: QualityConfig = QualityConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    enrollment: EnrollmentConfig = EnrollmentConfig()
    limits: LimitsConfig = LimitsConfig()
    storage: StorageConfig = StorageConfig()

    @property
    def db_path(self) -> Path:
        return self.home / "gallery" / "faces.db"

    @property
    def photos_dir(self) -> Path:
        return self.home / "gallery" / "photos"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"


def load_config(path: Path | None = None, home: Path | None = None) -> AppConfig:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    home = Path(home) if home else default_home()
    assert_safe_data_dir(home)
    sections = {
        "server": ServerConfig, "model": ModelConfig, "detection": DetectionConfig,
        "quality": QualityConfig, "recognition": RecognitionConfig,
        "enrollment": EnrollmentConfig, "limits": LimitsConfig, "storage": StorageConfig,
    }
    return AppConfig(home=home, **{k: cls(**raw.get(k, {})) for k, cls in sections.items()})
