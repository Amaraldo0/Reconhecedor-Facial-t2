"""Inicia a aplicação: python -m facerec.app"""
from __future__ import annotations

import ipaddress
import logging
from logging.handlers import RotatingFileHandler

import uvicorn

from ..config import load_config
from ..errors import UnsafeDataDirError
from ..service import FaceService
from ..storage import FaceStore
from ..vision import FaceNetEmbedder, MtcnnDetector
from .server import create_app


def configure_logging(logs_dir) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("facerec")
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)


def main() -> None:
    try:
        cfg = load_config()
    except UnsafeDataDirError as error:
        raise SystemExit(f"Inicialização recusada: {error}") from error
    if not ipaddress.ip_address(cfg.server.host).is_loopback:
        raise SystemExit("Inicialização recusada: server.host deve ser um endereço de loopback.")
    configure_logging(cfg.logs_dir)
    print("Carregando modelos...")
    try:
        embedder = FaceNetEmbedder(cfg.model, cfg.home)
    except FileNotFoundError as error:
        raise SystemExit(
            f"Checkpoint não encontrado em {cfg.home}. Gere-o com os scripts de experimentos "
            "ou use model.checkpoint: null em configs/app.yaml."
        ) from error
    store = FaceStore(cfg.db_path, cfg.photos_dir)
    service = FaceService(cfg, MtcnnDetector(cfg.detection, cfg.model.device), embedder, store)
    print(f"Pessoas cadastradas: {len(service.list_persons()) or 'Nenhuma pessoa cadastrada.'}")
    print(f"Abra http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(create_app(service, cfg), host=cfg.server.host, port=cfg.server.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
