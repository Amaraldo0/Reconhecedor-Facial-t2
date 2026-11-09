"""Avalia o modelo escolhido no dataset privado, localmente e com saída apenas agregada.

Estrutura esperada, FORA do repositório:
    FACEREC_HOME/private_dataset/<participante>/<fotos>

Uso: python experiments/evaluate_private.py
"""
from __future__ import annotations

import json

import numpy as np

from facerec.config import load_config
from facerec.imaging import decode_image
from facerec.training.private import summarize_private
from facerec.vision import FaceNetEmbedder, MtcnnDetector

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    cfg = load_config()
    root = cfg.home / "private_dataset"
    if not root.is_dir():
        raise SystemExit(f"Crie {root} com uma pasta por participante que assinou o Termo.")
    detector = MtcnnDetector(cfg.detection, cfg.model.device)
    embedder = FaceNetEmbedder(cfg.model, cfg.home)

    crops, labels, skipped = [], [], 0
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    for label, folder in enumerate(folders):
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            image = decode_image(path.read_bytes(), 50_000_000, 100_000_000)
            faces = detector.detect(image)
            if len(faces) != 1:
                skipped += 1
                continue
            crops.append(faces[0].crop)
            labels.append(label)
    if not crops:
        raise SystemExit("Nenhuma face utilizável encontrada.")

    embeddings = np.concatenate([embedder.embed(np.stack(crops[i:i + 64])) for i in range(0, len(crops), 64)])
    summary = summarize_private(embeddings, np.array(labels), cfg.recognition.threshold)
    summary["images_skipped"] = skipped
    output = cfg.home / "results" / "private_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
