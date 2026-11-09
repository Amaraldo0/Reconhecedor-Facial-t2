"""Baixa o LFW da origem, detecta e alinha as faces, gravando o cache fora do repositório.

Uso: python scripts/prepare_lfw.py
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from facerec.config import assert_safe_data_dir, default_home
from facerec.imaging import align_and_crop
from facerec.training.data import central_face_index, dhash
from facerec.vision import resolve_device

URLS = ("https://vis-www.cs.umass.edu/lfw/lfw.tgz", "https://ndownloader.figshare.com/files/5976018")
SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
IMAGE_SIZE, MARGIN, BATCH = 160, 14, 64


def download(target: Path) -> None:
    if target.exists() and sha256(target) == SHA256:
        return
    for url in URLS:
        try:
            print(f"Baixando {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "facerec-academic/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            if sha256(target) == SHA256:
                return
            print("Soma de verificação divergente; tentando próxima origem.")
        except OSError as error:
            print(f"Falhou: {error}")
    raise SystemExit("Não foi possível baixar o LFW com soma de verificação válida.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from facenet_pytorch import MTCNN

    home = default_home()
    assert_safe_data_dir(home)
    root = home / "public" / "lfw"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "lfw.tgz"
    download(archive)
    if not (root / "lfw").exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")

    paths = sorted((root / "lfw").glob("*/*.jpg"))
    names = sorted({path.parent.name for path in paths})
    label_of = {name: index for index, name in enumerate(names)}
    mtcnn = MTCNN(keep_all=True, device=resolve_device("auto"), post_process=False, thresholds=[0.6, 0.7, 0.7])

    aligned, cropped, labels, hashes, kept, failures = [], [], [], [], [], 0
    for start in tqdm(range(0, len(paths), BATCH), desc="Alinhando"):
        chunk = paths[start:start + BATCH]
        images = [np.asarray(Image.open(path).convert("RGB")) for path in chunk]
        boxes, _, points = mtcnn.detect(images, landmarks=True)
        for path, image, box, landmark in zip(chunk, images, boxes, points):
            if box is None:
                failures += 1
                continue
            index = central_face_index(box, 125.0, 125.0)
            eyes = landmark[index]
            rotated = align_and_crop(image, box[index], eyes[0], eyes[1], IMAGE_SIZE, MARGIN)
            aligned.append(rotated)
            cropped.append(align_and_crop(image, box[index], None, None, IMAGE_SIZE, MARGIN))
            labels.append(label_of[path.parent.name])
            hashes.append(dhash(rotated))
            kept.append(path.name)

    np.save(root / "aligned.npy", np.stack(aligned))
    np.save(root / "crop_only.npy", np.stack(cropped))
    np.save(root / "labels.npy", np.array(labels, dtype=np.int32))
    np.save(root / "hashes.npy", np.array(hashes, dtype=np.uint64))
    summary = {
        "images_total": len(paths), "images_kept": len(kept), "detection_failures": failures,
        "identities": len({int(v) for v in labels}),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
