"""Alinha o CelebA a partir do zip local, gravando o cache fora do repositório.

Baixe as imagens na fonte oficial (ou cópia equivalente) e coloque, em FACEREC_HOME/public/celeba,
o `identity_CelebA.txt` da pasta Anno oficial. O zip é lido sem extração.

Uso: python scripts/prepare_celeba.py --zip caminho/CelebA.zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from facerec.config import assert_safe_data_dir, default_home
from facerec.imaging import align_and_crop
from facerec.training.data import assert_partitions_disjoint, cap_per_identity, central_face_index, dhash
from facerec.vision import resolve_device

IDENTITY_MD5 = "32bd1bd63d3c78cd57e08160ec5ed1e2"
IMAGE_PREFIX = "img_align_celeba/img_align_celeba/"
PARTITION_FILE = "list_eval_partition.csv"
IMAGE_SIZE, MARGIN, BATCH, MAX_PER_IDENTITY, SEED = 160, 14, 64, 10, 42
CENTER_X, CENTER_Y = 89.0, 109.0


def verify_md5(data: bytes, expected: str = IDENTITY_MD5) -> None:
    if hashlib.md5(data).hexdigest() != expected:
        raise SystemExit("identity_CelebA.txt com soma MD5 divergente.")


def load_metadata(archive: zipfile.ZipFile, identity_file: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    data = identity_file.read_bytes()
    verify_md5(data)
    identity = {name: int(value) for name, value in (line.split() for line in data.decode().splitlines())}
    with archive.open(PARTITION_FILE) as handle:
        rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    names = [row["image_id"] for row in rows]
    return names, np.array([identity[name] for name in names]), np.array([int(row["partition"]) for row in rows])


def main() -> None:
    from facenet_pytorch import MTCNN

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    home = default_home()
    assert_safe_data_dir(home)
    root = home / "public" / "celeba"

    with zipfile.ZipFile(args.zip) as archive:
        names, labels, partitions = load_metadata(archive, root / "identity_CelebA.txt")
        assert_partitions_disjoint(labels, partitions)
        selected = cap_per_identity(labels, MAX_PER_IDENTITY, SEED)
        mtcnn = MTCNN(keep_all=True, device=resolve_device("auto"), post_process=False)
        crops, kept_labels, kept_partitions, hashes, failures = [], [], [], [], 0
        for start in tqdm(range(0, len(selected), BATCH), desc="Alinhando"):
            chunk = selected[start:start + BATCH]
            images = [
                np.asarray(Image.open(io.BytesIO(archive.read(IMAGE_PREFIX + names[i]))).convert("RGB"))
                for i in chunk
            ]
            boxes, _, points = mtcnn.detect(images, landmarks=True)
            for index, image, box, landmark in zip(chunk, images, boxes, points):
                if box is None:
                    failures += 1
                    continue
                face = central_face_index(box, CENTER_X, CENTER_Y)
                crop = align_and_crop(image, box[face], landmark[face][0], landmark[face][1], IMAGE_SIZE, MARGIN)
                crops.append(crop)
                kept_labels.append(labels[index])
                kept_partitions.append(partitions[index])
                hashes.append(dhash(crop))

    np.save(root / "aligned.npy", np.stack(crops))
    np.save(root / "labels.npy", np.array(kept_labels, dtype=np.int32))
    np.save(root / "partitions.npy", np.array(kept_partitions, dtype=np.int8))
    np.save(root / "hashes.npy", np.array(hashes, dtype=np.uint64))
    summary = {
        "images_in_dataset": len(names), "images_selected": len(selected), "images_kept": len(crops),
        "detection_failures": failures, "identities": int(len(set(kept_labels))),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
