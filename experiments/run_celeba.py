"""Treina no CelebA e avalia no LFW e no CelebA (SPEC-008).

Uso: python experiments/run_celeba.py [--only C03 C04]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from facerec.config import REPO_ROOT, assert_safe_data_dir, default_home
from facerec.training import engine
from facerec.training.data import (
    Split,
    has_cross_split_duplicates,
    make_pairs,
    overlapping_identities,
    remove_cross_split_duplicates,
)
from facerec.vision import build_backbone
from run_experiments import RESULTS, load_data, tensor

MATCHERS = ("cosine_max", "prototype", "svm")
APP_GALLERY_SIZE, APP_TARGET_FAR = 20, 0.001


@dataclass
class Benchmark:
    val_crops: torch.Tensor
    val_labels: np.ndarray
    test_crops: torch.Tensor
    test_labels: np.ndarray


def load_benchmarks(home: Path, seed: int, device):
    lfw_root, celeba_root = home / "public" / "lfw", home / "public" / "celeba"
    lfw_labels, lfw_split, _, _ = load_data(lfw_root, seed)
    lfw_crops = np.load(lfw_root / "aligned.npy", mmap_mode="r")
    lfw = Benchmark(
        tensor(lfw_crops, lfw_split.val, device), lfw_labels[lfw_split.val],
        tensor(lfw_crops, lfw_split.test, device), lfw_labels[lfw_split.test],
    )

    labels, partitions = np.load(celeba_root / "labels.npy"), np.load(celeba_root / "partitions.npy")
    hashes = np.load(celeba_root / "hashes.npy")
    split = Split(*(np.flatnonzero(partitions == value) for value in (0, 1, 2)))
    split, leakage = remove_cross_split_duplicates(hashes, split)
    if has_cross_split_duplicates(hashes, split):
        raise RuntimeError("Duplicatas entre partições permaneceram após a remoção.")
    crops = np.load(celeba_root / "aligned.npy", mmap_mode="r")
    celeba = Benchmark(
        tensor(crops, split.val, device), labels[split.val],
        tensor(crops, split.test, device), labels[split.test],
    )
    print("Carregando treino na RAM...", flush=True)
    train_crops = torch.from_numpy(crops[split.train].copy()).permute(0, 3, 1, 2)
    return lfw, celeba, train_crops, labels[split.train], leakage


def filter_overlap(train_crops, train_labels, lfw: Benchmark, threshold: float, device) -> tuple[np.ndarray, dict]:
    """Máscara das imagens de treino sem sobreposição com a validação e o teste do LFW."""
    model = build_backbone("vggface2", device)
    train_e = engine.embed_all(model, train_crops)
    reference = np.concatenate([engine.embed_all(model, lfw.val_crops), engine.embed_all(model, lfw.test_crops)])
    reference_labels = np.concatenate([lfw.val_labels, lfw.test_labels])
    removed = overlapping_identities(train_e, train_labels, reference, reference_labels, threshold)
    mask = ~np.isin(train_labels, removed)
    return mask, {
        "threshold": threshold, "identities_removed": len(removed),
        "identities_kept": int(len(np.unique(train_labels[mask]))), "images_kept": int(mask.sum()),
    }


def build_model(cfg: dict, home: Path, device):
    model = build_backbone(cfg["pretrained"], device)
    if cfg["checkpoint"]:
        model.load_state_dict(torch.load(home / cfg["checkpoint"], map_location=device, weights_only=True))
    return model


def run_one(cfg, home, device, lfw, celeba, train_crops, train_labels, context) -> dict:
    engine.set_seed(cfg["seed"])
    model = build_model(cfg, home, device)
    training = None
    if cfg["strategy"] != "baseline":
        val_pairs = make_pairs(celeba.val_labels, cfg["seed"])
        training = engine.train(model, cfg, train_crops, train_labels, celeba.val_crops, val_pairs)
        (home / "checkpoints" / "public").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), home / "checkpoints" / "public" / f"{cfg['id']}.pt")
    on_celeba = engine.evaluate(model, celeba.val_crops, celeba.val_labels, celeba.test_crops, celeba.test_labels, cfg["seed"])
    on_lfw = engine.evaluate(model, lfw.val_crops, lfw.val_labels, lfw.test_crops, lfw.test_labels, cfg["seed"])
    return {
        "config": cfg,
        "training": vars(training) if training else None,
        "selection_score": (on_celeba["val"]["eer"] + on_lfw["val"]["eer"]) / 2,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad) if training else 0,
        "environment": engine.environment(),
        **context,
        "celeba": on_celeba,
        "lfw": on_lfw,
    }


def select_and_calibrate(configs, home, device, lfw: Benchmark, celeba: Benchmark) -> dict:
    results = {
        cfg["id"]: json.loads((RESULTS / f"{cfg['id']}.json").read_text(encoding="utf-8"))
        for cfg in configs if (RESULTS / f"{cfg['id']}.json").exists()
    }
    scores = {key: value["selection_score"] for key, value in results.items()}
    best = min(scores, key=lambda key: (round(scores[key], 4), results[key]["training"]["seconds"] if results[key]["training"] else 0))
    cfg = results[best]["config"]
    if cfg["strategy"] != "baseline":
        cfg = {**cfg, "checkpoint": f"checkpoints/public/{best}.pt"}
    model = build_model(cfg, home, device)

    studies, methods, val_dir = {}, {}, {}
    embeddings = {
        name: (engine.embed_all(model, data.val_crops), data.val_labels, engine.embed_all(model, data.test_crops), data.test_labels)
        for name, data in (("lfw", lfw), ("celeba", celeba))
    }
    for name, (val_e, val_y, test_e, test_y) in embeddings.items():
        methods[name] = engine.compare_matchers(val_e, val_y, test_e, test_y, 42, MATCHERS)
    for method in ("cosine_max", "prototype"):
        val_dir[method] = (methods["lfw"][method]["val_dir"] + methods["celeba"][method]["val_dir"]) / 2
    chosen = max(val_dir, key=val_dir.get)
    for name, (val_e, val_y, test_e, test_y) in embeddings.items():
        studies[name] = engine.gallery_size_study(val_e, val_y, test_e, test_y, chosen, 42)
    thresholds = {
        name: next(row["threshold"] for row in rows if row["gallery_size"] == APP_GALLERY_SIZE and row["target_far"] == APP_TARGET_FAR)
        for name, rows in studies.items()
    }
    return {
        "scores": scores, "best": best, "chosen_method": chosen, "val_dir_mean": val_dir, "methods": methods,
        "gallery_study": studies, "thresholds_app": thresholds, "app_threshold": max(thresholds.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    home = default_home()
    assert_safe_data_dir(home)
    device = torch.device("cuda")
    spec = yaml.safe_load((REPO_ROOT / "configs" / "experiments_celeba.yaml").read_text(encoding="utf-8"))
    configs = [{**spec["defaults"], **item} for item in spec["experiments"]]
    seed = spec["defaults"]["seed"]

    started = time.time()
    lfw, celeba, train_crops, train_labels, leakage = load_benchmarks(home, seed, device)
    print(f"Dados em {time.time() - started:.0f}s; treino: {train_crops.shape}", flush=True)
    mask, overlap = filter_overlap(train_crops, train_labels, lfw, spec["overlap_threshold"], device)
    print("Sobreposição:", overlap, flush=True)
    train_crops, train_labels = train_crops[torch.from_numpy(mask)], train_labels[mask]
    context = {"dataset": "CelebA + LFW", "leakage_control": leakage, "overlap_filter": overlap}

    for cfg in configs:
        if args.only and cfg["id"] not in args.only:
            continue
        print(f"=== {cfg['id']}: {cfg['title']}", flush=True)
        result = run_one(cfg, home, device, lfw, celeba, train_crops, train_labels, context)
        (RESULTS / f"{cfg['id']}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  score={result['selection_score']:.4f} celeba EER val={result['celeba']['val']['eer']:.4f} lfw EER val={result['lfw']['val']['eer']:.4f}", flush=True)

    selection = select_and_calibrate(configs, home, device, lfw, celeba)
    (RESULTS / "celeba_selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    print("Melhor:", selection["best"], "limiar:", round(selection["app_threshold"], 4), "comparador:", selection["chosen_method"], flush=True)


if __name__ == "__main__":
    main()
