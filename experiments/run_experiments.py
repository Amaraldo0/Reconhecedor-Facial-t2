"""Executa os experimentos de configs/experiments.yaml sobre o LFW já preparado.

Uso: python experiments/run_experiments.py [--only E01 E04]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from facerec.config import REPO_ROOT, assert_safe_data_dir, default_home
from facerec.training import engine
from facerec.training.data import (
    has_cross_split_duplicates,
    make_pairs,
    remove_cross_split_duplicates,
    split_by_identity,
)
from facerec.vision import build_backbone

RESULTS = REPO_ROOT / "experiments" / "results"


def load_data(root: Path, seed: int):
    labels = np.load(root / "labels.npy")
    hashes = np.load(root / "hashes.npy")
    split, report = remove_cross_split_duplicates(hashes, split_by_identity(labels, seed))
    if has_cross_split_duplicates(hashes, split):
        raise RuntimeError("Duplicatas entre conjuntos permaneceram após a remoção.")
    train_labels = labels[split.train]
    identities, counts = np.unique(train_labels, return_counts=True)
    usable = np.isin(train_labels, identities[counts >= 2])
    report.update({
        "train_images": int(usable.sum()),
        "train_identities": int(len(np.unique(train_labels[usable]))),
        "val_images": len(split.val),
        "val_identities": int(len(np.unique(labels[split.val]))),
        "test_images": len(split.test),
        "test_identities": int(len(np.unique(labels[split.test]))),
    })
    return labels, split, split.train[usable], report


def tensor(array: np.ndarray, index: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(array[index].copy()).permute(0, 3, 1, 2).to(device)


def run_one(cfg: dict, data_root: Path, checkpoints: Path, device) -> dict:
    engine.set_seed(cfg["seed"])
    labels, split, train_index, leakage = load_data(data_root, cfg["seed"])
    crops = np.load(data_root / f"{cfg['variant']}.npy", mmap_mode="r")
    val_crops, val_labels = tensor(crops, split.val, device), labels[split.val]
    test_crops, test_labels = tensor(crops, split.test, device), labels[split.test]

    model = build_backbone(cfg["pretrained"], device)
    training = None
    if cfg["strategy"] != "baseline":
        train_crops = tensor(crops, train_index, device)
        val_pairs = make_pairs(val_labels, cfg["seed"])
        training = engine.train(model, cfg, train_crops, labels[train_index], val_crops, val_pairs)
        del train_crops
        checkpoints.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoints / f"{cfg['id']}.pt")

    results = engine.evaluate(model, val_crops, val_labels, test_crops, test_labels, cfg["seed"])
    model.eval()
    probe = test_crops[:64]
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.embed_all(model, probe)
    torch.cuda.synchronize()
    gpu_ms = 1000 * (time.perf_counter() - started) / len(probe)
    return {
        "config": cfg,
        "dataset": "LFW",
        "leakage_control": leakage,
        "training": vars(training) if training else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad) if training else 0,
        "gpu_ms_per_image": gpu_ms,
        "environment": engine.environment(),
        **results,
    }


def run_matchers(best_id: str, config: dict, data_root: Path, checkpoints: Path, device) -> dict:
    labels, split, _, _ = load_data(data_root, config["seed"])
    crops = np.load(data_root / f"{config['variant']}.npy", mmap_mode="r")
    model = build_backbone(config["pretrained"], device)
    if config["strategy"] != "baseline":
        model.load_state_dict(torch.load(checkpoints / f"{best_id}.pt", map_location=device, weights_only=True))
    val_e = engine.embed_all(model, tensor(crops, split.val, device))
    test_e = engine.embed_all(model, tensor(crops, split.test, device))
    methods = engine.compare_matchers(val_e, labels[split.val], test_e, labels[split.test], config["seed"])
    # Só comparadores suportados pela galeria sem retreino a cada cadastro; escolha pela validação.
    chosen = max(("cosine_max", "prototype"), key=lambda name: methods[name]["val_dir"])
    return {
        "model_experiment": best_id,
        "methods": methods,
        "chosen_method": chosen,
        "gallery_study": engine.gallery_size_study(
            val_e, labels[split.val], test_e, labels[split.test], chosen, config["seed"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    home = default_home()
    assert_safe_data_dir(home)
    data_root = home / "public" / "lfw"
    checkpoints = home / "checkpoints" / "public"
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    with open(REPO_ROOT / "configs" / "experiments.yaml", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    configs = [{**spec["defaults"], **item} for item in spec["experiments"]]
    for cfg in configs:
        if args.only and cfg["id"] not in args.only:
            continue
        print(f"=== {cfg['id']}: {cfg['title']}", flush=True)
        result = run_one(cfg, data_root, checkpoints, device)
        (RESULTS / f"{cfg['id']}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  val EER={result['val']['eer']:.4f}  test EER={result['test']['eer']:.4f}", flush=True)

    completed = {cfg["id"]: cfg for cfg in configs if (RESULTS / f"{cfg['id']}.json").exists()}
    val_eer = {
        key: json.loads((RESULTS / f"{key}.json").read_text(encoding="utf-8"))["val"]["eer"] for key in completed
    }
    best = min(val_eer, key=val_eer.get)
    matchers = run_matchers(best, completed[best], data_root, checkpoints, device)
    (RESULTS / "matchers.json").write_text(json.dumps(matchers, indent=1), encoding="utf-8")
    print(f"Melhor por validação: {best}")


if __name__ == "__main__":
    main()
