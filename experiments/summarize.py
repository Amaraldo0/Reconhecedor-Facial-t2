"""Gera a tabela e os gráficos agregados a partir de experiments/results/*.json.

Uso: python experiments/summarize.py
"""
from __future__ import annotations

import json

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from facerec.config import REPO_ROOT  # noqa: E402

RESULTS = REPO_ROOT / "experiments" / "results"
FIGURES = RESULTS / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STRATEGIES = {"baseline": "sem fine-tuning", "partial": "fine-tuning parcial", "full": "fine-tuning completo", "scratch": "do zero"}
MODELS = {"vggface2": "FaceNet VGGFace2", "casia-webface": "FaceNet CASIA", None: "Inception-ResNet"}


def load() -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS.glob("E*.json"))
    }


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(INK)


def figure(width: float, height: float):
    fig, axis = plt.subplots(figsize=(width, height), dpi=150, facecolor=SURFACE)
    style(axis)
    return fig, axis


def eer_plot(results: dict) -> None:
    ordered = sorted(results, key=lambda key: results[key]["test"]["eer"], reverse=True)
    positions = range(len(ordered))
    fig, axis = figure(8, 4.6)
    test = [100 * results[key]["test"]["eer"] for key in ordered]
    degraded = [100 * results[key]["test_degraded"]["eer"] for key in ordered]
    low = [100 * results[key]["test"]["ci95"]["eer"][0] for key in ordered]
    high = [100 * results[key]["test"]["ci95"]["eer"][1] for key in ordered]
    axis.hlines(positions, low, high, color=BLUE, linewidth=2, zorder=2)
    axis.scatter(test, positions, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Teste (IC 95%)")
    axis.scatter(degraded, positions, s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3, marker="s", label="Teste degradado")
    axis.set_yticks(list(positions), [f"{key} · {results[key]['config']['title']}" for key in ordered], color=INK, fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("EER no conjunto de teste (%, escala logarítmica; menor é melhor)")
    axis.set_title("Erro por experimento", loc="left", fontweight="bold")
    axis.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "eer_por_experimento.png", facecolor=SURFACE)
    plt.close(fig)


def roc_plot(results: dict, keys: list[str]) -> None:
    fig, axis = figure(6.5, 4.2)
    for key, color in zip(keys, (BLUE, ORANGE, AQUA)):
        roc = results[key]["curves"]["roc"]
        axis.plot(roc["fpr"], roc["tpr"], color=color, linewidth=2, label=f"{key} · {results[key]['config']['title']}")
    axis.set_xscale("log")
    axis.set_xlim(1e-3, 1)
    axis.set_ylim(0.9, 1.002)
    axis.set_xlabel("Taxa de falsa aceitação (escala logarítmica)")
    axis.set_ylabel("Taxa de verdadeira aceitação")
    axis.set_title("Curva ROC no conjunto de teste", loc="left", fontweight="bold")
    axis.legend(frameon=False, labelcolor=INK, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc.png", facecolor=SURFACE)
    plt.close(fig)


def distribution_plot(result: dict, key: str, threshold: float) -> None:
    histogram = result["curves"]["histogram"]
    centers = [(a + b) / 2 for a, b in zip(histogram["edges"], histogram["edges"][1:])]
    fig, axis = figure(6.5, 3.8)
    axis.plot(centers, histogram["same"], color=BLUE, linewidth=2, label="Mesma pessoa")
    axis.plot(centers, histogram["different"], color=ORANGE, linewidth=2, label="Pessoas diferentes")
    axis.axvline(threshold, color=MUTED, linewidth=1.2, linestyle="--")
    axis.annotate(f"threshold {threshold:.2f}", (threshold, axis.get_ylim()[1]), xytext=(-5, -12), textcoords="offset points", color=INK, fontsize=9, ha="right")
    axis.set_xlabel("Similaridade de cosseno")
    axis.set_ylabel("Densidade")
    axis.set_title(f"Distribuições de similaridade no teste ({key})", loc="left", fontweight="bold")
    axis.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "distribuicao_similaridade.png", facecolor=SURFACE)
    plt.close(fig)


def table(results: dict) -> str:
    lines = [
        "| Experimento | Modelo | Estratégia | Perda | Dataset | Accuracy | F1 | AUC | EER val | EER teste (IC 95%) | FAR | FRR | EER degradado | Tempo (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, result in results.items():
        config, test = result["config"], result["test"]
        at = test["at_val_eer_threshold"]
        low, high = test["ci95"]["eer"]
        seconds = f"{result['training']['seconds']:.0f}" if result["training"] else "-"
        lines.append(
            f"| {key} | {MODELS[config['pretrained']]} | {STRATEGIES[config['strategy']]} | {config['loss']} | LFW "
            f"| {at['accuracy']:.4f} | {at['f1']:.4f} | {test['auc']:.4f} | {result['val']['eer']:.4f} "
            f"| {test['eer']:.4f} ({low:.4f}–{high:.4f}) | {at['far']:.4f} | {at['frr']:.4f} "
            f"| {result['test_degraded']['eer']:.4f} | {seconds} |"
        )
    return "\n".join(lines)


def matcher_table() -> str:
    path = RESULTS / "matchers.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"Modelo: {data['model_experiment']}. FAR-alvo de 1% definido na validação.",
        "",
        "| Comparador | Identificação correta (validação) | Identificação correta (teste) | FAR entre desconhecidos (teste) | Identidade errada aceita (teste) | Threshold |",
        "|---|---|---|---|---|---|",
    ]
    for method, row in data["methods"].items():
        lines.append(
            f"| {method} | {row['val_dir']:.4f} | {row['dir']:.4f} | {row['far']:.4f} "
            f"| {row['wrong_identity_rate']:.4f} | {row['threshold']:.3f} |"
        )
    lines += [
        "",
        f"## Threshold por tamanho de galeria ({data['chosen_method']})",
        "",
        "Threshold calibrado na validação e medido no teste, com 30 sorteios de galeria.",
        "",
        "| Pessoas na galeria | FAR-alvo | Threshold | FAR no teste | Identificação correta no teste |",
        "|---|---|---|---|---|",
    ]
    for row in data["gallery_study"]:
        lines.append(
            f"| {row['gallery_size']} | {row['target_far']:.3f} | {row['threshold']:.3f} "
            f"| {row['test_far']:.4f} | {row['test_dir']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = load()
    best = min(results, key=lambda key: results[key]["val"]["eer"])
    eer_plot(results)
    keys = [key for key in dict.fromkeys(["E01", "E02", best]) if key in results]
    if len(keys) < 3 and "E09" in results:
        keys.append("E09")
    roc_plot(results, keys)
    app_config = yaml.safe_load((REPO_ROOT / "configs" / "app.yaml").read_text(encoding="utf-8"))
    distribution_plot(results[best], best, app_config["recognition"]["threshold"])

    leakage = results[best]["leakage_control"]
    report = [
        "# Resultados agregados (gerado por experiments/summarize.py)",
        "",
        f"Melhor por EER de validação: **{best}**.",
        "",
        table(results),
        "",
        "## Comparadores",
        "",
        matcher_table(),
        "",
        "## Controle de vazamento",
        "",
        "```json",
        json.dumps(leakage, indent=2),
        "```",
        "",
        "## Thresholds do melhor modelo (validação)",
        "",
        "```json",
        json.dumps({k: v for k, v in results[best]["val"].items() if "threshold" in k}, indent=2),
        "```",
    ]
    (RESULTS / "SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
