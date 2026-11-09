"""Gera a tabela e o gráfico agregados dos experimentos com CelebA.

Uso: python experiments/summarize_celeba.py
"""
from __future__ import annotations

import json

from summarize import BLUE, FIGURES, INK, ORANGE, RESULTS, STRATEGIES, SURFACE, figure, plt


def load() -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULTS.glob("C[0-9]*.json"))}


def main_table(results: dict) -> str:
    lines = [
        "| Exp. | Configuração | Estratégia | Perda | Score (média EER val) | EER val CelebA | EER val LFW | EER teste CelebA (IC 95%) | Acc. CelebA | EER teste LFW (IC 95%) | Acc. LFW | EER degradado CelebA | Tempo (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        cfg, c, l = r["config"], r["celeba"], r["lfw"]
        c_ci, l_ci = c["test"]["ci95"]["eer"], l["test"]["ci95"]["eer"]
        seconds = f"{r['training']['seconds']:.0f}" if r["training"] else "-"
        lines.append(
            f"| {key} | {cfg['title']} | {STRATEGIES[cfg['strategy']]} | {cfg['loss']} | {r['selection_score']:.4f} "
            f"| {c['val']['eer']:.4f} | {l['val']['eer']:.4f} "
            f"| {c['test']['eer']:.4f} ({c_ci[0]:.4f}–{c_ci[1]:.4f}) | {c['test']['at_val_eer_threshold']['accuracy']:.4f} "
            f"| {l['test']['eer']:.4f} ({l_ci[0]:.4f}–{l_ci[1]:.4f}) | {l['test']['at_val_eer_threshold']['accuracy']:.4f} "
            f"| {c['test_degraded']['eer']:.4f} | {seconds} |"
        )
    return "\n".join(lines)


def eer_plot(results: dict) -> None:
    ordered = sorted(results, key=lambda key: results[key]["selection_score"], reverse=True)
    positions = list(range(len(ordered)))
    fig, axis = figure(8, 3.6)
    for dataset, color, marker, label in (("celeba", BLUE, "o", "CelebA"), ("lfw", ORANGE, "s", "LFW")):
        values = [100 * results[key][dataset]["test"]["eer"] for key in ordered]
        low = [100 * results[key][dataset]["test"]["ci95"]["eer"][0] for key in ordered]
        high = [100 * results[key][dataset]["test"]["ci95"]["eer"][1] for key in ordered]
        offset = 0.12 if dataset == "celeba" else -0.12
        shifted = [position + offset for position in positions]
        axis.hlines(shifted, low, high, color=color, linewidth=2, zorder=2)
        axis.scatter(values, shifted, s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, marker=marker, label=f"{label} (IC 95%)")
    axis.set_yticks(positions, [f"{key} · {results[key]['config']['title']}" for key in ordered], color=INK, fontsize=8)
    axis.set_xlabel("EER no conjunto de teste (%; menor é melhor)")
    axis.set_title("Erro nos dois datasets", loc="left", fontweight="bold")
    axis.legend(frameon=False, labelcolor=INK, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "eer_celeba_lfw.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    results = load()
    selection = json.loads((RESULTS / "celeba_selection.json").read_text(encoding="utf-8"))
    eer_plot(results)
    any_result = next(iter(results.values()))
    report = [
        "# Resultados com CelebA (gerado por experiments/summarize_celeba.py)",
        "",
        f"Melhor pela média do EER de validação: **{selection['best']}**. "
        f"Comparador: **{selection['chosen_method']}**. Limiar da aplicação: **{selection['app_threshold']:.3f}**.",
        "",
        main_table(results),
        "",
        "## Controle de vazamento",
        "",
        "```json",
        json.dumps({"entre_particoes": any_result["leakage_control"], "sobreposicao_com_lfw": any_result["overlap_filter"]}, indent=2),
        "```",
        "",
        "## Comparadores (melhor modelo)",
        "",
        "| Dataset | Comparador | Identificação correta (validação) | Identificação correta (teste) | FAR desconhecidos (teste) |",
        "|---|---|---|---|---|",
    ]
    for dataset, methods in selection["methods"].items():
        for method, row in methods.items():
            report.append(f"| {dataset} | {method} | {row['val_dir']:.4f} | {row['dir']:.4f} | {row['far']:.4f} |")
    report += [
        "", "## Limiar por tamanho de galeria", "",
        "| Dataset | Pessoas | FAR-alvo | Limiar (validação) | FAR no teste | Identificação correta no teste |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, rows in selection["gallery_study"].items():
        for row in rows:
            report.append(
                f"| {dataset} | {row['gallery_size']} | {row['target_far']:.3f} | {row['threshold']:.3f} "
                f"| {row['test_far']:.4f} | {row['test_dir']:.4f} |"
            )
    text = "\n".join(report)
    (RESULTS / "SUMMARY_CELEBA.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
