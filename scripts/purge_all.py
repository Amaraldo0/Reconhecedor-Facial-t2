"""Elimina os dados privados ao término da disciplina (§24 do enunciado).

Uso: python scripts/purge_all.py --confirmar
Mantém apenas o ambiente, o LFW e os checkpoints públicos.
"""
from __future__ import annotations

import argparse
import shutil

from facerec.config import default_home

PRIVATE_DIRS = ("gallery", "private_dataset", "logs", "results", "checkpoints/private")


def purge(home) -> list[str]:
    removed = []
    for relative in PRIVATE_DIRS:
        target = home / relative
        if target.exists():
            shutil.rmtree(target)
            removed.append(relative)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmar", action="store_true", help="confirma a exclusão irreversível")
    args = parser.parse_args()
    home = default_home()
    if not args.confirmar:
        raise SystemExit(f"Nada foi apagado. Para apagar os dados privados em {home}, use --confirmar.")
    print("Removidos:", purge(home) or "nada a remover")


if __name__ == "__main__":
    main()
