"""Escribe models/<key>/terrain.key para los modelos YA habilitados.

enable_model.py lo escribe automaticamente para habilitaciones nuevas; este script
retro-llena los que se habilitaron antes de que existiera el fallback por hash.

Uso: python -m scripts.build_terrain_index
"""

from __future__ import annotations

from pathlib import Path

from scripts.enable_model import terrain_hash

REPO = Path(__file__).resolve().parent.parent
OFFICIAL = {"cramped_room", "asymmetric_advantages", "coordination_ring"}


def main():
    n = 0
    for marker in sorted((REPO / "models").glob("*/enabled")):
        key = marker.parent.name
        out = marker.parent / "terrain.key"
        if key in OFFICIAL:
            layout, layout_file = key, None
        else:
            lf = REPO / "configs" / "layouts" / f"{key}.layout"
            if not lf.exists():
                print(f"[index] {key}: sin .layout conocido, se omite")
                continue
            layout, layout_file = key, str(lf)
        h = terrain_hash(layout, layout_file)
        out.write_text(h + "\n")
        print(f"[index] {key}: terrain.key = {h}")
        n += 1
    print(f"[index] listo: {n} modelo(s) indexado(s).")


if __name__ == "__main__":
    main()
