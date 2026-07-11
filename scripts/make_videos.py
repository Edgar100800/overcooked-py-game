"""Simulacion en video del StudentAgent vs los 3 tipos de companero.

*** HERRAMIENTA BAJO DEMANDA: correr SOLO cuando se pida explicitamente. ***
No forma parte de gates, night_loop ni de ningun proceso automatico.

Corre un episodio real (rollout del runner oficial) del StudentAgent (selector hibrido
con sonda de cooperacion; usa el PPO habilitado si existe para el layout) contra
greedy / greedy+eps / random_motion, y guarda un GIF + MP4 por companero.

Uso (headless, en el login):
  python -m scripts.make_videos                                  # zigzag (PPO habilitado)
  python -m scripts.make_videos --layout cramped_room --layout-file ""   # otro layout
  python -m scripts.make_videos --seed 3 --fps 8
Salida: outputs/videos/<layout>_vs_<companero>.gif y .mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# pygame headless (nodo sin display) — ANTES de importar pygame/visualizer.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from evaluation import gate_configs as gc            # noqa: E402
from src.runner import run_from_config               # noqa: E402

PARTNERS = ["greedy", "greedy_eps", "random_motion"]


def to_mp4(gif_path: Path, fps: float) -> Path | None:
    """Convierte el GIF a MP4 (H.264) con imageio-ffmpeg. Devuelve None si falla."""
    try:
        import imageio.v2 as imageio
        frames = imageio.mimread(gif_path, memtest=False)
        # H.264/yuv420p exige dimensiones PARES: recortar 1px si hace falta.
        h, w = frames[0].shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)
        frames = [f[:h2, :w2, :3] for f in frames]
        mp4 = gif_path.with_suffix(".mp4")
        imageio.mimsave(mp4, frames, fps=fps, macro_block_size=1)
        return mp4
    except Exception as exc:
        print(f"  (sin mp4: {exc!r}; queda el GIF)")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="custom_zigzag_kitchen")
    ap.add_argument("--layout-file", default="configs/layouts/custom_zigzag_kitchen.layout",
                    help='"" para layouts oficiales')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8, help="fps del video (y del render)")
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--swap", action="store_true", help="ademas el episodio con roles invertidos")
    ap.add_argument("--out-dir", default="outputs/videos")
    args = ap.parse_args()

    lf = args.layout_file or None
    key = Path(lf).stem if lf else args.layout
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scfg = {"layout": args.layout, "layout_file": lf}   # carga el PPO habilitado si existe
    print(f"[videos] layout={key} seed={args.seed} (PPO habilitado: "
          f"{(Path('models') / key / 'enabled').exists()})")

    for pk in PARTNERS:
        gif_path = out_dir / f"{key}_vs_{pk}.gif"
        cfg = gc.make_config(args.layout, lf, gc.student_agent(dict(scfg)), gc.partner(pk),
                             horizon=args.horizon)
        cfg["execution"] = {"num_episodes": 1, "episode_seeds": [args.seed],
                            "swap_agent_positions": bool(args.swap)}
        cfg["rendering"] = {"mode": "gif", "fps": args.fps, "save_gif": True,
                            "gif_path": str(gif_path)}
        cfg["logging"] = {"output_dir": str(out_dir / f"_log_{key}_{pk}"),
                          "save_step_log": False, "save_episode_summary": False}
        print(f"[videos] simulando vs {pk} ...")
        res = run_from_config(cfg)
        soups = [int(round(r / 20.0)) for r in res["returns_sparse"]]
        mp4 = to_mp4(gif_path, args.fps)
        print(f"  sopas={soups}  gif={gif_path}" + (f"  mp4={mp4}" if mp4 else ""))

    print(f"\n[videos] listo. Archivos en {out_dir}/")


if __name__ == "__main__":
    main()
