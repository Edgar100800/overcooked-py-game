"""Simulacion en video del StudentAgent: vs companeros builtin o DUELOS entre modelos.

*** HERRAMIENTA BAJO DEMANDA: correr SOLO cuando se pida explicitamente. ***
No forma parte de gates, night_loop ni de ningun proceso automatico.

Cada video lleva un BANNER con la leyenda de gorros del visualizador oficial:
  gorro AZUL  = jugador 0 (agent_0)     gorro VERDE = jugador 1 (agent_1)
Salida organizada por layout: outputs/videos/<layout>/<match>.gif y .mp4
(logs del runner en outputs/videos/<layout>/_logs/).

Uso (headless, en el login):
  python -m scripts.make_videos                                   # student vs 3 builtin
  python -m scripts.make_videos --duel                            # duelos: student vs student / vs planner
  python -m scripts.make_videos --layout rehearsal_kitchen \
      --layout-file configs/layouts/rehearsal_kitchen.layout --duel
  python -m scripts.make_videos --seed 3 --fps 8 --swap
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
# Colores de los gorros del StateVisualizer oficial: player_colors = ['blue','green']
HAT_BLUE = (60, 100, 230)
HAT_GREEN = (40, 160, 70)


# ------------------------------------------------------------------ post-pro
def add_banner(gif_path: Path, label_p0: str, label_p1: str, fps: float):
    """Reescribe el GIF con una franja superior: gorro azul/verde = que agente es."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    frames = imageio.mimread(gif_path, memtest=False)
    w = frames[0].shape[1]
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:                       # Pillow viejo: sin size
        font = ImageFont.load_default()

    # Dos filas (una por gorro): nunca se corta aunque el frame sea angosto.
    bh = 48
    banner = Image.new("RGB", (w, bh), (24, 24, 24))
    d = ImageDraw.Draw(banner)
    for row, (color, label) in enumerate(((HAT_BLUE, f"gorro azul: {label_p0}"),
                                          (HAT_GREEN, f"gorro verde: {label_p1}"))):
        y = 5 + row * 22
        d.rectangle([6, y + 1, 20, y + 15], fill=color, outline=(255, 255, 255))
        d.text((26, y), label, fill=(240, 240, 240), font=font)
    banner = np.asarray(banner)

    out = [np.concatenate([banner, f[:, :, :3]], axis=0) for f in frames]
    imageio.mimsave(gif_path, out, duration=1000.0 / fps)
    return out


def to_mp4(gif_path: Path, fps: float, frames=None) -> Path | None:
    """Convierte el GIF (o frames ya en memoria) a MP4 H.264."""
    try:
        import imageio.v2 as imageio
        if frames is None:
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


# ------------------------------------------------------------------- partidas
def play(layout, lf, a0, a1, label0, label1, name, out_dir: Path, args):
    """Corre 1 episodio real, guarda GIF+MP4 con banner y reporta sopas."""
    gif_path = out_dir / f"{name}.gif"
    cfg = gc.make_config(layout, lf, a0, a1, horizon=args.horizon)
    cfg["execution"] = {"num_episodes": 1, "episode_seeds": [args.seed],
                        "swap_agent_positions": bool(args.swap)}
    cfg["rendering"] = {"mode": "gif", "fps": args.fps, "save_gif": True,
                        "gif_path": str(gif_path)}
    cfg["logging"] = {"output_dir": str(out_dir / "_logs" / name),
                      "save_step_log": False, "save_episode_summary": False}
    print(f"[videos] {name}: {label0} (azul) + {label1} (verde) ...")
    res = run_from_config(cfg)
    soups = [int(round(r / 20.0)) for r in res["returns_sparse"]]
    frames = add_banner(gif_path, label_p0=label0, label_p1=label1, fps=args.fps)
    mp4 = to_mp4(gif_path, args.fps, frames=frames)
    print(f"  sopas={soups}  {gif_path}" + (f"  {mp4}" if mp4 else ""))
    return soups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="custom_zigzag_kitchen")
    ap.add_argument("--layout-file", default="configs/layouts/custom_zigzag_kitchen.layout",
                    help='"" para layouts oficiales')
    ap.add_argument("--duel", action="store_true",
                    help="duelos: student vs student y student vs planner (en vez de builtin)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8, help="fps del video (y del render)")
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--swap", action="store_true", help="episodio con roles invertidos")
    ap.add_argument("--out-dir", default="outputs/videos")
    args = ap.parse_args()

    lf = args.layout_file or None
    key = Path(lf).stem if lf else args.layout
    out_dir = Path(args.out_dir) / key
    out_dir.mkdir(parents=True, exist_ok=True)

    scfg = {"layout": args.layout, "layout_file": lf}
    ppo_on = (Path("models") / key / "enabled").exists()
    stu_label = f"StudentAgent (PPO {key})" if ppo_on else "StudentAgent (planner)"
    print(f"[videos] layout={key} seed={args.seed} PPO habilitado: {ppo_on}")

    if args.duel:
        play(args.layout, lf, gc.student_agent(dict(scfg)), gc.student_agent(dict(scfg)),
             stu_label, stu_label, f"{key}__student_vs_student", out_dir, args)
        play(args.layout, lf, gc.student_agent(dict(scfg)), gc.planner_agent(),
             stu_label, "PlannerAgent (robot)", f"{key}__student_vs_planner", out_dir, args)
    else:
        for pk in PARTNERS:
            play(args.layout, lf, gc.student_agent(dict(scfg)), gc.partner(pk),
                 stu_label, f"builtin {pk}", f"{key}__vs_{pk}", out_dir, args)

    print(f"\n[videos] listo. Archivos en {out_dir}/")


if __name__ == "__main__":
    main()
