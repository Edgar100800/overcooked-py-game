"""Video "cerebro de la red": juego a la izquierda + nodos del PPO a la derecha.

*** HERRAMIENTA BAJO DEMANDA: correr SOLO cuando se pida explicitamente. ***
No forma parte de gates, night_loop ni de ningun proceso automatico.

Por cada paso del episodio se muestra, EN PARALELO al juego:
  - quien decide (PPO / planner por sonda / planner por fusible),
  - las probabilidades que la red asigna a las 6 acciones,
  - el valor V(s) estimado (con su historia),
  - los mapas de activacion de las 3 convs del SmallGridCNN y los 128 features.
La red "opina" en todos los pasos aunque decida el planner (panel apagado).

Uso (headless, en el login):
  python -m scripts.make_brain_videos                                  # zigzag vs greedy
  python -m scripts.make_brain_videos --partner random_motion          # sonda -> planner
  python -m scripts.make_brain_videos --layout rehearsal_kitchen \
      --layout-file configs/layouts/rehearsal_kitchen.layout --partner student
Salida: outputs/videos/<layout>/<layout>__brain_vs_<partner>.gif y .mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np                                     # noqa: E402
from overcooked_ai_py.agents.agent import AgentPair    # noqa: E402

from evaluation import gate_configs as gc              # noqa: E402
from evaluation.rollout import _seed_builtin_partners  # noqa: E402
from src.environment import build_env                  # noqa: E402
from src.observations import ObservationBuilder        # noqa: E402
from src.policy_loader import build_two_policies       # noqa: E402
from src.rendering import Renderer                     # noqa: E402
from src.runner import set_global_seed                 # noqa: E402
from scripts.make_videos import to_mp4, HAT_BLUE, HAT_GREEN  # noqa: E402

PANEL_W = 380
ACTION_LABELS = ["N (arriba)", "S (abajo)", "E (der)", "O (izq)", "quieto", "interactuar"]
DIM = 0.45          # atenuacion del panel cuando decide el planner


def unwrap_student(agent):
    """SafeActionWrapper.base_agent -> (Eps...) -> StudentAgentAdapter.student_agent."""
    x = agent
    for _ in range(6):
        if hasattr(x, "base_agent"):
            x = x.base_agent
        elif hasattr(x, "student_agent"):
            return x.student_agent
        else:
            break
    return None


def who_decides(student):
    """Replica el orden de decision de StudentAgent.act()."""
    if student is None or getattr(student, "model", None) is None:
        return "planner-sin-modelo"
    if getattr(student, "_fused", False):
        return "planner-fusible"
    if getattr(student, "partner_probe", False) and not getattr(student, "_partner_cooperative", False):
        return "planner-sonda"
    return "ppo"


def heat_rgb(a: np.ndarray) -> np.ndarray:
    """Colormap negro->naranja->blanco para un mapa 2D normalizado [0,1]."""
    a = np.clip(a, 0.0, 1.0)
    r = np.clip(a * 2.2, 0, 1)
    g = np.clip(a * 1.4 - 0.15, 0, 1)
    b = np.clip(a * 2.0 - 1.2, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


class BrainProbe:
    """Forward de la politica con hooks en las convs (misma obs que _ppo_act)."""

    def __init__(self, model):
        import torch
        self.torch = torch
        self.model = model
        self.acts = {}
        fe = model.policy.features_extractor
        for name, idx in (("conv1", 1), ("conv2", 3), ("conv3", 5)):
            fe.cnn[idx].register_forward_hook(self._save(name))
        fe.linear.register_forward_hook(self._save("features"))

    def _save(self, name):
        def hook(_m, _inp, out):
            self.acts[name] = out.detach().cpu().numpy()
        return hook

    def peek(self, mdp, state, agent_index: int):
        enc = mdp.lossless_state_encoding(state)[agent_index]
        x = np.ascontiguousarray(np.asarray(enc, dtype=np.float32).transpose(2, 0, 1))
        obs_t, _ = self.model.policy.obs_to_tensor(x)
        with self.torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy()[0]
            value = float(self.model.policy.predict_values(obs_t).item())
        maps = [self.acts[k][0].mean(axis=0) for k in ("conv1", "conv2", "conv3")]
        feats = self.acts["features"][0]
        return probs, value, maps, feats


def draw_panel(h: int, t: int, soups: int, who: str, probs, value, values_hist,
               maps, feats, exec_idx: int):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=13)
        small = ImageFont.load_default(size=11)
    except TypeError:
        font = small = ImageFont.load_default()

    img = Image.new("RGB", (PANEL_W, h), (18, 18, 22))
    d = ImageDraw.Draw(img)
    ppo_on = who == "ppo"
    mul = 1.0 if ppo_on else DIM

    def col(c):  # atenua colores cuando el planner decide
        return tuple(int(v * mul) for v in c)

    # -- cabecera: quien decide
    badge = {"ppo": ((30, 160, 60), "PPO DECIDE"),
             "planner-sonda": ((200, 140, 20), "PLANNER (sonda: esperando cooperacion)"),
             "planner-fusible": ((190, 40, 40), "PLANNER (fusible activado)"),
             "planner-sin-modelo": ((90, 90, 90), "PLANNER (sin modelo PPO)")}[who]
    d.rectangle([0, 0, PANEL_W, 22], fill=badge[0])
    d.text((8, 4), badge[1], fill=(255, 255, 255), font=font)
    d.text((8, 27), f"paso {t}    sopas {soups}", fill=(200, 200, 200), font=small)

    y = 48
    if probs is None:
        d.text((8, y), "sin red PPO para este layout", fill=(150, 150, 150), font=font)
        return np.asarray(img)

    # -- probabilidades de accion
    d.text((8, y), "que quiere hacer la red (prob. de accion)"
           + ("" if ppo_on else "  [opina, no decide]"), fill=col((235, 235, 235)), font=small)
    y += 16
    best = int(np.argmax(probs))
    for i, (lab, p) in enumerate(zip(ACTION_LABELS, probs)):
        bw = int(p * (PANEL_W - 150))
        color = (60, 200, 90) if i == best else (70, 110, 200)
        d.rectangle([120, y + 2, 120 + max(bw, 1), y + 13], fill=col(color))
        mark = " <- ejecutada" if i == exec_idx else ""
        d.text((8, y), lab, fill=col((215, 215, 215)), font=small)
        d.text((124 + max(bw, 1), y), f"{p:.2f}{mark}", fill=col((170, 170, 170)), font=small)
        y += 16

    # -- V(s) + sparkline
    y += 6
    d.text((8, y), f"V(s) = {value:+.2f}  (cuanta recompensa espera)", fill=col((235, 235, 235)), font=small)
    y += 16
    if len(values_hist) > 1:
        vs = np.array(values_hist[-160:])
        lo, hi = float(vs.min()), float(vs.max())
        rng = (hi - lo) or 1.0
        pts = [(8 + int(i * (PANEL_W - 16) / max(len(vs) - 1, 1)),
                y + 26 - int((v - lo) / rng * 24)) for i, v in enumerate(vs)]
        d.line(pts, fill=col((90, 190, 230)), width=1)
    y += 34

    # -- activaciones de las convs
    d.text((8, y), "nodos activos por capa (media de canales)", fill=col((235, 235, 235)), font=small)
    y += 15
    x = 8
    for name, m in zip(("conv1 32c", "conv2 64c", "conv3 64c"), maps):
        mm = m - m.min()
        mm = mm / (mm.max() or 1.0)
        tile = Image.fromarray((heat_rgb(mm) * mul).astype(np.uint8))
        scale = max(1, min(96 // tile.width, 96 // tile.height, 16))
        tile = tile.resize((tile.width * scale, tile.height * scale), Image.NEAREST)
        img.paste(tile, (x, y + 12))
        d.text((x, y), name, fill=col((170, 170, 170)), font=small)
        x += tile.width + 14
    y += 12 + max(m.shape[0] for m in maps) * 16 + 8
    y = min(y, h - 50)

    # -- los 128 features finales
    d.text((8, y), "128 features -> cabezas pi/vf", fill=col((235, 235, 235)), font=small)
    ftile = feats.reshape(8, 16)
    ftile = ftile - ftile.min()
    ftile = ftile / (ftile.max() or 1.0)
    fimg = Image.fromarray((heat_rgb(ftile) * mul).astype(np.uint8)).resize((16 * 14, 8 * 4), Image.NEAREST)
    img.paste(fimg, (8, y + 14))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="custom_zigzag_kitchen")
    ap.add_argument("--layout-file", default="configs/layouts/custom_zigzag_kitchen.layout")
    ap.add_argument("--partner", default="greedy",
                    choices=["greedy", "greedy_eps", "random_motion", "stay", "planner", "student"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--out-dir", default="outputs/videos")
    args = ap.parse_args()

    lf = args.layout_file or None
    key = Path(lf).stem if lf else args.layout
    out_dir = Path(args.out_dir) / key
    out_dir.mkdir(parents=True, exist_ok=True)

    scfg = {"layout": args.layout, "layout_file": lf}
    if args.partner == "planner":
        a1_spec = gc.planner_agent()
    elif args.partner == "student":
        a1_spec = gc.student_agent(dict(scfg))
    else:
        a1_spec = gc.partner(args.partner)
    cfg = gc.make_config(args.layout, lf, gc.student_agent(dict(scfg)), a1_spec,
                         horizon=args.horizon)
    cfg = _seed_builtin_partners(cfg, args.seed)

    set_global_seed(args.seed)
    env = build_env(cfg["environment"])
    obs_builder = ObservationBuilder(env, cfg["observation"])
    a0, a1 = build_two_policies(cfg, env, obs_builder, seed=args.seed)
    pair = AgentPair(a0, a1)
    env.reset(regen_mdp=False)
    pair.reset()
    pair.set_mdp(env.mdp)

    student = unwrap_student(a0)
    probe = BrainProbe(student.model) if (student is not None and student.model is not None) else None
    renderer = Renderer({"mode": "rgb_array"})
    print(f"[brain] layout={key} partner={args.partner} PPO cargado: {probe is not None}")

    frames, values_hist, soups = [], [], 0
    for t in range(args.horizon):
        state = env.state
        game = renderer.render_frame(env)

        probs = value = maps = feats = None
        if probe is not None:
            probs, value, maps, feats = probe.peek(env.mdp, state, 0)
            values_hist.append(value)

        ja_infos = pair.joint_action(state)
        joint_action = tuple(a for a, _ in ja_infos)
        exec_idx = ja_infos[0][1].get("action_index", -1) if len(ja_infos[0]) > 1 else -1

        panel = draw_panel(game.shape[0], t, soups, who_decides(student), probs, value,
                           values_hist, maps, feats, exec_idx)
        frames.append(np.concatenate([game[:, :, :3], panel], axis=1))

        _, _, done, info = env.step(joint_action)
        sparse = info.get("sparse_r_by_agent") or [0, 0]
        if sum(sparse) > 0:
            soups += int(round(sum(sparse) / 20.0))
        if done:
            break

    import imageio.v2 as imageio
    gif_path = out_dir / f"{key}__brain_vs_{args.partner}.gif"
    imageio.mimsave(gif_path, frames, duration=1000.0 / args.fps)
    mp4 = to_mp4(gif_path, args.fps, frames=frames)
    print(f"[brain] sopas={soups}  gif={gif_path}" + (f"  mp4={mp4}" if mp4 else ""))


if __name__ == "__main__":
    main()
