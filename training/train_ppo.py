"""Entrenamiento PPO por layout (PLAN.md §5, §16.2).

CLI:
  python -m training.train_ppo --layout cramped_room --timesteps 5000000 \
      --out models/cramped_room/seed0_default --device cuda --obs lossless_grid \
      --seed 0 --n-envs 8

Notas:
  - obs lossless_grid + CnnPolicy pequena (GPU/A100). featurized+MlpPolicy es opcion
    de CPU pero el StudentAgent desplegado usa lossless_grid (computable desde state).
  - best.zip se guarda por SCORE OFICIAL vs greedy (callback), no por reward.
  - OMP_NUM_THREADS=1 y torch 1 hilo por worker para no sobre-suscribir CPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def linear_schedule(start: float, end: float):
    def f(progress_remaining: float):  # 1.0 -> 0.0 durante el entrenamiento
        return end + (start - end) * progress_remaining
    return f


def build_ppo(venv, obs_kind="lossless_grid", device="cpu", seed=0,
              n_steps=400, batch_size=None, lr=3e-4, ent_coef=0.02,
              n_epochs=8, tb=None, verbose=0):
    from stable_baselines3 import PPO

    n_envs = getattr(venv, "num_envs", 1)
    rollout = n_steps * n_envs
    if batch_size is None:
        batch_size = max(1, rollout // 4)  # 4 minibatches; divide exacto

    if obs_kind == "lossless_grid":
        from training.networks import SmallGridCNN
        policy = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[128, 128],
        )
    else:
        policy = "MlpPolicy"
        policy_kwargs = dict(net_arch=[64, 64])

    return PPO(
        policy, venv, device=device, seed=seed,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
        learning_rate=linear_schedule(lr, lr * 0.1),
        ent_coef=ent_coef, gamma=0.99, gae_lambda=0.98,
        clip_range=0.2, max_grad_norm=0.5, vf_coef=0.5,
        policy_kwargs=policy_kwargs, tensorboard_log=tb, verbose=verbose,
    )


class EntCoefAnnealCallback:
    """ent_coef 0.02 -> 0.001 lineal (PLAN §5). Se aplica como callback simple."""
    pass


def bc_pretrain(model, npz_path: str, epochs: int = 8, batch_size: int = 512, lr: float = 3e-4):
    """Behavior cloning: cross-entropy sobre los logits de la politica con el dataset
    del planner (training/collect_bc_data.py). Implanta la habilidad de soloear ANTES
    del PPO (STEP A-3, M3). Solo entrena la cabeza de politica + extractor; el value
    head lo ajusta PPO despues."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    data = np.load(npz_path)
    obs = torch.as_tensor(data["obs"].astype(np.float32))
    acts = torch.as_tensor(data["actions"].astype(np.int64))
    n = len(acts)
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    model.policy.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        loss_sum, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            b_obs = obs[idx].to(model.device)
            b_act = acts[idx].to(model.device)
            dist = model.policy.get_distribution(b_obs)
            logits = dist.distribution.logits
            loss = F.cross_entropy(logits, b_act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(idx)
            correct += int((logits.argmax(dim=1) == b_act).sum())
        print(f"[bc] epoch {ep + 1}/{epochs} loss={loss_sum / n:.4f} acc={correct / n:.3f}",
              flush=True)
    model.policy.eval()


def make_vec_env(env_config, n_envs, partner_weights, seed, use_subproc):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from training.gym_env import make_env
    thunks = [make_env(env_config, partner_weights=partner_weights, seed=seed + i)
              for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(thunks, start_method="spawn")
    return DummyVecEnv(thunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="cramped_room")
    ap.add_argument("--layout-file", default=None)
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--obs", default="lossless_grid", choices=["lossless_grid", "featurized"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anneal-frac", type=float, default=0.6)
    ap.add_argument("--eval-freq", type=int, default=200_000)
    ap.add_argument("--horizon", type=int, default=200, help="horizon de ENTRENAMIENTO")
    ap.add_argument("--no-subproc", action="store_true")
    ap.add_argument("--partner", default="population",
                    choices=["population", "greedy", "greedy_heavy", "solo_heavy",
                             "sticky_heavy"],
                    help="greedy=100% (sobreajusta). greedy_heavy=55% greedy. "
                         "solo_heavy=60% no-cooperativo (stay+random) -> autosuficiente. "
                         "sticky_heavy=50% sticky (esc.2). population=default balanceado.")
    ap.add_argument("--curriculum", action="store_true",
                    help="fase-1 solo (stay/random 50/50) hasta 35%, luego solo_heavy (STEP A-2)")
    ap.add_argument("--bc-data", default=None,
                    help="npz de collect_bc_data: pre-entrena por imitacion del planner (M3)")
    ap.add_argument("--bc-epochs", type=int, default=8)
    ap.add_argument("--selfplay", action="store_true",
                    help="FCP-lite: poblacion con snapshots congelados propios (M4)")
    ap.add_argument("--no-anneal", action="store_true",
                    help="shaping fijo 1.0 sin annealing (§12-C.1)")
    args = ap.parse_args()

    # 1 hilo por worker (CPU-bound, evitar sobre-suscripcion) - antes de importar torch pesado
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    from stable_baselines3.common.vec_env import VecMonitor
    from training.callbacks import (ShapingAnnealCallback, OfficialScoreEvalCallback,
                                    PartnerCurriculumCallback, SelfPlaySnapshotCallback)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_config = {"layout_name": None if args.layout_file else args.layout,
                  "layout_file": args.layout_file, "horizon": args.horizon,
                  "old_dynamics": True}
    eval_env_config = dict(env_config)

    SOLO_HEAVY = {"stay": 0.25, "random_motion": 0.35, "greedy": 0.30, "greedy_sticky_eps": 0.10}
    if args.partner == "greedy":
        partner_weights = {"greedy": 1.0}
    elif args.partner == "greedy_heavy":
        # Fuerte vs greedy (esc.1-3) pero con random suficiente para NO colapsar en el
        # gate G8 (que prueba random_motion). Sin self-play (evita dependencia de ckpts).
        partner_weights = {"greedy": 0.55, "greedy_sticky_eps": 0.15,
                           "random_motion": 0.25, "stay": 0.05}
    elif args.partner == "solo_heavy":
        partner_weights = dict(SOLO_HEAVY)   # 60% no-cooperativo -> fuerza el solo
    elif args.partner == "sticky_heavy":
        # Especializado en el companero del escenario 2 (greedy+sticky PURO) sin
        # perder las celdas del enable-check: greedy 0.25 y random 0.20 conservan
        # la habilidad vs greedy limpio y el ciclo solo (que el BC ya implanta).
        partner_weights = {"greedy_sticky": 0.35, "greedy_sticky_eps": 0.15,
                           "greedy": 0.25, "random_motion": 0.20, "stay": 0.05}
    else:
        partner_weights = None

    # Curriculum: arrancar en fase-1 (solo puro) y dejar que el callback pase a solo_heavy.
    if args.curriculum:
        partner_weights = {"stay": 0.5, "random_motion": 0.5}   # fase-1

    # Self-play FCP-lite: 35% checkpoints propios (los aporta SelfPlaySnapshotCallback;
    # mientras no haya, la poblacion redistribuye ese peso a greedy).
    if args.selfplay:
        partner_weights = {"stay": 0.10, "random_motion": 0.20, "greedy": 0.25,
                           "greedy_sticky_eps": 0.10, "self_play": 0.35}
    venv = make_vec_env(env_config, args.n_envs, partner_weights=partner_weights,
                        seed=args.seed, use_subproc=not args.no_subproc)
    venv = VecMonitor(venv)

    model = build_ppo(venv, obs_kind=args.obs, device=args.device, seed=args.seed,
                      n_steps=args.n_steps, lr=args.lr, ent_coef=args.ent_coef,
                      tb=str(out / "tb"), verbose=1)

    callbacks = [OfficialScoreEvalCallback(eval_env_config, str(out), eval_freq=args.eval_freq)]
    if not args.no_anneal:
        callbacks.insert(0, ShapingAnnealCallback(args.timesteps, args.anneal_frac))
    # con --no-anneal el coef queda en initial_coef=1.0 (env) todo el entrenamiento
    if args.curriculum:
        callbacks.insert(0, PartnerCurriculumCallback(
            args.timesteps, switch_frac=0.35,
            phase1={"stay": 0.5, "random_motion": 0.5}, phase2=dict(SOLO_HEAVY), verbose=1))
    if args.selfplay:
        callbacks.insert(0, SelfPlaySnapshotCallback(
            every_steps=max(500_000, args.timesteps // 12),
            pool_dir=str(out / "selfplay_pool"), verbose=1))

    # M3: behavior cloning desde el planner ANTES del PPO (implanta el ciclo solo).
    if args.bc_data:
        print(f"[bc] pre-entrenando por imitacion desde {args.bc_data} ...", flush=True)
        bc_pretrain(model, args.bc_data, epochs=args.bc_epochs)

    # Config versionado junto al modelo (PLAN §16.2)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(str(out / "final.zip"))
    print(f"[train_ppo] listo. modelos en {out} (best.zip por score oficial, final.zip)")


if __name__ == "__main__":
    main()
