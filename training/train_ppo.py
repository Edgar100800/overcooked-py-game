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
                    choices=["population", "greedy", "greedy_heavy"],
                    help="greedy=100% greedy (sobreajusta, rompe G8 vs random). "
                         "greedy_heavy=fuerte vs greedy PERO robusto vs random (recomendado). "
                         "population=default balanceado.")
    ap.add_argument("--no-anneal", action="store_true",
                    help="shaping fijo 1.0 sin annealing (§12-C.1)")
    args = ap.parse_args()

    # 1 hilo por worker (CPU-bound, evitar sobre-suscripcion) - antes de importar torch pesado
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    from stable_baselines3.common.vec_env import VecMonitor
    from training.callbacks import ShapingAnnealCallback, OfficialScoreEvalCallback

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_config = {"layout_name": None if args.layout_file else args.layout,
                  "layout_file": args.layout_file, "horizon": args.horizon,
                  "old_dynamics": True}
    eval_env_config = dict(env_config)

    if args.partner == "greedy":
        partner_weights = {"greedy": 1.0}
    elif args.partner == "greedy_heavy":
        # Fuerte vs greedy (esc.1-3) pero con random suficiente para NO colapsar en el
        # gate G8 (que prueba random_motion). Sin self-play (evita dependencia de ckpts).
        partner_weights = {"greedy": 0.55, "greedy_sticky_eps": 0.15,
                           "random_motion": 0.25, "stay": 0.05}
    else:
        partner_weights = None
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

    # Config versionado junto al modelo (PLAN §16.2)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(str(out / "final.zip"))
    print(f"[train_ppo] listo. modelos en {out} (best.zip por score oficial, final.zip)")


if __name__ == "__main__":
    main()
