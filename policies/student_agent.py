"""StudentAgent — selector hibrido y ENTREGA FINAL (PLAN.md §6).

Compatible con el loader python_class: `__init__(config)`, `reset()`, `act(obs)->int`.
Requiere `observation.type: state` (obs = {"state","mdp","agent_index"}).

Estrategia:
  - Piso garantizado: PlannerAgent (sin aprendizaje, generaliza a cualquier layout).
  - Techo: modelo PPO por layout SI existe y esta habilitado (G7 paso en gate_seeds
    -> se escribe models/<layout>/enabled). El modelo usa el encoding lossless_grid,
    computable desde mdp+state (sin mlam, rapido).
  - Fusibles: cualquier excepcion -> planner. Latencia del PPO > fuse_ms -> planner el
    RESTO del episodio (proactivo, ademas del SafeActionWrapper de 100 ms). Nunca
    devuelve una accion invalida.

El modelo se precarga en __init__ (no en act(), para no arriesgar el timeout del
primer paso). Para activarlo, el config debe traer el `layout` (nombre oficial) o
`layout_file`; si no, o si no hay modelo habilitado, corre solo-planner. Esto tambien
cubre escenarios 5-6 (layout desconocido -> planner).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from policies.planner_agent import PlannerAgent

STAY_IDX = 4
REPO = Path(__file__).resolve().parent.parent


def _layout_key(config: dict) -> str | None:
    """Clave de layout desde el config (nombre oficial o basename del .layout)."""
    lf = config.get("layout_file")
    if lf:
        return Path(lf).stem
    return config.get("layout") or config.get("layout_name")


def terrain_key(mdp) -> str:
    """Hash estable del terrain_mtx (fallback para detectar layout en runtime)."""
    try:
        rows = ["".join(r) for r in mdp.terrain_mtx]
        return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


class StudentAgent:
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.config = config
        self.models_dir = REPO / config.get("models_dir", "models")
        self.fuse_ms = float(config.get("latency_fuse_ms", 60.0))
        self.force_planner = bool(config.get("force_planner", False))

        self.planner = PlannerAgent(config)
        self.model = None
        self.model_layout = None
        self._fused = False
        # Sonda de cooperacion (ad-hoc teaming, PLAN §15/HAHA): el PPO solo se usa si el
        # companero demostro cooperar (sostuvo un objeto alguna vez en el episodio).
        # random_motion/stay JAMAS sostienen nada -> planner todo el episodio (robusto);
        # greedy agarra una cebolla en ~5 pasos -> PPO el resto. Desactivable via config.
        self.partner_probe = bool(config.get("partner_probe", True))
        self._partner_cooperative = False
        # require_enabled=False permite a G6/G7 evaluar un modelo candidato ANTES de
        # habilitarlo. En despliegue normal queda True (anti-regresion §12-E.2).
        self.require_enabled = bool(config.get("require_enabled", True))
        self.explicit_model_path = config.get("model_path")

        if not self.force_planner:
            self._try_preload(_layout_key(config))

    # ------------------------------------------------------------------ carga
    def _try_preload(self, layout_key: str | None):
        if self.explicit_model_path:
            best = Path(self.explicit_model_path)
            enabled_ok = True  # ruta explicita (evaluacion de candidato)
        elif layout_key:
            model_dir = self.models_dir / layout_key
            best = model_dir / "best.zip"
            enabled_ok = (model_dir / "enabled").exists() or not self.require_enabled
        else:
            return
        # Activar PPO solo si el modelo existe y (esta habilitado o es evaluacion).
        if best.exists() and enabled_ok:
            try:
                import torch
                torch.set_num_threads(1)
                from stable_baselines3 import PPO
                self.model = PPO.load(str(best), device="cpu")
                self.model_layout = layout_key or "explicit"
            except Exception:
                self.model = None

    # ------------------------------------------------------------------- API
    def reset(self):
        self.planner.reset()
        self._fused = False
        self._partner_cooperative = False

    def act(self, obs) -> int:
        # Camino planner si: forzado, sin modelo, o fusible disparado.
        if self.force_planner or self.model is None or self._fused:
            return self._planner_act(obs)

        # Sonda de cooperacion: hasta que el companero sostenga un objeto, planner.
        if self.partner_probe and not self._partner_cooperative:
            try:
                state = obs["state"]
                idx = int(obs.get("agent_index", 0))
                partner = state.players[1 - idx]
                if partner.held_object is not None:
                    self._partner_cooperative = True   # coopera -> PPO desde ahora
            except Exception:
                pass
            if not self._partner_cooperative:
                return self._planner_act(obs)

        try:
            t0 = time.perf_counter()
            action = self._ppo_act(obs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > self.fuse_ms:
                # Latencia peligrosa -> degradar a planner el resto del episodio.
                self._fused = True
                return self._planner_act(obs)
            return int(action)
        except Exception:
            return self._planner_act(obs)

    # --------------------------------------------------------------- internos
    def _planner_act(self, obs) -> int:
        try:
            return int(self.planner.act(obs))
        except Exception:
            return STAY_IDX

    def _ppo_act(self, obs) -> int:
        state = obs["state"]
        mdp = obs["mdp"]
        agent_index = int(obs.get("agent_index", 0))
        enc = mdp.lossless_state_encoding(state)[agent_index]
        x = np.ascontiguousarray(np.asarray(enc, dtype=np.float32).transpose(2, 0, 1))
        act_idx, _ = self.model.predict(x, deterministic=True)
        act_idx = int(act_idx)
        if act_idx < 0 or act_idx > 5:
            raise ValueError(f"accion PPO invalida: {act_idx}")
        return act_idx
