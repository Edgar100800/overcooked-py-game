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

# Cache de modelos a nivel de MODULO: los arneses (rollout.py, el del profe) suelen
# instanciar un StudentAgent NUEVO por episodio; sin cache, cada episodio paga la
# carga del .zip (y el fallback por hash jamas llegaria a usar el modelo que su
# thread cargo). Clave = (path, mtime) para invalidar si best.zip se reemplaza.
_MODEL_CACHE: dict = {}
_LOADING: set = set()        # paths con un thread cargador en vuelo (dedup)


def _cache_key(path: Path):
    return (str(path), path.stat().st_mtime)


def _preload_all_enabled(models_dir: Path):
    """Carga al cache TODOS los modelos habilitados con terrain.key (sincronico).

    Se llama desde __init__ (fuera del SIGALRM de act(), igual que el preload
    clasico) cuando el config no trae layout: asi el fallback por hash encuentra
    el modelo YA cargado en el primer act() y el PPO juega desde el paso 1.
    Costo: ~3s la primera vez por proceso (import torch); luego sub-ms (cache).
    """
    for marker in sorted(models_dir.glob("*/enabled")):
        best = marker.parent / "best.zip"
        if best.exists() and (marker.parent / "terrain.key").exists():
            try:
                _load_ppo_cached(best)
            except Exception:
                pass


def _load_ppo_cached(path: Path):
    key = _cache_key(path)
    model = _MODEL_CACHE.get(key)
    if model is None:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        model = PPO.load(str(path), device="cpu")
        _MODEL_CACHE[key] = model     # los CNN son diminutos; cachear todos esta bien
    return model


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
        # Fallback por hash de terreno: si el arnes NO paso layout/layout_file en el
        # config, en el primer act() se busca un modelo habilitado cuyo terrain.key
        # coincida con el mdp observado, y se carga en un thread daemon (la carga
        # tarda >100ms; el SIGALRM del SafeActionWrapper solo corta el main thread).
        self.hash_fallback = bool(config.get("hash_fallback", True))
        self._hash_tried = False
        self._hash_path = None    # best.zip detectado por hash (el thread llena el cache)

        if not self.force_planner:
            self._try_preload(_layout_key(config))
            if self.model is None and self.hash_fallback and not self.explicit_model_path:
                # Sin layout en el config (o sin modelo para ese layout): dejar
                # listos en cache los habilitados para el fallback por hash.
                _preload_all_enabled(self.models_dir)

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
                self.model = _load_ppo_cached(best)
                self.model_layout = layout_key or "explicit"
            except Exception:
                self.model = None

    # ------------------------------------------------------------------- API
    def reset(self):
        self.planner.reset()
        self._fused = False
        self._partner_cooperative = False

    def act(self, obs) -> int:
        # Camino planner si: forzado o fusible disparado.
        if self.force_planner or self._fused:
            return self._planner_act(obs)

        # Sin modelo precargado: intentar el fallback por hash (no bloqueante) y
        # jugar con el planner hasta que el thread termine la carga (si hay match).
        if self.model is None:
            self._maybe_hash_load(obs)
            if self.model is None:
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
    def _maybe_hash_load(self, obs):
        """Fallback por hash de terreno (un solo intento por vida del agente).

        Busca models/<key>/ con marker `enabled` y `terrain.key` == hash del mdp
        observado; si hay match, carga el PPO en un thread daemon y mientras tanto
        el agente sigue con el planner (la sonda de cooperacion exige planner al
        inicio de todos modos). Cualquier excepcion -> queda en planner.
        """
        if self._hash_tried:
            # ¿El thread (de esta instancia o de una anterior) ya lleno el cache?
            if self.model is None and self._hash_path is not None:
                try:
                    self.model = _MODEL_CACHE.get(_cache_key(self._hash_path))
                except Exception:
                    pass
            return
        self._hash_tried = True
        if not self.hash_fallback or self.explicit_model_path:
            return
        try:
            tk = terrain_key(obs["mdp"])
            match = None
            for marker in self.models_dir.glob("*/enabled"):
                kf = marker.parent / "terrain.key"
                if (kf.exists() and kf.read_text().strip() == tk
                        and (marker.parent / "best.zip").exists()):
                    match = marker.parent
                    break
            if match is None:
                return
            self._hash_path = match / "best.zip"
            self.model_layout = match.name
            if _cache_key(self._hash_path) in _MODEL_CACHE:
                self.model = _MODEL_CACHE[_cache_key(self._hash_path)]
                return
            if str(self._hash_path) in _LOADING:
                return   # otra instancia (episodio previo) ya lo esta cargando
            _LOADING.add(str(self._hash_path))
            import threading

            def _load(path=self._hash_path):
                try:
                    _load_ppo_cached(path)
                except Exception:
                    pass
                finally:
                    _LOADING.discard(str(path))

            threading.Thread(target=_load, daemon=True).start()
        except Exception:
            pass

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
