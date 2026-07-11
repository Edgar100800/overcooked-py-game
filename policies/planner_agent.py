"""Planner robusto sin aprendizaje (FASE 1 de PLAN.md).

`PlannerAgent` es compatible con el loader `python_class` (contrato StudentAgent):
`__init__(config: dict)`, `reset()`, `act(obs) -> int` (0..5). Requiere
`observation.type: state`, es decir `obs == {"state", "mdp", "agent_index"}`.

Idea (PLAN §4):
  - Todo lo estatico del layout se precomputa en `reset()`/primer `act()` desde el
    `mdp` (posiciones de estaciones, tiles transitables). `act()` hace un BFS ligero
    que si tiene en cuenta al companero (posicion movil) -> objetivo < 5 ms.
  - FSM de sub-tareas que completa el ciclo COMPLETO en solitario (sirve para el
    escenario 4 con companero `random_motion`/`stay`): sopa->servir; plato->pot
    listo; ingrediente->pot no lleno; vacio+listo->plato; vacio->onion.
  - Modelado ligero del companero: se infiere su sub-tarea (objeto en mano) para
    tomar la complementaria y no pelear por la misma estacion; el companero se trata
    como obstaculo en la navegacion (avoid_teammate).
  - Anti-deadlock (§4.5): N pasos sin moverse -> desvio lateral aleatorio 1-2 pasos.
  - Independiente del indice: usa `obs["agent_index"]`, nunca 0 hardcodeado.

Reutiliza la logica probada de `policies/basic_policies.py:GreedyFullTaskPolicy`
(get_pot_states, get_*_locations, BFS con Direction/Action), reimplementada aqui de
forma autonoma porque el planner opera sobre la observacion `state` (no como Agent
con `self.mdp` inyectado).
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

from src.constants import overcooked_action_to_index

STAY_IDX = 4
INTERACT_IDX = 5


class PlannerAgent:
    """Agente-planner determinista y defensivo. Nunca debe crashear ni exceder 100 ms."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.ingredient = config.get("ingredient", "onion")
        if self.ingredient not in {"onion", "tomato"}:
            self.ingredient = "onion"
        self.avoid_teammate = bool(config.get("avoid_teammate", True))
        self.stuck_threshold = int(config.get("stuck_threshold", 3))
        self.detour_len = int(config.get("detour_len", 2))
        self.rng = np.random.default_rng(config.get("seed"))
        self._reset_state()

    # ------------------------------------------------------------------ API
    def reset(self):
        self._reset_state()

    def act(self, obs) -> int:
        state = obs["state"]
        mdp = obs["mdp"]
        agent_index = int(obs.get("agent_index", 0))

        if mdp is not self._mdp:
            self._precompute(mdp)

        self._navigating = False
        try:
            action_idx = self._decide(state, mdp, agent_index)
        except Exception:
            action_idx = STAY_IDX

        self._update_deadlock_tracking(state, agent_index, action_idx)
        return int(action_idx)

    # ---------------------------------------------------------------- estado
    def _reset_state(self):
        self._mdp = None
        self._valid_positions: set[tuple[int, int]] = set()
        self._pot_locations: list[tuple[int, int]] = []
        self._serving: list[tuple[int, int]] = []
        self._dish_disp: list[tuple[int, int]] = []
        self._onion_disp: list[tuple[int, int]] = []
        self._tomato_disp: list[tuple[int, int]] = []
        self._prev_pos: tuple[int, int] | None = None
        self._stuck_counter = 0
        self._detour_steps_left = 0
        self._navigating = False

    def _precompute(self, mdp):
        """Cachea todo lo estatico del layout (solo depende del mdp)."""
        self._mdp = mdp
        self._valid_positions = set(mdp.get_valid_player_positions())
        self._pot_locations = list(mdp.get_pot_locations())
        self._serving = list(mdp.get_serving_locations())
        self._dish_disp = list(mdp.get_dish_dispenser_locations())
        self._onion_disp = list(mdp.get_onion_dispenser_locations())
        try:
            self._tomato_disp = list(mdp.get_tomato_dispenser_locations())
        except Exception:
            self._tomato_disp = []

    def _ingredient_dispensers(self) -> list[tuple[int, int]]:
        return self._onion_disp if self.ingredient == "onion" else self._tomato_disp

    # ------------------------------------------------------------- decision
    def _decide(self, state, mdp, agent_index) -> int:
        player = state.players[agent_index]
        pos = player.position

        # Anti-deadlock: si venimos de un desvio o estamos atascados, mover lateral.
        if self._detour_steps_left > 0:
            self._detour_steps_left -= 1
            detour = self._detour_move(state, agent_index)
            if detour is not None:
                return detour
        if self._stuck_counter >= self.stuck_threshold:
            self._stuck_counter = 0
            self._detour_steps_left = self.detour_len
            detour = self._detour_move(state, agent_index)
            if detour is not None:
                return detour

        target = self._choose_target(state, mdp, agent_index)
        if target is None:
            return STAY_IDX
        player = state.players[agent_index]
        # Si ya estamos adyacentes al objetivo estamos interactuando (no navegando):
        # encarar/interactuar NO cuenta como atasco aunque la posicion no cambie.
        self._navigating = not self._is_adjacent(player.position, target)
        return self._move_or_interact_towards(state, agent_index, target)

    def _choose_target(self, state, mdp, agent_index) -> tuple[int, int] | None:
        """FSM de sub-tareas. Devuelve la celda-estacion objetivo (no transitable)."""
        player = state.players[agent_index]
        held = player.held_object
        pot_states = mdp.get_pot_states(state)

        partner = state.players[1 - agent_index] if len(state.players) > 1 else None
        partner_held = partner.held_object.name if (partner and partner.held_object) else None

        # --- Con objeto en mano: hay una unica accion correcta por objeto. ---
        if held is not None:
            if held.name == "soup":
                return self._nearest(player.position, self._serving)
            if held.name == "dish":
                ready = list(pot_states.get("ready", []))
                if ready:
                    return self._nearest(player.position, ready)
                # esperar cerca de un pot cocinando / lleno
                almost = list(pot_states.get("cooking", [])) + list(
                    pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", [])
                )
                return self._nearest(player.position, almost) if almost else None
            if held.name in {"onion", "tomato"}:
                return self._nearest(player.position, self._pots_accepting(pot_states))
            return None

        # --- Vacio: elegir rol, teniendo en cuenta al companero (complementariedad). ---
        ready = list(pot_states.get("ready", []))
        pots_need = self._pots_accepting(pot_states)

        # Si el companero ya lleva un plato y hay sopa lista, el la sirve:
        # yo empiezo la siguiente tanda (traer onion) en vez de pelear por el plato.
        if ready and partner_held != "dish":
            counter_dishes = self._counter_objects(state, "dish")
            if counter_dishes:
                return self._nearest(player.position, counter_dishes)
            if self._dish_disp:
                return self._nearest(player.position, self._dish_disp)

        # Si hay pots que aceptan ingredientes y el companero no esta trayendo el mismo
        # ingrediente, voy por el ingrediente.
        if pots_need and partner_held not in {"onion", "tomato"}:
            counter_ing = self._counter_objects(state, self.ingredient)
            if counter_ing:
                return self._nearest(player.position, counter_ing)
            disp = self._ingredient_dispensers()
            if disp:
                return self._nearest(player.position, disp)

        # Fallbacks (aseguran progreso en solitario aunque el companero haga lo mismo):
        if ready and self._dish_disp:
            return self._nearest(player.position, self._dish_disp)
        if pots_need and self._ingredient_dispensers():
            return self._nearest(player.position, self._ingredient_dispensers())

        # pot lleno sin cocinar (dinamicas nuevas): ir a interactuar para iniciarlo.
        full = list(pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", []))
        if full:
            return self._nearest(player.position, full)
        if list(pot_states.get("cooking", [])) and self._dish_disp:
            return self._nearest(player.position, self._dish_disp)
        return None

    def _pots_accepting(self, pot_states) -> list[tuple[int, int]]:
        out = list(pot_states.get("empty", []))
        for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
            out.extend(list(pot_states.get(f"{k}_items", [])))
        return out

    def _counter_objects(self, state, name: str) -> list[tuple[int, int]]:
        return [o.position for o in state.objects.values() if o.name == name]

    # ----------------------------------------------------------- navegacion
    def _move_or_interact_towards(self, state, agent_index, target) -> int:
        player = state.players[agent_index]
        pos = player.position
        orientation = player.orientation

        if self._is_adjacent(pos, target):
            desired = self._direction_from_to(pos, target)
            if orientation == desired:
                return INTERACT_IDX
            return overcooked_action_to_index(desired)

        next_pos = self._next_step(state, agent_index, target)
        if next_pos is None:
            # Sin ruta (companero bloquea): intentar un desvio lateral.
            detour = self._detour_move(state, agent_index)
            return detour if detour is not None else STAY_IDX
        action = Action.determine_action_for_change_in_pos(pos, next_pos)
        return overcooked_action_to_index(action)

    def _next_step(self, state, agent_index, target) -> tuple[int, int] | None:
        start = state.players[agent_index].position
        teammate = set()
        for idx, other in enumerate(state.players):
            if idx != agent_index:
                teammate.add(other.position)
        blocked = set(teammate) if self.avoid_teammate else set()

        adj = [p for p in self._adjacent(target) if p in self._valid_positions]
        goals_free = [p for p in adj if p not in blocked]

        # 1) Ruta que evita al companero. 2) Fallback: ruta ignorando al companero,
        #    para al menos ACERCARSE y esperar adyacente (util con companero movil:
        #    cuando se aparta, entramos). Con companero fijo en la unica celda util
        #    esto no entrega, pero es el mejor esfuerzo (§12-A.5).
        for goals, blk in ((goals_free, blocked), (adj, set())):
            if not goals:
                continue
            path = self._bfs(start, set(goals), blk)
            if path is not None and len(path) >= 2:
                nxt = path[1]
                # No pisar al companero: si el siguiente paso esta ocupado, esperar.
                if nxt in teammate:
                    return None
                return nxt
        return None

    def _bfs(self, start, goals, blocked) -> list[tuple[int, int]] | None:
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            pos, path = queue.popleft()
            if pos in goals:
                return path
            for d in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(pos, d)
                if nxt not in self._valid_positions:
                    continue
                if nxt in blocked and nxt not in goals:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    def _detour_move(self, state, agent_index) -> int | None:
        """Movimiento lateral aleatorio para romper deadlocks (§4.5)."""
        pos = state.players[agent_index].position
        blocked = set()
        for idx, other in enumerate(state.players):
            if idx != agent_index:
                blocked.add(other.position)
        candidates = []
        for d in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(pos, d)
            if nxt in self._valid_positions and nxt not in blocked:
                candidates.append(d)
        if not candidates:
            return None
        d = candidates[int(self.rng.integers(0, len(candidates)))]
        return overcooked_action_to_index(d)

    def _update_deadlock_tracking(self, state, agent_index, action_idx):
        pos = state.players[agent_index].position
        # Solo cuenta como atasco si estabamos NAVEGANDO (no interactuando/encarando)
        # y la posicion no cambio: el movimiento fue bloqueado de verdad.
        if self._navigating and self._prev_pos is not None and pos == self._prev_pos:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._prev_pos = pos

    # --------------------------------------------------------------- geometria
    @staticmethod
    def _nearest(origin, positions: Iterable[tuple[int, int]]):
        positions = list(positions)
        if not positions:
            return None
        return min(positions, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))

    @staticmethod
    def _is_adjacent(a, b) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @staticmethod
    def _adjacent(pos) -> list[tuple[int, int]]:
        return [Action.move_in_direction(pos, d) for d in Direction.ALL_DIRECTIONS]

    @staticmethod
    def _direction_from_to(a, b):
        direction = (b[0] - a[0], b[1] - a[1])
        if direction not in Direction.ALL_DIRECTIONS:
            raise ValueError(f"no adyacentes: {a} -> {b}")
        return direction


# Alias para que el loader pueda cargarlo como StudentAgent si se desea usar el
# planner puro como entrega (class_name: StudentAgent en el YAML).
StudentAgent = PlannerAgent
