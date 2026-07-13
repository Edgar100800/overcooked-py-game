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

from collections import Counter, deque
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
        # ventana del detector de oscilacion (pasos navegando en <=2 celdas unicas)
        self.osc_window = int(config.get("osc_window", 12))
        # Semilla FIJA por defecto (0): el unico componente estocastico es el desvio
        # anti-deadlock; fijarla hace al planner DETERMINISTA y reproducible (PLAN §14)
        # y garantiza que el StudentAgent en modo planner sea identico al planner puro
        # (necesario para G8: "el selector nunca empeora").
        self.rng = np.random.default_rng(config.get("seed", 0))
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
        self._orders: list[tuple[str, ...]] = []
        self._recipe_mode = False
        self._disjoint_kitchens = False
        self._prev_pos: tuple[int, int] | None = None
        self._pos_history: list[tuple[int, int]] = []
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
        self._precompute_orders(mdp)
        self._detect_disjoint_kitchens(mdp)

    def _detect_disjoint_kitchens(self, mdp):
        """Cocinas disjuntas (p.ej. asymmetric_advantages): cada jugador esta en una
        region transitable separada con TODAS las estaciones a su alcance. Ahi el
        companero jamas puede bloquearme ni competir por mis estaciones -> conviene
        el ciclo completo independiente (se ignoran las puertas de complementariedad
        en _choose_target). La autosuficiencia de AMBAS regiones es obligatoria:
        forced_coordination tambien es disjunto pero un jugador no tiene ollas ->
        alli la complementariedad es lo unico que funciona y el flag queda False."""
        self._disjoint_kitchens = False
        try:
            starts = list(getattr(mdp, "start_player_positions", None) or [])
            if len(starts) < 2:
                starts = [p.position for p in mdp.get_standard_start_state().players]
            if len(starts) < 2 or any(tuple(s) not in self._valid_positions for s in starts[:2]):
                return
            region_a = self._flood(tuple(starts[0]))
            if tuple(starts[1]) in region_a:
                return  # conectadas -> comportamiento historico
            region_b = self._flood(tuple(starts[1]))
            stations = [
                self._onion_disp + self._tomato_disp,
                self._dish_disp,
                self._pot_locations,
                self._serving,
            ]
            for region in (region_a, region_b):
                for group in stations:
                    if not any(a in region for s in group for a in self._adjacent(s)):
                        return  # region no autosuficiente
            self._disjoint_kitchens = True
        except Exception:
            self._disjoint_kitchens = False

    def _flood(self, start: tuple[int, int]) -> set[tuple[int, int]]:
        """Flood-fill determinista sobre celdas transitables (no consume self.rng)."""
        seen = {start}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            for d in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(pos, d)
                if nxt in self._valid_positions and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def _precompute_orders(self, mdp):
        """Modo receta: si NINGUNA orden del layout es la sopa mono-ingrediente
        clasica (p.ej. 3 cebollas), rellenar ollas a ciegas entrega sopas invalidas
        (valor 0). En ese caso el planner apunta a la orden valida mas rapida y
        rellena cada olla con los ingredientes que le FALTAN para completarla.
        Si la orden clasica existe (todos los layouts G3/G8), _recipe_mode queda
        False y el comportamiento es EXACTAMENTE el histórico (no regresiona G8)."""
        orders: list[tuple[str, ...]] = []
        try:
            for r in getattr(mdp, "start_all_orders", None) or []:
                ing = r["ingredients"] if isinstance(r, dict) else list(r.ingredients)
                orders.append(tuple(sorted(ing)))
        except Exception:
            orders = []
        mono = tuple([self.ingredient] * Recipe.MAX_NUM_INGREDIENTS)
        self._recipe_mode = bool(orders) and mono not in orders

        def cook_time(o: tuple[str, ...]) -> int:
            try:
                return int(Recipe(o).time)
            except Exception:
                return 20 * len(o)

        # menos ingredientes y menor coccion primero: maximiza sopas/episodio
        self._orders = sorted(set(orders), key=lambda o: (len(o), cook_time(o)))

    def _best_order(self, contents: tuple[str, ...]) -> tuple[str, ...] | None:
        """La orden valida mas rapida que CONTIENE lo que ya hay en la olla."""
        c = Counter(contents)
        for order in self._orders:
            oc = Counter(order)
            if all(oc[k] >= v for k, v in c.items()):
                return order
        return None

    def _nearest_empty_counter(self, state, mdp, origin):
        try:
            return self._nearest(origin, mdp.get_empty_counter_locations(state))
        except Exception:
            return None

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

        if self._disjoint_kitchens and partner_held == "dish" and \
                len(pot_states.get("ready", [])) > 1:
            # cocinas disjuntas y MAS de una olla lista: aunque el companero ya
            # lleve plato solo puede servir una; diferir mi plato dejaria la otra
            # sopa bloqueando su olla un viaje entero. Cada uno tiene su propio
            # dispensador de platos, asi que ir por el mio nunca compite.
            partner_held = None

        if self._recipe_mode:
            return self._choose_target_recipe(state, mdp, player, partner_held, pot_states)

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

    def _choose_target_recipe(self, state, mdp, player, partner_held, pot_states):
        """FSM para layouts con recetas multi-ingrediente (p.ej. counter_circuit:
        onion+tomato). Igual que la FSM clasica pero por-olla: calcula que le falta
        a cada olla para completar la orden valida mas rapida, inicia la coccion en
        cuanto la receta esta completa, y cocina/entrega (a valor 0) las ollas
        irreparables que el companero envenenó para liberarlas."""
        held = player.held_object
        pos = player.position

        # Clasificar ollas no-cocinando: rellenables (que falta) vs listas-a-iniciar.
        fillable: dict[tuple[int, int], Counter] = {}
        startable: list[tuple[int, int]] = []
        for ppos in self._pot_locations:
            if not state.has_object(ppos):
                best = self._best_order(())
                if best:
                    fillable[ppos] = Counter(best)
                continue
            soup = state.get_object(ppos)
            if soup.is_ready or soup.is_cooking:
                continue
            contents = tuple(sorted(soup.ingredients))
            best = self._best_order(contents)
            if best is None:
                # irreparable (p.ej. 3 cebollas sin orden que las acepte):
                # cocinarla y entregarla a valor 0 es la unica forma de liberarla.
                startable.append(ppos)
                continue
            missing = Counter(best) - Counter(contents)
            if missing:
                fillable[ppos] = missing
            else:
                startable.append(ppos)

        ready = list(pot_states.get("ready", []))

        # --- Con objeto en mano ---
        if held is not None:
            if held.name == "soup":
                return self._nearest(pos, self._serving)
            if held.name == "dish":
                if ready:
                    return self._nearest(pos, ready)
                almost = list(pot_states.get("cooking", [])) + startable
                return self._nearest(pos, almost) if almost else None
            if held.name in {"onion", "tomato"}:
                useful = [p for p, miss in fillable.items() if miss.get(held.name, 0) > 0]
                if useful:
                    return self._nearest(pos, useful)
                # ninguna olla lo necesita: soltarlo en un counter para no bloquearse
                return self._nearest_empty_counter(state, mdp, pos)
            return None

        # --- Vacio ---
        if ready and partner_held != "dish":
            counter_dishes = self._counter_objects(state, "dish")
            if counter_dishes:
                return self._nearest(pos, counter_dishes)
            if self._dish_disp:
                return self._nearest(pos, self._dish_disp)

        # Iniciar coccion cuanto antes (corre en paralelo mientras hago otra cosa).
        if startable:
            return self._nearest(pos, startable)

        if fillable:
            # olla mas cerca de completarse; desempate por distancia
            target_pot = min(
                fillable,
                key=lambda p: (
                    sum(fillable[p].values()),
                    abs(p[0] - pos[0]) + abs(p[1] - pos[1]),
                ),
            )
            needs = sorted(fillable[target_pot])
            # complementariedad: si el companero ya trae uno de los faltantes y hay
            # alternativa, yo voy por el otro ingrediente.
            if partner_held in needs and len(needs) > 1:
                needs = [n for n in needs if n != partner_held] + [partner_held]
            for name in needs:
                sources = self._counter_objects(state, name)
                sources += self._onion_disp if name == "onion" else self._tomato_disp
                if sources:
                    return self._nearest(pos, sources)

        if ready and self._dish_disp:
            return self._nearest(pos, self._dish_disp)
        if list(pot_states.get("cooking", [])) and self._dish_disp:
            return self._nearest(pos, self._dish_disp)
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
        # Oscilacion (baile A<->B): dos agentes replanificando en espejo se mueven
        # cada paso (el contador de arriba nunca dispara) pero no avanzan. Solo
        # cuentan pasos NAVEGANDO consecutivos (interactuar/encarar es quedarse en la
        # misma celda legitimamente y resetea la ventana). 8 pasos navegando dentro
        # de <=2 celdas unicas = atasco real.
        if self._navigating:
            w = self.osc_window
            self._pos_history.append(pos)
            if len(self._pos_history) > w:
                self._pos_history.pop(0)
            if len(self._pos_history) == w and len(set(self._pos_history)) <= 2:
                self._stuck_counter = self.stuck_threshold   # fuerza detour
                self._pos_history.clear()
        else:
            self._pos_history.clear()

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
