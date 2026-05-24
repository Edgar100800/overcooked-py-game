"""Keyboard-controlled human policy for collecting demonstrations.

This policy is intentionally simple: each call reads the current keyboard state
and returns one action index using the public convention:

    0 north, 1 south, 2 east, 3 west, 4 stay, 5 interact.

It is meant to be used with rendering.mode: window.
"""

from __future__ import annotations


from typing import Any

from overcooked_ai_py.agents.agent import Agent

from src.constants import action_index_to_overcooked_action, action_name_to_index


DEFAULT_KEYMAP = {
    "up": ["up", "w"],
    "down": ["down", "s"],
    "left": ["left", "a"],
    "right": ["right", "d"],
    "interact": ["space", "e", "return"],
    "stay": [],
}

DEFAULT_PRIORITY = ["interact", "up", "down", "left", "right", "stay"]

KEY_ALIASES = {
    "esc": "escape",
    "enter": "return",
    "spacebar": "space",
    "arrow_up": "up",
    "arrow_down": "down",
    "arrow_left": "left",
    "arrow_right": "right",
}


class HumanKeyboardPolicy(Agent):
    """Agent controlled by the keyboard.

    YAML options:
        keymap:
          up: ["up", "w"]
          down: ["down", "s"]
          left: ["left", "a"]
          right: ["right", "d"]
          interact: ["space", "e"]
        priority: ["interact", "up", "down", "left", "right", "stay"]

    If multiple keys are pressed, the first action in `priority` wins.
    If no mapped key is pressed, the policy returns stay.
    """

    def __init__(self, keymap: dict[str, Any] | None = None, priority: list[str] | None = None):
        super().__init__()
        self.keymap = _merge_keymap(DEFAULT_KEYMAP, keymap or {})
        self.priority = [str(a).lower() for a in (priority or DEFAULT_PRIORITY)]
        self._pygame = None
        self._key_codes_by_action: dict[str, list[int]] | None = None
        self._action_by_key_code: dict[int, str] | None = None

    def action(self, state):
        pygame = self._ensure_pygame()
        pygame.event.pump()

        buffered_action = self._action_from_keydown_events(pygame)
        if buffered_action is not None:
            return self._overcooked_action(buffered_action, source="keyboard_event")

        pressed = pygame.key.get_pressed()

        for action_name in self.priority:
            if action_name == "stay":
                continue
            for key_code in self._key_codes_by_action.get(action_name, []):
                if pressed[key_code]:
                    return self._overcooked_action(action_name, source="keyboard_hold")

        return self._overcooked_action("stay", source="keyboard_idle")

    def _ensure_pygame(self):
        if self._pygame is not None:
            return self._pygame

        import pygame

        if not pygame.get_init():
            pygame.init()
        self._pygame = pygame
        self._key_codes_by_action = {
            action: [_key_name_to_code(pygame, key_name) for key_name in key_names]
            for action, key_names in self.keymap.items()
        }
        self._action_by_key_code = {}
        for action_name in self.priority:
            for key_code in self._key_codes_by_action.get(action_name, []):
                self._action_by_key_code[key_code] = action_name
        return pygame

    def _action_from_keydown_events(self, pygame) -> str | None:
        """Return the highest-priority action from queued KEYDOWN events.

        This catches quick taps that happen between simulation steps. Events not
        used by gameplay are posted back so the renderer can still see quit keys.
        """
        assert self._action_by_key_code is not None

        seen_actions: set[str] = set()
        for event in pygame.event.get([pygame.KEYDOWN]):
            action_name = self._action_by_key_code.get(event.key)
            if action_name is None:
                pygame.event.post(event)
                continue
            seen_actions.add(action_name)

        for action_name in self.priority:
            if action_name in seen_actions:
                return action_name
        return None

    def _overcooked_action(self, action_name: str, source: str):
        action_idx = action_name_to_index(action_name)
        return action_index_to_overcooked_action(action_idx), {
            "policy_name": "human_keyboard",
            "action_index": action_idx,
            "source": source,
        }


def _merge_keymap(default: dict[str, list[str]], override: dict[str, Any]) -> dict[str, list[str]]:
    merged = {key: list(value) for key, value in default.items()}
    for action, value in override.items():
        action_key = str(action).lower()
        if value is None:
            merged[action_key] = []
        elif isinstance(value, str):
            merged[action_key] = [value]
        else:
            merged[action_key] = [str(v) for v in value]
    return merged


def _key_name_to_code(pygame, key_name: str) -> int:
    key = str(key_name).strip().lower().replace(" ", "_")
    key = KEY_ALIASES.get(key, key)

    # Common explicit aliases first, because pygame.key.key_code is strict about names.
    explicit = {
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "space": pygame.K_SPACE,
        "return": pygame.K_RETURN,
        "escape": pygame.K_ESCAPE,
        "tab": pygame.K_TAB,
        "shift": pygame.K_LSHIFT,
        "ctrl": pygame.K_LCTRL,
    }
    if key in explicit:
        return explicit[key]

    try:
        return pygame.key.key_code(key)
    except Exception as exc:
        raise ValueError(f"Unknown pygame key name: {key_name!r}") from exc
