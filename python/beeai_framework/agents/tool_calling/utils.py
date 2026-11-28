# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from pydantic import BaseModel

from beeai_framework.backend import MessageToolCallContent


class ToolCallCheckerConfig(BaseModel):
    max_strike_length: int = 1
    max_total_occurrences: int = 5
    window_size: int = 10


class ToolCallChecker:
    def __init__(self, config: ToolCallCheckerConfig) -> None:
        self._config = config
        self._strike_history: list[MessageToolCallContent] = []
        self._occurrences_history: deque[MessageToolCallContent] = deque(
            maxlen=max(config.max_total_occurrences + 1, config.window_size)
        )
        self.cycle_found = False
        self.enabled = True

    def register(self, value: MessageToolCallContent) -> None:
        if not self.enabled:
            return

        # A strike is a run of identical consecutive calls. A different call in between ends it.
        if self._strike_history and _is_same_tool_call(self._strike_history[-1], value):
            self._strike_history.append(value)
            if len(self._strike_history) > self._config.max_strike_length:
                self.cycle_found = True
        else:
            self._strike_history = [value]

        # Count the identical calls among the most recent calls (a sliding window).
        self._occurrences_history.append(value)
        occurrences = sum(1 for call in self._occurrences_history if _is_same_tool_call(call, value))
        if occurrences > self._config.max_total_occurrences:
            self.cycle_found = True

    def reset(self, current: MessageToolCallContent | None = None) -> None:
        self._strike_history = [current] if current else []
        self._occurrences_history.clear()
        if current:
            self._occurrences_history.append(current)
        self.cycle_found = False

    @property
    def config(self) -> ToolCallCheckerConfig:
        return self._config


def _is_same_tool_call(a: MessageToolCallContent | None, b: MessageToolCallContent | None) -> bool:
    return bool(a and b and a.tool_name == b.tool_name and a.args == b.args and a.type == b.type)
