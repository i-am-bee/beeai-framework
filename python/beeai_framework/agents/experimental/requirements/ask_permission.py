# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from beeai_framework.agents.experimental.requirements._utils import (
    MultiTargetType,
    _assert_all_rules_found,
    _extract_targets,
    _target_seen_in,
)
from beeai_framework.agents.experimental.requirements.requirement import (
    Requirement,
    Rule,
    run_with_context,
)
from beeai_framework.agents.experimental.types import RequirementAgentRunState
from beeai_framework.agents.experimental.utils._tool import FinalAnswerTool
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.tools import AnyTool, StringToolOutput
from beeai_framework.utils import MaybeAsync
from beeai_framework.utils.asynchronous import ensure_async

AskHandler = MaybeAsync[[AnyTool, dict[str, Any]], bool]


class AskPermissionRequirement(Requirement[RequirementAgentRunState]):
    name = "ask_permission"
    description = "Use to ask the user for a clarification"

    def __init__(
        self,
        include: MultiTargetType | None = None,
        *,
        handler: AskHandler | None = None,
        exclude: MultiTargetType | None = None,
        remember_choices: bool = False,
        hide_disallowed: bool = False,
        always_allow: bool = False,
    ) -> None:
        super().__init__()
        self.priority += 1
        self._include = _extract_targets(include)

        self._exclude = _extract_targets(exclude)
        self._exclude.add(FinalAnswerTool)

        self._state = dict[str, bool]()
        self._remember_choices = remember_choices
        self._hide_disallowed = hide_disallowed
        self._always_allow = always_allow
        self._handler = ensure_async(
            handler
            if handler
            else (
                lambda tool, tool_input: input(
                    f"The agent wants to use the '{tool.name} tool.'\nInput: {tool_input}\nDo you allow it? (yes/no): "
                )
                .strip()
                .startswith("yes")
            )
        )

    def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
        _assert_all_rules_found(self._include, tools)
        _assert_all_rules_found(self._exclude, tools)

        def setup_tool(tool: AnyTool) -> None:
            async def handler(data: Any, _: EventMeta) -> None:
                await self._ask(tool, data)

            ctx.emitter.match(
                create_internal_event_matcher("start", tool, parent_run_id=ctx.run_id),
                handler,
                EmitterOptions(is_blocking=True, persistent=True, match_nested=True),
            )

        for tool in tools:
            if _target_seen_in(tool, self._exclude):
                continue

            if not self._include or _target_seen_in(tool, self._include):
                setup_tool(tool)

    async def _ask(self, tool: AnyTool, data: RunContextStartEvent) -> None:
        allowed: bool | None = True if self._always_allow else self._state.get(tool.name)
        if allowed is None:
            allowed = await self._handler(tool, data.input)
            if self._remember_choices:
                self._state[tool.name] = allowed

        if not allowed:
            data.output = StringToolOutput("This tool is not allowed to be used.")

    @run_with_context
    async def run(self, input: RequirementAgentRunState, context: RunContext) -> list[Rule]:
        return [
            Rule(
                target=target,
                allowed=state,
                prevent_stop=False,
                hidden=not state and self._hide_disallowed,
                forced=False,
            )
            for target, state in self._state.items()
        ]
