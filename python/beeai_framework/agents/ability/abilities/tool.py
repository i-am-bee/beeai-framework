from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from beeai_framework.agents.ability.abilities.ability import Ability, AbilityCheckResult, with_run_context
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.context import RunContext
from beeai_framework.tools import Tool, ToolOutput

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=ToolOutput)


class ToolAbility(Generic[TInput, TOutput], Ability[TInput]):
    def __init__(
        self, tool: Tool[TInput, Any, TOutput], /, name: str | None = None, description: str | None = None
    ) -> None:
        super().__init__()
        self._tool = tool
        self.name = name or tool.name
        self.description = description or tool.description
        self.priority -= 1  # TODO: refactor

    @property
    def tool(self) -> Tool[TInput, Any, TOutput]:
        return self._tool

    @property
    def input_schema(self) -> type[TInput]:
        return self._tool.input_schema

    @with_run_context
    async def run(self, input: TInput, context: RunContext) -> TOutput:
        response = await self._tool.run(input)
        return response  # TODO: refactor

    def check(self, state: AbilityAgentRunState) -> AbilityCheckResult:
        return AbilityCheckResult(allowed=True, prevent_stop=False, forced=False, hidden=False)
