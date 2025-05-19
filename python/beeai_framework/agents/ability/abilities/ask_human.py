from typing import Any

from pydantic import BaseModel, Field

from beeai_framework.agents.ability.abilities.ability import Ability, AbilityCheckResult, with_run_context
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.context import RunContext


class HumanInTheLoopSchema(BaseModel):
    request: str = Field(
        ..., description="Clearly describe what you would like to know from the user to resolve the task."
    )


class HumanInTheLoopAbility(Ability[HumanInTheLoopSchema]):
    name = "ask_human"
    description = "Use to ask the user for a clarification"

    def __init__(self) -> None:
        super().__init__()
        self.priority += 1
        # TODO: OOTB CLI reader?

    @with_run_context
    async def run(self, input: HumanInTheLoopSchema, context: RunContext) -> Any:
        raise NotImplementedError("This ability is not yet implemented.")

    def check(self, state: AbilityAgentRunState) -> AbilityCheckResult:
        return AbilityCheckResult(allowed=True)

    @property
    def input_schema(self) -> type[HumanInTheLoopSchema]:
        return HumanInTheLoopSchema
