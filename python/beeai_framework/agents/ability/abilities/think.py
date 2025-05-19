from pydantic import BaseModel, Field

from beeai_framework.agents.ability.abilities.ability import Ability, AbilityCheckResult, with_run_context
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.context import RunContext


class ReasoningSchema(BaseModel):
    thoughts: str = Field(..., description="Precisely describe what you are thinking about.")
    next_step: list[str] = Field(..., description="Describe tool you would need to use next and why.", min_length=1)


class ThinkAbility(Ability[ReasoningSchema]):
    name = "think"
    description = "Use when you want to think through a problem, clarify your assumptions, or break down complex steps before acting or responding."  # noqa: E501

    def __init__(self, *, force: bool = False, extra_instructions: str = "") -> None:
        super().__init__()
        self.force = force
        self.priority = 5
        if extra_instructions:
            self.description += f" {extra_instructions}"

    @property
    def input_schema(self) -> type[ReasoningSchema]:
        return ReasoningSchema

    def check(self, states: AbilityAgentRunState) -> AbilityCheckResult:
        last_step = states.steps[-1] if states.steps else None
        return AbilityCheckResult(
            allowed=True,
            forced=last_step.ability.name != self.name and last_step.error is not None
            if last_step and last_step.ability
            else self.force,
            hidden=False,
            prevent_stop=False,
        )

    @with_run_context
    async def run(self, input: ReasoningSchema, context: RunContext) -> str:
        return "Saved"  # TODO: refactor
