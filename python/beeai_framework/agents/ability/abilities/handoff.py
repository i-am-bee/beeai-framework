from functools import cached_property

from pydantic import BaseModel, Field

from beeai_framework.agents.ability.abilities.ability import Ability, AbilityCheckResult, with_run_context
from beeai_framework.agents.ability.agent import AbilityAgent
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.context import RunContext
from beeai_framework.memory import BaseMemory


class HandoffSchema(BaseModel):
    prompt: str = Field(description="Clearly defined task for the agent to work on based on his abilities.")


class HandoffAbility(Ability[HandoffSchema]):
    """Delegates a task to an expert agent"""

    def __init__(self, target: "AbilityAgent", *, name: str | None = None, description: str | None = None) -> None:
        self._target = target

        self.name = name or target.meta.name
        self.description = description or target.meta.description
        # self.description += "(context must contain only verified information)"

        super().__init__()

    @cached_property
    def input_schema(self) -> type[HandoffSchema]:
        return HandoffSchema

    def check(self, states: AbilityAgentRunState) -> AbilityCheckResult:
        return AbilityCheckResult(allowed=True, forced=False, hidden=False, prevent_stop=False)

    @with_run_context
    async def run(self, obj: HandoffSchema, context: RunContext) -> str:
        memory: BaseMemory = context.context["state"]["memory"]

        target = await self._target.clone()
        await target.memory.add_many(memory.messages[1:])
        response = await target.run(prompt=obj.prompt)
        return response.result.text
