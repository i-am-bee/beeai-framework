import contextlib
from collections.abc import Sequence
from typing import Any, Generic, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypeVar

from beeai_framework.agents.ability.abilities.ability import Ability, AnyAbility
from beeai_framework.agents.ability.abilities.final_answer import FinalAnswerAbility
from beeai_framework.agents.ability.prompts import AbilityPromptTemplateDefinition
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.backend.types import ChatModelToolChoice
from beeai_framework.context import RunContext
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.tool import tool as create_tool
from beeai_framework.utils.lists import remove_by_reference
from beeai_framework.utils.strings import to_json

TAbility = TypeVar("TAbility", bound=Ability[Any, Any, Any], default=AnyAbility)


class AbilityModelAdapter(Generic[TAbility]):
    def __init__(self, ability: TAbility, tool: AnyTool | None = None) -> None:
        self._ability = ability
        self._tool = tool

    @property
    def name(self) -> str:
        return self._ability.name

    @property
    def ability(self) -> TAbility:
        return self._ability

    @property
    def tool(self) -> AnyTool:
        if self._tool is None:

            @create_tool(
                name=self.ability.name,
                description=self.ability.description,
                input_schema=self.ability.input_schema,
                with_context=True,
                # emitter=emitter,  # TODO: to be updated
            )
            async def tool(context: RunContext, **kwargs: Any) -> Any:
                input = self.ability.input_schema.model_validate(kwargs)
                return await self.ability.run(input)

            self._tool = tool

        return self._tool

    @property
    def describe(self) -> AbilityPromptTemplateDefinition:
        ability = self._ability

        return AbilityPromptTemplateDefinition(
            name=ability.name,
            description=ability.description,
            enabled=str(ability.enabled),  # TODO: bool vs str
            input_schema=to_json(
                ability.input_schema.model_json_schema(mode="serialization"), indent=2, sort_keys=False
            ),
        )


class AbilityModel:
    def __init__(
        self,
        *,
        abilities: Sequence[AnyAbility],
        final_answer: FinalAnswerAbility,  # TODO: refactor
    ) -> None:
        self._entries: list[AbilityModelAdapter] = []
        self.final_answer = AbilityModelAdapter(final_answer)
        self.update(abilities)

    def update(self, abilities: Sequence[AnyAbility]) -> None:
        self._entries.clear()

        for ability in abilities:
            self._entries.append(AbilityModelAdapter(ability))
        self._entries.append(self.final_answer)

        self._verify()

    def _verify(self) -> None:
        all_abilities = [entry.ability for entry in self._entries]

        for ability in all_abilities:
            ability.verify(abilities=all_abilities)

    def create_request(self, state: AbilityAgentRunState, *, force_tool_call: bool) -> "AbilityAgentRequest":
        tool_choice: Literal["required"] | AnyTool = "required"

        hidden: list[AbilityModelAdapter] = []
        allowed: list[AbilityModelAdapter] = []

        prevent_stop: bool = False
        forced: AbilityModelAdapter | None = None

        for entry in self._entries:
            status = entry.ability.can_use(state=state)
            if status.forced:
                allowed.clear()
                # TODO: we need to update the selection of tools instead of removing them

            if status.hidden:
                hidden.append(entry)

            if status.prevent_stop:
                prevent_stop = True

            if not status.allowed:
                continue

            allowed.append(entry)
            if status.forced and (forced is None or entry.ability.priority > forced.ability.priority):
                forced = entry
                tool_choice = entry.tool

        if prevent_stop:
            with contextlib.suppress(ValueError):
                remove_by_reference(allowed, self.final_answer)

        if not allowed:
            raise FrameworkError("Unknown state. Tools shouldn't not be empty.")

        if len(allowed) == 1:
            tool_choice = allowed[0].tool

        _assert_uniq_abilities(allowed)

        return AbilityAgentRequest(
            allowed=allowed,
            tool_choice=tool_choice if isinstance(tool_choice, Tool) or force_tool_call or prevent_stop else "auto",
            final_answer=self.final_answer,
            hidden=hidden,
            can_stop=not prevent_stop,
        )


def _assert_uniq_abilities(entries: list[AbilityModelAdapter]) -> None:
    seen = set()
    for entry in entries:
        if entry.name in seen:
            raise ValueError(f"Duplicate ability name '{entry.name}'!")
        seen.add(entry.name)


class AbilityAgentRequest(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    allowed: list[AbilityModelAdapter]
    hidden: list[AbilityModelAdapter]
    tool_choice: ChatModelToolChoice
    final_answer: AbilityModelAdapter[FinalAnswerAbility]
    can_stop: bool
