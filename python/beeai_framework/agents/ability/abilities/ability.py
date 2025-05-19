from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.strings import to_safe_word

if TYPE_CHECKING:
    from beeai_framework.agents.ability.types import AbilityAgentRunState

TAbilityInput = TypeVar("TAbilityInput", bound=BaseModel, default=BaseModel)
TAbilityCheckInput = TypeVar("TAbilityCheckInput", bound=BaseModel, default="AbilityAgentRunState")
TAbilityOutput = TypeVar("TAbilityOutput", bound=Any, default=Any)
AgentAbilityFactory = Callable[[], "Ability[Any]"]


class AbilityCheckResult(BaseModel):
    allowed: bool = Field(True, description="Can the agent use the tool?")
    prevent_stop: bool = Field(False, description="Prevent the agent from terminating.")
    forced: bool = Field(False, description="Must the agent use the tool?")
    hidden: bool = Field(False, description="Completely omit the tool.")

    def __bool__(self) -> bool:
        return self.allowed


class Ability(ABC, Cloneable, Generic[TAbilityInput, TAbilityCheckInput, TAbilityOutput]):
    name: str
    description: str
    state: dict[str, Any]
    enabled: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._priority = 10
        self.enabled = True
        self.state = {}

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Priority must be a positive integer.")

        self._priority = value

    @abstractmethod
    def run(self, input: TAbilityInput) -> Run[Any]: ...

    @abstractmethod
    def check(self, input: TAbilityCheckInput) -> AbilityCheckResult: ...

    @cached_property
    def emitter(self) -> Emitter:
        emitter = Emitter.root().child(namespace=["ability", to_safe_word(self.name)])
        return emitter

    @property
    @abstractmethod
    def input_schema(self) -> type[TAbilityInput]: ...

    _registered_classes: ClassVar[dict[str, AgentAbilityFactory]] = {}

    def verify(self, *, abilities: list["AnyAbility"]) -> None:
        pass

    @staticmethod
    def register(name: str, factory: AgentAbilityFactory) -> None:  # TODO: support cloneable?
        if name in Ability._registered_classes:
            raise ValueError(f"Ability with name '{name}' has been already registered!")

        Ability._registered_classes[name] = factory

    @staticmethod
    def lookup(name: str) -> "Ability[Any]":  # TODO: add support for lookup by a signature instead
        factory = Ability._registered_classes.get(name)
        if factory is None:
            raise ValueError(f"Ability with name '{name}' has not been registered!")
        return factory()

    def can_use(self, state: TAbilityCheckInput) -> AbilityCheckResult:  # TODO: rename?
        response = self.check(state)
        return response if isinstance(response, AbilityCheckResult) else AbilityCheckResult(allowed=response)

    async def clone(self) -> Self:
        instance = type(self).__new__(self.__class__)
        instance.name = self.name
        instance.description = self.description
        instance.priority = self.priority
        instance.enabled = self.enabled
        instance.state = self.state.copy()
        self.emitter.pipe(instance.emitter)
        return instance


AnyAbility = Ability[Any, Any, Any]

R = TypeVar("R")
TSelf = TypeVar("TSelf", bound=Ability[Any, Any, Any])


# TODO: refactor name
def with_run_context(
    fn: Callable[
        [TSelf, TAbilityInput, "RunContext"],
        Awaitable[R],
    ],
) -> Callable[[TSelf, TAbilityInput], Run[R]]:
    def decorated(self: TSelf, input: TAbilityInput) -> Run[R]:
        async def handler(context: RunContext) -> R:
            return await fn(self, input, context)

        return RunContext.enter(self, handler, run_params=input.model_dump())

    return decorated


TFunction = Callable[..., Any]
