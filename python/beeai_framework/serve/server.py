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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Generic, Self

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.agents import BaseAgent
from beeai_framework.agents.base import AnyAgent

TConfig = TypeVar("TConfig", bound=BaseModel, default=BaseModel)
TAgent = TypeVar("TAgent", bound=object, default=Any)
TSourceAgent = TypeVar("TSourceAgent", bound=BaseAgent[Any])


class AgentServer(Generic[TConfig, TAgent], ABC):
    _factories: ClassVar[dict[type[AnyAgent], Callable[[AnyAgent], TAgent]]] = {}  # type: ignore[misc]

    def __init__(self, *, config: TConfig) -> None:
        self._agents: list[AnyAgent] = []
        self._config = config

    @classmethod
    def register_factory(
        cls,
        ref: type[TSourceAgent],
        factory: Callable[[TSourceAgent], TAgent],
        *,
        override: bool = False,
    ) -> None:
        if ref not in cls._factories or override:
            cls._factories[ref] = factory  # type: ignore
        elif cls._factories[ref] is not factory:
            raise ValueError(f"Factory for {ref} is already registered.")

    def register(self, agents: list[AnyAgent] | AnyAgent) -> Self:
        for agent in agents if isinstance(agents, list) else [agents]:
            if not self.supports(agent):
                raise ValueError(f"Agent {type(agent)} is not supported by this server.")
            if agent not in self._agents:
                self._agents.append(agent)

        return self

    def deregister(self, agent: AnyAgent) -> Self:
        self._agents.remove(agent)
        return self

    @classmethod
    def supports(cls, agent: AnyAgent) -> bool:
        return type(agent) in cls._factories

    @property
    def agents(self) -> list[AnyAgent]:
        return self._agents

    @abstractmethod
    def serve(self) -> None:
        pass
