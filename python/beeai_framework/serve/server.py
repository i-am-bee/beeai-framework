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
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

TServeConfig = TypeVar("TServeConfig", bound=BaseModel)
TAgent = TypeVar("TAgent", bound=Any)


class Server(Generic[TServeConfig, TAgent], ABC):
    def __init__(self) -> None:
        self._agents: list[TAgent] = []

    def register(self, agents: list[TAgent]) -> "Server[Any, Any]":
        self._agents = agents
        return self

    @property
    def agents(self) -> list[TAgent]:
        return self._agents

    @abstractmethod
    def serve(self, *, config: TServeConfig | None = None) -> None:
        pass
