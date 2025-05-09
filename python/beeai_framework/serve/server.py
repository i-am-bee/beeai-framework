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
from collections.abc import Callable, Sequence
from typing import ClassVar, Generic, Self

from pydantic import BaseModel
from typing_extensions import TypeVar

TInput = TypeVar("TInput", bound=object, default=object, contravariant=True)
TInternal = TypeVar("TInternal", bound=object, default=object)
TConfig = TypeVar("TConfig", bound=BaseModel, default=BaseModel)


class Server(Generic[TInput, TInternal, TConfig], ABC):
    _factories: ClassVar[dict[type[TInput], Callable[[TInput], TInternal]]] = {}  # type: ignore[misc]

    def __init__(self, *, config: TConfig) -> None:
        self._members: list[TInput] = []
        self._config = config

    @classmethod
    def register_factory(
        cls,
        ref: type[TInput],
        factory: Callable[[TInput], TInternal],
        *,
        override: bool = False,
    ) -> None:
        if ref not in cls._factories or override:
            cls._factories[ref] = factory
        elif cls._factories[ref] is not factory:
            raise ValueError(f"Factory for {ref} is already registered.")

    def register(self, input: TInput | Sequence[TInput]) -> Self:
        for value in input if isinstance(input, Sequence) else [input]:
            if not self.supports(value):
                raise ValueError(f"Agent {type(value)} is not supported by this server.")
            if value not in self._members:
                self._members.append(value)

        return self

    def deregister(self, input: TInput) -> Self:
        self._members.remove(input)
        return self

    @classmethod
    def supports(cls, input: TInput) -> bool:
        return type(input) in cls._factories

    @property
    def members(self) -> list[TInput]:
        return self._members

    @abstractmethod
    def serve(self) -> None:
        pass
