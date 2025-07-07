# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar, Generic, Self

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.serve.errors import FactoryAlreadyRegisteredError

TInput = TypeVar("TInput", bound=object, default=object, contravariant=True)
TInternal = TypeVar("TInternal", bound=object, default=object)
TConfig = TypeVar("TConfig", bound=BaseModel, default=BaseModel)


class Server(Generic[TInput, TInternal, TConfig], ABC):
    _factories: ClassVar[dict[type[TInput], Callable[[TInput], TInternal]]] = {}  # type: ignore[misc]

    # TODO: later remove config property
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
            raise FactoryAlreadyRegisteredError(f"Factory for {ref} is already registered.")

    def register(self, input: TInput) -> Self:
        # check if the type has a factory registered
        type(self)._get_factory(input)
        if input not in self._members:
            self._members.append(input)

        return self

    def register_many(self, input: Sequence[TInput]) -> Self:
        for item in input:
            self.register(item)

        return self

    def deregister(self, input: TInput) -> Self:
        self._members.remove(input)
        return self

    @classmethod
    def _get_factory(cls, input: TInput) -> Callable[[TInput], TInternal]:
        factory = cls._factories.get(type(input))
        if factory is None:
            raise ValueError(f"No factory registered for {type(input)}.")
        return factory

    @property
    def members(self) -> list[TInput]:
        return self._members

    @abstractmethod
    def serve(self) -> None:
        pass
