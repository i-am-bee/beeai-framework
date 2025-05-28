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

import inspect
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from typing import Any, ClassVar, Generic, TypedDict, TypeVar

from pydantic import BaseModel
from typing_extensions import Unpack

from beeai_framework.context import Run
from beeai_framework.emitter import Emitter
from beeai_framework.registry import Registry, RegistryEntry
from beeai_framework.utils.cancellation import AbortSignal
from beeai_framework.utils.models import ModelLike, to_model

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class PluginKwargs(TypedDict, total=False):
    context: dict[str, Any]
    signal: AbortSignal


class PluginRegistry(Registry["AnyPlugin"]): ...


class Plugin(ABC, Generic[TInput, TOutput]):
    _auto_register: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._is_running = False
        self._uuid = uuid.uuid4()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> type[TInput]: ...

    @property
    @abstractmethod
    def output_schema(self) -> type[TOutput]: ...

    @property
    @abstractmethod
    def emitter(self) -> Emitter: ...

    @property
    def uuid(self) -> str:
        """plugin UUID."""
        return self._uuid.hex

    @abstractmethod
    def run(self, input: ModelLike[TInput], /, **kwargs: Unpack[PluginKwargs]) -> Run[TOutput]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Run[TOutput]:
        extra = PluginKwargs()
        for k in PluginKwargs.__annotations__:
            value = kwargs.get(k)
            if value is not None:
                extra[k] = value  # type: ignore

        input = (
            to_model(self.input_schema, args[0])
            if args and not kwargs
            else self.input_schema.model_validate(kwargs)  # TODO: improve
        )
        return self.run(input, **extra)

    def __init_subclass__(cls) -> None:
        if cls._auto_register and not inspect.isabstract(cls):
            with suppress(Exception):
                PluginRegistry.root().register(cls)

        return super().__init_subclass__()

    @staticmethod
    def register(
        ref: type["AnyPlugin"],
        /,
        name: str | None = None,
        description: str | None = None,
        override: bool = False,
    ) -> None:
        return PluginRegistry.root().register(ref, name=name, description=description, override=override)

    @staticmethod
    def lookup(name: str) -> RegistryEntry["AnyPlugin"]:
        return PluginRegistry.root().lookup(name)


AnyPlugin = Plugin[Any, Any]


TFunction = Callable[..., Any]
