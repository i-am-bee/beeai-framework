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

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.plugins.plugin import Plugin
from beeai_framework.registry import Registry

TInput = TypeVar("TInput", bound=BaseModel, default=Any)
TOutput = TypeVar("TOutput", bound=BaseModel, default=Any)


@runtime_checkable
class Pluggable(Protocol[TInput, TOutput]):
    def as_plugin(self) -> Plugin[TInput, TOutput]: ...


class PluggableRegistry(Registry[type[Pluggable]]): ...


class PluggableInstanceRegistry(Registry[Pluggable]): ...
