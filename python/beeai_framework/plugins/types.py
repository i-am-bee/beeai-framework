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
"""Definition of plugin types."""

from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, StringConstraints
from typing_extensions import TypeVar

from beeai_framework.plugins.plugin import Plugin
from beeai_framework.registry import Registry
from beeai_framework.utils import ModelLike


class DataContext(BaseModel):
    """Instance of the data exchange object for input and output."""

    data: str | list[Any] | ModelLike[BaseModel]
    context: dict[str, Any] | None = Field(default={})


TInput = TypeVar("TInput", bound=DataContext, default=DataContext)
TOutput = TypeVar("TOutput", bound=DataContext, default=DataContext)


@runtime_checkable
class Pluggable(Protocol[TInput, TOutput]):
    def as_plugin(self) -> Plugin[TInput, TOutput]: ...


DataContextPluggable = Pluggable[DataContext, DataContext]


class PluggableInstanceRegistry(Registry[type[Pluggable]]):
    def create(self, name: str, parameters: Any) -> Pluggable:
        pluggable = self.lookup(name)
        return pluggable.ref(**parameters)


class PluggableRegistry(Registry[Pluggable]): ...


class PluginSection(BaseModel):
    """The plugin registration and configuration."""

    configpaths: list[str] = []
    codepaths: list[str] = []
    plugins: list[str] = []


class Config(BaseModel):
    """The main SDK configuration object."""

    loader: PluginSection


class Runtime(BaseModel):
    """The plugin's runtime configuration object."""

    class_name: str = Field(alias="class")
    tests: list[str] | None = None


class PluginConfig(BaseModel):
    """The plugin object."""

    name: Annotated[str, StringConstraints(min_length=1)]
    id_: str = Field(alias="id")
    display_name: str | None = ""
    description: Annotated[str, StringConstraints(min_length=1)]
    version: str
    tags: list[str] | None = []
    based_on: str | None = None
    runtime: Runtime | None = None
    pipeline: str | None = None
    config: dict[Any, Any] | None = {}
    streamlit: dict[Any, Any] | None = {}
    analysis: dict[Any, Any] | None = None


class PluggableDef(BaseModel):
    """Pluggable definition."""

    type: str = ""
    arguments: dict[Any, Any] | None = {}
