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
"""
Define configuration types for the SDK.
"""

from enum import Enum
from typing import Annotated

from pydantic import AliasChoices, BaseModel, Field, StringConstraints


class OutputType(str, Enum):
    """Enumeration for output types."""

    PLAIN = "plain"
    SELECTION = "selection"
    DATE = "date"
    ERROR = "error"
    INTERRUPT = "interrupt"

    def __str__(self) -> str:
        return str(self.value)

class PluginSection(BaseModel):
    """The plugin registration and configuration."""

    configpaths: list[str] | str = Field(validation_alias=AliasChoices("configpaths", "configpath"))
    codepaths: list[str] = []
    plugins: list[str] | None = None

class Config(BaseModel):
    """The main SDK configuration object."""

    plugin: PluginSection


class Runtime(BaseModel):
    """The plugin's runtime configuration object."""

    class_name: str = Field(alias="class")
    tests: list[str] | None = None


class PluginConfig(BaseModel):
    """The plugin object."""

    name: Annotated[str, StringConstraints(min_length=1)]
    description: Annotated[str, StringConstraints(min_length=1)]
    version: str
    tags: list | None = []
    based_on: str | None = None
    runtime: Runtime | None = None
    pipeline: str | None = None
    config: dict | None = {}
    streamlit: dict | None = {}
    analysis: dict | None = None

class DataContext(BaseModel):
    """Instance of the data input."""

    data: str | dict | list
    context: dict | None = Field(default={})
