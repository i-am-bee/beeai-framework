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
"""Module for base configurable plugin."""

from typing import Any

from beeai_framework.emitter import Emitter
from beeai_framework.plugins.configurable.schemas import DataContext, PluginConfig
from beeai_framework.plugins.plugin import Plugin


class ConfigurablePlugin(Plugin[DataContext, DataContext]):
    """Base configurable plugin."""

    def __init__(self, config: dict, emmiter_type: str = "plugin", **kwargs: Any) -> None:
        """Initializes a base no-code plugin class.
        Args:
          config: configuration dictionary.
          kwargs: extra key word arguments.
        """
        super().__init__(**kwargs)
        self._config = PluginConfig.model_validate(config)
        self._name = self._config.name
        self._description = self._config.description
        self._emitter_type = emmiter_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> type[DataContext]:
        return DataContext

    @property
    def output_schema(self) -> type[DataContext]:
        return DataContext

    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=[self._emitter_type, self._name])
