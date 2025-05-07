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
"""A plugin loader module."""

import functools
from typing import Any

from beeai_framework.memory import SlidingMemory, SummarizeMemory, TokenMemory, UnconstrainedMemory
from beeai_framework.plugins.configurable.config import ConfigLoader
from beeai_framework.plugins.constants import CONFIG, ConfigurablePlugins, MemoryType
from beeai_framework.plugins.plugin import Plugin
from beeai_framework.plugins.types import (
        Pluggable,
        PluggableInstanceRegistry,
        PluggableRegistry,
        PluginInstanceRegistry,
        PluginRegistry,
)


class PluginLoader:
        """The plugn loader object."""
        def __init__(self, *, registry: PluggableInstanceRegistry | None = None) -> None:
            self.pluggable_type_factory = registry or PluggableInstanceRegistry()
            self.pluggable_instances = PluggableRegistry()
            self.plugin_type_factory = PluginInstanceRegistry()
            self.plugin_instances = PluginRegistry()
            self._config = None
            self._register_memory()
            self._register_configurable_plugins()

        @staticmethod
        @functools.cache
        def root() -> "PluginLoader":
            """Loads a singleton PluginLoader."""
            return PluginLoader()

        def load_config(self, config: str) -> None:
            """Loads a configuration file.
            Args:
                config: SDK configuration.
            """
            self._config = ConfigLoader.load_config(config)
            self.load_configurable_plugins(self._config.plugin.configpaths, self._config.plugin.plugins)

        def load(self, /, config: dict[str, Any]) -> None:
            """Loads a plugin given a configuration object.
            Args:
                config: a configuration dictionary.
            """
            for name, options in config.items():
                plugin_name = options["plugin"]
                if not isinstance(plugin_name, str):
                    raise ValueError(f"Plugin name must be a string, got {type(plugin_name)}")

                plugin_parameters = options.get("parameters", {})
                if not isinstance(plugin_parameters, dict):
                    raise ValueError(f"Plugin parameters must be a dict, got {type(plugin_parameters)}")

                instance = self.pluggable_type_factory.create(
                    plugin_name,
                    **self._parse_plugin_parameters(plugin_parameters)
                )
                self.pluggable_instances.register(instance, name=name)

        def _parse_plugin_parameters(self, /, input: Any) -> Any:
            if isinstance(input, dict):
                return {key: self._parse_plugin_parameters(value) for key, value in input.items()}
            elif isinstance(input, list):
                return [self._parse_plugin_parameters(item) for item in input]

            if isinstance(input, str) and input.startswith("#"):
                return self.pluggable_instances.lookup(input[1:]).ref
            else:
                return input

        def _register_memory(self) -> None:
            self.pluggable_type_factory.register(UnconstrainedMemory, MemoryType.UNCONSTRAINED)
            self.pluggable_type_factory.register(SummarizeMemory, MemoryType.SUMMARIZE)
            self.pluggable_type_factory.register(SlidingMemory, MemoryType.SLIDING)
            self.pluggable_type_factory.register(TokenMemory, MemoryType.TOKEN)

        def _register_configurable_plugins(self) -> None:
            from beeai_framework.plugins.model.chat import ChatModelPlugin
            self.plugin_type_factory.register(ChatModelPlugin, ConfigurablePlugins.CHAT_MODEL)

        def register_plugin_instance(self, name: str, plugin: Plugin) -> None:
            """Register a plugin instance."""
            self.plugin_instances.register(plugin, name)

        def register_pluggable_type(self, name: str, pluggable: Pluggable) -> None:
            """Register a plugin instance."""
            self.plugin_instances.register(pluggable, name)

        def create_pluggable(self, name: str, parameters: Any) -> Pluggable:
            """Create a pluggable object."""
            return self.pluggable_type_factory.create(name, parameters)

        def create_plugin(self, name: str, parameters: Any) -> Plugin:
            return self.plugin_type_factory.create(name, parameters)

        def load_configurable_plugins(self, configpaths: list[str], reg_plugin_list: list[str]) -> None:
            """Load configurable plugins.
            Args:
                configpaths: the paths to look for plugin configs.
                reg_plugin_list: the list of plugins to register.
            """
            plugin_configs = ConfigLoader.load_plugin_configs(configpaths)

            for name, conf in plugin_configs.items():
                if name in reg_plugin_list:
                    instance = self.create_plugin(conf.based_on, {CONFIG: conf})
                    self.register_plugin_instance(name, instance)

        def get_plugin(self, name: str) -> Plugin:
            """Get a plugin instance.
            Args:
                name: name of plugin.
            Returns:
                A plugin object.
            """
            return self.plugin_instances.lookup(name).ref
