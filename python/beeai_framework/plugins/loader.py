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
import re
from pydoc import locate
from typing import Any, cast

from beeai_framework.adapters.amazon_bedrock import AmazonBedrockChatModel
from beeai_framework.adapters.anthropic import AnthropicChatModel
from beeai_framework.adapters.azure_openai import AzureOpenAIChatModel
from beeai_framework.adapters.groq import GroqChatModel
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.adapters.watsonx import WatsonxChatModel
from beeai_framework.adapters.xai import XAIChatModel
from beeai_framework.backend import ChatModel
from beeai_framework.memory import SlidingMemory, SummarizeMemory, TokenMemory, UnconstrainedMemory
from beeai_framework.plugins.config import ConfigLoader
from beeai_framework.plugins.constants import ARGUMENTS, DESCRIPTION, ID, NAME, TYPE
from beeai_framework.plugins.plugin import AnyPlugin
from beeai_framework.plugins.schemas import Config, PluginConfig
from beeai_framework.plugins.types import (
    Pluggable,
    PluggableInstanceRegistry,
    PluggableRegistry,
)
from beeai_framework.plugins.utils import topological_sort
from beeai_framework.utils.dicts import merge_nested, sort_by_key


class PluginLoader:
    """The plugin loader object."""

    REF_PREFIX = "$"

    def __init__(self, *, registry: PluggableInstanceRegistry | None = None) -> None:
        self.pluggable_type_factory = registry or PluggableInstanceRegistry()
        self.pluggable_instances = PluggableRegistry()
        self._config: Config | None = None
        self._register_memory()
        self._register_toolkit_classes()
        self._register_chat_models()

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
        plugin = self._config.loader if self._config else None
        if plugin:
            self.load_configurable_plugins(plugin.configpaths, plugin.plugins)

    def load(self, /, config: dict[str, Any], interpret_variables: bool = False) -> None:
        """Loads a plugin given a configuration object.
        Args:
            config: a configuration dictionary.
            interpret_variables: if set to true, will interpret strings with '#'/'$' as variables.
        """

        # TODO: idea (plugin can be registered as a dynamic class for other plugins)
        def register_dynamic_plugin_class(instance: Pluggable, parameters: dict[str, Any], name: str) -> None:
            class DynamicPluginClass(type(instance)):  # type: ignore
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    new_kwargs = merge_nested(parameters, kwargs)
                    super().__init__(*args, **new_kwargs)

            DynamicPluginClass.__name__ = name
            self.pluggable_type_factory.register(DynamicPluginClass, name)

        for name, options in config.items():
            plugin_name = options[TYPE]
            if not isinstance(plugin_name, str):
                raise ValueError(f"Plugin name must be a string, got {type(plugin_name)}")

            plugin_parameters = options.get(ARGUMENTS, {})
            if not isinstance(plugin_parameters, dict):
                raise ValueError(f"Plugin parameters must be a dict, got {type(plugin_parameters)}")

            parameters = self._parse_plugin_parameters(
                sort_by_key(plugin_parameters), interpret_variables=interpret_variables
            )
            parameters_without_refs = {k: v for k, v in parameters.items() if not k.startswith(PluginLoader.REF_PREFIX)}
            instance = self.pluggable_type_factory.create(plugin_name, parameters_without_refs)
            self.pluggable_instances.register(instance, name=name)
            register_dynamic_plugin_class(instance, parameters_without_refs, name)

    def _parse_plugin_parameters(self, /, input: Any, interpret_variables: bool = False, key: str | None = None) -> Any:
        if isinstance(input, dict):
            if TYPE in input and (ARGUMENTS in input or len(input.keys()) == 1):
                if not self.is_pluggable_type_registered(input[TYPE]):
                    pluggable_type = locate(input[TYPE])
                    if pluggable_type is Pluggable:
                        self.register_pluggable_type(pluggable_type, input[TYPE])
                    else:
                        raise ValueError(f"Unable to register type: {input[TYPE]}")
                pluggable = self.create_pluggable(
                    input[TYPE], input.get(ARGUMENTS, {}), interpreter_variables=interpret_variables
                )
                # handle references specified in the config
                if key and key.startswith(PluginLoader.REF_PREFIX):
                    self.pluggable_instances.register(pluggable, name=key)
                return pluggable
            else:
                return {
                    key: self._parse_plugin_parameters(value, interpret_variables=interpret_variables, key=key)
                    for key, value in input.items()
                }
        elif isinstance(input, list):
            return [
                self._parse_plugin_parameters(item, interpret_variables=interpret_variables, key=key) for item in input
            ]

        # TODO: needs to decide whether # or $ ... (# needs to be escaped in quotes in YAML)
        ref_name = isinstance(input, str) and interpret_variables and re.search(r"^\$(\w+)", input)
        if ref_name:
            return self.pluggable_instances.lookup(input).ref
        else:
            return input

    def _register_toolkit_classes(self) -> None:
        from beeai_framework.toolkit.chat.agent import Agent
        from beeai_framework.toolkit.chat.prompting import PromptingChatModel

        self.pluggable_type_factory.register(PromptingChatModel)
        self.pluggable_type_factory.register(Agent)

    def _register_memory(self) -> None:
        self.pluggable_type_factory.register(UnconstrainedMemory)
        self.pluggable_type_factory.register(SummarizeMemory)
        self.pluggable_type_factory.register(SlidingMemory)
        self.pluggable_type_factory.register(TokenMemory)

    def _register_chat_models(self) -> None:
        self.pluggable_type_factory.register(AmazonBedrockChatModel)
        self.pluggable_type_factory.register(AzureOpenAIChatModel)
        self.pluggable_type_factory.register(AnthropicChatModel)
        self.pluggable_type_factory.register(GroqChatModel)
        self.pluggable_type_factory.register(OllamaChatModel)
        self.pluggable_type_factory.register(OpenAIChatModel)
        self.pluggable_type_factory.register(WatsonxChatModel)
        self.pluggable_type_factory.register(XAIChatModel)

        def dynamic_factory(model_id: str, **kwargs: Any) -> ChatModel:
            return ChatModel.from_name(model_id, **kwargs)

        self.pluggable_type_factory.register(cast(type[Pluggable], dynamic_factory), name="ChatModel")

    def register_pluggable_type(self, pluggable: type[Pluggable], name: str | None = None) -> None:
        """Register a plugin instance."""
        self.pluggable_type_factory.register(pluggable, name)

    def register_pluggable_instance(self, pluggable: Pluggable, name: str) -> None:
        """Register a pluggable instance."""
        self.pluggable_instances.register(pluggable, name)

    def create_pluggable(self, name: str, parameters: Any, interpreter_variables: bool = False) -> Pluggable:
        """Create a pluggable object."""
        return self.pluggable_type_factory.create(
            name, self._parse_plugin_parameters(parameters, interpreter_variables)
        )

    def create_plugin(self, name: str, parameters: Any) -> AnyPlugin:
        """Create pluggable object as a plugin."""
        pluggable = self.create_pluggable(name, parameters)
        return pluggable.as_plugin()

    def _get_pluggable_parameters(self, config: PluginConfig) -> dict[str, Any]:
        arguments = config.config.copy() if config.config else {}
        arguments[NAME] = config.name
        arguments[DESCRIPTION] = config.description
        arguments[ID] = config.id_
        plugin_config = {
            config.name: {
                TYPE: config.based_on,
                ARGUMENTS: arguments,
            }
        }
        return plugin_config

    def load_configurable_plugins(self, configpaths: list[str], reg_plugin_list: list[str]) -> None:
        """Load configurable plugins.
        Args:
            configpaths: the paths to look for plugin configs.
            reg_plugin_list: the list of plugins to register.
        """
        plugin_configs = ConfigLoader.load_plugin_configs(configpaths)
        for name, conf in topological_sort(plugin_configs, attr_name="based_on"):
            if name in reg_plugin_list:
                params = self._get_pluggable_parameters(conf)
                self.load(params, interpret_variables=True)

    def get_plugin(self, name: str) -> AnyPlugin:
        """Get a plugin instance.
        Args:
            name: name of plugin.
        Returns:
            A plugin object.
        """
        pluggable = self.pluggable_instances.lookup(name).ref
        return pluggable.as_plugin()

    def is_pluggable_type_registered(self, name: str) -> bool:
        """Returns true if a pluggable type is registered.
        Args:
            name - the pluggable name.
        Returns:
            True if the pluggable type is registered.
        """
        return self.pluggable_type_factory.is_registered(name)
