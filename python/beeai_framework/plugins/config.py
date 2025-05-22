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
"""Implement configuration loaders."""

import logging
import os
import re
from glob import glob

import chevron
import yaml
from pydantic import ValidationError

from beeai_framework.plugins.types import Config, PluginConfig


class ConfigLoader:
    """A configuration loader."""

    @staticmethod
    def load_config(config: str, use_mustache: bool = False) -> Config:
        """Load the SDK configuration from a file path.

        Args:
          config: the configuration path

        Returns:
          The SDK configuration object

        """
        with open(os.path.normpath(config), encoding="utf8") as file:
            template = file.read()
            rendered_template = chevron.render(template, data=dict(os.environ)) if use_mustache else template
            config_data = yaml.safe_load(rendered_template)
        return Config(**config_data)

    @staticmethod
    def dump_config(path: str, config: Config) -> None:
        """Dumps plugin configuration to a file.

        Args:
          path: configuration file path
          config: the plugin configuration path

        Returns:
          None
        """
        with open(os.path.normpath(path), "w", encoding="utf8") as file:
            yaml.safe_dump(config.model_dump(exclude_none=True), file)

    @staticmethod
    def load_plugin_config(config: str) -> PluginConfig:
        """Load a plugin configuration from a file path.

        This function autoescapes curly brackets in the 'instruction'
        and 'examples' keys under the config attribute.

        Args:
          config: the plugin configuration path

        Returns:
          The plugin configuration object

        """
        with open(os.path.normpath(config), encoding="utf8") as file:
            template = file.read()
            config_data = yaml.safe_load(template)
            if config_data.get("config"):
                for key in ["instruction", "examples"]:
                    if key in config_data["config"]:
                        value = re.sub(
                            r"(?<!{){(?!{)(.+)(?<!})}(?!})", r"{{\1}}", yaml.dump(config_data["config"][key])
                        )
                        config_data["config"][key] = yaml.load(value, Loader=yaml.FullLoader)
        return PluginConfig(**config_data)

    @staticmethod
    def load_plugin_configs(dirs: list[str] | str, ignore_duplicates: bool = True) -> dict[str, PluginConfig]:
        """Load a plugin configuration recursively from a file path.

        Args:
          root: the root path from which to load plugin configurations

        Returns:
          The dictionary of plugin configuration objects

        """
        roots = [dirs] if isinstance(dirs, str) else dirs
        plugs = {}
        for root in roots:
            path = os.path.join(os.path.normpath(root), "**", "*.yaml")
            files = glob(path, recursive=True)
            for f in files:
                try:
                    pconf = ConfigLoader.load_plugin_config(f)
                except ValidationError as e:
                    if f.endswith("plugin.yaml"):
                        logging.error("Plugin validation error for %s: %s", f, repr(e))
                    else:
                        logging.debug("YAML validation error for %s: %s", f, repr(e))
                    continue
                if pconf.name in plugs:
                    logging.warning("Multiple instances of plugin %s have been found", pconf.name)
                if not ignore_duplicates:
                    raise ValueError(f"Duplicate plugin name found {pconf.name}")
                plugs[pconf.name] = pconf
        return plugs
