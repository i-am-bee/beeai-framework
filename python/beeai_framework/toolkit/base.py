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
"""Configurable base Pluggable class."""

from functools import cached_property

from beeai_framework.emitter import Emitter
from beeai_framework.plugins.plugin import Plugin
from beeai_framework.plugins.schemas import DataContext
from beeai_framework.plugins.types import Pluggable
from beeai_framework.utils.strings import to_safe_word


class BasePlugin(Pluggable[DataContext, DataContext], Plugin[DataContext, DataContext]):
    """Base toolkit class for Data Context"""

    def __init__(self, name: str, description: str, id_: str) -> None:
        """Initialize a base plugin.
        Args:
            name: name of the plugin.
            description: description of the plugin.
        """
        super().__init__()
        self._name = name
        self._description = description
        self._id = id_

    @property
    def id(self) -> str:
        """Plugin ID."""
        return self._id

    @property
    def name(self) -> str:
        """Plugin name."""
        return self._name

    @property
    def description(self) -> str:
        """Plugin description."""
        return self._description

    @property
    def input_schema(self) -> type[DataContext]:
        """Input schema."""
        return DataContext

    @property
    def output_schema(self) -> type[DataContext]:
        """Output schema."""
        return DataContext

    @cached_property
    def emitter(self) -> Emitter:
        """Create an Emitter for the plugin."""
        return Emitter.root().child(namespace=["toolkit", "chat", to_safe_word(self._name)])

    def as_plugin(self) -> Plugin[DataContext, DataContext]:
        """Returns the model as a plugin.
        Returns:
            A plugin object.
        """
        return self
