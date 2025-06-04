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
"""Base agent plugin class."""

from functools import cached_property

from beeai_framework.backend import ChatModel
from beeai_framework.backend.events import chat_model_event_types
from beeai_framework.emitter import Emitter
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.toolkit.base import BasePlugin
from beeai_framework.toolkit.chat.types import Example, Templates
from beeai_framework.utils.models import ModelLike
from beeai_framework.utils.strings import to_safe_word


class BaseAgentPlugin(BasePlugin):
    """Base agent plugin class."""

    def __init__(
        self,
        name: str,
        description: str,
        id_: str,
        model: ChatModel | str,
        instruction: str | None = None,
        templates: ModelLike[Templates] | None = None,
        examples: list[Example] | None = None,
        examples_files: list[str] | None = None,
        stream: bool = False,
        memory: BaseMemory | None = None,
    ) -> None:
        """Initializes a prompting chat model.
        Args:
            name: the name of the object.
            description: a meaningful description of what the model does.
            id: an alternative ID for the model.
            parameters: the model parameters.
            instruction: an instruction prompt.
            templates: a set of user and few shot example prompting templates.
            examples: a list of few shot examples.
            examples_files: a list files containing few shot examples in jsonl format.
            stream: True if streaming turned on.
            memory: memory for storing chat history.
        """
        super().__init__(name, description, id_)
        self._instruction = instruction
        if templates:
            self._templates = templates if isinstance(templates, Templates) else Templates.model_validate(templates)
        else:
            self._templates = Templates()
        self._examples = examples or []
        self._examples_files = examples_files or []
        self._stream = stream
        self._model = ChatModel.from_name(model) if isinstance(model, str) else model
        self._memory = memory

    @property
    def id(self) -> str:
        """The plugin ID.
        Returns:
            An id.
        """
        return self._id

    @property
    def templates(self) -> Templates:
        """The prompt templates.
        Returns:
            Prompt templates.
        """
        return self._templates

    @property
    def examples(self) -> list[Example]:
        """A list of fewshot examples.
        Returns:
            A list of fewshot examples.
        """
        return self._examples

    @property
    def instruction(self) -> str:
        """A prompt instruction.
        Returns:
            A prompt instruction.
        """
        return self._instruction or ""

    @property
    def stream(self) -> bool:
        """Whether streaming is supported.
        Returns:
            True if streaming enabled.
        """
        return self._stream

    @cached_property
    def emitter(self) -> Emitter:
        """An event emitter.
        Returns:
            An emitter.
        """
        return self._create_emitter()

    @property
    def memory(self) -> BaseMemory | None:
        """The memory object.
        Returns:
            A memory object.
        """
        return self._memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["toolkit", "chat", to_safe_word(self._name)],
            creator=self,
            events=chat_model_event_types,
        )
