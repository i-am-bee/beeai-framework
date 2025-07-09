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

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from beeai_framework.backend import AssistantMessage
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import RunnableConfig
from beeai_framework.tools.tool import AnyTool


class AgentExecutionConfig(RunnableConfig):
    """An agent's execution configuration."""

    context: str | None = None
    """
    Additional piece of context to be passed to the model.
    """

    expected_output: str | type[BaseModel] | None = None
    """
    Instruction for steering the agent towards an expected output format.
    This can be a Pydantic model for structured output decoding and validation.
    """

    total_max_retries: int = 3
    """
    Maximum number of model retries.
    """

    max_retries_per_step: int = 20
    """
    Maximum number of model retries per step.
    """

    max_iterations: int = 10
    """
    Maximum number of iterations.
    """


class AgentMeta(BaseModel):
    """An agent's metadata."""

    name: str
    description: str
    tools: list[InstanceOf[AnyTool]]
    extra_description: str | None = None


class AgentRunOutput(BaseModel):
    """An agent's execution output."""

    model_config = ConfigDict(populate_by_name=True)

    answer: InstanceOf[AssistantMessage] = Field(alias="result")
    memory: InstanceOf[BaseMemory]

    @property
    def result(self) -> AssistantMessage:
        """
        This property is provided for compatibility reasons only.
        Use 'answer' instead.
        """
        return self.answer

    @result.setter
    def result(self, value: AssistantMessage) -> None:
        """
        This setter is provided for compatibility reasons only.
        Sets the 'answer' attribute.
        """
        self.answer = value
