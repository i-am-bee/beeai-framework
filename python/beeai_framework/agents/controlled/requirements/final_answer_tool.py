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

from pydantic import BaseModel, Field, create_model

from beeai_framework.agents.controlled.types import AbilityAgentRunState
from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions


# TODO -> change to something like
#   TerminateAbility
#   SubmitAbility
#   FinishAbility
#   CompleteAbility
class FinalAnswerTool(Tool[BaseModel, ToolRunOptions, StringToolOutput]):
    name = "final_answer"
    description = "Sends the final answer to the user"

    def __init__(
        self, expected_output: str | type[BaseModel] | None, state: AbilityAgentRunState
    ) -> None:  # TODO: propagate state with ability context..
        super().__init__()
        self._expected_output = expected_output
        self._state = state
        self.instructions = expected_output if isinstance(expected_output, str) else None
        self.custom_schema = isinstance(expected_output, type)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "final_answer"], context={}, creator=self)

    @property
    def input_schema(self) -> type[BaseModel]:
        return (
            self._expected_output
            if (
                self._expected_output is not None
                and isinstance(self._expected_output, type)
                and issubclass(self._expected_output, BaseModel)
            )
            else create_model(
                f"{self.name}Schema",
                response=(
                    str,
                    Field(description=self._expected_output or None),
                ),
            )
        )

    async def _run(self, input: BaseModel, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        if self.input_schema is self._expected_output:
            self._state.result = AssistantMessage(input.model_dump_json())
        else:
            self._state.result = AssistantMessage(input.response)  # type: ignore

        return StringToolOutput("Message has been sent")
