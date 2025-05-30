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

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions


class ThinkSchema(BaseModel):
    thoughts: str = Field(..., description="Precisely describe what you are thinking about.")
    next_step: list[str] = Field(..., description="Describe tool you would need to use next and why.", min_length=1)


class ThinkTool(Tool[ThinkSchema]):
    name = "think"
    description = "Use when you want to think through a problem, clarify your assumptions, or break down complex steps before acting or responding."  # noqa: E501

    def __init__(self, *, extra_instructions: str = "") -> None:
        super().__init__()
        if extra_instructions:
            self.description += f" {extra_instructions}"

    @property
    def input_schema(self) -> type[ThinkSchema]:
        return ThinkSchema

    async def _run(self, input: ThinkSchema, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        return StringToolOutput("OK")

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "think"],
            creator=self,
        )
