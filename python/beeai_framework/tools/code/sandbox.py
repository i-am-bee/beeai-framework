# Copyright 2025 IBM Corp.
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

import json
from typing import Any, Self

from pydantic import BaseModel

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.code.python import PythonTool
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import StringToolOutput, ToolRunOptions
from beeai_framework.utils.models import JSONSchemaModel


class SandboxToolCreateError(FrameworkError):
    pass


class SandboxToolExecuteError(FrameworkError):
    pass


class SandboxToolOptions(ToolRunOptions):
    code_interpreter_url: str
    source_code: str
    name: str
    description: str
    input_schema: dict[str, Any]
    env: dict[str, Any]


class SandboxTool(Tool[BaseModel, SandboxToolOptions, StringToolOutput]):
    def __init__(self, options: SandboxToolOptions) -> None:
        super().__init__()
        self.__options = options

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "sandbox"],
            creator=self,
        )

    @property
    def name(self) -> str:
        return self.__options.name

    @property
    def description(self) -> str:
        return self.__options.description

    @property
    def input_schema(self) -> type[BaseModel]:
        return JSONSchemaModel.create(self.name, self.__options.input_schema)

    async def _run(
        self, tool_input: BaseModel | dict[str, Any], options: SandboxToolOptions | None, context: RunContext
    ) -> StringToolOutput:
        try:
            result = await PythonTool.call_code_interpreter(
                f"{self.__options.code_interpreter_url}/v1/execute-custom-tool",
                {
                    "tool_source_code": self.__options.source_code,
                    "tool_input_json": tool_input.model_dump_json()
                    if isinstance(tool_input, BaseModel)
                    else json.dumps(tool_input),
                    "env": {**self.__options.env, **(options.env if options else {})},
                },
            )

            if result.get("stderr"):
                raise SandboxToolExecuteError(result["stderr"])

            return StringToolOutput(result["tool_output_json"])
        except Exception as err:
            raise SandboxToolExecuteError.ensure(err)

    @classmethod
    async def from_source_code(cls, url: str, env: dict[str, Any], source_code: str) -> Self:
        try:
            result = await PythonTool.call_code_interpreter(
                f"{url}/v1/parse-custom-tool", {"tool_source_code": source_code}
            )

            if result.get("error_messages"):
                raise SandboxToolCreateError(result["error_messages"].join("\n"))

            return cls(
                SandboxToolOptions(
                    code_interpreter_url=url,
                    source_code=source_code,
                    name=result["tool_name"],
                    description=result["tool_description"],
                    input_schema=json.loads(result["tool_input_schema_json"]),
                    env=env,
                )
            )
        except Exception as err:
            raise SandboxToolCreateError.ensure(err)
