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


from typing import Any, TypeVar

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools import Tool as LangChainSimpleTool
from pydantic import BaseModel, ConfigDict

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import StringToolOutput, ToolRunOptions
from beeai_framework.utils.strings import to_safe_word


class LangChainToolRunOptions(ToolRunOptions):
    langchain_runnable_config: RunnableConfig | None = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


T = TypeVar("T", bound=BaseModel)


class LangChainTool(Tool[T, LangChainToolRunOptions, StringToolOutput]):
    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def input_schema(self) -> type[T]:
        return self._tool.input_schema  # type: ignore

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "langchain", to_safe_word(self._tool.name)],
            creator=self,
        )

    def __init__(
        self, tool: StructuredTool | LangChainSimpleTool | BaseTool, options: dict[str, Any] | None = None
    ) -> None:
        super().__init__(options)
        self._tool = tool

    async def _run(self, input: T, options: LangChainToolRunOptions | None, context: RunContext) -> StringToolOutput:
        langchain_runnable_config = options.langchain_runnable_config or {} if options else {}
        args = (
            input if isinstance(input, dict) else input.model_dump(),
            {
                **langchain_runnable_config,
                "signal": context.signal or None if context else None,
            },
        )
        is_async = (isinstance(self._tool, StructuredTool) and self._tool.coroutine) or (
            isinstance(args[0].get("run_manager"), AsyncCallbackManagerForToolRun)
        )
        if is_async:
            response = await self._tool.ainvoke(*args)  # type: ignore
        else:
            response = self._tool.invoke(*args)  # type: ignore

        return StringToolOutput(result=str(response))
