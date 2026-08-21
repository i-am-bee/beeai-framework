# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

try:
    from feedo.memory import FeedoMemory
except ImportError as e:
    raise ImportError(
        "Optional module [feedo-sdk] not found.\nRun 'pip install \"feedo-sdk>=0.1.24\"' to install."
    ) from e

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools import ToolError, ToolOutput
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

logger = Logger(__name__)


class FeedoToolInput(BaseModel):
    action: Literal["add", "search", "update", "delete"] = Field(
        description="The action to perform on the Feedo memory network."
    )
    query: str | None = Field(
        default=None,
        description="The search query (required for 'search' action) or the text to add/update.",
    )
    memory_id: str | None = Field(
        default=None,
        description="The ID of the memory to update or delete (required for 'update' or 'delete').",
    )
    topic: str | None = Field(
        default=None,
        description="Optional category or topic when adding a memory.",
    )


class FeedoToolOutput(ToolOutput):
    def __init__(self, result: str, memories: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.result = result
        self.memories = memories

    def get_text_content(self) -> str:
        if self.memories:
            import json
            return f"{self.result}\nMemories:\n{json.dumps(self.memories, indent=2)}"
        return self.result

    def is_empty(self) -> bool:
        return not self.result


class FeedoSearchTool(Tool[FeedoToolInput, ToolRunOptions, FeedoToolOutput]):
    name = "FeedoMemory"
    description = (
        "Access the decentralized Feedo Memory Network to store, search, update, "
        "and delete permanent knowledge and context across sessions."
    )
    input_schema = FeedoToolInput

    def __init__(
        self,
        usage_key: str | None = None,
        private: bool = True,
        *,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(options)
        self.usage_key = usage_key or os.environ.get("FEEDO_USAGE_KEY")
        if not self.usage_key:
            raise ValueError(
                "FEEDO_USAGE_KEY is required to initialize Feedo memory.\n"
                "You can generate a free testnet usage key at: https://feedo.ink"
            )
        self.private = private
        self._memory = FeedoMemory(usage_key=self.usage_key, private=self.private)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "feedo"],
            creator=self,
        )

    async def _run(
        self, input: FeedoToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> FeedoToolOutput:
        try:
            if input.action == "add":
                if not input.query:
                    raise ValueError("The 'query' field is required to add a memory.")
                metadata = {"topic": input.topic} if input.topic else {}
                mem_id = self._memory.add(input.query, metadata=metadata)
                return FeedoToolOutput(result=f"Successfully saved to Feedo memory with ID: {mem_id}")

            elif input.action == "search":
                if not input.query:
                    raise ValueError("The 'query' field is required to search memory.")
                results = self._memory.search(input.query, limit=5)
                if not results:
                    return FeedoToolOutput(result="No relevant memories found.", memories=[])
                return FeedoToolOutput(result=f"Found {len(results)} memories.", memories=results)

            elif input.action == "update":
                if not input.memory_id or not input.query:
                    raise ValueError("Both 'memory_id' and 'query' are required to update a memory.")
                new_id = self._memory.update(input.memory_id, input.query)
                return FeedoToolOutput(result=f"Memory successfully updated. New ID: {new_id}")

            elif input.action == "delete":
                if not input.memory_id:
                    raise ValueError("The 'memory_id' field is required to delete a memory.")
                self._memory.delete(input.memory_id)
                return FeedoToolOutput(result=f"Memory {input.memory_id} successfully deleted.")

            else:
                raise ValueError(f"Unknown action: {input.action}")

        except Exception as e:
            raise ToolError(f"Error interacting with Feedo: {str(e)}") from e

    async def clone(self) -> Self:
        tool = self.__class__(
            usage_key=self.usage_key,
            private=self.private,
            options=self.options,
        )
        tool.name = self.name
        tool.description = self.description
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool
