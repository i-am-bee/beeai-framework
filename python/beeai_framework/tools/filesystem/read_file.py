# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import StringToolOutput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.filesystem._file_backend import get_file_backend
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class FileReadToolInput(BaseModel):
    path: str = Field(description="Absolute path to the file to read.")
    line: int | None = Field(default=None, description="Optional 1-based line to start reading from.")
    limit: int | None = Field(default=None, description="Optional maximum number of lines to read.")


class FileReadTool(Tool[FileReadToolInput, ToolRunOptions, StringToolOutput]):
    """Read a text file via the active `FileBackend`.

    Default: local disk via `pathlib`. Under the ACP Zed adapter: routed through
    `fs/read_text_file` so the editor's unsaved buffers are respected.
    """

    name = "read_file"
    description = "Read a text file. The `path` must be absolute."
    input_schema = FileReadToolInput

    async def clone(self) -> Self:
        return type(self)(options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "filesystem", "read_file"], creator=self)

    async def _run(
        self, input: FileReadToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        if not Path(input.path).is_absolute():
            raise ToolInputValidationError(f"`path` must be absolute, got: {input.path!r}")
        try:
            content = await get_file_backend().read_text(input.path, line=input.line, limit=input.limit)
        except FileNotFoundError as e:
            raise ToolError(f"File not found: {input.path}") from e
        return StringToolOutput(content)
