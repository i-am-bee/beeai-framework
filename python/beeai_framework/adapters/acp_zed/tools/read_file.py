# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import StringToolOutput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

if TYPE_CHECKING:
    from beeai_framework.adapters.acp_zed.serve.server import ACPZedServer


class ACPZedReadFileInput(BaseModel):
    path: str = Field(description="Absolute path to the file to read.")
    line: int | None = Field(default=None, description="Optional 1-based line number to start reading from.")
    limit: int | None = Field(default=None, description="Optional maximum number of lines to read.")


class ACPZedReadFileTool(Tool[ACPZedReadFileInput, ToolRunOptions, StringToolOutput]):
    """Read a text file via the ACP client (`fs/read_text_file`).

    Routes file access through the editor so its unsaved-buffer view wins and the user
    sees the read in the UI. Requires the client to advertise `fs.read_text_file` in
    `initialize`.
    """

    name = "acp_zed_read_file"
    description = "Read a text file from the user's workspace via the ACP client. The `path` must be absolute."
    input_schema = ACPZedReadFileInput

    def __init__(self, server: ACPZedServer[Any], options: dict[str, Any] | None = None) -> None:
        super().__init__(options=options)
        self._server = server

    async def clone(self) -> Self:
        return type(self)(server=self._server, options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "acp_zed", "read_file"], creator=self)

    async def _run(
        self, input: ACPZedReadFileInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        if not Path(input.path).is_absolute():
            raise ToolInputValidationError(f"`path` must be absolute, got: {input.path!r}")
        bridge = self._server.bridge
        if not bridge.can_read:
            raise ToolError("ACP client does not advertise fs.read_text_file capability")
        content = await bridge.read_text_file(input.path, line=input.line, limit=input.limit)
        return StringToolOutput(content)
