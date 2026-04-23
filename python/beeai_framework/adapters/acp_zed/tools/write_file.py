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


class ACPZedWriteFileInput(BaseModel):
    path: str = Field(description="Absolute path to the file to write.")
    content: str = Field(description="Full new text content of the file.")


class ACPZedWriteFileTool(Tool[ACPZedWriteFileInput, ToolRunOptions, StringToolOutput]):
    """Write a text file via the ACP client (`fs/write_text_file`).

    The editor receives the write and can render it as a diff; the agent never touches
    the local filesystem directly. Requires `fs.write_text_file` capability.
    """

    name = "acp_zed_write_file"
    description = (
        "Write a text file in the user's workspace via the ACP client. The `path` must be absolute. "
        "Overwrites existing files."
    )
    input_schema = ACPZedWriteFileInput

    def __init__(self, server: ACPZedServer[Any], options: dict[str, Any] | None = None) -> None:
        super().__init__(options=options)
        self._server = server

    async def clone(self) -> Self:
        return type(self)(server=self._server, options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "acp_zed", "write_file"], creator=self)

    async def _run(
        self, input: ACPZedWriteFileInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        if not Path(input.path).is_absolute():
            raise ToolInputValidationError(f"`path` must be absolute, got: {input.path!r}")
        bridge = self._server.bridge
        if not bridge.can_write:
            raise ToolError("ACP client does not advertise fs.write_text_file capability")
        await bridge.write_text_file(input.path, input.content)
        return StringToolOutput(f"Wrote {len(input.content)} bytes to {input.path}")
