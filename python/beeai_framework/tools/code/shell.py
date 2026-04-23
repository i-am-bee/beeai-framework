# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from typing import Any, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class ShellToolInput(BaseModel):
    command: list[str] = Field(
        description=(
            "Command and arguments as a list (no shell interpretation). "
            "Example: ['git', 'status', '--porcelain']. Use shlex.split on a user string if needed."
        )
    )
    cwd: str | None = Field(default=None, description="Working directory. Defaults to the current process cwd.")
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables merged over the current environment."
    )
    timeout_seconds: float | None = Field(
        default=60.0, description="Kill the process if it runs longer. Set to None to disable."
    )
    input_text: str | None = Field(default=None, description="Optional text piped to the process on stdin.")


class ShellTool(Tool[ShellToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Run a local subprocess and collect its output.

    Takes the command as a list of arguments (no shell interpretation), so callers
    don't have to worry about quoting or injection. Captures stdout, stderr, exit
    code, and a `timed_out` flag. Under Zed's ACP adapter, prefer
    `ACPZedTerminalTool` — it routes through the editor's terminal widget and shows
    live output; this tool is for harnesses without a client-side terminal.
    """

    name = "shell"
    description = (
        "Run a local command as a subprocess and return its exit code, stdout, and stderr. "
        "The command must be a list of arguments; it is NOT run through a shell."
    )
    input_schema = ShellToolInput

    async def clone(self) -> Self:
        return type(self)(options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "code", "shell"], creator=self)

    async def _run(
        self, input: ShellToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        if not input.command:
            raise ToolInputValidationError("`command` must be a non-empty list")

        merged_env = {**os.environ, **(input.env or {})} if input.env else None
        try:
            process = await asyncio.create_subprocess_exec(
                *input.command,
                cwd=input.cwd,
                env=merged_env,
                stdin=asyncio.subprocess.PIPE if input.input_text is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise ToolError(f"Command not found: {input.command[0]!r}") from e

        stdin_bytes = input.input_text.encode() if input.input_text is not None else None
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=stdin_bytes), timeout=input.timeout_seconds
            )
        except TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True

        return JSONToolOutput(
            {
                "exit_code": process.returncode if process.returncode is not None else -1,
                "stdout": stdout_bytes.decode(errors="replace"),
                "stderr": stderr_bytes.decode(errors="replace"),
                "timed_out": timed_out,
            }
        )
