# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.code._shell_backend import get_shell_backend
from beeai_framework.tools.errors import ToolInputValidationError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class ShellToolInput(BaseModel):
    command: list[str] = Field(
        description=(
            "Command and arguments as a list (no shell interpretation). "
            "Example: ['git', 'status', '--porcelain']. Use shlex.split on a user string if needed."
        )
    )
    cwd: str | None = Field(default=None, description="Working directory.")
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables merged over the current environment."
    )
    timeout_seconds: float | None = Field(
        default=60.0, description="Kill the process if it runs longer. Set to None to disable."
    )
    input_text: str | None = Field(
        default=None,
        description=(
            "Optional text piped to the process on stdin. Some backends (e.g. editor terminals under ACP) "
            "do not support stdin and will ignore this field."
        ),
    )


class ShellTool(Tool[ShellToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Run a command and capture its output.

    Delegates to whichever `ShellBackend` is installed in the current `ContextVar`.
    The default backend runs a local subprocess; the ACP Zed adapter installs a
    backend that routes through the editor's terminal widget (`terminal/*` methods)
    so output renders live in the UI. The tool itself doesn't know the difference —
    protocol-awareness is the backend's job.
    """

    name = "shell"
    description = (
        "Run a command and return its exit code, stdout, and stderr. The command must be a list of "
        "arguments; it is NOT run through a shell."
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

        result = await get_shell_backend().run(
            command=input.command,
            cwd=input.cwd,
            env=input.env,
            timeout_seconds=input.timeout_seconds,
            input_text=input.input_text,
        )
        return JSONToolOutput(dict(result))
