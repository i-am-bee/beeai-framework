# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

try:
    from acp.schema import EnvVariable
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp_zed] not found.\nRun 'pip install \"beeai-framework[acp-zed]\"' to install."
    ) from e

if TYPE_CHECKING:
    from beeai_framework.adapters.acp_zed.serve.server import ACPZedServer


class ACPZedTerminalToolInput(BaseModel):
    command: str = Field(description="Program to run (e.g. 'pytest', 'npm').")
    args: list[str] = Field(default_factory=list, description="Arguments passed to the program.")
    cwd: str | None = Field(default=None, description="Working directory (absolute path).")
    env: dict[str, str] | None = Field(default=None, description="Extra environment variables.")
    output_byte_limit: int | None = Field(default=None, description="Cap captured output in bytes. Editor-enforced.")
    wait: bool = Field(
        default=True,
        description=(
            "Wait for the process to exit and return its output. If false, returns the terminal id immediately."
        ),
    )


class ACPZedTerminalTool(Tool[ACPZedTerminalToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Run a command in the editor's terminal via ACP `terminal/*`.

    Live output shows in the Zed terminal widget while the process is running; the
    agent blocks on exit (by default) and gets back `{exit_code, stdout, stderr,
    terminal_id}`. Requires the client to advertise `terminal` capability.
    """

    name = "acp_zed_terminal"
    description = (
        "Run a command in the user's editor terminal. Returns exit code, stdout, and stderr after the "
        "process exits. Prefer this over generic shell when running under Zed."
    )
    input_schema = ACPZedTerminalToolInput

    def __init__(self, server: ACPZedServer[Any], options: dict[str, Any] | None = None) -> None:
        super().__init__(options=options)
        self._server = server

    async def clone(self) -> Self:
        return type(self)(server=self._server, options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "acp_zed", "terminal"], creator=self)

    async def _run(
        self, input: ACPZedTerminalToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        bridge = self._server.bridge
        if not bridge.can_terminal:
            raise ToolError("ACP client does not advertise terminal capability")

        env_list = [EnvVariable(name=k, value=v) for k, v in (input.env or {}).items()] or None
        terminal_id = await bridge.create_terminal(
            input.command,
            args=input.args,
            cwd=input.cwd,
            env=env_list,
            output_byte_limit=input.output_byte_limit,
        )

        if not input.wait:
            return JSONToolOutput({"terminal_id": terminal_id, "waited": False})

        try:
            exit_info = await bridge.wait_for_terminal_exit(terminal_id)
            output = await bridge.terminal_output(terminal_id)
        finally:
            await bridge.release_terminal(terminal_id)

        return JSONToolOutput(
            {
                "terminal_id": terminal_id,
                "exit_code": getattr(exit_info, "exit_code", None),
                "signal": getattr(exit_info, "signal", None),
                "stdout": getattr(output, "output", "") or "",
                "truncated": getattr(output, "truncated", False),
                "waited": True,
            }
        )
