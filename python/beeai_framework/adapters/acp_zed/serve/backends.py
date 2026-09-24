# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""ACP-routed shell + file backends.

When the Zed ACP adapter is serving a turn, the generic `ShellTool`, `FileReadTool`,
and `FileEditTool` automatically route through the editor: shell commands open a
terminal in Zed's UI, file I/O goes through `fs/read_text_file` / `fs/write_text_file`
so unsaved buffers are respected. The tools themselves don't change — they just see
a different backend in the `ContextVar`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

try:
    from acp.schema import EnvVariable
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp_zed] not found.\nRun 'pip install \"beeai-framework[acp-zed]\"' to install."
    ) from e

from beeai_framework.tools.code import ShellBackend, ShellResult
from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.filesystem import FileBackend

if TYPE_CHECKING:
    from beeai_framework.adapters.acp_zed.serve.agent import FsBridge


class ACPFileBackend(FileBackend):
    """Route reads + writes through ACP `fs/read_text_file` / `fs/write_text_file`."""

    def __init__(self, bridge: FsBridge) -> None:
        self._bridge = bridge

    async def read_text(self, path: str, *, line: int | None = None, limit: int | None = None) -> str:
        if not self._bridge.can_read:
            raise ToolError("ACP client does not advertise fs.read_text_file capability")
        return await self._bridge.read_text_file(path, line=line, limit=limit)

    async def write_text(self, path: str, content: str) -> None:
        if not self._bridge.can_write:
            raise ToolError("ACP client does not advertise fs.write_text_file capability")
        await self._bridge.write_text_file(path, content)


class ACPShellBackend(ShellBackend):
    """Run commands in the editor's terminal via ACP `terminal/*`.

    `input_text` (stdin) is not exposed by the ACP terminal protocol; passing it
    raises `ToolError` rather than silently dropping it. `timeout_seconds` is
    enforced by wrapping `wait_for_terminal_exit` in `asyncio.wait_for` and calling
    `kill_terminal` + `release_terminal` on timeout.
    """

    def __init__(self, bridge: FsBridge) -> None:
        self._bridge = bridge

    async def run(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        input_text: str | None = None,
    ) -> ShellResult:
        if not self._bridge.can_terminal:
            raise ToolError("ACP client does not advertise terminal capability")
        if input_text is not None:
            raise ToolError("ACP terminal backend does not support stdin (input_text)")

        env_list = [EnvVariable(name=k, value=v) for k, v in (env or {}).items()] or None
        program, *args = command
        terminal_id = await self._bridge.create_terminal(program, args=args, cwd=cwd, env=env_list)

        timed_out = False
        try:
            try:
                exit_info = await asyncio.wait_for(
                    self._bridge.wait_for_terminal_exit(terminal_id), timeout=timeout_seconds
                )
            except TimeoutError:
                timed_out = True
                with contextlib.suppress(Exception):
                    await self._bridge.kill_terminal(terminal_id)
                exit_info = None
            output = await self._bridge.terminal_output(terminal_id)
        finally:
            with contextlib.suppress(Exception):
                await self._bridge.release_terminal(terminal_id)

        return ShellResult(
            exit_code=getattr(exit_info, "exit_code", -1) if exit_info is not None else -1,
            stdout=getattr(output, "output", "") or "",
            stderr="",  # ACP conflates streams into `output`
            timed_out=timed_out,
            truncated=bool(getattr(output, "truncated", False)),
        )


__all__ = ["ACPFileBackend", "ACPShellBackend"]
