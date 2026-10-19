# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""ShellBackend ContextVar dispatch — same shape as `beeai_framework/utils/io.py`.

`ShellTool` calls `await get_shell_backend().run(...)` and doesn't care whether the
command ends up in a local `subprocess` or is routed through an editor terminal via
ACP's `terminal/*` methods. Serve adapters install their own backend with
`setup_shell_backend(...)` for the duration of a turn; outside of that scope the
local subprocess default applies.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from typing import Protocol, TypedDict

from beeai_framework.tools.errors import ToolError


class ShellResult(TypedDict, total=False):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class ShellBackend(Protocol):
    async def run(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        input_text: str | None = None,
    ) -> ShellResult: ...


class _SessionOptions(TypedDict, total=False):
    start_new_session: bool


_USE_PROCESS_GROUP = os.name == "posix"


def _kill_process(process: asyncio.subprocess.Process) -> None:
    if _USE_PROCESS_GROUP:
        # Descendants may keep pipes open after the direct child has exited.
        # macOS reports EPERM for a group whose only member is an unreaped zombie.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        with suppress(ProcessLookupError):
            process.kill()


class LocalShellBackend(ShellBackend):
    """Run a local command and capture stdout and stderr separately.

    On POSIX each command runs in its own session, so terminal signals such as
    Ctrl+C do not reach it directly. Timeout or cancellation immediately sends
    SIGKILL to its process group; descendants that leave that group are not
    covered. On Windows only the direct child is killed.
    """

    async def run(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        input_text: str | None = None,
    ) -> ShellResult:
        merged_env = {**os.environ, **env} if env else None
        session_options: _SessionOptions = {"start_new_session": True} if _USE_PROCESS_GROUP else {}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=merged_env,
                stdin=asyncio.subprocess.PIPE if input_text is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **session_options,
            )
        except FileNotFoundError as e:
            raise ToolError(f"Command not found: {command[0]!r}") from e

        stdin_bytes = input_text.encode() if input_text is not None else None
        timed_out = False
        try:
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input=stdin_bytes), timeout=timeout_seconds
                )
            except TimeoutError:
                _kill_process(process)
                stdout_bytes, stderr_bytes = await process.communicate()
                timed_out = True
        except asyncio.CancelledError:
            # Cancelling communicate() does not terminate the child process.
            _kill_process(process)
            # Keep draining/reaping even if cancellation is requested again.
            cleanup = asyncio.create_task(process.communicate())
            while not cleanup.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(cleanup)
            cleanup.result()
            raise

        return ShellResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            timed_out=timed_out,
        )


_storage: ContextVar[ShellBackend] = ContextVar("shell_backend")


def get_shell_backend() -> ShellBackend:
    return _storage.get(LocalShellBackend())


def setup_shell_backend(backend: ShellBackend) -> Callable[[], None]:
    """Install `backend` as the active shell backend. Returns a cleanup callable."""
    token = _storage.set(backend)
    return lambda: _storage.reset(token)


__all__ = [
    "LocalShellBackend",
    "ShellBackend",
    "ShellResult",
    "get_shell_backend",
    "setup_shell_backend",
]
