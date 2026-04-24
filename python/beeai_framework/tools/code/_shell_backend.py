# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""ShellBackend ContextVar dispatch — same shape as `beeai_framework/utils/io.py`.

The `ShellTool` calls `await get_shell_backend().run(...)` and doesn't care whether
the command ends up in a local `subprocess` or is routed through an editor terminal
via ACP's `terminal/*` methods. Serve adapters install their own backend with
`setup_shell_backend(...)` for the duration of a turn; outside of that scope the
local subprocess default applies.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Protocol, TypedDict


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
        output_byte_limit: int | None = None,
    ) -> ShellResult: ...


class LocalShellBackend(ShellBackend):
    """Runs the command as a local subprocess. Captures stdout + stderr separately."""

    async def run(
        self,
        *,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        input_text: str | None = None,
        output_byte_limit: int | None = None,
    ) -> ShellResult:
        merged_env = {**os.environ, **env} if env else None
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=merged_env,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdin_bytes = input_text.encode() if input_text is not None else None
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=stdin_bytes), timeout=timeout_seconds
            )
        except TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True

        return ShellResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            timed_out=timed_out,
        )


_storage: ContextVar[ShellBackend] = ContextVar("shell_backend")
_storage.set(LocalShellBackend())


def get_shell_backend() -> ShellBackend:
    return _storage.get()


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


# Re-export `Any` in case downstream adapters want to expose richer result shapes.
_ = Any  # keep linters happy with the unused typing import in the stub signature.
