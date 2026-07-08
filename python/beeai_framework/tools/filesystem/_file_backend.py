# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""FileBackend ContextVar dispatch.

Generic file tools (`FileReadTool`, `FileEditTool`) call `get_file_backend()` and
don't care whether the operation touches local disk or routes through an editor
via ACP's `fs/read_text_file` + `fs/write_text_file`. Serve adapters install
their own backend with `setup_file_backend(...)` for the duration of a turn.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Protocol


class FileBackend(Protocol):
    async def read_text(self, path: str, *, line: int | None = None, limit: int | None = None) -> str: ...
    async def write_text(self, path: str, content: str) -> None: ...


class LocalFileBackend(FileBackend):
    """Reads / writes from the local filesystem. Path must be absolute."""

    async def read_text(self, path: str, *, line: int | None = None, limit: int | None = None) -> str:
        content = Path(path).read_text()
        if line is None and limit is None:
            return content
        lines = content.splitlines(keepends=True)
        start = max(0, (line or 1) - 1)
        end = start + limit if limit is not None else len(lines)
        return "".join(lines[start:end])

    async def write_text(self, path: str, content: str) -> None:
        Path(path).write_text(content)


_storage: ContextVar[FileBackend] = ContextVar("file_backend")
_storage.set(LocalFileBackend())


def get_file_backend() -> FileBackend:
    return _storage.get()


def setup_file_backend(backend: FileBackend) -> Callable[[], None]:
    """Install `backend` as the active file backend. Returns a cleanup callable."""
    token = _storage.set(backend)
    return lambda: _storage.reset(token)


__all__ = [
    "FileBackend",
    "LocalFileBackend",
    "get_file_backend",
    "setup_file_backend",
]
