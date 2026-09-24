# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextvars import Context
from pathlib import Path
from typing import Any

import pytest

from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.filesystem import (
    FileBackend,
    FileEditTool,
    FileReadTool,
    LocalFileBackend,
    get_file_backend,
    setup_file_backend,
)


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["read", "overwrite", "replace"])
def test_local_default_in_new_context(tmp_path: Path, operation: str) -> None:
    target = tmp_path / "a.txt"
    target.write_text("before\n")

    async def run() -> None:
        if operation == "read":
            result = await FileReadTool().run({"path": str(target)})
            assert result.get_text_content() == "before\n"
        else:
            arguments: dict[str, Any] = {"path": str(target), "mode": operation}
            arguments.update({"content": "after\n"} if operation == "overwrite" else {"old": "before", "new": "after"})
            await FileEditTool().run(arguments)
            assert target.read_text() == "after\n"

    Context().run(asyncio.run, run())


@pytest.mark.unit
def test_backend_cleanup_in_new_context(tmp_path: Path) -> None:
    context = Context()
    backend = LocalFileBackend()
    nested_backend = LocalFileBackend()
    cleanup = context.run(setup_file_backend, backend)
    try:
        assert context.run(get_file_backend) is backend
        assert Context().run(get_file_backend) is not backend
        nested_cleanup = context.run(setup_file_backend, nested_backend)
        try:
            assert context.run(get_file_backend) is nested_backend
        finally:
            context.run(nested_cleanup)
        assert context.run(get_file_backend) is backend
    finally:
        context.run(cleanup)

    target = tmp_path / "a.txt"
    target.write_text("restored\n")

    async def read() -> None:
        result = await FileReadTool().run({"path": str(target)})
        assert result.get_text_content() == "restored\n"

    context.run(asyncio.run, read())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_local_default(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("line1\nline2\nline3\n")
    result = await FileReadTool().run({"path": str(target)})
    assert result.get_text_content() == "line1\nline2\nline3\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_line_and_limit(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("a\nb\nc\nd\n")
    result = await FileReadTool().run({"path": str(target), "line": 2, "limit": 2})
    assert result.get_text_content() == "b\nc\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_requires_absolute_path() -> None:
    with pytest.raises(ToolInputValidationError):
        await FileReadTool().run({"path": "relative.txt"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="File not found"):
        await FileReadTool().run({"path": str(tmp_path / "missing.txt")})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edit_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old content\n")
    result = await FileEditTool().run({"mode": "overwrite", "path": str(target), "content": "new content\n"})
    assert target.read_text() == "new content\n"
    assert result.to_json_safe()["mode"] == "overwrite"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edit_replace_single_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("x = 1\ny = 'hello'\n")
    result = await FileEditTool().run(
        {"mode": "replace", "path": str(target), "old": "hello", "new": "world", "expected_occurrences": 1}
    )
    assert target.read_text() == "x = 1\ny = 'world'\n"
    diff = result.to_json_safe()["diff"]
    assert "-y = 'hello'" in diff
    assert "+y = 'world'" in diff


@pytest.mark.unit
@pytest.mark.asyncio
async def test_edit_replace_occurrence_mismatch_fails(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("a\na\n")
    with pytest.raises(ToolError, match="found 2"):
        await FileEditTool().run(
            {"mode": "replace", "path": str(target), "old": "a", "new": "b", "expected_occurrences": 1}
        )
    assert target.read_text() == "a\na\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tools_route_through_installed_backend() -> None:
    """Swap in a stub backend via `setup_file_backend` and confirm both tools use it."""

    class StubBackend(FileBackend):
        def __init__(self) -> None:
            self.reads: list[dict[str, Any]] = []
            self.writes: list[dict[str, Any]] = []
            self.store: dict[str, str] = {"/tmp/virtual.txt": "hello\n"}

        async def read_text(self, path: str, *, line: int | None = None, limit: int | None = None) -> str:
            self.reads.append({"path": path, "line": line, "limit": limit})
            return self.store.get(path, "")

        async def write_text(self, path: str, content: str) -> None:
            self.writes.append({"path": path, "content": content})
            self.store[path] = content

    stub = StubBackend()
    cleanup = setup_file_backend(stub)
    try:
        read = await FileReadTool().run({"path": "/tmp/virtual.txt"})
        assert read.get_text_content() == "hello\n"
        assert stub.reads == [{"path": "/tmp/virtual.txt", "line": None, "limit": None}]

        await FileEditTool().run({"mode": "replace", "path": "/tmp/virtual.txt", "old": "hello", "new": "world"})
        assert stub.writes == [{"path": "/tmp/virtual.txt", "content": "world\n"}]
    finally:
        cleanup()
