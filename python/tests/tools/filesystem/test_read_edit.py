# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.filesystem import (
    FileBackend,
    FileEditTool,
    FileReadTool,
    setup_file_backend,
)


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
