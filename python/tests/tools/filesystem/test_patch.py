# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

pytest.importorskip("patch_ng", reason="Optional module [filesystem] not installed.")

from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.filesystem.patch import FilePatchTool


@pytest.fixture
def tool() -> FilePatchTool:
    return FilePatchTool()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_single_occurrence(tool: FilePatchTool, tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("x = 1\ny = 'hello'\n")
    result = await tool.run(
        {"mode": "replace", "path": str(target), "old": "hello", "new": "world", "expected_occurrences": 1}
    )
    data = result.to_json_safe()
    assert data["mode"] == "replace"
    assert target.read_text() == "x = 1\ny = 'world'\n"
    assert "-y = 'hello'" in data["diff"]
    assert "+y = 'world'" in data["diff"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_zero_occurrences_fails(tool: FilePatchTool, tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    with pytest.raises(ToolError, match="found 0"):
        await tool.run(
            {"mode": "replace", "path": str(target), "old": "hello", "new": "world", "expected_occurrences": 1}
        )
    assert target.read_text() == "x = 1\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_multiple_occurrences_fails(tool: FilePatchTool, tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("a\na\n")
    with pytest.raises(ToolError, match="found 2"):
        await tool.run({"mode": "replace", "path": str(target), "old": "a", "new": "b", "expected_occurrences": 1})
    assert target.read_text() == "a\na\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replace_requires_absolute_path(tool: FilePatchTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run({"mode": "replace", "path": "relative.txt", "old": "a", "new": "b"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_patch_unified_diff_round_trip(tool: FilePatchTool, tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("one\ntwo\nthree\n")

    diff = "--- a/hello.txt\n+++ b/hello.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
    result = await tool.run({"mode": "patch", "diff": diff, "root": str(tmp_path)})
    data = result.to_json_safe()
    assert data["mode"] == "patch"
    assert target.read_text() == "one\nTWO\nthree\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_patch_mode_requires_diff(tool: FilePatchTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run({"mode": "patch"})
