# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from beeai_framework.tools.filesystem import GrepTool


@pytest.fixture
def tool() -> GrepTool:
    return GrepTool()


@pytest.fixture
def fs_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\ny = 'hello world'\nz = 3\n")
    (tmp_path / "b.py").write_text("greeting = 'Hello'\n")
    (tmp_path / "c.txt").write_text("hello again\n")
    return tmp_path


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ripgrep_path(tool: GrepTool, fs_tree: Path) -> None:
    pytest.importorskip("shutil")
    import shutil

    if not shutil.which("rg"):
        pytest.skip("ripgrep not installed on this machine")
    result = await tool.run({"pattern": "hello", "root": str(fs_tree)})
    data = result.to_json_safe()
    assert data["used_ripgrep"] is True
    assert data["matches"]
    assert any("a.py" in m["path"] for m in data["matches"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stdlib_fallback(tool: GrepTool, fs_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import beeai_framework.tools.filesystem.grep_tool as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _name: None)
    result = await tool.run({"pattern": "hello", "root": str(fs_tree)})
    data = result.to_json_safe()
    assert data["used_ripgrep"] is False
    matched_paths = {Path(m["path"]).name for m in data["matches"]}
    assert "a.py" in matched_paths
    assert "c.txt" in matched_paths


@pytest.mark.unit
@pytest.mark.asyncio
async def test_case_insensitive_stdlib(tool: GrepTool, fs_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import beeai_framework.tools.filesystem.grep_tool as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _name: None)
    result = await tool.run({"pattern": "hello", "root": str(fs_tree), "case_sensitive": False})
    matched_paths = {Path(m["path"]).name for m in result.to_json_safe()["matches"]}
    assert "b.py" in matched_paths  # matched "Hello" with case-insensitive


@pytest.mark.unit
@pytest.mark.asyncio
async def test_max_results_truncates_stdlib(tool: GrepTool, fs_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import beeai_framework.tools.filesystem.grep_tool as grep_mod

    monkeypatch.setattr(grep_mod.shutil, "which", lambda _name: None)
    result = await tool.run({"pattern": "hello", "root": str(fs_tree), "max_results": 1})
    data = result.to_json_safe()
    assert len(data["matches"]) == 1
    assert data["truncated"] is True
