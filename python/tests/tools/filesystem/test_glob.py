# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from beeai_framework.tools.filesystem import GlobTool


@pytest.fixture
def tool() -> GlobTool:
    return GlobTool()


@pytest.fixture
def fs_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")
    (tmp_path / ".hidden.py").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("")
    (tmp_path / ".hiddendir").mkdir()
    (tmp_path / ".hiddendir" / "e.py").write_text("")
    return tmp_path


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recursive_glob(tool: GlobTool, fs_tree: Path) -> None:
    result = await tool.run({"pattern": "**/*.py", "root": str(fs_tree)})
    names = sorted(Path(p).name for p in result.to_json_safe()["matches"])
    assert names == ["a.py", "b.py", "d.py"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hidden_files_excluded_by_default(tool: GlobTool, fs_tree: Path) -> None:
    result = await tool.run({"pattern": "**/*.py", "root": str(fs_tree)})
    for p in result.to_json_safe()["matches"]:
        assert not any(part.startswith(".") for part in Path(p).relative_to(fs_tree).parts)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hidden_files_opt_in(tool: GlobTool, fs_tree: Path) -> None:
    result = await tool.run({"pattern": "**/*.py", "root": str(fs_tree), "include_hidden": True})
    names = {Path(p).name for p in result.to_json_safe()["matches"]}
    assert ".hidden.py" in names
    assert "e.py" in names


@pytest.mark.unit
@pytest.mark.asyncio
async def test_limit_truncates(tool: GlobTool, fs_tree: Path) -> None:
    result = await tool.run({"pattern": "**/*", "root": str(fs_tree), "limit": 1})
    data = result.to_json_safe()
    assert len(data["matches"]) == 1
    assert data["truncated"] is True
