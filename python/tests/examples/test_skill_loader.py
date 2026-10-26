# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from beeai_framework.tools import ToolError
from examples.tools.skill_loader import SKILLS_DIR, LoadSkillInput, LoadSkillTool, split_frontmatter

pytestmark = pytest.mark.unit


def _write_skill(root: Path, name: str, description: str, body: str, *, declared_name: str | None = None) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {declared_name or name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_catalog_lists_names_and_descriptions_without_bodies() -> None:
    tool = LoadSkillTool(SKILLS_DIR)

    assert "- release-notes:" in tool.description
    assert "- commit-message:" in tool.description
    assert "# Writing release notes" not in tool.description


def test_ignores_directories_without_skill_md(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha", "Use when testing alpha.", "# Alpha")
    (tmp_path / "notes.txt").write_text("not a skill", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    tool = LoadSkillTool(tmp_path)

    assert "alpha" in tool.description
    assert "notes" not in tool.description
    assert "empty" not in tool.description


def test_frontmatter_name_must_match_directory(tmp_path: Path) -> None:
    _write_skill(tmp_path, "mismatch", "desc", "body", declared_name="other")

    with pytest.raises(ValueError):
        LoadSkillTool(tmp_path)


def test_split_frontmatter() -> None:
    header, body = split_frontmatter("---\nname: x\ndescription: >-\n  first\n  second\n---\n\nHello\n")

    assert header == {"name": "x", "description": "first second"}
    assert body == "Hello\n"


def test_split_frontmatter_without_header() -> None:
    assert split_frontmatter("Just text\n") == ({}, "Just text\n")


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LoadSkillTool(tmp_path / "does-not-exist")


@pytest.mark.asyncio
async def test_loads_body_without_frontmatter() -> None:
    tool = LoadSkillTool(SKILLS_DIR)

    output = await tool.run(input=LoadSkillInput(name="release-notes"))

    assert output.get_text_content().startswith("# Writing release notes")


@pytest.mark.asyncio
async def test_omitting_name_lists_skills() -> None:
    tool = LoadSkillTool(SKILLS_DIR)

    output = await tool.run(input=LoadSkillInput())

    assert "release-notes" in output.get_text_content()
    assert "commit-message" in output.get_text_content()


@pytest.mark.asyncio
async def test_unknown_skill_raises() -> None:
    tool = LoadSkillTool(SKILLS_DIR)

    with pytest.raises(ToolError):
        await tool.run(input=LoadSkillInput(name="does-not-exist"))
