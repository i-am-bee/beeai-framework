"""Load long procedures on demand instead of keeping them in the prompt.

A skill is a directory holding a ``SKILL.md`` file: YAML frontmatter with a
``name`` and a one-line ``description``, followed by the instruction body.

``LoadSkillTool`` builds a catalog of names and descriptions into its own tool
description — that is the only part the model pays for on every call — and
returns a body only when the agent asks for it by name. Agents that carry large,
branch-specific instruction blocks can therefore keep a lean prompt.
"""

import asyncio
import sys
import traceback
from pathlib import Path

from pydantic import BaseModel, Field

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import StringToolOutput, Tool, ToolError, ToolRunOptions

SKILLS_DIR = Path(__file__).parent / "assets" / "skills"


class LoadSkillInput(BaseModel):
    name: str | None = Field(
        default=None,
        description="Name of the skill to load. Omit to list the available skills with their descriptions.",
    )


class Skill(BaseModel):
    name: str
    description: str
    path: Path


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``---``-delimited frontmatter block from a skill body.

    Understands `key: value` pairs and the `>` / `|` block scalars used by the
    Agent Skills format. A production loader would use a YAML parser instead.
    """
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---", 4)
    if end == -1:
        return {}, text

    header: dict[str, str] = {}
    key: str | None = None
    block: list[str] = []

    def flush() -> None:
        nonlocal key
        if key is not None:
            header[key] = " ".join(part for part in block if part).strip()
            key = None

    for line in text[4:end].splitlines():
        if key is not None:
            if line.startswith((" ", "\t")):
                block.append(line.strip())
                continue
            flush()

        entry = line.strip()
        if not entry or entry.startswith("#") or ":" not in entry:
            continue

        field, _, value = entry.partition(":")
        value = value.strip()
        if value in {">", ">-", "|", "|-"}:
            key = field.strip()
            block = []
        elif value:
            header[field.strip()] = value
    flush()

    body = text[end + len("\n---") :].lstrip("\n")
    return header, body


def load_skills(directory: Path) -> list[Skill]:
    """Load every ``<name>/SKILL.md`` under *directory*.

    The frontmatter ``name`` must match the directory name, as the Agent Skills
    standard requires.
    """
    skills: list[Skill] = []
    for path in sorted(directory.glob("*/SKILL.md")):
        header, _ = split_frontmatter(path.read_text(encoding="utf-8"))
        name = header.get("name", "").strip()
        description = header.get("description", "").strip()
        if not name or not description:
            raise ValueError(f"{path} needs 'name' and 'description' frontmatter")
        if name != path.parent.name:
            raise ValueError(f"{path} declares name {name!r}, expected {path.parent.name!r}")
        skills.append(Skill(name=name, description=description, path=path))
    return skills


class LoadSkillTool(Tool[LoadSkillInput, ToolRunOptions, StringToolOutput]):
    """Expose a skills directory; only names and descriptions stay in context."""

    name = "load_skill"
    description = ""
    input_schema = LoadSkillInput

    def __init__(self, skills_dir: Path, *, tool_name: str = "load_skill") -> None:
        super().__init__()
        directory = Path(skills_dir)
        if not directory.is_dir():
            raise ValueError(f"Skills directory does not exist: {directory}")
        self.name = tool_name
        self._skills = {skill.name: skill for skill in load_skills(directory)}
        self.description = self._build_catalog()

    def _build_catalog(self) -> str:
        if not self._skills:
            return "Load a named skill. No skills are currently available."
        lines = [
            "Load a detailed skill on demand. Each skill holds a procedure that is too long",
            "to keep in the prompt; call this with `name` set to one of the following:",
            "",
        ]
        lines.extend(f"- {skill.name}: {skill.description}" for skill in self._skills.values())
        return "\n".join(lines)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "load_skill"],
            creator=self,
        )

    async def _run(
        self, input: LoadSkillInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        if input.name is None:
            return StringToolOutput(result=self._build_catalog())

        skill = self._skills.get(input.name)
        if skill is None:
            raise ToolError(f"Unknown skill {input.name!r}. Available: {', '.join(self._skills) or 'none'}")

        _, body = split_frontmatter(skill.path.read_text(encoding="utf-8"))
        return StringToolOutput(result=body)


async def main() -> None:
    tool = LoadSkillTool(SKILLS_DIR)

    # What every request pays for: the catalog, not the procedures.
    print(f"Tool description:\n\n{tool.description}\n")

    # What the agent receives once it asks for a skill: the body only.
    output = await tool.run(input=LoadSkillInput(name="release-notes")).middleware(
        GlobalTrajectoryMiddleware(pretty=True)
    )
    print(f"Release notes skill body:\n\n{output.get_text_content()}")

    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[tool],
        instructions=["When the task matches a skill listed by `load_skill`, load it and follow it."],
    )

    prompt = (
        "Write the release notes for version 2.0.0 from these merged changes:\n"
        "- added a --json flag to the report command\n"
        "- fixed a crash when the config file is empty\n"
        "- removed the deprecated --legacy flag\n"
        "- refactored the HTTP client internals, no behavior change"
    )
    print(f"User: {prompt}")
    response = await agent.run(prompt).middleware(GlobalTrajectoryMiddleware(pretty=True))
    print("Agent 🤖 : ", response.last_message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
