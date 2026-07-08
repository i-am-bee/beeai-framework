# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class GlobToolInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.")
    root: str = Field(default=".", description="Directory to search from.")
    include_hidden: bool = Field(default=False, description="Include entries whose name starts with a dot.")
    limit: int = Field(default=1000, description="Maximum number of paths to return.")


class GlobTool(Tool[GlobToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Find paths matching a glob pattern under a root directory.

    Uses `pathlib.Path.glob` — patterns like `**/*.py` work out of the box. Hidden
    files/directories (names starting with `.`) are excluded by default.
    """

    name = "glob"
    description = "List files and directories matching a glob pattern (e.g. '**/*.py')."
    input_schema = GlobToolInput

    async def clone(self) -> Self:
        return type(self)(options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "filesystem", "glob"], creator=self)

    async def _run(
        self, input: GlobToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        root = Path(input.root).expanduser().resolve()
        matches: list[str] = []
        truncated = False
        for path in root.glob(input.pattern):
            if not input.include_hidden and any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            matches.append(str(path))
            if len(matches) >= input.limit:
                truncated = True
                break
        return JSONToolOutput({"matches": matches, "truncated": truncated})
