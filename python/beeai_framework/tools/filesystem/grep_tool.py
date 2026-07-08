# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

_DEFAULT_SKIP_DIRS = frozenset({".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox"})


class GrepToolInput(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    root: str = Field(default=".", description="Directory to search from.")
    glob: str | None = Field(default=None, description="Optional glob filter (e.g. '*.py').")
    case_sensitive: bool = Field(default=True, description="Match case. Disable for case-insensitive search.")
    max_results: int = Field(default=500, description="Cap total matches returned.")
    context_lines: int = Field(default=0, description="Lines of context to include before and after each match.")


class GrepTool(Tool[GrepToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Recursively search files for a regex pattern.

    Uses ripgrep (`rg`) when it's on `$PATH` for speed and `.gitignore` honoring;
    falls back to a pure-Python walk when it isn't. The output shape is the same
    either way — `{"matches": [{"path", "line", "text"}...], "truncated", "used_ripgrep"}`.
    """

    name = "grep"
    description = "Recursively search files for a regex. Uses ripgrep when available, stdlib otherwise."
    input_schema = GrepToolInput

    async def clone(self) -> Self:
        return type(self)(options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "filesystem", "grep"], creator=self)

    async def _run(
        self, input: GrepToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        if shutil.which("rg"):
            return await self._run_ripgrep(input)
        return await self._run_python(input)

    async def _run_ripgrep(self, input: GrepToolInput) -> JSONToolOutput[dict[str, Any]]:
        args = ["rg", "--json", "--line-number", f"--max-count={input.max_results}"]
        if not input.case_sensitive:
            args.append("-i")
        if input.context_lines:
            args.extend(["-A", str(input.context_lines), "-B", str(input.context_lines)])
        if input.glob:
            args.extend(["--glob", input.glob])
        args.extend([input.pattern, input.root])

        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        # rg exits 1 when no matches; any other non-zero is a real error
        if process.returncode not in (0, 1):
            raise ToolError(f"ripgrep failed (exit {process.returncode}): {stderr_bytes.decode(errors='replace')}")

        matches: list[dict[str, Any]] = []
        for raw_line in stdout_bytes.splitlines():
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path_text = data.get("path", {}).get("text", "")
            for sub in data.get("submatches", []):
                matches.append(
                    {
                        "path": path_text,
                        "line": data.get("line_number"),
                        "text": data.get("lines", {}).get("text", "").rstrip("\n"),
                        "match": sub.get("match", {}).get("text", ""),
                    }
                )
                if len(matches) >= input.max_results:
                    break
            if len(matches) >= input.max_results:
                break

        return JSONToolOutput(
            {"matches": matches, "truncated": len(matches) >= input.max_results, "used_ripgrep": True}
        )

    async def _run_python(self, input: GrepToolInput) -> JSONToolOutput[dict[str, Any]]:
        flags = 0 if input.case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(input.pattern, flags)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}") from e

        root = Path(input.root).expanduser().resolve()
        matches: list[dict[str, Any]] = []
        truncated = False
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _DEFAULT_SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            if input.glob and not path.match(input.glob):
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line_no, line in enumerate(fh, start=1):
                        if regex.search(line):
                            matches.append({"path": str(path), "line": line_no, "text": line.rstrip("\n")})
                            if len(matches) >= input.max_results:
                                truncated = True
                                break
            except (OSError, UnicodeError):
                continue
            if truncated:
                break

        return JSONToolOutput({"matches": matches, "truncated": truncated, "used_ripgrep": False})
