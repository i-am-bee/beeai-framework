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


def _add_context(match: dict[str, Any], lines: dict[int, str], context_lines: int) -> None:
    """Attach up to `context_lines` lines before and after a match, read from `lines` (line number -> text)."""
    line = match["line"]
    match["context_before"] = [lines[n] for n in range(line - context_lines, line) if n in lines]
    match["context_after"] = [lines[n] for n in range(line + 1, line + context_lines + 1) if n in lines]


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
    With `context_lines` set, each match also has `context_before` and `context_after`
    lists holding the text of up to that many neighbouring lines.
    """

    name = "grep"
    description = "Recursively search files for a regex. Uses ripgrep when available, stdlib otherwise."
    input_schema = GrepToolInput

    async def clone(self) -> Self:
        cloned = type(self)(options=self.options)
        cloned._cache = await self.cache.clone()
        cloned.middlewares.extend(self.middlewares)
        return cloned

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "filesystem", "grep"], creator=self)

    async def _run(
        self, input: GrepToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        if shutil.which("rg"):
            return await self._run_ripgrep(input)
        return await self._run_python(input)

    async def _run_ripgrep(self, input: GrepToolInput) -> JSONToolOutput[dict[str, Any]]:
        # Ask for one more than needed, so a result set that exactly fills
        # max_results is not mistaken for a truncated one.
        args = ["rg", "--json", "--line-number", f"--max-count={input.max_results + 1}"]
        if not input.case_sensitive:
            args.append("-i")
        if input.context_lines:
            args.extend(["-A", str(input.context_lines), "-B", str(input.context_lines)])
        if input.glob:
            args.extend(["--glob", input.glob])
        args.extend(["--", input.pattern, input.root])

        process = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        # rg exits 1 when no matches; any other non-zero is a real error
        if process.returncode not in (0, 1):
            raise ToolError(f"ripgrep failed (exit {process.returncode}): {stderr_bytes.decode(errors='replace')}")

        matches: list[dict[str, Any]] = []
        # Text of every match and context line ripgrep printed, per file.
        lines_by_path: dict[str, dict[int, str]] = {}
        for raw_line in stdout_bytes.splitlines():
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in ("match", "context"):
                continue
            data = event.get("data", {})
            path_text = data.get("path", {}).get("text", "")
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            lines_by_path.setdefault(path_text, {})[data.get("line_number")] = line_text
            # Past the cap, keep reading so the last match still gets its after-context.
            if event["type"] == "context" or len(matches) > input.max_results:
                continue
            for sub in data.get("submatches", []):
                matches.append(
                    {
                        "path": path_text,
                        "line": data.get("line_number"),
                        "text": line_text,
                        "match": sub.get("match", {}).get("text", ""),
                    }
                )
                if len(matches) > input.max_results:
                    break

        truncated = len(matches) > input.max_results
        matches = matches[: input.max_results]
        if input.context_lines:
            for match in matches:
                _add_context(match, lines_by_path[match["path"]], input.context_lines)
        return JSONToolOutput({"matches": matches, "truncated": truncated, "used_ripgrep": True})

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
                    file_lines = fh.readlines()
                stripped = {n: line.rstrip("\n") for n, line in enumerate(file_lines, start=1)}
                for line_no, line in enumerate(file_lines, start=1):
                    if regex.search(line):
                        match = {"path": str(path), "line": line_no, "text": stripped[line_no]}
                        if input.context_lines:
                            _add_context(match, stripped, input.context_lines)
                        matches.append(match)
                        if len(matches) > input.max_results:
                            truncated = True
                            break
            except (OSError, UnicodeError):
                continue
            if truncated:
                break

        return JSONToolOutput({"matches": matches[: input.max_results], "truncated": truncated, "used_ripgrep": False})
