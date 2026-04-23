# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.errors import ToolError, ToolInputValidationError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions

try:
    import patch_ng
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [filesystem] not found.\nRun 'pip install \"beeai-framework[filesystem]\"' to install."
    ) from e


class FilePatchToolInput(BaseModel):
    mode: Literal["patch", "replace"] = Field(
        description="'patch' applies a unified-diff blob. 'replace' does exact-text search-and-replace in one file."
    )
    # patch mode
    diff: str | None = Field(default=None, description="Unified diff text. Required for mode='patch'.")
    root: str = Field(default=".", description="Root directory for 'patch' mode (strip=0 relative to this).")
    # replace mode
    path: str | None = Field(default=None, description="File path. Required for mode='replace'.")
    old: str | None = Field(default=None, description="Exact text to find. Required for mode='replace'.")
    new: str | None = Field(default=None, description="Replacement text. Required for mode='replace'.")
    expected_occurrences: int = Field(
        default=1,
        description=(
            "Expected number of times `old` appears. If the count differs, the tool fails without writing — "
            "prevents silent over-matching."
        ),
    )


class FilePatchTool(Tool[FilePatchToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Apply a unified-diff patch or a single-file exact-text replacement.

    `mode='patch'` is the industry-standard unified diff (what `git diff` and
    `patch(1)` produce). Implementation uses `patch-ng`.

    `mode='replace'` is a single-file search-and-replace keyed on an exact text
    match. It fails unless the target text appears exactly `expected_occurrences`
    times — the common agent footgun is over-matching a common substring, and this
    makes that a hard error instead of a silent corruption.
    """

    name = "file_patch"
    description = (
        "Edit files by applying a unified diff ('patch' mode) or by exact-text search-and-replace in one file "
        "('replace' mode). Returns the list of changed files and the resulting diff."
    )
    input_schema = FilePatchToolInput

    async def clone(self) -> Self:
        return type(self)(options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "filesystem", "patch"], creator=self)

    async def _run(
        self, input: FilePatchToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        if input.mode == "patch":
            return self._apply_patch(input)
        return self._apply_replace(input)

    def _apply_patch(self, input: FilePatchToolInput) -> JSONToolOutput[dict[str, Any]]:
        if not input.diff:
            raise ToolInputValidationError("`diff` is required for mode='patch'")

        patch_set = patch_ng.fromstring(input.diff.encode())
        if not patch_set:
            raise ToolError("Could not parse the provided diff")

        root = Path(input.root).expanduser().resolve()
        if not patch_set.apply(strip=0, root=str(root)):
            raise ToolError("Patch did not apply cleanly")

        changed = [str((root / p.target.decode()).resolve()) for p in patch_set.items]
        return JSONToolOutput({"mode": "patch", "files_changed": changed, "diff": input.diff})

    def _apply_replace(self, input: FilePatchToolInput) -> JSONToolOutput[dict[str, Any]]:
        if not input.path or input.old is None or input.new is None:
            raise ToolInputValidationError("`path`, `old`, and `new` are required for mode='replace'")

        target = Path(input.path).expanduser()
        if not target.is_absolute():
            raise ToolInputValidationError(f"`path` must be absolute, got: {input.path!r}")
        if not target.is_file():
            raise ToolError(f"File does not exist: {target}")

        original = target.read_text()
        occurrences = original.count(input.old)
        if occurrences != input.expected_occurrences:
            raise ToolError(
                f"Expected {input.expected_occurrences} occurrence(s) of the target text in {target}, "
                f"found {occurrences}. Refine `old` or adjust `expected_occurrences`."
            )
        updated = original.replace(input.old, input.new)
        target.write_text(updated)

        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=str(target),
                tofile=str(target),
            )
        )
        return JSONToolOutput({"mode": "replace", "files_changed": [str(target)], "diff": diff})
