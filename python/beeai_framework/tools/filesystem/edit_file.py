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
from beeai_framework.tools.filesystem._file_backend import get_file_backend
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class FileEditToolInput(BaseModel):
    mode: Literal["overwrite", "replace"] = Field(
        description=(
            "'overwrite' replaces the file with `content`. 'replace' does exact-text "
            "search-and-replace keyed on `expected_occurrences`."
        )
    )
    path: str = Field(description="Absolute path to the file to edit.")
    # overwrite mode
    content: str | None = Field(default=None, description="Full new content. Required for mode='overwrite'.")
    # replace mode
    old: str | None = Field(default=None, description="Exact text to find. Required for mode='replace'.")
    new: str | None = Field(default=None, description="Replacement text. Required for mode='replace'.")
    expected_occurrences: int = Field(
        default=1,
        description=(
            "Expected number of times `old` appears. Replace fails without writing if the count differs, "
            "preventing the most common agent edit footgun (silent over-matching)."
        ),
    )


class FileEditTool(Tool[FileEditToolInput, ToolRunOptions, JSONToolOutput[dict[str, Any]]]):
    """Edit a file via the active `FileBackend`.

    Two modes:
    - `overwrite` — write `content` verbatim.
    - `replace` — exact-text search-and-replace keyed on expected occurrence count.

    Default backend hits local disk. Under the ACP Zed adapter the read + write go
    through `fs/read_text_file` / `fs/write_text_file`, so the editor sees the edit
    and can render it as a diff. Returns a unified diff of the change for audit.
    """

    name = "edit_file"
    description = (
        "Edit a file: either overwrite it entirely ('overwrite' mode) or exact-text "
        "search-and-replace ('replace' mode). Returns a unified diff of the change."
    )
    input_schema = FileEditToolInput

    async def clone(self) -> Self:
        return type(self)(options=self.options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "filesystem", "edit_file"], creator=self)

    async def _run(
        self, input: FileEditToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput[dict[str, Any]]:
        if not Path(input.path).is_absolute():
            raise ToolInputValidationError(f"`path` must be absolute, got: {input.path!r}")

        backend = get_file_backend()
        try:
            original = await backend.read_text(input.path)
        except FileNotFoundError:
            if input.mode != "overwrite":
                raise ToolError(f"File not found: {input.path}") from None
            original = ""

        if input.mode == "overwrite":
            if input.content is None:
                raise ToolInputValidationError("`content` is required for mode='overwrite'")
            updated = input.content
        else:  # replace
            if input.old is None or input.new is None:
                raise ToolInputValidationError("`old` and `new` are required for mode='replace'")
            occurrences = original.count(input.old)
            if occurrences != input.expected_occurrences:
                raise ToolError(
                    f"Expected {input.expected_occurrences} occurrence(s) of the target text in "
                    f"{input.path}, found {occurrences}. Refine `old` or adjust `expected_occurrences`."
                )
            updated = original.replace(input.old, input.new)

        await backend.write_text(input.path, updated)

        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=input.path,
                tofile=input.path,
            )
        )
        return JSONToolOutput({"mode": input.mode, "path": input.path, "diff": diff})
