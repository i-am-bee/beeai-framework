# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from typing import Any

import httpx
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import JSONToolOutput, Tool, ToolError, ToolInputValidationError, ToolRunOptions


class OpenLibraryToolInput(BaseModel):
    title: str | None = Field(description="Title of book to retrieve.", default=None)
    olid: str | None = Field(description="Open Library number of book to retrieve.", default=None)
    subjects: str | None = Field(description="Subject of a book to retrieve.", default=None)


class OpenLibraryToolResult(BaseModel):
    preview_url: str
    info_url: str
    bib_key: str


class OpenLibraryToolOutput(JSONToolOutput[OpenLibraryToolResult]):
    pass


class OpenLibraryTool(Tool[OpenLibraryToolInput, ToolRunOptions, OpenLibraryToolOutput]):
    name = "OpenLibrary"
    description = """Provides access to a library of books with information about book titles,
        authors, contributors, publication dates, publisher and isbn."""
    input_schema = OpenLibraryToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "openlibrary"],
            creator=self,
        )

    async def _run(
        self, tool_input: OpenLibraryToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> OpenLibraryToolOutput:
        key = ""
        value = ""
        input_vars = vars(tool_input)
        for val in input_vars:
            if input_vars[val] is not None:
                key = val
                value = input_vars[val]
                break
        else:
            raise ToolInputValidationError("All input values in OpenLibraryToolInput were empty.") from None

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://openlibrary.org/api/books?bibkeys={key}:{value}&jsmcd=data&format=json",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response.raise_for_status()

            result = response.json().get(f"{key}:{value}")
            if not result:
                raise ToolError(f"No book found with {key}={value}.")

            return OpenLibraryToolOutput(OpenLibraryToolResult.model_validate(result))


async def main() -> None:
    tool = OpenLibraryTool()
    tool_input = OpenLibraryToolInput(title="It")
    result = await tool.run(tool_input)
    print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())
