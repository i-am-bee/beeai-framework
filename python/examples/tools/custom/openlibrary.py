import asyncio
import sys
from typing import Any

import httpx
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import JSONToolOutput, ToolRunOptions


class OpenLibraryToolInput(BaseModel):
    title: str | None = Field(description="Title of book to retrieve.", default=None)
    olid: str | None = Field(description="Open Library number of book to retrieve.", default=None)
    subjects: str | None = Field(description="Subject of a book to retrieve.", default=None)


class OpenLibraryToolResult(BaseModel):
    preview_url: str
    info_url: str
    bib_key: str


class OpenLibraryTool(Tool[OpenLibraryToolInput, ToolRunOptions, JSONToolOutput]):
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
    ) -> JSONToolOutput:
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

        json_output = {}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://openlibrary.org/api/books?bibkeys={key}:{value}&jsmcd=data&format=json",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response.raise_for_status()

            json_output = response.json()[f"{key}:{value}"]

        return JSONToolOutput(
            result={
                "preview_url": json_output.get("preview_url", ""),
                "info_url": json_output.get("info_url", ""),
                "bib_key": json_output.get("bib_key", ""),
            }
        )


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
