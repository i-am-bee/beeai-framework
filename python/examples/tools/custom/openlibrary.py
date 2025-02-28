import asyncio
from typing import Any

import requests
from pydantic import BaseModel, Field

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.tool import Tool


class OpenLibraryToolInput(BaseModel):
    title: str | None = Field(description="Title of book to retrieve.", default=None)
    olid: str | None = Field(description="Open Library number of book to retrieve.", default=None)
    subjects: str | None = Field(description="Subject of a book to retrieve.", default=None)


class OpenLibraryToolResult(BaseModel):
    preview_url: str
    info_url: str
    bib_key: str


class OpenLibraryTool(Tool[OpenLibraryToolInput]):
    name = "OpenLibrary"
    description = """Provides access to a library of books with information about book titles,
        authors, contributors, publication dates, publisher and isbn."""
    input_schema = OpenLibraryToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.emitter = Emitter.root().child(
            namespace=["tool", "example", "openlibrary"],
            creator=self,
        )

    def _run(self, input: OpenLibraryToolInput, _: Any | None = None) -> OpenLibraryToolResult:
        key = ""
        value = ""
        input_vars = vars(input)
        for val in input_vars:
            if input_vars[val] is not None:
                key = val
                value = input_vars[val]
                break
        else:
            raise ToolInputValidationError("All input values in OpenLibraryToolInput were empty.") from None

        response = requests.get(
            f"https://openlibrary.org/api/books?bibkeys={key}:{value}&jsmcd=data&format=json",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        response.raise_for_status()

        json_output = response.json()[f"{key}:{value}"]

        return OpenLibraryToolResult(
            preview_url=json_output["preview_url"], info_url=json_output["info_url"], bib_key=json_output["bib_key"]
        )


async def main() -> None:
    tool = OpenLibraryTool()
    input = OpenLibraryToolInput(title="It")
    result = await tool.run(input)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
