# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field, create_model

from beeai_framework.backend.message import UserMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolRunOptions

logger = Logger(__name__)


class Language(Enum):
    PYTHON = "python"
    SHELL = "shell"


class PythonTool(Tool[BaseModel, ToolRunOptions, StringToolOutput]):
    name = "Python"
    input_schema = BaseModel
    description = """
    Run Python and/or shell code and return the console output. Use for isolated calculations,
    computations, data or file manipulation but still prefer assistant's capabilities
    (IMPORTANT: Do not use for text analysis or summarization).
    Files provided by the user, or created in a previous run, will be accessible
    if and only if they are specified in the input. It is necessary to always print() results.
    The following shell commands are available:
    Use ffmpeg to convert videos.
    Use yt-dlp to download videos, and unless specified otherwise use `-S vcodec:h264,res,acodec:m4a`
    for video and `-x --audio-format mp3` for audio-only.
    Use pandoc to convert documents between formats (like MD, DOC, DOCX, PDF) -- and don't forget that
    you can create PDFs by writing markdown and then converting.
    In Python, the following modules are available:
    Use numpy, pandas, scipy and sympy for working with data.
    Use matplotlib to plot charts.
    Use pillow (import PIL) to create and manipulate images.
    Use moviepy for complex manipulations with videos.
    Use PyPDF2, pikepdf, or fitz to manipulate PDFs.
    Use pdf2image to convert PDF to images.
    Other Python libraries are also available -- however, prefer using the ones above.
    Prefer using qualified imports -- `import library; library.thing()` instead of `import thing from library`.
    Do not attempt to install libraries manually -- it will not work.
    Each invocation of Python runs in a completely fresh VM -- it will not remember anything from before.
    Do not use this tool multiple times in a row, always write the full code you want to run in a single invocation."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.files = self.options["storage"].list()
        filenames = [file["filename"] for file in self.files]
        members = {}
        for filename in filenames:
            members[filename.upper()] = filename
        python_files = Enum("PythonFiles", members)
        self.input_schema = create_model(
            "PythonToolInput",
            language=(Language, Field(description="Use shell for ffmpeg, pandoc, yt-dlp")),
            code=(str, Field(description="full source code file that will be executed")),
            inputFiles=(
                list[python_files],
                Field(
                    description="""To access an existing file, you must specify it;
                    otherwise, the file will not be accessible.
                    IMPORTANT: If the file is not provided in the input, it will not be accessible."""
                ),
            ),
        )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "python", "code_interpreter"],
            creator=self,
        )

    async def _run(self, input: Any, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        async def get_source_code() -> str:
            if self.options.get("preprocess"):
                response = await options["preprocess"]["llm"].create(
                    {
                        "messages": [UserMessage(PromptTemplate.render({input: input.code}))],
                        "abortSignal": context.signal,
                    }
                )
                return response.getTextContent().trim()
            return input.code

        async def call_code_interpreter(url: str, body: json, files: dict) -> json:
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, data=json.dumps(body))
                response.raise_for_status()
                return response.json()

        prefix = "/workspace/"
        unique_files = {}
        for file in self.files:
            if file["filename"] in [python_file.value for python_file in input.inputFiles]:
                unique_files[file["filename"]] = file
        await self.options["storage"].upload(unique_files.values())

        url = self.options["codeInterpreter"]["url"] + "/v1/execute"
        files = {}
        for file in unique_files.values():
            files[prefix + file["filename"]] = open("localTmp/" + file["filename"], "rb")  # noqa: ASYNC230,SIM115
        result = await call_code_interpreter(
            url=url,
            body={
                "source_code": await get_source_code(),
            },
            files=files,
        )

        filtered_files = []
        for file in result["files"].items():
            if file.startswith(prefix):
                filtered_files.append({"filename": file.path[len(prefix) :], "pythonId": str(file.pythonId)})
        await self.options["storage"].download(filtered_files)

        return StringToolOutput(json.dumps(result))
