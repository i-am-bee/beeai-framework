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


import io
import json
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, Field, create_model

from beeai_framework.agents.react.runners.default.prompts import (
    UserPromptTemplateInput,
)
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.tools.output import PythonToolOutput
from beeai_framework.tools.tool import Tool, ToolRunOptions  # type: ignore[attr-defined]

logger = Logger(__name__)


class Language(Enum):
    PYTHON = "python"
    SHELL = "shell"


class PythonTool(Tool[BaseModel, ToolRunOptions, PythonToolOutput]):
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
        input_files = None
        if options and options.get("storage"):
            self.files = options["storage"].list_files()
            filenames = [file.filename for file in self.files]
            members = {}
            for filename in filenames:
                members[filename.upper()] = filename
                python_files = Enum("PythonFiles", members)  # type: ignore[misc]
            if "python_files" in locals():
                input_files = (
                    list[python_files],
                    Field(
                        description="""To access an existing file, you must specify it;
                        otherwise, the file will not be accessible.
                        IMPORTANT: If the file is not provided in the input, it will not be accessible."""
                    ),
                )
                self.input_schema = create_model(
                    "PythonToolInput",
                    language=(Language, Field(description="Use shell for ffmpeg, pandoc, yt-dlp")),
                    code=(str, Field(description="full source code file that will be executed")),
                    inputFiles=input_files
                    )
            else:
                self.input_schema = create_model(
                    "PythonToolInput",
                    language=(Language, Field(description="Use shell for ffmpeg, pandoc, yt-dlp")),
                    code=(str, Field(description="full source code file that will be executed"))
                    )

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "python", "code_interpreter"],
            creator=self,
        )

    async def _run(self, input: Any, options: ToolRunOptions | None, context: RunContext) -> PythonToolOutput:
        async def get_source_code() -> Any:
            if self.options and self.options.get("preprocess"):
                response = await self.options["preprocess"]["llm"].create(
                    {
                        "messages": PromptTemplate(
                            PromptTemplateInput(schema=UserPromptTemplateInput, template="{{input}}")
                        ),
                        "abortSignal": context.signal,
                    }
                )
                return response.getTextContent().trim()
            return input.code

        async def call_code_interpreter(url: str, body: dict[str, Any], files: dict[str, io.BufferedReader]) -> Any:
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            async with httpx.AsyncClient() as client:
                # response = await client.post(url, headers=headers, data=json.dumps(body), files=files)
                response = await client.post(url, headers=headers, data=json.dumps(body), files=files)  # type:  ignore[arg-type]
                response.raise_for_status()
                return response.json()

        if self.options and self.options.get("codeInterpreter") and self.options["codeInterpreter"].get("url"):
            url = self.options["codeInterpreter"]["url"] + "/v1/execute"
            prefix = "/workspace/"
            unique_files = {}
            if self.options and self.options.get("storage"):
                for file in self.files:
                    if file.filename in [python_file.value for python_file in input.inputFiles]:
                        unique_files[file.filename] = file
                self.options["storage"].upload(unique_files.values())

            url = self.options["codeInterpreter"]["url"] + "/v1/execute"
            files = {}
            for file in unique_files.values():
                files[prefix + file.filename] = open("localTmp/" + file.filename, "rb")  # noqa: ASYNC230,SIM115
            result = await call_code_interpreter(
                url=url,
                body={
                    "source_code": await get_source_code(),
                },
                files=files,
            )

            filtered_files = []
            if result["files"]:
                for file in result["files"].items():
                    if file.startswith(prefix):
                        file = {"filename": file.path[len(prefix) :], "pythonId": str(file.pythonId)}
                        for unique_file in unique_files:
                            if file["filename"] == unique_file.filename and file["pythonId"] == unique_file.pythonId:
                                filtered_files.append(
                                    {"filename": file.path[len(prefix) :], "pythonId": str(file.pythonId)}
                                )
                self.options["storage"].download(filtered_files)
            return PythonToolOutput(result)
        return PythonToolOutput({})
