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

from enum import Enum
from functools import cached_property
from typing import Any

import httpx
from pydantic import BaseModel, Field, InstanceOf, create_model

from beeai_framework import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.code.output import PythonToolOutput
from beeai_framework.tools.code.storage import PythonFile, PythonStorage
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions
from beeai_framework.utils.strings import create_strenum

logger = Logger(__name__)


class Language(Enum):
    PYTHON = "python"
    SHELL = "shell"


class PythonToolTemplate(BaseModel):
    input: str


class PreProcess(BaseModel):
    llm: InstanceOf[ChatModel]
    prompt_template: InstanceOf[PromptTemplate[PythonToolTemplate]]


class PythonToolInput(BaseModel):
    language: Language = Field(description="Use shell for ffmpeg, pandoc, yt-dlp")
    code: str = Field(description="full source code file that will be executed")


class PythonTool(Tool[PythonToolInput, ToolRunOptions, PythonToolOutput]):
    name = "Python"
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

    def __init__(self, code_interpreter_url: str, storage: PythonStorage, preprocess: PreProcess | None = None) -> None:
        super().__init__()
        self._code_interpreter_url = code_interpreter_url
        self._storage = storage
        self._preprocess = preprocess
        self.files: list[PythonFile] = []

    @cached_property
    def input_schema(self) -> type[PythonToolInput]:
        self.files = self._storage.list_files()
        filenames = [file.filename for file in self.files]
        python_files = create_strenum("PythonFiles", filenames) if filenames else None
        if python_files:
            input_files = (
                list[python_files],  # type: ignore
                Field(
                    description="""To access an existing file, you must specify it;
otherwise, the file will not be accessible.
IMPORTANT: If the file is not provided in the input, it will not be accessible."""
                ),
            )
            return create_model(
                "PythonToolInput",
                __base__=PythonToolInput,
                input_files=input_files,
            )

        return PythonToolInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "python", "code_interpreter"],
            creator=self,
        )

    async def _run(
        self, tool_input: PythonToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> PythonToolOutput:
        async def get_source_code() -> str:
            if self._preprocess:
                response = await self._preprocess.llm.create(
                    messages=[
                        UserMessage(self._preprocess.prompt_template.render(PythonToolTemplate(input=tool_input.code)))
                    ],
                    abort_signal=context.signal,
                )
                return response.get_text_content()
            return tool_input.code

        async def call_code_interpreter(url: str, body: dict[str, Any]) -> Any:
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()

        execute_url = self._code_interpreter_url + "/v1/execute"
        prefix = "/workspace/"

        unique_files: list[PythonFile] = []
        for file in self.files or self._storage.list_files():
            if file.filename in [python_file.value for python_file in tool_input.input_files or []]:  # type: ignore
                unique_files.append(file)
        self._storage.upload(unique_files)

        files_dict = {}
        for file in unique_files:
            files_dict[prefix + file.filename] = file.python_id

        result = await call_code_interpreter(
            url=execute_url,
            body={
                "source_code": await get_source_code(),
                "files": files_dict,
            },
        )

        files_output: list[PythonFile] = []
        if result["files"]:
            for file_path, python_id in result["files"].items():
                if file_path.startswith(prefix):
                    filename = file_path.removeprefix(prefix)
                    if all(filename != f.filename or python_id != f.python_id for f in unique_files):
                        files_output.append(PythonFile(filename=filename, id=python_id, python_id=python_id))

            self._storage.download(files_output)

        return PythonToolOutput(result["stdout"], result["stderr"], result["exit_code"], files_output)
