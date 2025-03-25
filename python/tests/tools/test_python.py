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

import os
import shutil
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

from beeai_framework.tools.code import LocalPythonStorage, PythonTool


@pytest_asyncio.fixture
def test_dirs() -> Generator[tuple[str, str], Any, None]:
    local_dir = "local_dir"
    interpreter_dir = "interpreter_dir"
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(interpreter_dir, exist_ok=True)

    # Create some test files
    test_files = [os.path.join(local_dir, f"file{i}.txt") for i in range(1, 4)]
    for file in test_files:
        with open(file, "w") as f:
            f.write(f"Content of {file}")
    with open(os.path.join(interpreter_dir, "dummyID"), "w") as f:
        f.write("Hello, World!")

    yield local_dir, interpreter_dir

    # Clean up: remove the temporary directory and its contents
    shutil.rmtree(local_dir)
    shutil.rmtree(interpreter_dir)


@pytest_asyncio.fixture
async def tool(test_dirs: tuple[str, str]) -> PythonTool:
    tool = PythonTool(
        code_interpreter_url="dummyURL",
        storage=LocalPythonStorage(local_working_dir=test_dirs[0], interpreter_working_dir=test_dirs[1]),
    )
    return tool


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_without_file(tool: PythonTool) -> None:
    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={"stdout": "2\n", "stderr": "", "exit_code": 0, "files": {}},
    ):
        result = await tool.run(
            {
                "language": "python",
                "code": "print(str(1+1))",
                "input_files": [],
            }
        )
    assert result.stdout
    assert "2" in result.stdout


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_with_file(tool: PythonTool) -> None:
    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "files": {"/workspace/file1.txt": "dummyID"},
        },
    ):
        first_result = await tool.run(
            {
                "language": "python",
                "code": """
with open('file1.txt', 'w') as f:
    f.write("Hello, World!")
""",
                "input_files": [],
            }
        )

    assert len(first_result.output_files) == 1
    assert first_result.output_files[0].filename == "file1.txt"

    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={
            "stdout": "4\n",
            "stderr": "",
            "exit_code": 0,
            "files": {"/workspace/file1.txt": "dummyID"},
        },
    ):
        result = await tool.run(
            {
                "language": "python",
                "code": "print(str(3+1))",
                "input_files": ["file1.txt"],
            }
        )
    assert result.stdout
    assert "4" in result.stdout
