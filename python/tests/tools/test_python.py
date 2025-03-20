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

import pytest
import pytest_asyncio

from beeai_framework.tools.code import LocalPythonStorage, PythonTool

code_interpreter_url = os.getenv("CODE_INTERPRETER_URL", "http://localhost:50081")


@pytest_asyncio.fixture
async def test_dir() -> str:  # type: ignore[misc]
    test_dir = "test_directory"
    os.makedirs(test_dir, exist_ok=True)
    # Create some test files
    test_files = [os.path.join(test_dir, f"file{i}.txt") for i in range(1, 4)]
    for file in test_files:
        with open(file, "w") as f:  # noqa: ASYNC230
            f.write(f"Content of {file}")

    yield test_dir

    # Clean up: remove the temporary directory and its contents
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(test_dir)


@pytest_asyncio.fixture
async def tool() -> PythonTool:
    test_dir = "test_directory"
    tool = PythonTool(
        {
            "codeInterpreter": {"url": code_interpreter_url},
            "storage": LocalPythonStorage(local_working_dir=test_dir, interpreter_working_dir="./pythonTmp"),
        }
    )
    return tool


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_without_file(tool: PythonTool) -> None:
    result = await tool.run(
        {
            "language": "python",
            "code": "print(str(1+1))",
            "inputFiles": [],
        }
    )
    assert result.stdout
    assert "2" in result.stdout


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_with_file(tool: PythonTool) -> None:
    result = await tool.run(
        {
            "language": "python",
            "code": "print(str(3+1))",
            "inputFiles": ["file2", "file4"],
        }
    )
    assert result.stdout
    assert "4" in result.stdout
