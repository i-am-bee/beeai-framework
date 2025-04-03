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
from unittest.mock import patch

import pytest

from beeai_framework.tools import StringToolOutput, ToolError
from beeai_framework.tools.code import SandboxTool
from beeai_framework.tools.code.sandbox import SandboxToolCreateError, SandboxToolExecuteError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_instantiate() -> None:
    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={
            "tool_name": "test",
            "tool_description": "A test tool",
            "tool_input_schema_json": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "string"},
                    },
                }
            ),
        },
    ):
        sandbox_tool = await SandboxTool.from_source_code(url="dummyURL", source_code="source code")

    assert sandbox_tool.name == "test"
    assert sandbox_tool.description == "A test tool"
    assert sandbox_tool.input_schema.model_json_schema() == {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_error() -> None:
    with (
        patch(
            "beeai_framework.tools.code.PythonTool.call_code_interpreter",
            return_value={"cause": {"name": "HTTPParserError"}},
        ),
        pytest.raises(SandboxToolCreateError),
    ):
        await SandboxTool.from_source_code(url="dummyURL", source_code="source code")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run() -> None:
    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={
            "tool_name": "test",
            "tool_description": "A test tool",
            "tool_input_schema_json": json.dumps(
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
                }
            ),
        },
    ):
        sandbox_tool = await SandboxTool.from_source_code(url="dummyURL", source_code="source code")

    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={"exit_code": 0, "tool_output_json": '{"something": "42"}'},
    ):
        result = await sandbox_tool.run({"a": 42, "b": "test"})

    assert isinstance(result, StringToolOutput)
    assert result.get_text_content() == '{"something": "42"}'


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_error() -> None:
    with patch(
        "beeai_framework.tools.code.PythonTool.call_code_interpreter",
        return_value={
            "tool_name": "test",
            "tool_description": "A test tool",
            "tool_input_schema_json": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "string"},
                    },
                }
            ),
        },
    ):
        sandbox_tool = await SandboxTool.from_source_code(url="dummyURL", source_code="source code")

    with (
        patch(
            "beeai_framework.tools.code.PythonTool.call_code_interpreter",
            return_value={"stderr": "Oh no, it does not work"},
        ),
        pytest.raises(SandboxToolExecuteError),
    ):
        await sandbox_tool.run({"a": 42, "b": "test"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_error() -> None:
    with pytest.raises(ToolError):
        await SandboxTool.from_source_code(url="dummyURL", source_code="source code")
