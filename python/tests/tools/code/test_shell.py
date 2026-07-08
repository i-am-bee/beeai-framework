# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.code import ShellTool
from beeai_framework.tools.errors import ToolError, ToolInputValidationError


@pytest.fixture
def tool() -> ShellTool:
    return ShellTool()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path(tool: ShellTool) -> None:
    result = await tool.run({"command": ["echo", "hello"], "timeout_seconds": 5})
    assert isinstance(result, JSONToolOutput)
    data = result.to_json_safe()
    assert data["exit_code"] == 0
    assert data["stdout"].strip() == "hello"
    assert data["timed_out"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_zero_exit(tool: ShellTool) -> None:
    result = await tool.run({"command": ["sh", "-c", "exit 3"], "timeout_seconds": 5})
    assert result.to_json_safe()["exit_code"] == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_kills_process(tool: ShellTool) -> None:
    result = await tool.run({"command": ["sleep", "5"], "timeout_seconds": 0.1})
    data = result.to_json_safe()
    assert data["timed_out"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_binary_raises(tool: ShellTool) -> None:
    with pytest.raises(ToolError, match="Command not found"):
        await tool.run({"command": ["this-command-does-not-exist-xyz"], "timeout_seconds": 5})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_command_rejected(tool: ShellTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run({"command": [], "timeout_seconds": 5})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stdin_plumbing(tool: ShellTool) -> None:
    result = await tool.run({"command": ["cat"], "input_text": "piped", "timeout_seconds": 5})
    assert result.to_json_safe()["stdout"] == "piped"
