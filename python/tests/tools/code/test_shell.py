# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from contextvars import Context

import pytest

from beeai_framework.tools import JSONToolOutput
from beeai_framework.tools.code import LocalShellBackend, ShellTool, get_shell_backend, setup_shell_backend
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


@pytest.mark.unit
def test_local_default_in_new_context() -> None:
    async def run() -> None:
        result = await ShellTool().run({"command": [sys.executable, "-c", "print('hello')"]})
        assert result.to_json_safe()["stdout"] == "hello\n"
        assert result.to_json_safe()["exit_code"] == 0

    Context().run(asyncio.run, run())


@pytest.mark.unit
def test_backend_cleanup_in_new_context() -> None:
    context = Context()
    backend = LocalShellBackend()
    nested_backend = LocalShellBackend()
    cleanup = context.run(setup_shell_backend, backend)
    try:
        assert context.run(get_shell_backend) is backend
        assert Context().run(get_shell_backend) is not backend
        nested_cleanup = context.run(setup_shell_backend, nested_backend)
        try:
            assert context.run(get_shell_backend) is nested_backend
        finally:
            context.run(nested_cleanup)
        assert context.run(get_shell_backend) is backend
    finally:
        context.run(cleanup)

    async def run() -> None:
        result = await ShellTool().run({"command": [sys.executable, "-c", "print('restored')"]})
        assert result.to_json_safe()["stdout"] == "restored\n"

    context.run(asyncio.run, run())
