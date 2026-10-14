# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.errors import AbortError
from beeai_framework.tools import ToolError, ToolRunOptions
from beeai_framework.tools.code import LocalShellBackend, ShellTool
from beeai_framework.utils import AbortController
from tests.agents._scripted import ScriptedChatModel, tool_call_message


@pytest_asyncio.fixture
async def running_process(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[list[asyncio.subprocess.Process], asyncio.Event]]:
    """Observe real child processes and always reap them, including on failing tests."""
    processes: list[asyncio.subprocess.Process] = []
    communicating = asyncio.Event()
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def create(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await create_subprocess_exec(*args, **kwargs)
        processes.append(process)
        communicate = process.communicate

        async def tracked_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
            communicating.set()
            return await communicate(input)

        monkeypatch.setattr(process, "communicate", tracked_communicate)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    try:
        yield processes, communicating
    finally:
        for process in processes:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.communicate()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [None, 60.0])
@pytest.mark.parametrize("input_text", [None, "hello"])
async def test_cancel_backend_reaps_process(
    running_process: tuple[list[asyncio.subprocess.Process], asyncio.Event],
    timeout_seconds: float | None,
    input_text: str | None,
) -> None:
    processes, communicating = running_process
    task = asyncio.create_task(
        LocalShellBackend().run(
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
            timeout_seconds=timeout_seconds,
            input_text=input_text,
        )
    )
    try:
        await asyncio.wait_for(communicating.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        assert processes[0].returncode is not None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ["task", "signal", "tool_timeout"])
async def test_cancel_shell_tool_reaps_process(
    running_process: tuple[list[asyncio.subprocess.Process], asyncio.Event],
    cancel_mode: str,
) -> None:
    processes, communicating = running_process
    controller = AbortController()
    tool = ShellTool()

    async def run() -> None:
        await tool.run(
            {"command": [sys.executable, "-c", "import time; time.sleep(60)"], "timeout_seconds": None},
            ToolRunOptions(signal=controller.signal, timeout=2.0 if cancel_mode == "tool_timeout" else None),
        )

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(communicating.wait(), timeout=10)
        if cancel_mode == "task":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10)
        elif cancel_mode == "signal":
            controller.abort("Stop this run")
            with pytest.raises(AbortError, match="Stop this run"):
                await asyncio.wait_for(task, timeout=10)
        else:
            with pytest.raises(ToolError, match="timed out after 2.0s"):
                await asyncio.wait_for(task, timeout=10)
        assert processes[0].returncode is not None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        tool.emitter.destroy()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_after_process_exit_preserves_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exit between cancellation and kill must not replace CancelledError."""
    process = Mock(spec=asyncio.subprocess.Process)
    process.communicate = AsyncMock(side_effect=[asyncio.CancelledError(), (b"", b"")])
    process.kill.side_effect = ProcessLookupError
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    with pytest.raises(asyncio.CancelledError):
        await LocalShellBackend().run(command=[sys.executable, "-c", "pass"])

    assert process.communicate.await_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("first_interruption", ["cancel", "timeout"])
async def test_cancellation_during_cleanup_still_reaps_process(
    running_process: tuple[list[asyncio.subprocess.Process], asyncio.Event],
    monkeypatch: pytest.MonkeyPatch,
    first_interruption: str,
) -> None:
    processes, communicating = running_process
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    task = asyncio.create_task(
        LocalShellBackend().run(
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
            timeout_seconds=1.0 if first_interruption == "timeout" else None,
        )
    )
    try:
        await asyncio.wait_for(communicating.wait(), timeout=10)
        process = processes[0]
        communicate = process.communicate

        async def delayed_cleanup(input: bytes | None = None) -> tuple[bytes, bytes]:
            cleanup_started.set()
            await allow_cleanup.wait()
            result = await communicate(input)
            cleanup_finished.set()
            return result

        monkeypatch.setattr(process, "communicate", delayed_cleanup)
        if first_interruption == "cancel":
            task.cancel("original cancellation")
        await asyncio.wait_for(cleanup_started.wait(), timeout=10)
        task.cancel("cancel during cleanup")
        allow_cleanup.set()
        expected_reason = "original cancellation" if first_interruption == "cancel" else "cancel during cleanup"
        with pytest.raises(asyncio.CancelledError, match=expected_reason):
            await asyncio.wait_for(task, timeout=10)
        assert cleanup_finished.is_set()
        assert process.returncode is not None
    finally:
        allow_cleanup.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ["task", "signal"])
async def test_cancel_agent_reaps_shell_process(
    running_process: tuple[list[asyncio.subprocess.Process], asyncio.Event],
    cancel_mode: str,
) -> None:
    processes, communicating = running_process
    controller = AbortController()
    tool = ShellTool()
    model = ScriptedChatModel(
        [[tool_call_message("shell", {"command": [sys.executable, "-c", "import time; time.sleep(60)"]})]]
    )
    agent = RequirementAgent(llm=model, tools=[tool])

    async def run() -> None:
        await agent.run("Run the command", signal=controller.signal)

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(communicating.wait(), timeout=10)
        if cancel_mode == "task":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10)
        else:
            controller.abort("Stop this agent")
            with pytest.raises(AbortError, match="Stop this agent"):
                await asyncio.wait_for(task, timeout=10)
        assert processes[0].returncode is not None
        assert model.call_count == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        agent.emitter.destroy()
        model.emitter.destroy()
        tool.emitter.destroy()
