# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.errors import AbortError
from beeai_framework.tools import ToolError, ToolRunOptions
from beeai_framework.tools.code import LocalShellBackend, ShellTool
from beeai_framework.tools.code import _shell_backend as shell_backend
from beeai_framework.utils import AbortController
from tests.agents._scripted import ScriptedChatModel, tool_call_message


@pytest_asyncio.fixture
async def running_process(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[list[asyncio.subprocess.Process], asyncio.Event]]:
    """Observe real child processes and always reap them, including on failing tests."""
    processes: list[asyncio.subprocess.Process] = []
    communicating = asyncio.Event()
    owned_groups: dict[int, int] = {}
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def create(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await create_subprocess_exec(*args, **kwargs)
        processes.append(process)
        if os.name == "posix" and kwargs.get("start_new_session"):
            with suppress(ProcessLookupError):
                if os.getpgid(process.pid) == process.pid:
                    owned_groups[process.pid] = process.pid
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
            if process.pid in owned_groups:
                with suppress(ProcessLookupError):
                    if os.getpgid(process.pid) == owned_groups[process.pid]:
                        os.killpg(owned_groups[process.pid], signal.SIGKILL)
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
    process.pid = 12345
    process.returncode = 0
    if os.name == "posix":
        monkeypatch.setattr(shell_backend.os, "killpg", Mock(side_effect=ProcessLookupError))
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


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "condition deadline exceeded"
        time.sleep(0.01)


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return False
    except ProcessLookupError:
        return False
    except FileNotFoundError:
        return False
    return True


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
@pytest.mark.parametrize("parent_exits", [False, True])
async def test_descendant_is_killed_without_waiting_for_its_pipes(
    running_process: tuple[list[asyncio.subprocess.Process], asyncio.Event],
    tmp_path: Path,
    interruption: str,
    parent_exits: bool,
) -> None:
    processes, communicating = running_process
    ready = tmp_path / "descendant.pid"
    heartbeat = tmp_path / "heartbeat"
    child_code = (
        "import os, pathlib, sys, time; "
        "ready, heartbeat = map(pathlib.Path, sys.argv[1:]); "
        "heartbeat.write_text('0'); ready.write_text(str(os.getpid())); "
        "counter = 0\n"
        "while True:\n"
        " counter += 1; heartbeat.write_text(str(counter)); time.sleep(0.02)\n"
    )
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1], sys.argv[2]]); "
        "ready = pathlib.Path(sys.argv[1])\n"
        "while not ready.exists() or not ready.read_text().strip(): time.sleep(0.01)\n"
        + ("" if parent_exits else "time.sleep(60)")
    )
    task = asyncio.create_task(
        LocalShellBackend().run(
            command=[sys.executable, "-c", parent_code, str(ready), str(heartbeat)],
            timeout_seconds=2 if interruption == "timeout" else None,
        )
    )
    descendant_pid: int | None = None
    descendant_group: int | None = None
    try:
        await asyncio.wait_for(communicating.wait(), timeout=5)
        descendant_group = processes[0].pid if getattr(shell_backend, "_USE_PROCESS_GROUP", False) else os.getpgrp()
        await asyncio.to_thread(wait_until, lambda: ready.exists() and bool(ready.read_text().strip()))
        descendant_pid = int(ready.read_text())
        assert os.getpgid(descendant_pid) == descendant_group
        if parent_exits:
            await asyncio.to_thread(wait_until, lambda: processes[0].returncode is not None)
        if interruption == "cancel":
            task.cancel()
        # asyncio.wait bounds the assertion without waiting for cancellation
        # cleanup in a broken backend that ignores further cancellation.
        done, _ = await asyncio.wait({task}, timeout=5)
        assert task in done, "descendant still holds the backend's output pipes open"
        if interruption == "cancel":
            with pytest.raises(asyncio.CancelledError):
                task.result()
        else:
            assert task.result()["timed_out"] is True
        # Without /proc (e.g. macOS), a killed descendant stays visible until it is reaped.
        await asyncio.to_thread(wait_until, lambda: not process_is_running(descendant_pid))
        last_heartbeat = heartbeat.read_text()
        await asyncio.sleep(0.08)
        assert heartbeat.read_text() == last_heartbeat
    finally:
        # Recover the owned PID even if an assertion failed just after readiness.
        if descendant_pid is None and ready.exists() and ready.read_text().strip():
            descendant_pid = int(ready.read_text())
        # Kill only the recorded descendant, and only in its recorded group.
        if descendant_pid is not None and descendant_group is not None:
            with suppress(ProcessLookupError):
                if os.getpgid(descendant_pid) == descendant_group:
                    os.kill(descendant_pid, signal.SIGKILL)
        for process in processes:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_timeout_after_process_exit_preserves_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock(spec=asyncio.subprocess.Process)
    process.pid = 12345
    process.returncode = 0
    process.communicate = AsyncMock(side_effect=[TimeoutError(), (b"", b"")])
    killpg = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(shell_backend.os, "killpg", killpg)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    result = await LocalShellBackend().run(command=[sys.executable, "-c", "pass"], timeout_seconds=1)

    assert result["timed_out"] is True
    assert process.communicate.await_count == 2
    killpg.assert_called_once_with(process.pid, signal.SIGKILL)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_windows_branch_keeps_direct_child_kill(
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
) -> None:
    process = Mock(spec=asyncio.subprocess.Process)
    process.returncode = -9
    error = asyncio.CancelledError() if interruption == "cancel" else TimeoutError()
    process.communicate = AsyncMock(side_effect=[error, (b"", b"")])
    create = AsyncMock(return_value=process)
    killpg = Mock()
    monkeypatch.setattr(shell_backend, "_USE_PROCESS_GROUP", False)
    monkeypatch.setattr(shell_backend.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    if interruption == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await LocalShellBackend().run(command=[sys.executable, "-c", "pass"])
    else:
        result = await LocalShellBackend().run(command=[sys.executable, "-c", "pass"], timeout_seconds=1)
        assert result["timed_out"] is True
    process.kill.assert_called_once()
    killpg.assert_not_called()
    assert "start_new_session" not in create.call_args.kwargs
