# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from beeai_framework.adapters.hol_guard import HOLGuardMiddleware
from beeai_framework.tools import tool


class StubHOLGuardMiddleware(HOLGuardMiddleware):
    def __init__(self, *, allowed: bool, reason: str = "test decision") -> None:
        super().__init__()
        self.allowed = allowed
        self.reason = reason
        self.payloads: list[dict[str, Any]] = []

    async def _evaluate(self, payload: dict[str, Any]) -> tuple[bool, str]:
        self.payloads.append(payload)
        return self.allowed, self.reason


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int,
        stdout: bytes = b"",
        stderr: bytes = b"",
        hang_until_killed: bool = False,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.hang_until_killed = hang_until_killed
        self.killed = False

    async def communicate(self, _: bytes | None = None) -> tuple[bytes, bytes]:
        if self.hang_until_killed and not self.killed:
            await asyncio.sleep(60)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True


class _FakeEmitter:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []
        self.removed: list[Any] = []

    def on(self, *args: Any, **kwargs: Any):
        def decorator(callback: Any) -> Any:
            self.callbacks.append(callback)
            return callback

        return decorator

    def off(self, *, callback: Any) -> None:
        self.removed.append(callback)
        if callback in self.callbacks:
            self.callbacks.remove(callback)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_hol_guard_allows_tool_execution() -> None:
    calls: list[str] = []

    @tool(name="write_file", description="Write a file for the test")
    async def write_file(path: str) -> str:
        calls.append(path)
        return "written"

    middleware = StubHOLGuardMiddleware(allowed=True)
    write_file.middlewares.append(middleware)

    result = await write_file.run({"path": "/tmp/example.txt"})

    assert calls == ["/tmp/example.txt"]
    assert result.get_text_content() == "written"
    assert middleware.payloads[0]["tool_name"] == "write_file"
    assert middleware.payloads[0]["tool_input"] == {"path": "/tmp/example.txt"}
    assert middleware.payloads[0]["hook_event_name"] == "PreToolUse"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_hol_guard_blocks_tool_before_execution() -> None:
    calls: list[str] = []

    @tool(name="delete_file", description="Delete a file for the test")
    async def delete_file(path: str) -> str:
        calls.append(path)
        return "deleted"

    middleware = StubHOLGuardMiddleware(allowed=False, reason="policy action: block")
    delete_file.middlewares.append(middleware)

    result = await delete_file.run({"path": "/tmp/important.txt"})

    assert calls == []
    assert "HOL Guard blocked BeeAI tool 'delete_file' before execution" in result.get_text_content()
    assert "policy action: block" in result.get_text_content()


@pytest.mark.unit
def test_hol_guard_fails_closed_when_unavailable_by_default() -> None:
    middleware = HOLGuardMiddleware()
    assert middleware._availability_result("unavailable") == (False, "unavailable")


@pytest.mark.unit
def test_hol_guard_can_fail_open_when_explicitly_configured() -> None:
    middleware = HOLGuardMiddleware(fail_closed=False)
    assert middleware._availability_result("unavailable") == (True, "unavailable")


@pytest.mark.unit
def test_hol_guard_formats_json_policy_action() -> None:
    reason = HOLGuardMiddleware._decision_reason(
        stdout=b'{"policy_action":"review"}',
        stderr=b"",
        returncode=1,
    )
    assert reason == "policy action: review"


@pytest.mark.unit
def test_hol_guard_requires_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be greater than zero"):
        HOLGuardMiddleware(timeout=0)


@pytest.mark.unit
def test_bind_does_not_unregister_a_concurrent_context() -> None:
    middleware = HOLGuardMiddleware()
    first_emitter = _FakeEmitter()
    second_emitter = _FakeEmitter()
    first = SimpleNamespace(run_id="run-1", emitter=first_emitter)
    second = SimpleNamespace(run_id="run-2", emitter=second_emitter)

    middleware.bind(first)
    middleware.bind(second)

    assert len(first_emitter.callbacks) == 2
    assert len(second_emitter.callbacks) == 2
    assert first_emitter.removed == []
    assert middleware._bound_run_ids == {"run-1", "run-2"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_nonzero_policy_decision_blocks_even_when_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(
        returncode=2,
        stdout=b'{"policy_action":"block","reason":"policy denied"}',
    )

    async def create_process(*args: Any, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    middleware = HOLGuardMiddleware(fail_closed=False)

    assert await middleware._evaluate({"tool_name": "delete_file"}) == (False, "policy denied")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_unexpected_nonzero_exit_uses_availability_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(returncode=127, stderr=b"guard process crashed")

    async def create_process(*args: Any, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    middleware = HOLGuardMiddleware(fail_closed=False)

    assert await middleware._evaluate({"tool_name": "delete_file"}) == (True, "guard process crashed")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_start_failure_uses_availability_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create_process(*args: Any, **kwargs: Any) -> _FakeProcess:
        raise OSError("cannot execute")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    assert await HOLGuardMiddleware()._evaluate({}) == (False, "HOL Guard could not start: cannot execute")
    assert await HOLGuardMiddleware(fail_closed=False)._evaluate({}) == (
        True,
        "HOL Guard could not start: cannot execute",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_timeout_uses_availability_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess(returncode=-9, hang_until_killed=True)

    async def create_process(*args: Any, **kwargs: Any) -> _FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    middleware = HOLGuardMiddleware(timeout=0.001, fail_closed=False)

    assert await middleware._evaluate({}) == (True, "HOL Guard timed out after 0.001s")
    assert process.killed is True
