# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("acp", reason="Optional module [acp_zed] not installed.")

from acp.schema import ClientCapabilities, FileSystemCapabilities

from beeai_framework.adapters.acp_zed import ACPZedServer, ACPZedTerminalTool
from beeai_framework.adapters.acp_zed.serve.agent import _active_session
from beeai_framework.tools.errors import ToolError


class _StubExit:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.signal: str | None = None


class _StubOutput:
    def __init__(self, text: str) -> None:
        self.output = text
        self.truncated = False


class _StubCreateTerminalResponse:
    def __init__(self, terminal_id: str) -> None:
        self.terminal_id = terminal_id


class _StubConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def create_terminal(self, **kwargs: Any) -> _StubCreateTerminalResponse:
        self.calls.append(("create_terminal", kwargs))
        return _StubCreateTerminalResponse("term-1")

    async def wait_for_terminal_exit(self, **kwargs: Any) -> _StubExit:
        self.calls.append(("wait_for_terminal_exit", kwargs))
        return _StubExit(exit_code=0)

    async def terminal_output(self, **kwargs: Any) -> _StubOutput:
        self.calls.append(("terminal_output", kwargs))
        return _StubOutput("ok\n")

    async def release_terminal(self, **kwargs: Any) -> None:
        self.calls.append(("release_terminal", kwargs))


@pytest.fixture
def server() -> ACPZedServer[Any]:
    return ACPZedServer()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path(server: ACPZedServer[Any]) -> None:
    conn = _StubConn()
    server.bridge.bind(conn)  # type: ignore[arg-type]
    server.bridge.set_capabilities(
        ClientCapabilities(fs=FileSystemCapabilities(), terminal=True)  # type: ignore[call-arg]
    )
    tool = ACPZedTerminalTool(server)

    token = _active_session.set("sess-1")
    try:
        result = await tool.run({"command": "pytest", "args": ["-q"], "wait": True})
    finally:
        _active_session.reset(token)

    data = result.to_json_safe()
    assert data["exit_code"] == 0
    assert data["stdout"] == "ok\n"
    assert data["terminal_id"] == "term-1"
    assert data["waited"] is True

    called = [c[0] for c in conn.calls]
    assert called == ["create_terminal", "wait_for_terminal_exit", "terminal_output", "release_terminal"]
    # create_terminal received session id + args
    assert conn.calls[0][1]["command"] == "pytest"
    assert conn.calls[0][1]["args"] == ["-q"]
    assert conn.calls[0][1]["session_id"] == "sess-1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_capability_raises(server: ACPZedServer[Any]) -> None:
    server.bridge.bind(_StubConn())  # type: ignore[arg-type]
    server.bridge.set_capabilities(ClientCapabilities(terminal=False))  # type: ignore[call-arg]
    tool = ACPZedTerminalTool(server)
    token = _active_session.set("sess-1")
    try:
        with pytest.raises(ToolError, match="terminal capability"):
            await tool.run({"command": "pytest"})
    finally:
        _active_session.reset(token)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_wait_returns_early(server: ACPZedServer[Any]) -> None:
    conn = _StubConn()
    server.bridge.bind(conn)  # type: ignore[arg-type]
    server.bridge.set_capabilities(ClientCapabilities(terminal=True))  # type: ignore[call-arg]
    tool = ACPZedTerminalTool(server)
    token = _active_session.set("sess-1")
    try:
        result = await tool.run({"command": "sleep", "args": ["10"], "wait": False})
    finally:
        _active_session.reset(token)
    data = result.to_json_safe()
    assert data["waited"] is False
    assert data["terminal_id"] == "term-1"
    # Only create_terminal, no waits
    assert [c[0] for c in conn.calls] == ["create_terminal"]
