# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("acp", reason="Optional module [acp_zed] not installed.")

from acp.schema import ClientCapabilities, FileSystemCapabilities, ReadTextFileResponse

from beeai_framework.adapters.acp_zed.serve.agent import FsBridge, _active_session
from beeai_framework.adapters.acp_zed.serve.backends import ACPFileBackend, ACPShellBackend
from beeai_framework.adapters.acp_zed.serve.io import ACPZedIOContext
from beeai_framework.tools.code import ShellTool, get_shell_backend
from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.filesystem import FileEditTool, FileReadTool, get_file_backend


class _StubExit:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code


class _StubOutput:
    def __init__(self, text: str) -> None:
        self.output = text
        self.truncated = False


class _StubCreateTerminalResponse:
    def __init__(self, tid: str) -> None:
        self.terminal_id = tid


class _StubConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.store: dict[str, str] = {}

    async def read_text_file(self, **kwargs: Any) -> ReadTextFileResponse:
        self.calls.append(("read_text_file", kwargs))
        return ReadTextFileResponse(content=self.store.get(kwargs["path"], ""))

    async def write_text_file(self, **kwargs: Any) -> None:
        self.calls.append(("write_text_file", kwargs))
        self.store[kwargs["path"]] = kwargs["content"]

    async def create_terminal(self, **kwargs: Any) -> _StubCreateTerminalResponse:
        self.calls.append(("create_terminal", kwargs))
        return _StubCreateTerminalResponse("term-1")

    async def wait_for_terminal_exit(self, **kwargs: Any) -> _StubExit:
        self.calls.append(("wait_for_terminal_exit", kwargs))
        return _StubExit(exit_code=0)

    async def terminal_output(self, **kwargs: Any) -> _StubOutput:
        self.calls.append(("terminal_output", kwargs))
        return _StubOutput("tests pass\n")

    async def release_terminal(self, **kwargs: Any) -> None:
        self.calls.append(("release_terminal", kwargs))


@pytest.fixture
def bridge_with_stub() -> tuple[FsBridge, _StubConn]:
    bridge = FsBridge()
    conn = _StubConn()
    bridge.bind(conn)  # type: ignore[arg-type]
    bridge.set_capabilities(
        ClientCapabilities(
            fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),  # type: ignore[call-arg]
            terminal=True,  # type: ignore[call-arg]
        )
    )
    return bridge, conn


@pytest.mark.unit
@pytest.mark.asyncio
async def test_file_backend_routes_through_bridge(bridge_with_stub: tuple[FsBridge, _StubConn]) -> None:
    bridge, conn = bridge_with_stub
    conn.store["/tmp/x.txt"] = "hello\n"
    backend = ACPFileBackend(bridge)
    tok = _active_session.set("sess-1")
    try:
        content = await backend.read_text("/tmp/x.txt")
        await backend.write_text("/tmp/y.txt", "new\n")
    finally:
        _active_session.reset(tok)

    assert content == "hello\n"
    methods = [c[0] for c in conn.calls]
    assert methods == ["read_text_file", "write_text_file"]
    assert conn.store["/tmp/y.txt"] == "new\n"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shell_backend_routes_through_terminal(bridge_with_stub: tuple[FsBridge, _StubConn]) -> None:
    bridge, conn = bridge_with_stub
    backend = ACPShellBackend(bridge)
    tok = _active_session.set("sess-1")
    try:
        result = await backend.run(command=["pytest", "-q"])
    finally:
        _active_session.reset(tok)

    assert result["exit_code"] == 0
    assert result["stdout"] == "tests pass\n"
    # Chained create → wait → output → release
    assert [c[0] for c in conn.calls] == [
        "create_terminal",
        "wait_for_terminal_exit",
        "terminal_output",
        "release_terminal",
    ]
    assert conn.calls[0][1]["command"] == "pytest"
    assert conn.calls[0][1]["args"] == ["-q"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shell_backend_requires_capability() -> None:
    bridge = FsBridge()
    bridge.bind(_StubConn())  # type: ignore[arg-type]
    bridge.set_capabilities(ClientCapabilities())  # no terminal capability
    backend = ACPShellBackend(bridge)
    tok = _active_session.set("sess-1")
    try:
        with pytest.raises(ToolError, match="terminal capability"):
            await backend.run(command=["echo", "hi"])
    finally:
        _active_session.reset(tok)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_installs_acp_backends(bridge_with_stub: tuple[FsBridge, _StubConn]) -> None:
    """End-to-end: with `ACPZedIOContext` active, the generic tools hit the stub."""
    bridge, conn = bridge_with_stub
    conn.store["/tmp/foo.txt"] = "content\n"

    tok = _active_session.set("sess-1")
    try:
        with ACPZedIOContext(bridge):
            # File read goes through bridge
            r = await FileReadTool().run({"path": "/tmp/foo.txt"})
            assert r.get_text_content() == "content\n"

            # File edit routes read+write through bridge
            await FileEditTool().run({"mode": "replace", "path": "/tmp/foo.txt", "old": "content", "new": "updated"})
            assert conn.store["/tmp/foo.txt"] == "updated\n"

            # Shell goes through terminal
            shell = await ShellTool().run({"command": ["echo", "hi"]})
            assert shell.to_json_safe()["exit_code"] == 0
            assert shell.to_json_safe()["stdout"] == "tests pass\n"

        # After the context exits, defaults are restored (local backends are instances
        # of the local classes, not the ACP ones).
        from beeai_framework.tools.code import LocalShellBackend
        from beeai_framework.tools.filesystem import LocalFileBackend

        assert isinstance(get_shell_backend(), LocalShellBackend)
        assert isinstance(get_file_backend(), LocalFileBackend)
    finally:
        _active_session.reset(tok)
