# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("acp", reason="Optional module [acp_zed] not installed.")

from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
)

from beeai_framework.adapters.acp_zed.serve.agent import FsBridge, _active_session
from beeai_framework.adapters.acp_zed.serve.io import ACPZedIOContext
from beeai_framework.utils.io import io_confirm, io_read


class _StubConn:
    def __init__(self, *, allow: bool) -> None:
        self.allow = allow
        self.calls: list[dict[str, Any]] = []

    async def request_permission(self, **kwargs: Any) -> RequestPermissionResponse:
        self.calls.append(kwargs)
        if self.allow:
            return RequestPermissionResponse(
                outcome=AllowedOutcome(outcome="selected", option_id="allow")  # type: ignore[call-arg]
            )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confirm_allow_routes_through_request_permission() -> None:
    bridge = FsBridge()
    conn = _StubConn(allow=True)
    bridge.bind(conn)  # type: ignore[arg-type]

    tok = _active_session.set("sess-1")
    try:
        with ACPZedIOContext(bridge):
            result = await io_confirm("Are you sure?", data={"tool": "shell"})
    finally:
        _active_session.reset(tok)

    assert result is True
    assert len(conn.calls) == 1
    call = conn.calls[0]
    assert call["session_id"] == "sess-1"
    assert [opt.option_id for opt in call["options"]] == ["allow", "deny"]
    assert call["tool_call"].raw_input == {"tool": "shell"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confirm_deny_returns_false() -> None:
    bridge = FsBridge()
    bridge.bind(_StubConn(allow=False))  # type: ignore[arg-type]
    tok = _active_session.set("sess-1")
    try:
        with ACPZedIOContext(bridge):
            result = await io_confirm("Run this?", data={})
    finally:
        _active_session.reset(tok)
    assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_raises_under_context() -> None:
    bridge = FsBridge()
    bridge.bind(_StubConn(allow=True))  # type: ignore[arg-type]
    tok = _active_session.set("sess-1")
    try:
        with ACPZedIOContext(bridge), pytest.raises(RuntimeError, match="io_read is not supported"):
            await io_read("anything: ")
    finally:
        _active_session.reset(tok)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_context_restores_previous_handlers() -> None:
    """After exiting the context, the previous (default) handlers are restored."""
    bridge = FsBridge()
    bridge.bind(_StubConn(allow=True))  # type: ignore[arg-type]
    tok = _active_session.set("sess-1")
    try:
        with ACPZedIOContext(bridge):
            assert await io_confirm("x", data={}) is True
        # Outside the context, io_read would block on stdin; just confirm it's a
        # different function now (the default one, not our _read).
        from beeai_framework.utils.io import _storage

        restored = _storage.get()
        assert restored.read.__name__ != "_read"
    finally:
        _active_session.reset(tok)
