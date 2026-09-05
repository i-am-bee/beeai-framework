# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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


def test_hol_guard_fails_closed_when_unavailable_by_default() -> None:
    middleware = HOLGuardMiddleware()
    assert middleware._availability_result("unavailable") == (False, "unavailable")


def test_hol_guard_can_fail_open_when_explicitly_configured() -> None:
    middleware = HOLGuardMiddleware(fail_closed=False)
    assert middleware._availability_result("unavailable") == (True, "unavailable")


def test_hol_guard_formats_json_policy_action() -> None:
    reason = HOLGuardMiddleware._decision_reason(
        stdout=b'{"policy_action":"review"}',
        stderr=b"",
        returncode=1,
    )
    assert reason == "policy action: review"


def test_hol_guard_requires_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be greater than zero"):
        HOLGuardMiddleware(timeout=0)
