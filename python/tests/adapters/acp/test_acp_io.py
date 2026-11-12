# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _require_acp_io() -> Any:
    pytest.importorskip("acp_sdk")
    from beeai_framework.adapters.acp.serve.io import ACPIOContext

    return ACPIOContext


@pytest.mark.asyncio
async def test_confirm_shows_the_tool_input() -> None:
    acp_io_context = _require_acp_io()
    context = MagicMock()
    requests: list[Any] = []

    async def yield_async(request: Any) -> Any:
        requests.append(request)
        return MagicMock(message="yes")

    context.yield_async = yield_async
    io_context = acp_io_context(context)
    assert await io_context._confirm("The agent wants to use the weather tool.", data={"city": "Prague"}) is True
    assert len(requests) == 1
    content = requests[0].message.parts[0].content
    assert content.startswith("The agent wants to use the weather tool.")
    assert '"city": "Prague"' in content


@pytest.mark.asyncio
async def test_confirm_without_the_tool_input_keeps_the_prompt() -> None:
    acp_io_context = _require_acp_io()
    context = MagicMock()
    requests: list[Any] = []

    async def yield_async(request: Any) -> Any:
        requests.append(request)
        return MagicMock(message="yes")

    context.yield_async = yield_async
    io_context = acp_io_context(context)
    assert await io_context._confirm("Confirm: ") is True
    content = requests[0].message.parts[0].content
    assert content == "Confirm: "
