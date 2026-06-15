# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest

from beeai_framework.tools.code.sandbox import SandboxTool, SandboxToolCreateError


@pytest.mark.asyncio
async def test_sandbox_error_messages_joined_correctly() -> None:
    error_messages = ["SyntaxError on line 1", "Missing return statement"]

    with patch(
        "beeai_framework.tools.code.sandbox.PythonTool.call_code_interpreter",
        new_callable=AsyncMock,
        return_value={"error_messages": error_messages},
    ), pytest.raises(SandboxToolCreateError) as exc_info:
        await SandboxTool.from_source_code(
            url="http://fake-url",
            source_code="def broken(): pass",
        )

    assert "SyntaxError on line 1\nMissing return statement" in str(exc_info.value)
    