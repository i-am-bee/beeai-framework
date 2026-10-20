# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for MiniMax chat model.

These tests require a valid MINIMAX_API_KEY environment variable.
Run with: pytest tests/adapters/minimax/test_minimax_integration.py -v
"""

import os

import pytest

from beeai_framework.adapters.minimax.backend.chat import MiniMaxChatModel
from beeai_framework.backend.message import UserMessage

pytestmark = pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


@pytest.fixture(params=["MiniMax-M3.1-Flash", "MiniMax-M3"])
def chat_model(request: pytest.FixtureRequest) -> MiniMaxChatModel:
    return MiniMaxChatModel(request.param)


class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    @pytest.mark.asyncio
    async def test_simple_chat(self, chat_model: MiniMaxChatModel) -> None:
        output = await chat_model.run(
            [UserMessage("What is 2 + 2? Reply with just the number.")],
        )
        text = output.get_text_content()
        assert "4" in text

    @pytest.mark.asyncio
    async def test_streaming(self, chat_model: MiniMaxChatModel) -> None:
        output = await chat_model.run(
            [UserMessage("Count from 1 to 3.")],
            stream=True,
        )
        text = output.get_text_content()
        assert "1" in text
