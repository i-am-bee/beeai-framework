# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.backend import AssistantMessage, ChatModelOutput
from beeai_framework.backend.types import ChatModelUsage


@pytest.mark.unit
def test_merge_keeps_cached_token_counts() -> None:
    target = ChatModelOutput(
        output=[AssistantMessage("Hello")],
        usage=ChatModelUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            cached_prompt_tokens=8,
            cached_creation_tokens=64,
        ),
    )
    other = ChatModelOutput(
        output=[],
        usage=ChatModelUsage(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            cached_prompt_tokens=96,
            cached_creation_tokens=24,
        ),
    )

    target.merge(other)

    assert target.usage.prompt_tokens == 120
    assert target.usage.completion_tokens == 30
    assert target.usage.total_tokens == 150
    assert target.usage.cached_prompt_tokens == 96
    assert target.usage.cached_creation_tokens == 64


@pytest.mark.unit
def test_from_chunks_keeps_cached_token_counts() -> None:
    chunks = [
        ChatModelOutput(output=[AssistantMessage("Hel")]),
        ChatModelOutput(output=[AssistantMessage("lo")]),
        ChatModelOutput(
            output=[],
            usage=ChatModelUsage(
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
                cached_prompt_tokens=96,
                cached_creation_tokens=24,
            ),
        ),
    ]

    result = ChatModelOutput.from_chunks(chunks)

    assert result.usage.total_tokens == 150
    assert result.usage.cached_prompt_tokens == 96
    assert result.usage.cached_creation_tokens == 24
