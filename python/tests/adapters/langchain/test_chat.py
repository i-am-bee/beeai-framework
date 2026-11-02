# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from beeai_framework.adapters.langchain.backend.chat import LangChainChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import tool


@tool
def weather(city: str) -> str:
    """Return the weather for a city."""
    return "sunny"


class _ToolCallingFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_keeps_the_chat_model_options() -> None:
    middleware = GlobalTrajectoryMiddleware()
    model = LangChainChatModel(
        GenericFakeChatModel(messages=iter([])),
        allow_parallel_tool_calls=True,
        ignore_parallel_tool_calls=True,
        supports_top_level_unions=False,
        fix_invalid_tool_calls=False,
        retry_on_empty_response=False,
        allow_prompt_caching=False,
        middlewares=[middleware],
    )

    cloned = await model.clone()

    assert cloned.allow_parallel_tool_calls is True
    assert cloned.ignore_parallel_tool_calls is True
    assert cloned.supports_top_level_unions is False
    assert cloned.fix_invalid_tool_calls is False
    assert cloned.retry_on_empty_response is False
    assert cloned.allow_prompt_caching is False
    assert cloned.middlewares == [middleware]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_accepts_parallel_tool_calls_like_the_original() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "weather", "args": {"city": "Paris"}},
            {"id": "call_2", "name": "weather", "args": {"city": "Rome"}},
        ],
    )
    model = LangChainChatModel(
        _ToolCallingFakeChatModel(messages=iter([message, message])), allow_parallel_tool_calls=True
    )
    cloned = await model.clone()

    for chat_model in (model, cloned):
        output = await chat_model.run(
            [UserMessage("Weather in Paris and Rome?")], tools=[weather], tool_choice="required"
        )

        assert [call.args for call in output.get_tool_calls()] == ['{"city": "Paris"}', '{"city": "Rome"}']


@pytest.mark.unit
def test_transform_output_reports_cached_tokens() -> None:
    model = LangChainChatModel(GenericFakeChatModel(messages=iter([])))
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_token_details": {"cache_read": 60, "cache_creation": 30},
        },
    )

    output = model._transform_output({"output": message, "output_structured": None})  # type: ignore[typeddict-item]

    assert output.usage is not None
    assert output.usage.prompt_tokens == 100
    assert output.usage.cached_prompt_tokens == 60
    assert output.usage.cached_creation_tokens == 30


@pytest.mark.unit
def test_transform_output_reports_anthropic_cache_writes_by_ttl() -> None:
    model = LangChainChatModel(GenericFakeChatModel(messages=iter([])))
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            # What langchain-anthropic reports when the response has cache writes.
            "input_token_details": {
                "cache_read": 60,
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 20,
                "ephemeral_1h_input_tokens": 10,
            },
        },
    )

    output = model._transform_output({"output": message, "output_structured": None})  # type: ignore[typeddict-item]

    assert output.usage is not None
    assert output.usage.cached_prompt_tokens == 60
    assert output.usage.cached_creation_tokens == 30


@pytest.mark.unit
def test_transform_output_without_token_details_has_no_cached_tokens() -> None:
    model = LangChainChatModel(GenericFakeChatModel(messages=iter([])))
    message = AIMessage(content="hi", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})

    output = model._transform_output({"output": message, "output_structured": None})  # type: ignore[typeddict-item]

    assert output.usage is not None
    assert output.usage.cached_prompt_tokens == 0
    assert output.usage.cached_creation_tokens == 0
