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
