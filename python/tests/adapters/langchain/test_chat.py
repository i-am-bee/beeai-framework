# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGenerationChunk

from beeai_framework.adapters.langchain.backend.chat import LangChainChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.backend.types import ChatModelInput
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response_metadata", "expected"),
    [
        ({"done_reason": "stop"}, "stop"),
        ({"finish_reason": "tool_calls"}, "tool_calls"),
        ({"stop_reason": "max_tokens"}, "max_tokens"),
        ({"stopReason": "end_turn"}, "end_turn"),
        ({}, None),
    ],
)
def test_transform_output_reports_the_finish_reason(response_metadata: dict[str, str], expected: str | None) -> None:
    model = LangChainChatModel(GenericFakeChatModel(messages=iter([])))
    message = AIMessage(content="hi", response_metadata=response_metadata)

    output = model._transform_output({"output": message, "output_structured": None})  # type: ignore[typeddict-item]

    assert output.finish_reason == expected


class _NamedFakeChatModel(GenericFakeChatModel):
    model: str = "granite4:latest"


@pytest.mark.unit
def test_model_id_uses_the_model_attribute() -> None:
    model = LangChainChatModel(_NamedFakeChatModel(messages=iter([])))

    assert model.model_id == "granite4:latest"


class _StopRecordingFakeChatModel(GenericFakeChatModel):
    seen_stop: Any = "unset"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self.seen_stop = stop
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_stop", "call_stop", "expected"),
    [(["END"], None, ["END"]), (["END"], ["STOP"], ["STOP"]), (None, ["STOP"], ["STOP"])],
)
async def test_stop_sequences_fall_back_to_model_parameters(
    model_stop: list[str] | None, call_stop: list[str] | None, expected: list[str]
) -> None:
    fake = _StopRecordingFakeChatModel(messages=iter([AIMessage(content="hi")]))
    model = LangChainChatModel(fake)
    model.parameters.stop_sequences = model_stop

    await model.run([UserMessage("hello")], stop_sequences=call_stop)

    assert fake.seen_stop == expected


class _StopFieldFakeChatModel(_StopRecordingFakeChatModel):
    stop: list[str] | None = None

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        # LangChain integrations fall back to their own "stop" field when the call passes none.
        return super()._generate(messages, stop=stop or self.stop, run_manager=run_manager, **kwargs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stop_sequences_keep_the_langchain_model_stop() -> None:
    fake = _StopFieldFakeChatModel(messages=iter([AIMessage(content="hi")]), stop=["END"])

    await LangChainChatModel(fake).run([UserMessage("hello")])

    assert fake.stop == ["END"]
    assert fake.seen_stop == ["END"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_stop_sequences_do_not_stick_to_the_langchain_model() -> None:
    fake = _StopFieldFakeChatModel(messages=iter([AIMessage(content="hi")]), stop=["END"])

    await LangChainChatModel(fake).run([UserMessage("hello")], stop_sequences=["STOP"])

    assert fake.stop == ["END"]


@pytest.mark.unit
@pytest.mark.parametrize(("call_stop", "expected"), [(None, ["END"]), (["STOP"], ["STOP"])])
def test_prepared_model_carries_the_stop_sequences(call_stop: list[str] | None, expected: list[str]) -> None:
    fake = _StopFieldFakeChatModel(messages=iter([]), stop=["END"])
    input = ChatModelInput(messages=[UserMessage("hello")], stop_sequences=call_stop)

    prepared = LangChainChatModel(fake)._prepare_model(input)

    # The structured output path does not forward invoke kwargs, so the model itself must hold them.
    assert prepared.first.stop == expected  # type: ignore[attr-defined]
    assert fake.stop == ["END"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_emits_each_chunk() -> None:
    model = LangChainChatModel(GenericFakeChatModel(messages=iter([AIMessage(content="one two three")])))
    tokens: list[str] = []

    output = await model.run([UserMessage("Count")], stream=True).on(
        "new_token", lambda data, _: tokens.append(data.value.get_text_content())
    )

    assert [token for token in tokens if token] == ["one", " ", "two", " ", "three"]
    assert output.get_text_content() == "one two three"


class _UsageStreamingFakeChatModel(GenericFakeChatModel):
    # LangChain reports usage as per-chunk deltas that add up to the total
    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        for text, input_tokens, output_tokens in (("Hello", 10, 1), (" world", 0, 2)):
            usage = UsageMetadata(
                input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens
            )
            yield ChatGenerationChunk(message=AIMessageChunk(content=text, usage_metadata=usage, id="msg_1"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_adds_up_usage_across_chunks() -> None:
    model = LangChainChatModel(_UsageStreamingFakeChatModel(messages=iter([])))

    output = await model.run([UserMessage("Hi")], stream=True)

    assert output.get_text_content() == "Hello world"
    assert output.usage.prompt_tokens == 10
    assert output.usage.completion_tokens == 3
    assert output.usage.total_tokens == 13
