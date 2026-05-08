# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from litellm import ModelResponse, ModelResponseStream
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Message,
    StreamingChoices,
)

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import MessageReasoningContent, MessageTextContent, MessageToolCallContent


class _TestLiteLLMChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "openai"

    def __init__(self) -> None:
        super().__init__("gpt-4o", provider_id="openai")


@pytest.fixture(scope="module")
def model() -> _TestLiteLLMChatModel:
    return _TestLiteLLMChatModel()


def _make_tool_call(
    call_id: str = "call_1",
    name: str = "search",
    arguments: str = '{"q": "test"}',
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function={"name": name, "arguments": arguments},
    )


class TestTransformOutput:
    @pytest.mark.unit
    def test_content_only(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-1",
            model="test",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="hello", role="assistant"))],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        msg = result.output[0]
        texts = msg.get_by_type(MessageTextContent)
        assert len(texts) == 1
        assert texts[0].text == "hello"
        assert msg.get_by_type(MessageToolCallContent) == []

    @pytest.mark.unit
    def test_reasoning_content_only(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-2",
            model="test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content=None, role="assistant", reasoning_content="thinking deeply"),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        reasoning = result.output[0].get_by_type(MessageReasoningContent)
        assert len(reasoning) == 1
        assert reasoning[0].text == "thinking deeply"

    @pytest.mark.unit
    def test_tool_calls_only(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-3",
            model="test",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(content=None, role="assistant", tool_calls=[_make_tool_call()]),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        msg = result.output[0]
        tool_calls = msg.get_by_type(MessageToolCallContent)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "search"
        assert tool_calls[0].args == '{"q": "test"}'
        assert msg.get_by_type(MessageTextContent) == []

    @pytest.mark.unit
    def test_content_and_reasoning_content(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-4",
            model="test",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="the answer is 42", role="assistant", reasoning_content="let me think"),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        reasoning = result.output[0].get_by_type(MessageReasoningContent)
        texts = result.output[0].get_by_type(MessageTextContent)
        assert len(reasoning) == 1
        assert reasoning[0].text == "let me think"
        assert len(texts) == 1
        assert texts[0].text == "the answer is 42"

    @pytest.mark.unit
    def test_tool_calls_with_content(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-5",
            model="test",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content="I'll search for that",
                        role="assistant",
                        tool_calls=[_make_tool_call()],
                    ),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        msg = result.output[0]
        texts = msg.get_by_type(MessageTextContent)
        tool_calls = msg.get_by_type(MessageToolCallContent)
        assert len(texts) == 1
        assert texts[0].text == "I'll search for that"
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "search"

    @pytest.mark.unit
    def test_tool_calls_with_reasoning(self, model: _TestLiteLLMChatModel) -> None:
        """Core regression test: reasoning must not be dropped when tool calls are present."""
        response = ModelResponse(
            id="chatcmpl-6",
            model="test",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        reasoning_content="I need to search for this",
                        tool_calls=[_make_tool_call()],
                    ),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        msg = result.output[0]
        reasoning = msg.get_by_type(MessageReasoningContent)
        tool_calls = msg.get_by_type(MessageToolCallContent)
        assert len(reasoning) == 1
        assert reasoning[0].text == "I need to search for this"
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "search"

    @pytest.mark.unit
    def test_all_three_present(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-7",
            model="test",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content="searching now",
                        role="assistant",
                        reasoning_content="I should look this up",
                        tool_calls=[_make_tool_call()],
                    ),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        msg = result.output[0]
        reasoning = msg.get_by_type(MessageReasoningContent)
        texts = msg.get_by_type(MessageTextContent)
        tool_calls = msg.get_by_type(MessageToolCallContent)
        assert len(reasoning) == 1
        assert reasoning[0].text == "I should look this up"
        assert len(texts) == 1
        assert texts[0].text == "searching now"
        assert len(tool_calls) == 1

    @pytest.mark.unit
    def test_empty_update(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponse(
            id="chatcmpl-8",
            model="test",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content=None, role="assistant"))],
        )
        result = model._transform_output(response)

        assert result.output == []

    @pytest.mark.unit
    def test_streaming_reasoning_and_tool_calls(self, model: _TestLiteLLMChatModel) -> None:
        response = ModelResponseStream(
            id="chatcmpl-9",
            model="test",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content=None,
                        reasoning_content="analyzing the query",
                        tool_calls=[
                            {
                                "index": 0,
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q": "test"}'},
                            }
                        ],
                    ),
                )
            ],
        )
        result = model._transform_output(response)

        assert len(result.output) == 1
        msg = result.output[0]
        reasoning = msg.get_by_type(MessageReasoningContent)
        tool_calls = msg.get_by_type(MessageToolCallContent)
        assert len(reasoning) == 1
        assert reasoning[0].text == "analyzing the query"
        assert len(tool_calls) == 1

    @pytest.mark.unit
    def test_no_reasoning_attribute(self, model: _TestLiteLLMChatModel) -> None:
        """Litellm deletes reasoning_content attr when None; getattr must handle this."""
        response = ModelResponse(
            id="chatcmpl-10",
            model="test",
            choices=[Choices(finish_reason="stop", index=0, message=Message(content="plain", role="assistant"))],
        )
        msg = response.choices[0].message
        assert not hasattr(msg, "reasoning_content")

        result = model._transform_output(response)
        texts = result.output[0].get_by_type(MessageTextContent)
        assert len(texts) == 1
        assert texts[0].text == "plain"
