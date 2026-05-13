# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import pytest
from litellm import ModelResponse as LiteLLMModelResponse

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.message import AssistantMessage


class DummyLiteLLMChatModel(LiteLLMChatModel):
    def __init__(self) -> None:
        super().__init__("test-model", provider_id="openai")

    @property
    def provider_id(self) -> ProviderName:
        return "openai"


class ModelDumpNamespace(SimpleNamespace):
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {key: value for key, value in vars(self).items() if value is not None}


class ModelResponse(dict[str, Any]):
    choices: list[Any]
    id: str

    def __init__(self, message: Any) -> None:
        super().__init__(model="test-model", usage=None)
        self.id = "response-id"
        self.choices = [SimpleNamespace(message=message, finish_reason="tool_calls")]


@pytest.mark.unit
def test_transform_output_preserves_text_with_tool_calls() -> None:
    message = ModelDumpNamespace(
        content="I will call the tool.",
        tool_calls=[
            SimpleNamespace(
                id="call-id",
                function=SimpleNamespace(name="search", arguments='{"query":"bee"}'),
            )
        ],
    )

    output = DummyLiteLLMChatModel()._transform_output(cast(LiteLLMModelResponse, ModelResponse(message)))

    assert len(output.output) == 2
    text_output = cast(AssistantMessage, output.output[0])
    tool_output = cast(AssistantMessage, output.output[1])
    assert text_output.text == "I will call the tool."
    tool_call = tool_output.get_tool_calls()[0]
    assert tool_call.id == "call-id"
    assert tool_call.tool_name == "search"
    assert tool_call.args == '{"query":"bee"}'


@pytest.mark.unit
def test_transform_output_handles_missing_reasoning_content() -> None:
    message = ModelDumpNamespace(content=None, tool_calls=None, role="assistant")

    output = DummyLiteLLMChatModel()._transform_output(cast(LiteLLMModelResponse, ModelResponse(message)))

    assert len(output.output) == 1
    assert output.output[0].text == ""
