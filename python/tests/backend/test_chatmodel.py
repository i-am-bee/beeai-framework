# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from pydantic import BaseModel

from beeai_framework.adapters.amazon_bedrock.backend.chat import AmazonBedrockChatModel
from beeai_framework.adapters.anthropic.backend.chat import AnthropicChatModel
from beeai_framework.adapters.azure_openai.backend.chat import AzureOpenAIChatModel
from beeai_framework.adapters.groq.backend.chat import GroqChatModel
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel
from beeai_framework.adapters.vertexai.backend.chat import VertexAIChatModel
from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel
from beeai_framework.adapters.xai.backend.chat import XAIChatModel
from beeai_framework.backend.chat import (
    ChatModel,
)
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    CustomMessage,
    UserMessage,
)
from beeai_framework.backend.types import (
    ChatModelInput,
    ChatModelOutput,
    ChatModelStructureInput,
    ChatModelStructureOutput,
)
from beeai_framework.cancellation import AbortSignal
from beeai_framework.context import RunContext
from beeai_framework.errors import AbortError

"""
Utility functions and classes
"""


class ReverseWordsDummyModel(ChatModel):
    """Dummy model that simply reverses every word in a UserMessages"""

    model_id = "reversed_words_model"
    provider_id = "ollama"

    def reverse_message_words(self, messages: list[AnyMessage]) -> list[str]:
        reversed_words_messages = []
        for message in messages:
            if isinstance(message, UserMessage):
                reversed_words = " ".join(word[::-1] for word in message.text.split())
                reversed_words_messages.append(reversed_words)
        return reversed_words_messages

    async def _create(self, input: ChatModelInput, _: RunContext) -> ChatModelOutput:
        reversed_words_messages = self.reverse_message_words(input.messages)
        return ChatModelOutput(messages=[AssistantMessage(w) for w in reversed_words_messages])

    async def _create_stream(self, input: ChatModelInput, context: RunContext) -> AsyncGenerator[ChatModelOutput]:
        words = self.reverse_message_words(input.messages)[0].split(" ")

        last = len(words) - 1
        for count, chunk in enumerate(words):
            if context.signal.aborted:
                break
            await asyncio.sleep(5)
            yield ChatModelOutput(messages=[AssistantMessage(f"{chunk} " if count != last else chunk)])

    async def _create_structure(self, input: ChatModelStructureInput[Any], run: RunContext) -> ChatModelStructureOutput:
        reversed_words_messages = self.reverse_message_words(input.messages)
        response_object = {"reversed": "".join(reversed_words_messages)}
        return ChatModelStructureOutput(object=response_object)


@pytest_asyncio.fixture
def reverse_words_chat() -> ChatModel:
    return ReverseWordsDummyModel()


@pytest.fixture
def chat_messages_list() -> list[AnyMessage]:
    user_message = UserMessage("tell me something interesting")
    custom_message = CustomMessage(role="custom", content="this is a custom message")
    return [user_message, custom_message]


"""
Unit Tests
"""


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_create(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    response = await reverse_words_chat.create(messages=chat_messages_list)

    assert len(response.messages) == 1
    assert all(isinstance(message, AssistantMessage) for message in response.messages)
    assert response.messages[0].get_texts()[0].text == "llet em gnihtemos gnitseretni"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_structure(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    class ReverseWordsSchema(BaseModel):
        reversed: str

    reverse_words_chat = ReverseWordsDummyModel()
    response = await reverse_words_chat.create_structure(schema=ReverseWordsSchema, messages=chat_messages_list)

    ReverseWordsSchema.model_validate(response.object)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_stream(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    response = await reverse_words_chat.create(messages=chat_messages_list, stream=True)

    assert len(response.messages) == 4
    assert all(isinstance(message, AssistantMessage) for message in response.messages)
    assert "".join([m.get_texts()[0].text for m in response.messages]) == "llet em gnihtemos gnitseretni"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_abort(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    with pytest.raises(AbortError):
        await reverse_words_chat.create(
            messages=chat_messages_list,
            stream=True,
            abort_signal=AbortSignal.timeout(1),
        )


@pytest.mark.unit
def test_chat_model_from(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ollama with Llama model and base_url specified in code
    ollama_chat_model = ChatModel.from_name("ollama:llama3.1", {"base_url": "http://somewhere:12345"})
    assert isinstance(ollama_chat_model, OllamaChatModel)
    assert ollama_chat_model._settings["base_url"] == "http://somewhere:12345/v1"

    # Ollama with Granite model and base_url specified in env var
    monkeypatch.setenv("OLLAMA_API_BASE", "http://somewhere-else:12345")
    ollama_chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")
    assert isinstance(ollama_chat_model, OllamaChatModel)
    assert ollama_chat_model._settings["base_url"] == "http://somewhere-else:12345/v1"

    # Watsonx with Granite model and settings specified in code
    watsonx_chat_model = ChatModel.from_name(
        "watsonx:ibm/granite-3-8b-instruct",
        {
            "url": "http://somewhere",
            "project_id": "proj_id_123",
            "api_key": "api_key_123",
        },
    )
    assert isinstance(watsonx_chat_model, WatsonxChatModel)
    assert watsonx_chat_model._settings["url"] == "http://somewhere"
    assert watsonx_chat_model._settings["project_id"] == "proj_id_123"
    assert watsonx_chat_model._settings["api_key"] == "api_key_123"

    # Watsonx with Granite model and settings specified in env vars
    monkeypatch.setenv("WATSONX_URL", "http://somewhere-else")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "proj_id_456")
    monkeypatch.setenv("WATSONX_API_KEY", "api_key_456")

    watsonx_chat_model = ChatModel.from_name("watsonx:ibm/granite-3-8b-instruct")
    assert isinstance(watsonx_chat_model, WatsonxChatModel)

    openai_chat_model = ChatModel.from_name("openai:gpt-4o")
    assert isinstance(openai_chat_model, OpenAIChatModel)

    groq_chat_model = ChatModel.from_name("groq:gemma2-9b-it")
    assert isinstance(groq_chat_model, GroqChatModel)

    xai_chat_model = ChatModel.from_name("xai:grok-2")
    assert isinstance(xai_chat_model, XAIChatModel)

    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "myproject")
    vertexai_chat_model = ChatModel.from_name("vertexai:gemini-2.0-flash-lite-001")
    assert isinstance(vertexai_chat_model, VertexAIChatModel)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "apikey")
    anthropic_chat_model = ChatModel.from_name("anthropic:claude-3-haiku-20240307")
    assert isinstance(anthropic_chat_model, AnthropicChatModel)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "secret1")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret2")
    monkeypatch.setenv("AWS_REGION", "region1")
    amazon_bedrock_chat_model = ChatModel.from_name("amazon_bedrock:meta.llama3-8b-instruct-v1:0")
    assert isinstance(amazon_bedrock_chat_model, AmazonBedrockChatModel)

    monkeypatch.setenv("AZURE_API_KEY", "secret")
    monkeypatch.setenv("AZURE_API_BASE", "base")
    monkeypatch.setenv("AZURE_API_VERSION", "version")
    azure_openai_chat_model = ChatModel.from_name("azure_openai:gpt-4o")
    assert isinstance(azure_openai_chat_model, AzureOpenAIChatModel)
