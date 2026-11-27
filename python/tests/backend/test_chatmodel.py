# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from google.oauth2.credentials import Credentials
from pydantic import BaseModel, create_model

from beeai_framework.adapters.amazon_bedrock import AmazonBedrockChatModel
from beeai_framework.adapters.anthropic import AnthropicChatModel
from beeai_framework.adapters.azure_openai import AzureOpenAIChatModel
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.adapters.groq import GroqChatModel
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.adapters.vertexai import VertexAIChatModel
from beeai_framework.adapters.watsonx import WatsonxChatModel
from beeai_framework.adapters.xai import XAIChatModel
from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    ChatModel,
    ChatModelOutput,
    CustomMessage,
    UserMessage,
)
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.cache import SlidingCache, UnconstrainedCache
from beeai_framework.context import RunContext
from beeai_framework.errors import AbortError
from beeai_framework.tools import tool
from beeai_framework.utils import AbortSignal

"""
Utility functions and classes
"""


class ReverseWordsDummyModel(ChatModel):
    """Dummy model that simply reverses every word in a UserMessages"""

    model_id = "reversed_words_model"
    # pyrefly: ignore [bad-override]
    provider_id = "ollama"

    def reverse_message_words(self, messages: list[AnyMessage]) -> list[str]:
        reversed_words_messages = []
        for message in messages:
            if isinstance(message, UserMessage):
                reversed_words = " ".join(word[::-1] for word in message.text.split())
                reversed_words_messages.append(reversed_words)
        return reversed_words_messages

    # pyrefly: ignore [bad-param-name-override]
    async def _create(self, input: ChatModelInput, _: RunContext) -> ChatModelOutput:
        reversed_words_messages = self.reverse_message_words(input.messages)

        output_structured: Any = {"reversed": "".join(reversed_words_messages)} if input.response_format else None
        if isinstance(input.response_format, type) and issubclass(input.response_format, BaseModel):
            # pyrefly: ignore [missing-attribute]
            output_structured = input.response_format.model_validate(output_structured)

        return ChatModelOutput(
            output=[AssistantMessage(w) for w in reversed_words_messages],
            output_structured=output_structured,
        )

    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, context: RunContext) -> AsyncGenerator[ChatModelOutput]:
        words = self.reverse_message_words(input.messages)[0].split(" ")

        last = len(words) - 1
        for count, chunk in enumerate(words):
            if context.signal.aborted:
                break
            await asyncio.sleep(5)
            yield ChatModelOutput(output=[AssistantMessage(f"{chunk} " if count != last else chunk)])


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
    response = await reverse_words_chat.run(chat_messages_list)

    assert len(response.output) == 1
    assert all(isinstance(message, AssistantMessage) for message in response.output)
    assert response.output[0].get_texts()[0].text == "llet em gnihtemos gnitseretni"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_structure(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    class ReverseWordsSchema(BaseModel):
        reversed: str

    reverse_words_chat = ReverseWordsDummyModel()
    response = await reverse_words_chat.run(chat_messages_list, response_format=ReverseWordsSchema)
    assert isinstance(response.output_structured, ReverseWordsSchema)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_stream(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    response = await reverse_words_chat.run(chat_messages_list, stream=True)

    assert len(response.output) == 1
    assert len(response.output[0].content) == 4
    assert all(isinstance(message, AssistantMessage) for message in response.output)
    assert response.output[0].text == "llet em gnihtemos gnitseretni"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_abort(reverse_words_chat: ChatModel, chat_messages_list: list[AnyMessage]) -> None:
    with pytest.raises(AbortError):
        await reverse_words_chat.run(
            chat_messages_list,
            stream=True,
            signal=AbortSignal.timeout(1),
        )


@pytest.mark.unit
def test_chat_model_from(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ollama with Llama model and base_url specified in code
    ollama_chat_model = ChatModel.from_name("ollama:llama3.1", {"base_url": "http://somewhere:12345"})
    assert isinstance(ollama_chat_model, OllamaChatModel)
    assert ollama_chat_model._settings["base_url"] == "http://somewhere:12345/v1"

    # Ollama with Granite model and base_url specified in env var
    monkeypatch.setenv("OLLAMA_API_BASE", "http://somewhere-else:12345")
    ollama_chat_model = ChatModel.from_name("ollama:granite4:micro")
    assert isinstance(ollama_chat_model, OllamaChatModel)
    assert ollama_chat_model._settings["base_url"] == "http://somewhere-else:12345/v1"

    # Watsonx with Granite model and settings specified in code
    watsonx_chat_model = ChatModel.from_name(
        "watsonx:ibm/granite-3-3-8b-instruct",
        {
            "base_url": "http://somewhere",
            "project_id": "proj_id_123",
            "api_key": "api_key_123",
        },
    )
    assert isinstance(watsonx_chat_model, WatsonxChatModel)
    assert watsonx_chat_model._settings["base_url"] == "http://somewhere"
    assert watsonx_chat_model._settings["project_id"] == "proj_id_123"
    assert watsonx_chat_model._settings["api_key"] == "api_key_123"

    # Watsonx with Granite model and settings specified in env vars
    monkeypatch.setenv("WATSONX_URL", "http://somewhere-else")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "proj_id_456")
    monkeypatch.setenv("WATSONX_API_KEY", "api_key_456")

    watsonx_chat_model = ChatModel.from_name("watsonx:ibm/granite-3-3-8b-instruct")
    assert isinstance(watsonx_chat_model, WatsonxChatModel)

    openai_chat_model = ChatModel.from_name("openai:gpt-4o", api_key="test")
    assert isinstance(openai_chat_model, OpenAIChatModel)

    groq_chat_model = ChatModel.from_name("groq:gemma2-9b-it", api_key="test")
    assert isinstance(groq_chat_model, GroqChatModel)

    xai_chat_model = ChatModel.from_name("xai:grok-2", api_key="test")
    assert isinstance(xai_chat_model, XAIChatModel)

    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "myproject")
    vertexai_chat_model = ChatModel.from_name("vertexai:gemini-2.0-flash-lite-001", vertex_location="test")
    assert isinstance(vertexai_chat_model, VertexAIChatModel)
    # Ref: https://github.com/BerriAI/litellm/issues/17696
    assert vertexai_chat_model.allow_prompt_caching is False

    vertexai_chat_model_explicit = ChatModel.from_name(
        "vertexai:gemini-2.0-flash-lite-001", vertex_location="test", allow_prompt_caching=True
    )
    assert vertexai_chat_model_explicit.allow_prompt_caching is True

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


@pytest.mark.unit
def test_chat_model_does_not_share_the_settings_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two models built from one settings template must not share (or mutate) that dict."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    template: dict[str, Any] = {"base_url": "https://proxy.example/v1"}
    first = OpenAIChatModel("gpt-4o", settings=template, api_key="key-one")
    second = OpenAIChatModel("gpt-4o", settings=template, api_key="key-two")

    assert first._settings is not second._settings
    assert first._settings["api_key"] == "key-one"
    assert second._settings["api_key"] == "key-two"
    assert template == {"base_url": "https://proxy.example/v1"}


class FailThenFixDummyModel(ChatModel):
    """Dummy model that returns invalid tool-call JSON on the first attempt,
    then valid JSON on subsequent attempts. Used to verify that the cache
    is invalidated on ChatModelToolCallError so retries hit the model again."""

    model_id = "fail_then_fix_model"
    # pyrefly: ignore [bad-override]
    provider_id = "ollama"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._attempt = 0

    # pyrefly: ignore [bad-param-name-override]
    async def _create(self, input: ChatModelInput, _: RunContext) -> ChatModelOutput:
        self._attempt += 1
        if self._attempt == 1:
            return ChatModelOutput(output=[AssistantMessage('{"bad": "data"}')])
        return ChatModelOutput(output=[AssistantMessage('{"name": "tool_add", "parameters": {"a": 1, "b": 2}}')])

    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, context: RunContext) -> AsyncGenerator[ChatModelOutput]:
        raise NotImplementedError
        yield  # pyrefly: ignore [invalid-yield]


@tool()
def tool_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider,model_id",
    [
        ("gemini", "gemini-3-flash-preview"),
        ("vertex_ai", "gemini-3-flash-preview"),
        ("vertex_ai", "publishers/google/models/gemini-3-flash-preview"),
    ],
)
@pytest.mark.asyncio
async def test_litellm_guided_tool_request(provider: str, model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if provider == "gemini":
        model = GeminiChatModel(model_id, api_key="test", tool_choice_support={"auto"})
    else:
        credentials = {
            "type": "authorized_user",
            "token": "test",
            "client_id": "test",
            "client_secret": "test",
            "refresh_token": "test",
        }
        monkeypatch.setattr(Credentials, "refresh", lambda self, request: None)
        model = VertexAIChatModel(
            model_id, project="test", location="us-central1", credentials=credentials, tool_choice_support={"auto"}
        )

    @tool()
    def think(thought: str) -> str:
        """Think about the problem."""
        return thought

    bodies = []

    async def capture_request(client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        bodies.append(json.loads(request.content))
        # Stop at the wire: no model response or background success callbacks are needed.
        return httpx.Response(400, request=request, json={"error": {"message": "Request captured"}})

    monkeypatch.setattr(httpx.AsyncClient, "send", capture_request)
    with pytest.raises(ChatModelError) as error:
        await model.run(
            [UserMessage("add")],
            tools=[think, tool_add],
            tool_choice="required",
            parallel_tool_calls=False,
            max_retries=0,
        )
    assert "Request captured" in error.value.explain()
    (body,) = bodies
    assert "tools" not in body
    generation = body["generationConfig"]
    schema = generation.get("response_json_schema") or generation["response_schema"]
    names = [variant["properties"]["name"] for variant in schema["anyOf"]]
    assert [name["enum"] for name in names] == [["think"], ["tool_add"]]
    assert all("const" not in name for name in names)


@pytest.fixture
def completion_request(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    # These cases inspect the request only; stop before any response is parsed.
    completion = AsyncMock(side_effect=ChatModelError("Request captured"))
    monkeypatch.setattr("beeai_framework.adapters.litellm.chat.acompletion", completion)
    return completion


@pytest.mark.unit
@pytest.mark.asyncio
async def test_gemini_preserves_native_tools_and_custom_response_format(completion_request: AsyncMock) -> None:
    model = GeminiChatModel("gemini-3-flash-preview", api_key="test", tool_choice_support={"auto"})
    messages: list[AnyMessage] = [UserMessage("add")]
    with pytest.raises(ChatModelError, match="Request captured"):
        await model.run(messages, tools=[tool_add], tool_choice="auto", max_retries=0)
    request = completion_request.await_args_list[-1].kwargs
    assert request["tools"][0]["function"]["name"] == tool_add.name
    assert "response_format" not in request

    custom = {"type": "json_schema", "json_schema": {"name": "Custom", "schema": {"type": "string", "const": "x"}}}
    with pytest.raises(ChatModelError, match="Request captured"):
        await model.run(messages, tools=[tool_add], tool_choice="required", response_format=custom, max_retries=0)
    assert completion_request.await_args_list[-1].kwargs["response_format"] == custom


@pytest.mark.unit
@pytest.mark.asyncio
async def test_guided_schema_enum_is_limited_to_gemini_adapters(completion_request: AsyncMock) -> None:
    models: list[tuple[ChatModel, str]] = [
        (OpenAIChatModel("gpt-4o", api_key="test", tool_choice_support={"auto"}), "const"),
        (GeminiChatModel("gemini-3-flash-preview", api_key="test", tool_choice_support={"auto"}), "enum"),
        (
            VertexAIChatModel(
                "gemini-3-flash-preview", project="test", location="us-central1", tool_choice_support={"auto"}
            ),
            "enum",
        ),
        (
            VertexAIChatModel("non-gemini-test", project="test", location="us-central1", tool_choice_support={"auto"}),
            "const",
        ),
    ]

    for model, expected_keyword in models:
        with pytest.raises(ChatModelError, match="Request captured"):
            await model.run([UserMessage("add 1 and 2")], tools=[tool_add], tool_choice="required", max_retries=0)
        name_schema = completion_request.await_args_list[-1].kwargs["response_format"]["json_schema"]["schema"][
            "properties"
        ]["name"]
        assert expected_keyword in name_schema
        assert ({"enum", "const"} - {expected_keyword}).isdisjoint(name_schema)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_cleared_on_tool_call_error() -> None:
    model = FailThenFixDummyModel(
        cache=SlidingCache(size=10),
        model_supports_tool_calling=False,
    )

    result = await model.run(
        [UserMessage("add 1 and 2")],
        tools=[tool_add],
        tool_choice="required",
        max_retries=3,
    )

    assert model._attempt == 2, (
        f"Expected model to be called twice (initial + 1 retry), got {model._attempt} calls. "
        "Cache was not cleared after ChatModelToolCallError."
    )
    assert result.get_tool_calls(), "Expected result to contain tool calls"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_separates_response_formats_with_the_same_name(chat_messages_list: list[AnyMessage]) -> None:
    model = ReverseWordsDummyModel(cache=UnconstrainedCache())
    first_schema = create_model("Answer", reversed=(str, ...))
    second_schema = create_model("Answer", reversed=(str, ...), language=(str, "en"))

    first = await model.run(chat_messages_list, response_format=first_schema)
    second = await model.run(chat_messages_list, response_format=second_schema)

    assert isinstance(first.output_structured, first_schema)
    assert isinstance(second.output_structured, second_schema)
    assert await model.cache.size() == 2
