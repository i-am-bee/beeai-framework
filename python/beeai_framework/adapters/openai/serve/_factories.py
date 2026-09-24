# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import json
from collections.abc import AsyncIterable
from typing import Any

from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve._types import OpenAIEvent
from beeai_framework.adapters.openai.serve.server import OpenAIServerMetadata
from beeai_framework.agents import BaseAgent
from beeai_framework.agents.react import ReActAgent, ReActAgentSuccessEvent, ReActAgentUpdateEvent
from beeai_framework.agents.react.types import ReActAgentIterationResult
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.backend import AnyMessage, ChatModel, ChatModelNewTokenEvent, ChatModelSuccessEvent
from beeai_framework.runnable import Runnable
from beeai_framework.utils.lists import find_index


def _runnable_factory(runnable: Runnable[Any], *, metadata: OpenAIServerMetadata | None = None) -> OpenAIModel:
    if metadata is None:
        metadata = {}

    name = metadata.get(
        "name",
        runnable.meta.name
        if isinstance(runnable, BaseAgent)
        else runnable.model_id
        if isinstance(runnable, ChatModel)
        else runnable.__class__.__name__,
    )

    return OpenAIModel(runnable, model_id=name)


def _react_factory(agent: ReActAgent, *, metadata: OpenAIServerMetadata | None = None) -> OpenAIModel:
    if metadata is None:
        metadata = {}

    async def stream(input: list[AnyMessage]) -> AsyncIterable[OpenAIEvent]:
        cloned_agent = await agent.clone()
        async for data, _ in cloned_agent.run(input):
            if (
                isinstance(data, ReActAgentUpdateEvent)
                and isinstance(data.data, ReActAgentIterationResult)
                and data.update.key == "final_answer"
                and data.data.final_answer is None
            ):
                yield OpenAIEvent(text=data.update.value)
            if isinstance(data, ReActAgentSuccessEvent):
                yield OpenAIEvent(finish_reason=data.iterations[-1].raw.finish_reason)

    return OpenAIModel(agent, model_id=metadata.get("name") or agent.meta.name, stream=stream)


def _requirement_agent_factory(agent: RequirementAgent, *, metadata: OpenAIServerMetadata | None = None) -> OpenAIModel:
    if metadata is None:
        metadata = {}

    async def stream(input: list[AnyMessage]) -> AsyncIterable[OpenAIEvent]:
        cloned_agent = await agent.clone()
        last_msg = None
        async for data, _ in cloned_agent.run(input):
            messages = data.state.memory.messages
            if last_msg is None:
                last_msg = messages[-1]

            cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
            for message in messages[cur_index + 1 :]:
                last_msg = message
                if isinstance(message, FinalAnswerTool):
                    continue
                if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                    yield OpenAIEvent(text=data.state.answer.text, type="message", append=False)
                    continue

                yield OpenAIEvent(
                    text=json.dumps([m.model_dump() for m in message.content]), type="custom_tool_call", append=False
                )

    return OpenAIModel(agent, model_id=metadata.get("name") or agent.meta.name, stream=stream)


def _chat_model_factory(llm: ChatModel, *, metadata: OpenAIServerMetadata | None = None) -> OpenAIModel:
    if metadata is None:
        metadata = {}

    async def stream(input: list[AnyMessage]) -> AsyncIterable[OpenAIEvent]:
        cloned_llm = await llm.clone()
        async for data, event in cloned_llm.run(input, stream=True):
            if isinstance(data, ChatModelNewTokenEvent):
                yield OpenAIEvent(text=data.value.last_message.text)
            if isinstance(event, ChatModelSuccessEvent):
                yield OpenAIEvent(append=False, finish_reason=data.value.finish_reason)

    return OpenAIModel(llm, model_id=metadata.get("name") or llm.model_id, stream=stream)
