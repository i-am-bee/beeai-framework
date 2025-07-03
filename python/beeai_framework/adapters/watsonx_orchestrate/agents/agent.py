# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

import httpx

import beeai_framework.adapters.watsonx_orchestrate._api as watsonx_orchestrate_api
from beeai_framework.adapters.watsonx_orchestrate._api import ChatCompletionRequestBody, ChatCompletionResponse
from beeai_framework.adapters.watsonx_orchestrate._utils import (
    beeai_message_to_watsonx_orchestrate_message,
    map_watsonx_orchestrate_agent_input_to_bee_messages,
    watsonx_orchestrate_message_to_beeai_message,
)
from beeai_framework.adapters.watsonx_orchestrate.agents.types import WatsonxOrchestrateAgentRunOutput
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.errors import AgentError
from beeai_framework.backend import AssistantMessage
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.lists import flatten
from beeai_framework.utils.strings import to_safe_word


class WatsonxOrchestrateAgent(BaseAgent[WatsonxOrchestrateAgentRunOutput]):
    def __init__(
        self,
        *,
        agent_id: str,
        instance_url: str,
        memory: BaseMemory | None = None,
        name: str | None = None,
        thread_id: str | None = None,
        api_key: str | None = None,
        auth_type: Literal["iam", "jwt"] = "iam",
    ) -> None:
        super().__init__()
        self._memory = memory or UnconstrainedMemory()
        self._instance_url = instance_url
        self._name = name or f"watsonx_orchestrate_agent_{agent_id}"
        self._thread_id: str = thread_id or str(uuid.uuid4().hex)
        self._auth_type = auth_type
        self._api_key = api_key
        self._agent_id = agent_id

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, input: str | list[str] | AnyMessage | list[AnyMessage] | None, *, signal: AbortSignal | None = None
    ) -> Run[WatsonxOrchestrateAgentRunOutput]:
        async def handler(_: RunContext) -> WatsonxOrchestrateAgentRunOutput:
            async with self._create_client() as client:
                input_messages = map_watsonx_orchestrate_agent_input_to_bee_messages(input)
                await self.memory.add_many(input_messages)

                # TODO: support streaming

                response = await client.post(
                    f"{self._agent_id}/chat/completions",
                    json=ChatCompletionRequestBody(
                        messages=flatten(
                            [beeai_message_to_watsonx_orchestrate_message(msg) for msg in self.memory.messages]
                        ),
                        stream=False,
                    ).model_dump(),
                )
                response.raise_for_status()

                response_data = ChatCompletionResponse.model_validate(response.json())
                result = watsonx_orchestrate_message_to_beeai_message(
                    watsonx_orchestrate_api.ChatMessage(
                        role=response_data.choices[-1].message.role,
                        content=response_data.choices[-1].message.content,
                        tool_calls=None,
                        tool_call_id=None,
                    )
                )
                assert isinstance(result, AssistantMessage), "Result must be instanceof AssistantMessage"
                await self.memory.add(result)
                return WatsonxOrchestrateAgentRunOutput(
                    result=result,
                    raw=response_data.model_dump(),
                )

        return self._to_run(
            handler,
            signal=signal,
            run_params={
                "prompt": input,
                "signal": signal,
            },
        )

    async def check_agent_exists(
        self,
    ) -> None:
        async with self._create_client() as client:
            response = await client.get("agents")
            response.raise_for_status()
            agents = response.json()
            agent = any(agent["name"] == self._name for agent in agents)
            if not agent:
                raise AgentError(f"Agent with ID {self._agent_id} does not exist.")

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    async def clone(self) -> "WatsonxOrchestrateAgent":
        cloned = WatsonxOrchestrateAgent(
            agent_id=self._agent_id,
            instance_url=self._instance_url,
            memory=await self.memory.clone(),
            name=self._name,
            thread_id=self._thread_id,
            api_key=self._api_key,
            auth_type=self._auth_type,
        )
        cloned.emitter = await self.emitter.clone()
        return cloned

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["watsonx_orchestrate", "agent", to_safe_word(self._name)],
            creator=self,
        )

    @asynccontextmanager
    async def _create_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        async with httpx.AsyncClient(
            base_url=f"{self._instance_url}/v1/orchestrate",
            headers=exclude_none(
                {
                    "IAM-API-KEY": self._api_key if self._auth_type == "iam" else None,
                    "Authorization": f"Bearer {self._api_key}" if self._auth_type == "jwt" else None,
                }
            ),
        ) as client:
            yield client
