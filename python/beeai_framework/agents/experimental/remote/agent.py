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

import json
from contextlib import AsyncExitStack

from acp import ClientSession
from acp.client.sse import sse_client
from acp.types import (
    AgentRunProgressNotification,
    AgentRunProgressNotificationParams,
    RunAgentRequest,
    RunAgentRequestParams,
    RunAgentResult,
    ServerNotification,
)
from beeai_sdk.utils.api import send_request_with_notifications

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.experimental.remote.types import (
    RemoteAgentInput,
    RemoteAgentRunInput,
    RemoteAgentRunOptions,
    RemoteAgentRunOutput,
)
from beeai_framework.backend.message import AssistantMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.utils.models import ModelLike


class RemoteAgent(BaseAgent[RemoteAgentRunInput, RemoteAgentRunOptions, RemoteAgentRunOutput]):
    def __init__(self, agent: str, url: str) -> None:
        self.input = RemoteAgentInput(agent=agent, url=url)
        self.emitter = Emitter.root().child(
            namespace=["agent", "remote"],
            creator=self,
        )
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    @property
    def memory(self) -> BaseMemory:
        raise NotImplementedError()

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        raise NotImplementedError()

    async def _run(
        self,
        run_input: ModelLike[RemoteAgentRunInput],
        options: ModelLike[RemoteAgentRunOptions] | None,
        context: RunContext,
    ) -> RemoteAgentRunOutput:
        if not self.session:
            await self.connect_to_server()

        try:
            input = json.loads(run_input.get("prompt"))
            async for message in send_request_with_notifications(
                url_or_session=self.session,
                req=RunAgentRequest(
                    method="agents/run",
                    params=RunAgentRequestParams(name=self.input.agent, input=input),
                ),
                result_type=RunAgentResult,
            ):
                match message:
                    case ServerNotification(
                        root=AgentRunProgressNotification(params=AgentRunProgressNotificationParams(delta=delta))
                    ):
                        await context.emitter.emit(
                            "update",
                            {
                                "update": {
                                    "key": "update",
                                    "value": delta,
                                }
                            },
                        )
                    case RunAgentResult() as result:
                        await context.emitter.emit(
                            "update",
                            {
                                "update": {
                                    "key": "final_answer",
                                    "value": result.output,
                                }
                            },
                        )
                        return RemoteAgentRunOutput(result=AssistantMessage(json.dumps(result.output)))
        except Exception as e:
            raise AgentError("Error during agent's run", cause=e)
        finally:
            await self.exit_stack.aclose()
            self.session = None

    async def connect_to_server(
        self,
    ) -> None:
        try:
            sse_transport = await self.exit_stack.enter_async_context(sse_client(url=self.input.url))
            self.read, self.write = sse_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
            await self.session.initialize()
            response = await self.session.list_agents()
            agents = response.agents

            agent = any(agent.name == self.input.agent for agent in agents)
            if not agent:
                raise AgentError(f"Agent {self.input.agent} is not registered in the platform")
        except Exception as e:
            raise AgentError("Can't connect to Beeai Platform.", cause=e)
