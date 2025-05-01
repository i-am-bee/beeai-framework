# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable

from acp_sdk.models.models import Message as AcpMessage
from acp_sdk.models.models import Metadata
from acp_sdk.server.agent import Agent as AcpBaseAgent
from acp_sdk.server.context import Context
from acp_sdk.server.types import RunYield, RunYieldResume


class AcpAgent(AcpBaseAgent):
    """A wrapper for a BeeAI agent to be used with the ACP server."""

    def __init__(
        self,
        fn: Callable[[list[AcpMessage], Context], AsyncGenerator[RunYield, RunYieldResume]],
        name: str,
        description: str | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self._name = name
        self._description = description
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description or ""

    @property
    def metadata(self) -> Metadata:
        return self._metadata or Metadata()

    async def run(self, input: list[AcpMessage], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
        try:
            gen: AsyncGenerator[RunYield, RunYieldResume] = self.fn(input, context)
            value = None
            while True:
                value = yield await gen.asend(value)
        except StopAsyncIteration:
            pass


class ACPAdapter(ABC):
    @abstractmethod
    def to_acp(self) -> AcpAgent:
        pass
