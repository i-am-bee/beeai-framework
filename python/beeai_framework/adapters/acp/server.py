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


from acp_sdk.server.server import Server as AcpHttpServer
from pydantic import BaseModel

from beeai_framework.adapters.acp.adapter import ACPAdapter, AcpAgent
from beeai_framework.serve.server import Server


class AcpServerConfig(BaseModel):
    """Configuration for the AcpServer."""

    host: str = "127.0.0.1"
    port: int = 8000


class AcpServer(Server[AcpServerConfig]):
    def __init__(self) -> None:
        super().__init__()
        self.server = AcpHttpServer()

    def serve(self, *, config: AcpServerConfig | None = None) -> None:
        if not config:
            config = AcpServerConfig()
        agents = [self.conver_to_acp_agent(agent) for agent in self.agents]
        self.server.register(*agents)
        self.server.run(
            host=config.host,
            port=config.port,
        )

    def conver_to_acp_agent(self, agent: ACPAdapter) -> AcpAgent:
        """Convert a BeeAI agent to an ACP agent."""

        if not isinstance(agent, ACPAdapter):
            raise ValueError(f"Agent {agent} is not an ACPAdapter")

        return agent.to_acp()
