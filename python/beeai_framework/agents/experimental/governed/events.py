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


from pydantic import BaseModel

from beeai_framework.agents.experimental.governed.types import GovernedAgentRequest, GovernedAgentRunState
from beeai_framework.backend import ChatModelOutput


class GovernedAgentStartEvent(BaseModel):
    state: GovernedAgentRunState
    request: GovernedAgentRequest


class GovernedAgentSuccessEvent(BaseModel):
    state: GovernedAgentRunState
    response: ChatModelOutput


ability_agent_event_types: dict[str, type] = {
    "start": GovernedAgentStartEvent,
    "success": GovernedAgentSuccessEvent,
}
