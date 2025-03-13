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


from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.message import AnyMessage
from beeai_framework.cancellation import AbortSignal


class RemoteAgentRunInput(BaseModel):
    prompt: str | None = None


class RemoteAgentRunOptions(BaseModel):
    signal: AbortSignal | None = None


class RemoteAgentRunOutput(BaseModel):
    result: InstanceOf[AnyMessage]


class RemoteAgentInput(BaseModel):
    agent_name: str
    url: str
