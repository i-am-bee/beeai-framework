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

from beeai_framework.cancellation import AbortSignal
from beeai_framework.tools.tool import AnyTool


class BaseAgentRunOptions(BaseModel):
    signal: AbortSignal | None = None


class AgentExecutionConfig(BaseModel):
    total_max_retries: int | None = None
    max_retries_per_step: int | None = None
    max_iterations: int | None = None


class AgentMeta(BaseModel):
    name: str
    description: str
    tools: list[InstanceOf[AnyTool]]
    extra_description: str | None = None
