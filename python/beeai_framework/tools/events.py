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

from typing import Generic, TypeVar

from pydantic import BaseModel, InstanceOf

from beeai_framework.tools import ToolError
from beeai_framework.tools.types import ToolOutput, ToolRunOptions

TInput = TypeVar("TInput", bound=BaseModel)


class ToolStartEvent(BaseModel, Generic[TInput]):
    input: TInput
    options: ToolRunOptions | None = None


class ToolSuccessEvent(BaseModel, Generic[TInput]):
    output: InstanceOf[ToolOutput]
    input: TInput
    options: ToolRunOptions | None = None


class ToolErrorEvent(BaseModel, Generic[TInput]):
    error: InstanceOf[ToolError]
    input: TInput
    options: ToolRunOptions | None = None
