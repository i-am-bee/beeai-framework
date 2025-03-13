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

from collections.abc import Callable
from types import NoneType

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.errors import FrameworkError


class ChatModelNewTokenEvent(BaseModel):
    value: InstanceOf[ChatModelOutput]
    abort: Callable[[], None]


class ChatModelSuccessEvent(BaseModel):
    value: InstanceOf[ChatModelOutput]


class ChatModelStartEvent(BaseModel):
    input: InstanceOf[ChatModelInput]


class ChatModelErrorEvent(BaseModel):
    input: InstanceOf[ChatModelInput]
    error: InstanceOf[FrameworkError]


chat_model_event_types: dict[str, type] = {
    "new_token": ChatModelNewTokenEvent,
    "success": ChatModelSuccessEvent,
    "start": ChatModelStartEvent,
    "error": ChatModelErrorEvent,
    "finish": NoneType,
}
