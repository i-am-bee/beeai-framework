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

from beeai_framework.backend.backend import Backend
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.errors import BackendError, ChatModelError, EmbeddingModelError, MessageError
from beeai_framework.backend.events import (
    ChatModelErrorEvent,
    ChatModelNewTokenEvent,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
    EmbeddingModelErrorEvent,
    EmbeddingModelStartEvent,
    EmbeddingModelSuccessEvent,
)
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    AssistantMessageContent,
    CustomMessage,
    CustomMessageContent,
    Message,
    MessageImageContent,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
    Role,
    SystemMessage,
    ToolMessage,
    UserMessage,
    UserMessageContent,
)
from beeai_framework.backend.types import (
    ChatModelOutput,
    ChatModelParameters,
    ChatModelStructureOutput,
    EmbeddingModelOutput,
)

__all__ = [
    "AnyMessage",
    "AssistantMessage",
    "AssistantMessageContent",
    "Backend",
    "BackendError",
    "ChatModel",
    "ChatModelError",
    "ChatModelErrorEvent",
    "ChatModelNewTokenEvent",
    "ChatModelOutput",
    "ChatModelParameters",
    "ChatModelStartEvent",
    "ChatModelStructureOutput",
    "ChatModelSuccessEvent",
    "CustomMessage",
    "CustomMessage",
    "CustomMessageContent",
    "EmbeddingModel",
    "EmbeddingModelError",
    "EmbeddingModelErrorEvent",
    "EmbeddingModelOutput",
    "EmbeddingModelStartEvent",
    "EmbeddingModelSuccessEvent",
    "Message",
    "MessageError",
    "MessageImageContent",
    "MessageTextContent",
    "MessageToolCallContent",
    "MessageToolResultContent",
    "Role",
    "SystemMessage",
    "ToolMessage",
    "UserMessage",
    "UserMessage",
    "UserMessageContent",
]
