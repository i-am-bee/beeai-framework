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

from beeai_framework.backend import AssistantMessage, ToolMessage
from beeai_framework.memory import BaseMemory
from beeai_framework.utils.lists import find_index


def extract_last_tool_call_pair(memory: BaseMemory) -> tuple[AssistantMessage, ToolMessage] | None:
    tool_call_index = find_index(
        memory.messages,
        lambda msg: bool(isinstance(msg, AssistantMessage) and msg.get_tool_calls()),
        reverse_traversal=True,
        fallback=-1,
    )
    if tool_call_index < 0:
        return None

    tool_call: AssistantMessage = memory.messages[tool_call_index]  # type: ignore

    tool_response_index = find_index(
        memory.messages,
        lambda msg: bool(
            isinstance(msg, ToolMessage) and msg.get_tool_results()[0].tool_call_id == tool_call.get_tool_calls()[0].id
        ),
        reverse_traversal=True,
        fallback=-1,
    )

    if tool_response_index < 0:
        return None

    tool_response: ToolMessage = memory.messages[tool_response_index]  # type: ignore
    return tool_call, tool_response
