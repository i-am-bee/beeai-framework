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
"""Util functions for the toolkit chat module."""

from typing import Any

from pydantic import BaseModel

from beeai_framework.backend.message import AnyMessage, UserMessage
from beeai_framework.plugins.schemas import DataContext
from beeai_framework.template import PromptTemplate
from beeai_framework.toolkit.chat.constants import USER_PROMPT_VARIABLE
from beeai_framework.toolkit.chat.template import FewShotChatPromptTemplate


def get_formatted_prompt(
    data: str | BaseModel | dict[str, Any] | UserMessage, template: PromptTemplate | FewShotChatPromptTemplate
) -> UserMessage:
    """Returns a prompt as a UserMessage.
    Args:
        data: the data to be formatted as a UserMessage.
        template: the template object to format the input.
    Returns:
        A formatted user message.
    """
    datum: dict[str, Any] = {}
    if isinstance(data, str):
        datum = {USER_PROMPT_VARIABLE: data}
    elif isinstance(data, dict):
        datum = data
    elif isinstance(data, BaseModel):
        datum = data.model_dump()
    elif isinstance(data, UserMessage):
        datum = {USER_PROMPT_VARIABLE: data.text}
    query = template.render(datum)
    return UserMessage(content=query)


def get_input_messages(cxt: DataContext, template: PromptTemplate | FewShotChatPromptTemplate) -> list[AnyMessage]:
    """Returns a prompt as a set of formatted messages.
    Args:
        cxt: the data to be formatted as a UserMessage.
        template: the template object to format the input.
    Returns:
        A list of formatted user messages.
    """
    input_messages: list[AnyMessage] = []
    data = cxt.data
    if isinstance(data, list) and len(data) > 0:
        for msg in data:
            if isinstance(msg, UserMessage):
                message = get_formatted_prompt(msg, template)
                input_messages.append(message)
            else:
                input_messages.append(msg)
    elif isinstance(data, str | dict | BaseModel | UserMessage):
        message = get_formatted_prompt(data, template)
        input_messages.append(message)
    return input_messages
