# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.openai.serve.responses._types import (
    ResponsesRequestInputMessage,
)
from beeai_framework.adapters.openai.serve.responses._utils import (
    openai_input_to_beeai_message,
)
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageTextContent,
    SystemMessage,
    UserMessage,
)

@pytest.mark.unit
@pytest.mark.parametrize(
    ("role", "content", "expected_message_type"),
    [
        ("user", "hello user", UserMessage),
        ("system", "system content", SystemMessage),
        ("developer", "dev content", SystemMessage),
        ("assistant", "assistant content", AssistantMessage),
    ],
)
def test_openai_input_to_beeai_message_roles(
    role: str, content: str, expected_message_type: type
) -> None:
    msg = ResponsesRequestInputMessage(role=role, content=content)
    result = openai_input_to_beeai_message(msg)

    assert isinstance(result, expected_message_type)
    assert isinstance(result.content[0], MessageTextContent)
    assert result.content[0].text == content


@pytest.mark.unit
def test_none_content_defaults_to_empty_string() -> None:
    msg = ResponsesRequestInputMessage(role="user", content=None)
    result = openai_input_to_beeai_message(msg)

    assert isinstance(result, UserMessage)
    assert isinstance(result.content[0], MessageTextContent)
    assert result.content[0].text == ""


@pytest.mark.unit
def test_openai_input_to_beeai_message_invalid_role() -> None:
    msg = ResponsesRequestInputMessage.model_construct(role="unknown", content="x")

    with pytest.raises(ValueError) as exc:
        openai_input_to_beeai_message(msg)

    assert str(exc.value) == "Invalid role: unknown"
