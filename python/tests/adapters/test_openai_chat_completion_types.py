# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.openai.serve.chat_completion._types import (
    ChatToolFunctionDefinition,
)


@pytest.mark.unit
def test_arguments_accept_a_json_string() -> None:
    """OpenAI sends tool-call arguments as a JSON string, and parse_arguments
    exists to decode it. It only registers when field_validator is the outer
    decorator."""
    assert ChatToolFunctionDefinition(name="f", arguments='{"a": 1}').arguments == {"a": 1}


@pytest.mark.unit
def test_arguments_accept_a_dict_unchanged() -> None:
    assert ChatToolFunctionDefinition(name="f", arguments={"a": 1}).arguments == {"a": 1}


@pytest.mark.unit
def test_invalid_json_arguments_raise() -> None:
    with pytest.raises(ValueError):
        ChatToolFunctionDefinition(name="f", arguments="not json at all")
