# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.utils.strings import validate_class_name


def test_validate_class_name_valid() -> None:
    validate_class_name("MyClass")
    validate_class_name("my_function")
    validate_class_name("_InternalClass")
    validate_class_name("Class123")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "123Class",
        "My-Class",
        "My Class",
        "My.Class",
        "!",
        "@",
        "#",
        "import os; os.system('ls')",
        "__import__('os').system('ls')",
    ],
)
def test_validate_class_name_invalid(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid class name"):
        validate_class_name(name)


@pytest.mark.parametrize("name", ["class", "import", "def", "for", "return", "None", "True", "False"])
def test_validate_class_name_rejects_keywords(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid class name"):
        validate_class_name(name)


def test_validate_class_name_rejects_none() -> None:
    with pytest.raises(ValueError, match="Invalid class name"):
        validate_class_name(None)
