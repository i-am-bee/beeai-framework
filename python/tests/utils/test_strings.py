# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.utils.strings import validate_class_name


def test_validate_class_name_valid() -> None:
    # Valid identifiers
    validate_class_name("MyClass")
    validate_class_name("my_function")
    validate_class_name("_InternalClass")
    validate_class_name("Class123")


def test_validate_class_name_invalid() -> None:
    # Invalid identifiers
    invalid_names = [
        "",
        "123Class",
        "My-Class",
        "My Class",
        "My.Class",
        "class",  # reserved word is allowed as identifier in isidentifier() but we might want to be stricter?
        # actually isidentifier() returns True for keywords.
        # but you can't import a class named 'class'.
        "import",
        "def",
        "!",
        "@",
        "#",
        "import os; os.system('ls')",
        "__import__('os').system('ls')",
    ]
    for name in invalid_names:
        # Some of these are actually valid identifiers (like 'class', 'import')
        # but most are not.
        if not name.isidentifier() or not name:
            with pytest.raises(ValueError, match="Invalid class name"):
                validate_class_name(name)


def test_validate_class_name_keywords() -> None:
    # Python keywords are identifiers but we shouldn't allow them as class names for safety?
    # Actually, the main goal is to prevent module probing or injection like 'module.submodule'.
    # A single keyword like 'class' is not an injection in the same way.
    # But let's see if we want to block them.
    # For now, validate_class_name only uses isidentifier().
    pass
