# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal
from enum import IntEnum
from typing import Any

import pytest

from beeai_framework.tools import JSONToolOutput


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, "false"),
        (0, "0"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (True, "true"),
        (1, "1"),
        (-1, "-1"),
        (1.5, "1.5"),
        ({"count": 0, "enabled": False}, '{"count": 0, "enabled": false}'),
        ([0, False], "[0, false]"),
        ("value", '"value"'),
        (" \t\n", '" \\t\\n"'),
    ],
)
def test_json_output_preserves_scalar_content(value: Any, expected: str) -> None:
    output = JSONToolOutput(value)

    assert not output.is_empty()
    assert output.get_text_content() == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "", [], {}, (), set()])
def test_json_output_empty_values(value: Any) -> None:
    assert JSONToolOutput(value).is_empty()


class FalsyValue:
    def __bool__(self) -> bool:
        return False


class FalsyNumber(IntEnum):
    ZERO = 0


@pytest.mark.unit
@pytest.mark.parametrize("value", [FalsyValue(), FalsyNumber.ZERO, Decimal(0)])
def test_json_output_retains_other_falsy_values(value: Any) -> None:
    assert JSONToolOutput(value).is_empty()
