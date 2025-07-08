# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

T = TypeVar("T")


def identity(value: T) -> T:
    return value
