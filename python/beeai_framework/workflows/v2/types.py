# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Coroutine
from typing import Any, Literal

from pydantic import BaseModel

AsyncMethod = Callable[..., Coroutine[Any, Any, Any]]

ExecutionCondition = Literal["and", "or"]


class AsyncMethodSet(BaseModel):
    methods: list[str] = []
    condition: ExecutionCondition = "and"
