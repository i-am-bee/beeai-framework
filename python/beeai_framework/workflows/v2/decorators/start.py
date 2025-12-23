# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from beeai_framework.backend.message import AnyMessage
from beeai_framework.workflows.v2.workflow import Workflow

S = TypeVar("S", bound=Workflow)
AnyCoroutine = Coroutine[Any, Any, Any]
StartMethod = Callable[[S, list[AnyMessage]], AnyCoroutine]


def start(func: StartMethod[S]) -> StartMethod[S]:
    func._is_step = True  # type: ignore[attr-defined]
    func._is_start = True  # type: ignore[attr-defined]
    return func
