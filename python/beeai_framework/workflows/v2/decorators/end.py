# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Coroutine
from typing import Any, Concatenate, ParamSpec, TypeVar

from beeai_framework.backend.message import AnyMessage
from beeai_framework.workflows.v2.workflow import Workflow

S = TypeVar("S", bound=Workflow)
P = ParamSpec("P")

EndMethod = Callable[Concatenate[S, P], Coroutine[Any, Any, list[AnyMessage]]]


def end(func: EndMethod[S, P]) -> EndMethod[S, P]:
    func._is_step = True  # type: ignore[attr-defined]
    func._is_end = True  # type: ignore[attr-defined]
    return func
