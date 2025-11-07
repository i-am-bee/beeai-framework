# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeAlias, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

MaybeAsync: TypeAlias = Callable[P, R] | Callable[P, Awaitable[R]]
