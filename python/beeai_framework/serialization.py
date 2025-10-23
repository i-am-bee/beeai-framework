# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable
from typing import Protocol, Self, TypeVar

from beeai_framework.utils.cloneable import Cloneable

T = TypeVar("T")


class Serializable(Cloneable, Protocol[T]):
    """Lightweight serialization protocol shared by stateful components."""

    def create_snapshot(self) -> T | Awaitable[T]: ...

    def load_snapshot(self, snapshot: T) -> None | Awaitable[None]: ...

    @classmethod
    def from_snapshot(cls, snapshot: T) -> Self | Awaitable[Self]: ...
