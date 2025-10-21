"""Lightweight serialization protocol shared by stateful components."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, Self, TypeVar

SnapshotT = TypeVar("SnapshotT")


class Serializable(Protocol[SnapshotT]):
    """Minimal contract for classes that can persist and restore their state."""

    def create_snapshot(self) -> SnapshotT | Awaitable[SnapshotT]: ...

    def load_snapshot(self, snapshot: SnapshotT) -> None | Awaitable[None]: ...

    @classmethod
    def from_snapshot(cls, snapshot: SnapshotT) -> Self | Awaitable[Self]: ...

    async def clone(self) -> Self: ...
