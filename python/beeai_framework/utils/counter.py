# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from beeai_framework.errors import FrameworkError


class RetryCounter:
    def __init__(self, error_type: type[BaseException], max_retries: int = 0) -> None:
        if not issubclass(error_type, FrameworkError):
            raise ValueError("error_type must be a subclass of FrameworkError")

        self._max_retries = max_retries
        self.error_type = error_type
        self.remaining = max_retries

        self._error_class: type[BaseException] = error_type  # TODO: FrameworkError
        self._lastError: BaseException | None = None
        self._finalError: BaseException | None = None

    def use(self, error: BaseException) -> None:
        if self._finalError:
            raise self._finalError

        self._lastError = error or self._lastError
        self.remaining -= 1

        # TODO: ifFatal, isRetryable etc
        if self.remaining < 0:
            self._finalError = self._error_class(  # type: ignore
                f"Maximal amount of global retries ({self._max_retries}) has been reached.", cause=self._lastError
            )
            raise self._finalError

    async def clone(self) -> "RetryCounter":
        cloned = RetryCounter(self.error_type, self._max_retries)
        cloned.remaining = self.remaining
        cloned._lastError = (
            await self._lastError.clone() if isinstance(self._lastError, FrameworkError) else self._lastError
        )
        cloned._finalError = (
            await self._finalError.clone() if isinstance(self._finalError, FrameworkError) else self._finalError
        )
        cloned._error_class = self._error_class
        return cloned


T = TypeVar("T")


class OccurrencesCounterEntry(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    value: T
    distance: int
    occurrences: int


class OccurrencesCounter(Generic[T]):
    def __init__(
        self,
        default: T | None = None,
        *,
        comparator: Callable[[T | None, T | None], bool] | None = None,
        n: int = 1,
    ) -> None:
        self.visits = 0
        self._default = default
        self._n = n
        self._comparator = comparator if comparator else lambda x, y: x == y
        self._entries: list[OccurrencesCounterEntry[T]] = []

    @property
    def leader(self) -> OccurrencesCounterEntry[T]:
        entry = max(self._entries, key=lambda e: e.occurrences)
        if entry is None:
            raise ValueError(f"{self} has no entry")
        return entry

    @property
    def entries(self) -> list[OccurrencesCounterEntry[T]]:
        return list(self._entries)

    def update(self, value: T) -> int:
        self.visits += 1

        for entry in self.entries:
            entry.distance += 1
            if entry.distance > self._n:
                self._entries.remove(entry)

        for entry in self.entries:
            if self._comparator(entry.value, value):
                entry.occurrences += 1
                return entry.occurrences

        entry = OccurrencesCounterEntry(value=value, distance=0, occurrences=1)
        self._entries.append(entry)
        return entry.occurrences

    def reset(self, value: T | None = None) -> None:
        self.visits = 0
        self._entries = []
        self._default = value
