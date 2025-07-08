# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

from beeai_framework.cache.base import BaseCache

T = TypeVar("T")


class NullCache(BaseCache[T]):
    def __init__(self) -> None:
        super().__init__()
        self._enabled: bool = False

    async def size(self) -> int:
        return 0

    async def set(self, _key: str, _value: T) -> None:
        pass

    async def get(self, key: str) -> T | None:
        return None

    async def has(self, key: str) -> bool:
        return False

    async def delete(self, key: str) -> bool:
        return True

    async def clear(self) -> None:
        pass
