from typing import TypeVar

from beeai_framework.cache.base import BaseCache

T = TypeVar("T")


class CustomCache(BaseCache[T]):
    async def size(self) -> int:
        raise NotImplementedError("CustomCache 'size' not yet implemented")

    async def set(self, _key: str, _value: T) -> None:
        raise NotImplementedError("CustomCache 'set' not yet implemented")

    async def get(self, key: str) -> T | None:
        raise NotImplementedError("CustomCache 'get' not yet implemented")

    async def has(self, key: str) -> bool:
        raise NotImplementedError("CustomCache 'has' not yet implemented")

    async def delete(self, key: str) -> bool:
        raise NotImplementedError("CustomCache 'delete' not yet implemented")

    async def clear(self) -> None:
        raise NotImplementedError("CustomCache 'clear' not yet implemented")
