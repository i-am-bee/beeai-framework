import asyncio
import random
import sys
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Generic, ParamSpec, TypedDict, TypeVar

from beeai_framework.cache import BaseCache
from beeai_framework.errors import FrameworkError

P = ParamSpec("P")
R = TypeVar("R")


class TokenResponse(TypedDict):
    token: str
    expires_in: float


class CacheFn(Generic[P, R]):
    """Callable wrapper that memoizes async functions with adjustable TTL."""

    def __init__(self, fn: Callable[P, Awaitable[R]], *, default_ttl: float | None = None) -> None:
        self._fn = fn
        self._entries: dict[str, tuple[R, float | None]] = {}
        self._default_ttl = default_ttl
        self._pending_ttl: float | None = None

    @classmethod
    def create(cls, fn: Callable[P, Awaitable[R]], *, default_ttl: float | None = None) -> "CacheFn[P, R]":
        return cls(fn, default_ttl=default_ttl)

    def update_ttl(self, ttl: float | None) -> None:
        """Adjust TTL for the next value written to the cache."""
        self._pending_ttl = ttl

    def clear(self) -> None:
        self._entries.clear()

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key = BaseCache.generate_key({"args": args, "kwargs": kwargs})
        entry = self._entries.get(key)
        now = time.time()
        if entry:
            value, expires_at = entry
            if expires_at is None or expires_at > now:
                return value

        result = await self._fn(*args, **kwargs)
        ttl = self._pending_ttl if self._pending_ttl is not None else self._default_ttl
        self._pending_ttl = None
        expires_at = now + ttl if ttl is not None else None
        self._entries[key] = (result, expires_at)
        return result


async def main() -> None:
    async def fetch_api_token() -> str:
        response: TokenResponse = {"token": f"TOKEN-{random.randint(1000, 9999)}", "expires_in": 0.2}
        get_token.update_ttl(response["expires_in"])
        await asyncio.sleep(0.05)
        return response["token"]

    get_token = CacheFn.create(fetch_api_token, default_ttl=0.1)

    first = await get_token()
    second = await get_token()
    print(first == second)  # True -> cached value

    await asyncio.sleep(0.25)
    refreshed = await get_token()
    print(first == refreshed)  # False -> TTL elapsed, value refreshed


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
