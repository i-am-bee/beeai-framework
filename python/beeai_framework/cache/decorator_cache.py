# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Awaitable, Callable
from typing import Any, Generic, ParamSpec, TypeVar

from beeai_framework.cache.base import BaseCache

P = ParamSpec("P")
R = TypeVar("R")

CacheKeyFn = Callable[[tuple[Any, ...], dict[str, Any]], str]


def cached(
    cache: BaseCache[R],
    *,
    enabled: bool = True,
    key_fn: CacheKeyFn | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Async caching decorator built on top of BeeAI cache providers."""

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not enabled:
                return await fn(*args, **kwargs)

            key_builder = key_fn or (lambda a, kw: BaseCache.generate_key({"args": a, "kwargs": kw}))
            cache_key = key_builder(args, kwargs)

            cached_value = await cache.get(cache_key)
            if cached_value is not None or await cache.has(cache_key):
                return cached_value  # type: ignore[return-value]

            result = await fn(*args, **kwargs)
            await cache.set(cache_key, result)
            return result

        return wrapper

    return decorator


class CacheFn(Generic[P, R]):
    """Callable wrapper that memoizes async functions with adjustable TTL."""

    def __init__(
        self,
        fn: Callable[P, Awaitable[R]],
        *,
        default_ttl: float | None = None,
        key_fn: CacheKeyFn | None = None,
    ) -> None:
        self._fn = fn
        self._entries: dict[str, tuple[R, float | None]] = {}
        self._default_ttl = default_ttl
        self._pending_ttl: float | None = None
        self._key_fn = key_fn

    @classmethod
    def create(
        cls,
        fn: Callable[P, Awaitable[R]],
        *,
        default_ttl: float | None = None,
        key_fn: CacheKeyFn | None = None,
    ) -> "CacheFn[P, R]":
        return cls(fn, default_ttl=default_ttl, key_fn=key_fn)

    @property
    def default_ttl(self) -> float | None:
        return self._default_ttl

    def update_ttl(self, ttl: float | None) -> None:
        """Adjust TTL for the next cached value."""
        self._pending_ttl = ttl

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key_builder = self._key_fn or (lambda a, kw: BaseCache.generate_key({"args": a, "kwargs": kw}))
        cache_key = key_builder(args, kwargs)

        entry = self._entries.get(cache_key)
        now = time.time()
        if entry:
            value, expires_at = entry
            if expires_at is None or expires_at > now:
                return value

        result = await self._fn(*args, **kwargs)
        ttl = self._pending_ttl if self._pending_ttl is not None else self._default_ttl
        self._pending_ttl = None
        expires_at = now + ttl if ttl is not None else None
        self._entries[cache_key] = (result, expires_at)
        return result
