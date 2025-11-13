import asyncio
import datetime as dt
import random
import sys
import traceback
from typing import Any

from beeai_framework.cache import BaseCache, SlidingCache, cached
from beeai_framework.errors import FrameworkError

activity_cache: SlidingCache[dict[str, Any]] = SlidingCache(size=16, ttl=5)


def session_cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    user_id = kwargs.get("user_id") or args[0]
    scope = kwargs.get("scope", "default")
    bucket: int | None = kwargs.get("minute_bucket")
    payload = {"user_id": user_id, "scope": scope}
    if bucket is not None:
        payload["minute_bucket"] = bucket
    return BaseCache.generate_key(payload)


class FeatureFlagService:
    def __init__(self, *, caching_enabled: bool = True) -> None:
        self._enabled = caching_enabled
        self._db_hits = 0

    @cached(activity_cache, enabled=True, key_fn=session_cache_key)
    async def load_flags(
        self, user_id: str, scope: str = "default", minute_bucket: int | None = None
    ) -> dict[str, Any]:
        self._db_hits += 1
        await asyncio.sleep(0.05)
        return {
            "user": user_id,
            "scope": scope,
            "db_hits": self._db_hits,
            "flags": {"beta_search": random.choice([True, False])},
            "refreshed_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }


async def main() -> None:
    service = FeatureFlagService()
    bucket = int(dt.datetime.now(dt.UTC).timestamp() // 60)

    first = await service.load_flags("42", scope="admin", minute_bucket=bucket)
    second = await service.load_flags("42", scope="admin", minute_bucket=bucket)
    print(first == second)  # True -> same cache key within a minute bucket

    await activity_cache.clear()  # Manual invalidation when new feature set deployed
    refreshed = await service.load_flags("42", scope="admin", minute_bucket=bucket)
    print(refreshed["db_hits"])  # 2 -> cache miss due to clear

    # Changing scope hits a different cache entry without flushing existing data.
    other_scope = await service.load_flags("42", scope="viewer", minute_bucket=bucket)
    print(other_scope["scope"])  # viewer


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
