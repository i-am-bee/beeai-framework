import asyncio
import sys
import tempfile
import traceback
from pathlib import Path
from typing import TypeVar

from beeai_framework.cache import UnconstrainedCache
from beeai_framework.errors import FrameworkError

from .file_cache import JsonFileCache  # noqa: TID252

T = TypeVar("T")


async def export_cache(provider: UnconstrainedCache[T]) -> dict[str, T]:
    """Clone an in-memory cache so that we can safely persist its content."""
    cloned = await provider.clone()
    # UnconstrainedCache stores entries in a simple dict, so cloning is inexpensive here.
    return getattr(cloned, "_provider", {}).copy()


async def main() -> None:
    memory_cache: UnconstrainedCache[int] = UnconstrainedCache()
    await memory_cache.set("tasks:open", 7)
    await memory_cache.set("tasks:closed", 12)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bee_cache.json"

        file_cache = await JsonFileCache.from_mapping(path, await export_cache(memory_cache), size=10, ttl=10)
        print(f"Promoted cache to disk: {file_cache.source}")

        print(await file_cache.get("tasks:open"))  # 7
        await file_cache.set("tasks:stale", 1)
        print(await file_cache.size())  # 3

        reloaded: JsonFileCache[int] = JsonFileCache(path, size=10, ttl=10)
        print(await reloaded.get("tasks:closed"))  # 12


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
