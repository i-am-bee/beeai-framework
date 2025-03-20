# 🗄️ Cache

> [!NOTE]  
> **COMING SOON! 🚀 Cache is not yet implemented in Python, but is available today in [TypeScript](/typescript/docs/cache.md)**

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
  - [Cache Types](#cache-types)
  - [Usage Patterns](#usage-patterns)
- [Basic Usage](#basic-usage)
  - [Caching Function Output](#caching-function-output)
  - [Using with Tools](#using-with-tools)
  - [Using with LLMs](#using-with-llms)
- [Cache Types](#cache-types)
  - [UnconstrainedCache](#unconstrainedcache)
  - [SlidingCache](#slidingcache)
  - [FileCache](#filecache)
  - [NullCache](#nullcache)
- [Advanced Usage](#advanced-usage)
  - [Cache Decorator](#cache-decorator)
  - [CacheFn Helper](#cachefn-helper)
- [Creating a Custom Cache Provider](#creating-a-custom-cache-provider)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

Caching is a technique used to temporarily store copies of data or computation results to improve performance by reducing the need to repeatedly fetch or compute the same data from slower or more resource-intensive sources.

In the context of AI applications, caching provides several important benefits:

- 🚀 **Performance improvement** - Avoid repeating expensive operations like API calls or complex calculations
- 💰 **Cost reduction** - Minimize repeated calls to paid services (like external APIs or LLM providers)
- ⚡ **Latency reduction** - Deliver faster responses to users by serving cached results
- 🔄 **Consistency** - Ensure consistent responses for identical inputs

BeeAI framework provides a robust caching system with multiple implementations to suit different use cases.

---

## Core concepts

### Cache types

BeeAI framework offers several cache implementations out of the box:

| Type               | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| **UnconstrainedCache** | Simple in-memory cache with no limits                            |
| **SlidingCache**       | In-memory cache that maintains a maximum number of entries       |
| **FileCache**          | Persistent cache that stores data on disk                        |
| **NullCache**          | Special implementation that performs no caching (useful for testing) |

Each cache type implements the `BaseCache` interface, making them interchangeable in your code.

### Usage patterns

BeeAI framework supports several caching patterns:

| Usage pattern          | Description                                      |
|------------------------|--------------------------------------------------|
| **Direct caching**      | Manually store and retrieve values              |
| **Function decoration** | Automatically cache function returns            |
| **Tool integration**    | Cache tool execution results                    |
| **LLM integration**     | Cache model responses                           |

---

## Basic usage

### Caching function output

The simplest way to use caching is to wrap a function that produces deterministic output:

<!-- embedme examples/cache/unconstrained_cache_function.py -->

```python
import asyncio
import sys
import traceback

from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: UnconstrainedCache[int] = UnconstrainedCache()

    async def fibonacci(n: int) -> int:
        cache_key = str(n)
        cached = await cache.get(cache_key)
        if cached:
            return int(cached)

        if n < 1:
            result = 0
        elif n <= 2:
            result = 1
        else:
            result = await fibonacci(n - 1) + await fibonacci(n - 2)

        await cache.set(cache_key, result)
        return result

    print(await fibonacci(10))  # 55
    print(await fibonacci(9))  # 34 (retrieved from cache)
    print(f"Cache size {await cache.size()}")  # 10


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/cache/unconstrained_cache_function.py](/python/examples/cache/unconstrained_cache_function.py)_

### Using with tools

BeeAI framework's caching system seamlessly integrates with tools:

<!-- embedme examples/cache/tool_cache.py -->

```python
import asyncio
import sys
import traceback

from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaTool({"full_text": True, "cache": SlidingCache(size=100, ttl=5 * 60)})

    print(await wikipedia_client.cache.size())  # 0
    tool_input = WikipediaToolInput(query="United States")
    first = await wikipedia_client.run(tool_input)
    print(await wikipedia_client.cache.size())  # 1

    # new request with the EXACTLY same input will be retrieved from the cache
    tool_input = WikipediaToolInput(query="United States")
    second = await wikipedia_client.run(tool_input)
    print(first.get_text_content() == second.get_text_content())  # True
    print(await wikipedia_client.cache.size())  # 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/cache/tool_cache.py](/python/examples/cache/tool_cache.py)_

### Using with LLMs

You can also cache LLM responses to save on API costs:

<!-- embedme examples/cache/llm_cache.py -->

```python
import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    llm.config(parameters=ChatModelParameters(max_tokens=25), cache=SlidingCache(size=50))

    print(await llm.cache.size())  # 0
    first = await llm.create(messages=[UserMessage("Who is Amilcar Cabral?")])
    print(await llm.cache.size())  # 1

    # new request with the EXACTLY same input will be retrieved from the cache
    second = await llm.create(messages=[UserMessage("Who is Amilcar Cabral?")])
    print(first.get_text_content() == second.get_text_content())  # True
    print(await llm.cache.size())  # 1


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/cache/llm_cache.py](/python/examples/cache/llm_cache.py)_

---

## Cache types

### UnconstrainedCache

The simplest cache type with no constraints on size or entry lifetime. Good for development and smaller applications.

<!-- embedme examples/cache/unconstrained_cache.py -->

```python
import asyncio
import sys
import traceback

from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: UnconstrainedCache[int] = UnconstrainedCache()

    # Save
    await cache.set("a", 1)
    await cache.set("b", 2)

    # Read
    result = await cache.has("a")
    print(result)  # True

    # Meta
    print(cache.enabled)  # True
    print(await cache.has("a"))  # True
    print(await cache.has("b"))  # True
    print(await cache.has("c"))  # False
    print(await cache.size())  # 2

    # Delete
    await cache.delete("a")
    print(await cache.has("a"))  # False

    # Clear
    await cache.clear()
    print(await cache.size())  # 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/cache/unconstrained_cache.py](/python/examples/cache/unconstrained_cache.py)_

### SlidingCache

Maintains a maximum number of entries, removing the oldest entries when the limit is reached.

<!-- embedme examples/cache/sliding_cache.py -->

```python
import asyncio
import sys
import traceback

from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.errors import FrameworkError


async def main() -> None:
    cache: SlidingCache[int] = SlidingCache(
        size=3,  # (required) number of items that can be live in the cache at a single moment
        ttl=1,  # // (optional, default is Infinity) Time in seconds after the element is removed from a cache
    )

    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.set("c", 3)

    await cache.set("d", 4)  # overflow - cache internally removes the oldest entry (key "a")

    print(await cache.has("a"))  # False
    print(await cache.size())  # 3


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/cache/sliding_cache.py](/python/examples/cache/sliding_cache.py)_

### FileCache

Persists cache data to disk, allowing data to survive if application restarts.

```text
Coming soon
```

_Source: examples/cache/fileCache.py_

#### With custom provider

You can customize how the FileCache stores data:

```text
Coming soon
```

_Source: examples/cache/fileCacheCustomProvider.py_

### NullCache

A special cache that implements the `BaseCache` interface but performs no caching. Useful for testing or temporarily disabling caching.

The reason for implementing is to enable [Null object pattern](https://en.wikipedia.org/wiki/Null_object_pattern).

---

## Advanced usage

### Cache decorator

The framework provides a convenient decorator for automatically caching function results:

```text
Coming soon
```

_Source: examples/cache/decoratorCache.py_

For more complex caching logic, you can customize the key generation:

```text
Coming soon
```

_Source: /examples/cache/decoratorCacheComplex.py_

### CacheFn helper

For more dynamic caching needs, the `CacheFn` helper provides a functional approach:

```text
Coming soon
```

_Source: /examples/cache/cacheFn.py_

---

## Creating a custom cache provider

You can create your own cache implementation by extending the `BaseCache` class:

<!-- embedme examples/cache/custom.py -->

```python
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

```

_Source: [examples/cache/custom.py](/python/examples/cache/custom.py)_


---

## Examples

- All cache examples are coming soon in python.
<!-- - All cache examples can be found in [here](/python/examples/cache). -->
