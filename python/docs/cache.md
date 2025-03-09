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

```text
Coming soon
```

_Source: examples/cache/unconstrainedCacheFunction.py_

### Using with tools

BeeAI framework's caching system seamlessly integrates with tools:

```text
Coming soon
```

_Source: examples/cache/toolCache.py_

### Using with LLMs

You can also cache LLM responses to save on API costs:

```text
Coming soon
```

_Source: examples/cache/llmCache.py_

---

## Cache types

### UnconstrainedCache

The simplest cache type with no constraints on size or entry lifetime. Good for development and smaller applications.

```text
Coming soon
```

_Source: examples/cache/unconstrainedCache.py_

### SlidingCache

Maintains a maximum number of entries, removing the oldest entries when the limit is reached.

```text
Coming soon
```

_Source: examples/cache/slidingCache.py_

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

```text
Coming soon
```

_Source: /examples/cache/custom.py_

---

## Examples

- All cache examples can be found in [here](/python/examples/cache).
