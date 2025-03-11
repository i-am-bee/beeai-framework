# Cache

> [!TIP]
>
> Location within the framework `beeai-framework/cache`.

Caching is a process used to temporarily store copies of data or computations in a cache (a storage location) to facilitate faster access upon future requests. The primary purpose of caching is to improve the efficiency and performance of systems by reducing the need to repeatedly fetch or compute the same data from a slower or more resource-intensive source.

## Usage

### Capabilities showcase

<!-- embedme examples/cache/unconstrainedCache.ts -->

```ts
import { UnconstrainedCache } from "beeai-framework/cache/unconstrainedCache";

const cache = new UnconstrainedCache();

// Save
await cache.set("a", 1);
await cache.set("b", 2);

// Read
const result = await cache.get("a");
console.log(result); // 1

// Meta
console.log(cache.enabled); // true
console.log(await cache.has("a")); // true
console.log(await cache.has("b")); // true
console.log(await cache.has("c")); // false
console.log(await cache.size()); // 2

// Delete
await cache.delete("a");
console.log(await cache.has("a")); // false

// Clear
await cache.clear();
console.log(await cache.size()); // 0
```

_Source: [examples/cache/unconstrainedCache.ts](/typescript/examples/cache/unconstrainedCache.ts)_

### Caching function output + intermediate steps

<!-- embedme examples/cache/unconstrainedCacheFunction.ts -->

```ts
import { UnconstrainedCache } from "beeai-framework/cache/unconstrainedCache";

const cache = new UnconstrainedCache<number>();

async function fibonacci(n: number): Promise<number> {
  const cacheKey = n.toString();
  const cached = await cache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const result = n < 1 ? 0 : n <= 2 ? 1 : (await fibonacci(n - 1)) + (await fibonacci(n - 2));
  await cache.set(cacheKey, result);
  return result;
}

console.info(await fibonacci(10)); // 55
console.info(await fibonacci(9)); // 34 (retrieved from cache)
console.info(`Cache size ${await cache.size()}`); // 10
```

_Source: [examples/cache/unconstrainedCacheFunction.ts](/typescript/examples/cache/unconstrainedCacheFunction.ts)_

### Usage with tools

<!-- embedme examples/cache/toolCache.ts -->

```ts
import { SlidingCache } from "beeai-framework/cache/slidingCache";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";

const ddg = new WikipediaTool({
  cache: new SlidingCache({
    size: 100, // max 100 entries
    ttl: 5 * 60 * 1000, // 5 minutes lifespan
  }),
});

const response = await ddg.run({
  query: "United States",
});
// upcoming requests with the EXACTLY same input will be retrieved from the cache
```

_Source: [examples/cache/toolCache.ts](/typescript/examples/cache/toolCache.ts)_

> [!IMPORTANT]
>
> Cache key is created by serializing function parameters (the order of keys in the object does not matter).

### Usage with LLMs

<!-- embedme examples/cache/llmCache.ts -->

```ts
import { SlidingCache } from "beeai-framework/cache/slidingCache";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { UserMessage } from "beeai-framework/backend/message";

const llm = new OllamaChatModel("llama3.1");
llm.config({
  cache: new SlidingCache({
    size: 50,
  }),
  parameters: {
    maxTokens: 25,
  },
});

console.info(await llm.cache.size()); // 0
const first = await llm.create({
  messages: [new UserMessage("Who was Alan Turing?")],
});
// upcoming requests with the EXACTLY same input will be retrieved from the cache
console.info(await llm.cache.size()); // 1
const second = await llm.create({
  messages: [new UserMessage("Who was Alan Turing?")],
});
console.info(first.getTextContent() === second.getTextContent()); // true
console.info(await llm.cache.size()); // 1
```

_Source: [examples/cache/llmCache.ts](/typescript/examples/cache/llmCache.ts)_

## Cache types

The framework provides multiple out-of-the-box cache implementations.

### UnconstrainedCache

Unlimited in size.

```ts
import { UnconstrainedCache } from "beeai-framework/cache/unconstrainedCache";
const cache = new UnconstrainedCache();

await cache.set("a", 1);
console.log(await cache.has("a")); // true
console.log(await cache.size()); // 1
```

### SlidingCache

Keeps last `k` entries in the memory. The oldest ones are deleted.

<!-- embedme examples/cache/slidingCache.ts -->

```ts
import { SlidingCache } from "beeai-framework/cache/slidingCache";

const cache = new SlidingCache<number>({
  size: 3, // (required) number of items that can be live in the cache at a single moment
  ttl: 1000, // (optional, default is Infinity) Time in milliseconds after the element is removed from a cache
});

await cache.set("a", 1);
await cache.set("b", 2);
await cache.set("c", 3);

await cache.set("d", 4); // overflow - cache internally removes the oldest entry (key "a")
console.log(await cache.has("a")); // false
console.log(await cache.size()); // 3
```

_Source: [examples/cache/slidingCache.ts](/typescript/examples/cache/slidingCache.ts)_

### FileCache

One may want to persist data to a file so that the data can be later loaded. In that case the `FileCache` is ideal candidate.
You have to provide a location where the cache is persisted.

<!-- embedme examples/cache/fileCache.ts -->

```ts
import { FileCache } from "beeai-framework/cache/fileCache";
import * as os from "node:os";

const cache = new FileCache({
  fullPath: `${os.tmpdir()}/bee_file_cache_${Date.now()}.json`,
});
console.log(`Saving cache to "${cache.source}"`);
await cache.set("abc", { firstName: "John", lastName: "Doe" });
```

_Source: [examples/cache/fileCache.ts](/typescript/examples/cache/fileCache.ts)_

> [!NOTE]
>
> Provided location (`fullPath`) doesn't have to exist. It gets automatically created when needed.

> [!NOTE]
>
> Every modification to the cache (adding, deleting, clearing) immediately updates the target file.

#### Using a custom provider

<!-- embedme examples/cache/fileCacheCustomProvider.ts -->

```ts
import { FileCache } from "beeai-framework/cache/fileCache";
import { UnconstrainedCache } from "beeai-framework/cache/unconstrainedCache";
import os from "node:os";

const memoryCache = new UnconstrainedCache<number>();
await memoryCache.set("a", 1);

const fileCache = await FileCache.fromProvider(memoryCache, {
  fullPath: `${os.tmpdir()}/bee_file_cache.json`,
});
console.log(`Saving cache to "${fileCache.source}"`);
console.log(await fileCache.get("a")); // 1
```

_Source: [examples/cache/fileCacheCustomProvider.ts](/typescript/examples/cache/fileCacheCustomProvider.ts)_

### NullCache

The special type of cache is `NullCache` which implements the `BaseCache` interface but does nothing.

The reason for implementing is to enable [Null object pattern](https://en.wikipedia.org/wiki/Null_object_pattern).

### @Cache (decorator cache)

<!-- embedme examples/cache/decoratorCache.ts -->

```ts
import { Cache } from "beeai-framework/cache/decoratorCache";

class Generator {
  @Cache()
  get(seed: number) {
    return (Math.random() * 1000) / Math.max(seed, 1);
  }
}

const generator = new Generator();
const a = generator.get(5);
const b = generator.get(5);
console.info(a === b); // true
console.info(a === generator.get(6)); // false
```

_Source: [examples/cache/decoratorCache.ts](/typescript/examples/cache/decoratorCache.ts)_

**Complex example**

<!-- embedme examples/cache/decoratorCacheComplex.ts -->

```ts
import { Cache, SingletonCacheKeyFn } from "beeai-framework/cache/decoratorCache";

class MyService {
  @Cache({
    cacheKey: SingletonCacheKeyFn,
    ttl: 3600,
    enumerable: true,
    enabled: true,
  })
  get id() {
    return Math.floor(Math.random() * 1000);
  }

  reset() {
    Cache.getInstance(this, "id").clear();
  }
}

const service = new MyService();
const a = service.id;
console.info(a === service.id); // true
service.reset();
console.info(a === service.id); // false
```

_Source: [examples/cache/decoratorCacheComplex.ts](/typescript/examples/cache/decoratorCacheComplex.ts)_

> [!NOTE]
>
> Default `cacheKey` function is `ObjectHashKeyFn`

> [!CAUTION]
>
> Calling an annotated method with the `@Cache` decorator with different parameters (despite the fact you are not using them) yields in cache bypass (different arguments = different cache key) generated.
> Be aware of that. If you want your method always to return the same response, use `SingletonCacheKeyFn`.

### CacheFn

Because previously mentioned `CacheDecorator` can be applied only to class methods/getter the framework
provides a way how to do caching on a function level.

<!-- embedme examples/cache/cacheFn.ts -->

```ts
import { CacheFn } from "beeai-framework/cache/decoratorCache";
import { setTimeout } from "node:timers/promises";

const getSecret = CacheFn.create(
  async () => {
    // instead of mocking response you would do a real fetch request
    const response = await Promise.resolve({ secret: Math.random(), expiresIn: 100 });
    getSecret.updateTTL(response.expiresIn);
    return response.secret;
  },
  {}, // options object
);

const token = await getSecret();
console.info(token === (await getSecret())); // true
await setTimeout(150);
console.info(token === (await getSecret())); // false
```

_Source: [examples/cache/cacheFn.ts](/typescript/examples/cache/cacheFn.ts)_

> [!NOTE]
>
> Internally, the function is wrapped as a class; therefore, the same rules apply here as if it were a method annotated with the `@Cache` decorator.

## Creating a custom cache provider

To create your cache implementation, you must implement the `BaseCache` class.

<!-- embedme examples/cache/custom.ts -->

```ts
import { BaseCache } from "beeai-framework/cache/base";
import { NotImplementedError } from "beeai-framework/errors";

export class CustomCache<T> extends BaseCache<T> {
  size(): Promise<number> {
    throw new NotImplementedError();
  }

  set(key: string, value: T): Promise<void> {
    throw new NotImplementedError();
  }

  get(key: string): Promise<T | undefined> {
    throw new NotImplementedError();
  }

  has(key: string): Promise<boolean> {
    throw new NotImplementedError();
  }

  delete(key: string): Promise<boolean> {
    throw new NotImplementedError();
  }

  clear(): Promise<void> {
    throw new NotImplementedError();
  }

  createSnapshot() {
    throw new NotImplementedError();
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    throw new NotImplementedError();
  }
}
```

_Source: [examples/cache/custom.ts](/typescript/examples/cache/custom.ts)_

The simplest implementation is `UnconstrainedCache`, which can be found [here](/typescript/src/cache/unconstrainedCache.ts).
