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
