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
