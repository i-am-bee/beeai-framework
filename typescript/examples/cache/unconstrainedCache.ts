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
