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
