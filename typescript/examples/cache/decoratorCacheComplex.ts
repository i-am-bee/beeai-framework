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
