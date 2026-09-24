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
