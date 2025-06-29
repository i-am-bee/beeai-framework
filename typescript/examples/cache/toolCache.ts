/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

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
