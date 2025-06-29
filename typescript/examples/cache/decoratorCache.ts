/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

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
