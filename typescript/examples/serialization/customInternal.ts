/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Serializable } from "beeai-framework/internals/serializable";

class MyClass extends Serializable {
  constructor(public readonly name: string) {
    super();
  }

  static {
    // register class to the global serializer register
    this.register();
  }

  createSnapshot(): unknown {
    return {
      name: this.name,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

const instance = new MyClass("Bee");
const serialized = await instance.serialize();
const deserialized = await MyClass.fromSerialized(serialized);

console.info(instance);
console.info(deserialized);
