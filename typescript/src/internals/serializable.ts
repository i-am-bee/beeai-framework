/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Serializer } from "@/serializer/serializer.js";
import { ClassConstructor } from "@/internals/types.js";
import { extractClassName } from "@/serializer/utils.js";
import { SerializerError } from "@/serializer/error.js";
import { Cache } from "@/cache/decoratorCache.js";

export type SerializableClass<T> = ClassConstructor<Serializable<T>> &
  Pick<typeof Serializable<T>, "fromSnapshot" | "fromSerialized">;

interface SerializableStructure<T> {
  target: string;
  snapshot: T;
}

export interface DeserializeOptions {
  extraClasses?: SerializableClass<unknown>[];
}

export abstract class Serializable<T = unknown> {
  abstract createSnapshot(): T | Promise<T>;
  abstract loadSnapshot(snapshot: T): void | Promise<void>;

  constructor() {
    Object.getPrototypeOf(this).constructor.register();
    Cache.init(this);
  }

  public static register<T>(this: SerializableClass<T>, aliases?: string[]) {
    Serializer.registerSerializable(this, undefined, aliases);
  }

  async clone<T extends Serializable>(this: T): Promise<T> {
    const snapshot = await this.createSnapshot();

    const target = Object.create(this.constructor.prototype) as T;
    await target.loadSnapshot(snapshot);
    return target;
  }

  async serialize(): Promise<string> {
    const snapshot = await this.createSnapshot();
    return await Serializer.serialize<SerializableStructure<T>>({
      target: extractClassName(this),
      snapshot,
    });
  }

  protected async deserialize(value: string, options?: DeserializeOptions): Promise<T> {
    const { __root } = await Serializer.deserializeWithMeta<SerializableStructure<T>>(
      value,
      options?.extraClasses,
    );
    if (!__root.target) {
      // eslint-disable-next-line
      console.warn(
        `Serializable class must be serialized via "serialize" method and not via Serializer class. This may lead to incorrect deserialization.`,
      );
      return __root as T;
    }

    const current = extractClassName(this);
    if (current !== __root.target) {
      throw new SerializerError(
        `Snapshot has been created for class '${__root.target}' but you want to use it for class '${current}'.`,
      );
    }
    return __root.snapshot;
  }

  static async fromSnapshot<P, T extends Serializable<P>>(
    this: new (...args: any[]) => T,
    state: P,
  ): Promise<T> {
    const target = Object.create(this.prototype);
    await target.loadSnapshot(state);
    Cache.init(target);
    return target;
  }

  static async fromSerialized<T extends Serializable>(
    this: abstract new (...args: any[]) => T,
    serialized: string,
    options: DeserializeOptions = {},
  ): Promise<T> {
    const target = Object.create(this.prototype) as T;
    const state = await target.deserialize(serialized, options);
    await target.loadSnapshot(state);
    Cache.init(target);
    return target;
  }
}
