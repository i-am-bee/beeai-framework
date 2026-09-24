/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { BaseCache } from "@/cache/base.js";
import fs from "node:fs";
import { SlidingCache } from "@/cache/slidingCache.js";
import { Cache } from "@/cache/decoratorCache.js";
import { Serializer } from "@/serializer/serializer.js";
import { SerializableClass } from "@/internals/serializable.js";

interface Input {
  fullPath: string;
}

export class FileCache<T> extends BaseCache<T> {
  constructor(protected readonly input: Input) {
    super();
  }

  static {
    this.register();
  }

  get source() {
    return this.input.fullPath;
  }

  static async fromProvider<A>(provider: BaseCache<A>, input: Input) {
    await fs.promises.writeFile(input.fullPath, await provider.serialize());
    return new FileCache<A>(input);
  }

  @Cache()
  protected async getProvider(): Promise<BaseCache<T>> {
    const exists = await fs.promises
      .stat(this.input.fullPath)
      .then((r) => r.isFile())
      .catch(() => false);

    if (exists) {
      const serialized = await fs.promises.readFile(this.input.fullPath, "utf8");
      const { target, snapshot } = await Serializer.deserialize<any>(serialized);
      const Target = Serializer.getFactory(target).ref as SerializableClass<BaseCache<T>>;
      const instance = await Target.fromSnapshot(snapshot);
      if (!(instance instanceof BaseCache)) {
        throw new TypeError("Provided file does not serialize any instance of BaseCache class.");
      }
      return instance;
    } else {
      return new SlidingCache({
        size: Infinity,
        ttl: Infinity,
      });
    }
  }

  async reload() {
    // @ts-expect-error protected property
    Cache.getInstance(this, "getProvider").clear();
    void (await this.getProvider());
  }

  protected async save() {
    const provider = await this.getProvider();
    return await fs.promises.writeFile(this.input.fullPath, await provider.serialize());
  }

  async size() {
    const provider = await this.getProvider();
    return provider.size();
  }

  async set(key: string, value: T) {
    const provider = await this.getProvider();
    await provider.set(key, value);
    void provider.get(key).finally(() => {
      void this.save();
    });
  }

  async get(key: string) {
    const provider = await this.getProvider();
    return await provider.get(key);
  }

  async has(key: string) {
    const provider = await this.getProvider();
    return await provider.has(key);
  }

  async delete(key: string) {
    const provider = await this.getProvider();
    const result = await provider.delete(key);
    await this.save();
    return result;
  }

  async clear() {
    const provider = await this.getProvider();
    await provider.clear();
    await this.save();
  }

  async createSnapshot() {
    return {
      input: { fullPath: this.input.fullPath },
      provider: await this.getProvider(),
    };
  }

  loadSnapshot(snapshot: Awaited<ReturnType<typeof this.createSnapshot>>): void {
    Object.assign(this, snapshot);
  }
}
