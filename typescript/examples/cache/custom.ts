import { BaseCache } from "beeai-framework/cache/base";
import { NotImplementedError } from "beeai-framework/errors";

export class CustomCache<T> extends BaseCache<T> {
  size(): Promise<number> {
    throw new NotImplementedError();
  }

  set(key: string, value: T): Promise<void> {
    throw new NotImplementedError();
  }

  get(key: string): Promise<T | undefined> {
    throw new NotImplementedError();
  }

  has(key: string): Promise<boolean> {
    throw new NotImplementedError();
  }

  delete(key: string): Promise<boolean> {
    throw new NotImplementedError();
  }

  clear(): Promise<void> {
    throw new NotImplementedError();
  }

  createSnapshot() {
    throw new NotImplementedError();
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    throw new NotImplementedError();
  }
}
