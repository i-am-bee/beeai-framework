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

import { AsyncLocalStorage } from "node:async_hooks";
import { Emitter } from "@/emitter/emitter.js";
import { createRandomHash } from "@/internals/helpers/hash.js";
import { omit } from "remeda";
import { Callback } from "@/emitter/types.js";
import { registerSignals } from "@/internals/helpers/cancellation.js";
import { Serializable } from "@/internals/serializable.js";
import { executeSequentially, LazyPromise } from "@/internals/helpers/promise.js";
import { FrameworkError } from "@/errors.js";
import { shallowCopy } from "@/serializer/utils.js";
import { isAsyncIterable } from "@/internals/helpers/stream.js";

export interface RunInstance<T = any> {
  emitter: Emitter<T>;
}

export interface RunContextCallbacks {
  start: Callback<null>;
  success: Callback<unknown>;
  error: Callback<Error>;
  finish: Callback<null>;
}

export type GetRunContext<T, P = any> = T extends RunInstance ? RunContext<T, P> : never;
export type GetRunInstance<T> = T extends RunInstance<infer P> ? P : never;

export class Run<R, I extends RunInstance, P = any> extends LazyPromise<R> {
  protected readonly tasks: (() => Promise<void>)[] = [];

  constructor(
    handler: () => Promise<R>,
    protected readonly runContext: GetRunContext<I, P>,
  ) {
    super(handler);
  }

  readonly [Symbol.toStringTag] = "Promise";

  observe(fn: (emitter: Emitter<GetRunInstance<I>>) => void) {
    this.tasks.push(async () => fn(this.runContext.emitter));
    return this;
  }

  context(value: object) {
    this.tasks.push(async () => {
      Object.assign(this.runContext.context, value);
      Object.assign(this.runContext.emitter.context, value);
    });
    return this;
  }

  middleware(fn: (context: GetRunContext<I, P>) => void) {
    this.tasks.push(async () => fn(this.runContext));
    return this;
  }

  protected async before(): Promise<void> {
    await super.before();
    await executeSequentially(this.tasks.splice(0, Infinity));
  }

  // @ts-expect-error too complex
  async *[Symbol.asyncIterator](): R extends AsyncIterable<infer X> ? AsyncIterator<X> : never {
    const response = await this.then();
    if (isAsyncIterable(response)) {
      yield* response;
    } else {
      throw new Error("Result is not iterable!");
    }
  }
}

export interface RunContextInput<P> {
  params: P;
  signal?: AbortSignal;
}

export class RunContext<T extends RunInstance, P = any> extends Serializable {
  static #storage = new AsyncLocalStorage<RunContext<any>>();

  protected readonly controller: AbortController;
  public readonly runId: string;
  public readonly groupId: string;
  public readonly parentId?: string;
  public readonly emitter;
  public readonly context: object;
  public readonly runParams: P;
  public readonly createdAt: Date;

  get signal() {
    return this.controller.signal;
  }

  abort(reason?: Error) {
    this.controller.abort(reason);
  }

  constructor(
    public readonly instance: T,
    protected readonly input: RunContextInput<P>,
    parent?: RunContext<any>,
  ) {
    super();
    this.createdAt = new Date();
    this.runParams = input.params;
    this.runId = createRandomHash(5);
    this.parentId = parent?.runId;
    this.groupId = parent?.groupId ?? createRandomHash();
    this.context = omit((parent?.context ?? {}) as any, ["id", "parentId"]);

    this.controller = new AbortController();
    registerSignals(this.controller, [input.signal, parent?.signal]);

    this.emitter = instance.emitter.child<GetRunInstance<T>>({
      context: this.context,
      trace: {
        id: this.groupId,
        runId: this.runId,
        parentRunId: parent?.runId,
      },
    });
    if (parent) {
      this.emitter.pipe(parent.emitter);
    }
  }

  destroy() {
    this.emitter.destroy();
    this.controller.abort(new FrameworkError("Context destroyed."));
  }

  static enter<C2 extends RunInstance, R2, P2>(
    instance: C2,
    input: RunContextInput<P2>,
    fn: (context: GetRunContext<C2, P2>) => Promise<R2>,
  ) {
    const parent = RunContext.#storage.getStore();
    const runContext = new RunContext(instance, input, parent) as GetRunContext<C2, P2>;

    return new Run(async () => {
      const emitter = runContext.emitter.child<RunContextCallbacks>({
        namespace: ["run"],
        creator: runContext,
        context: { internal: true },
      });

      try {
        await emitter.emit("start", null);
        const result = await Promise.race([
          RunContext.#storage.run(runContext, fn, runContext),
          new Promise<never>((_, reject) =>
            runContext.signal.addEventListener("abort", () =>
              setTimeout(() => reject(runContext.signal.reason), 0),
            ),
          ),
        ]);
        await emitter.emit("success", result);
        return result;
      } catch (_e) {
        const e = FrameworkError.ensure(_e);
        await emitter.emit("error", e);
        throw e;
      } finally {
        await emitter.emit("finish", null);
        runContext.destroy();
      }
    }, runContext);
  }

  static {
    this.register();
  }

  createSnapshot() {
    return {
      controller: this.controller,
      runId: this.runId,
      groupId: this.groupId,
      parentId: this.parentId,
      emitter: this.emitter,
      context: shallowCopy(this.context),
      runParams: shallowCopy(this.runParams),
      createdAt: new Date(this.createdAt),
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}
