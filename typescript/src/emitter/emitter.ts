/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Callback,
  CleanupFn,
  EmitterOptions,
  EventMeta,
  EventTrace,
  Matcher,
  MatcherFn,
  StringKey,
} from "@/emitter/types.js";
export type { EventMeta, EventTrace, Callback };
import { Cache } from "@/cache/decoratorCache.js";
import { createFullPath, isPath, assertValidName, assertValidNamespace } from "@/emitter/utils.js";
import { EmitterError } from "@/emitter/errors.js";
import { createRandomHash } from "@/internals/helpers/hash.js";
import { shallowCopy, toBoundedFunction } from "@/serializer/utils.js";
import { RequiredAll } from "@/internals/types.js";
import { Serializable } from "@/internals/serializable.js";
import { pick } from "remeda";

export interface EmitterInput<E extends object = object> {
  groupId?: string;
  namespace?: string[];
  creator?: object;
  context?: E & object;
  trace?: EventTrace;
}
export interface EmitterChildInput<E extends object = object> {
  groupId?: string;
  namespace?: string[];
  creator?: object;
  context?: E & object;
  trace?: EventTrace;
}

interface Listener<T = any> {
  readonly match: MatcherFn;
  readonly raw: Matcher;
  readonly callback: Callback<T>;
  readonly options?: EmitterOptions;
}

export class Emitter<T = Record<keyof any, Callback<unknown>>> extends Serializable {
  protected listeners = new Set<Listener>();
  protected readonly groupId?: string;
  public readonly namespace: string[];
  public readonly creator?: object;
  public readonly context: object;
  public readonly trace?: EventTrace;
  protected readonly cleanups: CleanupFn[] = [];

  constructor(input: EmitterInput = {}) {
    super();
    this.groupId = input?.groupId;
    this.namespace = input?.namespace ?? [];
    this.creator = input.creator ?? Object.prototype;
    this.context = input?.context ?? {};
    this.trace = input.trace;
    assertValidNamespace(this.namespace);
  }

  static {
    this.register();
  }

  @Cache()
  static get root() {
    return new Emitter<Record<string, Callback<any>>>({ creator: Object.create(null) });
  }

  child<R = T>(input: EmitterChildInput = {}): Emitter<R> {
    const child = new Emitter<R>({
      trace: input.trace ?? this.trace,
      groupId: input?.groupId ?? this.groupId,
      context: { ...this.context, ...input?.context },
      creator: input?.creator ?? this.creator,
      namespace: input?.namespace
        ? [...this.namespace, ...input.namespace]
        : this.namespace.slice(),
    });

    const cleanup = child.pipe(this);
    this.cleanups.push(cleanup);

    return child;
  }

  pipe(target: Emitter<any>): CleanupFn {
    return this.on(
      // @ts-expect-error
      "*.*",
      toBoundedFunction(
        // @ts-expect-error
        (...args) => target.invoke(...args),
        [{ value: target, name: "target" }],
      ),
      {
        isBlocking: true,
        once: false,
        persistent: true,
      },
    );
  }

  destroy() {
    this.listeners.clear();
    this.cleanups.forEach((child) => child());
    this.cleanups.length = 0;
  }

  reset() {
    for (const listener of this.listeners) {
      if (!listener.options?.persistent) {
        this.listeners.delete(listener);
      }
    }
  }

  registerCallbacks<K extends StringKey<RequiredAll<T>>>(
    callbacks: Partial<
      Record<K, NonNullable<T[K]> extends Callback<infer X> ? Callback<X> : never>
    >,
    options?: Partial<Record<K, EmitterOptions>>,
  ): CleanupFn {
    const listeners: CleanupFn[] = [];
    Object.entries(callbacks).forEach(([key, value]) => {
      if (value) {
        // @ts-expect-error
        listeners.push(this.on(key, value, options?.[key]));
      }
    });
    return () => listeners.forEach((cleanup) => cleanup());
  }

  on<K extends StringKey<RequiredAll<T>>>(
    event: K,
    callback: NonNullable<T[K]> extends Callback<any> ? T[K] : never,
    options?: EmitterOptions,
  ): CleanupFn {
    return this.match(event as Matcher, callback as Callback<any>, options);
  }

  match<L>(matcher: Matcher, callback: Callback<L>, options?: EmitterOptions): CleanupFn {
    const createMatcher = (): MatcherFn => {
      const matchers: MatcherFn[] = [];
      let matchNested = options?.matchNested;

      if (matcher === "*") {
        matchNested ??= false;
        matchers.push((event) => event.path === createFullPath(this.namespace, event.name));
      } else if (matcher === "*.*") {
        matchNested ??= true;
        matchers.push(() => true);
      } else if (matcher instanceof RegExp) {
        matchNested ??= true;
        matchers.push((event) => matcher.test(event.path));
      } else if (typeof matcher === "function") {
        matchNested ??= false;
        matchers.push(matcher);
      } else if (typeof matcher === "string") {
        if (isPath(matcher)) {
          matchNested ??= true;
          matchers.push((event) => event.path === matcher);
        } else {
          matchNested ??= false;
          matchers.push(
            (event) =>
              event.name === matcher && event.path === createFullPath(this.namespace, event.name),
          );
        }
      } else {
        throw new EmitterError("Invalid matcher provided!");
      }

      if (!matchNested) {
        const matchSameRun = (event: EventMeta) => {
          // global observability x scoped observability
          return !this.trace?.runId ? true : this.trace?.runId === event.trace?.runId;
        };
        matchers.unshift(matchSameRun);
      }

      return (event) => matchers.every((matcher) => matcher(event));
    };

    const listener: Listener = {
      options,
      callback,
      raw: matcher,
      match: createMatcher(),
    };
    this.listeners.add(listener);

    return () => this.listeners.delete(listener);
  }

  async emit<K extends StringKey<RequiredAll<T>>>(
    name: K,
    value: NonNullable<T[K]> extends Callback<infer X> ? X : unknown,
  ): Promise<void> {
    assertValidName(name);

    const event = this.createEvent(name);
    return await this.invoke(value, event);
  }

  protected async invoke(data: unknown, event: EventMeta) {
    const executions: unknown[] = [];
    for (const listener of this.listeners.values()) {
      if (!listener.match(event)) {
        continue;
      }

      if (listener.options?.once) {
        this.listeners.delete(listener);
      }

      const run = async () => listener.callback(data, event);
      executions.push(listener.options?.isBlocking ? await run() : run());
    }
    await Promise.all(executions);
  }

  protected createEvent(name: string): EventMeta {
    return {
      id: createRandomHash(),
      groupId: this.groupId,
      name,
      path: createFullPath(this.namespace, name),
      createdAt: new Date(),
      source: this,
      creator: this.creator!,
      context: Object.assign({}, this.context, {}),
      trace: shallowCopy(this.trace), // TODO
    };
  }

  createSnapshot() {
    return {
      groupId: this.groupId,
      namespace: shallowCopy(this.namespace),
      creator: this.creator,
      context: this.context,
      trace: this.trace,
      listeners: Array.from(this.listeners).map(pick(["raw", "options", "callback"])),
      cleanups: this.cleanups,
    };
  }

  loadSnapshot({ listeners, ...snapshot }: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot, { listeners: new Set() });
    listeners.forEach(({ raw, callback, options }) => this.match(raw, callback, options));
  }
}
