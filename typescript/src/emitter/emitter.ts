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
import { removeFromArray } from "@/internals/helpers/array.js";

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

interface ParentPipe {
  parent: Emitter<any>;
  callback: Callback<any>;
  cleanup: CleanupFn;
  parentCleanup: CleanupFn;
}

export class Emitter<T = Record<keyof any, Callback<unknown>>> extends Serializable {
  protected listeners: Listener[] = [];
  protected readonly groupId?: string;
  public readonly namespace: string[];
  public readonly creator?: object;
  public readonly context: object;
  public readonly trace?: EventTrace;
  protected readonly cleanups: CleanupFn[] = [];
  private parentPipe?: ParentPipe;
  // Field initializers don't run for instances made with Object.create(Emitter.prototype)
  // (serializer placeholders, clone(), fromSnapshot), so this stays undefined for those.
  private initialized = true;

  private static readonly childRefs = new WeakMap<CleanupFn, WeakRef<Emitter<any>>>();
  private static readonly childRegistry = new FinalizationRegistry<{
    parent: WeakRef<Emitter<any>>;
    cleanup: WeakRef<CleanupFn>;
  }>(({ parent, cleanup }) => {
    const owner = parent.deref();
    const callback = cleanup.deref();
    if (owner && callback) {
      removeFromArray(owner.cleanups, callback);
      Emitter.childRefs.delete(callback);
    }
  });

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

    child.attachParent(this);

    return child;
  }

  pipe(target: Emitter<any>): CleanupFn {
    return this.createPipe(target).cleanup;
  }

  private createPipe(target: Emitter<any>) {
    const callback = toBoundedFunction(
      (data: unknown, event: EventMeta) => target.invoke(data, event),
      [{ value: target, name: "target" }],
    );
    const cleanup = this.on(
      // @ts-expect-error
      "*.*",
      callback,
      {
        isBlocking: true,
        once: false,
        persistent: true,
        matchNested: false,
      },
    );
    return { callback, cleanup };
  }

  private static createChildCleanup(childRef: WeakRef<Emitter<any>>): CleanupFn {
    // Keep this closure outside attachParent's scope: it must not capture the child
    // or the original pipe cleanup, including through a shared closure context.
    return toBoundedFunction(() => {
      const child = childRef.deref();
      if (child instanceof Emitter) {
        child.detachParent();
      }
    }, [{ value: childRef, name: "childRef" }, Emitter]);
  }

  private attachParent(parent: Emitter<any>) {
    const childRef = new WeakRef(this);
    const parentCleanup = Emitter.createChildCleanup(childRef);
    this.parentPipe = { parent, ...this.createPipe(parent), parentCleanup };
    // During deserialization, a child may refer to an uninitialized parent placeholder.
    if (!parent.cleanups) {
      Object.assign(parent, { cleanups: [] });
    }
    parent.cleanups.push(parentCleanup);
    Emitter.childRefs.set(parentCleanup, childRef);
    Emitter.childRegistry.register(
      this,
      {
        parent: new WeakRef(parent),
        cleanup: new WeakRef(parentCleanup),
      },
      parentCleanup,
    );
  }

  private detachParent() {
    const pipe = this.parentPipe;
    this.parentPipe = undefined;
    if (pipe) {
      pipe.cleanup();
      removeFromArray(pipe.parent.cleanups, pipe.parentCleanup);
      Emitter.childRefs.delete(pipe.parentCleanup);
      Emitter.childRegistry.unregister(pipe.parentCleanup);
    }
  }

  destroy() {
    this.detachParent();
    this.listeners.length = 0;
    this.cleanups.splice(0).forEach((cleanup) => cleanup());
  }

  reset() {
    this.listeners = this.listeners.filter((listener) => listener.options?.persistent);
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

    const priority = options?.priority ?? 0;
    const insertIndex = this.listeners.findIndex((l) => (l.options?.priority ?? 0) >= priority);
    if (insertIndex === -1) {
      this.listeners.push(listener);
    } else {
      this.listeners.splice(insertIndex, 0, listener);
    }

    return () => {
      removeFromArray(this.listeners, listener);
    };
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
    // Iterate in reverse to match Python's behavior (reversed list iteration)
    for (let i = this.listeners.length - 1; i >= 0; i--) {
      const listener = this.listeners[i];
      if (!listener.match(event)) {
        continue;
      }

      if (listener.options?.once) {
        this.listeners.splice(i, 1);
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
      listeners: this.listeners
        .filter((listener) => listener.callback !== this.parentPipe?.callback)
        .map(pick(["raw", "options", "callback"])),
      cleanups: this.cleanups.slice(),
      childCleanupIndices: this.cleanups.flatMap((cleanup, index) =>
        Emitter.childRefs.has(cleanup) ? [index] : [],
      ),
      parent: this.parentPipe?.parent,
    };
  }

  loadSnapshot(
    input: Omit<ReturnType<typeof this.createSnapshot>, "parent" | "childCleanupIndices"> & {
      parent?: Emitter<any>;
      childCleanupIndices?: number[];
    },
  ): void {
    // Cyclic graphs are finalized by loading a temporary instance into a placeholder.
    const temporary = input instanceof Emitter ? input : undefined;
    const {
      listeners,
      parent,
      childCleanupIndices = [],
      ...snapshot
    } = temporary ? temporary.createSnapshot() : input;
    const pending = !this.initialized ? (this.cleanups?.slice() ?? []) : [];
    if (this.initialized) {
      this.detachParent();
      this.cleanups.slice().forEach((cleanup) => {
        if (Emitter.childRefs.has(cleanup)) {
          cleanup();
        }
      });
    }
    const children =
      temporary?.cleanups
        .map((cleanup) => Emitter.childRefs.get(cleanup)?.deref())
        .filter((child): child is Emitter<any> => child !== undefined) ?? [];
    temporary?.detachParent();
    children.forEach((child) => child.detachParent());

    Object.assign(this, snapshot, {
      listeners: [],
      cleanups: snapshot.cleanups
        .filter(
          (cleanup, index) =>
            !childCleanupIndices.includes(index) && !Emitter.childRefs.has(cleanup),
        )
        .concat(pending),
    });
    this.parentPipe = undefined;
    this.initialized = true;
    listeners.forEach(({ raw, callback, options }) => this.match(raw, callback, options));
    if (parent) {
      this.attachParent(parent);
    }
    children.forEach((child) => child.attachParent(this));
  }
}
