/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { removeFromArray } from "@/internals/helpers/array.js";

type ServerFactory<TInput, TInternal> = (input: TInput) => Promise<TInternal>;
export type FactoryMember<TInput> = abstract new (
  ...args: any[]
) => TInput | (new (...args: any[]) => TInput) | ((...args: any[]) => TInput);

export abstract class Server<
  TInput extends object = object,
  TInternal extends object = object,
  TConfig extends object = object,
> {
  // @ts-expect-error
  public static readonly factories = new Map<object, ServerFactory<TInput, TInternal>>();

  public readonly members: TInput[] = [];

  constructor(protected config: TConfig) {}

  public static registerFactory<TInput2 extends object, TInternal2 extends object>(
    this: typeof Server<TInput2, TInternal2, any>,
    ref: FactoryMember<TInput2>,
    factory: ServerFactory<TInput2, TInternal2>,
    override = false,
  ): void {
    if (!this.factories.get(ref) || override) {
      this.factories.set(ref, factory);
    } else if (this.factories.get(ref) !== factory) {
      throw new Error(`Factory is already registered.`);
    }
  }

  public register(input: TInput): this {
    // check if the type has a factory registered
    this.getFactory(input);
    if (!this.members.includes(input)) {
      this.members.push(input);
    }
    return this;
  }

  public registerMany(input: TInput[]): this {
    input.forEach((item) => this.register(item));
    return this;
  }

  public deregister(input: TInput): this {
    removeFromArray(this.members, input);
    return this;
  }

  protected getFactory(input: TInput): ServerFactory<TInput, TInternal> {
    const factory = (this.constructor as typeof Server).factories.get(input);
    if (!factory) {
      throw new Error(`No factory registered for ${input.constructor.name}.`);
    }
    return factory;
  }

  public abstract serve(): void;
}
