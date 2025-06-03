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
