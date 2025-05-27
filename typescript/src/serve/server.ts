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

export abstract class Server<
  TInput extends object = object,
  TInternal extends object = object,
  TConfig extends object = object,
> {
  private factories = new Map<TInput, (input: TInput) => TInternal>();

  protected _members: TInput[] = [];

  constructor(protected config: TConfig) {}

  public registerFactory(
    ref: TInput,
    factory: (input: TInput) => TInternal,
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
    if (!this._members.includes(input)) {
      this._members.push(input);
    }
    return this;
  }

  public registerMany(input: TInput[]): this {
    input.forEach((item) => this.register(item));
    return this;
  }

  public deregister(input: TInput): this {
    this._members = this._members.filter((member) => member !== input);
    return this;
  }

  protected getFactory(input: TInput): (input: TInput) => TInternal {
    const factory = this.factories.get(input);
    if (!factory) {
      throw new Error(`No factory registered for ${input.constructor.name}.`);
    }
    return factory;
  }

  public get members(): TInput[] {
    return this._members;
  }

  public abstract serve(): void;
}
