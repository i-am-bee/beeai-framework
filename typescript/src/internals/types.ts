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

import type { Primitive } from "@/internals/helpers/guards.js";
type ExcludeArrayLike<T> = object & Exclude<T, any[]>;
export type ObjectLike<K extends keyof any = string, V = any> = Record<K, V>;
export type PlainObject<K extends string = string, V = Primitive> = Exclude<
  ExcludeArrayLike<Record<K & string, any extends V ? never : V>>,
  any[]
>;

export type ExcludeNonStringIndex<T> = {
  [K in keyof T as string extends K ? never : number extends K ? never : K]: T[K];
};

export type AnyFn = (...args: any[]) => any;

export type Enum<T> = Readonly<{
  [K in keyof T]: K;
}>;
export type EnumFromUnion<T extends string> = Readonly<{
  [K in T as Uppercase<K>]: K;
}>;
export type EnumValue<T> = T extends Enum<infer P> ? P[keyof P] : never;
export type EnumLowerCaseValue<T> = Readonly<{
  [K in keyof T]: K extends string ? Lowercase<K> : never;
}>[keyof T];
export type EmptyObject = Record<string, never>;

export type OmitType<T, O = never> = { [K in keyof T as T[K] extends O ? never : K]: T[K] };

export type Class<T = any> = new (...args: any[]) => T;

type TupleUnion<U extends string, R extends any[] = []> = {
  [S in U]: Exclude<U, S> extends never ? [...R, S] : TupleUnion<Exclude<U, S>, [...R, S]>;
}[U];
export type ArrayKeys<T> = TupleUnion<Extract<keyof T, string>>;
export type Common<A, B> = {
  [P in keyof A & keyof B]: A[P] | B[P];
};
export type ValueOf<T extends NonNullable<unknown>> = T[keyof T];

export type Head<T> = T extends [infer A, ...any] ? A : never;
export type Tail<T> = T extends [any, ...infer B] ? B : never;
export type RequiredExcept<T, K extends keyof T> = Omit<Required<T>, K> & Pick<Partial<T>, K>;
export type OptionalExcept<T, K extends keyof T> = Pick<Required<T>, K> & Omit<Partial<T>, K>;
export type NonUndefined<T, K extends keyof T = keyof T> = Pick<Required<T>, K> & Omit<T, K>;

export type RequiredAll<T> = {
  [P in keyof T]-?: NonNullable<T[P]>;
};

export type RequiredNested<T> = {
  [P in keyof T]-?: Required<T[P]>;
};

export type OmitEmpty<T> = OmitType<T, never | void>;
export type NonEmptyArray<T> = [T, ...T[]];
export type Unwrap<T> = T extends (infer X)[] ? X : T;
export type Constructable<T> = new (...args: any[]) => T;
export type AbstractConstructable = abstract new (...args: any[]) => any;
export type AnyConstructable = Constructable<any>;

export type Constructor<T extends new (...args: any) => any> = T extends new (
  ...args: infer A
) => infer R
  ? new (...args: A) => R
  : never;

export interface ClassConstructor<T = any> {
  name: string;
  new (...args: any[]): T;
}
export type AbstractClassConstructor<T = any> = abstract new (...args: any[]) => T;
export interface NamedFunction<T = any> {
  name: string;
  (...args: any[]): T;
}

export type PromiseOrPlain<T, Decider extends AnyFn> =
  ReturnType<Decider> extends Promise<unknown> ? Promise<T> : T;

export type NoPromise<T> = T extends Promise<unknown> ? never : T;

export type TypedFn<P extends any[], R> = (...args: P) => R;

export function narrowTo<T>(value: unknown, fn: boolean | ((value: T) => boolean)): value is T {
  if (typeof fn === "function") {
    return fn(value as T);
  }
  return fn;
}

type Without<T, U> = Partial<Record<Exclude<keyof T, keyof U>, never>>;
export type XOR<T, U> = T | U extends object ? (Without<T, U> & U) | (Without<U, T> & T) : T | U;
export type OneOf<T extends any[]> = T extends [infer Only]
  ? Only
  : T extends [infer A, infer B, ...infer Rest]
    ? OneOf<[XOR<A, B>, ...Rest]>
    : never;

export type AnyVoid = Promise<unknown> | unknown;

export type OmitPrivateKeys<T> = {
  [K in keyof T as K extends `_${string}` ? never : K]: T[K];
};

type MergeElements<A, B> = A extends never ? B : B extends never ? A : A | B;
export type MergeArrays<A extends any[], B extends any[]> = A extends []
  ? B
  : B extends []
    ? A
    : [MergeElements<Head<A>, Head<B>>, ...MergeArrays<Tail<A>, Tail<B>>];

export type MergeFunctions<A extends AnyFn, B extends AnyFn> = (
  ...args: MergeArrays<Parameters<A>, Parameters<B>>
) => MergeElements<ReturnType<B>, ReturnType<A>>;

export type WithoutLast<T> = T extends [...infer A, any] ? A : [];
