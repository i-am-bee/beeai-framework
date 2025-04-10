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

import { TypeOf, ZodType, ZodEffects, ZodTypeAny, AnyZodObject, input } from "zod";
import { zodToJsonSchema, Options } from "zod-to-json-schema";
import { Ajv, SchemaObject, ValidateFunction, Options as AjvOptions } from "ajv";
import addFormats from "ajv-formats";
import { findFirstPair } from "@/internals/helpers/string.js";
import { FrameworkErrorOptions, ValueError } from "@/errors.js";
import { jsonrepair } from "jsonrepair";
export type AnyToolSchemaLike = AnyZodObject | SchemaObject;
export type AnySchemaLike = ZodTypeAny | SchemaObject;
export type FromSchemaLike<T> = T extends ZodTypeAny ? TypeOf<T> : unknown;
export type FromSchemaLikeRaw<T> = T extends ZodTypeAny ? input<T> : unknown;

export function validateSchema<T extends AnySchemaLike>(
  schema: T | ZodEffects<any>,
  errorOptions?: FrameworkErrorOptions,
): asserts schema is T {
  if (schema && schema instanceof ZodEffects) {
    throw new ValueError(
      "zod effects (refine, superRefine, transform, ...) cannot be converted to JSONSchema!",
      [],
      errorOptions,
    );
  }
}

export function toJsonSchema<T extends AnySchemaLike>(
  schema: T,
  options?: Partial<Options>,
): SchemaObject {
  validateSchema(schema);
  if (schema instanceof ZodType) {
    return zodToJsonSchema(schema, options);
  }
  return schema;
}

export function createSchemaValidator<T extends AnySchemaLike>(
  schema: T,
  options?: AjvOptions,
): ValidateFunction<FromSchemaLike<T>> {
  const jsonSchema = toJsonSchema(schema);

  const ajv = new Ajv({
    coerceTypes: "array",
    useDefaults: true,
    strict: false,
    strictSchema: false,
    strictTuples: true,
    strictNumbers: true,
    strictTypes: true,
    strictRequired: true,
    parseDate: true,
    allowDate: true,
    allowUnionTypes: true,
    ...options,
  });
  addFormats.default(ajv);
  return ajv.compile<FromSchemaLike<T>>(jsonSchema);
}

interface ParseBrokenJsonOptions {
  pair?: [string, string] | null;
}
export function parseBrokenJson(input: string | undefined, options?: ParseBrokenJsonOptions) {
  input = (input ?? "")?.trim();

  try {
    try {
      return JSON.parse(input);
    } catch {
      const pair = options?.pair;
      if (pair) {
        const { outer } = findFirstPair(input, pair) ?? { outer: input };
        return JSON.parse(jsonrepair(outer));
      } else {
        return JSON.parse(jsonrepair(input));
      }
    }
  } catch {
    return null;
  }
}
