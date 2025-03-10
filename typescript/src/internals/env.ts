/**
 * Copyright 2025 IBM Corp.
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

import { z, ZodSchema } from "zod";
import { FrameworkError } from "@/errors.js";
import { getProp } from "@/internals/helpers/object.js";

export function getEnv(key: string, fallback?: never): string | undefined;
export function getEnv(key: string, fallback: string): string;
export function getEnv(key: string, fallback?: string) {
  return getProp(process.env, [key], fallback);
}

export function parseEnv<T extends ZodSchema>(
  key: string,
  schema: T,
  defaultValue?: string,
): z.output<T> {
  const value = getEnv(key) ?? defaultValue;
  const result = schema.safeParse(value);
  if (!result.success) {
    if (value === undefined) {
      throw new FrameworkError(`The required variable '${key}' is not set!`);
    }

    throw new FrameworkError(`Failed to parse the environment variable (${key})!`, [result.error]);
  }
  return result.data;
}
parseEnv.asBoolean = (key: string, fallback = false) => {
  return parseEnv(key, z.string(), String(fallback)).trim().toLowerCase() === "true";
};

export function hasEnv(key: string) {
  return getProp(process.env, [key]) !== undefined;
}
