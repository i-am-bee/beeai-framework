/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
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
