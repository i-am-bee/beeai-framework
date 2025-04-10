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

import { setTimeout } from "node:timers/promises";
import { signalRace } from "@/internals/helpers/promise.js";
import { FrameworkError } from "@/errors.js";

interface Meta {
  attempt: number;
  remaining: number;
}

interface Options {
  signal?: AbortSignal;
  factor?: number;
  retries?: number;
  shouldRetry?: (e: Error, meta: Meta) => boolean | Promise<boolean>;
  onFailedAttempt?: (e: Error, meta: Meta) => void | Promise<void>;
}

export async function pRetry<T>(
  fn: (attempt: number) => Promise<T>,
  options?: Options,
): Promise<T> {
  const handler = async (attempt: number, remaining: number) => {
    try {
      const factor = options?.factor ?? 2;
      if (attempt > 1) {
        const ms = Math.round(Math.pow(factor, attempt - 1)) * 1000;
        await setTimeout(ms, undefined, {
          signal: options?.signal,
        });
      }

      return await fn(attempt);
    } catch (e) {
      const meta: Meta = {
        attempt,
        remaining,
      };

      if (FrameworkError.isAbortError(e)) {
        throw e.cause || e;
      }

      await options?.onFailedAttempt?.(e, meta);
      if (remaining <= 0) {
        throw e;
      }

      if ((await options?.shouldRetry?.(e, meta)) === false) {
        throw e;
      }
      return await handler(attempt + 1, remaining - 1);
    }
  };

  return await signalRace(() => handler(1, options?.retries ?? 0), options?.signal);
}
