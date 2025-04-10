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

import { setTimeout as pSetTimeout } from "node:timers/promises";
import { Retryable, RunStrategy } from "@/internals/helpers/retryable.js";
import { expect } from "vitest";

describe("Retryable", () => {
  describe("Execution", () => {
    it("Executes only once", async () => {
      const executor = vi.fn().mockReturnValue(1);

      const instance = new Retryable({
        executor,
        config: {
          maxRetries: 100,
        },
      });

      expect(executor).not.toBeCalled();
      const results = await Promise.all([
        instance.get(),
        instance.get(),
        instance.get(),
        instance.get(),
      ]);
      expect(results.every((result) => result === 1)).toBe(true);
      await expect(instance.get()).resolves.toBe(1);
      expect(executor).toBeCalledTimes(1);
    });

    it("Re-Executes on error", async () => {
      const executor = vi.fn();
      const instance = new Retryable({
        executor,
        config: {
          maxRetries: 0,
        },
      });

      executor.mockImplementationOnce(() => {
        throw new Error("something went wrong");
      });
      await expect(instance.get()).rejects.toThrowError("something went wrong");
      await expect(instance.get()).rejects.toThrowError("something went wrong");

      instance.reset();

      executor.mockResolvedValueOnce(1);
      await expect(instance.get()).resolves.toBe(1);

      expect(executor).toBeCalledTimes(2);
    });

    it("Re-Executes on reset", async () => {
      const executor = vi.fn();
      const instance = new Retryable({
        executor,
        config: {
          maxRetries: 100,
        },
      });

      executor.mockResolvedValueOnce(1);
      await expect(instance.get()).resolves.toBe(1);
      instance.reset();
      executor.mockResolvedValueOnce(2);
      await expect(instance.get()).resolves.toBe(2);
      expect(executor).toBeCalledTimes(2);
    });
  });

  describe("Aborting", () => {
    it.each([-1, 0, 50, 100, 150])("Aborts after specific amount of time", async (delay) => {
      vi.useRealTimers();

      const startFn = vi.fn();
      const endFn = vi.fn();

      const controller = new AbortController();

      const instance = new Retryable({
        executor: async ({ attempt }) => {
          startFn();
          await pSetTimeout(200);
          if (attempt === 1) {
            throw new Error("Last error");
          }
          endFn();
        },
        config: {
          signal: controller.signal,
          maxRetries: 100,
        },
      });

      if (delay < 0) {
        controller.abort(new Error("Stop!"));
      }
      setTimeout(() => controller.abort(new Error("Stop!")), delay);
      await expect(instance.get()).rejects.toThrowError("Stop!");
      expect(endFn).not.toBeCalled();
      if (delay >= 0) {
        expect(startFn).toBeCalled();
      } else {
        expect(startFn).not.toBeCalled();
      }
    });
  });

  describe("Grouping", () => {
    class FatalError extends Error {}
    class BaseError extends Error {}

    const createContext = () => {
      const log: [number, "START" | "FATAL_ERROR" | "ERROR" | "RETRY" | "SUCCESS" | "END"][] =
        [] as const;
      const inputs = Array(5)
        .fill(null)
        .map(
          (_, idx) =>
            new Retryable({
              executor: async (meta) => {
                log.push([idx, "START"]);
                try {
                  if (idx === 0) {
                    await pSetTimeout(500);
                    throw new FatalError();
                  }
                  if (idx % 2 === 1 && meta.attempt % 2 === 1) {
                    await pSetTimeout(100);
                    throw new BaseError("Dummy error");
                  }
                  await pSetTimeout(300 + idx * 100);
                  log.push([idx, "SUCCESS"]);
                  return idx;
                } finally {
                  log.push([idx, "END"]);
                }
              },
              onRetry: () => {
                log.push([idx, "RETRY"]);
              },
              onError: (e) => {
                if (e instanceof FatalError) {
                  log.push([idx, "FATAL_ERROR"]);
                  throw e;
                } else {
                  log.push([idx, "ERROR"]);
                }
              },
            }),
        );

      return { log, inputs };
    };

    it.each(Object.values(RunStrategy))("Strategy (%s)", async (strategy) => {
      const { log, inputs } = createContext();
      await expect(Retryable.runGroup(strategy, inputs)).rejects.toThrowError(BaseError);
      expect(log.length).toBeGreaterThan(0);
      expect(log).toMatchSnapshot();
    });
  });
});
