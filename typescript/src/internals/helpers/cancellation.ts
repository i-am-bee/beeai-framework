/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export function createAbortController(...signals: (AbortSignal | undefined)[]) {
  const controller = new AbortController();
  registerSignals(controller, signals);
  return controller;
}

export function registerSignals(controller: AbortController, signals: (AbortSignal | undefined)[]) {
  signals.forEach((signal) => {
    if (signal?.aborted) {
      controller.abort(signal.reason);
    }

    signal?.addEventListener?.(
      "abort",
      () => {
        controller.abort(signal?.reason);
      },
      {
        once: true,
        signal: controller.signal,
      },
    );
  });
}
