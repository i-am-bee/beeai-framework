/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import pinoPretty from "pino-pretty";
import picocolors from "picocolors";

const compose =
  (...fns) =>
  (value) =>
    fns.reduce((res, f) => f(res), value);

export default (opts) => {
  return pinoPretty({
    colorize: true,
    colorizeObjects: true,
    singleLine: true,
    hideObject: false,
    sync: true,
    levelFirst: true,
    ...opts,
    translateTime: "HH:MM:ss",
    customPrettifiers: {
      level: (() => {
        const levels = {
          TRACE: { letters: "TRC", icon: "🔎", formatter: picocolors.gray },
          DEBUG: { letters: "DBG", icon: "🪲", formatter: picocolors.yellow },
          INFO: { letters: "INF", icon: "ℹ️", formatter: picocolors.green },
          WARN: { letters: "WRN", icon: "⚠️", formatter: picocolors.yellow },
          ERROR: { letters: "ERR", icon: "🔥", formatter: picocolors.red },
          FATAL: {
            letters: "FTL",
            icon: "💣",
            formatter: compose(picocolors.black, picocolors.bgRed),
          },
        };
        const fallback = { letters: "???", icon: "🤷‍", formatter: picocolors.gray };

        return (logLevel) => {
          const target = levels[logLevel] || fallback;
          return `${target.formatter(target.letters)}  ${target.icon} `;
        };
      })(),
      time: (timestamp) => picocolors.dim(timestamp),
      caller: (caller, key, log, { colors }) => `${colors.greenBright(caller)}`,
    },
    messageFormat: (log, messageKey) => {
      return `${log[messageKey]}`;
    },
  });
};
