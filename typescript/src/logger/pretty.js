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
