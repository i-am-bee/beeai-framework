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

import { pino as pinoLogger, ChildLoggerOptions, LoggerOptions, DestinationStream } from "pino";
import { FrameworkError } from "@/errors.js";
import { Serializable } from "@/internals/serializable.js";
import { Cache } from "@/cache/decoratorCache.js";
import { EnumFromUnion, ValueOf } from "@/internals/types.js";
import { parseEnv } from "@/internals/env.js";
import { z } from "zod";
import { isTruthy } from "remeda";
import { PrettyOptions } from "pino-pretty";
import { shallowCopy } from "@/serializer/utils.js";
import { stdout } from "node:process";
import * as url from "node:url";
import path from "node:path";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));

export interface LoggerBindings extends Record<string, any> {}

export class LoggerError extends FrameworkError {}

export const LoggerLevel: EnumFromUnion<pinoLogger.LevelWithSilent> = {
  DEBUG: "debug",
  ERROR: "error",
  FATAL: "fatal",
  INFO: "info",
  TRACE: "trace",
  WARN: "warn",
  SILENT: "silent",
} as const;
export type LoggerLevelType = ValueOf<typeof LoggerLevel>;

export interface LoggerInput {
  name?: string;
  bindings?: LoggerBindings;
  level?: LoggerLevelType;
  raw?: ChildLoggerOptions;
}

export class Logger extends Serializable implements pinoLogger.BaseLogger {
  protected raw!: pinoLogger.Logger;

  info!: pinoLogger.LogFn;
  warn!: pinoLogger.LogFn;
  fatal!: pinoLogger.LogFn;
  error!: pinoLogger.LogFn;
  debug!: pinoLogger.LogFn;
  trace!: pinoLogger.LogFn;
  silent!: pinoLogger.LogFn;

  static {
    this.register();
  }

  get level(): LoggerLevelType {
    return this.raw.level as LoggerLevelType;
  }

  set level(value: LoggerLevelType) {
    this.raw.level = value;
  }

  constructor(
    public readonly input: LoggerInput,
    raw?: pinoLogger.Logger,
  ) {
    super();
    this.raw = raw!;
    this.init();
  }

  static of(input: LoggerInput) {
    return new Logger(input);
  }

  private init() {
    const parent = this.raw || Logger.root.raw;
    const instance = parent.child(
      {
        ...this.input.bindings,
        name: this.input.name ?? this.input.bindings?.name,
      },
      {
        ...this.input.raw,
        level: this.input.level ?? this.input.raw?.level,
      },
    );

    this.raw = instance;
    this.info = instance.info.bind(instance);
    this.warn = instance.warn.bind(instance);
    this.fatal = instance.fatal.bind(instance);
    this.error = instance.error.bind(instance);
    this.debug = instance.debug.bind(instance);
    this.trace = instance.trace.bind(instance);
    this.silent = instance.silent.bind(instance);
  }

  @Cache()
  static get root() {
    return new Logger(Logger.defaults, Logger.createRaw(Logger.defaults));
  }

  @Cache()
  static get defaults(): Omit<LoggerInput, "raw"> & { pretty: boolean } {
    return {
      name: undefined,
      pretty: parseEnv.asBoolean("BEE_FRAMEWORK_LOG_PRETTY", false),
      bindings: {},
      level: parseEnv(
        "BEE_FRAMEWORK_LOG_LEVEL",
        z.nativeEnum(LoggerLevel).default(LoggerLevel.INFO),
      ),
    };
  }

  child(input?: LoggerInput) {
    const name = [this.input.name, input?.name].filter(isTruthy).join(".");

    return new Logger(
      {
        ...this.input,
        level: this.level,
        ...input,
        name,
        bindings: {
          name,
        },
      },
      this.raw,
    );
  }

  createSnapshot() {
    return {
      input: shallowCopy(this.input),
      level: this.raw.level,
    };
  }

  loadSnapshot({ level, ...extra }: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, extra);
    this.init();
    this.raw.level = level;
  }

  public static createRaw(options?: LoggerOptions, stream?: DestinationStream | undefined) {
    const defaults = Logger.defaults;
    const isPretty = defaults.pretty;

    const targetStream = stream ?? (isPretty ? pinoLogger.destination(stdout) : undefined);

    return pinoLogger(
      {
        ...(isPretty && {
          transport: {
            target: path.join(__dirname, "pretty.js"),
            options: {
              messageKey: "message",
              nestedKey: undefined,
              errorKey: "error",
              colorize: true,
              sync: true,
              singleLine: parseEnv.asBoolean("BEE_FRAMEWORK_LOG_SINGLE_LINE", false),
            } as PrettyOptions,
          },
        }),
        messageKey: "message",
        nestedKey: defaults.pretty ? undefined : "payload",
        errorKey: "error",
        timestamp: true,
        name: defaults.name,
        level: defaults.level,
        ...options,
        formatters: {
          bindings: ({ pid: _, hostname: __, ...others }) => {
            return others;
          },
          log: (record) => {
            return record;
          },
          level: (label) => {
            return { level: label.toUpperCase() };
          },
          ...options?.formatters,
        },
      },
      targetStream,
    );
  }
}
