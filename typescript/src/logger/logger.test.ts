/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Logger } from "@/logger/logger.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";
import pinoTest from "pino-test";
import * as R from "remeda";

describe("Logger", () => {
  const createInstance = () => {
    const stream = pinoTest.sink();
    const logger = new Logger(
      {},
      Logger.createRaw(
        {
          level: "info",
          transport: undefined,
          timestamp: true,
          formatters: {
            bindings: R.identity(),
          },
        },
        stream,
      ),
    );

    return {
      stream,
      logger,
    };
  };

  it("Logs", async () => {
    const { logger, stream } = createInstance();

    logger.info("Hello world!");
    await pinoTest.once(stream, {
      level: "INFO",
      message: "Hello world!",
    });
  });

  it("Forks", async () => {
    const { logger: root, stream } = createInstance();

    root.info("Root");
    const child = root.child({
      name: "A",
    });
    child.info("A");

    const subchild = child.child({
      name: "B",
    });
    subchild.info("B");

    await pinoTest.consecutive(stream, [
      {
        level: "INFO",
        message: "Root",
      },
      {
        level: "INFO",
        message: "A",
        name: "A",
      },
      {
        level: "INFO",
        message: "B",
        name: "A.B",
      },
    ]);
  });

  it("Serializes", async () => {
    const instance = new Logger({
      name: "Root",
      bindings: {
        id: 123,
      },
    });
    instance.level = "fatal";

    const serialized = await instance.serialize();
    const deserialized = await Logger.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized);
  });
});
