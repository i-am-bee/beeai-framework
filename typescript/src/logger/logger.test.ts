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
