/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { tool as createTool } from "@langchain/core/tools";
import { z } from "zod";
import { LangChainTool } from "@/adapters/langchain/tools.js";
import { verifyDeserialization } from "@tests/e2e/utils.js";

describe("Langchain Tools", () => {
  const getLangChainTool = () => {
    return createTool(
      async ({ min, max }): Promise<number> => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
      },
      {
        name: "GenerateRandomNumber",
        description: "Generates a random number in the given interval.",
        schema: z.object({
          min: z.number().int().min(0),
          max: z.number().int().min(0),
        }),
        metadata: {},
        responseFormat: "content",
      },
    );
  };

  it("Serializes", async () => {
    const lcTool = getLangChainTool();
    const instance = new LangChainTool({
      tool: lcTool,
    });

    const serialized = await instance.serialize();
    const deserialized = await LangChainTool.fromSerialized(serialized);
    verifyDeserialization(instance, deserialized, undefined, [], ["tool.schema"]);
  });
});
