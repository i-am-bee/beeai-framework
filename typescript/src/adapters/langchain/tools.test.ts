/**
 * Copyright 2025 IBM Corp.
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
