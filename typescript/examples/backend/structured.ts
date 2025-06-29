/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { ChatModel, UserMessage } from "beeai-framework/backend/core";
import { z } from "zod";

const model = await ChatModel.fromName("ollama:llama3.1");
const response = await model.createStructure({
  schema: z.union([
    z.object({
      firstName: z.string().min(1),
      lastName: z.string().min(1),
      address: z.string(),
      age: z.number().int().min(1),
      hobby: z.string(),
    }),
    z.object({
      error: z.string(),
    }),
  ]),
  messages: [new UserMessage("Generate a profile of a citizen of Europe.")],
});
console.log(response.object);
