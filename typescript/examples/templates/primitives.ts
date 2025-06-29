/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const greetTemplate = new PromptTemplate({
  template: `Hello {{name}}`,
  schema: z.object({
    name: z.string(),
  }),
});

const output = greetTemplate.render({
  name: "Alex",
});
console.log(output); // Hello Alex!
