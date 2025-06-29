/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const template = new PromptTemplate({
  schema: z.object({
    colors: z.array(z.string()).min(1),
  }),
  template: `Colors: {{#trim}}{{#colors}}{{.}},{{/colors}}{{/trim}}`,
});

const output = template.render({
  colors: ["Green", "Yellow"],
});
console.log(output); // Colors: Green,Yellow
