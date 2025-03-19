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

import { PromptTemplate } from "@/template.js";
import { z } from "zod";

export const ToolCallingAgentSystemPrompt = new PromptTemplate({
  schema: z.object({
    role: z.string(),
    instructions: z.string().optional(),
  }),
  functions: {
    formatDate: function () {
      return new Date().toISOString();
    },
  },
  defaults: { role: "A helpful AI assistant", instructions: "" },
  template: `Assume the role of {{role}}.
{{#instructions}}

Your instructions are:
{{.}}
{{/instructions}}

When the user sends a message, figure out a solution and provide a final answer.
You can use tools to improve your answers if available.

# Best practices
- Use markdown syntax to format code snippets, links, JSON, tables, images, and files.
- If the provided task is unclear, ask the user for clarification.
- Do not refer to tools or tool outputs by name when responding.

# Date and Time
The current date and time is: {{formatDate}}
You do not need a tool to get the current Date and Time. Use the information available here.
`,
});

export const ToolCallingAgentTaskPrompt = new PromptTemplate({
  schema: z.object({
    prompt: z.string(),
    context: z.string().optional(),
    expectedOutput: z.string().optional(),
  }),
  template: `{{#context}}This is the context that you are working with:
{{.}}

{{/context}}
{{#expectedOutput}}
This is the expected criteria for your output:
{{.}}

{{/expectedOutput}}
Your task: {{prompt}}
`,
});
