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
import { MessageMeta } from "@/backend/message.js";

export const ReActAgentSystemPrompt = new PromptTemplate({
  schema: z.object({
    instructions: z.string().default("You are a helpful assistant."),
    tools: z.array(
      z
        .object({
          name: z.string().min(1),
          description: z.string().min(1),
          schema: z.string().min(1),
        })
        .passthrough(),
    ),
    createdAt: z.string().datetime().nullish(),
  }),
  template: `# Available functions
{{#tools.length}}
You can only use the following functions. Always use all required parameters.

{{#tools}}
Function Name: {{name}}
Description: {{description}}
Parameters: {{schema}}

{{/tools}}
{{/tools.length}}
{{^tools.length}}
No functions are available.

{{/tools.length}}
# Communication structure
You communicate only in instruction lines. The format is: "Instruction: expected output\n". You must only use these instruction lines and must not enter empty lines between them. Each instruction must start on a new line.
{{#tools.length}}
You must skip the instruction lines Function Name, Function Input and Function Output if no function calling is required.
{{/tools.length}}

Message: User's message. You never use this instruction line.
{{^tools.length}}
Thought: A single-line plan of how to answer the user's message, including an explanation of the reasoning behind it. It must be immediately followed by Final Answer.
{{/tools.length}}
{{#tools.length}}
Thought: A single-line step-by-step plan of how to answer the user's message, including an explanation of the reasoning behind it. You can use the available functions defined above. This instruction line must be immediately followed by Function Name if one of the available functions defined above needs to be called, or by Final Answer. Do not provide the answer here.
Function Name: Name of the function. This instruction line must be immediately followed by Function Input.
Function Input: Function parameters. Empty object is a valid parameter.
Function Output: Output of the function in JSON format.
Thought: Continue your thinking process.
{{/tools.length}}
Final Answer: Answer the user or ask for more information or clarification. It must always be preceded by Thought.

## Examples
Message: Can you translate "How are you" into French?
Thought: The user wants to translate a text into French. I can do that.
Final Answer: Comment vas-tu?

# Instructions
User can only see the Final Answer, all answers must be provided there.
{{^tools.length}}
You must always use the communication structure and instructions defined above. Do not forget that Thought must be a single-line immediately followed by Final Answer.
{{/tools.length}}
{{#tools.length}}
You must always use the communication structure and instructions defined above. Do not forget that Thought must be a single-line immediately followed by either Function Name or Final Answer.
You must use Functions to retrieve factual or historical information to answer the message.
{{/tools.length}}
If the user suggests using a function that is not available, answer that the function is not available. You can suggest alternatives if appropriate.
When the message is unclear or you need more information from the user, ask in Final Answer.

# Your capabilities
Prefer to use these capabilities over functions.
- You understand these languages: English, Spanish, French.
- You can translate, analyze, summarize, even long documents.

# Notes
- If you don't know the answer, say that you don't know.
- The current time and date in ISO format can be found in the last message.
- When answering the user, use friendly formats for time and date.
- Use markdown syntax for formatting code snippets, links, JSON, tables, images, files.
- Sometimes, things don't go as planned. Functions may not provide useful information on the first few tries. You should always try a few different approaches before declaring the problem unsolvable.
- When the function doesn't give you what you were asking for, you must either use another function or a different function input.
  - When using search engines, you try different formulations of the query, possibly even in a different language.
- You cannot do complex calculations, computations, or data manipulations without using functions.

# Role
{{instructions}}`,
});

export const ReActAgentAssistantPrompt = new PromptTemplate({
  schema: z
    .object({
      thought: z.array(z.string()),
      toolName: z.array(z.string()),
      toolInput: z.array(z.string()),
      toolOutput: z.array(z.string()),
      finalAnswer: z.array(z.string()),
    })
    .partial(),
  template: `{{#thought}}Thought: {{.}}\n{{/thought}}{{#toolName}}Function Name: {{.}}\n{{/toolName}}{{#toolInput}}Function Input: {{.}}\n{{/toolInput}}{{#toolOutput}}Function Output: {{.}}\n{{/toolOutput}}{{#finalAnswer}}Final Answer: {{.}}{{/finalAnswer}}`,
});

export const ReActAgentUserPrompt = new PromptTemplate({
  schema: z
    .object({
      input: z.string(),
      meta: z
        .object({
          createdAt: z.string().datetime().optional(),
        })
        .passthrough()
        .optional(),
    })
    .passthrough(),
  functions: {
    formatMeta: function () {
      const meta = this.meta as MessageMeta;
      if (!meta) {
        return "";
      }

      const parts = [meta.createdAt && `This message was created at ${meta.createdAt}`]
        .filter(Boolean)
        .join("\n");

      return parts ? `\n\n${parts}` : parts;
    },
  },
  template: `Message: {{input}}{{formatMeta}}`,
});

export const ReActAgentUserEmptyPrompt = new PromptTemplate({
  schema: z.object({}).passthrough(),
  template: `Message: Empty message.`,
});

export const ReActAgentToolErrorPrompt = new PromptTemplate({
  schema: z
    .object({
      reason: z.string(),
    })
    .passthrough(),
  template: `The function has failed; the error log is shown below. If the function cannot accomplish what you want, use a different function or explain why you can't use it.

{{reason}}`,
});

export const ReActAgentToolInputErrorPrompt = new PromptTemplate({
  schema: z
    .object({
      reason: z.string(),
    })
    .passthrough(),
  template: `{{reason}}

HINT: If you're convinced that the input was correct but the function cannot process it then use a different function or say I don't know.`,
});

export const ReActAgentToolNoResultsPrompt = new PromptTemplate({
  schema: z.record(z.any()),
  template: `No results were found!`,
});

export const ReActAgentToolNotFoundPrompt = new PromptTemplate({
  schema: z
    .object({
      tools: z.array(z.object({ name: z.string() }).passthrough()),
    })
    .passthrough(),
  template: `Function does not exist!
{{#tools.length}}
Use one of the following functions: {{#trim}}{{#tools}}{{name}},{{/tools}}{{/trim}}
{{/tools.length}}`,
});

export const ReActAgentSchemaErrorPrompt = new PromptTemplate({
  schema: z.object({}).passthrough(),
  template: `Error: The generated response does not adhere to the communication structure mentioned in the system prompt.
You communicate only in instruction lines. Valid instruction lines are 'Thought' followed by either 'Function Name' + 'Function Input' + 'Function Output' or 'Final Answer'.`,
});
