/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { PromptTemplate } from "@/template.js";
import {
  RequirementAgentSystemPromptInputSchema,
  RequirementAgentTaskPromptInputSchema,
  RequirementAgentToolErrorPromptInputSchema,
  RequirementAgentToolNoResultPromptInputSchema,
} from "@/agents/requirement/types.js";

export const RequirementAgentSystemPrompt = new PromptTemplate({
  schema: RequirementAgentSystemPromptInputSchema,
  template: `# Role
Assume the role of {{role}}.

# Instructions
{{#instructions}}
{{{.}}}
{{/instructions}}
When the user sends a message, figure out a solution and provide a final answer to the user by calling the '{{finalAnswerName}}' tool.
{{#finalAnswerSchema}}
The final answer must fulfill the following.

\`\`\`
{{{finalAnswerSchema}}}
\`\`\`
{{/finalAnswerSchema}}
{{#finalAnswerInstructions}}
{{{finalAnswerInstructions}}}
{{/finalAnswerInstructions}}

IMPORTANT: The facts mentioned in the final answer must be backed by evidence provided by relevant tool outputs.

# Tools
You must use a tool to retrieve factual or historical information.
Never use the tool twice with the same input if not stated otherwise.

{{#tools.0}}
{{#tools}}
Name: {{name}}
Description: {{description}}
Allowed: {{allowed}}{{#reason}}
Reason: {{{.}}}{{/reason}}

{{/tools}}
{{/tools.0}}

# Notes
- Use markdown syntax to format code snippets, links, JSON, tables, images, and files.
- If the provided task is unclear, ask the user for clarification.
- Do not refer to tools or tool outputs by name when responding.
- Always take it one step at a time. Don't try to do multiple things at once.
- When the tool doesn't give you what you were asking for, you must either use another tool or a different tool input.
- You should always try a few different approaches before declaring the problem unsolvable.
- If you can't fully answer the user's question, answer partially and describe what you couldn't achieve.
- You cannot do complex calculations, computations, or data manipulations without using tools.
- The current date and time is: {{formatDate}}
{{#notes}}
{{{.}}}
{{/notes}}`,
  defaults: {
    role: "a helpful AI assistant",
    instructions: "",
  },
  functions: {
    formatDate: () => new Date().toISOString().split("T")[0],
  },
});

export const RequirementAgentTaskPrompt = new PromptTemplate({
  schema: RequirementAgentTaskPromptInputSchema,
  template: `{{#context}}This is the context relevant to the task:
{{{.}}}

{{/context}}
{{#expectedOutput}}
This is the expected criteria for your output:
{{.}}

{{/expectedOutput}}
Your task: {{prompt}}`,
});

export const RequirementAgentToolErrorPrompt = new PromptTemplate({
  schema: RequirementAgentToolErrorPromptInputSchema,
  template: `The tool has failed; the error log is shown below. If the tool cannot accomplish what you want, use a different tool or explain why you can't use it.

{{{reason}}}`,
});

export const RequirementAgentToolNoResultPrompt = new PromptTemplate({
  schema: RequirementAgentToolNoResultPromptInputSchema,
  template: `No results were found! Try to reformulate your query or use a different tool.`,
});
