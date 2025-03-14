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

import {
  ReActAgentAssistantPrompt,
  ReActAgentSchemaErrorPrompt,
  ReActAgentSystemPrompt,
  ReActAgentToolErrorPrompt,
  ReActAgentToolInputErrorPrompt,
  ReActAgentToolNotFoundPrompt,
  ReActAgentUserPrompt,
} from "@/agents/react/prompts.js";

export const DeepThinkReActAgentAssistantPrompt = ReActAgentAssistantPrompt.fork((config) => {
  config.template = `{{#thought}}<think>{{.}}</think>\n\n{{/thought}}{{#toolName}}Tool Name: {{.}}\n{{/toolName}}{{#toolInput}}Tool Input: {{.}}\n{{/toolInput}}{{#finalAnswer}}Response: {{.}}{{/finalAnswer}}`;
});

export const DeepThinkReActAgentSystemPrompt = ReActAgentSystemPrompt.fork((config) => {
  config.defaults.instructions = "";
  config.functions.formatDate = function () {
    const date = this.createdAt ? new Date(this.createdAt) : new Date();
    return new Intl.DateTimeFormat("en-US", {
      dateStyle: "full",
      timeStyle: "medium",
    }).format(date);
  };
  config.template = `You are an AI assistant.
{{#tools.length}}
# Tools
You have access to a set of tools that can be used to retrieve information and perform actions.

{{#tools}}
Tool Name: {{name}}
Tool Description: {{description}}
Tool Input Schema: {{schema}}

{{/tools}}

## Guidelines for using tools
- Do not directly mention the existence of tools. Act as if these are your inherent capabilities.
- Do not speculate about specific numbers or facts, obtain them using tools.
- The tool responses appear as if provided by the user, but in reality they are inserted externally and the user does not see them. Act as if the tool responses were obtained by your inherent capabilities.
- When you don't have enough information, ask the user for details. Make reasonable assumptions. Do not ask the user to directly provide tool input, ask like you would in a normal conversation.
- If you need to call multiple tools, simply think out what tools you need to call, and then call the first one and end your message. Don't worry, you will be given opportunity to call as many tools as you want.

# Assistant message structure
- After thinking, write a message that has to follow a rigid structure. Deviations will result in an error.
- If you don't need to use a tool, already have all the information and want to answer the user's query, or you want to ask the user for more information, start the message with 'Response:', like this:
Response: [response message sent to the user]
- If you want to call a tool, start your message with 'Tool Name:', followed by the tool name, and then on the next line 'Tool Input:', followed by the tool input JSON, like this:
Tool Name: [name of the tool, as listed above]
Tool Input: [JSON object, following the schema defined above]
- DO NOT make the line prefixes ('Response:', 'Tool Name:', etc.) **bold**, always keep them plain.
- As soon as you write these lines, you will be provided the tool result. You will then have the opportunity to call more tools, or to finalize by sending a 'Response:' to the user.
{{/tools.length}}

# Additional context
The current date and time is: {{formatDate}}

{{#instructions}} 
# Additional instructions
{{.}} 
{{/instructions}}
`;
});

export const DeepThinkReActAgentSchemaErrorPrompt = ReActAgentSchemaErrorPrompt.fork((config) => {
  config.template = `Error: The generated response does not adhere to the communication structure mentioned in the system prompt.
You communicate only in instruction lines. Valid instruction lines are 'Tool Name:' and then 'Tool Input:', or 'Response:'.`;
});

export const DeepThinkReActAgentUserPrompt = ReActAgentUserPrompt.fork((config) => {
  config.template = `{{input}}`;
});

export const DeepThinkReActAgentToolNotFoundPrompt = ReActAgentToolNotFoundPrompt.fork((config) => {
  config.template = `Tool does not exist!
{{#tools.length}}
Use one of the following tools: {{#trim}}{{#tools}}{{name}},{{/tools}}{{/trim}}
{{/tools.length}}`;
});

export const DeepThinkReActAgentToolErrorPrompt = ReActAgentToolErrorPrompt.fork((config) => {
  config.template = `The tool has failed; the error log is shown below. If the tool cannot accomplish what you want, use a different tool or explain why you can't use it.

{{reason}}`;
});

export const DeepThinkReActAgentToolInputErrorPrompt = ReActAgentToolInputErrorPrompt.fork(
  (config) => {
    config.template = `{{reason}}

HINT: If you're convinced that the input was correct but the tool cannot process it then use a different tool or say I don't know.`;
  },
);
