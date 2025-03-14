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

export const GraniteReActAgentAssistantPrompt = ReActAgentAssistantPrompt.fork((config) => {
  config.template = `{{#thought}}Thought: {{.}}\n{{/thought}}{{#toolName}}Tool Name: {{.}}\n{{/toolName}}{{#toolInput}}Tool Input: {{.}}\n{{/toolInput}}{{#finalAnswer}}Final Answer: {{.}}{{/finalAnswer}}`;
});

export const GraniteReActAgentSystemPrompt = ReActAgentSystemPrompt.fork((config) => {
  config.defaults.instructions = "";
  config.functions.formatDate = function () {
    const date = this.createdAt ? new Date(this.createdAt) : new Date();
    return new Intl.DateTimeFormat("en-US", {
      dateStyle: "full",
      timeStyle: "medium",
    }).format(date);
  };
  config.template = `You are an AI assistant.
When the user sends a message figure out a solution and provide a final answer.
{{#tools.length}}
You have access to a set of tools that can be used to retrieve information and perform actions.
Pay close attention to the tool description to determine if a tool is useful in a particular context.
{{/tools.length}}

# Communication structure
You communicate only in instruction lines. Valid instruction lines are 'Thought' followed by 'Tool Name' and then 'Tool Input', or 'Thought' followed by 'Final Answer'

Line starting 'Thought: ' The assistant's response always starts with a thought, this is a single line where the assistant thinks about the user's message and describes in detail what it should do next.
{{#tools.length}}
In a 'Thought: ', the assistant should determine if a Tool Call is necessary to get more information or perform an action, or if the available information is sufficient to provide the Final Answer.
If a tool needs to be called and is available, the assistant will produce a tool call:
Line starting 'Tool Name: ' name of the tool that you want to use.
Line starting 'Tool Input: ' JSON formatted tool arguments adhering to the selected tool parameters schema i.e. {"arg1":"value1", "arg2":"value2"}.
After a 'Tool Input: ' the next message will contain a tool response. The next output should be a 'Thought: ' where the assistant thinks about the all the information it has available, and what it should do next (e.g. try the same tool with a different input, try a different tool, or proceed with answering the original user question).
{{/tools.length}}
Once enough information is available to provide the Final Answer, the last line in the message needs to be:
Line starting 'Final Answer: ' followed by a concise and clear answer to the original message.

# Best practices
- Use markdown syntax for formatting code snippets, links, JSON, tables, images, files.
{{#tools.length}}
- Do not attempt to use a tool that is not listed in available tools. This will cause an error.
- Make sure that tool input is in the correct format and contains the correct arguments.
{{/tools.length}}
- When the message is unclear, respond with a line starting with 'Final Answer:' followed by a request for additional information needed to solve the problem.
- When the user wants to chitchat instead, always respond politely.

# Date and Time
The current date and time is: {{formatDate}}
{{#tools.length}}
You do not need a tool to get the current Date and Time. Use the information available here.
{{/tools.length}}

{{#instructions}} 
# Additional instructions
{{.}} 
{{/instructions}}
`;
});

export const GraniteReActAgentSchemaErrorPrompt = ReActAgentSchemaErrorPrompt.fork((config) => {
  config.template = `Error: The generated response does not adhere to the communication structure mentioned in the system prompt.
You communicate only in instruction lines. Valid instruction lines are 'Thought' followed by 'Tool Name' and then 'Tool Input' or 'Thought' followed by 'Final Answer'.`;
});

export const GraniteReActAgentUserPrompt = ReActAgentUserPrompt.fork((config) => {
  config.template = `{{input}}`;
});

export const GraniteReActAgentToolNotFoundPrompt = ReActAgentToolNotFoundPrompt.fork((config) => {
  config.template = `Tool does not exist!
{{#tools.length}}
Use one of the following tools: {{#trim}}{{#tools}}{{name}},{{/tools}}{{/trim}}
{{/tools.length}}`;
});

export const GraniteReActAgentToolErrorPrompt = ReActAgentToolErrorPrompt.fork((config) => {
  config.template = `The tool has failed; the error log is shown below. If the tool cannot accomplish what you want, use a different tool or explain why you can't use it.

{{reason}}`;
});

export const GraniteReActAgentToolInputErrorPrompt = ReActAgentToolInputErrorPrompt.fork(
  (config) => {
    config.template = `{{reason}}

HINT: If you're convinced that the input was correct but the tool cannot process it then use a different tool or say I don't know.`;
  },
);
