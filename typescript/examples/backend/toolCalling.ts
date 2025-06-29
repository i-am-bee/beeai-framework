/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import "dotenv/config";
import {
  ChatModel,
  Message,
  SystemMessage,
  ToolMessage,
  UserMessage,
} from "beeai-framework/backend/core";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { AnyTool, ToolOutput } from "beeai-framework/tools/base";

const model = await ChatModel.fromName("ollama:llama3.1");
const tools: AnyTool[] = [new DuckDuckGoSearchTool(), new OpenMeteoTool()];
const messages: Message[] = [
  new SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
  new UserMessage("What's the fastest marathon time?"),
];

while (true) {
  const response = await model.create({
    messages,
    tools,
  });
  messages.push(...response.messages);

  const toolCalls = response.getToolCalls();
  const toolResults = await Promise.all(
    toolCalls.map(async ({ args, toolName, toolCallId }) => {
      console.log(`-> running '${toolName}' tool with ${JSON.stringify(args)}`);
      const tool = tools.find((tool) => tool.name === toolName)!;
      const response: ToolOutput = await tool.run(args as any);
      const result = response.getTextContent();
      console.log(
        `<- got response from '${toolName}'`,
        result.replaceAll(/\s+/g, " ").substring(0, 90).concat(" (truncated)"),
      );
      return new ToolMessage({
        type: "tool-result",
        result,
        isError: false,
        toolName,
        toolCallId,
      });
    }),
  );
  messages.push(...toolResults);

  const answer = response.getTextContent();
  if (answer) {
    console.info(`Agent: ${answer}`);
    break;
  }
}
