// NOTE: ensure you have installed following packages
// - @langchain/core
// - @langchain/cohere (or any other provider related package that you would like to use)
// List of available providers: https://js.langchain.com/v0.2/docs/integrations/chat/

import { LangChainChatModel } from "beeai-framework/adapters/langchain/backend/chat";
// @ts-expect-error package not installed
import { ChatCohere } from "@langchain/cohere";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { z } from "zod";
import { ChatModelError } from "beeai-framework/backend/errors";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new LangChainChatModel(
  new ChatCohere({
    model: "command-r-plus",
    temperature: 0,
  }),
);

async function langchainSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function langchainStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function langchainAbort() {
  try {
    const response = await llm.create({
      messages: [new UserMessage("What is the smallest of the Cape Verde islands?")],
      stream: true,
      abortSignal: AbortSignal.timeout(1 * 1000),
    });
    console.info(response.getTextContent());
  } catch (err) {
    if (err instanceof ChatModelError) {
      console.error("Aborted", { err });
    }
  }
}

async function langchainStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function langchainToolCalling() {
  const userMessage = new UserMessage("What is the weather in Boston?");
  const weatherTool = new OpenMeteoTool({ retryOptions: { maxRetries: 3 } });
  const response = await llm.create({ messages: [userMessage], tools: [weatherTool] });
  const toolCallMsg = response.getToolCalls()[0];
  console.debug(JSON.stringify(toolCallMsg));
  const toolResponse = await weatherTool.run(toolCallMsg.args as any);
  const toolResponseMsg = new ToolMessage({
    type: "tool-result",
    result: toolResponse.getTextContent(),
    toolName: toolCallMsg.toolName,
    toolCallId: toolCallMsg.toolCallId,
  });
  console.info(toolResponseMsg.toPlain());
  const finalResponse = await llm.create({ messages: [userMessage, toolResponseMsg], tools: [] });
  console.info(finalResponse.getTextContent());
}

async function langchainDebug() {
  // Log every request
  llm.emitter.match("*", (value, event) =>
    console.debug(
      `Time: ${event.createdAt.toISOString()}`,
      `Event: ${event.name}`,
      `Data: ${JSON.stringify(value)}`,
    ),
  );

  const response = await llm.create({
    messages: [new UserMessage("Hello world!")],
  });
  console.info(response.messages[0].toPlain());
}

console.info(" langchainSync".padStart(25, "*"));
await langchainSync();
console.info(" langchainStream".padStart(25, "*"));
await langchainStream();
console.info(" langchainAbort".padStart(25, "*"));
await langchainAbort();
console.info(" langchainStructure".padStart(25, "*"));
await langchainStructure();
console.info(" langchainToolCalling".padStart(25, "*"));
await langchainToolCalling();
console.info(" langchainDebug".padStart(25, "*"));
await langchainDebug();
