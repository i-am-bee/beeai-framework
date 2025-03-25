import "dotenv/config";
import { AnthropicChatModel } from "beeai-framework/adapters/anthropic/backend/chat";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { z } from "zod";
import { ChatModelError } from "beeai-framework/backend/errors";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new AnthropicChatModel(
  "claude-3-5-sonnet-latest",
  // {},
  // {
  //   baseURL: "ANTHROPIC_API_BASE_URL",
  //   apiKey: "ANTHROPIC_API_KEY",
  // },
);

llm.config({
  parameters: {
    topK: 1,
    temperature: 0,
    topP: 1,
  },
});

async function anthropicSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function anthropicStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function anthropicAbort() {
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

async function anthropicStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function anthropicToolCalling() {
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

async function anthropicDebug() {
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

console.info(" anthropicSync".padStart(25, "*"));
await anthropicSync();
console.info(" anthropicStream".padStart(25, "*"));
await anthropicStream();
console.info(" anthropicAbort".padStart(25, "*"));
await anthropicAbort();
console.info(" anthropicStructure".padStart(25, "*"));
await anthropicStructure();
console.info(" anthropicToolCalling".padStart(25, "*"));
await anthropicToolCalling();
console.info(" anthropicDebug".padStart(25, "*"));
await anthropicDebug();
