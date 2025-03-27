import "dotenv/config";
import { AzureOpenAIChatModel } from "beeai-framework/adapters/azure-openai/backend/chat";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { z } from "zod";
import { ChatModelError } from "beeai-framework/backend/errors";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new AzureOpenAIChatModel(
  "gpt-4o-mini",
  // {},
  // {
  //   apiKey: "AZURE_OPENAI_API_KEY",
  //   baseURL: "AZURE_OPENAI_API_ENDPOINT",
  //   resourceName: "AZURE_OPENAI_API_RESOURCE",
  //   apiVersion: "AZURE_OPENAI_API_VERSION"
  // },
);

llm.config({
  parameters: {
    topK: 1,
    temperature: 0,
    topP: 1,
  },
});

async function azureOpenaiFromName() {
  const azureOpenaiLLM = await ChatModel.fromName("azure:gpt-4o-mini");
  const response = await azureOpenaiLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function azureOpenaiSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function azureOpenaiStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function azureOpenaiAbort() {
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

async function azureOpenaiStructure() {
  const response = await llm.createStructure({
    schema: z.object({
      answer: z.string({ description: "your final answer" }),
    }),
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
  });
  console.info(response.object);
}

async function azureOpenaiToolCalling() {
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

async function azureOpenaiDebug() {
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

console.info(" azureOpenaiFromName".padStart(25, "*"));
await azureOpenaiFromName();
console.info(" azureOpenaiSync".padStart(25, "*"));
await azureOpenaiSync();
console.info(" azureOpenaiStream".padStart(25, "*"));
await azureOpenaiStream();
console.info(" azureOpenaiAbort".padStart(25, "*"));
await azureOpenaiAbort();
console.info(" azureOpenaiStructure".padStart(25, "*"));
await azureOpenaiStructure();
console.info(" azureOpenaiToolCalling".padStart(25, "*"));
await azureOpenaiToolCalling();
console.info(" azureOpenaiDebug".padStart(25, "*"));
await azureOpenaiDebug();
