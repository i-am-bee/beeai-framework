import "dotenv/config";
import { OrcaRouterChatModel } from "beeai-framework/adapters/orcarouter/backend/chat";
import "dotenv/config.js";
import { ToolMessage, UserMessage } from "beeai-framework/backend/message";
import { ChatModel } from "beeai-framework/backend/chat";
import { AbortError } from "beeai-framework/errors";
import { z } from "zod";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new OrcaRouterChatModel(
  "orcarouter/auto",
  // {},
  // {
  //   apiKey: "ORCAROUTER_API_KEY",
  //   baseURL: "https://api.orcarouter.ai/v1",
  // },
);

llm.config({
  parameters: {
    temperature: 0.7,
    maxTokens: 1024,
    topP: 1,
  },
});

async function orcarouterFromName() {
  const orcarouterLLM = await ChatModel.fromName("orcarouter:orcarouter/auto");
  const response = await orcarouterLLM.create({
    messages: [new UserMessage("what states are part of New England?")],
  });
  console.info(response.getTextContent());
}

async function orcarouterSync() {
  const response = await llm.create({
    messages: [new UserMessage("what is the capital of Massachusetts?")],
  });
  console.info(response.getTextContent());
}

async function orcarouterStream() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    stream: true,
  });
  console.info(response.getTextContent());
}

async function orcarouterStreamAbort() {
  try {
    const response = await llm.create({
      messages: [new UserMessage("What is the smallest of the Cape Verde islands?")],
      stream: true,
      signal: AbortSignal.timeout(500),
    });
    console.info(response.getTextContent());
  } catch (error) {
    if (error instanceof AbortError) {
      console.info("Aborted.");
    }
  }
}

async function orcarouterStructure() {
  const response = await llm.create({
    messages: [new UserMessage("How many islands make up the country of Cape Verde?")],
    output: {
      type: "object",
      schema: z.object({
        answer: z.string(),
      }),
    },
  });
  console.info(response.getTextContent());
}

async function orcarouterWithTool() {
  const response = await llm.create({
    messages: [new UserMessage("What is the current weather in Salzburg?")],
    tools: [new OpenMeteoTool()],
  });
  console.info(response.getTextContent());
}

async function orcarouterToolCalls() {
  const response = await llm.create({
    messages: [new UserMessage("What is the current weather in Salzburg?")],
    tools: [new OpenMeteoTool()],
  });
  const toolResults = await Promise.all(
    response.getToolCalls().map(async (toolCall) => {
      const result = await toolCall.tool.run(toolCall.input);
      return new ToolMessage({
        toolCallId: toolCall.id,
        toolResult: result.getTextContent(),
      });
    }),
  );
  const response2 = await llm.create({
    messages: [
      new UserMessage("What is the current weather in Salzburg?"),
      ...response.getMessages(),
      ...toolResults,
    ],
    tools: [new OpenMeteoTool()],
  });
  console.info(response2.getTextContent());
}

async function main() {
  console.log("*".repeat(10), "orcarouterFromName");
  await orcarouterFromName();
  console.log("*".repeat(10), "orcarouterSync");
  await orcarouterSync();
  console.log("*".repeat(10), "orcarouterStream");
  await orcarouterStream();
  console.log("*".repeat(10), "orcarouterStreamAbort");
  await orcarouterStreamAbort();
  console.log("*".repeat(10), "orcarouterStructure");
  await orcarouterStructure();
  console.log("*".repeat(10), "orcarouterWithTool");
  await orcarouterWithTool();
  console.log("*".repeat(10), "orcarouterToolCalls");
  await orcarouterToolCalls();
}

main().catch(console.error);
