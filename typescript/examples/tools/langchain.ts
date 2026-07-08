import { tool as createTool } from "@langchain/core/tools";
import { z } from "zod";
import { LangChainTool } from "beeai-framework/adapters/langchain/tools";

// You can use an arbitrary LangChain tool (see https://js.langchain.com/docs/integrations/tools/)
const generateRandomNumber = createTool(
  ({ min, max }) => {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  },
  {
    name: "GenerateRandomNumber",
    description: "Generates a random number in the given interval.",
    schema: z.object({
      min: z.number().int().min(0),
      max: z.number().int().min(0),
    }),
  },
);

const tool = new LangChainTool({
  tool: generateRandomNumber,
});
const response = await tool.run(
  // input to the tool
  {
    min: 1,
    max: 10,
  },
  // (optional) LangChain's run options
  {
    timeout: 10 * 1000,
  },
);
console.info(response);
