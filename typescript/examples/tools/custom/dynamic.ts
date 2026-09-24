import { DynamicTool, StringToolOutput } from "beeai-framework/tools/base";
import { z } from "zod";

const tool = new DynamicTool({
  name: "GenerateRandomNumber",
  description: "Generates a random number in the given interval.",
  inputSchema: z.object({
    min: z.number().int().min(0),
    max: z.number().int(),
  }),
  async handler(input) {
    const min = Math.min(input.min, input.max);
    const max = Math.max(input.max, input.min);

    const number = Math.floor(Math.random() * (max - min + 1)) + min;
    return new StringToolOutput(number.toString());
  },
});
