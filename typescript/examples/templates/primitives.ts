import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const greetTemplate = new PromptTemplate({
  template: `Hello {{name}}`,
  schema: z.object({
    name: z.string(),
  }),
});

const output = greetTemplate.render({
  name: "Alex",
});
console.log(output); // Hello Alex!
