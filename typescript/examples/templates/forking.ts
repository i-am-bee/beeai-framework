import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const original = new PromptTemplate({
  template: `You are a helpful assistant called {{name}}. Your objective is to {{objective}}.`,
  schema: z.object({
    name: z.string(),
    objective: z.string(),
  }),
});

const modified = original.fork((config) => ({
  ...config,
  template: `${config.template} Your answers must be concise.`,
  defaults: {
    name: "Bee",
  },
}));

const output = modified.render({
  name: undefined, // default will be used
  objective: "fulfill the user needs",
});
console.log(output); // You are a helpful assistant called Bee. Your objective is to fulfill the user needs. Your answers must be concise.
