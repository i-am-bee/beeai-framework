import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const template = new PromptTemplate({
  template: `Expected Duration: {{expected}}ms; Retrieved: {{#responses}}{{duration}}ms {{/responses}}`,
  schema: z.object({
    expected: z.number(),
    responses: z.array(z.object({ duration: z.number() })),
  }),
  defaults: {
    expected: 5,
  },
});

const output = template.render({
  expected: undefined, // default value will be used
  responses: [{ duration: 3 }, { duration: 5 }, { duration: 6 }],
});
console.log(output); // Expected Duration: 5ms; Retrieved: 3ms 5ms 6ms
