import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const messageTemplate = new PromptTemplate({
  schema: z
    .object({
      text: z.string(),
      author: z.string().optional(),
      createdAt: z.string().datetime().optional(),
    })
    .passthrough(),
  functions: {
    formatMeta: function () {
      if (!this.author && !this.createdAt) {
        return "";
      }

      const author = this.author || "anonymous";
      const createdAt = this.createdAt || new Date().toISOString();

      return `\nThis message was created at ${createdAt} by ${author}.`;
    },
  },
  template: `Message: {{text}}{{formatMeta}}`,
});

// Message: Hello from 2024!
// This message was created at 2024-01-01T00:00:00.000Z by John.
console.log(
  messageTemplate.render({
    text: "Hello from 2024!",
    author: "John",
    createdAt: new Date("2024-01-01").toISOString(),
  }),
);

// Message: Hello from the present!
console.log(
  messageTemplate.render({
    text: "Hello from the present!",
  }),
);
