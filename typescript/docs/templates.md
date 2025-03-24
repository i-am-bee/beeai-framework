# Templates (Prompt Templates)

> [!TIP]
>
> Location within the framework `beeai-framework/template`.

**Template** is a predefined structure or format used to create consistent documents or outputs. It often includes placeholders for specific information that can be filled in later.

**Prompt template**, on the other hand, is a specific type of template used in the context of language models or AI applications.
It consists of a structured prompt that guides the model in generating a response or output. The prompt often includes variables or placeholders for user input, which helps to elicit more relevant or targeted responses.

The Framework exposes such functionality via the [`PromptTemplate`](/typescript/src/template.ts) class, which is based on the well-known [`Mustache.js`](https://github.com/janl/mustache.js) template system, which is supported almost in every programming language.
In addition, the framework provides type safety and validation against appropriate [`code](https://zod.dev/) schema, as you can see in the following examples.

> [!TIP]
>
> The Prompt Template concept is used anywhere - especially in our agents.

## Usage

### Primitives

<!-- embedme examples/templates/primitives.ts -->

```ts
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
```

_Source: [examples/templates/primitives.ts](/typescript/examples/templates/primitives.ts)_

### Arrays

<!-- embedme examples/templates/arrays.ts -->

```ts
import { PromptTemplate } from "beeai-framework/template";
import { z } from "zod";

const template = new PromptTemplate({
  schema: z.object({
    colors: z.array(z.string()).min(1),
  }),
  template: `Colors: {{#trim}}{{#colors}}{{.}},{{/colors}}{{/trim}}`,
});

const output = template.render({
  colors: ["Green", "Yellow"],
});
console.log(output); // Colors: Green,Yellow
```

_Source: [examples/templates/arrays.ts](/typescript/examples/templates/arrays.ts)_

### Objects

<!-- embedme examples/templates/objects.ts -->

```ts
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
```

_Source: [examples/templates/objects.ts](/typescript/examples/templates/objects.ts)_

### Forking

<!-- embedme examples/templates/forking.ts -->

```ts
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
```

_Source: [examples/templates/forking.ts](/typescript/examples/templates/forking.ts)_

### Functions

<!-- embedme examples/templates/functions.ts -->

```ts
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
```

_Source: [examples/templates/functions.ts](/typescript/examples/templates/functions.ts)_

## Agents

The ReAct Agent internally uses multiple prompt templates, and because now you know how to work with them, you can alter the agent’s behavior.

The internal prompt templates can be modified [here](/typescript/examples/agents/react_advanced.ts).
