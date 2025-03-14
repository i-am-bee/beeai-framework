# Error Handling

> [!TIP]
>
> Location within the framework `beeai-framework/error`.

Error handling is a critical part of any JavaScript application, especially when dealing with asynchronous operations, various error types, and error propagation across multiple layers. In the BeeAI Framework, we provide a robust and consistent error-handling structure that ensures reliability and ease of debugging.

## The `FrameworkError` class

All errors thrown within the BeeAI Framework extend from the base [FrameworkError](/typescript/src/errors.ts) class, which itself extends Node.js's native [AggregateError](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/AggregateError).

Benefits of using `FrameworkError`:

- **Multiple Error Handling**: Supports handling multiple errors simultaneously, which is particularly useful in asynchronous or concurrent operations.
- **Preserved Error Chains**: Retains the full history of errors, giving developers greater context for debugging.
- **Consistent Structure:** All errors across the framework share a uniform structure, simplifying error tracking and management.
- **Native Support:** Built on native Node.js functionality, avoiding the need for additional dependencies while leveraging familiar mechanisms.
- **Utility Functions:** Includes methods for formatting error stack traces and explanations, making them suitable for use with LLMs and other external tools.

This structure ensures that users can trace the complete error history while clearly identifying any errors originating from the BeeAI Framework.

<!-- embedme examples/errors/base.ts -->

```ts
import { FrameworkError } from "beeai-framework/errors";

const error = new FrameworkError(
  "Function 'getUser' has failed.",
  [
    new FrameworkError("Cannot retrieve data from the API.", [
      new Error("User with given ID does not exist!"),
    ]),
  ],
  {
    context: { input: { id: "123" } },
    isFatal: true,
    isRetryable: false,
  },
);

console.log("Message", error.message); // Main error message
console.log("Meta", { fatal: error.isFatal, retryable: error.isRetryable }); // Is the error fatal/retryable?
console.log("Context", error.context); // Context in which the error occurred
console.log(error.explain()); // Human-readable format without stack traces (ideal for LLMs)
console.log(error.dump()); // Full error dump, including sub-errors
console.log(error.getCause()); // Retrieve the initial cause of the error
```

_Source: [examples/errors/base.ts](/typescript/examples/errors/base.ts)_

> [!NOTE]
>
> Every error thrown from the framework is an instance of the `FrameworkError` class, ensuring consistency across the codebase.

> [!TIP]
>
> The `explain()` method is particularly useful for returning a simplified, human-readable error message to an LLM, as used by the ReAct Agent.

## Specialized Error Classes

The BeeAI Framework extends FrameworkError to create specialized error classes for different components. This ensures that each part of the framework has clear and well-defined error types, improving debugging and error handling.

> [!TIP]
>
> Casting an unknown error to a `FrameworkError` can be done by calling the `FrameworkError.ensure` static method ([example](/typescript/examples/errors/cast.ts)).

### Tools

When a tool encounters an error, it throws a `ToolError`, which extends `FrameworkError`. If input validation fails, a `ToolInputValidationError` (which extends `ToolError`) is thrown.

<!-- embedme examples/errors/tool.ts -->

```ts
import { DynamicTool, ToolError } from "beeai-framework/tools/base";
import { FrameworkError } from "beeai-framework/errors";
import { z } from "zod";

const tool = new DynamicTool({
  name: "dummy",
  description: "dummy",
  inputSchema: z.object({}),
  handler: async () => {
    throw new Error("Division has failed.");
  },
});

try {
  await tool.run({});
} catch (e) {
  const err = e as FrameworkError;
  console.log(e instanceof ToolError); // true
  console.log("===DUMP===");
  console.log(err.dump());

  console.log("===EXPLAIN===");
  console.log(err.explain());
}
```

_Source: [examples/errors/tool.ts](/typescript/examples/errors/tool.ts)_

> [!TIP]
>
> If you throw a `ToolError` intentionally in a custom tool, the framework will not apply any additional "wrapper" errors, preserving the original error context.

### Agents

Throw `AgentError` class which extends `FrameworkError` class.

### Prompt Templates

Throw `PromptTemplateError` class which extends `FrameworkError` class.

### Loggers

Throw `LoggerError` class which extends `FrameworkError` class.

### Serializers

Throw `SerializerError` class which extends `FrameworkError` class.
