# Error Handling

Error handling is a critical part of any Python application, especially when dealing with asynchronous operations, various error types, and error propagation across multiple layers. In the BeeAI Framework, we provide a robust and consistent error-handling structure that ensures reliability and ease of debugging.

## The `FrameworkError` class

All errors thrown within the BeeAI Framework extend from the base FrameworkError class.

Benefits of using `FrameworkError`:

[//] # TODO: REVIEW -The first two are standard Python functionality - remove?

- **Multiple Error Handling**: Supports handling multiple errors simultaneously, which is particularly useful in asynchronous or concurrent operations.
- **Preserved Error Chains**: Retains the full history of errors, giving developers greater context for debugging.
- **Consistent Structure:** All errors across the framework share a uniform structure, simplifying error tracking and management.
- **Native Support:** Built on native Python Exceptions functionality, avoiding the need for additional dependencies while leveraging familiar mechanisms.
- **Utility Functions:** Includes methods for formatting error stack traces and explanations, making them suitable for use with LLMs and other external tools.

This structure ensures that users can trace the complete error history while clearly identifying any errors originating from the BeeAI Framework.

[//] # TODO: Example needs adding with embedding. Include an ensure()
<!--
```py
```

_Source: /examples/errors/base.py TODO
-->
## Specialized Error Classes

The BeeAI Framework extends `FrameworkError` to create specialized error classes for different components or scenarios. This ensures that each part of the framework has clear and well-defined error types, improving debugging and error handling.

Unless otherwise stated, the exceptions below are subclasses of `FrameworkError`.

> [!TIP]
>
> Casting an unknown error to a `FrameworkError` can be done by calling the `FrameworkError.ensure` static method as shown in example above

[//] # TODO: REVIEW Consider better formats - table? Plus more description on parms used & better description of when used

[//] # TODO: Implementation also includes UnimplementedError and ArgumentError which only occur in test case - remove?
[//] # TODO: Does this fit better somewhere else?

### Aborts

- `AbortError`: Raised when an operation has been aborted. 

### Tools

- `ToolError` : Raised when a problem is reported by a tool.
  - `ToolInputValidationError`, which extends ToolError, raised when input validation fails.

[//] # TODO: Add toolError example and verify behaviour
<!--
```py
```

_Source: /examples/errors/tool.py TODO

> [!TIP]
>
> If you throw a `ToolError` intentionally in a custom tool, the framework will not apply any additional "wrapper" errors, preserving the original error context.
-->

### Agents

- `AgentError` : Raised when problems occur in agents.

### Prompt Templates

- `PromptTemplateError` : Raised when problems occur processing prompt templates.

### Loggers

- `LoggerError` : Raised when errors occur during logging.

### Serializers

- `SerializerError` : Raised when problems occur serializing or deserializing objects.

### Workflow

- `WorkflowError` : Raised when a workflow encounters an error.

### Parser

[//] # TODO: REVIEW - Do we want to enumerate the reason codes or link?

- `ParserError` : Raised when a parser fails to parse the input data. Includes additional *Reason*.

### Memory

- `Resource Error` : Raised when an error occurs with processing agent memory.
  - `ResourceFatalError` : Raised for particularly severe errors that are likely to be fatal.

### Emitter

- `EmitterError` : Raised when a problem occurs in the emitter.

### Backend

- `BackendError` : Raised when a backend encounters an error.
  - `ChatModelError` : Raised when a chat model fails to process input data. Subclass of BackendError.
- `MessageError` : Raised when a message processing fails.
