# Error Handling

Error handling is a critical part of any Python application, especially when dealing with asynchronous operations, various error types, and error propagation across multiple layers. In the BeeAI Framework, we provide a robust and consistent error-handling structure that ensures reliability and ease of debugging.

## The `FrameworkError` class

Within the BeeAI Framework, regular Python Exceptions are used to handle common issues such as `ValueError`, `TypeError`. However, to provide a more comprehensive error handling experience, we have introduced `FrameworkError`, which is a subclass of Exception. Where additional context is needed, we can use `FrameworkError` to provide additional information about the nature of the error. This may wrap the original exception following the standard Python approach.


Benefits of using `FrameworkError`:


- **Additional properties**: Exceptions may include additional properties to provide a more detailed view of the error.
- **Preserved Error Chains**: Retains the full history of errors, giving developers full context for debugging.
- **Utility Functions:** Includes methods for formatting error stack traces and explanations, making them suitable for use with LLMs and other external tools.
- **Native Support:** Built on native Python Exceptions functionality, avoiding the need for additional dependencies while leveraging familiar mechanisms.

This structure ensures that users can trace the complete error history while clearly identifying any errors originating from the BeeAI Framework.

## Specialized Error Classes

The BeeAI Framework extends `FrameworkError` to create specialized error classes for different components or scenarios. This ensures that each part of the framework has clear and well-defined error types, improving debugging and error handling.

Framework error has two additional properties which help with agent processing, though ultimately the code that catches the exception will determine the appropriate action.

- **is_retryable** : hints that the error is retryable.
- **is_fatal** : hints that the error is fatal.

> [!TIP]
>
> Wrapping a standard Exception within a `FrameworkError` can be done by calling the `FrameworkError.ensure` static method. This will have no effect if the passed exception is already a `FrameworkError`.

### Aborts

- `AbortError`: Raised when an operation has been aborted.

### Tools

- `ToolError` : Raised when a problem is reported by a tool.
  - `ToolInputValidationError`, which extends ToolError, raised when input validation fails.

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

- `ParserError` : Raised when a parser fails to parse the input data. Includes additional *Reason*.

### Memory

- `Resource Error` : Raised when an error occurs with processing agent memory.
  - `ResourceFatalError` : Raised for particularly severe errors that are likely to be fatal (subclass of Resource Error).

### Emitter

- `EmitterError` : Raised when a problem occurs in the emitter.

### Backend

- `BackendError` : Raised when a backend encounters an error.
  - `ChatModelError` : Raised when a chat model fails to process input data. Subclass of BackendError.
- `MessageError` : Raised when a message processing fails.
