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

<!-- embedme examples/errors/base.py -->
```py
from beeai_framework.errors import FrameworkError

error = FrameworkError("Fuction 'getUser' has failed.", is_fatal=True, is_retryable=False)
inner_error = FrameworkError("Cannot retrieve data from the API.")
innermost_error = ValueError("User with Given ID Does not exist!")

inner_error.__cause__ = innermost_error
error.__cause__ = inner_error

print(f"Message: {error.message}")  # Main error message
# Is the error fatal/retryable?
print(f"Meta: fatal:{FrameworkError.is_fatal(error)} retryable:{FrameworkError.is_retryable(error)}")
print(f"Cause: {error.get_cause()}")  # Prints the cause of the error
print(error.explain())  # Human-readable format without stack traces (ideal for LLMs)

```

_Source: /examples/errors/base.py_

Framework error also has two additional properties which help with agent processing, though ultimately the code that catches the exception will determine the appropriate action.

- **is_retryable** : hints that the error is retryable.
- **is_fatal** : hints that the error is fatal.

## Specialized Error Classes

The BeeAI Framework extends `FrameworkError` to create specialized error classes for different components or scenarios. This ensures that each part of the framework has clear and well-defined error types, improving debugging and error handling.

> [!TIP]
>
> Casting an unknown error to a `FrameworkError` can be done by calling the `FrameworkError.ensure` static method ([example](/examples/errors/cast.py)).

The definitions for these classes are typically local to the module where they are raised.

### Aborts

- `AbortError`: Raised when an operation has been aborted.

### Tools
<!-- embedme examples/errors/tool.py -->
```py
import asyncio

from beeai_framework import tool
from beeai_framework.tools import ToolError


async def main() -> None:
    @tool
    def dummy() -> None:
        """
        A dummy tool.
        """
        raise ToolError("Dummy error.")

    await dummy.run({})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ToolError as e:
        print("===CAUSE===")
        print(e.get_cause())
        print("===EXPLAIN===")
        print(e.explain())

```

_Source: /examples/errors/tool.py_

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

## Usage

To use Framework error, add the following import
```python
from beeai_framework.errors import FrameworkError
```

Add any additional custom errors you need in your code to the import, for example
```python
from beeai_framework.errors import FrameworkError, ChatModelError,ToolError
```

If you wish to create additional errors, you can extend `FrameworkError` or any of the other errors above:

```python
from beeai_framework.errors import FrameworkError

class MyCustomError(FrameworkError):
    def __init__(self, message: str = "My custom error", *, cause: Exception | None = None) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause)
```

You can wrap existing errors in a `FrameworkError`, for example:
```python
inner_err: Exception = ValueError("Value error")
error = FrameworkError.ensure(inner_err)
raise(error)
```

Framework error also has two additional properties which help with agent processing, though ultimately the code that catches the exception will determine the appropriate action.

- **is_retryable** : hints that the error is retryable.
- **is_fatal** : hints that the error is fatal.

these can be accessed via:

```python
err = FrameworkError("error")
isfatal: bool = FrameworkError.is_fatal(err)
isretryable: bool = FrameworkError.is_retryable(err)
```

This allows use of some useful functions within the error class.

For example the `explain` static method will return a string that may be more useful for an LLM to interpret:

```python
message: str = FrameworkError.ensure(error).explain()
```

See the source file [errors.py](python/beeai_framework/errors.py) for additional methods.
