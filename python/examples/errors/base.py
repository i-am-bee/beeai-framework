from beeai_framework.errors import FrameworkError

# Create the main FrameworkError instance
error = FrameworkError("Function 'getUser' has failed.", is_fatal=True, is_retryable=False)
inner_error = FrameworkError("Cannot retrieve data from the API.")
innermost_error = ValueError("User with Given ID Does not exist!")

# Chain the errors together using __cause__
inner_error.__cause__ = innermost_error
error.__cause__ = inner_error

# Set the context dictionary for the top level error
# Add any additional context here. This will help with debugging
error.context["workflow"] = "activity_planner"
error.context["provider"] = "ollama"
error.context["chat_model"] = "granite3.2:8b"

# Print some properties of the error
print("\n-- Error properties:")
print(f"Message: {error.message}")  # Main error message
# Is the error fatal/retryable?
print(f"Meta: fatal:{FrameworkError.is_fatal(error)} retryable:{FrameworkError.is_retryable(error)}")
print(f"Cause: {error.get_cause()}")  # Prints the cause of the error
print(f"Context: {error.context}")  # Prints the dictionary of the error context

print("\n-- Explain:")
print(error.explain())  # Human-readable format without stack traces (ideal for LLMs)
print("\n-- str():")
print(str(error))  # Human-readable format (for debug)
