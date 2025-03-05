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
