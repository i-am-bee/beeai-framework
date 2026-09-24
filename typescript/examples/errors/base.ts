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
