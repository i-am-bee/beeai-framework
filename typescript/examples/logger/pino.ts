import { Logger } from "beeai-framework/logger/logger";
import { pino } from "pino";

// Create a custom pino logger
const customLogger = pino({
  name: "app",
});

// Use the custom pino instance within the framework
const frameworkLogger = new Logger(
  {
    level: "info", // Set the log level
    name: "framework", // Set the logger name
  },
  customLogger, // Pass the custom pino instance
);
