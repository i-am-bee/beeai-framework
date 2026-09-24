import { Logger, LoggerLevel } from "beeai-framework/logger/logger";

// Configure logger defaults
Logger.defaults.pretty = true; // Pretty-print logs (default: false, can also be set via ENV: BEE_FRAMEWORK_LOG_PRETTY=true)
Logger.defaults.level = LoggerLevel.TRACE; // Set log level to trace (default: TRACE, can also be set via ENV: BEE_FRAMEWORK_LOG_LEVEL=trace)
Logger.defaults.name = undefined; // Optional name for logger (default: undefined)
Logger.defaults.bindings = {}; // Optional bindings for structured logging (default: empty)

// Create a child logger for your app
const logger = Logger.root.child({ name: "app" });

// Log at different levels
logger.trace("Trace!");
logger.debug("Debug!");
logger.info("Info!");
logger.warn("Warning!");
logger.error("Error!");
logger.fatal("Fatal!");
