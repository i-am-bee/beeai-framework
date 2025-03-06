import logging

from beeai_framework.logger import Logger

# Configure logger with default log level
logger = Logger("app", level=logging.TRACE)

# Log at different levels
logger.trace("Trace!")
logger.debug("Debug!")
logger.info("Info!")
logger.warning("Warning!")
logger.error("Error!")
logger.fatal("Fatal!")
