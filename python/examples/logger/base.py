import logging

from beeai_framework.logger import BeeLogger

# Configure logger with default log level
logger = BeeLogger("app", level=logging.TRACE)

# Log at different levels
logger.trace("Trace!")
logger.debug("Debug!")
logger.info("Info!")
logger.warning("Warning!")
logger.error("Error!")
logger.fatal("Fatal!")
