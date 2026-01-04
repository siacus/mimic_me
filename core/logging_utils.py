import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent logging for the app."""
    logger = logging.getLogger()
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(level)
