import logging
import sys

# Logging setup
logging.getLogger(__package__).addHandler(logging.NullHandler())
logger = logging.getLogger(__package__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())

def set_log_to_file(filename, level=logging.INFO):
    logger.addHandler(logging.FileHandler(filename))
    logger.setLevel(level)

def set_level(level):
    if isinstance(level, int):
        logger.setLevel(level)

    elif isinstance(level, str):
        if level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif level == 'info':
            logger.setLevel(logging.INFO)
        elif level == 'warning':
            logger.setLevel(logging.WARNING)
        else:
            raise ValueError(f'unknown logging level {level}')
    else:
        raise ValueError(f'unknown logging level {level}')

def set_log_to_console(level=logging.INFO):
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    set_level(level)