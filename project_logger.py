import logging

FILE_HANDLER = "debug.log"

def setup_logger():
    FORMAT = "%(name)s :: %(levelname)s :: %(funcName)s :: %(message)s"

    logger = logging.getLogger("Methylation")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(FORMAT)
    log_file_handler = logging.FileHandler(FILE_HANDLER, 'w')
    log_file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.ERROR)

    logger.addHandler(log_file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_logger(name):
    logger = logging.getLogger("Methylation." + name)
    return logger
