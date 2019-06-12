import logging
import datetime
from Ldata_helper import *
from Lglobal_defs import *


def init_logger(name, fn):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(join(FOLDER_LOGS, fn + '_'+datetime.datetime.now().strftime('%d%H%M%S')+'.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger


logger = init_logger
