import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore')

def setup_logger(logger_name='logger'):
    '''
    logging.info("info")
    logging.debug("debug")
    logging.warning("warning")
    logging.error("error")
    logging.critical("critical")
    '''
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logging.basicConfig(filemode='w')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG) 
        stream_handler.setFormatter(formatter)
    
        file_handler = logging.FileHandler(f"./logs/{logger_name}.log",mode='w')
        file_handler.setLevel(logging.DEBUG) 
        file_handler.setFormatter(formatter)
    
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger