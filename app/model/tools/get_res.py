# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore') 

def get_res(pred,gt):
    pred=np.array(pred)
    gt=np.array(gt)
    res=pd.DataFrame()
    res['pred'],res['gt']=pred,gt
    return res