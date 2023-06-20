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
    date_index=gt.index
    pred=np.array(pred).reshape(-1)
    gt=np.array(gt).reshape(-1)
    res=pd.DataFrame(index=date_index)
    res['pred'],res['gt']=pred,gt
    return res