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


def eval_res(res, capacity):
    ## res(DataFrame) must at least have 'gt' and 'pred' cols
    df=deepcopy(res)
    def _nanwang_error(row):
        if row['gt']<0.1*capacity and row['pred']<0.1*capacity:
            return np.nan
        if row['gt']<0.2*capacity:
            return (row['gt'] - row['pred'])**2/(0.2*capacity)**2
        else:
            return (row['gt'] - row['pred'])**2/row['gt']**2                                
    df['error']=df.apply(_nanwang_error,axis=1)        
    return 1-np.sqrt(df['error'].mean()) ##NaN value will not be considered by mean() 