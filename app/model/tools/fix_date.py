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

def fix_date(data_cleaned,data,freq='15T'):
    '''
    make date index of data_cleaned as same as data
    '''
    df=deepcopy(data)
    df_cleaned=deepcopy(data_cleaned)
    time=pd.DataFrame(pd.date_range(start=df.index[0], end=df.index[-1],freq=freq),columns=['date'])
    time.index=time['date']
    df_cleaned = df_cleaned.join(time, how='right')
    # df_cleaned=pd.merge(df_cleaned,time,on='date',how='right')        
    return df_cleaned