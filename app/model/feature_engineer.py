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

def date_to_timeFeatures(data):
    df=deepcopy(data)
    df['date'] = df.index
    df['HourOfDay']=df['date'].apply(lambda x : x.hour / 23.0 - 0.5)
    # df['DayOfWeek'] = df['date'].apply(lambda x: x.dayofweek / 6.0 - 0.5)
    # df['DayOfMonth'] = df['date'].apply(lambda x: (x.day - 1) / 30.0 - 0.5)
    df['DayOfYear'] = df['date'].apply(lambda x: (x.dayofyear - 1) / 365.0 - 0.5)
    del df['date']  
   
    return df

def fea_shift(data,name='load',N=4):
    df=deepcopy(data)
    for i in range(96,96+N+1):
        # print(i)
        df['{}-{}-point'.format(name,i)]=df[name].shift(i)
    df=df.dropna()
    return df

def speed_diff(data,ws_cols):
    df=deepcopy(data)
    for col in ws_cols:        
        df[f'{col}_diff']=df[col]-df[col].shift(1)
    df=df.dropna()
    return df

def dir_to_sincos_dir(data,wd_cols,delete=True):
    df=deepcopy(data)
    for col in wd_cols:        
        df[f'{col}_sin']=np.sin(np.pi*df[f'{col}']/180)
        df[f'{col}_cos']=np.cos(np.pi*df[f'{col}']/180)
        if delete==True:
            del df[f'{col}']
    return df

def rad_diff(data,rad_cols):
    df=deepcopy(data)
    for col in rad_cols:        
        df[f'{col}_diff']=df[col]-df[col].shift(1)
    df=df.dropna()
    return df

