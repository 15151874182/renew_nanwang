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

def plot_peroid(res,filename,cols = ["gt","pred"],start_day = "2021-11-12",end_day=None,days = 30,maxmin=True):
    import matplotlib as mpl

    start_day = pd.to_datetime(start_day)
    
    plt.figure(figsize=(30,10))
    #解决中文或者是负号无法显示的情况
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams['axes.unicode_minus'] = False
    
    #取需要画图的时间段数据
    if end_day ==None:
        end_day = start_day + pd.Timedelta(days=days)
    else:
        end_day = pd.to_datetime(end_day)

    df=res[start_day:end_day]
    
    for col in cols:
        if maxmin==False:
            plt.plot(df[col],label=col,alpha=1,linewidth =1.5)
        elif maxmin==True:
            df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
            plt.plot(df[col],label=col,alpha=1,linewidth =1.5)
            
    plt.legend(loc="upper left",fontsize='x-large')
    plt.title(f"{filename}",fontsize='x-large')
    # plt.savefig(f"./figure/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()
    plt.close()