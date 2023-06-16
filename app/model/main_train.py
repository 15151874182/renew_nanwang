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

from tools.plot_view import plot_peroid
from tools.logger import setup_logger

# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area",
        type=str,
        default="all",  # all for all areas // NARI-19008-Xibei-dtqlyf,NARI-19008-Xibei-dtqlyf for 2 areas
        help="name of areas to predict, 1, more, all areas both ok",
    )
    args = parser.parse_args()
# get project_path=============================================================================
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# get areas_list=============================================================
    if args.area == "all":
        areas = os.listdir(os.path.join(project_path, 'model','data', 'DQYC','20220807'))
    else:
        areas = args.area.split(',')  # list type
# get config=============================================================================
    from config.parameter_config import ConfigParser    
    config = ConfigParser('1078',project_path)
    # config = ConfigParser('2365',project_path)
    setattr(config, 'mode', 'train')
# get config====================================================================    
    from short_term_agent import ShortTermAgent
    agent = ShortTermAgent()
    agent.load_data(config, mode='train', unit='MW') ##unit of MW in power file
    
    
    setattr(config, 'data', agent.data)
# data_clean=============================================================================
    from data_clean import Clean
    cleaner = Clean()        
    cln_data = cleaner.clean_area(config,online=True,plot=False) ##cln means cleaned
    # cln_data = cleaner.clean_area(config,online=True,plot=[['Power', 'Speed100'], "2022-05-02", 30])
    cln_data=cln_data.dropna()# cleaned_data change unwanted data into NaN,so dropna used in 'train' mode    
# feature_engineering======================================================================        
    cln_FE_data=agent.feature_engineering(cln_data,config)  ##FE means feature_engineering
# data_split_scale=============================================================        
    agent.data_split_scale(cln_FE_data,random_state=123)# renew.x_train/x_val/x_test/x_train_scaled/x_val_scaled/x_test_scaled/y_train/y_val/y_test
# model train and finetune===========================================================        
    from model import MyLGB
    mylgb=MyLGB(config)
    best_model=mylgb.finetune(agent.x_train, agent.y_train, agent.x_val, agent.y_val, n_trials=200) ##finetune include train process
    
    
    
    # best_model=mylgb.train(renew.x_train, renew.y_train, renew.x_val, renew.y_val)
    
# model save===========================================================  
    import joblib          
    setattr(config, 'model_path', os.path.join(config.area_path,'model'))
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path) 
    joblib.dump(best_model, os.path.join(config.model_path,'lgb.pkl'))
    
# model load===========================================================    
    best_model = joblib.load(os.path.join(config.model_path,'lgb.pkl'))
# model predict===========================================================            
    y_pred=mylgb.predict(best_model, agent.x_test)
    acc_mape=mylgb.eval_result(y_pred, agent.y_test, config.capacity)
    res=pd.concat([y_pred, agent.y_test],axis=1)
    res.columns=['pred','gt']
    plot_peroid(res,filename='res',cols = ['pred','gt'],start_day = res.index[0],end_day=None,days = 30,maxmin=False)
    print(acc_mape)
