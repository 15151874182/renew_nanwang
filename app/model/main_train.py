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

# get project_path=============================================================================
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_path)
sys.path.insert(0,project_path) ##search mine dir first, because model.tools in somewhere else of sys.path
    
from model.tools.plot_view import plot_peroid
from model.tools.logger import setup_logger
logger = setup_logger('logger')
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

# get areas_list=============================================================
    if args.area == "all":
        areas = os.listdir(os.path.join(project_path, 'model','data', 'DQYC','20220807'))
    else:
        areas = args.area.split(',')  # list type
        
        
    for area in tqdm(areas):
    # get config=============================================================================
        from model.config.parameter_config import ConfigParser    
        config = ConfigParser(area,project_path)
        setattr(config, 'mode', 'train')
    # get config====================================================================    
        from model.short_term_agent import ShortTermAgent
        agent = ShortTermAgent()
        agent.load_data(config, mode='train', unit='MW') ##unit of MW in power file
        setattr(config, 'data', agent.data)
    # data_clean=============================================================================
        from model.data_clean import Clean
        cleaner = Clean()        
        cln_data = cleaner.clean_area(config,online=True,plot=False) ##cln means cleaned
        # cln_data = cleaner.clean_area(config,online=True,plot=[['Power', 'Speed100'], "2022-05-02", 30])
        cln_data=cln_data.dropna()# cleaned_data change unwanted data into NaN,so dropna used in 'train' mode    
    # feature_engineering======================================================================        
        cln_FE_data=agent.feature_engineering(cln_data,config)  ##FE means feature_engineering
    # data_split_scale=============================================================        
        agent.data_split_scale(cln_FE_data,random_state=123)# renew.x_train/x_val/x_test/x_train_scaled/x_val_scaled/x_test_scaled/y_train/y_val/y_test
    # create model by config===========================================================        
        from model.model import MyLGB
        mylgb=MyLGB(config)

    # subset_selection======================================================================        
        base_model=mylgb.build_model()
        feas_selected=agent.subset_selection(base_model,config)  ##use base model for feas selection
    # save feas_selected to area_info.csv==========================================================
        feas_selected_str='+'.join(feas_selected)
        selected_idx=config.area_info[config.area_info["FarmCode"] == config.area].index
        config.area_info.loc[selected_idx, "feas_selected"]=feas_selected_str
        config.area_info.to_csv(config.area_info_path)
    # model train and finetune=======================================================        
        best_model=mylgb.finetune(config, agent.x_train[feas_selected], agent.y_train,
                                  agent.x_val[feas_selected],   agent.y_val, n_trials=200) ##finetune include train process
        # best_model=mylgb.train(agent.x_train, agent.y_train, agent.x_val, agent.y_val)
        
    # model save===========================================================  
        import joblib          
        setattr(config, 'model_path', os.path.join(config.area_train_path,'model'))
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path) 
        joblib.dump(best_model, os.path.join(config.model_path,'lgb.pkl'))
        
    # # model load===========================================================    
    #     best_model = joblib.load(os.path.join(config.model_path,'lgb.pkl'))
    
    # model predict===========================================================            
        y_pred=mylgb.predict(best_model, agent.x_test[feas_selected])
    # predict post process===========================================================    
        if config.area_type=='pv':
            y_pred=cleaner.sunset_zero(y_pred,'pred')   
    # eval_res===========================================================
        from model.tools.get_res import get_res
        from model.tools.eval_res import eval_res
        res=get_res(y_pred, agent.y_test)
        acc_mape=eval_res(res, config.capacity)
        print(acc_mape)    
        plot_peroid(res,filename='res',cols = ['pred','gt'],start_day = res.index[0],end_day=None,days = 30,maxmin=False)

