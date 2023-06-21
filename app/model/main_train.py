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

from model.tools.logger import setup_logger
logger = setup_logger('logger')

# =============================================================================
try:  ##any bug of whole procedure will be recorded in ./logs/logger.log
    if __name__ == "__main__":
        
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--area",
            type=str,
            default="2365",  # all for all areas // NARI-19008-Xibei-dtqlyf,NARI-19008-Xibei-dtqlyf for 2 areas
            help="name of areas to predict, 1, more, all areas both ok",
        )
        args = parser.parse_args()
    
    # get areas_list=============================================================
        if args.area == "all":
            areas = os.listdir(os.path.join(project_path, 'model','data', 'DQYC','20220807'))
        else:
            areas = args.area.split(',')  # list type
            
            
        for area in tqdm(areas):
            try: ##jump the bug area in training,any bug of this step will be recorded in ./logs/logger.log
                logger.info(f'{area} start training!##################################')
            # get_config=============================================================================
                from model.config.parameter_config import ConfigParser    
                config = ConfigParser(area,project_path)
                
                logger.info(f'area_train_path:{config.area_train_path}')
                logger.info(f'area_type:{config.area_type}')
                logger.info(f'capacity:{config.capacity}')
                logger.info(f'feas_used:{config.feas_used}')
                logger.info(f'feas_selected:{config.feas_selected}')
                logger.info(f'trend:{config.trend}')
                
                setattr(config, 'mode', 'train')
                logger.info('get_config succeed!')
            # create_agent====================================================================    
                from model.short_term_agent import ShortTermAgent
                agent = ShortTermAgent()
                logger.info('create_agent succeed!')
            # load_data===============================================================   
                agent.load_data(config, mode='train', unit='MW') ##unit of MW in power file
                setattr(config, 'data', agent.data)
                logger.info('load_data succeed!')
            # data_clean=============================================================================
                from model.data_clean import Clean
                cleaner = Clean()        
                # cln_data = cleaner.clean_area(config,online=False,plot=False) ##cln means cleaned
                cln_data = cleaner.clean_area(config,online=False,plot=[['Power', 'Groundradiation'], "2022-05-02", 30])
                cln_data=cln_data.dropna()# cleaned_data change unwanted data into NaN,so dropna used in 'train' mode    
                logger.info('data_clean succeed!')
            # feature_engineering======================================================================        
                cln_FE_data=agent.feature_engineering(cln_data,config)  ##FE means feature_engineering
                logger.info('feature_engineering succeed!')
            # data_split_scale=============================================================        
                agent.data_split_scale(cln_FE_data,random_state=123)# renew.x_train/x_val/x_test/x_train_scaled/x_val_scaled/x_test_scaled/y_train/y_val/y_test
                logger.info('data_split_scale succeed!')
            # create_model===========================================================        
                from model.model import MyLGB
                mylgb=MyLGB(config)
                logger.info('create_model succeed!')
            # subset_selection======================================================================        
                base_model=mylgb.build_model()
                feas_selected=agent.subset_selection(base_model,config)  ##use base model for feas selection
                logger.info('subset_selection succeed!')
            # save feas_selected to area_info.csv==========================================================
                feas_selected_str='+'.join(feas_selected)
                selected_idx=config.area_info[config.area_info["FarmCode"] == config.area].index
                config.area_info.loc[selected_idx, "feas_selected"]=feas_selected_str
                config.area_info.to_csv(config.area_info_path,index=False)
                logger.info('save_feas_selected succeed!')
            # model train and finetune=======================================================        
                best_model=mylgb.finetune(config, agent.x_train[feas_selected], agent.y_train,
                                          agent.x_val[feas_selected],   agent.y_val, n_trials=100) ##finetune include train process
                # best_model=mylgb.train(agent.x_train, agent.y_train, agent.x_val, agent.y_val)
                logger.info('model_finetune succeed!')
            # model_save===========================================================  
                import joblib          
                setattr(config, 'model_path', os.path.join(config.area_train_path,'model'))
                if not os.path.exists(config.model_path):
                    os.makedirs(config.model_path) 
                joblib.dump(best_model, os.path.join(config.model_path,'lgb.pkl'))
                logger.info('model_save succeed!')
            # # model load===========================================================    
            #     best_model = joblib.load(os.path.join(config.model_path,'lgb.pkl'))
            
            # model_predict===========================================================            
                y_pred=mylgb.predict(best_model, agent.x_test[feas_selected])
                logger.info('model_predict succeed!')
            # predict post_process===========================================================    
                if config.area_type=='pv':
                    y_pred=cleaner.sunset_zero(y_pred,'pred')   
                    logger.info('post_process succeed!')
            # eval_res===========================================================
                from model.tools.get_res import get_res
                from model.tools.eval_res import eval_res
                res=get_res(y_pred, agent.y_test)
                acc_mape=eval_res(res, config.capacity)
                logger.info(f'acc_mape: {acc_mape} !!!!!!!!!!!')    
                from model.tools.plot_view import plot_peroid
                plot_peroid(res,filename='res',cols = ['pred','gt'],start_day = res.index[0],end_day=None,days = 30,maxmin=False)
                logger.info('eval_res succeed!')
            except:
                logger.info(f'{area} fail!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                logger.info(traceback.format_exc())
                continue
except:
    logger.info(traceback.format_exc())
