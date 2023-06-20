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
sys.path.insert(0,project_path) ##search mine dir first, because model.tools in somewhere else of sys.path

from model.tools.logger import setup_logger
logger = setup_logger('logger')

# =============================================================================
try:
    if __name__ == "__main__":

        # get sys.argv=============================================================================
        day_area_path=sys.argv[1]
        # get area=============================================================
        area=day_area_path.split('/')[1]
        logger.info(f'{area} start predicting!##################################')
        # write_STATUS_file=============================================================
        STATUS_path=os.path.join(project_path,'predict/data/DQYC',day_area_path,'STATUS.TXT').replace('IN', 'OUT')
        f=open(STATUS_path,mode='w') ##follow rules of competition
        f.write('1\n') ##successful startup the procedure
        logger.info('write_STATUS_file succeed!')
        try:
        # get_config=============================================================================
            from model.config.parameter_config import ConfigParser    
            config = ConfigParser(area,project_path)
            setattr(config, 'mode', 'predict')
            logger.info('get_config succeed!')
        # create_agent====================================================================    
            from model.short_term_agent import ShortTermAgent
            agent = ShortTermAgent()
            logger.info('create_agent succeed!')
        # load_data===============================================================     
            agent.load_data(config, mode='predict', unit='MW') ##unit of MW in power file
            setattr(config, 'data', agent.data)
            logger.info('load_data succeed!')
        # data_clean=============================================================================
            from model.data_clean import Clean
            cleaner = Clean()        
            config.feas_used.remove('Power') ##'Power' not need in predict
            config.trend=None 
            try: # make sure clean_area wont fail!
                cln_data = cleaner.clean_area(config,online=True,plot=False) ##cln means cleaned
            except:
                cln_data=config.data[config.feas_selected].interpolate(method='linear')
                logger.info(traceback.format_exc())
            # cln_data = cleaner.clean_area(config,online=True,plot=[['Power', 'Speed100'], "2022-05-02", 30])
            # cln_data=cln_data.dropna()# ## nothing can be dropped in 'predict' mode! 
            logger.info('data_clean succeed!')
        # feature_engineering======================================================================        
            cln_FE_data=agent.feature_engineering(cln_data,config)  ##FE means feature_engineering         
            cln_FE_data=cln_FE_data[config.feas_selected]
            logger.info('feature_engineering succeed!')
        # model_load===========================================================   
            from model.model import MyLGB
            mylgb=MyLGB(config)
            import joblib 
            setattr(config, 'model_path', os.path.join(config.area_train_path,'model'))
            best_model = joblib.load(os.path.join(config.model_path,'lgb.pkl'))
            logger.info('model_load succeed!')
        # model_predict===========================================================            
            y_pred=mylgb.predict(best_model, cln_FE_data)
            logger.info('model_predict succeed!')
        # predict post_process===========================================================    
            if config.area_type=='pv':
                y_pred=cleaner.sunset_zero(y_pred,'pred')   
                logger.info('post_process succeed!')
        # save_result into output===========================================================
            y_pred.index.name='Datetime'
            y_pred.columns=['Power']
            y_pred.to_csv(os.path.join(config.area_predict_path,'OUT','DQYC_OUT_PREDICT_POWER.txt'))
            f.write('2') #successful endup the procedure!
        except:
            f.write('3') #something wrong with procedure!
        f.close()
        logger.info('save_result succeed!')
except:
    logger.info(traceback.format_exc())