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

import lightgbm as lgb
import xgboost as xgb
from model.tools.logger import setup_logger
from model.data_clean import Clean
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
logger = setup_logger('logger')

class MyLGB():
    def __init__(self,config):
        self.config=config
    
    def build_model(self):
        return lgb.LGBMRegressor(**self.config.lgb_param)  
       
    def train(self, x_train, y_train,x_val, y_val):
        model = self.build_model()
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_val, y_val)],
                  early_stopping_rounds=20
                  )
        return model
    
    def predict(self, model,x_test):
        y_pred = model.predict(x_test)
        y_pred=pd.DataFrame(y_pred,index=x_test.index,columns=['pred']) ##put date info into datframe index
        return y_pred

    def eval_result(self, res, Capacity):
        ## res(DataFrame) must at least have 'gt' and 'pred' cols
        df=deepcopy(res)
        def _fun(row):
            if row['gt']<0.2*Capacity:
                return 1-np.sqrt((row['gt'] - row['pred'])**2/(0.2*Capacity)**2)
            else:
                return 1-np.sqrt((row['gt'] - row['pred'])**2/Capacity**2)                                 
        df['acc']=df.apply(_fun,axis=1)        
        
        return df['acc'].mean()

    def finetune(self, x_train,y_train,x_val,y_val, n_trials=100):

        import optuna

        def objective(trial):
            
            # Define hyperparameter Search Scope
            param = {
                'boosting_type':'gbdt',
                'class_weight':None, 
                'colsample_bytree':1.0, 
                'device':'cpu',
                'importance_type':'split', 
                'learning_rate':trial.suggest_float('learning_rate', 1e-5,1e-1),
                'max_depth':trial.suggest_int('max_depth', 2,10,step=1),
                'min_child_samples':91, 
                'min_child_weight':0.001,
                'min_split_gain':0.2, 
                'n_estimators':trial.suggest_int('n_estimators', 50,300,step=10),
                'n_jobs':-1, 
                'num_leaves':trial.suggest_int('max_depth', 2,50,step=1),
                'objective':None, 
                'random_state':1822, 
                'reg_alpha':trial.suggest_float('reg_alpha', 0.1, 1,step=0.1),
                'reg_lambda':trial.suggest_float('reg_lambda', 0.1, 1,step=0.1),
                'silent':True, 
                'subsample':trial.suggest_float('subsample', 0.1, 1,step=0.1), 
                'subsample_for_bin':200000,
                'subsample_freq':0
            }

            model = lgb.LGBMRegressor(**param)
            model.fit(x_train,y_train)
            y_val_pred = model.predict(x_val)
            mse = mean_squared_error(y_val, y_val_pred)
            return mse
        
        study = optuna.create_study(
            direction='minimize')  # maximize the auc
        study.optimize(objective, n_trials=n_trials)
        print("Best parameters:", study.best_params)
        best_model = lgb.LGBMRegressor( **study.best_params)
        best_model.fit(x_train,y_train)
        return best_model        