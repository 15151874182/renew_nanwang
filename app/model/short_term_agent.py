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

import lightgbm as lgb
import xgboost as xgb
from model.tools.logger import setup_logger
from model.data_clean import Clean
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
logger = setup_logger('logger')

# =============================================================================
class ShortTermAgent():
    def __init__(self):
        self.feas_used = []  # final feas used to train and predict model
        
    # define my zscore function which can use feas_used
    def zscore(self, data, mode, feas_used):
        df = deepcopy(data)  # not to affect origin data
        if mode == 'fit_transform':
            self.u = df.mean()
            self.std = df.std()
            scaled_data = (df-self.u)/self.std
        elif mode == 'transform':
            scaled_data = (df-self.u)/self.std
        elif mode == 'feas_used_transform':
            u_selected = self.u[feas_used]
            std_selected = self.std[feas_used]
            scaled_data = (df[feas_used]-u_selected)/std_selected
        return scaled_data

    # customerize for different project
    def load_data(self, config, mode='train', unit='MW'): #mode='train'/'predict'  unit='MW'/'KW'
        if mode=='train':
            self.real_load = pd.read_csv(os.path.join(
                config.area_train_path, 'IN','DQYC_IN_HISTORY_POWER_LONG.txt'), sep=' ',index_col='Datetime', parse_dates=True)
            del self.real_load['PlantID']
            self.fore_wea = pd.read_csv(os.path.join(
                config.area_train_path, 'IN', '0','DQYC_IN_FORECAST_WEATHER_H.txt'), sep=' ', index_col='Datetime', parse_dates=True)
            self.data = self.fore_wea.join(self.real_load).dropna(how='all')  ##only all NAN in one row will be dropped
        elif mode=='predict':
            self.fore_wea = pd.read_csv(os.path.join(
                config.area_predict_path, 'IN', '0','DQYC_IN_FORECAST_WEATHER.txt'), sep=' ', index_col='Datetime', parse_dates=True)
            self.data=self.fore_wea
            
        # self.data.rename(columns={'GlobalR': 'rad_GlobalR', 'AirT': 'temp',
        #                           'DirectR': 'rad_DirectR', 'RH': 'hum'}, inplace=True)
        if mode=='KW':
            self.data['load']=self.data['load']/1000
            
    def data_clean(self, data, config):
        df = deepcopy(data)  # not to affect origin data
        cleaner = Clean()
        cleaned_data = cleaner.clean(data, config)
        cleaned_data = cleaner.clean_area(data, config)
        return cleaned_data

    def data_split_scale(self, data, label='Power',random_state=123):
        df = deepcopy(data)  # not to affect origin data
        y = df[label]
        del df[label]
        x = df
        # split dataset into train,val,test and scale
        x_trainval, self.x_test, y_trainval, self.y_test = train_test_split(
            x, y, random_state=random_state, shuffle=False, test_size=0.2)  # split test first
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_trainval, y_trainval, random_state=random_state, shuffle=False, test_size=0.2)  # then split val
        self.x_train_scaled = self.zscore(
            self.x_train, mode='fit_transform', feas_used=None)
        self.x_val_scaled = self.zscore(
            self.x_val, mode='transform', feas_used=None)
        self.x_test_scaled = self.zscore(
            self.x_test, mode='transform', feas_used=None)

    def feature_engineering(self, data, config):
        df = deepcopy(data)
        from model.feature_engineer import date_to_timeFeatures, dir_to_sincos_dir, fea_shift, speed_diff, rad_diff
        
        ####general FE
        df=date_to_timeFeatures(df)# time discretization feature        
        df['capacity']=config.capacity
        
        if config.area_type=='wind':##wind special FE
            speed_cols=[fea for fea in config.feas_used if 'Speed' in fea] 
            df=speed_diff(df,speed_cols)
            
            dir_feas=[fea for fea in config.feas_used if 'Dir' in fea] 
            df=dir_to_sincos_dir(df,dir_feas,delete=True)
            
        elif config.area_type=='pv':##pv special FE
            # df=fea_shift(df,name='Power',N=4)# lagged load feature  
        
            # rad_cols=[fea for fea in config.feas_used if 'rad' in fea] 
            # df=rad_diff(df,rad_cols)
            pass
        return df

    def subset_selection(self, model,config):
        # filter method
        from sklearn.feature_selection import SelectKBest, f_classif
        # Use correlation coefficients to select most relevant features
        selector = SelectKBest(score_func=f_classif,
                               k=int(len(self.x_train.columns)*0.8))
        x_selected = selector.fit_transform(self.x_train, self.y_train)
        feature_indices = selector.get_support(indices=True)
        filter_feas_used = self.x_train.columns[feature_indices]
        print('filter_feas_used:', filter_feas_used)

        # wrapper method use  Backward Elimination
        model.fit(self.x_train, self.y_train)
        y_val_pred = model.predict(self.x_val)
        from model.tools.get_res import get_res
        from model.tools.eval_res import eval_res
        res=get_res(y_val_pred, self.y_val)
        best_score = eval_res(res, config.capacity)
        wrapper_feas_used = list(self.x_train.columns)  # init fea_list
        for fea in tqdm(self.x_train.columns):
            wrapper_feas_used.remove(fea)
            model.fit(self.x_train[wrapper_feas_used], self.y_train)
            y_val_pred = model.predict(self.x_val[wrapper_feas_used])
            
            res=get_res(y_val_pred, self.y_val)
            score = eval_res(res, config.capacity)
            if score < best_score:  # if remove this fea and score decrease means this fea should remain
                wrapper_feas_used.append(fea)  # recovery this fea
            else:
                best_score = score  # update best_score
        print('wrapper_feas_used:', wrapper_feas_used)

        # Embedded method use Lasso
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.001)
        lasso.fit(self.x_train_scaled.values, self.y_train)
        embedded_feas_used = self.x_train.columns[lasso.coef_ != 0]
        print('embedded_feas_used:', embedded_feas_used)

        return list(filter_feas_used), list(wrapper_feas_used), list(embedded_feas_used)
        


    def investigate_model(self, model, feas_used, name='tuned_SVC'):
        pass


    