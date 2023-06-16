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
from renew_center.tools.logger import setup_logger
from renew_center.dataclean.renew_clean import Clean
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from center.model.lgb_model import LGBModel
logger = setup_logger('logger')
# =============================================================================
class MyLGB(LGBModel):
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
        y_pred=pd.Series(y_pred,index=x_test.index) ##put date info into datframe index
        return y_pred

    def eval_result(self, pred, gt, Capacity):
        return 1-np.sqrt(((gt - pred)**2/Capacity**2).mean())


        
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
        
        
    
class Renew():
    def __init__(self, project_path):
        self.project_path = project_path
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
    def load_data(self, config,mode='kw'):
        self.real_load = pd.read_csv(os.path.join(
            config.area_path, 'data', 'history_cleaned_real_load.csv'), index_col='date', parse_dates=True)
        self.fore_wea = pd.read_csv(os.path.join(
            config.area_path, 'data', 'history_cleaned_XXL_nwp_weather.csv'), index_col='date', parse_dates=True)
        self.daily_wea = pd.read_csv(os.path.join(
            config.area_path, 'data', 'daily_cleaned_pred_weather.csv'), index_col='date', parse_dates=True)
        self.data = self.fore_wea.join(self.real_load).dropna()
        self.data.rename(columns={'GlobalR': 'rad_GlobalR', 'AirT': 'temp',
                                  'DirectR': 'rad_DirectR', 'RH': 'hum'}, inplace=True)
        if mode=='kw':
            self.data['load']=self.data['load']/1000
            
    def data_clean(self, data, config):
        df = deepcopy(data)  # not to affect origin data
        cleaner = Clean()
        cleaned_data = cleaner.clean(data, config)
        cleaned_data = cleaner.clean_area(data, config)
        return cleaned_data

    def make_label(self, data):
        pass

    def data_split_scale(self, data, label='load',random_state=123):
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
        from fea_engine import date_to_timeFeatures, dir_to_sincos_dir, fea_shift, speed_diff, rad_diff
        
        ####general FE
        df=date_to_timeFeatures(df)# time discretization feature        
        
        if config.area_type=='wind':##wind special FE
            speed_cols=[fea for fea in config.feas_used if 'speed' in fea] 
            df=speed_diff(df,speed_cols)
            
            dir_feas=[fea for fea in config.feas_used if 'dir' in fea] 
            df=dir_to_sincos_dir(df,dir_feas,delete=True)
            
        elif config.area_type=='pv':##pv special FE
            df=fea_shift(df,name='load',N=4)# lagged load feature  
        
            rad_cols=[fea for fea in config.feas_used if 'rad' in fea] 
            df=rad_diff(df,rad_cols)

        return df

    def subset_selection(self, model):
        # filter method
        from sklearn.feature_selection import SelectKBest, f_classif
        # Use correlation coefficients to select most relevant features
        selector = SelectKBest(score_func=f_classif,
                               k=int(len(self.x_train.columns)*0.8))
        x_selected = selector.fit_transform(self.x_train_scaled, self.y_train)
        feature_indices = selector.get_support(indices=True)
        filter_feas_used = self.x_train.columns[feature_indices]
        print('filter_feas_used:', filter_feas_used)

        # wrapper method use  Backward Elimination
        model.fit(self.x_train_scaled, self.y_train)
        y_val_pred = model.predict_proba(self.x_val_scaled)[:, 1]
        best_score = roc_auc_score(self.y_val, y_val_pred)
        wrapper_feas_used = list(self.x_train.columns)  # init fea_list
        for fea in tqdm(self.x_train.columns):
            wrapper_feas_used.remove(fea)
            model.fit(self.zscore(self.x_train[wrapper_feas_used],
                                  verbose='feas_used_transform', feas_used=wrapper_feas_used), self.y_train)
            y_val_pred = model.predict_proba(self.zscore(
                self.x_val[wrapper_feas_used], verbose='feas_used_transform', feas_used=wrapper_feas_used))[:, 1]
            score = roc_auc_score(self.y_val, y_val_pred)
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
        # Display confussion matrix
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            self.x_test_scaled[feas_used],
            self.y_test,
            display_labels=model.classes_,
            cmap=plt.cm.Blues
        )
        disp.ax_.set_title('Confusion matrix')
        plt.show()

        # Classification Report
        y_test_pred = model.predict(self.x_test_scaled[feas_used])
        print(classification_report(self.y_test, y_test_pred))

        # Display ROCCurve
        disp_roc = RocCurveDisplay.from_estimator(
            model,
            self.x_test_scaled[feas_used],
            self.y_test,
            name=name)
        disp_roc.ax_.set_title('ROC Curve')
        plt.show()


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
    parser.add_argument(
        "--mode",
        type=str,
        default="train",  # train/predcit
        help="train/predcit",
    )
    args = parser.parse_args()
# =============================================================================
    # get areas
    from config import EnvConfig, AreaConfig
    project_path = EnvConfig().project_path
    if args.area == "all":
        areas = os.listdir(os.path.join(project_path, "data", "area"))
    else:
        areas = args.area.split(',')  # list type
# =============================================================================
    renew = Renew(project_path)
    config = AreaConfig('NARI-19001-Xibei-abhnfd')
    # config = AreaConfig('NARI-19551-Xibei-jchjgf')
    
    renew.load_data(config,mode='kw')
    if len(renew.data)//96 < 30:
        logger.warning('len(data)//96<30')

    if args.mode == 'train':
        setattr(config, 'mode', 'train')
        setattr(config, 'data', renew.data)
# data_clean=============================================================================
        cleaner = Clean()
        # cln_data = cleaner.clean(data=config.data,
        #                              feas_used=list(renew.data.columns),
        #                              area='xxx',
        #                              area_type=config.area_type,
        #                              # 'train'(may delete bad feas)/'predcit'(fix data only)
        #                              capacity=config.capacity,
        #                              online=True,  # None for wind clean
        #                              Longitude=config.Longitude,  # None for wind clean
        #                              Latitude=config.Latitude,  # None for wind clean
        #                              # 3 inputs only, compare 2 feas' trend to clean, 0.8 means remain top 80% data, False if unwanted
        #                              trend=['load', 'rad', 0.8],
        #                              plot=[['load', 'rad'], "2022-03-19", 30])
        
        cln_data = cleaner.clean_area(config,online=True,plot=False) ##cln means cleaned
        # cleaned_data = cleaner.clean_area(config,online=True,plot=[['load', 'speed_30'], "2022-06-19", 30])
        cln_data=cln_data.dropna()# cleaned_data change unwanted data into NaN,so dropna used in 'train' mode    
# feature_engineering======================================================================        
        cln_FE_data=renew.feature_engineering(cln_data,config)  ##FE means feature_engineering
# data_split_scale=============================================================        
        renew.data_split_scale(cln_FE_data,random_state=123)# renew.x_train/x_val/x_test/x_train_scaled/x_val_scaled/x_test_scaled/y_train/y_val/y_test
# model train and finetune===========================================================        
        mylgb=MyLGB(config)
        best_model=mylgb.finetune(renew.x_train, renew.y_train, renew.x_val, renew.y_val, n_trials=200) ##finetune include train process
        
        
        
        # best_model=mylgb.train(renew.x_train, renew.y_train, renew.x_val, renew.y_val)
        
# model save===========================================================            
        setattr(config, 'model_path', os.path.join(config.area_path,'model'))
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path) 
        joblib.dump(best_model, os.path.join(config.model_path,'lgb.pkl'))
        
# model load===========================================================    
        best_model = joblib.load(os.path.join(config.model_path,'lgb.pkl'))
# model predict===========================================================            
        y_pred=mylgb.predict(best_model, renew.x_test)
        acc_mape=mylgb.eval_result(y_pred, renew.y_test, config.capacity)
        res=pd.concat([y_pred, renew.y_test],axis=1)
        res.columns=['pred','gt']
        mylgb.plot_peroid(res,filename='res',cols = ['pred','gt'],start_day = res.index[0],end_day=None,days = 10,maxmin=True)
        print(acc_mape)

    elif args.mode == 'predict':
        setattr(config, 'mode', 'predict')
        cln_data = cleaner.clean_area(config,online=True,plot=False) ##cln means cleaned
        cln_data=cln_data.interpolate(method='linear',limit_direction='both')# cleaned_data change unwanted data into NaN,so interpolate used in 'predict' mode  
        cln_FE_data=renew.feature_engineering(cln_data,config)
        mylgb=MyLGB(config)
# model load===========================================================    
        model = joblib.load(os.path.join(config.model_path,'lgb.pkl'))
# model predict===========================================================            
        y_pred=mylgb.predict(model, renew.x_test)        