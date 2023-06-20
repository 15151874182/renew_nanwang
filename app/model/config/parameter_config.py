# -*- coding: utf-8 -*-
"""
@author: cty
"""

import os
import pandas as pd
import glob
import json
from copy import deepcopy


# =============================================================================
class ConfigParser:
    
    def __init__(self, area, project_path):
        self.area = area
        self.area_train_path = os.path.join(project_path, 'model','data', 'DQYC','20220807',area)
        self.area_predict_path = os.path.join(project_path, 'predict','data', 'DQYC','20220807',area)
        self.area_info_path=os.path.join(project_path, 'model','config','area_info.csv')
        self.area_info = pd.read_csv(self.area_info_path,dtype={'FarmCode':str})
        
        self.capacity = self.area_info[self.area_info["FarmCode"] == area]['capacity'].iloc[0]
        self.Longitude = self.area_info[self.area_info["FarmCode"] == area]['Longitude'].iloc[0]
        self.Latitude = self.area_info[self.area_info["FarmCode"] == area]['Latitude'].iloc[0]
        self.area_type = self.area_info[self.area_info["FarmCode"] == area]['area_type'].iloc[0]

        self.models_used = self.area_info[self.area_info["FarmCode"] == area]['models_used'].iloc[0]
        self.feas_used = self.area_info[self.area_info["FarmCode"] == area]['feas_used'].iloc[0].split('+')
        self.feas_selected = self.area_info[self.area_info["FarmCode"] == area]['feas_selected'].iloc[0].split('+')
        
        #In json.loads(dict), '' for key and value must be replaced by "" in dict, use true not True 
        self.xgb_param = json.loads(self.area_info[self.area_info["FarmCode"] == area]['xgb_param'].iloc[0])
        self.lgb_param = json.loads(self.area_info[self.area_info["FarmCode"] == area]['lgb_param'].iloc[0])
        self.lr_param = json.loads(self.area_info[self.area_info["FarmCode"] == area]['lr_param'].iloc[0])
        
        self.day_point = self.area_info[self.area_info["FarmCode"] == area]['day_point'].iloc[0]   
        self.trend = self.area_info[self.area_info["FarmCode"] == area]['trend'].iloc[0].split('+')
        
        
if __name__ == '__main__':
    config=ConfigParser('NARI-19012-Xibei-gbdyfd')