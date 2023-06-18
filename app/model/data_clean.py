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

from datetime import date, timedelta
from tools.plot_view import plot_peroid
from tools.logger import setup_logger
from tools.fix_date import fix_date
logger=setup_logger('logger')

class Clean():
    def __init__(self):
        pass
    def clean(self,data,feas_used,
              area,
              area_type,
              capacity,
              online,   ##None for wind clean
              Longitude, ##None for wind clean
              Latitude, ##None for wind clean
              trend,   ##['fea1','fea2'], None if unwanted
              plot=False): ##['fea1','fea2'..], False if unwanted

#preparing ====================================================================        
        day_point=96  ##15min time freq everyday
        df=deepcopy(data)
        df=df[feas_used]
        if area_type=='pv' and online==True:
            from astral.sun import sun
            from astral import LocationInfo     
            self.sun=sun
            self.LocationInfo=LocationInfo            
            #suntime是日出日落信息，需要联网
            self.suntime = self.download_suntime(area=area,
                                                belong='china',
                                                Longitude=Longitude,
                                                Latitude=Latitude,
                                                begin="2018-1-1",
                                                end="2024-12-31")
            self.suntime["sunrise_before"] = self.suntime["Sunrise"].apply(lambda x:pd.to_datetime(x).round('15T'))
            self.suntime["sunset_after"] = self.suntime["Sunset"].apply(lambda x:pd.to_datetime(x).round('15T'))  
            self.suntime.to_csv('data/suntime.csv')
            
        elif area_type=='pv' and online==False:
            self.suntime = pd.read_csv('data/suntime.csv')
            
# =============================================================================             
        for col in tqdm(feas_used): ##every feas should handle_NAN,handle_limit,handle_constant
            
        #### fea name should follow rules as below
            if "Speed" in col:    ##wind speed
                upper,lower = 35,0  
            elif "Dir" in col:    ##wind direction
                upper,lower = 360,0  
            elif "rad" in col:    ##radiation
                upper,lower = 1400,0     
            elif "Temp" in col: ##temperature   
                upper,lower = 48,-15  
            elif "Hum" in col:  ##humidity
                upper,lower = 100,-0  
            elif "Perss" in col:##pressure    ##misspelling in original data
                upper,lower = 1500,640  
            elif "Power" in col:
                upper,lower = capacity,0  
            else:
                upper,lower=float('inf'),float('-inf')
            logging.info(f"{col},upper={upper},lower={lower}")   
            
#handle_NAN =================================================================  
            df=self.handle_NAN(df,col,threshold=day_point//6)
#handle_limit================================================================            
            df=self.handle_limit(df,col,lower,upper)
#special strategy for wind/pv================================================================             
            if area_type=='wind':
                pass  ##special strategy for wind if have will be put here 
            elif area_type=='pv':
                if 'rad' in col or 'Power' in col:  ## fea related to suntime
                    df=self.sunset_zero(df,col)
#handle_constant================================================================        
        ##constant=0 or capacity will be ignored,constant=0 problem will be put into similarity detect part             
            df=self.handle_constant(df,col,capacity,threshold=day_point//6) 




#handle_trend if want==============================================================
        if trend:
            df=self. handle_trend(df, trend, day_point)
#fix_date==================================================================     
        df=fix_date(df,data,freq='15T')
#if plot cleaned result========================================================================        
        if plot:
            feas,start_day,days=plot[0],plot[1],plot[2]
            plot_peroid(data,'before',cols = feas,start_day = start_day,end_day=None,days = days,maxmin=True)
            plot_peroid(df,'after',cols = feas,start_day = start_day,end_day=None,days = days,maxmin=True)   
        del df['date']
        return df   ##after clean process, this return df has NaN which must drop or interpolate according to train or predict model
            
            
    def clean_area(self,config,online=True,plot=True):
        return self.clean(data=config.data,
                          feas_used=config.feas_used,
                          area=config.area,
                          area_type=config.area_type,
                          capacity=config.capacity,
                          online=online,   ##None for wind clean
                          Longitude=config.Longitude, ##None for wind clean
                          Latitude=config.Latitude, ##None for wind clean
                          trend=config.trend,   ##['fea1','fea2'], None if unwanted
                          plot=plot) ##['fea1','fea2'..], False if unwanted
    
    def check_length(self,data,day_point,threshold):
        days=len(data)//day_point
        ##return True if number of data is enough
        return True if days>=threshold else False

    def handle_NAN(self,data,col,threshold):
        df=deepcopy(data)
        #### spline interpolate if continues num of NAN is within threshold
        df[col].interpolate(method='linear', limit=threshold,axis=0)
        return df

    def handle_limit(self,data,col,lower,upper):
        df=deepcopy(data)
        #### give NAN if over the lower/upper limit
        df[col] = df[col].apply(lambda x:np.nan if (x>upper and x<lower) else x)
        return df

    def handle_constant(self,data,col,capacity,threshold):
        '''
        delete threshold points constant data[col] except constant=0 or capacity
        threshold->int, how many continues points considered
        '''
        df=deepcopy(data)
        #### continues constant array's std must be zero
        df['constant_std'] = df[col].rolling(window=threshold).std()
        df['constant_mu'] = df[col].rolling(window=threshold).mean()
        df['index']=range(len(df)) ##data's index are date so....
        condition=(df['constant_std']==0) & ((df['constant_mu']!=0) & (df['constant_mu']!=capacity))
        constant_end_index=df[condition]['index'] ##find end index
        constant_index=[] ##find all index not wanted
        for i in constant_end_index:
            constant_index+=[j for j in range(i-threshold+1,i+1)]
        df['index']=df['index'].apply(lambda x: False if x in constant_index else True)
        df=df[df['index']==True] ##remain index wanted
        del df['index'],df['constant_std'],df['constant_mu'] ##delete auxiliary varible
        return df

    def handle_trend(self, data, trend, freq=96):
        '''
        quantile=0.8 means remain top 80% data
        '''
        df=deepcopy(data)
        col1,col2,quantile=trend[0],trend[1],trend[2]
        # use MAE method to delete 2 feas with different trend
        df['col1'] = (df[col1] - df[col1].min()) / \
            (df[col1].max() - df[col1].min())  # max-min-normlize
        df['col2'] = (df[col2] - df[col2].min()) / \
            (df[col2].max() - df[col2].min())
        df_list = [(df.iloc[i * freq:(i + 1) * freq]['col1'], df.iloc[i *
                    freq:(i + 1) * freq]['col2']) for i in range(int(len(df) / freq))]
        res = pd.Series(map(lambda x: np.mean(abs(x[0] - x[1])) * 1000, df_list))
        res=res.fillna(np.float('inf'))  ##asign inf for NAN value
        filted_res = res[res <= np.quantile(res, float(quantile))].index.tolist() 
        df_list = [df.iloc[i * freq:(i + 1) * freq] for i in filted_res]
        df = pd.concat(df_list, axis=0)
        del df['col1'], df['col2']
        return df

    def download_suntime(self,area,belong,Longitude,Latitude,begin,end):
        """
        根据输入的经度、维度，下载从起始到结束之间的每天日出日落时间
        该函数使用的时候需要联网环境
        Parameters
        ----------
        area : str
            目标区域名称.
        belong : str
            目标区域所属区域/国家名称.
        longitude : float
            经度.
        latitude : float
            纬度.
        begin : str
            起始时间 “2020-1-1”
        end : str
            结束时间 “2023-1-1”.
    
        Returns
        -------
        None.
    
        """
    
        #example:    
        # city = LocationInfo("Nanjing", "China", "China/Nanjing", latitude, longitude)#纬度，经度
        city = self.LocationInfo(area, belong, f"{belong}/{area}", Latitude, Longitude)#纬度，经度
        logging.info((
            f"Information for {city.name}/{city.region}\n"
            f"Timezone: {city.timezone}\n"
            f"Latitude: {city.latitude:.02f}; Longitude: {city.longitude:.02f}\n"
        ))
        
        #将输入的起始和结束时间转换为指定的date格式
        begin_list = begin.split("-")
        end_list = end.split("-")
        begin = date(int(begin_list[0]),int(begin_list[1]),int(begin_list[2]))
        end = date(int(end_list[0]),int(end_list[1]),int(end_list[2]))
    
        #根据起始和结束时间，遍历的获取每日的日出、日落时间    
        date_list = []
        sunrise_list = []
        sunset_list = []
        for i in range((end - begin).days + 1):
            current_date = begin + timedelta(days=i)
            current_date_str = current_date.strftime('%Y-%m-%d')
            s = self.sun(city.observer, date=current_date)
            
            #modify UTC timezone into BeiJing timezone
            sunrise_list.append(s["sunrise"]+timedelta(hours=8))
            sunset_list.append(s["sunset"]+timedelta(hours=8))
            date_list.append(current_date_str)
            
        data = {"Date":date_list,"Sunrise":sunrise_list,"Sunset":sunset_list}
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] =  df["Date"].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
        df["Sunrise"] =  df["Sunrise"].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
        df["Sunset"] =  df["Sunset"].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
    
        return df    

    def sunset_zero(self,data,col):
        df=deepcopy(data)
        sunset_point=pd.DataFrame(sorted(pd.concat([self.suntime['sunrise_before'],self.suntime['sunset_after']]).drop_duplicates()))
        sunset_point.columns=['sunset_point']
        sunset_point.index=pd.to_datetime(sunset_point['sunset_point']) ##change index into datetime
        df=df.join(sunset_point)
        sunset_list=list(df[~df['sunset_point'].isna()]['sunset_point']) ##sunset Separation point list
        res=[] ##full sunset point list
        for idx,time in enumerate(sunset_list):
            if idx==0 and time.hour<12: ##create first point sunset points 
                res+=list(pd.date_range(time-pd.Timedelta(8,unit='H'),time,freq='15T'))
            elif idx==len(sunset_list)-1 and time.hour>12: ##create last point sunset points 
                res+=list(pd.date_range(time,time+pd.Timedelta(8,unit='H'),freq='15T'))
            elif time.hour>12:  ##create middle sunset points 
                res+=list(pd.date_range(sunset_list[idx],sunset_list[idx+1],freq='15T'))
        df['date']=df.index
        df['sunset_flag']=df['date'].apply(lambda x:True if x in res else False)
        df[col][df['sunset_flag']==True]=0 ##sunset points set to zero
        del df['sunset_flag'],df['date'],df['sunset_point']
        return df

