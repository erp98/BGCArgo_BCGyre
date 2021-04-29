#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:30:08 2021

@author: Ellen
"""

import glob
import pandas as pd
import numpy as np

InitialDataDir='/Users/Ellen/Desktop/GOSHIP_Data/NAtlantic_Sorted/Reformat/'
OutDir='/Users/Ellen/Desktop/GOSHIP_Data/NAtlantic_Sorted/'
# Load in data
DataFileList=glob.glob(InitialDataDir+'*.csv')

VariablesOfInterest=['OXYGEN']#,'NITRAT','PHSPHT','SILICAT','NO2+NO3']
minvar=1
# Questions: Do we want pH?

# AllData=pd.DataFrame({'EXPOCODE':[np.NaN], 'DATE':[np.NaN],'LATITUDE':[np.NaN],'LONGITUDE':[np.NaN],
#                       'PRES':[np.NaN],'TEMP':[np.NaN],'SAL':[np.NaN],'OXY':[np.NaN],'NITR':[np.NaN],'PHSP':[np.NaN],
#                       'SILI':[np.NaN]})

AllData=pd.DataFrame({'EXPOCODE':[np.NaN], 'DATE':[np.NaN],'LATITUDE':[np.NaN],'LONGITUDE':[np.NaN],
                      'PRES':[np.NaN],'TEMP':[np.NaN],'SAL':[np.NaN],'OXY':[np.NaN]})

BadQC=[1.0,3.0,4.0,5.0,7.0,8.0,9.0]

for GOFile in DataFileList:
    
    # Read in data file
    Data=pd.read_csv(GOFile, engine='python')
    
    Variables=Data.columns.to_list()
    
    # Determine if this cruise measures ALL variables of interest
    var_count=[0,0,0,0]
    
    n_flag=[0,0]
    
    s_flag=[0,0]
    for var in Variables:
        #print(var)
        if var == 'OXYGEN':
            var_count[0]=1
        # elif var == 'NITRAT' or var == 'NO2+NO3':
        #     var_count[1]=1
        #     if var == 'NITRAT':
        #         n_flag[0]=1
        #     elif var == 'NO2+NO3':
        #         n_flag[1]=1
        # elif var=='SILCAT':
        #     var_count[2]=1
        # elif var=='PHSPHT':
        #     var_count[3]=1
            
        if var == 'CTDSAL':
            s_flag[0]=1
        elif var == 'SALNTY':
            s_flag[1]=1
    
    #print(var_count)
    if np.sum(np.array(var_count)) == minvar:
        # Save and manage this dataset
        # Only save varaible of interest and quality control
        
        # Use bottle data except for pres, sal, and temp
        # Note: not every cruise uses bottle sal so use CTD sal for 
        # Consistency
        
        for num in BadQC:
            # Salinity
            if s_flag[0]==1:
                for var in Variables:
                    if var == 'CTDSAL_FLAG_W':
                        Data.loc[Data['CTDSAL_FLAG_W']==num,'CTDSAL_FLAG_W']=np.NaN
            elif s_flag[1]==1:
                for var in Variables:
                    if var == 'SALNTY_FLAG_W':
                        Data.loc[Data['SALNTY_FLAG_W']==num,'SALNTY_FLAG_W']=np.NaN
            
            # Oxygen 
            Data.loc[Data['OXYGEN_FLAG_W']==num,'OXYGEN_FLAG_W']=np.NaN
            
            # Nitrate
            # Use just nitrate if provided
            # if n_flag[0]==1:
            #     Data.loc[Data['NITRAT_FLAG_W']==num,'NITRAT_FLAG_W']=np.NaN
            # elif n_flag[1]==1:
            #     Data.loc[Data['NO2+NO3_FLAG_W']==num,'NO2+NO3_FLAG_W']=np.NaN
            
            # # Silicate
            # Data.loc[Data['SILCAT_FLAG_W']==num,'SILCAT_FLAG_W']=np.NaN
            
            # # Phosphate
            # Data.loc[Data['PHSPHT_FLAG_W']==num,'PHSPHT_FLAG_W']=np.NaN
        
        # Drop bad data
        Data=Data.dropna()

        # Save good data
        exp=Data.loc[:,'EXPOCODE']
        date=Data.loc[:,'DATE']
        lat=Data.loc[:,'LATITUDE']
        lon=Data.loc[:,'LONGITUDE']
        pres=Data.loc[:,'CTDPRS']
        temp=Data.loc[:,'CTDTMP']
        oxy=Data.loc[:,'OXYGEN']
        # phs=Data.loc[:,'PHSPHT']
        # sil=Data.loc[:,'SILCAT']
        
        if s_flag[0]==1:
            sal=Data.loc[:,'CTDSAL']
        elif s_flag[1]==1:
            sal=Data.loc[:,'SALNTY']
            
        # if n_flag[0]==1:
        #     nit=Data.loc[:,'NITRAT']
        # elif n_flag[1]==1:
        #     nit=Data.loc[:,'NO2+NO3']
        
        # df_temp=pd.DataFrame({'EXPOCODE':exp, 'DATE':date,'LATITUDE':lat,'LONGITUDE':lon,
        #               'PRES':pres,'TEMP':temp,'SAL':sal,'OXY':oxy,'NITR':nit,'PHSP':phs,
        #               'SILI':sil})
        
        df_temp=pd.DataFrame({'EXPOCODE':exp, 'DATE':date,'LATITUDE':lat,'LONGITUDE':lon,
                      'PRES':pres,'TEMP':temp,'SAL':sal,'OXY':oxy})
        
        AllData=AllData.append(df_temp)
        
AllData=AllData.iloc[1:,:]   

## Reformat dates
new_dates=[[]]*AllData.shape[0]
old_dates=AllData.loc[:,'DATE'].to_numpy()
months=[[]]*AllData.shape[0]
for i in np.arange(len(new_dates)):
    t=str(int(old_dates[i]))
    new_dates[i]=t[0:4]+'-'+t[4:6]+'-'+t[6:]
    months[i]=int(t[4:6])

AllData['DATE']=new_dates
AllData['MONTH']=months

AllData.to_csv(OutDir+'QCFilteredData.csv')     
        
        