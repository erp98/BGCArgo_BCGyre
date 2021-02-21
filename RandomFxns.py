#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:10:59 2021

@author: Ellen
"""
import numpy as np
import pandas as pd
import xarray as xr

def DetermineAdjusted(raw_data, raw_data_QC,a_data, a_data_QC):
    # raw_data = not adjusted BGC Argo data
    # a_data = adjuasted BGC Argo data
    search_flag=0
    j=0
    adj_flag=np.NaN
    
    raw_ss=raw_data[0,:]
    a_ss=a_data[0,:]
    
    while (j < len(raw_ss) and search_flag == 0):
        
        if (np.isnan(raw_ss[j]) != True and np.isnan(a_ss[j]) != True):
            search_flag=1
            adj_flag=1
        elif (np.isnan(raw_ss[j]) != True and np.isnan(a_ss[j]) == True):
            search_flag=1
            adj_flag=0
        elif (np.isnan(raw_ss[j]) == True and np.isnan(a_ss[j]) == True):
            j=j+1
    
    if adj_flag == 1:
        output_data = a_data
        output_data_QC = a_data_QC
    elif adj_flag == 0:
        output_data = raw_data
        output_data_QC= raw_data_QC
    elif np.isnan(adj_flag) == True:
        print('ERROR: Using raw data')
        output_data=raw_data
        output_data_QC=raw_data_QC
        
    return output_data, output_data_QC, adj_flag

def last_day_of_month(given_date):
    
    if given_date.month == 1:
        # January
        last_day = 31
    elif (given_date.month == 2 and given_date.year%4 != 0):
        # February - nonleap year
        last_day = 28
    elif (given_date.month == 2 and given_date.year%4 == 0):
        # February - leap year
        last_day = 29
    elif given_date.month == 3:
        # March
        last_day = 31
    elif given_date.month == 4:
        # April
        last_day = 30
    elif given_date.month == 5:
        # May
        last_day = 31
    elif given_date.month == 6:
        # June
        last_day = 30
    elif given_date.month == 7:
        # July
        last_day = 31
    elif given_date.month == 8:
        # August
        last_day = 31
    elif given_date.month == 9:
        # September 
        last_day = 30
    elif given_date.month == 10:
        # October
        last_day = 31
    elif given_date.month == 11:
        # November
        last_day = 30
    elif given_date.month == 12:
        # December
        last_day = 31
    
    return last_day

def PositionCheck(RegionOI, LatData, LonData, TempData, SalData, OxyData):
    # RegionOI = [LatN, LatS, LatE, LatW]
    LatNMax=RegionOI[0]
    LatSMin=RegionOI[1]
    LonEMax=RegionOI[2]
    LonWMin=RegionOI[3]
    
    #var_nans=[]
    # Make a pandas dataframe of all the data
    all_df=pd.DataFrame({'Lat': LatData, 'Lon': LonData, 'Temp': TempData, 'Sal': SalData,'Oxy': OxyData})
    all_df.loc[all_df.loc[:,'Lat']>LatNMax,:]=np.NaN
    all_df.loc[all_df.loc[:,'Lat']<LatSMin,:]=np.NaN
    all_df.loc[all_df.loc[:,'Lon']>LonEMax,:]=np.NaN
    all_df.loc[all_df.loc[:,'Lon']<LonWMin,:]=np.NaN
    
    # Turn data back to arrays
    LatData_C = all_df.loc[:,'Lat'].to_numpy()
    LonData_C = all_df.loc[:,'Lon'].to_numpy()
    TempData_C = all_df.loc[:,'Temp'].to_numpy()
    SalData_C = all_df.loc[:,'Sal'].to_numpy()
    OxyData_C = all_df.loc[:,'Oxy'].to_numpy()
    
    return LatData_C, LonData_C, TempData_C, SalData_C, OxyData_C

def ArgoQC(Data, Data_QC, goodQC_flags):
    
    AllQCLevels=[b'1',b'2',b'3',b'4',b'5',b'6',b'7',b'8',b'9']
    AllQCLevels_i=[1,2,3,4,5,6,7,8,9]
    QCData=np.zeros((Data.shape))
    
    for i in goodQC_flags:
        #print(i)
        AllQCLevels_i.remove(i)
    
    if len(Data.shape)==2:
        for i in np.arange(Data.shape[0]):
            
            t_df=pd.DataFrame({'Data':Data[i,:],'QC':Data_QC[i,:]})
            #print(t_df)
            for j in AllQCLevels_i:
                qc_l=AllQCLevels[j-1]
                t_df.loc[t_df.loc[:,'QC']==qc_l,'Data']=np.NaN
            
            t_df = t_df.loc[:,'Data'].to_numpy()
            QCData[i,:]=t_df
    
    return QCData

def ArgoDataLoader(WMO, DAC):
    BGCfile='/Users/Ellen/Desktop/ArgoGDAC/dac/'+DAC+'/'+str(WMO)+'/'+str(WMO)+'_Sprof.nc'
    Data = xr.open_dataset(BGCfile)
    
    return Data

def SurfacePIndex (pres, minP, maxP):
    
    j=0
    pressure_check=0
    pres_ind=[]
    prev_val=0
    
    while j < len(pres) and pressure_check == 0:
        
        # Determine if pressure value falls in range 
        if pres[j] >= minP and pres[j] <= maxP:
            pres_ind=pres_ind+[j]
            current_val=1
        else:
            current_val=0
            
        if current_val == 0 and prev_val == 1:
            pressure_check=1
            
        if current_val == 1:
            prev_val=1
        
        j=j+1
    
    return pres_ind

def MatchData2Dates(alldates, dates, L13, L13_24hr, L13_1wk, N16, N16_24hr, N16_1wk):
    
    L13_reform=np.zeros(len(alldates))
    L13_reform[:]=np.NaN
    L13_24hr_reform=np.zeros(len(alldates))
    L13_24hr_reform[:]=np.NaN
    L13_1wk_reform=np.zeros(len(alldates))
    L13_1wk_reform[:]=np.NaN
    
    N16_reform=np.zeros(len(alldates))
    N16_reform[:]=np.NaN
    N16_24hr_reform=np.zeros(len(alldates))
    N16_24hr_reform[:]=np.NaN
    N16_1wk_reform=np.zeros(len(alldates))
    N16_1wk_reform[:]=np.NaN
        
    for i in np.arange(len(dates)):
        date_ind=np.where(alldates == dates[i])
        #print(i)
        L13_reform[date_ind]=L13[i]
        L13_24hr_reform[date_ind]=L13_24hr[i]
        L13_1wk_reform[date_ind]=L13_1wk[i]
        
        N16_reform[date_ind]=N16[i]
        N16_24hr_reform[date_ind]=N16_24hr[i]
        N16_1wk_reform[date_ind]=N16_1wk[i]
        
        a=1
    return L13_reform,L13_24hr_reform,L13_1wk_reform,N16_reform,N16_24hr_reform,N16_1wk_reform
    

