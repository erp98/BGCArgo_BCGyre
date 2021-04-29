#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:10:59 2021

@author: Ellen
"""
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib.pyplot as plt
from scipy import interpolate 

def DetermineAdjusted(raw_data, raw_data_QC,a_data, a_data_QC):
    # raw_data = not adjusted BGC Argo data
    # a_data = adjuasted BGC Argo data
    # Note! Update this function copying coreargo_fxns
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


def MatchData2Dates(alldates, dates, Data):   
    
    blankdata=np.zeros((len(alldates), Data.shape[1]))
    blankdata[:]=np.NaN
    
    ReformattedData=pd.DataFrame(blankdata,index=alldates,columns=Data.columns.to_list())
    
    # L13_reform=np.zeros(len(alldates))
    # L13_reform[:]=np.NaN
    # L13_24hr_reform=np.zeros(len(alldates))
    # L13_24hr_reform[:]=np.NaN
    # L13_1wk_reform=np.zeros(len(alldates))
    # L13_1wk_reform[:]=np.NaN
    
    # N16_reform=np.zeros(len(alldates))
    # N16_reform[:]=np.NaN
    # N16_24hr_reform=np.zeros(len(alldates))
    # N16_24hr_reform[:]=np.NaN
    # N16_1wk_reform=np.zeros(len(alldates))
    # N16_1wk_reform[:]=np.NaN
        
    for i in np.arange(len(dates)):
        #print(i)
        date_ind=np.where(alldates == dates[i])
        #print(date_ind)
        a=1
        # L13_reform[date_ind]=L13[i]
        # L13_24hr_reform[date_ind]=L13_24hr[i]
        # L13_1wk_reform[date_ind]=L13_1wk[i]
        
        # N16_reform[date_ind]=N16[i]
        # N16_24hr_reform[date_ind]=N16_24hr[i]
        # N16_1wk_reform[date_ind]=N16_1wk[i]
        #print(Data.iloc[i,:])
        #print(ReformattedData.iloc[date_ind[0],:])
        
        ReformattedData.iloc[date_ind[0][0],:]=Data.iloc[i,:]
    
    return ReformattedData

def MLD(Pres, Temp, Sal, Lat, Lon):

    
    MinSurfP=0
    MaxSurfP=10
     
    j=0
    surf_flag=0
    surf_pres_i=[]
    
    dense_offset=.03
    while (j<len(Pres) and surf_flag ==0):
        if (Pres[j]>= MinSurfP and Pres[j] <= MaxSurfP):
            surf_pres_i=surf_pres_i+[j]
        
        if Pres[j] > MaxSurfP:
            surf_flag=1
        
        j=j+1
    
    if surf_pres_i != []:
        
        #print(surf_pres_i)
        s_start=surf_pres_i[0]
        s_end=surf_pres_i[-1]+1
        #print(s_start, s_end)
        P_mean=np.nanmean(Pres[s_start:s_end])
        
        # Calculate density
        SA=gsw.SA_from_SP(Sal,Pres,Lat,Lon)
        #print(SA)
        CT=gsw.CT_from_t(SA,Temp,P_mean)
        #print(CT)
        density = np.zeros(len(Pres))
        density[:]=np.NaN
        
        for k in np.arange(len(density)):
            density[k]=gsw.density.sigma0(SA=SA[k],CT=CT[k])

        surf_dense=np.nanmean(density[s_start:s_end])
        #print(surf_dense)
        rho_mld=surf_dense+dense_offset
        
        #print(mld_dense)
        # Find MLD
        mld_flag=0
        inter_flag=np.NaN
        mld_pres=np.NaN
        
        P_L=np.NaN
        rho_L=np.NaN
        P_D=np.NaN
        rho_D=np.NaN
        
        j = s_end -1
        # Start search at base of surface layer
        while(j<len(density) and mld_flag ==0):
        
            if density[j] == rho_mld:
                mld_pres=Pres[j]
                mld_flag=1
            elif density[j]<rho_mld:
                P_L=Pres[j]
                rho_L=density[j]
            elif density[j]>rho_mld:
                P_D=Pres[j]
                rho_D=density[j]
                inter_flag=1
                mld_flag=1
            
            j=j+1
        
        if (np.isnan(mld_flag) == True):
            # Ran through entire profile and did not find mld
            # i.e. MLD is deeper than deepest pressure measurement
            # So make md deepest pressure measurement
            # OR 2000 dbar?
            mld_pres=np.nanmax(Pres)
        elif (np.isnan(mld_flag)== False and inter_flag == 1):
            # Need to interpolate to get MLD
            #mld_pres=M_D-(((M_D-mld_dense)/(M_D-L_D))*(M_P-L_P))
            mld_pres = P_D-(P_D - P_L)*((rho_D-rho_mld)/(rho_D-rho_L))
        elif (np.isnan(mld_flag)== False and inter_flag == 0):
            mld_pres=mld_pres
        else:
            print('ERROR')
        
        # plt.figure()
        # plt.plot(density, Pres)
        # plt.gca().invert_yaxis()
        # plt.show()
        
    else:
        mld_pres=np.NaN
        #mld_dense=np.NaN
    
    return mld_pres#, mld_dense
        
        
def PresInterpolation(OriginalData, OriginalPressure, NewPressure, Pres_StepSize, NewData):
    
    for l in np.arange(NewData.shape[0]):
    
        if np.sum(np.isnan(OriginalData[l])) != len(OriginalData[l]):
            
            # Determine min and max values of pres range to use
            min_ind=np.NaN
            max_ind=np.NaN
            #print(np.nanmin(OriginalPressure[l]))
            if np.isnan(np.nanmin(OriginalPressure[l]))==False:
                if np.nanmin(OriginalPressure[l])%Pres_StepSize==0:
                    min_ind=int(np.nanmin(OriginalPressure[l])//Pres_StepSize)
                else:
                    min_ind=int(np.nanmin(OriginalPressure[l])//Pres_StepSize+1)
            
            if np.isnan(np.nanmax(OriginalPressure[l]))==False:
                max_ind=int(np.nanmax(OriginalPressure[l])//Pres_StepSize)+1
            
            if np.isnan(min_ind) == False and np.isnan(max_ind)==False:
                if max_ind > min_ind:
                    T_depth_interp=interpolate.interp1d(OriginalPressure[l], OriginalData[l])
                    NewData[l,min_ind:max_ind]=T_depth_interp(NewPressure[min_ind:max_ind])
                else:
                    # There is only one data point 
                    min_ind=int(np.round(np.nanmin(OriginalPressure[l]))//Pres_StepSize)
                    NewData[l,min_ind]=OriginalData[l,np.argwhere(np.isnan(OriginalData[l])==False)[0][0]]
                    
    return NewData

def TimeInterpolatation(PresInterpData, PresInterpTime, NewTime, NewData):
    
    for l in np.arange(NewData.shape[0]):
        # For each pressure level interp with time
        
        ## Temperature ##
        T_p_level=PresInterpData[:,l].T
        if np.sum(np.isnan(T_p_level)) != len(T_p_level):
            T_time_interp=interpolate.interp1d(PresInterpTime, T_p_level)
            NewData[l,:]=T_time_interp(NewTime)
    
    return NewData

def PresInterpolation1m(OriginalData, OriginalPressure, NewPressure, Pres_StepSize, NewData):
    
    for l in np.arange(NewData.shape[0]):
    
        if np.sum(np.isnan(OriginalData[l])) != len(OriginalData[l]):
            
            T_depth_interp=interpolate.interp1d(OriginalPressure[l], OriginalData[l],fill_value='extrapolate')
            NewData[l,:]=T_depth_interp(NewPressure)
            
            # if np.isnan(min_ind) == False and np.isnan(max_ind)==False:
            #     if max_ind > min_ind:
            #         T_depth_interp=interpolate.interp1d(OriginalPressure[l], OriginalData[l])
            #         NewData[l,min_ind:max_ind]=T_depth_interp(NewPressure[min_ind:max_ind])
            #     else:
            #         # There is only one data point 
            #         min_ind=int(np.round(np.nanmin(OriginalPressure[l]))//Pres_StepSize)
            #         NewData[l,min_ind]=OriginalData[l,np.argwhere(np.isnan(OriginalData[l])==False)[0][0]]
                    
    return NewData


        
def WMODacPair():
    PairFile = '/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/DacWMO_NAtlantic.txt'
    dacs = []
    wmos = []
    
    count = 0
    with open(PairFile) as fp:
        Lines = fp.readlines()
        for line in Lines:
            count += 1
            x=line.strip()
            xs = x.split('/')
            dacs=dacs+[xs[0]]
            wmos = wmos + [xs[1]]
    
    Dict = {}
    for i in np.arange(len(dacs)):
        Dict[int(wmos[i])]=dacs[i]
    
    return Dict
        

