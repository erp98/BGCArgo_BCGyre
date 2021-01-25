#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:51:54 2021

@author: Ellen
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from RandomFxns import FindInd

#####################
## Some Parameters ##
#####################

# N Atlantic Region
lat_N=80.000
lat_S= 40.00
lon_E= 0.00
lon_W= -80.00

# Labrador Sea Region
lab_N=65
lab_S=48
lab_E=-45
lab_W=-80

fsize_x=10
fsize_y=6

###################
data_types=[0,1]
for data_i in data_types:
    if data_i == 0:
        # Load BC float data
        print('\n%% Loading BC Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BCFloats_'
        FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries/BC_'
    elif data_i == 1:
        # Load Lab Gyre float data
        print('\n%% Loading Lab Sea Gyre Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
        FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries/LabSea_'
    
    FileList=glob.glob(CSVDir_AS+'*.csv')
    df_count = 0
    
    file_count=0
    
    for filedir in FileList:
        
        # Read in each data file
        fluxdata = pd.read_csv(filedir)

        # And combine into one big data frame
        if df_count == 0:
            AllData = pd.read_csv(filedir)
            df_count = 1
        else:
            AllData=AllData.append(fluxdata)
        
        file_count=file_count+1
    
    # If BC floats, crop data so it is in Lab Sea
    
    if data_i == 0:
        AllData.loc[AllData.loc[:,'Lat']>lab_N,:]=np.NaN
        AllData.loc[AllData.loc[:,'Lat']<lab_S,:]=np.NaN
        AllData.loc[AllData.loc[:,'Lon']>lab_E,:]=np.NaN
        AllData.loc[AllData.loc[:,'Lon']<lab_W,:]=np.NaN
        AllData=AllData.dropna()

    # Plot float trajectories
    float_wmo= AllData.loc[:,'WMO'].unique().tolist()
    wmo_count =0
    print('\n%% Plotting Trajectories %%\n')
    for wmo in float_wmo:
        lat=AllData.loc[AllData.loc[:,'WMO']==wmo,'Lat'].to_numpy()
        lon=AllData.loc[AllData.loc[:,'WMO']==wmo,'Lon'].to_numpy()
        
        plt.figure(1,figsize=(fsize_x,fsize_y))
        NA = plt.axes(projection=ct.crs.PlateCarree())
        NA.set_extent([lon_E, lon_W, lat_S, lat_N])
        lonval=-1*np.arange(-lon_E,-lon_W+1,10)
        latval=np.arange(lat_S,lat_N+1,10)
        NA.set_xticks(lonval, crs=ct.crs.PlateCarree())
        NA.set_yticks(latval, crs=ct.crs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        NA.add_feature(ct.feature.COASTLINE)
        NA.add_feature(ct.feature.OCEAN)
        NA.xaxis.set_major_formatter(lon_formatter)
        NA.yaxis.set_major_formatter(lat_formatter)
        #plt.title('Trajectory for Float '+str(WMO))
        plt.plot(lon[0],lat[0],'go')
        plt.plot(lon[len(lon)-1],lat[len(lat)-1],'ro')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.plot(lon, lat)
                
        if wmo_count == len(float_wmo)-1:
            plt.savefig(FigDir+'Trajectory.jpg')
            plt.clf(); plt.close()
        
        wmo_count=wmo_count+1
    
    # Calculate the mean at given time in specific region 
    MeanData=AllData.groupby(by='Date').mean()
    StdData=AllData.groupby(by='Date').std()
    
    date_list=MeanData.index.values.tolist()
    num_ticks=20
    xticks_ind=np.arange(0,(len(date_list)//num_ticks*num_ticks)+1, step=len(date_list)//num_ticks)
    xticks_labels=[]
    for i in xticks_ind:
        xticks_labels=xticks_labels+[date_list[i]]
    
    print('\n%% Plotting Gas Flux Time Series %%\n')
    
    ## Raw Data ##
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(date_list, MeanData.loc[:,'Ft_N16'])
    #plt.fill_between(date_list, MeanData.loc[:,'Ft_N16']+StdData.loc[:,'Ft_N16'],MeanData.loc[:,'Ft_N16']-StdData.loc[:,'Ft_N16'])
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (N16) - Raw')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'N16_Raw.jpg')
    plt.close()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(date_list, MeanData.loc[:,'Ft_L13'])
    #plt.fill_between(date_list, MeanData.loc[:,'Ft_L13']+StdData.loc[:,'Ft_L13'],MeanData.loc[:,'Ft_L13']-StdData.loc[:,'Ft_L13'])
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (L13) - Raw')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'L13_Raw.jpg')
    plt.close()
    
    ## Moving Average: 24 hrs ##
    Ft_N16_24hr=MeanData.loc[:,'Fd_N16'].rolling(int((24/6)),min_periods=1).mean()+MeanData.loc[:,'Fp_N16'].rolling(int((24/6)),min_periods=1).mean()+MeanData.loc[:,'Fc_N16'].rolling(int((24/6)),min_periods=1).mean()
    Ft_L13_24hr=MeanData.loc[:,'Fd_L13'].rolling(int((24/6)),min_periods=1).mean()+MeanData.loc[:,'Fp_L13'].rolling(int((24/6)),min_periods=1).mean()+MeanData.loc[:,'Fc_L13'].rolling(int((24/6)),min_periods=1).mean()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(date_list, Ft_N16_24hr)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (N16) - 24 hr Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'N16_24hr.jpg')
    plt.close()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(date_list,Ft_L13_24hr)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (L13) - 24 hr Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'L13_24hr.jpg')
    plt.close()
    
    ## Moving Average: 1 week ##
    Ft_N16_1wk=MeanData.loc[:,'Fd_N16'].rolling(int((7*24)/6),min_periods=1).mean()+MeanData.loc[:,'Fp_N16'].rolling(int((7*24)/6),min_periods=1).mean()+MeanData.loc[:,'Fc_N16'].rolling(int((7*24)/6),min_periods=1).mean()
    Ft_L13_1wk=MeanData.loc[:,'Fd_L13'].rolling(int((7*24)/6),min_periods=1).mean()+MeanData.loc[:,'Fp_L13'].rolling(int((7*24)/6),min_periods=1).mean()+MeanData.loc[:,'Fc_L13'].rolling(int((7*24)/6),min_periods=1).mean()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(date_list, Ft_N16_1wk)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (N16) - 1 week Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'N16_1wk.jpg')
    plt.close()
    
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(date_list, Ft_L13_1wk)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (L13) - 1 week Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'L13_1wk.jpg')
    plt.close()
    
    if data_i == 0:
        dates_BC=date_list
        data_BC=MeanData.loc[:,'Ft_N16']
    elif data_i ==1:
        dates_LSG=date_list
        data_LSG=MeanData.loc[:,'Ft_N16']

# Make complete date range for data
min_flag = np.NaN
# min_flag = 0; min date is LSG
# min_flag = 1; min date is BC

if dates_LSG[0]<dates_BC[0]:
    MinDate=dates_LSG[0]
    min_flag =0
else:
    MinDate=dates_BC[0]
    min_flag=1

if dates_LSG[-1]>dates_BC[-1]:
    MaxDate=dates_LSG[-1]
else:
    MaxDate=dates_BC[-1]
    
dates_total=pd.date_range(MinDate,MaxDate,freq='6H')
BC_N16=np.zeros(len(dates_total))
BC_N16[:]=np.NaN
LSG_N16=np.zeros(len(dates_total))
LSG_N16[:]=np.NaN

if min_flag == 0:
    LSG_N16[0:len(data_LSG)]=data_LSG
    startind=FindInd(ThingToSearch=dates_total, ThingLookingFor=dates_BC[0])
    endind=startind+len(dates_BC)
    BC_N16[startind:endind]=data_BC
    # Check that ends match
    if (str(dates_total[len(data_LSG)-1])==dates_LSG[-1] and str(dates_total[endind])==dates_BC[-1]):
        print('Everything Matches')
    else:
        print('ERROR!')
    
elif min_flag ==1:
    BC_N16[0:len(data_BC)]=data_BC
    startind=FindInd(ThingToSearch=dates_total, ThingLookingFor=dates_LSG[0])
    endind=startind+len(dates_LSG)
    LSG_N16[startind:endind]=data_LSG
     
    # Check that ends match
    if (dates_total[len(data_BC)-1]==dates_BC[-1] and dates_total[endind]==dates_LSG[-1]):
        print('Everything Matches')
    else:
        print('ERROR!')
        
# plt.figure(figsize=(fsize_x,fsize_y))
# plt.scatter(dates_LSG,data_LSG, s= 1)
# plt.scatter(dates_BC,data_BC, s=1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
# plt.subplots_adjust(bottom=0.3)
# plt.savefig('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries/CompareFlux.jpg')





