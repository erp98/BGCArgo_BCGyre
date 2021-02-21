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
        title_add='LS-BC Floats'
    elif data_i == 1:
        # Load Lab Gyre float data
        print('\n%% Loading Lab Sea Gyre Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
        FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries/LabSea_'
        title_add='LS-G Floats'
        
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
    
    print('\nThere are '+str(len(float_wmo))+' floats\n')
    # Calculate the mean at given time in specific region 
    MeanData=AllData.groupby(by='Date').mean()
    StdData=AllData.groupby(by='Date').std()
    
    date_list=MeanData.index.values.tolist()
    FirstDate=date_list[0]
    LastDate=date_list[-1]
    
    # Get full range of dates in between first and last profile date
    AllDates_df=pd.date_range(str(FirstDate),str(LastDate),freq='6H')
    AllDates=[[]]*len(AllDates_df)
    for i in np.arange(len(AllDates)):
        AllDates[i]=str(AllDates_df[i])
    AllDates=np.array(AllDates)
    
    ## N16 ##
    Ft_N16=np.zeros(len(AllDates_df))
    Ft_N16[:]=np.NaN
    Fd_N16=np.zeros(len(AllDates_df))
    Fd_N16[:]=np.NaN
    Fp_N16=np.zeros(len(AllDates_df))
    Fp_N16[:]=np.NaN
    Fc_N16=np.zeros(len(AllDates_df))
    Fc_N16[:]=np.NaN
    
    ## L13 ##
    Ft_L13=np.zeros(len(AllDates_df))
    Ft_L13[:]=np.NaN
    Fd_L13=np.zeros(len(AllDates_df))
    Fd_L13[:]=np.NaN
    Fp_L13=np.zeros(len(AllDates_df))
    Fp_L13[:]=np.NaN
    Fc_L13=np.zeros(len(AllDates_df))
    Fc_L13[:]=np.NaN  
    
    # Go through dates and match data to correct time period
    for j in np.arange(len(date_list)):
        date_ind=np.where(AllDates == date_list[j])
        
        # Mean Data column values
        # 1: WMO, 2: Lat, 3: Lon, 
        # 4: Fp_l13, 5:Fc_L13, 6: Fd_L13, 7 Ft_L13
        # 8: Fp_N16, 9: Fc_N16, 10: Fd_N16, 11 Ft_N16
        Ft_N16[date_ind]=MeanData.iloc[j,11]
        Fd_N16[date_ind]=MeanData.iloc[j,10]
        Fp_N16[date_ind]=MeanData.iloc[j,8]
        Fc_N16[date_ind]=MeanData.iloc[j,9]
    
        Ft_L13[date_ind]=MeanData.iloc[j,7]
        Fd_L13[date_ind]=MeanData.iloc[j,6]
        Fp_L13[date_ind]=MeanData.iloc[j,4]
        Fc_L13[date_ind]=MeanData.iloc[j,5]
    
    num_ticks=20
    xticks_ind=np.arange(0,(len(AllDates)//num_ticks*num_ticks)+1, step=len(AllDates)//num_ticks)
    xticks_labels=[]
    for i in xticks_ind:
        xticks_labels=xticks_labels+[AllDates[i]]
    
    print('\n%% Plotting Gas Flux Time Series %%\n')
    
    ## Raw Data ##
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(AllDates, Ft_N16)
    #plt.scatter(AllDates, Ft_N16, s=1)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (N16) - '+title_add+' - Raw')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'N16_Raw.jpg')
    plt.close()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(AllDates, Ft_L13)
    #plt.scatter(AllDates, Ft_L13, s=1)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (L13) - '+title_add+' - Raw')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'L13_Raw.jpg')
    plt.close()
    
    ## Moving Average: 24 hrs ##
    Ft_N16_24hr=pd.Series(Fd_N16).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fp_N16).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fc_N16).rolling(int((24/6)),min_periods=1).mean()
    Ft_L13_24hr=pd.Series(Fd_L13).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fp_L13).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fc_L13).rolling(int((24/6)),min_periods=1).mean()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(AllDates, Ft_N16_24hr)
    #plt.scatter(AllDates, Ft_N16_24hr,s=1)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (N16) - '+title_add+' - 24 hr Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'N16_24hr.jpg')
    plt.close()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(AllDates,Ft_L13_24hr)
    #plt.scatter(AllDates,Ft_L13_24hr,s=1)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (L13) - '+title_add+' - 24 hr Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'L13_24hr.jpg')
    plt.close()
    
    ## Moving Average: 1 week ##
    Ft_N16_1wk=pd.Series(Fd_N16).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fp_N16).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fc_N16).rolling(int((7*24)/6),min_periods=1).mean()
    Ft_L13_1wk=pd.Series(Fd_L13).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fp_L13).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fc_L13).rolling(int((7*24)/6),min_periods=1).mean()
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(AllDates, Ft_N16_1wk)
    #plt.scatter(AllDates, Ft_N16_1wk,s=1)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (N16) - '+title_add+' - 1 week Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'N16_1wk.jpg')
    plt.close()
    
    
    plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(AllDates, Ft_L13_1wk)
    #plt.scatter(AllDates, Ft_L13_1wk,s=1)
    plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
    plt.xlabel('Date')
    plt.ylabel('Flux (mol/m^2-s)')
    plt.title('Total Air-Sea Flux (L13) - '+title_add+' - 1 week Moving Average')
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(FigDir+'L13_1wk.jpg')
    plt.close()
    
    temp_Data_store=pd.DataFrame({'Ft_L13': Ft_L13,'Ft_L13_24hr': Ft_L13_24hr,'Ft_L13_1wk': Ft_L13_1wk,'Ft_N16': Ft_N16,'Ft_N16_24hr': Ft_N16_24hr,'Ft_N16_1wk': Ft_N16_1wk})
    if data_i == 0:
        dates_BC=AllDates
        data_BC=temp_Data_store
    elif data_i ==1:
        dates_LSG=AllDates
        data_LSG=temp_Data_store

# Make complete date range for data
print('\n%% Comparing Gas Flux Time Series %%\n')

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
    
dates_total_df=pd.date_range(str(MinDate),str(MaxDate),freq='6H')
dates_total=[[]]*len(dates_total_df)
for i in np.arange(len(dates_total_df)):
    dates_total[i]=str(dates_total_df[i])
dates_total=np.array(dates_total)

######### BC #########
## N16 ##
BC_N16=np.zeros(len(dates_total))
BC_N16[:]=np.NaN

BC_N16_24hr=np.zeros(len(dates_total))
BC_N16_24hr[:]=np.NaN

BC_N16_1wk=np.zeros(len(dates_total))
BC_N16_1wk[:]=np.NaN

## L13 ##
BC_L13=np.zeros(len(dates_total))
BC_L13[:]=np.NaN

BC_L13_24hr=np.zeros(len(dates_total))
BC_L13_24hr[:]=np.NaN

BC_L13_1wk=np.zeros(len(dates_total))
BC_L13_1wk[:]=np.NaN

######### LSG #########
## N16 ##
LSG_N16=np.zeros(len(dates_total))
LSG_N16[:]=np.NaN

LSG_N16_24hr=np.zeros(len(dates_total))
LSG_N16_24hr[:]=np.NaN

LSG_N16_1wk=np.zeros(len(dates_total))
LSG_N16_1wk[:]=np.NaN

## L13 ##
LSG_L13=np.zeros(len(dates_total))
LSG_L13[:]=np.NaN

LSG_L13_24hr=np.zeros(len(dates_total))
LSG_L13_24hr[:]=np.NaN

LSG_L13_1wk=np.zeros(len(dates_total))
LSG_L13_1wk[:]=np.NaN

# BC floats
for i in np.arange(len(dates_BC)):
    date_ind=np.where(dates_total == dates_BC[i])
    # if dates_total[date_ind] != dates_BC[i]:
    #     print(date_ind)
    BC_L13[date_ind]=data_BC.iloc[i,0]
    BC_L13_24hr[date_ind]=data_BC.iloc[i,1]
    BC_L13_1wk[date_ind]=data_BC.iloc[i,2]
    
    BC_N16[date_ind]=data_BC.iloc[i,3]
    BC_N16_24hr[date_ind]=data_BC.iloc[i,4]
    BC_N16_1wk[date_ind]=data_BC.iloc[i,5]

# LSG floats
for i in np.arange(len(dates_LSG)):
    date_ind=np.where(dates_total == dates_LSG[i])
    # if dates_total[date_ind] != dates_LSG[i]:
    #     print(date_ind)
    LSG_L13[date_ind]=data_LSG.iloc[i,0]
    LSG_L13_24hr[date_ind]=data_LSG.iloc[i,1]
    LSG_L13_1wk[date_ind]=data_LSG.iloc[i,2]
    
    LSG_N16[date_ind]=data_LSG.iloc[i,3]
    LSG_N16_24hr[date_ind]=data_LSG.iloc[i,4]
    LSG_N16_1wk[date_ind]=data_LSG.iloc[i,5]

# Make figure labels
num_ticks=20
xticks_ind2=np.arange(0,(len(dates_total)//num_ticks*num_ticks)+1, step=len(dates_total)//num_ticks)
xticks_labels2=[]
for i in xticks_ind2:
    xticks_labels2=xticks_labels2+[dates_total[i]]

FigDir_Compare='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries/CompareFlux_'
############ L13 ###############
plt.figure(figsize=(fsize_x,fsize_y))
plt.plot(dates_total,BC_L13)
plt.plot(dates_total,LSG_L13)
# plt.scatter(dates_total,BC_N16, s=1)
# plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (L13) - Raw')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'L13_Raw.jpg')
plt.close()


plt.figure(figsize=(fsize_x,fsize_y))
plt.plot(dates_total,BC_L13_24hr)
plt.plot(dates_total,LSG_L13_24hr)
# plt.scatter(dates_total,BC_N16, s=1)
# plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (L13) - 24hr')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'L13_24hr.jpg')
plt.close()

plt.figure(figsize=(fsize_x,fsize_y))
plt.plot(dates_total,BC_L13_1wk)
plt.plot(dates_total,LSG_L13_1wk)
# plt.scatter(dates_total,BC_N16, s=1)
# plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (L13) - 1wk')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'L13_1wk.jpg')
plt.close()

############ N16 ###############
plt.figure(figsize=(fsize_x,fsize_y))
plt.plot(dates_total,BC_N16)
plt.plot(dates_total,LSG_N16)
# plt.scatter(dates_total,BC_N16, s=1)
# plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (N16) - Raw')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'N16_Raw.jpg')
plt.close()


plt.figure(figsize=(fsize_x,fsize_y))
plt.plot(dates_total,BC_N16_24hr)
plt.plot(dates_total,LSG_N16_24hr)
# plt.scatter(dates_total,BC_N16, s=1)
# plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (N16) - 24hr')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'N16_24hr.jpg')
plt.close()

plt.figure(figsize=(fsize_x,fsize_y))
plt.plot(dates_total,BC_N16_1wk)
plt.plot(dates_total,LSG_N16_1wk)
# plt.scatter(dates_total,BC_N16, s=1)
# plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (N16) - 1wk')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'N16_1wk.jpg')
plt.close()


print('\n%% Comparing Gas Flux Time Series (Monthly) %%\n')

# Add a month column to data_BC and data_LSG
BC_Months=np.zeros(len(dates_BC))
BC_Months[:]=np.NaN

LSG_Months=np.zeros(len(dates_LSG))
LSG_Months[:]=np.NaN

for i in np.arange(len(BC_Months)):
    BC_Months[i]=int(dates_BC[i][5:7])

for i in np.arange(len(LSG_Months)):
    LSG_Months[i]=int(dates_LSG[i][5:7])

data_BC['Month']=BC_Months
data_LSG['Month']=LSG_Months

BC_Data_M=data_BC.groupby(by='Month').mean()
LSG_Data_M=data_LSG.groupby(by='Month').mean()

BC_Data_M_std=data_BC.groupby(by='Month').std()
LSG_Data_M_std=data_LSG.groupby(by='Month').std()

MonthList=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']

x=np.arange(len(MonthList))
width = 0.35  # the width of the bars

############## L13 ##################
fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13'], width, label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13'], width,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_L13_Raw.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13'], width, yerr=BC_Data_M_std.loc[:,'Ft_L13'],capsize=2,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13'], width, yerr=LSG_Data_M_std.loc[:,'Ft_L13'],capsize=2,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_L13_Raw_wErr.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13_24hr'], width, label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13_24hr'], width,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (L13) - 24hr')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_L13_24hr.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13_24hr'], width, yerr=BC_Data_M_std.loc[:,'Ft_L13_24hr'],capsize=2,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13_24hr'], width, yerr=LSG_Data_M_std.loc[:,'Ft_L13_24hr'],capsize=2,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (L13) - 24hr')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_L13_24hr_wErr.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13_1wk'], width,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13_1wk'], width,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (L13) - 1wk')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_L13_1wk.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13_1wk'], width, yerr=BC_Data_M_std.loc[:,'Ft_L13_1wk'],capsize=2,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13_1wk'], width, yerr=LSG_Data_M_std.loc[:,'Ft_L13_1wk'],capsize=2,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (L13) - 1wk')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_L13_1wk_wErr.jpg')
plt.close()

############## N16 ##################
fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16'], width, label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16'], width,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (N16) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_N16_Raw.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16'], width, yerr=BC_Data_M_std.loc[:,'Ft_N16'],capsize=2,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16'], width, yerr=LSG_Data_M_std.loc[:,'Ft_N16'],capsize=2,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (N16) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_N16_Raw_wErr.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16_24hr'], width, label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16_24hr'], width,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (N16) - 24hr')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_N16_24hr.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16_24hr'], width, yerr=BC_Data_M_std.loc[:,'Ft_N16_24hr'],capsize=2,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16_24hr'], width, yerr=LSG_Data_M_std.loc[:,'Ft_N16_24hr'],capsize=2,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (N16) - 24hr')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_N16_24hr_wErr.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16_1wk'], width,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16_1wk'], width,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (N16) - 1wk')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_N16_1wk.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16_1wk'], width, yerr=BC_Data_M_std.loc[:,'Ft_N16_1wk'],capsize=2,label='BC')
rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16_1wk'], width, yerr=LSG_Data_M_std.loc[:,'Ft_N16_1wk'],capsize=2,label='LSG')
ax.set_ylabel('Flux (mol/m^2-s)')
ax.set_title('Monthly Total Air Sea Gas Flux (N16) - 1wk')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir_Compare+'Month_N16_1wk_wErr.jpg')
plt.close()


