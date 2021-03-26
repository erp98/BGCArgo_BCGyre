#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:25:58 2021

@author: Ellen
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import RandomFxns as RF

MODEL_TYPE=1
# 0: BC by Shape
# 1: BC by bathymetry

if MODEL_TYPE ==0:
    BCData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Shape/GasFluxData_BC.csv')
    GyreData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Shape/GasFluxData_Gyre.csv')
    FigDir_Compare='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_BC_Shape/CompareFlux_'
    print('\nUSING SHAPE MODEL\n')
elif MODEL_TYPE ==1:
    BCData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/GasFluxData_BC.csv')
    GyreData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/GasFluxData_Gyre.csv')
    FigDir_Compare='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_BC_Bath/CompareFlux_'
    print('\nUSING BATHYMETRY MODEL\n')
    

lat_N=65.000
lat_S= 48.00
lon_E= -45.00
lon_W= -70.00

fsize_x=10
fsize_y=6

# Plot data points on a map
lat_BC=BCData.loc[:,'Lat']
lon_BC=BCData.loc[:,'Lon']

lat_G=GyreData.loc[:,'Lat']
lon_G=GyreData.loc[:,'Lon']

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
plt.scatter(lon_BC,lat_BC, s=1, c='blue',label='Boundary Current')
plt.scatter(lon_G,lat_G, s=1, c='orange',label='Gyre')
plt.legend(['Boundary Current (n='+str(len(lon_BC))+')','Gyre (n='+str(len(lon_G))+')'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(FigDir_Compare+'Map.jpg')
plt.close()

## Group data by date
Mean_BCData=BCData.groupby(by='Date_6hr').mean()
Std_BCData=BCData.groupby(by='Date_6hr').std()

Mean_GData=GyreData.groupby(by='Date_6hr').mean()
Std_GData=GyreData.groupby(by='Date_6hr').std()

# Get full range of dates in between first and last profile date
# Boundary Curreny
date_list_BC=Mean_BCData.index.values.tolist()
# Reformat date list
dl_t=[[]]*len(date_list_BC)
for i in np.arange(len(dl_t)):
    dl_t[i]=date_list_BC[i][:19]
date_list_BC=dl_t

FirstDate_BC=date_list_BC[0]
LastDate_BC=date_list_BC[-1]

AllDates_BC_df=pd.date_range(str(FirstDate_BC),str(LastDate_BC),freq='6H')
AllDates_BC=[[]]*len(AllDates_BC_df)
for i in np.arange(len(AllDates_BC)):
    AllDates_BC[i]=str(AllDates_BC_df[i])
AllDates_BC=np.array(AllDates_BC)

# Gyre
date_list_G=Mean_GData.index.values.tolist()
# Reformat date list
dl_t=[[]]*len(date_list_G)
for i in np.arange(len(dl_t)):
    dl_t[i]=date_list_G[i][:19]
date_list_G=dl_t
FirstDate_G=date_list_G[0]
LastDate_G=date_list_G[-1]

AllDates_G_df=pd.date_range(str(FirstDate_G),str(LastDate_G),freq='6H')
AllDates_G=[[]]*len(AllDates_G_df)
for i in np.arange(len(AllDates_G)):
    AllDates_G[i]=str(AllDates_G_df[i])
AllDates_G=np.array(AllDates_G)

min_flag = np.NaN
# min_flag = 0; min date is LSG
# min_flag = 1; min date is BC

if AllDates_G[0]<AllDates_BC[0]:
    MinDate=AllDates_G[0]
    min_flag =0
else:
    MinDate=AllDates_BC[0]
    min_flag=1

if AllDates_G[-1]>AllDates_BC[-1]:
    MaxDate=AllDates_G[-1]
else:
    MaxDate=AllDates_BC[-1]
    
dates_total_df=pd.date_range(str(MinDate),str(MaxDate),freq='6H')
dates_total=[[]]*len(dates_total_df)
for i in np.arange(len(dates_total_df)):
    dates_total[i]=str(dates_total_df[i])
dates_total=np.array(dates_total)

######### BC #########
#L13_reform,L13_24hr_reform,L13_1wk_reform,N16_reform,N16_24hr_reform,N16_1wk_reform
#  #[L13_reform=BC_L13, L13_24hr_reform=BC_L13_24hr, L13_1wk_reform=BC_L13_1wk, N16_reform=BC_N16, N16_24hr_reform=BC_N16_24hr, N16_1wk_reform=BC_N16_1wk]  
# [BC_L13, BC_L13_24hr, BC_L13_1wk, BC_N16, BC_N16_24hr, BC_N16_1wk]=RF.MatchData2Dates(alldates=dates_total, dates=date_list_BC, L13=Ft_L13_BC, L13_24hr=Ft_L13_24hr_BC, L13_1wk=Ft_L13_1wk_BC,
#                                                                                      N16=Ft_N16_BC, N16_24hr=Ft_N16_24hr_BC , N16_1wk=Ft_N16_1wk_BC)
BC_Data_ref=RF.MatchData2Dates(alldates=dates_total, dates=date_list_BC,Data=Mean_BCData)

######### LSG #########
# [LSG_L13, LSG_L13_24hr, LSG_L13_1wk, LSG_N16, LSG_N16_24hr, LSG_N16_1wk]=RF.MatchData2Dates(alldates=dates_total, dates=date_list_G, L13=Ft_L13_G, L13_24hr=Ft_L13_24hr_G, L13_1wk=Ft_L13_1wk_G,
#                                                                                       N16=Ft_N16_G, N16_24hr=Ft_N16_24hr_G , N16_1wk=Ft_N16_1wk_G)
LSG_Data_ref=RF.MatchData2Dates(alldates=dates_total, dates=date_list_G,Data=Mean_GData)

BC_L13=BC_Data_ref.loc[:,'Ft_L13']
BC_N16=BC_Data_ref.loc[:,'Ft_N16']
LSG_L13=LSG_Data_ref.loc[:,'Ft_L13']
LSG_N16=LSG_Data_ref.loc[:,'Ft_N16']

## Moving Averages ##
## Moving Average: 24 hrs ##
# BC
BC_N16_24hr=BC_Data_ref.loc[:,'Fd_N16'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_N16'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_N16'].rolling(int((24/6)),min_periods=1).mean()
BC_L13_24hr=BC_Data_ref.loc[:,'Fd_L13'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_L13'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_L13'].rolling(int((24/6)),min_periods=1).mean()

BC_N16_1wk=BC_Data_ref.loc[:,'Fd_N16'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_N16'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_N16'].rolling(int((24*7/6)),min_periods=1).mean()
BC_L13_1wk=BC_Data_ref.loc[:,'Fd_L13'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_L13'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_L13'].rolling(int((24*7/6)),min_periods=1).mean()

# gyre
LSG_N16_24hr=LSG_Data_ref.loc[:,'Fd_N16'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_N16'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_N16'].rolling(int((24/6)),min_periods=1).mean()
LSG_L13_24hr=LSG_Data_ref.loc[:,'Fd_L13'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_L13'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_L13'].rolling(int((24/6)),min_periods=1).mean()

LSG_N16_1wk=LSG_Data_ref.loc[:,'Fd_N16'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_N16'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_N16'].rolling(int((24*7/6)),min_periods=1).mean()
LSG_L13_1wk=LSG_Data_ref.loc[:,'Fd_L13'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_L13'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_L13'].rolling(int((24*7/6)),min_periods=1).mean()


# Make figure labels
num_ticks=20
xticks_ind2=np.arange(0,(len(dates_total)//num_ticks*num_ticks)+1, step=len(dates_total)//num_ticks)
xticks_labels2=[]
for i in xticks_ind2:
    xticks_labels2=xticks_labels2+[dates_total[i]]

############ L13 ###############
plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_L13)
# plt.plot(dates_total,LSG_L13)
plt.scatter(dates_total,BC_L13, s=1)
plt.scatter(dates_total,LSG_L13, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (L13) - Raw')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'L13_Raw.jpg')
plt.close()


plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_L13_24hr)
# plt.plot(dates_total,LSG_L13_24hr)
plt.scatter(dates_total,BC_L13_24hr, s=1)
plt.scatter(dates_total,LSG_L13_24hr, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (L13) - 24hr')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'L13_24hr.jpg')
plt.close()

plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_L13_1wk)
# plt.plot(dates_total,LSG_L13_1wk)
plt.scatter(dates_total,BC_L13_1wk, s=1)
plt.scatter(dates_total,LSG_L13_1wk, s= 1)
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
# plt.plot(dates_total,BC_N16)
# plt.plot(dates_total,LSG_N16)
plt.scatter(dates_total,BC_N16, s=1)
plt.scatter(dates_total,LSG_N16, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (N16) - Raw')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'N16_Raw.jpg')
plt.close()


plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_N16_24hr)
# plt.plot(dates_total,LSG_N16_24hr)
plt.scatter(dates_total,BC_N16_24hr, s=1)
plt.scatter(dates_total,LSG_N16_24hr, s= 1)
plt.legend(['LS-BC','LS-G'])
plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
plt.xlabel('Date')
plt.ylabel('Flux (mol/m^2-s)')
plt.title('Total Air Sea Gas Flux (N16) - 24hr')
plt.subplots_adjust(bottom=0.3)
plt.savefig(FigDir_Compare+'N16_24hr.jpg')
plt.close()

plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_N16_1wk)
# plt.plot(dates_total,LSG_N16_1wk)
plt.scatter(dates_total,BC_N16_1wk, s=1)
plt.scatter(dates_total,LSG_N16_1wk, s= 1)
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
BC_Months=np.zeros(len(dates_total))
BC_Months[:]=np.NaN

LSG_Months=np.zeros(len(dates_total))
LSG_Months[:]=np.NaN

for i in np.arange(len(BC_Months)):
    BC_Months[i]=int(dates_total[i][5:7])

for i in np.arange(len(LSG_Months)):
    LSG_Months[i]=int(dates_total[i][5:7])

BC_Data_ref['Month']=BC_Months
LSG_Data_ref['Month']=LSG_Months

BC_Data_M=BC_Data_ref.groupby(by='Month').mean()
LSG_Data_M=LSG_Data_ref.groupby(by='Month').mean()

BC_Data_M_std=BC_Data_ref.groupby(by='Month').std()
LSG_Data_M_std=LSG_Data_ref.groupby(by='Month').std()

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
