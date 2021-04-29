#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 07:19:19 2021

@author: Ellen
"""

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
import RandomFxns as RF

#####################
## Some Parameters ##
#####################
badfloat=4901141

# N Atlantic Region
lat_N=80.000
lat_S= 40.00
lon_E= 0.00
lon_W= -80.00

# Labrador Sea Region
lab_N=65
lab_S=48
lab_E=-45
lab_W=-70

fsize_x=10
fsize_y=6

###################

data_types=[0,1]

bc_counter = 0
gyre_counter = -0.25

bc_wmo =[]
bc_start = []
bc_end = []
gyre_wmo =[]
gyre_start = []
gyre_end =[]

ODir = '/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/OverlapFiles_BC/'
for data_i in data_types:
    if data_i == 0:
        # Load BC float data
        print('\n%% Loading BC Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BCFloats_'
        FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_Float/BC_'
        TrajDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_Float/Trajectories/BC_'

        title_add='LS-BC Floats'
    elif data_i == 1:
        # Load Lab Gyre float data
        print('\n%% Loading Lab Sea Gyre Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
        FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_Float/LabSea_'
        TrajDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_Float/Trajectories/LabSea_'

        title_add='LS-G Floats'
        
    FileList=glob.glob(CSVDir_AS+'*.csv')
    df_count = 0
    
    file_count=0
    
    for filedir in FileList:
        
        # Read in each data file
        fluxdata = pd.read_csv(filedir)
        
        wmo = int(fluxdata.loc[:,'WMO'].unique()[0])
        
        if wmo != badfloat:
            
            # Crop data into correct region 
            fluxdata.loc[fluxdata.loc[:,'Lat']>lab_N,:]=np.NaN
            fluxdata.loc[fluxdata.loc[:,'Lat']<lab_S,:]=np.NaN
            fluxdata.loc[fluxdata.loc[:,'Lon']>lab_E,:]=np.NaN
            fluxdata.loc[fluxdata.loc[:,'Lon']<lab_W,:]=np.NaN
                
            fluxdata=fluxdata.dropna()
            
            if fluxdata.shape[0] > 0:
            
                # Plot 
                a=1
                
                date_float=fluxdata.loc[:,'Date']
                lat_float=fluxdata.loc[:,'Lat']
                lon_float=fluxdata.loc[:,'Lon']
                
                min_date = np.nanmin(date_float)
                max_date = np.nanmax(date_float)
                
                values = np.zeros(len(date_float))
                
                if data_i == 0:
                    values[:]=bc_counter
                    bc_counter = bc_counter +1
                    bc_wmo=bc_wmo+[wmo]
                    bc_start=bc_start+[min_date]
                    bc_end = bc_end + [max_date]
                    marker_type = '-'
                elif data_i == 1:
                    values[:]=gyre_counter
                    gyre_counter = gyre_counter +1
                    gyre_wmo=gyre_wmo+[wmo]
                    gyre_start=gyre_start+[min_date]
                    gyre_end = gyre_end + [max_date]
                    marker_type = ':'
            
                plt.figure(1)
                plt.plot(date_float, values,marker_type, label = str(wmo))
                
                
                # Save individual float trajectories
                plt.figure(figsize=(fsize_x,fsize_y))
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
                plt.title('Trajectory for Float '+str(wmo))
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.scatter(lon_float, lat_float,s=1)
                plt.savefig(TrajDir+str(wmo)+'_Trajectory.jpg')
                plt.clf(); plt.close()

plt.figure(1)
locs, labels = plt.xticks()

# Make figure labels
num_ticks=20
step_size = np.floor(len(locs)/num_ticks)

locs_new=np.zeros(num_ticks)
locs_new[:]=np.NaN
labels_new=[[]]*num_ticks

for i in np.arange(20):
    if i != 0:
        ind = int(i * step_size -1)
    else:
        ind = int(i * step_size)
    
    locs_new[i]=locs[ind]
    labels_new[i]=labels[ind]
    
plt.legend(bbox_to_anchor=(1.05, 1))
plt.xticks(ticks=locs_new, labels =labels_new)
plt.show()

print('\n Starting Overlap')
for i in np.arange(len(bc_wmo)):
    b_wmo = bc_wmo[i]
    b_s = bc_start[i]
    b_e = bc_end[i]
    
    print('BC Float: ', b_wmo)
    o_floats = []
    for j in np.arange(len(gyre_wmo)):
        g_s = gyre_start[j]
        g_e = gyre_end[j]
        # Check to see if there is overlap
        o_flag = 0
        if g_s >= b_s and g_e<=b_e:
            o_flag =1
        elif g_s <= b_s and g_e <= b_e:
            o_flag = 1
        elif g_s < b_e and g_e > b_e:
            o_flag = 1
        elif g_s < b_s and g_e > b_e:
            o_flaf = 1
        
        if o_flag == 1:
            o_floats = o_floats + [gyre_wmo[j]]
    
    if len(o_floats) >0:
        w=np.zeros(len(o_floats))
        w[:]=int(b_wmo)
        
        odf = pd.DataFrame({'BC_Float': w, 'Gyre_Float': o_floats})
        odf.to_csv(ODir+str(b_wmo)+'.csv')
        
    #     # And combine into one big data frame
    #     if df_count == 0:
    #         AllData = pd.read_csv(filedir)
    #         df_count = 1
    #     else:
    #         AllData=AllData.append(fluxdata)
        
    #     file_count=file_count+1
    
    # # If BC floats, crop data so it is in Lab Sea
    
    # AllData.loc[AllData.loc[:,'Lat']>lab_N,:]=np.NaN
    # AllData.loc[AllData.loc[:,'Lat']<lab_S,:]=np.NaN
    # AllData.loc[AllData.loc[:,'Lon']>lab_E,:]=np.NaN
    # AllData.loc[AllData.loc[:,'Lon']<lab_W,:]=np.NaN
    
    # # Remove bad float data
    # AllData.loc[AllData.loc[:,'WMO']==badfloat,:]=np.NaN
    
    # AllData=AllData.dropna()
    
    
    # # Plot float trajectories
    # float_wmo= AllData.loc[:,'WMO'].unique().tolist()
    # wmo_count =0
    # print('\n%% Plotting Trajectories %%\n')
    # for wmo in float_wmo:
        
        
    #     lat=AllData.loc[AllData.loc[:,'WMO']==wmo,'Lat'].to_numpy()
    #     lon=AllData.loc[AllData.loc[:,'WMO']==wmo,'Lon'].to_numpy()
        
    #     plt.figure(1,figsize=(fsize_x,fsize_y))
    #     NA = plt.axes(projection=ct.crs.PlateCarree())
    #     NA.set_extent([lon_E, lon_W, lat_S, lat_N])
    #     lonval=-1*np.arange(-lon_E,-lon_W+1,10)
    #     latval=np.arange(lat_S,lat_N+1,10)
    #     NA.set_xticks(lonval, crs=ct.crs.PlateCarree())
    #     NA.set_yticks(latval, crs=ct.crs.PlateCarree())
    #     lon_formatter = LongitudeFormatter()
    #     lat_formatter = LatitudeFormatter()
    #     NA.add_feature(ct.feature.COASTLINE)
    #     NA.add_feature(ct.feature.OCEAN)
    #     NA.xaxis.set_major_formatter(lon_formatter)
    #     NA.yaxis.set_major_formatter(lat_formatter)
    #     #plt.title('Trajectory for Float '+str(WMO))
    #     plt.plot(lon[0],lat[0],'go')
    #     plt.plot(lon[len(lon)-1],lat[len(lat)-1],'ro')
    #     plt.xlabel('Longitude')
    #     plt.ylabel('Latitude')
    #     plt.plot(lon, lat)
                
    #     if wmo_count == len(float_wmo)-1:
    #         plt.savefig(TrajDir+'Trajectory.jpg')
    #         plt.clf(); plt.close()
        
    #     wmo_count=wmo_count+1
    
    # lat=AllData.loc[:,'Lat']
    # lon=AllData.loc[:,'Lon']
    
    # # Save all trajectories
    # plt.figure(2,figsize=(fsize_x,fsize_y))
    # NA = plt.axes(projection=ct.crs.PlateCarree())
    # NA.set_extent([lon_E, lon_W, lat_S, lat_N])
    # lonval=-1*np.arange(-lon_E,-lon_W+1,10)
    # latval=np.arange(lat_S,lat_N+1,10)
    # NA.set_xticks(lonval, crs=ct.crs.PlateCarree())
    # NA.set_yticks(latval, crs=ct.crs.PlateCarree())
    # lon_formatter = LongitudeFormatter()
    # lat_formatter = LatitudeFormatter()
    # NA.add_feature(ct.feature.COASTLINE)
    # NA.add_feature(ct.feature.OCEAN)
    # NA.xaxis.set_major_formatter(lon_formatter)
    # NA.yaxis.set_major_formatter(lat_formatter)
    # #plt.title('Trajectory for Float '+str(WMO))
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # if data_i ==0:
    #     plt.scatter(lon, lat,s=1, color='blue')
    # elif data_i==1:
    #     plt.scatter(lon, lat,s=1, color='orange')
    #     plt.savefig('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries_Float/Trajectories/CompareFlux_Map.jpg')
    #     plt.clf(); plt.close()
    
    
#    print('\nThere are '+str(len(float_wmo))+' floats\n')
   
#     # Calculate the mean at given time in specific region 
#     MeanData=AllData.groupby(by='Date').mean()
#     StdData=AllData.groupby(by='Date').std()
    
#     date_list=MeanData.index.values.tolist()
#     FirstDate=date_list[0]
#     LastDate=date_list[-1]
    
#     # Get full range of dates in between first and last profile date
#     AllDates_df=pd.date_range(str(FirstDate),str(LastDate),freq='6H')
#     AllDates=[[]]*len(AllDates_df)
#     for i in np.arange(len(AllDates)):
#         AllDates[i]=str(AllDates_df[i])
#     AllDates=np.array(AllDates)
    
#     # Go through dates and match data to correct time period
#     MeanData = RF.MatchData2Dates(alldates=AllDates, dates=date_list, Data=MeanData)
    
#     Ft_N16=MeanData.loc[:,'Ft_N16']
#     Fd_N16=MeanData.loc[:,'Fd_N16']
#     Fp_N16=MeanData.loc[:,'Fp_N16']
#     Fc_N16=MeanData.loc[:,'Fc_N16']

#     Ft_L13=MeanData.loc[:,'Ft_L13']
#     Fd_L13=MeanData.loc[:,'Fd_L13']
#     Fp_L13=MeanData.loc[:,'Fp_L13']
#     Fc_L13=MeanData.loc[:,'Fc_L13']
        
#     num_ticks=20
#     xticks_ind=np.arange(0,(len(AllDates)//num_ticks*num_ticks)+1, step=len(AllDates)//num_ticks)
#     xticks_labels=[]
#     for i in xticks_ind:
#         xticks_labels=xticks_labels+[AllDates[i]]
    
#     print('\n%% Plotting Gas Flux Time Series %%\n')
    
#     ## Raw Data ##
#     plt.figure(figsize=(fsize_x,fsize_y))
#     plt.plot(AllDates, Ft_N16)
#     #plt.scatter(AllDates, Ft_N16, s=1)
#     plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
#     plt.xlabel('Date')
#     plt.ylabel('Flux (mol/m^2-s)')
#     plt.title('Total Air-Sea Flux (N16) - '+title_add+' - Raw')
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(FigDir+'N16_Raw.jpg')
#     plt.close()
    
#     plt.figure(figsize=(fsize_x,fsize_y))
#     plt.plot(AllDates, Ft_L13)
#     #plt.scatter(AllDates, Ft_L13, s=1)
#     plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
#     plt.xlabel('Date')
#     plt.ylabel('Flux (mol/m^2-s)')
#     plt.title('Total Air-Sea Flux (L13) - '+title_add+' - Raw')
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(FigDir+'L13_Raw.jpg')
#     plt.close()
    
#     ## Moving Average: 24 hrs ##
#     Ft_N16_24hr=pd.Series(Fd_N16).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fp_N16).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fc_N16).rolling(int((24/6)),min_periods=1).mean()
#     Ft_L13_24hr=pd.Series(Fd_L13).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fp_L13).rolling(int((24/6)),min_periods=1).mean()+pd.Series(Fc_L13).rolling(int((24/6)),min_periods=1).mean()
    
#     plt.figure(figsize=(fsize_x,fsize_y))
#     plt.plot(AllDates, Ft_N16_24hr)
#     #plt.scatter(AllDates, Ft_N16_24hr,s=1)
#     plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
#     plt.xlabel('Date')
#     plt.ylabel('Flux (mol/m^2-s)')
#     plt.title('Total Air-Sea Flux (N16) - '+title_add+' - 24 hr Moving Average')
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(FigDir+'N16_24hr.jpg')
#     plt.close()
    
#     plt.figure(figsize=(fsize_x,fsize_y))
#     plt.plot(AllDates,Ft_L13_24hr)
#     #plt.scatter(AllDates,Ft_L13_24hr,s=1)
#     plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
#     plt.xlabel('Date')
#     plt.ylabel('Flux (mol/m^2-s)')
#     plt.title('Total Air-Sea Flux (L13) - '+title_add+' - 24 hr Moving Average')
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(FigDir+'L13_24hr.jpg')
#     plt.close()
    
#     ## Moving Average: 1 week ##
#     Ft_N16_1wk=pd.Series(Fd_N16).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fp_N16).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fc_N16).rolling(int((7*24)/6),min_periods=1).mean()
#     Ft_L13_1wk=pd.Series(Fd_L13).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fp_L13).rolling(int((7*24)/6),min_periods=1).mean()+pd.Series(Fc_L13).rolling(int((7*24)/6),min_periods=1).mean()
    
#     plt.figure(figsize=(fsize_x,fsize_y))
#     plt.plot(AllDates, Ft_N16_1wk)
#     #plt.scatter(AllDates, Ft_N16_1wk,s=1)
#     plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
#     plt.xlabel('Date')
#     plt.ylabel('Flux (mol/m^2-s)')
#     plt.title('Total Air-Sea Flux (N16) - '+title_add+' - 1 week Moving Average')
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(FigDir+'N16_1wk.jpg')
#     plt.close()
    
    
#     plt.figure(figsize=(fsize_x,fsize_y))
#     plt.plot(AllDates, Ft_L13_1wk)
#     #plt.scatter(AllDates, Ft_L13_1wk,s=1)
#     plt.xticks(ticks=xticks_ind,labels=xticks_labels, rotation=90)
#     plt.xlabel('Date')
#     plt.ylabel('Flux (mol/m^2-s)')
#     plt.title('Total Air-Sea Flux (L13) - '+title_add+' - 1 week Moving Average')
#     plt.subplots_adjust(bottom=0.3)
#     plt.savefig(FigDir+'L13_1wk.jpg')
#     plt.close()
    
#     temp_Data_store=pd.DataFrame({'Ft_L13': Ft_L13,'Fd_L13': Fd_L13,'Fp_L13': Fp_L13,'Fc_L13': Fc_L13,'Ft_N16': Ft_N16,'Fp_N16': Fp_N16,'Fd_N16': Fd_N16, 'Fc_N16': Fc_N16})
#     if data_i == 0:
#         dates_BC=AllDates
#         data_BC=temp_Data_store
#     elif data_i ==1:
#         dates_LSG=AllDates
#         data_LSG=temp_Data_store

# # Make complete date range for data
# print('\n%% Comparing Gas Flux Time Series %%\n')
     
# min_flag = np.NaN
# # min_flag = 0; min date is LSG
# # min_flag = 1; min date is BC

# if dates_LSG[0]<dates_BC[0]:
#     MinDate=dates_LSG[0]
#     min_flag =0
# else:
#     MinDate=dates_BC[0]
#     min_flag=1

# if dates_LSG[-1]>dates_BC[-1]:
#     MaxDate=dates_LSG[-1]
# else:
#     MaxDate=dates_BC[-1]
    
# dates_total_df=pd.date_range(str(MinDate),str(MaxDate),freq='6H')
# dates_total=[[]]*len(dates_total_df)
# for i in np.arange(len(dates_total_df)):
#     dates_total[i]=str(dates_total_df[i])
# dates_total=np.array(dates_total)

# BC_Data_ref=RF.MatchData2Dates(alldates=dates_total, dates=dates_BC,Data=data_BC)

# ######### LSG #########
# # [LSG_L13, LSG_L13_24hr, LSG_L13_1wk, LSG_N16, LSG_N16_24hr, LSG_N16_1wk]=RF.MatchData2Dates(alldates=dates_total, dates=date_list_G, L13=Ft_L13_G, L13_24hr=Ft_L13_24hr_G, L13_1wk=Ft_L13_1wk_G,
# #                                                                                       N16=Ft_N16_G, N16_24hr=Ft_N16_24hr_G , N16_1wk=Ft_N16_1wk_G)
# LSG_Data_ref=RF.MatchData2Dates(alldates=dates_total, dates=dates_LSG,Data=data_LSG)

# BC_L13=BC_Data_ref.loc[:,'Ft_L13']
# BC_N16=BC_Data_ref.loc[:,'Ft_N16']
# LSG_L13=LSG_Data_ref.loc[:,'Ft_L13']
# LSG_N16=LSG_Data_ref.loc[:,'Ft_N16']

# ## Moving Averages ##
# ## Moving Average: 24 hrs ##
# # BC
# BC_N16_24hr=BC_Data_ref.loc[:,'Fd_N16'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_N16'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_N16'].rolling(int((24/6)),min_periods=1).mean()
# BC_L13_24hr=BC_Data_ref.loc[:,'Fd_L13'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_L13'].rolling(int((24/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_L13'].rolling(int((24/6)),min_periods=1).mean()

# BC_N16_1wk=BC_Data_ref.loc[:,'Fd_N16'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_N16'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_N16'].rolling(int((24*7/6)),min_periods=1).mean()
# BC_L13_1wk=BC_Data_ref.loc[:,'Fd_L13'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fp_L13'].rolling(int((24*7/6)),min_periods=1).mean()+BC_Data_ref.loc[:,'Fc_L13'].rolling(int((24*7/6)),min_periods=1).mean()

# # gyre
# LSG_N16_24hr=LSG_Data_ref.loc[:,'Fd_N16'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_N16'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_N16'].rolling(int((24/6)),min_periods=1).mean()
# LSG_L13_24hr=LSG_Data_ref.loc[:,'Fd_L13'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_L13'].rolling(int((24/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_L13'].rolling(int((24/6)),min_periods=1).mean()

# LSG_N16_1wk=LSG_Data_ref.loc[:,'Fd_N16'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_N16'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_N16'].rolling(int((24*7/6)),min_periods=1).mean()
# LSG_L13_1wk=LSG_Data_ref.loc[:,'Fd_L13'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fp_L13'].rolling(int((24*7/6)),min_periods=1).mean()+LSG_Data_ref.loc[:,'Fc_L13'].rolling(int((24*7/6)),min_periods=1).mean()


# # Make figure labels
# num_ticks=20
# xticks_ind2=np.arange(0,(len(dates_total)//num_ticks*num_ticks)+1, step=len(dates_total)//num_ticks)
# xticks_labels2=[]
# for i in xticks_ind2:
#     xticks_labels2=xticks_labels2+[dates_total[i]]

# FigDir_Compare='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Flux_TimeSeries/CompareFlux_'

# ############ L13 ###############
# plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_L13)
# plt.plot(dates_total,LSG_L13)
# # plt.scatter(dates_total,BC_N16, s=1)
# # plt.scatter(dates_total,LSG_N16, s= 1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Flux (mol/m^2-s)')
# plt.title('Total Air Sea Gas Flux (L13) - Raw')
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(FigDir_Compare+'L13_Raw.jpg')
# plt.close()


# plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_L13_24hr)
# plt.plot(dates_total,LSG_L13_24hr)
# # plt.scatter(dates_total,BC_N16, s=1)
# # plt.scatter(dates_total,LSG_N16, s= 1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Flux (mol/m^2-s)')
# plt.title('Total Air Sea Gas Flux (L13) - 24hr')
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(FigDir_Compare+'L13_24hr.jpg')
# plt.close()

# plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_L13_1wk)
# plt.plot(dates_total,LSG_L13_1wk)
# # plt.scatter(dates_total,BC_N16, s=1)
# # plt.scatter(dates_total,LSG_N16, s= 1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Flux (mol/m^2-s)')
# plt.title('Total Air Sea Gas Flux (L13) - 1wk')
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(FigDir_Compare+'L13_1wk.jpg')
# plt.close()

# ############ N16 ###############
# plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_N16)
# plt.plot(dates_total,LSG_N16)
# # plt.scatter(dates_total,BC_N16, s=1)
# # plt.scatter(dates_total,LSG_N16, s= 1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Flux (mol/m^2-s)')
# plt.title('Total Air Sea Gas Flux (N16) - Raw')
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(FigDir_Compare+'N16_Raw.jpg')
# plt.close()


# plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_N16_24hr)
# plt.plot(dates_total,LSG_N16_24hr)
# # plt.scatter(dates_total,BC_N16, s=1)
# # plt.scatter(dates_total,LSG_N16, s= 1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Flux (mol/m^2-s)')
# plt.title('Total Air Sea Gas Flux (N16) - 24hr')
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(FigDir_Compare+'N16_24hr.jpg')
# plt.close()

# plt.figure(figsize=(fsize_x,fsize_y))
# plt.plot(dates_total,BC_N16_1wk)
# plt.plot(dates_total,LSG_N16_1wk)
# # plt.scatter(dates_total,BC_N16, s=1)
# # plt.scatter(dates_total,LSG_N16, s= 1)
# plt.legend(['LS-BC','LS-G'])
# plt.xticks(ticks=xticks_ind2,labels=xticks_labels2, rotation=90)
# plt.xlabel('Date')
# plt.ylabel('Flux (mol/m^2-s)')
# plt.title('Total Air Sea Gas Flux (N16) - 1wk')
# plt.subplots_adjust(bottom=0.3)
# plt.savefig(FigDir_Compare+'N16_1wk.jpg')
# plt.close()


# print('\n%% Comparing Gas Flux Time Series (Monthly) %%\n')

# # Add a month column to data_BC and data_LSG
# BC_Months=np.zeros(len(dates_BC))
# BC_Months[:]=np.NaN

# LSG_Months=np.zeros(len(dates_LSG))
# LSG_Months[:]=np.NaN

# for i in np.arange(len(BC_Months)):
#     BC_Months[i]=int(dates_BC[i][5:7])

# for i in np.arange(len(LSG_Months)):
#     LSG_Months[i]=int(dates_LSG[i][5:7])

# data_BC['Month']=BC_Months
# data_LSG['Month']=LSG_Months

# BC_Data_M=data_BC.groupby(by='Month').mean()
# LSG_Data_M=data_LSG.groupby(by='Month').mean()

# BC_Data_M_std=data_BC.groupby(by='Month').std()
# LSG_Data_M_std=data_LSG.groupby(by='Month').std()

# MonthList=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']

# x=np.arange(len(MonthList))
# width = 0.35  # the width of the bars

# ############## L13 ##################
# fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
# rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13'], width, label='BC')
# rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13'], width,label='LSG')
# ax.set_ylabel('Flux (mol/m^2-s)')
# ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
# ax.set_xticks(x)
# ax.set_xticklabels(MonthList)
# ax.legend()
# plt.savefig(FigDir_Compare+'Month_L13_Raw.jpg')
# plt.close()

# fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
# rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_L13'], width, yerr=BC_Data_M_std.loc[:,'Ft_L13'],capsize=2,label='BC')
# rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_L13'], width, yerr=LSG_Data_M_std.loc[:,'Ft_L13'],capsize=2,label='LSG')
# ax.set_ylabel('Flux (mol/m^2-s)')
# ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
# ax.set_xticks(x)
# ax.set_xticklabels(MonthList)
# ax.legend()
# plt.savefig(FigDir_Compare+'Month_L13_Raw_wErr.jpg')
# plt.close()


# ############## N16 ##################
# fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
# rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16'], width, label='BC')
# rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16'], width,label='LSG')
# ax.set_ylabel('Flux (mol/m^2-s)')
# ax.set_title('Monthly Total Air Sea Gas Flux (N16) - Raw')
# ax.set_xticks(x)
# ax.set_xticklabels(MonthList)
# ax.legend()
# plt.savefig(FigDir_Compare+'Month_N16_Raw.jpg')
# plt.close()

# fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
# rects1 = ax.bar(x - width/2, BC_Data_M.loc[:,'Ft_N16'], width, yerr=BC_Data_M_std.loc[:,'Ft_N16'],capsize=2,label='BC')
# rects2 = ax.bar(x + width/2, LSG_Data_M.loc[:,'Ft_N16'], width, yerr=LSG_Data_M_std.loc[:,'Ft_N16'],capsize=2,label='LSG')
# ax.set_ylabel('Flux (mol/m^2-s)')
# ax.set_title('Monthly Total Air Sea Gas Flux (N16) - Raw')
# ax.set_xticks(x)
# ax.set_xticklabels(MonthList)
# ax.legend()
# plt.savefig(FigDir_Compare+'Month_N16_Raw_wErr.jpg')
# plt.close()


