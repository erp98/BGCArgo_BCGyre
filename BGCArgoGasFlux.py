#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:31:59 2021

@author: Ellen
"""

import xarray as xr
from datetime import datetime#, timedelta, timezone
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import KDTree
import gsw
import gasex.airsea as AS
import RandomFxns as RF
import time
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)

## Depth_time_interp: Produce sections (date-time-variable)
# 0 = no sections
# 1 = sections
depth_time_interp = 0

print('What types of floats do you want to analyze?')

input_check = 0

while input_check == 0:
    floattype=input('1 (All), 2 (Boundary Current), 3 (Labrador), 4 (Both), 5(Other): ')
    floattype=int(floattype)
    
    if floattype == 1:
        print('\nAnalyzing ALL floats\n')
        input_check=1
    elif floattype ==2:
        print('\nAnalyzing BOUNDARY CURRENT floats\n')
        input_check=1
    elif floattype==3:
        print('\nAnalyzing LABRADOR SEA floats\n')
        input_check=1
    elif floattype==4:
        print('\nAnalyzing BOTH floats\n')
        input_check=1
    elif floattype==5:
        print('\nAnalyzing OTHER floats\n')
        input_check=1

tic_start=time.time()

################################
########## SOME VALUES #########
# Random parameters you can change  
refdate=datetime(1990,1,1)  # Not really important, only used for converting dates to numbers for interpolation
timestep=6 # hours --> matches ERA5 data spacing
freq_str=str(timestep)+'H'

# Surface pressure range used to calculate air-sea flux
surfP_val=5    # dbar
surfP_var=3  # dbar

minsurfP=surfP_val-surfP_var
maxsurfP=surfP_val+surfP_var

last_date_time=datetime(2020, 12, 31,18,0, 0)

lat_N=80.000
lat_S= 40.00
lon_E= 0.00
lon_W= -80.00

fsize_x=10
fsize_y=6

#good_QC=[b'1',b'2',b'5',b'8']
good_QC=[1,2,5,8]

# interp values
maxP_sec=2000
step_P=5

##############################

# Open ERA5 Reanalysis Data
print('\n%%% Loading ERA5 Data Files %%%\n')
print('U-10 data')
era5_u10_file='/Users/Ellen/Desktop/ERA5/era5_u10_6hr_2003_2020_E0_W-80_S40_N80.nc'
era_u10_data=xr.open_dataset(era5_u10_file)

print('V-10 data')
era5_v10_file='/Users/Ellen/Desktop/ERA5/era5_v10_6hr_2003_2020_E0_W-80_S40_N80.nc'
era_v10_data=xr.open_dataset(era5_v10_file)

print('SLP data')
era5_slp_file='/Users/Ellen/Desktop/ERA5/era5_mSLP_6hr_2003_2020_E0_W-80_S40_N80.nc'
era_slp_data=xr.open_dataset(era5_slp_file)
print('\n%%% Data Loaded %%%\n')

era5_time=era_u10_data.time.values
x=era_u10_data.longitude.values
y=era_u10_data.latitude.values
x_grid, y_grid= np.meshgrid(x,y)
print('Making Nearest Neighbor Tree...')
tree = KDTree(np.c_[x_grid.ravel(), y_grid.ravel()])

print('\n%% Making BGC Argo Float Lists %%\n')

## Directory information for saving figures
#FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Test/'
if floattype == 1:
    FloatDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_AllFloats.txt'
    FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/AllFloats/'
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_AllFloats.csv'
    CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_AllFloats_'
elif floattype ==2:
    FloatDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_BCFloats.txt'
    FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/BCFloats/'
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_BCFloats.csv'
    CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BCFloats_'
elif floattype==3:
    FloatDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_LabradorSeaFloats.txt'
    FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/LabradorFloats/'
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_LabradorFloats.csv'
    CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
elif floattype==4:
    FloatDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_Bothloats.txt'
    FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/BothFloats/'
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_BothFloats.csv'
    CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BothFloats_'
elif floattype==5:
    FloatDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_OtherFloats.txt'
    FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/OtherFloats/'
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_OtherFloats.csv'
    CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_OtherFloats_'

FigDir_Traj='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Trajectories/'

floatlist=[]
daclist=[]
count= 0
# Read in float info
with open(FloatDir) as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        t=line.strip()
        r=t.split('/')
        floatlist=floatlist+[int(r[1])]
        daclist=daclist+[r[0]]
 
pres_flag_total=np.zeros(len(floatlist))
pres_flag_total[:]=np.NaN
temp_flag_total=np.zeros(len(floatlist))
temp_flag_total[:]=np.NaN
sal_flag_total=np.zeros(len(floatlist))
sal_flag_total[:]=np.NaN
oxy_flag_total=np.zeros(len(floatlist))
oxy_flag_total[:]=np.NaN
flux_count=np.zeros(len(floatlist))
flux_count[:]=np.NaN

print('Number of Floats: ',len(floatlist))

for b in np.arange(len(floatlist)):
### For debugging and looking at specific floats ###
#for b in [10]:
    WMO=floatlist[b]
    dac=daclist[b]
    
    bad_index=[]
    #WMO=4900610
    #dac='coriolis'
    
    tic_float=time.time()
    print('\n%%% '+str(WMO)+' %%%\n')
    print(b, ' Floats completed; ', len(floatlist)-b,' Floats Left')
    BGCfile='/Users/Ellen/Desktop/ArgoGDAC/dac/'+dac+'/'+str(WMO)+'/'+str(WMO)+'_Sprof.nc'
    f = xr.open_dataset(BGCfile)
    
    float_vars=list(f.keys())
    float_vars=np.array(float_vars)
    
    oxy_check = np.where(float_vars=='DOXY')
    
    if oxy_check[0].size != 0:
        reft=f.REFERENCE_DATE_TIME.values
        dates=f.JULD.values
        
        date_reform=[[]]*dates.shape[0]
        for i in np.arange(len(date_reform)):
            if np.isnat(dates[i]) == False:
                date_reform[i]=datetime.fromisoformat(str(dates[i])[:-3])
            else:
                date_reform[i]=dates[i]
                bad_index=bad_index+[i]
    
        mindate=date_reform[0]
        minhour=mindate.hour
        
        if (minhour%timestep == 0 and mindate.minute == 0 and mindate.second ==0):
            minddate=mindate
        else:
            new_minhour=(minhour//timestep+1)*timestep
            if new_minhour == 24:
                
                # get last day of the month
                last_day = RF.last_day_of_month(mindate)
                
                if mindate.day == last_day:
                    mindate_roundup=datetime(mindate.year, mindate.month+1, 1, 0, 0, 0 )
                else:
                    mindate_roundup=datetime(mindate.year, mindate.month, mindate.day+1,0,0, 0)
            else:
                mindate_roundup=datetime(mindate.year, mindate.month, mindate.day,new_minhour,0, 0)

        q=1
        time_check=0
        
        while (q<=len(date_reform) and time_check==0):
            if isinstance(date_reform[-q],datetime) == True:
            #if np.isnat(date_reform[-q]) == False:
                maxdate=date_reform[-q]
                time_check=1
            else:
                  q=q+1  
        
        maxhour=maxdate.hour
        
        if maxhour%timestep != 0:
            new_maxhour=(maxhour//timestep)*timestep
            maxdate_rounddown=datetime(maxdate.year, maxdate.month, maxdate.day,new_maxhour,0, 0)
        elif maxhour%timestep == 0:
            if (maxdate.minute == 0 and maxdate.second ==0):
                maxdate_rounddown=maxdate
            else:
                new_maxhour=(maxhour//timestep)*timestep
                maxdate_rounddown=datetime(maxdate.year, maxdate.month, maxdate.day,new_maxhour,0, 0)
                
        ## Data only goes to 2020-12-31 18:000
        if maxdate.year == 2021:
            maxdate_rounddown=last_date_time
        date_6hr=np.arange(mindate_roundup, maxdate_rounddown,timestep*60*60,dtype='datetime64[s]')
        #date_6hr_pd=pd.date_range(mindate_roundup,maxdate_rounddown,freq=freq_str)
        
        # Convert numpy time to datetime
        date_6hr_reform=[[]]*len(date_6hr)
        for i in np.arange(len(date_6hr_reform)):
            date_6hr_reform[i]=datetime.fromisoformat(str(date_6hr[i]))

        
        ##############################
        #### Get surface values #######
        ##############################
        print('\nCompleting data quality control...\n')
        # Determine if using adjusted or not adjusted values
        [pres, pres_QC, pres_flag]=RF.DetermineAdjusted(raw_data=f.PRES.values,raw_data_QC=f.PRES_QC.values,a_data=f.PRES_ADJUSTED.values, a_data_QC=f.PRES_ADJUSTED_QC.values)
        [temp, temp_QC, temp_flag]=RF.DetermineAdjusted(raw_data=f.TEMP.values, raw_data_QC= f.TEMP_QC.values,a_data=f.TEMP_ADJUSTED.values, a_data_QC=f.TEMP_ADJUSTED_QC.values)
        [sal, sal_QC, sal_flag]=RF.DetermineAdjusted(raw_data=f.PSAL.values, raw_data_QC=f.PSAL_QC.values,a_data=f.PSAL_ADJUSTED.values,a_data_QC=f.PSAL_ADJUSTED_QC.values)
        [doxy, doxy_QC, doxy_flag]=RF.DetermineAdjusted(raw_data=f.DOXY.values, raw_data_QC=f.DOXY_QC.values,a_data=f.DOXY_ADJUSTED.values,a_data_QC=f.DOXY_ADJUSTED_QC.values)
        
        # Store flags for adjusted data
        pres_flag_total[b]=pres_flag
        temp_flag_total[b]=temp_flag
        sal_flag_total[b]=sal_flag
        oxy_flag_total[b]=doxy_flag
        
        ## Quality control float data ##
        pres=RF.ArgoQC(Data=pres, Data_QC=pres_QC, goodQC_flags=good_QC)
        temp=RF.ArgoQC(Data=temp, Data_QC=temp_QC, goodQC_flags=good_QC)
        sal=RF.ArgoQC(Data=sal, Data_QC=sal_QC, goodQC_flags=good_QC)
        doxy=RF.ArgoQC(Data=doxy, Data_QC=doxy_QC, goodQC_flags=good_QC)
        
        lat=f.LATITUDE.values
        lon=f.LONGITUDE.values
        
        # Surface values used to calculate flux
        surf_T=np.zeros(pres.shape[0])
        surf_T[:]=np.NaN
        surf_S=np.zeros(pres.shape[0])
        surf_S[:]=np.NaN
        surf_O=np.zeros(pres.shape[0])
        surf_O[:]=np.NaN
        
        # For each profile...
        print('\nGetting surface values...\n')
        for i in np.arange(pres.shape[0]):
            ## Determine what values fall in the given pressure range and save index
            j=0
            pressure_check=0
            pres_ind=[]
            #print(pres_ind)
            prev_val=0
            
            while j < len(pres[i]) and pressure_check == 0:
                
                # Determine if pressure value falls in range 
                if pres[i][j] >= minsurfP and pres[i][j] <= maxsurfP:
                    pres_ind=pres_ind+[j]
                    current_val=1
                else:
                    current_val=0
                    
                if current_val == 0 and prev_val == 1:
                    pressure_check=1
                    
                if current_val == 1:
                    prev_val=1
                
                j=j+1
                    
            # Use pressure indeces and calculate average surface T, S, and O for each profile
            #print(pres_ind)
            if pres_ind==[]:
                # No pressures in the correct range
                surf_T[i]=np.NaN
                surf_S[i]=np.NaN
                surf_O[i]=np.NaN
            elif len(pres_ind)==1:
                # only one pressure surface in range
                surf_T[i]=temp[i][pres_ind[0]]
                surf_S[i]=sal[i][pres_ind[0]]
                surf_O[i]=doxy[i][pres_ind[0]]
            else:
                # more than one
                minind=pres_ind[0]
                maxind=pres_ind[len(pres_ind)-1]+1
                
                # Subset of surface values
                T_sub=temp[i][minind:maxind]
                S_sub=sal[i][minind:maxind]
                O_sub=doxy[i][minind:maxind]
                
                # Use mean values
                if len(T_sub) == 0:
                    surf_T[i]=np.NaN
                else:
                    surf_T[i]=np.nanmean(T_sub)
                    
                if len(S_sub)==0:
                    surf_S[i]=np.NaN
                else:
                    surf_S[i]=np.nanmean(S_sub)
                
                if len(O_sub) == 0:
                    surf_O[i]=np.NaN
                else:
                    surf_O[i]=np.nanmean(O_sub)
            
        # Remove bad times
        
        if len(bad_index)>0:
            for b_i in bad_index:
                
                # Convert arrays to lists to use pop
                surf_T=list(surf_T)
                surf_S=list(surf_S)
                surf_O=list(surf_O)
                lat=list(lat)
                lon=list(lon)
                
                # POP
                date_reform.pop(b_i)
                surf_T.pop(b_i)
                surf_S.pop(b_i)
                surf_O.pop(b_i)
                lat.pop(b_i)
                lon.pop(b_i)
                
                # Convert lists back to arrays
                surf_T=np.array(surf_T)
                surf_S=np.array(surf_S)
                surf_O=np.array(surf_O)
                lat=np.array(lat)
                lon=np.array(lon)
        
        ## If all of the surface values are nans, do not make figures or do calculations
        if (sum(np.isnan(surf_T))==len(surf_T) or sum(np.isnan(surf_S))==len(surf_S) or sum(np.isnan(surf_O))==len(surf_O)):
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
            plt.title('Trajectory for Float '+str(WMO))
            plt.plot(lon[0],lat[0],'go')
            plt.plot(lon[len(lon)-1],lat[len(lat)-1],'ro')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.plot(lon, lat)
            plt.savefig(FigDir_Traj+str(WMO)+'_Trajectory_NAtlantic.jpg')
            plt.close()
            
            flux_count[b]=0
            print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('%% Not Enough Data to Calculate Flux %%')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
        else:
            ## Plot surface values vs time
            figs, axs=plt.subplots(3,1, figsize=(fsize_x,fsize_y))
            ## Temp ##
            axs[0].plot(date_reform, surf_T)
            axs[0].set_title('Surface Temperature (ºC)')
            ## Salinity ##
            axs[1].plot(date_reform, surf_S)
            axs[1].set_title('Surface Salinity (PSU)')
            ## DOXY ##
            axs[2].plot(date_reform, surf_O)
            axs[2].set_title('Dissolved Oxygen (µmol/kg)')
            
            for ax in axs.flat:
                ax.label_outer()
            for tick in axs[2].get_xticklabels():
                tick.set_rotation(45)
            plt.xlabel('Date')
            plt.subplots_adjust(hspace=0.5)
            figs.suptitle('Surface Values for Float '+str(WMO))
            figs.subplots_adjust(bottom=0.18)
            plt.savefig(FigDir+str(WMO)+'_Surface_STO.jpg')
            plt.close()
            
            ####################################
            #### Interpolate data with time ###
            ###################################
            print('\nInterpolating...\n')
            
            dates_num=np.zeros(len(date_reform))
            dates6hr_num=np.zeros(len(date_6hr))
            
            # Convert times to numbers
            for i in np.arange(len(date_reform)):
                dates_num[i]=(date_reform[i]-refdate).total_seconds()
            for i in np.arange(len(dates6hr_num)):
                dates6hr_num[i]=(date_6hr_reform[i]-refdate).total_seconds()
        
            temp_interp=interpolate.interp1d(dates_num, surf_T)
            sal_interp=interpolate.interp1d(dates_num, surf_S)
            oxy_interp=interpolate.interp1d(dates_num, surf_O)
            lat_interp=interpolate.interp1d(dates_num, lat)
            lon_interp=interpolate.interp1d(dates_num, lon)
            
            surf_T_interp=temp_interp(dates6hr_num)
            surf_S_interp=sal_interp(dates6hr_num)
            surf_O_interp=oxy_interp(dates6hr_num)
            surf_lat_interp=lat_interp(dates6hr_num)
            surf_lon_interp=lon_interp(dates6hr_num)
            
            ## Make sure all data are in the correct lat-lon region
            RegionOI=[lat_N, lat_S, lon_E, lon_W]
            [surf_lat_interp,surf_lon_interp,surf_T_interp,surf_S_interp,surf_O_interp]=RF.PositionCheck(RegionOI=RegionOI, LatData=surf_lat_interp,LonData=surf_lon_interp,TempData=surf_T_interp, SalData=surf_S_interp,OxyData=surf_O_interp)
            
            # ## Plot interpolated surface values vs time
            figs, axs=plt.subplots(3,1,figsize=(10,8))
            ## Temp ##
            axs[0].plot(date_reform, surf_T,':', date_6hr, surf_T_interp,'-')
            axs[0].set_title('Surface Temperature (ºC)')
            #axs[0].legend(['Data','Interpolated'])
            ## Salinity ##
            axs[1].plot(date_reform, surf_S,':', date_6hr, surf_S_interp,'-')
            axs[1].set_title('Surface Salinity (PSU)')
            #axs[1].legend(['Data','Interpolated'])
            ## DOXY ##
            axs[2].plot(date_reform, surf_O,':', date_6hr, surf_O_interp,'-')
            axs[2].set_title('Dissolved Oxygen (µmol/kg)')
            #axs[2].legend(['Data','Interpolated'])
            
            for ax in axs.flat:
                ax.label_outer()
            for tick in axs[2].get_xticklabels():
                tick.set_rotation(45)
            
            plt.subplots_adjust(hspace=0.3)
            plt.xlabel('Date')
            plt.legend(['Data','Interpolated'],loc="upper center", bbox_to_anchor=(0.5, 3.96), ncol=2)
            figs.suptitle('Surface Values Interpolated for Float '+str(WMO))
            figs.subplots_adjust(bottom=0.12)
            plt.savefig(FigDir+str(WMO)+'_Surface_STO_Interpolated.jpg')
            plt.close()
            
            # Plot trajectory
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.plot(lon, lat,':', surf_lon_interp, surf_lat_interp,'-')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend(['Data','Interpolated'])
            plt.title('Trajectory Interpolated for Float '+str(WMO))
            plt.savefig(FigDir+str(WMO)+'_Trajectory_Interpolated.jpg')
            plt.close()
            
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
            plt.title('Trajectory for Float '+str(WMO))
            plt.plot(lon[0],lat[0],'go')
            plt.plot(lon[len(lon)-1],lat[len(lat)-1],'ro')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.plot(lon, lat)
            plt.savefig(FigDir+str(WMO)+'_Trajectory_NAtlantic.jpg')
            plt.savefig(FigDir_Traj+str(WMO)+'_Trajectory_NAtlantic.jpg')
            plt.close()
        
            
            ############################
            ### Find closes ERA data ###
            ############################
            
            # before calculating nearest neighbor, need to remove data that is outside 
            # data range
            
            
            ## to calculate the diffusive gas flux
            ## Need the following parameters:
            # C = surface dissolved gas concentration in mol/m^3
            # u10 = 10 m windspeed (m/s)
            # SP = PSS
            # T = Surface water temp (øC)
            # slp = sea level pressure (atm)
            
            # Convert surface oxygen concentration (µmol/kg) to mol/m^3
            
            # Calculate the density at each point
            surf_density = np.zeros(len(surf_T_interp))
            surf_density[:]=np.NaN
            
            P=surfP_val # dbar
            for i in np.arange(len(surf_density)):
                # Calculate absolute salinity
                SA=gsw.SA_from_SP(surf_S_interp[i], P, surf_lon_interp[i], surf_lat_interp[i])
                
                # Calculate conservative temp or temp
                T=surf_T_interp[i]
                
                #CT=gsw.CT_from_t(SA, T, P)
                
                # Calculate density
                dense=gsw.density.rho_t_exact(SA, T, P) # kg/m^3
                surf_density[i]=dense
            
            # Convert oxygen to appropriate units
            surf_O_interp_units = np.multiply(surf_O_interp, surf_density)*(10**-6)
            
            Fd_L13_float=np.zeros(len(date_6hr))
            Fd_L13_float[:]=np.NaN
            Fc_L13_float=np.zeros(len(date_6hr))
            Fc_L13_float[:]=np.NaN
            Fp_L13_float=np.zeros(len(date_6hr))
            Fp_L13_float[:]=np.NaN
            
            Fd_N16_float=np.zeros(len(date_6hr))
            Fd_N16_float[:]=np.NaN
            Fc_N16_float=np.zeros(len(date_6hr))
            Fc_N16_float[:]=np.NaN
            Fp_N16_float=np.zeros(len(date_6hr))
            Fp_N16_float[:]=np.NaN
            
            # For each point in time
            print('\nFinding Nearest Neighbor...\n')
            for i in np.arange(len(date_6hr)):
                
                # Make sure float has a valid postion
                float_lat=surf_lat_interp[i]
                float_lon=surf_lon_interp[i]
                
                # Make sure float has a valid postion
                if (np.isnan(float_lat) == False and np.isnan(float_lon) == False):
                    
                    #Find the matching date-time pair
                    profdate=date_6hr[i]
                    date_ind=np.where(era5_time==profdate)
                    
                    # Should add a check to make sure there is only 1 date ind
                    
                    # use index to get lat-lon slices and other variable
                    # wind speed structure (time,lat, lon)
                    
                    # load u_10 data
                    u10_slice=era_u10_data.sel(time=profdate).u10.values[0,:,:]
                    
                    # load v_10 data
                    v10_slice=era_v10_data.sel(time=profdate).v10.values[0,:,:]
                    
                    # load slp
                    slp_slice=era_slp_data.sel(time=profdate).msl.values[0,:,:]
                    
                    # Find the closest lat, lon point
        
                    # Use nearest neighbors to find nearest point
                    dd, ii = tree.query([float_lon,float_lat])
                    #print(ii)
                    
                    # Use ii get row column pairing 
                    # ravel unravels row by row 
                    
                    num_col=len(x)
                    lat_ind=ii//num_col
                    lon_ind=ii%num_col
                    
                    u10_point=u10_slice[lat_ind, lon_ind]
                    v10_point=v10_slice[lat_ind, lon_ind]
                    slp_point=slp_slice[lat_ind, lon_ind]
                    # convert SLP from Pa to atmosphere
                    slp_point=slp_point/101325
                    
                    ## Use nearest values and calculate air-sea flux at each point
                    c=surf_O_interp_units[i]
                    S=surf_S_interp[i]
                    U10=(u10_point**2 + v10_point**2)**.5
                    T=surf_T_interp[i]
                    SLP=slp_point
                    
                    ###########################################
                    ## L13 -- ISSUE with wrapper and ret ?? ##
                    #def L13(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None):
                        
                    [Fd_L13, Fc_L13, Fp_L13, Deq_L13, k_L13]=AS.L13(C=c,u10=U10, SP=S, pt=T, slp=SLP, gas='O2', rh=1)
                    Fd_L13_float[i]=Fd_L13
                    Fc_L13_float[i]=Fc_L13
                    Fp_L13_float[i]=Fp_L13
                    
                    # Outputs 
                    # Fd = surface gas flux mmol/m^2-s
                    # Fc = flyx from fully collapsing large bubbles
                    # Fp = flux from partially collapsing large bubbles
                    # Deq = equilibirum saturation ( %sat/100)
                    # k = diffusive gas traner velocity 
                    # Ft = Fd + Fc + Fp
                    
                    ##############
                    ## N16 #######
                    #def N16(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None):
                    [Fd_N16, Fc_N16, Fp_N16, Deq_N16, k__N16]=AS.N16(C=c,u10=U10, SP=S, pt=T, slp=SLP, gas='O2', rh=1)
                    Fd_N16_float[i]=Fd_N16
                    Fc_N16_float[i]=Fc_N16
                    Fp_N16_float[i]=Fp_N16
                         
            print('\nCalculating Total Air-Sea Flux...\n')
            Ft_L13_float=Fd_L13_float+Fc_L13_float+Fp_L13_float
            Ft_N16_float=Fd_N16_float+Fc_N16_float+Fp_N16_float
            
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.plot(date_6hr_reform, Ft_L13_float,date_6hr_reform, Fd_L13_float,date_6hr_reform, Fc_L13_float,date_6hr_reform, Fp_L13_float)
            plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Flux (mmol/m^2-s)')
            plt.title('Air-Sea Oxygen Flux (L13) for Float '+str(WMO))
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_L13_Total_RAW.jpg')
            plt.close() 
            
            plt.figure(figsize=(fsize_x,fsize_y))
            #plt.plot(date_6hr_reform, Ft_L13_float,date_6hr_reform, Fd_L13_float,date_6hr_reform, Fc_L13_float,date_6hr_reform, Fp_L13_float)
            plt.plot(date_6hr_reform, Ft_N16_float,date_6hr_reform, Fd_N16_float,date_6hr_reform, Fc_N16_float,date_6hr_reform, Fp_N16_float)
            plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Flux (mmol/m^2-s)')
            plt.title('Air-Sea Oxygen Flux (N16) for Float '+str(WMO))
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_N16_Total_RAW.jpg')
            plt.close() 
            
            ##############################
            ## Calculate moving average ##
            ##############################
            
            print('Calculating Moving Averages...')
            ## Surface Gas Flux ##
            df_Fd_L13=pd.DataFrame({'Fd_L13': Fd_L13_float})
            Fd_L13_MA_24hr=df_Fd_L13.rolling(int((24/6)),min_periods=1).mean()
            Fd_L13_MA_1wk=df_Fd_L13.rolling(int((7*24)/6),min_periods=1).mean()
            
            df_Fd_N16=pd.DataFrame({'Fd_N16': Fd_N16_float})
            Fd_N16_MA_24hr=df_Fd_N16.rolling(int((24/6)),min_periods=1).mean()
            Fd_N16_MA_1wk=df_Fd_N16.rolling(int((7*24)/6),min_periods=1).mean()
            
            ## Small bubble gas flux ##
            df_Fc_L13=pd.DataFrame({'Fd_L13': Fc_L13_float})
            Fc_L13_MA_24hr=df_Fc_L13.rolling(int((24/6)),min_periods=1).mean()
            Fc_L13_MA_1wk=df_Fc_L13.rolling(int((7*24)/6),min_periods=1).mean()
            
            df_Fc_N16=pd.DataFrame({'Fd_N16': Fc_N16_float})
            Fc_N16_MA_24hr=df_Fc_N16.rolling(int((24/6)),min_periods=1).mean()
            Fc_N16_MA_1wk=df_Fc_N16.rolling(int((7*24)/6),min_periods=1).mean()
            
            ## Large bubble gas flux ##
            df_Fp_L13=pd.DataFrame({'Fd_L13': Fp_L13_float})
            Fp_L13_MA_24hr=df_Fp_L13.rolling(int((24/6)),min_periods=1).mean()
            Fp_L13_MA_1wk=df_Fp_L13.rolling(int((7*24)/6),min_periods=1).mean()
            
            df_Fp_N16=pd.DataFrame({'Fd_N16': Fp_N16_float})
            Fp_N16_MA_24hr=df_Fp_N16.rolling(int((24/6)),min_periods=1).mean()
            Fp_N16_MA_1wk=df_Fp_N16.rolling(int((7*24)/6),min_periods=1).mean()
            
            ## Total Air-Sea Gas Flux ##
            Ft_L13_MA_24hr=Fd_L13_MA_24hr+Fc_L13_MA_24hr+Fp_L13_MA_24hr
            Ft_L13_MA_1wk=Fd_L13_MA_1wk+Fc_L13_MA_1wk+Fp_L13_MA_1wk
            
            Ft_N16_MA_24hr=Fd_N16_MA_24hr+Fc_N16_MA_24hr+Fp_N16_MA_24hr
            Ft_N16_MA_1wk=Fd_N16_MA_1wk+Fc_N16_MA_1wk+Fp_N16_MA_1wk
            
            # Break down by component
            ## L13 ## 
            figs, axs=plt.subplots(4,1,figsize=(10,9))
            ## Surface Flux ##
            axs[0].plot(date_6hr_reform, Fd_L13_float, date_6hr_reform, Fd_L13_MA_24hr,date_6hr_reform,Fd_L13_MA_1wk)
            axs[0].set_title('Surface Gas Flux (mmol/m^2-s)')
            #axs[0].legend(['Raw','24-hr average','1-week average'])
            ## Small Bubble ##
            axs[1].plot(date_6hr_reform, Fc_L13_float, date_6hr_reform, Fc_L13_MA_24hr,date_6hr_reform,Fc_L13_MA_1wk)
            axs[1].set_title('Small Bubble Gas Flux (mmol/m^2-s)')
            #axs[1].legend(['Raw','24-hr average','1-week average'])
            
            ## Large Bubble ##
            axs[2].plot(date_6hr_reform, Fp_L13_float, date_6hr_reform, Fp_L13_MA_24hr,date_6hr_reform,Fp_L13_MA_1wk)
            axs[2].set_title('Large Bubble Gas Flux (mmol/m^2-s)')
            #axs[2].legend(['Raw','24-hr average','1-week average'])
            ## Total ##
            axs[3].plot(date_6hr_reform, Ft_L13_float, date_6hr_reform, Ft_L13_MA_24hr,date_6hr_reform,Ft_L13_MA_1wk)
            axs[3].set_title('Total Air-Sea Gas Flux (mmol/m^2-s)')
            #axs[3].legend(['Raw','24-hr average','1-week average'])
            
            for ax in axs.flat:
                ax.label_outer()
            for tick in axs[3].get_xticklabels():
                tick.set_rotation(45)
            
            plt.subplots_adjust(hspace=0.5)
            plt.xlabel('Date')
            #plt.legend(['Raw','24-hr average','1-week average'],bbox_to_anchor=(1.05,2.5), loc="lower left", borderaxespad=0)
            plt.legend(['Raw','24-hr average','1-week average'],loc="upper center", bbox_to_anchor=(0.5, 6), ncol=3)
            figs.suptitle('Air-Sea Oxygen Flux (L13) by Component for Float '+str(WMO))
            figs.subplots_adjust(bottom=0.1)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_L13_Component_Average.jpg')
            plt.close()
            
            ## N16 ##
            figs, axs=plt.subplots(4,1,figsize=(10,9))
            ## Surface Flux ##
            axs[0].plot(date_6hr_reform, Fd_N16_float, date_6hr_reform, Fd_N16_MA_24hr,date_6hr_reform,Fd_N16_MA_1wk)
            axs[0].set_title('Surface Gas Flux (mmol/m^2-s)')
            #axs[0].legend(['Raw','24-hr average','1-week average'])
            ## Small Bubble ##
            axs[1].plot(date_6hr_reform, Fc_N16_float, date_6hr_reform, Fc_N16_MA_24hr,date_6hr_reform,Fc_N16_MA_1wk)
            axs[1].set_title('Small Bubble Gas Flux (mmol/m^2-s)')
            #axs[1].legend(['Raw','24-hr average','1-week average'])
            
            ## Large Bubble ##
            axs[2].plot(date_6hr_reform, Fp_N16_float, date_6hr_reform, Fp_N16_MA_24hr,date_6hr_reform,Fp_N16_MA_1wk)
            axs[2].set_title('Large Bubble Gas Flux (mmol/m^2-s)')
            #axs[2].legend(['Raw','24-hr average','1-week average'])
            ## Total ##
            axs[3].plot(date_6hr_reform, Ft_N16_float, date_6hr_reform, Ft_N16_MA_24hr,date_6hr_reform,Ft_N16_MA_1wk)
            axs[3].set_title('Total Air-Sea Gas Flux (mmol/m^2-s)')
            #axs[3].legend(['Raw','24-hr average','1-week average'])
            
            for ax in axs.flat:
                ax.label_outer()
            for tick in axs[3].get_xticklabels():
                tick.set_rotation(45)
            
            plt.subplots_adjust(hspace=0.5)
            plt.xlabel('Date')
            #plt.legend(['Raw','24-hr average','1-week average'],bbox_to_anchor=(1.05,2.5), loc="lower left", borderaxespad=0)
            plt.legend(['Raw','24-hr average','1-week average'],loc="upper center", bbox_to_anchor=(0.5, 6), ncol=3)
            figs.suptitle('Air-Sea Oxygen Flux (N16) by Component for Float '+str(WMO))
            figs.subplots_adjust(bottom=0.1)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_N16_Component_Average.jpg')
            plt.close()
            
            ## 24 hr moving average ## 
            ## L13 ##
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.plot(date_6hr_reform, Ft_L13_MA_24hr,date_6hr_reform, Fd_L13_MA_24hr,date_6hr_reform, Fc_L13_MA_24hr,date_6hr_reform, Fp_L13_MA_24hr)
            plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Flux (mmol/m^2-s)')
            plt.title('Air-Sea Oxygen Flux (L13) with 24-hr Average for Float '+str(WMO), y=1.05)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_L13_Total_24hr.jpg')
            plt.close() 
            
            ## N16 ##
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.plot(date_6hr_reform, Ft_N16_MA_24hr,date_6hr_reform, Fd_N16_MA_24hr,date_6hr_reform, Fc_N16_MA_24hr,date_6hr_reform, Fp_N16_MA_24hr)
            plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Flux (mmol/m^2-s)')
            plt.title('Air-Sea Oxygen Flux (N16) with 24-hr Average for Float '+str(WMO), y=1.05)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_N16_Total_24hr.jpg')
            plt.close() 
            
            ## 1 week moving average ##
            
            ## L13 ##
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.plot(date_6hr_reform, Ft_L13_MA_1wk,date_6hr_reform, Fd_L13_MA_1wk,date_6hr_reform, Fc_L13_MA_1wk,date_6hr_reform, Fp_L13_MA_1wk)
            plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Flux (mmol/m^2-s)')
            plt.title('Air-Sea Oxygen Flux (L13) with 1-week Average for Float '+str(WMO),y=1.05)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_L13_Total_1week.jpg')
            plt.close()
            
            ## N16 ##
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.plot(date_6hr_reform, Ft_N16_MA_1wk,date_6hr_reform, Fd_N16_MA_1wk,date_6hr_reform, Fc_N16_MA_1wk,date_6hr_reform, Fp_N16_MA_1wk)
            plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Flux (mmol/m^2-s)')
            plt.title('Air-Sea Oxygen Flux (N16) with 1-week Average for Float '+str(WMO),y=1.05)
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_N16_Total_1week.jpg')
            plt.close()
            
            flux_count[b]=1
            
            WMO_df=np.zeros(len(date_6hr_reform))
            WMO_df[:]=WMO
            TotalAirSea=pd.DataFrame({'WMO':WMO_df,'Date': date_6hr_reform,'Lat': surf_lat_interp, 'Lon': surf_lon_interp,'Fp_L13': Fp_L13_float,'Fc_L13': Fc_L13_float,'Fd_L13': Fd_L13_float, 'Ft_L13': Ft_L13_float,
                                      'Fp_N16': Fp_N16_float,'Fc_N16': Fc_N16_float,'Fd_N16': Fd_N16_float, 'Ft_N16': Ft_N16_float})
            TotalAirSea.to_csv(CSVDir_AS+str(WMO)+'.csv')
        
        if depth_time_interp == 1:
            ###### DEPTH INTERPOLATION ###########
            pres_range=np.arange(step_P,maxP_sec+1,step_P)
            temp_interp_p=np.zeros((len(pres_range),pres.shape[0]))
            temp_interp_p[:]=np.NaN
            sal_interp_p=np.zeros((len(pres_range),pres.shape[0]))
            sal_interp_p[:]=np.NaN
            doxy_interp_p=np.zeros((len(pres_range),pres.shape[0]))
            doxy_interp_p[:]=np.NaN
            
            # Depth and time interpolated variables
            temp_pt=np.zeros((len(pres_range), len(date_6hr)))
            temp_pt[:]=np.NaN
            sal_pt=np.zeros((len(pres_range), len(date_6hr)))
            sal_pt[:]=np.NaN
            oxy_pt=np.zeros((len(pres_range), len(date_6hr)))
            oxy_pt[:]=np.NaN
    
            ######################################
            for i in np.arange(len(pres)):
                
                ## Depth interpolation ##
                temp_f=interpolate.interp1d(pres[i], temp[i])
                sal_f=interpolate.interp1d(pres[i], sal[i])
                doxy_f=interpolate.interp1d(pres[i],doxy[i])
                
                # Not every float goes as deep/shallow as specified pressure range
                # Go through and find shallowest and deepest profile measurements
                # and use subsection of pres_range so interpolation works
                # interpolate to desired pressure range
                
                ## get subset of press range
                minP_sub=np.nanmin(pres[i])
                maxP_sub=np.nanmax(pres[i])
                
                if minP_sub%step_P == 0:
                    startind_P=int(minP_sub//step_P)+1
                else:
                    startind_P=int(minP_sub//step_P)
                
                if maxP_sub >= maxP_sec:
                    endind_P=len(pres_range)+1
                else:
                    endind_P=int(maxP_sub//step_P)
                
                temp_interp_p[startind_P:endind_P,i]=temp_f(pres_range[startind_P:endind_P])
                sal_interp_p[startind_P:endind_P,i]=sal_f(pres_range[startind_P:endind_P])
                doxy_interp_p[startind_P:endind_P,i]=doxy_f(pres_range[startind_P:endind_P])
            
            # for each pressure level interpolate with time
            for i in np.arange(pres_range.shape[0]):
                level_temp=temp_interp_p[i,:]
                level_sal=sal_interp_p[i,:]
                level_oxy=doxy_interp_p[i,:]
                
                level_temp_interp=interpolate.interp1d(dates_num, level_temp)
                level_sal_interp=interpolate.interp1d(dates_num, level_sal)
                level_oxy_interp=interpolate.interp1d(dates_num, level_oxy)
                
                t_pt=level_temp_interp(dates6hr_num)
                s_pt=level_sal_interp(dates6hr_num)
                o_pt=level_oxy_interp(dates6hr_num)
                
                temp_pt[i,:]=t_pt
                sal_pt[i,:]=s_pt
                oxy_pt[i,:]=o_pt
            
            
            X, Y = np.meshgrid(date_6hr_reform, pres_range)
            
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.contourf(X,Y,temp_pt)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Pressure (dbar)')
            plt.title('Temperature Time Series for Float '+str(WMO))
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_PresDateSection_Temp.jpg')
            plt.close()
            
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.contourf(X,Y, sal_pt)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Pressure (dbar)')
            plt.title('Salinity Time Series for Float '+str(WMO))
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_PresDateSection_Sal.jpg')
            plt.close()
            
            plt.figure(figsize=(fsize_x,fsize_y))
            plt.contourf(X,Y,oxy_pt)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Pressure (dbar)')
            plt.title('Oxygen Time Series for Float '+str(WMO))
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(FigDir+str(WMO)+'_PresDateSection_Oxy.jpg')
            plt.close()
            
            plt.show()

    else:
        print('This float does not measure oxygen')
        
    toc_float=time.time()
    print('Time elapsed (s): ', toc_float-tic_float)
    print('Time elapsed (min): ', (toc_float-tic_float)/60)

print('\n%% Saving adjusted variable flags %%\n')
adj_df=pd.DataFrame({'FloatWMO': floatlist, 'Pres_Adj': pres_flag_total,'Temp_Adj': temp_flag_total,'Sal_Adj': sal_flag_total, 'Oxy_Adj':oxy_flag_total,'Flux_Calc': flux_count})
adj_df=adj_df.sort_values(by=['FloatWMO'])
adj_df.to_csv(CSVDir)

toc_end=time.time()
print('\n%%Total Time%%\n')
print('Time elapsed (s): ', toc_end-tic_start)
print('Time elapsed (min): ', (toc_end-tic_start)/60)
print('Time elapsed (hr): ', (toc_end-tic_start)/(60*60))

        