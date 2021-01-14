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
import glob


floatlist=[4900610]#,1900652,5902299]   
refdate=datetime(1990,1,1)
timestep=6 # hours
freq_str=str(timestep)+'H'
surfP_val=5    # dbar
surfP_var=3  # dbar

minsurfP=surfP_val-surfP_var
maxsurfP=surfP_val+surfP_var

## Directory information
FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Test/'

# Open ERA5 Reanalysis Data
print('\n%%% Loading ERA5 Data Files%%%\n')
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

##Change these for different files
era5_time=era_u10_data.time.values

#####################################
## USES UP TOO MUCH MEMORY##
# print('U10')
# era5_u10=era_u10_data.u10.values
# print('V10')
# era5_v10=era_v10_data.v10.values
# print('SLP')
# era5_slp=era_slp_data.msl.values

# ## Check that lat/lon/time are the same
# u_time=era_u10_data.time.values
# v_time=era_v10_data.time.values
# slp_time=era_slp_data.time.values
# print('Time check')
# print(u_time==v_time)
# print(slp_time==v_time)

# u_lat=era_u10_data.latitude.values
# v_lat=era_v10_data.latitude.values
# slp_lat=era_slp_data.latitude.values
# print('Lat check')
# print(u_lat==v_lat)
# print(slp_lat==v_lat)

######################################
#print('Lon')
x=era_u10_data.longitude.values
#print('Lat')
y=era_u10_data.latitude.values
x_grid, y_grid= np.meshgrid(x,y)
print('Making Nearest Neighbor Tree')
tree = KDTree(np.c_[x_grid.ravel(), y_grid.ravel()])

for WMO in floatlist:
    print('\n%%% '+str(WMO)+' %%%\n')
    BGCfile='/Users/Ellen/Desktop/ArgoGDAC/dac/coriolis/'+str(WMO)+'/'+str(WMO)+'_Sprof.nc'
    f = xr.open_dataset(BGCfile)
    
    reft=f.REFERENCE_DATE_TIME.values
    #print(reft)
    dates=f.JULD.values
    #print(dates[0])
    #print(datetime.fromisoformat(str(dates[0])[:-3]))
    
    date_reform=[[]]*dates.shape[0]
    date_reform_str=[[]]*dates.shape[0]
    for i in np.arange(len(date_reform)):
        date_reform[i]=datetime.fromisoformat(str(dates[i])[:-3])
        date_reform_str[i]=str(date_reform[i])
    
    #date_reform=pd.DataFrame(date_reform)
    mindate=date_reform[0]
    #print(mindate)
    
    minhour=mindate.hour
    if minhour%timestep != 0:
        new_minhour=( minhour//timestep+1)*timestep
        mindate_roundup=datetime(mindate.year, mindate.month, mindate.day,new_minhour,0, 0)
    elif minhour%timestep == 0:
        if (mindate.minute == 0 and mindate.second ==0):
            mindate_roundup=mindate
        else:
            new_minhour=( minhour//timestep+1)*timestep
            mindate_roundup=datetime(mindate.year, mindate.month, mindate.day,new_minhour,0, 0)
    #print(mindate_roundup)
    
    maxdate=date_reform[-1]
    #print(maxdate)
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
    #print(maxdate_rounddown) 
    
    date_6hr=np.arange(mindate_roundup, maxdate_rounddown,timestep*60*60,dtype='datetime64[s]')
    date_6hr_pd=pd.date_range(mindate_roundup,maxdate_rounddown,freq=freq_str)
    
    # Convert numpy time to datetime
    date_6hr_reform=[[]]*len(date_6hr)
    date_6hr_reform_str=[[]]*len(date_6hr)
    for i in np.arange(len(date_6hr_reform)):
        date_6hr_reform[i]=datetime.fromisoformat(str(date_6hr[i]))
        date_6hr_reform_str[i]=str(date_6hr_reform[i])
      
    ###### DEPTH INTERPOLATION ###########
    # pres_range=np.arange(0,2001,10)
    # temp_interp_p=np.zeros((pres.shape))
    # temp_f=interpolate.interp1d(pres, temp)
    # sal_f=interpolate.interp1d(pres, sal)
    # doxy_f=interpolate.interp1d(pres,doxy)
    ######################################
    
    ##############################
    #### Get surface values #######
    ##############################
    
    ## NOTE! Change this for adjusted, etc.
    ## Write and use a new function
    pres=f.PRES.values
    temp=f.TEMP.values
    sal=f.PSAL.values
    doxy=f.DOXY.values
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
    print('Getting surface values...')
    for i in np.arange(pres.shape[1]):
        ## Determine what values fall in the given pressure range and save index
        j=0
        pressure_check=0
        pres_ind=[]
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
            
            j=j+1
                
        # Use pressure indeces and calculate average surface T, S, and O for each profile
        if pres_ind==[]:
            surf_T[i]=np.NaN
            surf_S[i]=np.NaN
            surf_O[i]=np.NaN
        else:
            minind=pres_ind[0]
            maxind=len(pres_ind)
            
            # Subset of surface values
            T_sub=temp[i][minind:maxind]
            S_sub=sal[i][minind:maxind]
            O_sub=doxy[i][minind:maxind]
            
            # Use mean values
            surf_T[i]=np.nanmean(T_sub)
            surf_S[i]=np.nanmean(S_sub)
            surf_O[i]=np.nanmean(O_sub)
    
    ## Plot surface values vs time
    figs, axs=plt.subplots(3,1)
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
        
    plt.subplots_adjust(hspace=0.5)
    figs.suptitle('Surface Values for Float '+str(WMO))
    figs.subplots_adjust(bottom=0.2)
    plt.savefig(FigDir+str(WMO)+'_Surface_STO.jpg')
    plt.close()
    
    ####################################
    #### Interpolate data with time ###
    ###################################
    print('Interpolating...')
    
    dates_num=np.zeros(len(dates))
    dates6hr_num=np.zeros(len(date_6hr))
    
    # Convert times to numbers
    for i in np.arange(len(dates)):
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
    
    
    # ## Plot interpolated surface values vs time
    figs, axs=plt.subplots(3,1)
    ## Temp ##
    axs[0].plot(date_reform, surf_T,':', date_6hr, surf_T_interp,'-')
    axs[0].set_title('Surface Temperature (ºC)')
    axs[0].legend(['Data','Interpolated'])
    ## Salinity ##
    axs[1].plot(date_reform, surf_S,':', date_6hr, surf_S_interp,'-')
    axs[1].set_title('Surface Salinity (PSU)')
    axs[1].legend(['Data','Interpolated'])
    ## DOXY ##
    axs[2].plot(date_reform, surf_O,':', date_6hr, surf_O_interp,'-')
    axs[2].set_title('Dissolved Oxygen (µmol/kg)')
    axs[2].legend(['Data','Interpolated'])
    
    for ax in axs.flat:
        ax.label_outer()
    for tick in axs[2].get_xticklabels():
        tick.set_rotation(45)
    
    plt.subplots_adjust(hspace=0.5)
    figs.suptitle('Surface Values Interpolated for Float '+str(WMO))
    figs.subplots_adjust(bottom=0.2)
    plt.savefig(FigDir+str(WMO)+'_Surface_STO_Interpolated.jpg')
    plt.close()
    
    # Plot trajectory
    plt.figure()
    plt.plot(lon, lat,':', surf_lon_interp, surf_lat_interp,'-')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(['Data','Interpolated'])
    plt.title('Trajectory Interpolated for Float '+str(WMO))
    plt.savefig(FigDir+str(WMO)+'_Trajectory_Interpolated.jpg')
    plt.close()

    
    ############################
    ### Find closes ERA data ###
    ############################
    
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
    print('Finding Nearest Neighbor...')
    for i in np.arange(len(date_6hr)):
        
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
        float_lat=surf_lat_interp[i]
        float_lon=surf_lon_interp[i]
        
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
            
        # [Fd_L13, Fc_L13, Fp_L13, Deq_L13, k_L13]=AS.L13(C=c,u10=U10, SP=S, pt=T, slp=SLP, gas='O2', rh=1)
        # Fd_L13_float[i]=Fd_L13
        # Fc_L13_float[i]=Fc_L13
        # Fp_L13_float[i]=Fp_L13
        
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
    
    print('Calculating Total Air-Sea Flux...')
    #Ft_L13_float=Fd_L13_float+Fc_L13_float+Fp_L13_float
    Ft_N16_float=Fd_N16_float+Fc_N16_float+Fp_N16_float
    
    plt.figure()
    #plt.plot(date_6hr_reform, Ft_L13_float,date_6hr_reform, Fd_L13_float,date_6hr_reform, Fc_L13_float,date_6hr_reform, Fp_L13_float)
    plt.plot(date_6hr_reform, Ft_N16_float,date_6hr_reform, Fd_N16_float,date_6hr_reform, Fc_N16_float,date_6hr_reform, Fp_N16_float)
    plt.legend(['Total Air-Sea Flux','Surface Gas Flux','Small Bubble Flux','Large Bubble Flux'])
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Flux (mmol/m^2-s)')
    plt.title('Air-Sea Oxygen Flux (N16) for Float '+str(WMO))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(FigDir+str(WMO)+'_AirSeaFlux_Oxy_N16.jpg')
    plt.close()

        