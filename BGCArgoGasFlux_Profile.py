#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:44:58 2021

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
import BCClassFxns as BCF
import time
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)

# Read in float list
# Get FloatID
tic1=time.time()

Argofiles='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/DacWMO_NAtlantic.txt'

ArgoDACWMO=[]
ArgoDac=[]
ArgoWMO=[]
count=0

with open(Argofiles) as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        x=line.strip()
        ArgoDACWMO=ArgoDACWMO + [x]
        t=x.split('/')
        ArgoDac=ArgoDac+[t[0]]
        ArgoWMO=ArgoWMO+[t[1]]
        
# Load float data

MODEL_TYPE=1
# 0: BC by Shape
# 1: BC by bathymetry

if MODEL_TYPE==0:
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Shape/'
    print('\nUSING SHAPE MODEL\n')
elif MODEL_TYPE == 1:
    CSVDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/'
    print('\nUSING BATHYMETRY MODEL\n')
    
#####################
## Some parameters ##
#####################
good_QC=[1,2,5,8]
lat_N=80.000
lat_S= 40.00
lon_E= -30.00
lon_W= -70.00

# Surface pressure range used to calculate air-sea flux
surfP_val=5    # dbar
surfP_var=3  # dbar

minsurfP=surfP_val-surfP_var
maxsurfP=surfP_val+surfP_var

P = surfP_val

timestep=6 # hours --> matches ERA5 data spacing
##########################
#########################

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

#####################
#####################

pres_flag_total=np.zeros(len(ArgoDACWMO))
pres_flag_total[:]=np.NaN
temp_flag_total=np.zeros(len(ArgoDACWMO))
temp_flag_total[:]=np.NaN
sal_flag_total=np.zeros(len(ArgoDACWMO))
sal_flag_total[:]=np.NaN
oxy_flag_total=np.zeros(len(ArgoDACWMO))
oxy_flag_total[:]=np.NaN
flux_count=np.zeros(len(ArgoDACWMO))
flux_count[:]=np.NaN

BCData=pd.DataFrame({'FloatWMO': [np.NaN], 'ProfNum': [np.NaN],'Date':[np.NaN],'Date_6hr':[np.NaN],'Lat':[np.NaN],'Lon':[np.NaN],'MLD': [np.NaN], 'Surf_T': [np.NaN],'Surf_S': [np.NaN],'Surf_O': [np.NaN],
                     'Oxy_Sat': [np.NaN], 'Oxy_Dev': [np.NaN],'k_L13': [np.NaN],'Fp_L13': [np.NaN],'Fc_L13': [np.NaN],'Fd_L13': [np.NaN], 'Ft_L13': [np.NaN],
                     'k_N16': [np.NaN],'Fp_N16': [np.NaN],'Fc_N16': [np.NaN],'Fd_N16': [np.NaN], 'Ft_N16': [np.NaN]})
GyreData=pd.DataFrame({'FloatWMO': [np.NaN], 'ProfNum': [np.NaN],'Date':[np.NaN],'Date_6hr':[np.NaN],'Lat':[np.NaN],'Lon':[np.NaN],'MLD': [np.NaN], 'Surf_T': [np.NaN],'Surf_S': [np.NaN],'Surf_O': [np.NaN],
                     'Oxy_Sat': [np.NaN], 'Oxy_Dev': [np.NaN],'k_L13': [np.NaN],'Fp_L13': [np.NaN],'Fc_L13': [np.NaN],'Fd_L13': [np.NaN], 'Ft_L13': [np.NaN],
                     'k_N16': [np.NaN],'Fp_N16': [np.NaN],'Fc_N16': [np.NaN],'Fd_N16': [np.NaN], 'Ft_N16': [np.NaN]})

for i in np.arange(len(ArgoWMO)):
#for i in [124]:
    tic2=time.time()
    
    dac=ArgoDac[i]
    WMO=ArgoWMO[i]
    
    print('\n%%% ',WMO,' %%%\n')
    print(i, ' Floats completed; ', len(ArgoWMO)-i,' Floats Left')
    f = RF.ArgoDataLoader(DAC=dac, WMO=WMO)
    
    # Make sure float measures oxygen
    float_vars=list(f.keys())
    float_vars=np.array(float_vars)
    oxy_check = np.where(float_vars=='DOXY')
    
    flux_profcount=0
    
    if oxy_check[0].size != 0:

        # Load float data and determine if use adjusted or not adjusted data
        [pres, pres_QC, pres_flag]=RF.DetermineAdjusted(raw_data=f.PRES.values,raw_data_QC=f.PRES_QC.values,a_data=f.PRES_ADJUSTED.values, a_data_QC=f.PRES_ADJUSTED_QC.values)
        [temp, temp_QC, temp_flag]=RF.DetermineAdjusted(raw_data=f.TEMP.values, raw_data_QC= f.TEMP_QC.values,a_data=f.TEMP_ADJUSTED.values, a_data_QC=f.TEMP_ADJUSTED_QC.values)
        [sal, sal_QC, sal_flag]=RF.DetermineAdjusted(raw_data=f.PSAL.values, raw_data_QC=f.PSAL_QC.values,a_data=f.PSAL_ADJUSTED.values,a_data_QC=f.PSAL_ADJUSTED_QC.values)
        [doxy, doxy_QC, doxy_flag]=RF.DetermineAdjusted(raw_data=f.DOXY.values, raw_data_QC=f.DOXY_QC.values,a_data=f.DOXY_ADJUSTED.values,a_data_QC=f.DOXY_ADJUSTED_QC.values)
        
        # Store flags for adjusted data
        pres_flag_total[i]=pres_flag
        temp_flag_total[i]=temp_flag
        sal_flag_total[i]=sal_flag
        oxy_flag_total[i]=doxy_flag
        
        ## Quality control float data ##
        pres=RF.ArgoQC(Data=pres, Data_QC=pres_QC, goodQC_flags=good_QC)
        temp=RF.ArgoQC(Data=temp, Data_QC=temp_QC, goodQC_flags=good_QC)
        sal=RF.ArgoQC(Data=sal, Data_QC=sal_QC, goodQC_flags=good_QC)
        doxy=RF.ArgoQC(Data=doxy, Data_QC=doxy_QC, goodQC_flags=good_QC)
        
        lat=f.LATITUDE.values
        lon=f.LONGITUDE.values
        
        dates=f.JULD.values
        
        date_reform=[[]]*dates.shape[0]
        bad_index=[]
        
        for k in np.arange(len(date_reform)):
            if np.isnat(dates[k]) == False:
                date_reform[k]=datetime.fromisoformat(str(dates[k])[:-3])
            else:
                date_reform[k]=dates[k]
                bad_index=bad_index+[k]
        
        # For each profile, determine if data point is in BC, Gyre, or N/A
        for j in np.arange(len(lat)):
            
            # Make sure data point is not a bad index 
            bad_check = 0
            if len(bad_index)>0:
                for b_i in bad_index:
                    if b_i == j:
                        bad_check =1
             
            if bad_check == 0:          
                # Make sure profile is in the generally correct region and a valid position
                if (lat[j]<=lat_N and lat[j]>=lat_S and lon[j]>=lon_W and lon[j]<=lon_E and np.isnan(lat[j]) == False and np.isnan(lon[j]) == False):
                    # Detetmine if data point is BC, Gyre, N/A
                    if MODEL_TYPE==0:
                        bc_flag=BCF.BoundaryCurrent_Shape(Lon=lon[j], Lat=lat[j])
                    elif MODEL_TYPE==1:
                        bc_flag=BCF.BoundaryCurrent_Bath(Lon=lon[j], Lat=lat[j])
                    
                    if np.isnan(bc_flag)==False:
                        # If it is within one of the two shapes, proceed with analyis
                        # Get surface data
                        prof_pres=pres[j]
                        
                        ##############################################
                        # Increase search range and add MLD check!! ##
                        ##############################################
                        MLD=np.NaN
                        
                        pres_ind=RF.SurfacePIndex(pres=prof_pres, minP=minsurfP, maxP=maxsurfP)
                        
                        if pres_ind!=[]:
                            # Get average surface values
                            if len(pres_ind)==1:
                                surf_T=temp[j][pres_ind[0]]
                                surf_S=sal[j][pres_ind[0]]
                                surf_O=doxy[j][pres_ind[0]]
                            else:
                                minind=pres_ind[0]
                                maxind=pres_ind[len(pres_ind)-1]+1
                                
                                # Subset of surface values
                                T_sub=temp[j][minind:maxind]
                                S_sub=sal[j][minind:maxind]
                                O_sub=doxy[j][minind:maxind]
                            
                                # Use mean values
                                if len(T_sub) == 1:
                                    surf_T=T_sub
                                else:
                                    surf_T=np.nanmean(T_sub)
                                    
                                if len(S_sub)==1:
                                    surf_S=S_sub
                                else:
                                    surf_S=np.nanmean(S_sub)
                                
                                if len(O_sub) == 1:
                                    surf_O=O_sub
                                else:
                                    surf_O=np.nanmean(O_sub)
                            
                            # If surface data exists...
                            # If all of the surface values are nans, do not make figures or do calculations
                            if (np.isnan(surf_T)==False and np.isnan(surf_S)==False and np.isnan(surf_O)==False):
                                # Find nearest 6:00 time
                                prof_date=date_reform[j]
                                prof_hour=prof_date.hour
                                
                                ## Data only goes to 2020-12-31 18:000
                                if prof_date.year != 2021:
                                    # Date: Round Up
                                    if (prof_hour%timestep == 0 and prof_date.minute == 0 and prof_date.second ==0):
                                        roundup_date=np.NaN
                                        rounddown_date=np.NaN
                                        close_date=prof_date
                                    else:
                                        new_minhour=(prof_hour//timestep+1)*timestep
                                        
                                        if new_minhour == 24:
                                            # get last day of the month
                                            last_day = RF.last_day_of_month(prof_date)
                    
                                            if prof_date.day == last_day:
                                                if prof_date.month == 12:
                                                    roundup_date=datetime(prof_date.year+1, 1, 1, 0, 0, 0 )
                                                else:
                                                    roundup_date=datetime(prof_date.year, prof_date.month+1, 1, 0, 0, 0 )
                                            else:
                                                roundup_date=datetime(prof_date.year, prof_date.month, prof_date.day+1,0,0, 0)
                                        else:
                                            roundup_date=datetime(prof_date.year, prof_date.month, prof_date.day,new_minhour,0, 0)
        
                                        # Date: Round Down
                                        if prof_hour%timestep != 0:
                                            new_maxhour=(prof_hour//timestep)*timestep
                                            rounddown_date=datetime(prof_date.year, prof_date.month, prof_date.day,new_maxhour,0, 0)
                                        elif prof_hour%timestep == 0:
                                            if (prof_date.minute == 0 and prof_date.second ==0):
                                                rounddown_date=prof_date
                                            else:
                                                new_maxhour=(prof_hour//timestep)*timestep
                                                rounddown_date=datetime(prof_date.year, prof_date.month, prof_date.day,new_maxhour,0, 0)
                                    
                                        # Determine if the roundup or rounddown date is closer
                                        rd_dif=abs((prof_date-rounddown_date).total_seconds())
                                        ru_dif=abs((prof_date-roundup_date).total_seconds())
                                        
                                        if ru_dif<=rd_dif:
                                            close_date=roundup_date
                                        else:
                                            close_date=rounddown_date
                                    
                                    # Unit Conversion
                                    SA=gsw.SA_from_SP(surf_S, P, lon[j], lat[j])
                
                                    # Calculate conservative temp or temp 
                                    CT=gsw.CT_from_t(SA, surf_T, P)
                                    
                                    # Calculate density
                                    #dense=gsw.density.rho_t_exact(SA, T, P) # kg/m^3
                                    surf_dense=gsw.density.sigma0(SA,CT)+1000
                                    
                                    # Calculate oxygen saturation at each point 
                                    O2_sol=gsw.O2sol(SA,CT,P,lon[j],lat[j])
                                    oxy_sat=np.round(surf_O/O2_sol*100,2)
                                    oxy_dev=surf_O-O2_sol
                                    
                                    surf_O_units = (surf_O*surf_dense)*(10**-6)
                                    
                                    # Nearest neighbor
                                    date_ind=np.where(era5_time==close_date)
                                    # Should add a check to make sure there is only 1 date ind
                                    # use index to get lat-lon slices and other variable
                                    # wind speed structure (time,lat, lon)
                                    
                                    # load u_10 data
                                    u10_slice=era_u10_data.sel(time=close_date).u10.values[0,:,:]
                                    
                                    # load v_10 data
                                    v10_slice=era_v10_data.sel(time=close_date).v10.values[0,:,:]
                                    
                                    # load slp
                                    slp_slice=era_slp_data.sel(time=close_date).msl.values[0,:,:]
                                    
                                    # Find the closest lat, lon point
                        
                                    # Use nearest neighbors to find nearest point
                                    dd, ii = tree.query([lon[j],lat[j]])
                                    
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
                                    
                                    # Calculate gas flux
                                    ## Use nearest values and calculate air-sea flux at each point
                                    c=surf_O_units
                                    S=surf_S
                                    U10=(u10_point**2 + v10_point**2)**.5
                                    T=surf_T
                                    SLP=slp_point
                                    
                                    # Outputs 
                                    # Fd = surface gas flux mol/m^2-s
                                    # Fc = flyx from fully collapsing large bubbles
                                    # Fp = flux from partially collapsing large bubbles
                                    # Deq = equilibirum saturation ( %sat/100)
                                    # k = diffusive gas traner velocity 
                                    # Ft = Fd + Fc + Fp
                                    
                                    [Fd_L13, Fc_L13, Fp_L13, Deq_L13, k_L13]=AS.L13(C=c,u10=U10, SP=S, pt=T, slp=SLP, gas='O2', rh=1)
                                    [Fd_N16, Fc_N16, Fp_N16, Deq_N16, k_N16]=AS.N16(C=c,u10=U10, SP=S, pt=T, slp=SLP, gas='O2', rh=1)
                                    
                                    Ft_L13=Fp_L13+Fc_L13+Fd_L13
                                    Ft_N16=Fp_N16+Fc_N16+Fd_N16
                                    
                                    flux_profcount=flux_profcount+1
                                    
                                    # Save data
                                    df_temp=pd.DataFrame({'FloatWMO': [WMO], 'ProfNum': [int(f.CYCLE_NUMBER.values[j])],'Date':[prof_date],'Date_6hr':[close_date],
                                                          'Lat': [lat[j]], 'Lon': [lon[j]],'MLD': [MLD], 'Surf_T': [surf_T],'Surf_S':[surf_S],'Surf_O': [surf_O],
                                                          'Oxy_Sat':[oxy_sat], 'Oxy_Dev': [oxy_dev],'k_L13': [k_L13],'Fp_L13': [Fp_L13],'Fc_L13': [Fc_L13],'Fd_L13': [Fd_L13], 
                                                          'Ft_L13': [Ft_L13],'k_N16': [k_N16],'Fp_N16': [Fp_N16],'Fc_N16': [Fc_N16],'Fd_N16': [Fd_N16], 'Ft_N16': [Ft_N16]})
                                    
                                    if bc_flag ==1:
                                        BCData=BCData.append(df_temp)
                                    elif bc_flag == 0:
                                        GyreData=GyreData.append(df_temp)
                                    else:
                                        print('\n%%% ERROR IN SAVING DATA %%%\n')
    else:
        print('%%% THIS FLOAT DOES NOT MEASURE OXYGEN %%%')
    
    flux_count[i]=np.round(flux_profcount/len(lat)*100,2)
    
    toc2=time.time()
    print('\nFloat Run Time: ')
    print(toc2-tic2,' seconds elapsed')
    print((toc2-tic2)/60,' minutes elapsed')

print('\n%% Saving adjusted variable flags %%\n')
adj_df=pd.DataFrame({'FileName': ArgoDACWMO, 'FloatWMO': ArgoWMO, 'Pres_Adj': pres_flag_total,'Temp_Adj': temp_flag_total,'Sal_Adj': sal_flag_total, 'Oxy_Adj':oxy_flag_total,'Flux_Calc': flux_count})
adj_df=adj_df.sort_values(by=['FloatWMO'])
adj_df.to_csv(CSVDir+'AdjDataFlags.csv')

print('\n%% Saving gas flux data %%\n')
BCData=BCData.iloc[1:,:]
BCData.to_csv(CSVDir+'GasFluxData_BC.csv')
GyreData=GyreData.iloc[1:,:]
GyreData.to_csv(CSVDir+'GasFluxData_Gyre.csv')

toc1=time.time()
print('\nTOTAL RUN TIME: ')
print(toc1-tic1,' seconds elapsed')
print((toc1-tic1)/60,' minutes elapsed')

    