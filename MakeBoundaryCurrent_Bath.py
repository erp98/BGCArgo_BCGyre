#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:55:54 2021

@author: Ellen
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import glob
import RandomFxns as RF
from datetime import datetime
from scipy.spatial import KDTree
from scipy.stats import ttest_ind

# Load bathmyetry data
BathFile='/Users/Ellen/Desktop/GEBCO/gebco_2020_n70.0_s40.0_w-70.0_e-30.0.nc'
BathData = xr.open_dataset(BathFile)

print('Making bath contours...')

BathLat=BathData.lat.values
BathLon=BathData.lon.values
BathHeight=BathData.elevation.values
BathHeight=np.where(BathHeight>0, np.NaN, BathHeight)
BathHeight=BathHeight*-1

C1000=np.where(BathHeight!=1000, np.NaN, BathHeight)
C1500=np.where(BathHeight!=1500, np.NaN, BathHeight)
C2000=np.where(BathHeight!=2000, np.NaN, BathHeight)

BX, BY = np.meshgrid(BathLon, BathLat)

plt.figure()
plt.contour(BX, BY, BathHeight, levels=[1000,1500,2000])
# plt.pcolormesh(BX, BY, C1000)
# plt.pcolormesh(BX, BY, C1500)
# plt.pcolormesh(BX, BY, C2000)
plt.colorbar()

# load float data
lab_N=65
lab_S=48
lab_E=-40
lab_W=-80

# # Load boundary current data
# CSVDir_BC='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BCFloats_'
# FileList_BC=glob.glob(CSVDir_BC+'*.csv')

# df_count = 0
# file_count=0
# for filedir in FileList_BC:
    
#     # Read in each data file
#     data = pd.read_csv(filedir)

#     # And combine into one big data frame
#     if df_count == 0:
#         BCData = pd.read_csv(filedir)
#         df_count = 1
#     else:
#         BCData=BCData.append(data)
    
#     file_count=file_count+1

# # Load gyre data
# CSVDir_LSG='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
# FileList_LSG=glob.glob(CSVDir_LSG+'*.csv')

# df_count = 0
# file_count=0
# for filedir in FileList_LSG:
    
#     # Read in each data file
#     data = pd.read_csv(filedir)

#     # And combine into one big data frame
#     if df_count == 0:
#         LSGData = pd.read_csv(filedir)
#         df_count = 1
#     else:
#         LSGData=LSGData.append(data)
    
#     file_count=file_count+1

# # Crop data to the Labrador Sea region
# BCData.loc[BCData.loc[:,'Lat']>lab_N,:]=np.NaN
# BCData.loc[BCData.loc[:,'Lat']<lab_S,:]=np.NaN
# BCData.loc[BCData.loc[:,'Lon']>lab_E,:]=np.NaN
# BCData.loc[BCData.loc[:,'Lon']<lab_W,:]=np.NaN
# BCData=BCData.dropna()
# new_ind=np.arange(BCData.shape[0])
# BCData=BCData.set_index(pd.Index(list(new_ind)))

# LSGData=LSGData.dropna()
# new_ind=np.arange(LSGData.shape[0])
# LSGData=LSGData.set_index(pd.Index(list(new_ind)))

# All_Lat_BC=BCData.loc[:,'Lat']
# All_Lon_BC=BCData.loc[:,'Lon']

# All_Lat_LSG=LSGData.loc[:,'Lat']
# All_Lon_LSG=LSGData.loc[:,'Lon']

# bc_floatlist= BCData.loc[:,'WMO'].unique().tolist()
# lsg_floatlist= LSGData.loc[:,'WMO'].unique().tolist()

# Use adjusted data files to get float list
CheckBCDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_BCFloats.csv'
CheckGyreDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/AdjDataFlags_LabradorFloats.csv'

BC_Check= pd.read_csv(CheckBCDir)
BC_Check=BC_Check.loc[:,['FileName','FloatWMO','Pres_Adj']]
BC_Check=BC_Check.dropna()
BC_CheckList=BC_Check.loc[:,'FileName'].to_numpy()
BC_List=(['BC']*len(BC_CheckList))


Gyre_Check= pd.read_csv(CheckGyreDir)
Gyre_Check=Gyre_Check.loc[:,['FileName','FloatWMO','Pres_Adj']]
Gyre_Check=Gyre_Check.dropna()
Gyre_CheckList=Gyre_Check.loc[:,'FileName']
Gyre_List=(['Gyre']*len(Gyre_CheckList))

CheckFloats=pd.DataFrame({'FloatType':BC_List, 'FloatDir':BC_CheckList})
CheckFloats=CheckFloats.append(pd.DataFrame({'FloatType':Gyre_List, 'FloatDir':Gyre_CheckList}))

df_count = 0

print('Loading float data...')
for i in np.arange(CheckFloats.shape[0]):
    fline=CheckFloats.iloc[i,1]
    dac=fline.split('/')[0]
    wmo=fline.split('/')[1]
    
    data = RF.ArgoDataLoader(DAC=dac, WMO=wmo)
    
    lat=data.LATITUDE.values
    lon=data.LONGITUDE.values
    
    dates=data.JULD.values
    date_reform=[[]]*dates.shape[0]
    bad_index=[]
    for j in np.arange(len(date_reform)):
        if np.isnat(dates[j]) == False:
            date_reform[j]=datetime.fromisoformat(str(dates[j])[:-3])
        else:
            date_reform[j]=dates[j]
            bad_index=bad_index+[j]
            
    if len(bad_index)>0:
        for b_i in bad_index:
            # Convert arrays to lists to use pop
            lat=list(lat)
            lon=list(lon)
            
            # POP
            date_reform.pop(b_i)
            lat.pop(b_i)
            lon.pop(b_i)
            
            # Convert lists back to arrays
            lat=np.array(lat)
            lon=np.array(lon)
    
    typelist=[CheckFloats.iloc[i,0]]*len(date_reform)
    wmolist=[wmo]*len(date_reform)
    
    if df_count == 0:
        AllCheckList = pd.DataFrame({'FloatType': typelist, 'WMO': wmolist,'Date': date_reform,'Lat': lat,'Lon':lon})
        df_count =1
    else:
        df_temp=pd.DataFrame({'FloatType': typelist, 'WMO': wmolist,'Date': date_reform,'Lat': lat,'Lon':lon})
        AllCheckList=AllCheckList.append(df_temp)

AllCheckList.loc[AllCheckList.loc[:,'Lat']>lab_N,:]=np.NaN
AllCheckList.loc[AllCheckList.loc[:,'Lat']<lab_S,:]=np.NaN
AllCheckList.loc[AllCheckList.loc[:,'Lon']>lab_E,:]=np.NaN
AllCheckList.loc[AllCheckList.loc[:,'Lon']<lab_W,:]=np.NaN
AllCheckList=AllCheckList.dropna()

new_ind=np.arange(AllCheckList.shape[0])
AllCheckList=AllCheckList.set_index(pd.Index(list(new_ind)))
ftype=AllCheckList.loc[:,'FloatType'].to_numpy()
bc_count=0
j=0
found=0
while (found ==0 and j<len(ftype)):
    
    if ftype[j]=='BC':
        bc_count = bc_count +1
    elif ftype[j]=='Gyre':
        found =1
    
    j=j+1

gyre_count=AllCheckList.shape[0]-bc_count

lat_bc=AllCheckList.iloc[:bc_count,3].to_numpy()
lon_bc=AllCheckList.iloc[:bc_count,4].to_numpy()
lat_lsg=AllCheckList.iloc[bc_count:,3].to_numpy()
lon_lsg=AllCheckList.iloc[bc_count:,4].to_numpy()

bc_dist=np.zeros(len(lat_bc))
bc_dist[:]=np.NaN
gyre_dist=np.zeros(len(lat_lsg))
gyre_dist[:]=np.NaN

## Nearest Neighbor
print('Getting Contour Lat-Lon...')
a=np.where(np.isnan(C1000)==False)
cont_lat=np.zeros((len(a[0])))
cont_lat[:]=np.NaN

cont_lon=np.zeros((len(a[0])))
cont_lon[:]=np.NaN

for f in np.arange(len(a[0])):
    i=a[0][f]
    j=a[1][f]
    cont_lon[f]=BX[i,j]
    cont_lat[f]=BY[i,j]
            
cont_lon =cont_lon[~np.isnan(cont_lon)] 
cont_lat =cont_lat[~np.isnan(cont_lat)] 
       
tree = KDTree(np.c_[cont_lon, cont_lat])
print('BC Nearest Neighbor...')
for i in np.arange(len(bc_dist)):
    dd, ii = tree.query([lon_bc[i],lat_bc[i]])
    bc_dist[i]=dd
    
print('Gyre Nearest Neighbor...')
for i in np.arange(len(gyre_dist)):
    dd, ii = tree.query([lon_lsg[i],lat_lsg[i]])
    gyre_dist[i]=dd
    
print('Making Histogram...')
plt.figure()
plt.hist(gyre_dist, color='orange')
plt.hist(bc_dist, color='blue')
plt.legend(['Gyre','Boundary Current'])

print('\n%%%%%%%%%% RESULTS %%%%%%%%%%%\n')
print('Boundary Current')
print('Mean Distance: ', np.nanmean(bc_dist))
print('Std: ',np.nanstd(bc_dist))
print('\nGyre')
print('Mean Distance: ', np.nanmean(gyre_dist))
print('Std: ',np.nanstd(gyre_dist))

plt.figure()
plt.contour(BX, BY, BathHeight, levels=[1000,1500,2000])
plt.scatter(lon_bc,lat_bc, s=1)
plt.scatter(lon_lsg,lat_lsg, s=1)

print('T-Test Results...')
print(ttest_ind(bc_dist,gyre_dist))
print(ttest_ind(bc_dist,gyre_dist, equal_var=False))

min_dist=1.6
max_dist=2.9
step_size=0.1

dist_range=np.arange(min_dist,max_dist+step_size, step_size)
bc_bc_r=np.zeros(len(dist_range))
bc_lsg_r=np.zeros(len(dist_range))
lsg_bc_r=np.zeros(len(dist_range))
lsg_lsg_r=np.zeros(len(dist_range))

bc_bc_r[:]=np.NaN
bc_lsg_r[:]=np.NaN
lsg_bc_r[:]=np.NaN
lsg_lsg_r[:]=np.NaN


for h in np.arange(len(dist_range)):
    
    bc_dist=dist_range[h]
    bc_bc=0
    bc_lsg=0
    
    lsg_lsg=0
    lsg_bc=0
    
    lon_bc_new=[]
    lat_bc_new=[]
    
    lon_g_new=[]
    lat_g_new=[]
    for i in np.arange(AllCheckList.shape[0]):
        lon_c=AllCheckList.iloc[i,4]
        lat_c=AllCheckList.iloc[i,3]
        ft_c=AllCheckList.iloc[i,0]
        
        dd, ii = tree.query([lon_c,lat_c])
        
        if dd<= bc_dist:   
            if ft_c == 'BC':
                bc_bc=bc_bc+1
            elif ft_c == 'Gyre':
                bc_lsg=bc_lsg+1
            
            lon_bc_new=lon_bc_new+[lon_c]
            lat_bc_new=lat_bc_new+[lat_c]
        else:
            if ft_c == 'BC':
                lsg_bc=lsg_bc+1
            elif ft_c == 'Gyre':
                lsg_lsg=lsg_lsg+1
            
            lon_g_new=lon_g_new+[lon_c]
            lat_g_new=lat_g_new+[lat_c]

    per_bc_bc_tot=np.round(bc_bc/bc_count*100,2)
    per_bc_lsg_tot=np.round(bc_lsg/gyre_count*100,2)
    per_lsg_bc_tot=np.round(lsg_bc/bc_count*100,2)
    per_lsg_lsg_tot=np.round(lsg_lsg/gyre_count*100,2)
    
    # print('Boundary Current Bathymetry Analysis \n')
    # print(bc_bc,' BC profiles of ',bc_count,' are in the BC')
    # print(per_bc_bc_tot,' % of points')
    # print(bc_lsg,' LSG profiles of ',gyre_count,' are in the BC')
    # print(per_bc_lsg_tot,' % of points')
    
    # print('\nGyre Shape Analysis \n')
    # print(lsg_bc,' BC profiles of ',bc_count,' are in the LSG')
    # print(per_lsg_bc_tot,' % of points')
    # print(lsg_lsg,' LSG profiles of ',gyre_count,' are in the LSG')
    # print(per_lsg_lsg_tot,' % of points')
    # print('%%%%%%%%%%%%%%%%%\n')
    # plt.figure()
    # plt.contour(BX, BY, BathHeight, levels=[1000,1500,2000])
    # plt.scatter(lon_bc_new,lat_bc_new, s=1)
    # plt.scatter(lon_g_new,lat_g_new, s=1)

    bc_bc_r[h]=per_bc_bc_tot
    bc_lsg_r[h]=per_bc_lsg_tot
    lsg_bc_r[h]=per_lsg_bc_tot
    lsg_lsg_r[h]=per_lsg_lsg_tot

Dist_df=pd.DataFrame({'Distance': dist_range, 'BC_BC': bc_bc_r,'BC_LSG': bc_lsg_r,'LSG_BC': lsg_bc_r,'LSG_LSG':lsg_lsg_r})
Dist_df.to_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BCDistances.csv')

plt.figure()
plt.plot(Dist_df.loc[:,'BC_BC'], Dist_df.loc[:,'BC_LSG'])

bc_dist=2.2
bc_bc=0
bc_lsg=0

lsg_lsg=0
lsg_bc=0

lon_bc_new=[]
lat_bc_new=[]

lon_g_new=[]
lat_g_new=[]
for i in np.arange(AllCheckList.shape[0]):
    lon_c=AllCheckList.iloc[i,4]
    lat_c=AllCheckList.iloc[i,3]
    ft_c=AllCheckList.iloc[i,0]
    
    dd, ii = tree.query([lon_c,lat_c])
    
    if dd<= bc_dist:   
        if ft_c == 'BC':
            bc_bc=bc_bc+1
        elif ft_c == 'Gyre':
            bc_lsg=bc_lsg+1
        
        lon_bc_new=lon_bc_new+[lon_c]
        lat_bc_new=lat_bc_new+[lat_c]
    else:
        if ft_c == 'BC':
            lsg_bc=lsg_bc+1
        elif ft_c == 'Gyre':
            lsg_lsg=lsg_lsg+1
        
        lon_g_new=lon_g_new+[lon_c]
        lat_g_new=lat_g_new+[lat_c]

per_bc_bc_tot=np.round(bc_bc/bc_count*100,2)
per_bc_lsg_tot=np.round(bc_lsg/gyre_count*100,2)
per_lsg_bc_tot=np.round(lsg_bc/bc_count*100,2)
per_lsg_lsg_tot=np.round(lsg_lsg/gyre_count*100,2)

print('Boundary Current Bathymetry Analysis \n')
print(bc_bc,' BC profiles of ',bc_count,' are in the BC')
print(per_bc_bc_tot,' % of points')
print(bc_lsg,' LSG profiles of ',gyre_count,' are in the BC')
print(per_bc_lsg_tot,' % of points')

print('\nGyre Shape Analysis \n')
print(lsg_bc,' BC profiles of ',bc_count,' are in the LSG')
print(per_lsg_bc_tot,' % of points')
print(lsg_lsg,' LSG profiles of ',gyre_count,' are in the LSG')
print(per_lsg_lsg_tot,' % of points')
print('%%%%%%%%%%%%%%%%%\n')
plt.figure()
plt.contour(BX, BY, BathHeight, levels=[1000,1500,2000])
plt.scatter(lon_bc_new,lat_bc_new, s=1)
plt.scatter(lon_g_new,lat_g_new, s=1)

plt.show()