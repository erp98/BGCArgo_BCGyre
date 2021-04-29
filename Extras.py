#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:27:20 2021

@author: Ellen
"""
import numpy as np
import RandomFxns as RF
import pandas as pd
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import matplotlib.pyplot as plt

BC_Dir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_BCFloats.txt'
Gyre_Dir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/Sorted_DACWMO_LabradorSeaFloats.txt'

fsize_x=10
fsize_y=6

# N Atlantic Region
lat_N=80.000
lat_S= 40.00
lon_E= 0.00
lon_W= -80.00

## BC Data
bc_floatlist=[]
bc_daclist=[]
bc_dacwmo=[]
count= 0
b_c=0
# Read in float info
with open(BC_Dir) as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        t=line.strip()
        r=t.split('/')
        bc_floatlist=bc_floatlist+[int(r[1])]
        bc_daclist=bc_daclist+[r[0]]
        bc_dacwmo=bc_dacwmo+[t]

BC_Data = pd.DataFrame({'Date': [np.NaN],'Lat': [np.NaN], 'Lon': [np.NaN]})
for b in np.arange(len(bc_floatlist)):
### For debugging and looking at specific floats ###
#for b in [10]:
    WMO=bc_floatlist[b]
    dac=bc_daclist[b]
    
    f = RF.ArgoDataLoader(DAC=dac, WMO=WMO)
    float_vars=list(f.keys())
    float_vars=np.array(float_vars)
    oxy_check = np.where(float_vars=='DOXY')
    
    if oxy_check[0].size != 0:
        lat = f.LATITUDE.values
        lon = f.LONGITUDE.values
        date = f.JULD.values
        
        dft=pd.DataFrame({'Date': date,'Lat': lat, 'Lon': lon})
        BC_Data=BC_Data.append(dft)
        b_c = b_c+1

## gyre Data
g_floatlist=[]
g_daclist=[]
g_dacwmo=[]
g_c = 0
count= 0
# Read in float info
with open(Gyre_Dir) as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        t=line.strip()
        r=t.split('/')
        g_floatlist=g_floatlist+[int(r[1])]
        g_daclist=g_daclist+[r[0]]
        g_dacwmo=g_dacwmo+[t]

G_Data = pd.DataFrame({'Date': [np.NaN],'Lat': [np.NaN], 'Lon': [np.NaN]})
for b in np.arange(len(g_floatlist)):
### For debugging and looking at specific floats ###
#for b in [10]:
    WMO=g_floatlist[b]
    dac=g_daclist[b]
    
    f = RF.ArgoDataLoader(DAC=dac, WMO=WMO)
    float_vars=list(f.keys())
    float_vars=np.array(float_vars)
    oxy_check = np.where(float_vars=='DOXY')
    date = f.JULD.values
    
    if oxy_check[0].size != 0:
        lat = f.LATITUDE.values
        lon = f.LONGITUDE.values
        
        dft=pd.DataFrame({'Date': date,'Lat': lat, 'Lon': lon})
        G_Data=G_Data.append(dft)
        g_c= g_c+1
    
BC_Data=BC_Data.iloc[1:,:]
G_Data = G_Data.iloc[1:,:]

## Plot Data
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
#NA.add_feature(ct.feature.OCEAN)
NA.xaxis.set_major_formatter(lon_formatter)
NA.yaxis.set_major_formatter(lat_formatter)
#plt.title('Trajectory for Float '+str(WMO))
plt.scatter(BC_Data.loc[:,'Lon'], BC_Data.loc[:,'Lat'], s=1,label='Boundary Current (n = '+str(b_c)+')')
plt.scatter(G_Data.loc[:,'Lon'], G_Data.loc[:,'Lat'], s=1,label='Gyre (n = '+str(g_c)+')')
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('BGC Argo Floats')
plt.savefig('./Figures/Extras/OOIUpdate2021.jpg')
        