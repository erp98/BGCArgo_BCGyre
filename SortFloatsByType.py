#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:05:33 2020

@author: Ellen
"""

import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import glob
import xarray as xr


#######################
# Plot trajectories of argo floats in designated area
#######################

Dir1='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/'
# Get FloatID
Argofiles=Dir1+'DacWMO_NAtlantic.txt'

lat_N=80.000
lat_S= 40.00
lon_E= 0.00
lon_W= -80.00

ArgoNum=[]
count=0

Irm=[]
Lab=[]
BC=[]
Other=[]
AllFloats=[]

print('Start of file reading...')
with open(Argofiles) as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        x=line.strip()
        ArgoNum=ArgoNum + [x]

#print(ArgoNum)

floatnumlist=ArgoNum[:]
fname_list=[]

for i in floatnumlist:
    floatfname=i
    fname_t=glob.glob('/Users/Ellen/Desktop/ArgoGDAC/dac/'+floatfname+'/*_prof.nc')
    #print('this is f_namet:',fname_t)
    fname_list=fname_list + fname_t

#print(fname_list)
for BGCfile in fname_list:

    # MR - merged file (US) | SR - merged file (?) (FR)
    # B - bio
    # R/D raw/delated

    f = xr.open_dataset(BGCfile)
    
    fname_complete=BGCfile.split('/')
    dac=fname_complete[6]
    titlename=fname_complete[7]

    # Plot trajectory
    # Get Latitidue Data
    LatData = f.LATITUDE.values
    # Get Longitude Data
    LonData = f.LONGITUDE.values

    plt.figure(1)

    NA = plt.axes(projection=ct.crs.PlateCarree())

    NA.set_extent([lon_E, lon_W, lat_S, lat_N])

    lonval=-1*np.arange(-lon_E,-lon_W+1,10)
    latval=np.arange(lat_S,lat_N,10)

    NA.set_xticks(lonval, crs=ct.crs.PlateCarree())
    NA.set_yticks(latval, crs=ct.crs.PlateCarree())

    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    NA.add_feature(ct.feature.COASTLINE)
    NA.add_feature(ct.feature.OCEAN)

    NA.xaxis.set_major_formatter(lon_formatter)
    NA.yaxis.set_major_formatter(lat_formatter)

    plt.title(titlename)
    plt.plot(LonData[0],LatData[0],'go')
    plt.plot(LonData[len(LonData)-1],LatData[len(LonData)-1],'ro')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.plot(LonData, LatData)

    plt.pause(0.1)

    check = 0

    while check == 0:

        var_int=input('1 (Lab), 2 (Irm), 3 (BC), 4 (Other), 5(N/A):')
        print(var_int)

        if var_int == '1':
            Lab=Lab+[titlename]
            AllFloats=AllFloats+[titlename]
            check = 1
        elif var_int == '2':
            Irm=Irm+[titlename]
            AllFloats=AllFloats+[titlename]
            check=1
        elif var_int == '3':
            BC=BC+[titlename]
            AllFloats=AllFloats+[titlename]
            check=1
        elif var_int == '4':
            Other=Other+[titlename]
            AllFloats=AllFloats+[titlename]
            check=1
        elif var_int == '5':
            check=1

    plt.clf()


with open(Dir1+'Sorted_LabradorSeaFloats.txt','w') as f:
    for ele in Lab:
        f.write(ele+'\n')

with open(Dir1+'Sorted_BCFloats.txt','w') as f:
    for ele in BC:
        f.write(ele+'\n')

with open(Dir1+'Sorted_IrmingerFloats.txt','w') as f:
    for ele in Irm:
        f.write(ele+'\n')

with open(Dir1+'Sorted_OtherFloats.txt','w') as f:
    for ele in Other:
        f.write(ele+'\n')

with open(Dir1+'Sorted_AllFloats.txt','w') as f:
    for ele in AllFloats:
        f.write(ele+'\n')
