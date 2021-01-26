#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:07:41 2021

@author: Ellen
"""

from shapely.geometry import LineString, Point, Polygon
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import cartopy as ct
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from haversine import haversine
from datetime import datetime
from scipy.spatial import ConvexHull

# Make a shape representing the boundary current
# Use on float's pat and add a spatial buffer?

#### Some parameters ######
# N Atlantic Region
lat_N=80.000
lat_S= 40.00
lon_E= -40.00
lon_W= -80.00

# Labrador Sea Region
lab_N=65
lab_S=48
lab_E=-45
lab_W=-80
##############################

# Load boundary current data
CSVDir_BC='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BCFloats_'
FileList_BC=glob.glob(CSVDir_BC+'*.csv')

df_count = 0
file_count=0
for filedir in FileList_BC:
    
    # Read in each data file
    data = pd.read_csv(filedir)

    # And combine into one big data frame
    if df_count == 0:
        BCData = pd.read_csv(filedir)
        df_count = 1
    else:
        BCData=BCData.append(data)
    
    file_count=file_count+1

# Load gyre data
CSVDir_LSG='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
FileList_LSG=glob.glob(CSVDir_LSG+'*.csv')

df_count = 0
file_count=0
for filedir in FileList_LSG:
    
    # Read in each data file
    data = pd.read_csv(filedir)

    # And combine into one big data frame
    if df_count == 0:
        LSGData = pd.read_csv(filedir)
        df_count = 1
    else:
        LSGData=LSGData.append(data)
    
    file_count=file_count+1

# Crop data to the Labrador Sea region
BCData.loc[BCData.loc[:,'Lat']>lab_N,:]=np.NaN
BCData.loc[BCData.loc[:,'Lat']<lab_S,:]=np.NaN
BCData.loc[BCData.loc[:,'Lon']>lab_E,:]=np.NaN
BCData.loc[BCData.loc[:,'Lon']<lab_W,:]=np.NaN
BCData=BCData.dropna()
new_ind=np.arange(BCData.shape[0])
BCData=BCData.set_index(pd.Index(list(new_ind)))

LSGData=LSGData.dropna()
new_ind=np.arange(LSGData.shape[0])
LSGData=LSGData.set_index(pd.Index(list(new_ind)))

All_Lat_BC=BCData.loc[:,'Lat']
All_Lon_BC=BCData.loc[:,'Lon']

All_Lat_LSG=LSGData.loc[:,'Lat']
All_Lon_LSG=LSGData.loc[:,'Lon']

bc_floatlist= BCData.loc[:,'WMO'].unique().tolist()
lsg_floatlist= LSGData.loc[:,'WMO'].unique().tolist()

goodfloat=5904988

# Get lat-lon positions for this "good float"
lat_BC=BCData.loc[BCData.loc[:,'WMO']==goodfloat,'Lat'].to_numpy()
lon_BC=BCData.loc[BCData.loc[:,'WMO']==goodfloat,'Lon'].to_numpy()

## Make a piece-wise boundary current
break1_lon=-51
break1_flag=0
break1_count=0
break1_ind=np.NaN

while (break1_flag ==0 and break1_count<len(lon_BC)):
    
    if lon_BC[break1_count]<=break1_lon:
        break1_flag=1
        break1_ind=break1_count
    
    break1_count=break1_count+1
    
break2_lat=57.5
break2_flag=0
break2_count=0
break2_ind=np.NaN

while (break2_flag ==0 and break2_count<len(lat_BC)):
    
    if lat_BC[break2_count]<=break2_lat:
        break2_flag=1
        break2_ind=break2_count
    
    break2_count=break2_count+1

break3_lat=52
break3_flag=0
break3_count=0
break3_ind=np.NaN

while (break3_flag ==0 and break3_count<len(lat_BC)):
    
    if lat_BC[break3_count]<=break3_lat:
        break3_flag=1
        break3_ind=break3_count
    
    break3_count=break3_count+1

## Make a linestring from position data
GoodPoints=[[]]*len(lat_BC)

lat_shift=-1.5
lon_shift=1.5
for i in np.arange(len(GoodPoints)):
    lon_val=lon_BC[i]
    lat_val=lat_BC[i]
    if i<=break1_ind:
        # Shift lat-values
        lat_val=lat_val+lat_shift
    if i>=break3_ind:
        # Shift lat-values
        lon_val=lon_val+lon_shift
        
    GoodPoints[i]=Point(lon_val,lat_val)
    
GoodPath1 = LineString(GoodPoints[0:break1_ind+1])
GoodPath2 = LineString(GoodPoints[break1_ind+1:break2_ind+1])
GoodPath3 = LineString(GoodPoints[break2_ind+1:break3_ind+1])
GoodPath4 = LineString(GoodPoints[break3_ind+1:])

BoundaryCurrent1=GoodPath1.buffer(1.4)
BoundaryCurrent2=GoodPath2.buffer(2)
BoundaryCurrent3=GoodPath3.buffer(1)
BoundaryCurrent4=GoodPath4.buffer(1.4)

x1,y1 = BoundaryCurrent1.exterior.xy
x2,y2 = BoundaryCurrent2.exterior.xy
x3,y3 = BoundaryCurrent3.exterior.xy
x4,y4 = BoundaryCurrent4.exterior.xy

included=0
# Determine what percentage of points fall in boundary current
for i in np.arange(len(All_Lat_BC)):
    p=Point(All_Lon_BC[i], All_Lat_BC[i])
    if p.within(BoundaryCurrent1) == True:
        included=included+1
    elif p.within(BoundaryCurrent2) == True:
        included=included+1
    elif p.within(BoundaryCurrent3) == True:
        included=included+1
    elif p.within(BoundaryCurrent4) == True:
        included=included+1
        
print('\n%%%%%%%%%%%%%%%%%')
print(included,' BC profiles of ',len(All_Lat_BC),' are in the BC')
print((included/len(All_Lat_BC))*100,' % of points')
print('%%%%%%%%%%%%%%%%%\n') 

## Determine what percentage of gyre flots are in the BC
included_lsg=0
# Determine what percentage of points fall in boundary current
for i in np.arange(len(All_Lat_LSG)):
    p=Point(All_Lon_LSG[i], All_Lat_LSG[i])
    if p.within(BoundaryCurrent1) == True:
        included_lsg=included_lsg+1
    elif p.within(BoundaryCurrent2) == True:
        included_lsg=included_lsg+1
    elif p.within(BoundaryCurrent3) == True:
        included_lsg=included_lsg+1
    elif p.within(BoundaryCurrent4) == True:
        included_lsg=included_lsg+1
        
print('\n%%%%%%%%%%%%%%%%%')
print(included_lsg,' LSG profiles of ',len(All_Lat_LSG),' are in the BC')
print((included_lsg/len(All_Lat_LSG))*100,' % of points')
print('%%%%%%%%%%%%%%%%%\n') 

## Make the gyre ##
gyre_points=np.array((All_Lon_LSG, All_Lat_LSG)).T
hull = ConvexHull(gyre_points)
g_p=[[]]*hull.simplices.shape[0]
for i in np.arange(hull.simplices.shape[0]):
    simplex=hull.simplices[i]
    g_p=Point(np.array(gyre_points[simplex, 0], gyre_points[simplex, 1]))
Gyre=Polygon(hull.simplices)
px,py=Gyre.exterior.xy

plt.figure()
plt.plot(x1,y1,c='r')
plt.plot(x2,y2,c='r')
plt.plot(x3,y3,c='r')
plt.plot(x4,y4,c='r')
plt.plot(px,py,c='k')
plt.scatter(All_Lon_BC,All_Lat_BC,s=1, c='b')
plt.scatter(All_Lon_LSG,All_Lat_LSG,s=1, c='g')

plt.figure()
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
plt.plot(x1,y1,c='r')
plt.plot(x2,y2,c='r')
plt.plot(x3,y3,c='r')
plt.plot(x4,y4,c='r')
plt.scatter(All_Lon_BC,All_Lat_BC,s=1)

## Calculate float speeds
bc_floatspeeds=np.zeros(len(All_Lat_BC))
bc_floatspeeds[:]=np.NaN
bc_scount=0

for wmo in bc_floatlist:
    
    lat_pos=BCData.loc[BCData.loc[:,'WMO']==wmo,'Lat']
    lon_pos=BCData.loc[BCData.loc[:,'WMO']==wmo,'Lon']
    time_pos=BCData.loc[BCData.loc[:,'WMO']==wmo,'Date']
    
    indlist=lon_pos.index.to_list()
    for j in np.arange(len(lat_pos)-1):
        
        i1=indlist[j]
        i2=indlist[j+1]
        # Calculate change in distance
        pos1=np.array((lon_pos[i1],lat_pos[i1]))
        pos2=np.array((lon_pos[i2],lat_pos[i2]))
        dist=haversine(pos1, pos2, unit='m')
        #print(dist)
        
        # Calculate change in time
        t1=datetime.strptime(time_pos[i1], '%Y-%m-%d %H:%M:%S')
        t2=datetime.strptime(time_pos[i2], '%Y-%m-%d %H:%M:%S')
        tt=(t2-t1).total_seconds()
        #print(tt)
        
        bc_floatspeeds[bc_scount]=dist/tt
        bc_scount=bc_scount+1
    
    # plt.figure(1)
    # plt.plot(bc_floatspeeds)

## Calculate float speeds
lsg_floatspeeds=np.zeros(len(All_Lat_LSG))
lsg_floatspeeds[:]=np.NaN
lsg_scount=0

for wmo in lsg_floatlist:
    
    lat_pos=LSGData.loc[LSGData.loc[:,'WMO']==wmo,'Lat']
    lon_pos=LSGData.loc[LSGData.loc[:,'WMO']==wmo,'Lon']
    time_pos=LSGData.loc[LSGData.loc[:,'WMO']==wmo,'Date']
    
    indlist=lon_pos.index.to_list()
    for j in np.arange(len(lat_pos)-1):
        
        i1=indlist[j]
        i2=indlist[j+1]
        # Calculate change in distance
        pos1=np.array((lon_pos[i1],lat_pos[i1]))
        pos2=np.array((lon_pos[i2],lat_pos[i2]))
        dist=haversine(pos1, pos2, unit='m')
        #print(dist)
        
        # Calculate change in time
        t1=datetime.strptime(time_pos[i1], '%Y-%m-%d %H:%M:%S')
        t2=datetime.strptime(time_pos[i2], '%Y-%m-%d %H:%M:%S')
        tt=(t2-t1).total_seconds()
        #print(tt)
        
        lsg_floatspeeds[lsg_scount]=dist/tt
        lsg_scount=lsg_scount+1
    
    # plt.figure(2)
    # plt.plot(lsg_floatspeeds)
    # plt.pause(2)
    
lsg_floatspeeds = lsg_floatspeeds[~np.isnan(lsg_floatspeeds)]

plt.figure()
plt.hist(lsg_floatspeeds)
plt.hist(bc_floatspeeds)
plt.legend(['Gyre','BC'])

print('\n%%%%%%%%%% RESULTS %%%%%%%%%%%\n')
print('Boundary Current')
print('Mean speed (m/s): ', np.nanmean(bc_floatspeeds))
print('Std: ',np.nanstd(bc_floatspeeds))
print('\nGyre')
print('Mean speed (m/s): ', np.nanmean(lsg_floatspeeds))
print('Std: ',np.nanstd(lsg_floatspeeds))


