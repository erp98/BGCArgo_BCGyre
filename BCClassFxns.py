#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:51:09 2021

@author: Ellen
"""

from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import ConvexHull
import numpy as np
import glob
import pandas as pd

# Labrador Sea Region
lab_N=65
lab_S=48
lab_E=-45
lab_W=-80

goodfloat=5904988

########################
### BOUNDARY CURRENT ###
########################

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

# Crop data to the Labrador Sea region
BCData.loc[BCData.loc[:,'Lat']>lab_N,:]=np.NaN
BCData.loc[BCData.loc[:,'Lat']<lab_S,:]=np.NaN
BCData.loc[BCData.loc[:,'Lon']>lab_E,:]=np.NaN
BCData.loc[BCData.loc[:,'Lon']<lab_W,:]=np.NaN
BCData=BCData.dropna()
new_ind=np.arange(BCData.shape[0])
BCData=BCData.set_index(pd.Index(list(new_ind)))

# Get lat-lon positions for this "good float"
lat_BC=BCData.loc[BCData.loc[:,'WMO']==goodfloat,'Lat'].to_numpy()
lon_BC=BCData.loc[BCData.loc[:,'WMO']==goodfloat,'Lon'].to_numpy()

## Make a piece-wise boundary current ##
break1_lon=-51
break1_flag=0
break1_count=0
break1_ind=np.NaN

break2_lat=57.5
break2_flag=0
break2_count=0
break2_ind=np.NaN

break3_lat=52
break3_flag=0
break3_count=0
break3_ind=np.NaN

while (break1_flag ==0 and break1_count<len(lon_BC)):
    
    if lon_BC[break1_count]<=break1_lon:
        break1_flag=1
        break1_ind=break1_count
    
    break1_count=break1_count+1

while (break2_flag ==0 and break2_count<len(lat_BC)):
    
    if lat_BC[break2_count]<=break2_lat:
        break2_flag=1
        break2_ind=break2_count
    
    break2_count=break2_count+1
    
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

buff1=1
buff2=2.3
buff3=1
buff4=1.6

BoundaryCurrent1=GoodPath1.buffer(buff1)
BoundaryCurrent2=GoodPath2.buffer(buff2)
BoundaryCurrent3=GoodPath3.buffer(buff3)
BoundaryCurrent4=GoodPath4.buffer(buff4)

########################
### LABRADOR GYRE ###
########################
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

LSGData=LSGData.dropna()
new_ind=np.arange(LSGData.shape[0])
LSGData=LSGData.set_index(pd.Index(list(new_ind)))
All_Lat_LSG=LSGData.loc[:,'Lat']
All_Lon_LSG=LSGData.loc[:,'Lon']

## Make the gyre ##
gyre_points=np.array((All_Lon_LSG, All_Lat_LSG)).T
hull = ConvexHull(gyre_points)
[hull_simp_unique, hs_ind]=np.unique(hull.simplices.flatten(),return_index=True)
split_lat1=57
split_lat2=56
ext_points=gyre_points[hull_simp_unique]
top_gyre=pd.DataFrame(ext_points[ext_points[:,1]>split_lat1]).sort_values(by=0).to_numpy()
ext_points2=ext_points[ext_points[:,1]<=split_lat1]
middle_gyre=pd.DataFrame(ext_points2[ext_points2[:,1]>split_lat2]).sort_values(by=0).to_numpy()
bot_gyre=pd.DataFrame(ext_points2[ext_points2[:,1]<=split_lat2]).sort_values(by=0,ascending=False).to_numpy()
all_gyre=np.append(top_gyre,middle_gyre, axis=0)
all_gyre=np.append(all_gyre,bot_gyre, axis=0)
annoying_point=all_gyre[8].T
all_gyre=np.delete(all_gyre,8,0)
all_gyre= np.concatenate((all_gyre, np.array([annoying_point.T])),axis=0)
Gyre=Polygon(all_gyre)

def BoundaryCurrent_Shape(Lon, Lat):
    # Given a profiles particular lat, lon pairing
    # Determine what profile are in the boundary current or gyre
    # bc_flag = 1 (profile in boundary current)
    # bc_flag = 0 (progile in gyre)
    # bc_flag = np.NaN (Not in either region)
    
    bc_flag = np.NaN
    p=Point(Lon,Lat)
                        
    # Check if in boundary current
    if p.within(BoundaryCurrent1) == True:
        bc_flag=1
    elif p.within(BoundaryCurrent2) == True:
        bc_flag=1
    elif p.within(BoundaryCurrent3) == True:
        bc_flag=1
    elif p.within(BoundaryCurrent4) == True:
        bc_flag=1
    # Check if in gyre
    elif p.within(Gyre) == True:
        bc_flag=0

    
    return bc_flag
    
    
    