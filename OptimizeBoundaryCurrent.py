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
import RandomFxns as RF
import time

# Make a shape representing the boundary current
# Use on float's path and add a spatial buffer?

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

## Load other gyre and boundary current float trajectories and check performance
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
## Load Float data
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

# Crop boundary current data
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

BottomCheckList=AllCheckList.iloc[bc_count:,:]

AllCheckList=AllCheckList.where(AllCheckList.iloc[:bc_count,3]<=lab_N, np.NaN)
AllCheckList=AllCheckList.where(AllCheckList.iloc[:bc_count,3]>=lab_S, np.NaN)
AllCheckList=AllCheckList.where(AllCheckList.iloc[:bc_count,4]<=lab_E, np.NaN)
AllCheckList=AllCheckList.where(AllCheckList.iloc[:bc_count,4]>=lab_W, np.NaN)
TopCheckList=AllCheckList.iloc[:bc_count,:].dropna()
AllCheckList=TopCheckList.append(BottomCheckList)
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

gyre_count=len(ftype)-bc_count

lat_bc=AllCheckList.iloc[:bc_count,3]
lon_bc=AllCheckList.iloc[:bc_count,4]
lat_lsg=AllCheckList.iloc[bc_count:,3]
lon_lsg=AllCheckList.iloc[bc_count:,4]

goodfloat=5904988

# Get lat-lon positions for this "good float"
lat_BC=BCData.loc[BCData.loc[:,'WMO']==goodfloat,'Lat'].to_numpy()
lon_BC=BCData.loc[BCData.loc[:,'WMO']==goodfloat,'Lon'].to_numpy()

########################
## Construct Regions! ##
########################

## Make the gyre ##
gyre_points=np.array((All_Lon_LSG, All_Lat_LSG)).T
hull = ConvexHull(gyre_points)
[hull_simp_unique, hs_ind]=np.unique(hull.simplices.flatten(),return_index=True)
#a=pd.DataFrame({'Vals':hull_simp_unique,'Ind':hs_ind})
#a=a.sort_values(by='Ind')
#HS_IND=a.loc[:,'Vals'].to_numpy()
split_lat1=57
split_lat2=56
ext_points=gyre_points[hull_simp_unique]
top_gyre=pd.DataFrame(ext_points[ext_points[:,1]>split_lat1]).sort_values(by=0).to_numpy()
ext_points2=ext_points[ext_points[:,1]<=split_lat1]
middle_gyre=pd.DataFrame(ext_points2[ext_points2[:,1]>split_lat2]).sort_values(by=0).to_numpy()
bot_gyre=pd.DataFrame(ext_points2[ext_points2[:,1]<=split_lat2]).sort_values(by=0,ascending=False).to_numpy()
# g_p=[[]]*len(hull_simp_unique)
# for i in np.arange(len(hull_simp_unique)):
#     s_ind=hull_simp_unique[i]
#     g_p[i]=np.array((gyre_points[s_ind, 0], gyre_points[s_ind, 1]))

all_gyre=np.append(top_gyre,middle_gyre, axis=0)
all_gyre=np.append(all_gyre,bot_gyre, axis=0)
annoying_point=all_gyre[8].T
all_gyre=np.delete(all_gyre,8,0)
all_gyre= np.concatenate((all_gyre, np.array([annoying_point.T])),axis=0)
Gyre=Polygon(all_gyre)
px,py=Gyre.exterior.xy
#plt.plot(px,py)

# GyreT=Polygon(top_gyre)
# GyreB=Polygon(bot_gyre)
# px1,py1=GyreT.exterior.xy
# px2,py2=GyreB.exterior.xy


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


## Iterate through buffer sizes
step_size=0.1
minVal=1.2
maxVal=2+step_size

# buff1=1.1
# buff2=2
# buff3=1.3
# buff4=1.3

buffer1=np.arange(1,1.5+step_size,step_size)
buffer2=np.arange(1.8,2.3+step_size,step_size)
buffer3=np.arange(1,1.5+step_size,step_size)
buffer4=np.arange(1,1.5+step_size,step_size)

total_itter=len(buffer1)*len(buffer2)*len(buffer3)*len(buffer4)
BC_1=np.zeros(total_itter)
BC_2=np.zeros(total_itter)
BC_3=np.zeros(total_itter)
BC_4=np.zeros(total_itter)

BC_BC=np.zeros(total_itter)
BC_LSG=np.zeros(total_itter)
LSG_BC=np.zeros(total_itter)
LSG_LSG=np.zeros(total_itter)

print('Starting Iterations')
print(total_itter,' Total iterations')
itter_counts=0
tic1=time.time()
tic2=time.time()
for buff1 in buffer1:
    tic_l2=time.time()
    for buff2 in buffer2:
        tic_l3=time.time()
        for buff3 in buffer3:
            tic_l4=time.time()
            for buff4 in buffer4:
                
                tic=time.time()
                BoundaryCurrent1=GoodPath1.buffer(buff1)
                BoundaryCurrent2=GoodPath2.buffer(buff2)
                BoundaryCurrent3=GoodPath3.buffer(buff3)
                BoundaryCurrent4=GoodPath4.buffer(buff4)
                
                # x1,y1 = BoundaryCurrent1.exterior.xy
                # x2,y2 = BoundaryCurrent2.exterior.xy
                # x3,y3 = BoundaryCurrent3.exterior.xy
                # x4,y4 = BoundaryCurrent4.exterior.xy
                toc=time.time()
                #print('Buffer:', toc-tic)
                #bc_included_bc=0
                #gyre_included_bc=0
                # Determine what percentage of points fall in boundary current
                
                # tic=time.time()
                # for i in np.arange(len(All_Lat_BC)):
                #     p=Point(All_Lon_BC[i], All_Lat_BC[i])
                    
                #     # Check BC
                #     if p.within(BoundaryCurrent1) == True:
                #         bc_included_bc=bc_included_bc+1
                #     elif p.within(BoundaryCurrent2) == True:
                #         bc_included_bc=bc_included_bc+1
                #     elif p.within(BoundaryCurrent3) == True:
                #         bc_included_bc=bc_included_bc+1
                #     elif p.within(BoundaryCurrent4) == True:
                #         bc_included_bc=bc_included_bc+1
                    
                    # # Check gyre
                    # if p.within(Gyre) == True:
                    #     gyre_included_bc=gyre_included_bc+1
                # toc=time.time()
                # print('BC In': toc-tic)       
                # ## Determine what percentage of gyre flots are in the BC
                
                # tic=time.time()
                #bc_included_lsg=0
                #gyre_included_lsg=0
                # # Determine what percentage of points fall in boundary current
                # for i in np.arange(len(All_Lat_LSG)):
                #     p=Point(All_Lon_LSG[i], All_Lat_LSG[i])
                #     if p.within(BoundaryCurrent1) == True:
                #         bc_included_lsg=bc_included_lsg+1
                #     elif p.within(BoundaryCurrent2) == True:
                #         bc_included_lsg=bc_included_lsg+1
                #     elif p.within(BoundaryCurrent3) == True:
                #         bc_included_lsg=bc_included_lsg+1
                #     elif p.within(BoundaryCurrent4) == True:
                #         bc_included_lsg=bc_included_lsg+1
                
                # toc=time.time()
                # print('Gyre In': toc-tic)        

                
                # Go through each point and check polygon
                bc_count_bc=0
                bc_count_lsg=0
                
                gyre_count_bc=0
                gyre_count_lsg=0
                
                tic=time.time()
                for i in np.arange(AllCheckList.shape[0]):
                    
                    p=Point(AllCheckList.iloc[i,4], AllCheckList.iloc[i,3])
                    
                    if AllCheckList.iloc[i,0] == 'BC':
                    
                        # Check if in boundary current
                        if p.within(BoundaryCurrent1) == True:
                            bc_count_bc=bc_count_bc+1
                        elif p.within(BoundaryCurrent2) == True:
                            bc_count_bc=bc_count_bc+1
                        elif p.within(BoundaryCurrent3) == True:
                            bc_count_bc=bc_count_bc+1
                        elif p.within(BoundaryCurrent4) == True:
                            bc_count_bc=bc_count_bc+1
                            
                        # Check if in gyre
                        if p.within(Gyre) == True:
                            gyre_count_bc=gyre_count_bc+1
                    else:
                        # Check if in boundary current
                        if p.within(BoundaryCurrent1) == True:
                            bc_count_lsg=bc_count_lsg+1
                        elif p.within(BoundaryCurrent2) == True:
                            bc_count_lsg=bc_count_lsg+1
                        elif p.within(BoundaryCurrent3) == True:
                            bc_count_lsg=bc_count_lsg+1
                        elif p.within(BoundaryCurrent4) == True:
                            bc_count_lsg=bc_count_lsg+1
                            
                        # Check if in gyre
                        if p.within(Gyre) == True:
                            gyre_count_lsg=gyre_count_lsg+1
            
                toc=time.time()
                #print('Within-Check:', toc-tic)
                    
                per_bc_bc_tot=np.round((bc_count_bc)/(bc_count)*100,2)
                per_bc_lsg_tot=np.round((bc_count_lsg)/(gyre_count)*100,2)
                per_lsg_bc_tot=np.round((gyre_count_bc)/(bc_count)*100,2)
                per_lsg_lsg_tot=np.round((gyre_count_lsg)/(gyre_count)*100,2)
                
                BC_1[itter_counts]=buff1
                BC_2[itter_counts]=buff2
                BC_3[itter_counts]=buff3
                BC_4[itter_counts]=buff4
                
                BC_BC[itter_counts]=per_bc_bc_tot
                BC_LSG[itter_counts]=per_bc_lsg_tot
                LSG_BC[itter_counts]=per_lsg_bc_tot
                LSG_LSG[itter_counts]=per_lsg_lsg_tot
                
                itter_counts=itter_counts+1
                
                if itter_counts%500==0:
                    print(itter_counts,' iterations completed')
                    print((total_itter-itter_counts),' iterations left')
                    print(np.round((itter_counts/total_itter)*100,2),'% Complete')
                    toc2=time.time()
                    print((toc2-tic1)/60,' TOTAL minutes have elapsed')
                    print((toc2-tic2)/60,' minutes have elapsed since last check') 
                    
                    tic2=time.time()
            
            toc_l4=time.time()
            #print('L4:', toc_l4-tic_l4)
        toc_l3=time.time()
        print('L3:', toc_l3-tic_l3)
    toc_l2=time.time()
    print('L2:', toc_l2-tic_l2)
            
ShapeChecks=pd.DataFrame({'BC_1':BC_1,'BC_2':BC_2,'BC_3':BC_3,'BC_4':BC_4
                          ,'BC_BC':BC_BC,'BC_LSG':BC_LSG,
                          'LSG_BC':LSG_BC,'LSG_LSG':LSG_LSG})
ShapeChecks.to_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BCBufferSizes.csv')
toc1=time.time()
print((toc1-tic1)/60,' TOTAL minutes have elapsed')

# #############################
# ## Calculate float speeds ##
# ############################
# bc_floatspeeds=np.zeros(len(All_Lat_BC))
# bc_floatspeeds[:]=np.NaN
# bc_scount=0

# for wmo in bc_floatlist:
    
#     lat_pos=BCData.loc[BCData.loc[:,'WMO']==wmo,'Lat']
#     lon_pos=BCData.loc[BCData.loc[:,'WMO']==wmo,'Lon']
#     time_pos=BCData.loc[BCData.loc[:,'WMO']==wmo,'Date']
    
#     indlist=lon_pos.index.to_list()
#     for j in np.arange(len(lat_pos)-1):
        
#         i1=indlist[j]
#         i2=indlist[j+1]
#         # Calculate change in distance
#         pos1=np.array((lon_pos[i1],lat_pos[i1]))
#         pos2=np.array((lon_pos[i2],lat_pos[i2]))
#         dist=haversine(pos1, pos2, unit='m')
#         #print(dist)
        
#         # Calculate change in time
#         t1=datetime.strptime(time_pos[i1], '%Y-%m-%d %H:%M:%S')
#         t2=datetime.strptime(time_pos[i2], '%Y-%m-%d %H:%M:%S')
#         tt=(t2-t1).total_seconds()
#         #print(tt)
        
#         bc_floatspeeds[bc_scount]=dist/tt
#         bc_scount=bc_scount+1
    
#     # plt.figure(1)
#     # plt.plot(bc_floatspeeds)

# ## Calculate float speeds
# lsg_floatspeeds=np.zeros(len(All_Lat_LSG))
# lsg_floatspeeds[:]=np.NaN
# lsg_scount=0

# for wmo in lsg_floatlist:
    
#     lat_pos=LSGData.loc[LSGData.loc[:,'WMO']==wmo,'Lat']
#     lon_pos=LSGData.loc[LSGData.loc[:,'WMO']==wmo,'Lon']
#     time_pos=LSGData.loc[LSGData.loc[:,'WMO']==wmo,'Date']
    
#     indlist=lon_pos.index.to_list()
#     for j in np.arange(len(lat_pos)-1):
        
#         i1=indlist[j]
#         i2=indlist[j+1]
#         # Calculate change in distance
#         pos1=np.array((lon_pos[i1],lat_pos[i1]))
#         pos2=np.array((lon_pos[i2],lat_pos[i2]))
#         dist=haversine(pos1, pos2, unit='m')
#         #print(dist)
        
#         # Calculate change in time
#         t1=datetime.strptime(time_pos[i1], '%Y-%m-%d %H:%M:%S')
#         t2=datetime.strptime(time_pos[i2], '%Y-%m-%d %H:%M:%S')
#         tt=(t2-t1).total_seconds()
#         #print(tt)
        
#         lsg_floatspeeds[lsg_scount]=dist/tt
#         lsg_scount=lsg_scount+1
    
#     # plt.figure(2)
#     # plt.plot(lsg_floatspeeds)
#     # plt.pause(2)
    
# lsg_floatspeeds = lsg_floatspeeds[~np.isnan(lsg_floatspeeds)]


# plt.figure()
# plt.hist(lsg_floatspeeds)
# plt.hist(bc_floatspeeds)
# plt.legend(['Gyre','BC'])

# print('\n%%%%%%%%%% RESULTS %%%%%%%%%%%\n')
# print('Boundary Current')
# print('Mean speed (m/s): ', np.nanmean(bc_floatspeeds))
# print('Std: ',np.nanstd(bc_floatspeeds))
# print('\nGyre')
# print('Mean speed (m/s): ', np.nanmean(lsg_floatspeeds))
# print('Std: ',np.nanstd(lsg_floatspeeds))


