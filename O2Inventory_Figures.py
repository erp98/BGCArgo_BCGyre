#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:24:16 2021

@author: Ellen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean.cm as cmo
import cartopy as ct
import cartopy.crs as ccrs
import cmocean.cm as cmo
import BCClassFxns as BC

lab_N=65
lab_S=48
lab_E=-45
lab_W=-65

PresLevel=100
FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/O2Inventory/Pres_'+str(PresLevel)+'_'
cmap_choice=cmo.dense 
#cmap_choice='winter'

minval=25
maxval=35

mindiff=-3
maxdiff=3

badfloat=4901141

O2invData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/Pres_'+str(PresLevel)+'_O2WaterColumn_1m.csv')
O2invData.loc[O2invData.loc[:,'O2Inv']<0,'O2Inv']=np.NaN
O2invData.loc[O2invData.loc[:,'WMO']==badfloat,:]=np.NaN
O2invData=O2invData.dropna()
new_ind=np.arange(O2invData.shape[0])
O2invData=O2invData.set_index(pd.Index(list(new_ind)))
    

# Add months
Months=np.zeros(O2invData.shape[0])
Months[:]=np.NaN
for i in np.arange(len(Months)):
    #Months[i]=int(str(AllData.iloc[i,2])[5:7])
    Months[i]=int(str(O2invData.iloc[i,3])[5:7])
    
O2invData['Month']=Months

lon=O2invData.loc[:,'Lon']
lat=O2invData.loc[:,'Lat']
O2inv=O2invData.loc[:,'O2Inv']
O2eqinv=O2invData.loc[:,'O2EqInv']
DiffInv=O2inv-O2eqinv
O2invData['DiffEq']=DiffInv

fig, axs = plt.subplots(1,1,figsize=(10,6))
ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax.coastlines('50m')
ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
cm_cbr=ax.scatter(lon, lat, c=O2inv, vmin=minval, vmax=maxval, marker ='o', s=2,cmap=cmap_choice) #cmo.balance
ax.set_title('Total Oxygen Inventory (mol/m2)')
fig.colorbar(cm_cbr)
plt.savefig(FigDir+'Map_Inventory_Total.jpg')
plt.close()

fig, axs = plt.subplots(1,1,figsize=(10,6))
ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
ax.coastlines('50m')
ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
cm_cbr=ax.scatter(lon, lat, c=DiffInv,vmin=mindiff, vmax=maxdiff, marker ='o', s=2,cmap=cmo.balance) #cmo.balance
ax.set_title('Difference From Equilibrium Oxygen Inventory (mol/m2)')
fig.colorbar(cm_cbr)
plt.savefig(FigDir+'Map_Inventory_DiffEq_Total.jpg')

month_list=['January','February', 'March', 'April','May', 'June', 'July','August','September','October','November', 'December']


## Plot by month
for i in np.arange(len(month_list)):
    m=i+1
    
    fig, axs = plt.subplots(1,1,figsize=(10,6))
    
    # Get BC data
    lon_m=O2invData.loc[O2invData.loc[:,'Month']==m,'Lon'].to_numpy()
    lat_m=O2invData.loc[O2invData.loc[:,'Month']==m,'Lat'].to_numpy()
    O2inv_m=O2invData.loc[O2invData.loc[:,'Month']==m,'O2Inv'].to_numpy()
    
    
    ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_m, lat_m, c=O2inv_m, vmin=minval, vmax=maxval, marker ='o', s=2,cmap=cmap_choice) #cmo.balance
    ax.set_title(month_list[i])
    fig.colorbar(cm_cbr)
    plt.savefig(FigDir+'Map_Month_'+str(i)+'_'+month_list[i]+'.jpg')
    plt.close()
    
  
fig, axs = plt.subplots(4,3,figsize=(10,8))
for i in np.arange(len(month_list)):
    r=i//3
    c=i%3
    m=i+1
    
    # Get BC data
    lon_m=O2invData.loc[O2invData.loc[:,'Month']==m,'Lon'].to_numpy()
    lat_m=O2invData.loc[O2invData.loc[:,'Month']==m,'Lat'].to_numpy()
    O2inv_m=O2invData.loc[O2invData.loc[:,'Month']==m,'O2Inv'].to_numpy()

    ax=plt.subplot(4,3,i+1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_m, lat_m, c=O2inv_m, vmin=minval, vmax=maxval, marker ='o', s=2,cmap=cmap_choice)  #cmo.balance
    ax.set_title(month_list[i])

fig.colorbar(cm_cbr, ax=axs[:, :], location='right')
plt.suptitle('Oxygen Inventory (mol/m2)')
#plt.colorbar(cm_cbr,ax=ax, location = right)
plt.subplots_adjust(right=.75)#,hspace=0.9, wspace=0.4)
plt.savefig(FigDir+'Map_Monthly.jpg')
plt.close()

## Plot by season
Seasons=[[1,2,12],[3,4,5],[6,7,8],[9,10,11]]
SeasonType=['Winter','Spring','Summer','Fall']

## Inventroy
fig, axs = plt.subplots(2,2,figsize=(10,8))
for h in np.arange(len(Seasons)):
    s=Seasons[h]
    m_count=0
    
    lon_s_tot=np.zeros(1)
    lat_s_tot=np.zeros(1)
    O2inv_s_tot=np.zeros(1)
    O2diff_s_tot=np.zeros(1)
    
    for i in s:
        m=i
        # Get BC data
        lon_s=O2invData.loc[O2invData.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_s=O2invData.loc[O2invData.loc[:,'Month']==m,'Lat'].to_numpy()
        O2inv_s=O2invData.loc[O2invData.loc[:,'Month']==m,'O2Inv'].to_numpy()
        O2_diff_s=O2invData.loc[O2invData.loc[:,'Month']==m,'DiffEq'].to_numpy()
        
        lon_s_tot=np.concatenate((lon_s_tot,lon_s))
        lat_s_tot=np.concatenate((lat_s_tot,lat_s))
        O2inv_s_tot=np.concatenate((O2inv_s_tot,O2inv_s))
        O2diff_s_tot=np.concatenate((O2diff_s_tot,O2_diff_s,))
        
    lon_s_tot=lon_s_tot[1:]
    lat_s_tot=lat_s_tot[1:]
    O2inv_s_tot=O2inv_s_tot[1:]
    O2diff_s_tot=O2diff_s_tot[1:]
    
    ax=plt.subplot(2,2,h+1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_s_tot, lat_s_tot, c=O2inv_s_tot, vmin=minval, vmax=maxval, marker ='o', s=2,cmap=cmap_choice)  #cmo.balance
    ax.set_title(SeasonType[h])

fig.colorbar(cm_cbr, ax=axs[:, :], location='right')
plt.suptitle('Oxygen Inventory (mol/m2)')
#plt.colorbar(cm_cbr,ax=ax, location = right)
plt.subplots_adjust(right=.75)#,hspace=0.9, wspace=0.4)
plt.savefig(FigDir+'Map_Seasonaly.jpg')
plt.close()

## DIff Inventory
fig, axs = plt.subplots(2,2,figsize=(10,8))
for h in np.arange(len(Seasons)):
    s=Seasons[h]
    m_count=0
    
    lon_s_tot=np.zeros(1)
    lat_s_tot=np.zeros(1)
    O2inv_s_tot=np.zeros(1)
    O2diff_s_tot=np.zeros(1)
    
    for i in s:
        m=i
        # Get BC data
        lon_s=O2invData.loc[O2invData.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_s=O2invData.loc[O2invData.loc[:,'Month']==m,'Lat'].to_numpy()
        O2_diff_s=O2invData.loc[O2invData.loc[:,'Month']==m,'DiffEq'].to_numpy()
        
        lon_s_tot=np.concatenate((lon_s_tot,lon_s))
        lat_s_tot=np.concatenate((lat_s_tot,lat_s))
        O2diff_s_tot=np.concatenate((O2diff_s_tot,O2_diff_s,))
        
    lon_s_tot=lon_s_tot[1:]
    lat_s_tot=lat_s_tot[1:]
    O2diff_s_tot=O2diff_s_tot[1:]
    
    ax=plt.subplot(2,2,h+1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_s_tot, lat_s_tot, c=O2diff_s_tot, vmin=mindiff, vmax=maxdiff, marker ='o', s=2,cmap=cmo.balance)  #cmo.balance
    ax.set_title(SeasonType[h])

fig.colorbar(cm_cbr, ax=axs[:, :], location='right')
plt.suptitle('Difference Equilibrium Oxygen Inventory (mol/m2)')
#plt.colorbar(cm_cbr,ax=ax, location = right)
plt.subplots_adjust(right=.75)#,hspace=0.9, wspace=0.4)
plt.savefig(FigDir+'Map_DiffEq_Seasonaly.jpg')
plt.close()

for h in np.arange(len(Seasons)):
    s=Seasons[h]
    m_count=0
    
    fig, axs = plt.subplots(1,1,figsize=(10,6))
    
    lon_s_tot=np.zeros(1)
    lat_s_tot=np.zeros(1)
    O2inv_s_tot=np.zeros(1)

    for i in s:
        m=i
        # Get BC data
        lon_s=O2invData.loc[O2invData.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_s=O2invData.loc[O2invData.loc[:,'Month']==m,'Lat'].to_numpy()
        O2inv_s=O2invData.loc[O2invData.loc[:,'Month']==m,'O2Inv'].to_numpy()
        
        lon_s_tot=np.concatenate((lon_s_tot,lon_s))
        lat_s_tot=np.concatenate((lat_s_tot,lat_s))
        O2inv_s_tot=np.concatenate((O2inv_s_tot,O2inv_s))
    
    lon_s_tot=lon_s_tot[1:]
    lat_s_tot=lat_s_tot[1:]
    O2inv_s_tot=O2inv_s_tot[1:]
     
    ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_s_tot, lat_s_tot, c=O2inv_s_tot, vmin=minval, vmax=maxval, marker ='o', s=2,cmap=cmap_choice)
    ax.set_title(SeasonType[h])
    
    fig.colorbar(cm_cbr)
    plt.suptitle('Oxygen Inventory (mol/m2)')
    plt.savefig(FigDir+'Map_Season_'+str(h)+'_'+SeasonType[h]+'.jpg')
    plt.clf(); plt.close()
    
## Make time sereis by month for BC versus gyre (generally)
bc_flag=np.zeros(O2invData.shape[0])
bc_flag[:]=np.NaN

for i in np.arange(O2invData.shape[0]):
    bc_flag[i]=BC.BoundaryCurrent_Bath(O2invData.iloc[i,5],O2invData.iloc[i,4])

O2invData['BCFlag']=bc_flag

BCData=O2invData.loc[O2invData.loc[:,'BCFlag']==1,:]
GData=O2invData.loc[O2invData.loc[:,'BCFlag']==0,:]


BC_Mean=BCData.groupby(by='Month').mean()
BC_Std=BCData.groupby(by='Month').std()

G_Mean=GData.groupby(by='Month').mean()
G_Std=GData.groupby(by='Month').std()

MonthList=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']

x=np.arange(len(MonthList))
width = 0.35  # the width of the bars

############## L13 ##################
fig, ax = plt.subplots(figsize=(10,8))
rects1 = ax.bar(x - width/2, BC_Mean.loc[:,'O2Inv'], width,yerr=BC_Std.loc[:,'O2Inv'],capsize=2, label='BC')
rects2 = ax.bar(x + width/2, G_Mean.loc[:,'O2Inv'], width,yerr=G_Std.loc[:,'O2Inv'],capsize=2,label='LSG')
ax.set_ylabel('O2 Inventory (mol/m2)')
#ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
#plt.ylim((400,600))
plt.savefig(FigDir+'TimeSeries_Monthly.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(10,8))
rects1 = ax.bar(x - 3*width/4, BC_Mean.loc[:,'O2Inv'], width/2,yerr=BC_Std.loc[:,'O2Inv'],capsize=2, label='BC')
rects3 = ax.bar(x - width/4, G_Mean.loc[:,'O2Inv'], width/2,yerr=G_Std.loc[:,'O2Inv'],capsize=2,label='LSG')
rects2 = ax.bar(x + width/4, BC_Mean.loc[:,'O2EqInv'], width/2,yerr=BC_Std.loc[:,'O2EqInv'],capsize=2, label='BC Eq')
rects4 = ax.bar(x + 3*width/4, G_Mean.loc[:,'O2EqInv'], width/2,yerr=G_Std.loc[:,'O2EqInv'],capsize=2,label='LSG Eq')
ax.set_ylabel('O2 Inventory (mol/m2)')
#ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
#plt.ylim((400,600))
plt.savefig(FigDir+'TimeSeries_Monthly_wEq.jpg')
plt.close()

fig, ax = plt.subplots(figsize=(10,8))
rects1 = ax.bar(x - width/2, BC_Mean.loc[:,'DiffEq'], width,yerr=BC_Std.loc[:,'DiffEq'],capsize=2, label='BC')
rects2 = ax.bar(x + width/2, G_Mean.loc[:,'DiffEq'], width,yerr=G_Std.loc[:,'DiffEq'],capsize=2,label='LSG')
ax.set_ylabel('Difference O2 Inventory (mol/m2)')
#ax.set_title('Monthly Total Air Sea Gas Flux (L13) - Raw')
ax.set_xticks(x)
ax.set_xticklabels(MonthList)
ax.legend()
plt.savefig(FigDir+'TimeSeries_DiffEq_Monthly.jpg')
plt.close()



