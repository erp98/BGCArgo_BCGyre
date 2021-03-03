#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:26:37 2021

@author: Ellen
"""
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmocean.cm as cmo
import cartopy as ct
import cartopy.crs as ccrs

# Labrador Sea Region
lab_N=65
lab_S=48
lab_E=-40
lab_W=-65

fsize_x=10
fsize_y=8

FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/FluxType_Monthly/'
# Load float flux data 
data_types=[0,1]
for data_i in data_types:
    if data_i == 0:
        # Load BC float data
        print('\n%% Loading BC Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_BCFloats_'
        filedir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/GasFluxData_BC.csv'
    elif data_i == 1:
        # Load Lab Gyre float data
        print('\n%% Loading Lab Sea Gyre Float Data %%\n')
        CSVDir_AS='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/O2Flux/AirSeaO2Flux_LabradorFloats_'
        filedir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/GasFluxData_Gyre.csv'
     
    AllData = pd.read_csv(filedir)
    AllData=AllData.drop('MLD', axis=1)
    # FileList=glob.glob(CSVDir_AS+'*.csv')
    # df_count = 0
    
    # file_count=0
    
    # for filedir in FileList:
        
    #     # Read in each data file
    #     fluxdata = pd.read_csv(filedir)

    #     # And combine into one big data frame
    #     if df_count == 0:
    #         AllData = pd.read_csv(filedir)
    #         df_count = 1
    #     else:
    #         AllData=AllData.append(fluxdata)
        
    #     file_count=file_count+1
    
    # If BC floats, crop data so it is in Lab Sea
    AllData.loc[AllData.loc[:,'Lat']>lab_N,:]=np.NaN
    AllData.loc[AllData.loc[:,'Lat']<lab_S,:]=np.NaN
    AllData.loc[AllData.loc[:,'Lon']>lab_E,:]=np.NaN
    AllData.loc[AllData.loc[:,'Lon']<lab_W,:]=np.NaN
    AllData=AllData.dropna()
    new_ind=np.arange(AllData.shape[0])
    AllData=AllData.set_index(pd.Index(list(new_ind)))
    
    Months=np.zeros(AllData.shape[0])
    Months[:]=np.NaN
    for i in np.arange(len(Months)):
        #Months[i]=int(str(AllData.iloc[i,2])[5:7])
        Months[i]=int(str(AllData.iloc[i,4])[5:7])
    
    AllData['Month']=Months
    
    if data_i == 0:
        data_BC=AllData
    elif data_i ==1:
        data_LSG=AllData

# Make subplots 
month_list=['January','February', 'March', 'April','May', 'June', 'July','August','September','October','November', 'December']

fig, axs = plt.subplots(4,3,figsize=(10,8))

minflux=-10**-6
maxflux=10**-6
m_size=5

cmap_choice=cmo.balance
#plt.figure(figsize=(fsize_x,fsize_y))
for i in np.arange(len(month_list)):
    r=i//3
    c=i%3
    m=i+1
    
    # Get BC data
    lon_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lon'].to_numpy()
    lat_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lat'].to_numpy()
    Ft_N16_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Ft_N16'].to_numpy()
    
    # Get LSG data
    lon_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lon'].to_numpy()
    lat_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lat'].to_numpy()
    Ft_N16_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Ft_N16'].to_numpy()
    
    # Plot data
    # BC: circles
    # LSG: triangles
    
    ax=plt.subplot(4,3,i+1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_BC, lat_BC, c=Ft_N16_BC, marker ='o',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice) #cmo.balance
    ax.scatter(lon_LSG, lat_LSG, c=Ft_N16_LSG, marker ='^',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice)
    ax.set_title(month_list[i])
    # cm_cbr=axs[r,c].scatter(lon_BC, lat_BC, c=Ft_N16_BC, marker ='o',vmin=minflux, vmax=maxflux, s=1,cmap=cmo.balance)
    # axs[r,c].scatter(lon_LSG, lat_LSG, c=Ft_N16_LSG, marker ='^',vmin=-minflux, vmax=maxflux, s=1,cmap=cmo.balance)
    # axs[r,c].set_title(month_list[i])

fig.colorbar(cm_cbr, ax=axs[:, :], location='right')
#plt.colorbar(cm_cbr,ax=ax, location = right)
plt.subplots_adjust(right=.75)#,hspace=0.9, wspace=0.4)
plt.savefig(FigDir+'Monthly.jpg')

# By Each Month

#plt.figure(figsize=(fsize_x,fsize_y))
for i in np.arange(len(month_list)):
    m=i+1
    
    fig, axs = plt.subplots(1,1,figsize=(10,6))
    
    # Get BC data
    lon_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lon'].to_numpy()
    lat_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lat'].to_numpy()
    Ft_N16_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Ft_N16'].to_numpy()
    
    # Get LSG data
    lon_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lon'].to_numpy()
    lat_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lat'].to_numpy()
    Ft_N16_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Ft_N16'].to_numpy()
    
    # Plot data
    # BC: circles
    # LSG: triangles
    
    ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_BC, lat_BC, c=Ft_N16_BC, marker ='o',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice) #cmo.balance
    ax.scatter(lon_LSG, lat_LSG, c=Ft_N16_LSG, marker ='^',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice)
    ax.set_title(month_list[i])
    # cm_cbr=axs[r,c].scatter(lon_BC, lat_BC, c=Ft_N16_BC, marker ='o',vmin=minflux, vmax=maxflux, s=1,cmap=cmo.balance)
    # axs[r,c].scatter(lon_LSG, lat_LSG, c=Ft_N16_LSG, marker ='^',vmin=-minflux, vmax=maxflux, s=1,cmap=cmo.balance)
    # axs[r,c].set_title(month_list[i])
    
    fig.colorbar(cm_cbr)
    plt.savefig(FigDir+'Month_'+str(i)+'_'+month_list[i]+'.jpg')
    plt.close()

# By Season
Seasons=[[1,2,12],[3,4,5],[6,7,8],[9,10,11]]
SeasonType=['Winter','Spring','Summer','Fall']

fig, axs = plt.subplots(2,2,figsize=(10,8))


for h in np.arange(len(Seasons)):
    s=Seasons[h]
    m_count=0
    
    lon_BC_tot=np.zeros(1)
    lat_BC_tot=np.zeros(1)
    Ft_N16_BC_tot=np.zeros(1)
    lon_LSG_tot=np.zeros(1)
    lat_LSG_tot=np.zeros(1)
    Ft_N16_LSG_tot=np.zeros(1)

    for i in s:
        m=i
        # Get BC data
        lon_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lat'].to_numpy()
        Ft_N16_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Ft_N16'].to_numpy()
        
        # Get LSG data
        lon_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lat'].to_numpy()
        Ft_N16_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Ft_N16'].to_numpy()
        
        lon_BC_tot=np.concatenate((lon_BC_tot,lon_BC))
        lat_BC_tot=np.concatenate((lat_BC_tot,lat_BC))
        Ft_N16_BC_tot=np.concatenate((Ft_N16_BC_tot,Ft_N16_BC))
        lon_LSG_tot=np.concatenate((lon_LSG_tot,lon_LSG))
        lat_LSG_tot=np.concatenate((lat_LSG_tot,lat_LSG))
        Ft_N16_LSG_tot=np.concatenate((Ft_N16_LSG_tot,Ft_N16_LSG))
    
    lon_BC_tot=lon_BC_tot[1:]
    lat_BC_tot=lat_BC_tot[1:]
    Ft_N16_BC_tot=Ft_N16_BC_tot[1:]
    lon_LSG_tot=lon_LSG_tot[1:]
    lat_LSG_tot=lat_LSG_tot[1:]
    Ft_N16_LSG_tot=Ft_N16_LSG_tot[1:]

    ax=plt.subplot(2,2,h+1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_BC_tot, lat_BC_tot, c=Ft_N16_BC_tot, marker ='o',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice) #cmo.balance
    ax.scatter(lon_LSG_tot, lat_LSG_tot, c=Ft_N16_LSG_tot, marker ='^',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice)
    ax.set_title(SeasonType[h])

fig.colorbar(cm_cbr, ax=axs[:, :], location='right')
#plt.colorbar(cm_cbr,ax=ax, location = right)
plt.subplots_adjust(right=.75)#,hspace=0.9, wspace=0.4)
plt.savefig(FigDir+'Seasonaly.jpg')    

for h in np.arange(len(Seasons)):
    s=Seasons[h]
    m_count=0
    fig, axs = plt.subplots(1,1,figsize=(10,6))
    
    lon_BC_tot=np.zeros(1)
    lat_BC_tot=np.zeros(1)
    Ft_N16_BC_tot=np.zeros(1)
    lon_LSG_tot=np.zeros(1)
    lat_LSG_tot=np.zeros(1)
    Ft_N16_LSG_tot=np.zeros(1)

    for i in s:
        m=i
        # Get BC data
        lon_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Lat'].to_numpy()
        Ft_N16_BC=data_BC.loc[data_BC.loc[:,'Month']==m,'Ft_N16'].to_numpy()
        
        # Get LSG data
        lon_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lon'].to_numpy()
        lat_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Lat'].to_numpy()
        Ft_N16_LSG=data_LSG.loc[data_LSG.loc[:,'Month']==m,'Ft_N16'].to_numpy()
        
        lon_BC_tot=np.concatenate((lon_BC_tot,lon_BC))
        lat_BC_tot=np.concatenate((lat_BC_tot,lat_BC))
        Ft_N16_BC_tot=np.concatenate((Ft_N16_BC_tot,Ft_N16_BC))
        lon_LSG_tot=np.concatenate((lon_LSG_tot,lon_LSG))
        lat_LSG_tot=np.concatenate((lat_LSG_tot,lat_LSG))
        Ft_N16_LSG_tot=np.concatenate((Ft_N16_LSG_tot,Ft_N16_LSG))
    
    lon_BC_tot=lon_BC_tot[1:]
    lat_BC_tot=lat_BC_tot[1:]
    Ft_N16_BC_tot=Ft_N16_BC_tot[1:]
    lon_LSG_tot=lon_LSG_tot[1:]
    lat_LSG_tot=lat_LSG_tot[1:]
    Ft_N16_LSG_tot=Ft_N16_LSG_tot[1:]
     
    ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines('50m')
    ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
    cm_cbr=ax.scatter(lon_BC_tot, lat_BC_tot, c=Ft_N16_BC_tot, marker ='o',vmin=minflux, vmax=maxflux, s=m_size,cmap=cmap_choice) #cmo.balance
    ax.scatter(lon_LSG_tot, lat_LSG_tot, c=Ft_N16_LSG_tot, marker ='^',s=m_size,cmap=cmap_choice,vmin=minflux, vmax=maxflux)
    ax.set_title(SeasonType[h])
    
    fig.colorbar(cm_cbr)
    plt.savefig(FigDir+'Season_'+str(h)+'_'+SeasonType[h]+'.jpg')
    plt.clf(); plt.close()

plt.show()
    

    
    