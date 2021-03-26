#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:05:41 2021

@author: Ellen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import gsw
import cartopy as ct
import cartopy.crs as ccrs
import time

FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Spice/'

lab_N=65
lab_S=48
lab_E=-45
lab_W=-65

mindense=27
maxdense=28

dense_ss=0.05

dense_range=np.arange(mindense, maxdense+dense_ss, dense_ss)

minO=250
maxO=400

minOsat=80
maxOsat=120

figx=10
figy=8

minT=0
maxT=12

minS=34
maxS=36


# Load data
tic1=time.time()
SpiceFile='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/Spice.csv'
SpiceData=pd.read_csv(SpiceFile)

var_list=SpiceData.columns.to_list()
check=0
for i in var_list:
    if i == 'Spice':
        check =1

if check == 0:
    SA=gsw.SA_from_SP(SpiceData.loc[:,'Sal'],SpiceData.loc[:,'Pres'],SpiceData.loc[:,'Lon'],SpiceData.loc[:,'Lat'])
    CT=gsw.CT_from_t(SA,SpiceData.loc[:,'Temp'],SpiceData.loc[:,'Pres'])
    Spicy=gsw.spiciness0(SA,CT)
    SpiceData['Spice']=Spicy
    SpiceData.to_csv(SpiceFile)

# General plots

for i in np.arange(len(dense_range)-1):
    tic2=time.time()
    lower_bound=np.round(dense_range[i],3)
    upper_bound=np.round(dense_range[i+1],3)
    print('\nDense range: ',lower_bound,'-',upper_bound)
    
    # Load data
    SpiceData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/Spice.csv')
    
    # Crop data to only look at data along specific isopycnals
    SpiceData.loc[SpiceData.loc[:,'Sigma0']<lower_bound,'Sigma0']=np.NaN
    SpiceData.loc[SpiceData.loc[:,'Sigma0']>=upper_bound,'Sigma0']=np.NaN
    
    SpiceData=SpiceData.dropna()
    
    if SpiceData.shape[0] >0:
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Temp'],SpiceData.loc[:,'Sal'],c=SpiceData.loc[:,'OxySat'],vmin=minOsat, vmax=maxOsat,cmap=cmo.balance)
        plt.colorbar()
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_TSOxySat.jpg')
        plt.clf(); plt.close()
        
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Temp'],SpiceData.loc[:,'Sal'],c=SpiceData.loc[:,'Oxy'],vmin=minO, vmax=maxO,cmap=cmo.matter)
        plt.colorbar()
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_TSOxy.jpg')
        plt.clf(); plt.close()
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Temp'],SpiceData.loc[:,'Sal'],c=SpiceData.loc[:,'Spice'],vmin=-2, vmax=2,cmap=cmo.balance)
        plt.colorbar()
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_TSSpice.jpg')
        plt.clf(); plt.close()
        
        num_ticks=20
        dates_total=SpiceData.loc[:,'Date']
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Date'],SpiceData.loc[:,'Sigma0'],c=SpiceData.loc[:,'OxySat'],vmin=minOsat, vmax=maxOsat,cmap=cmo.balance)
        plt.colorbar()
        plt.xticks([])
        # new_locs=[]
        # new_labels=[]
        # xticks_ind=np.arange(0,len(locs), step=len(locs)//num_ticks)
        # for i in xticks_ind:
        #     new_locs=new_locs+[locs[i]]
        #     new_labels=new_labels+[labels[i]]
        # plt.xticks(new_locs,new_labels)
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_Sigma0DateOxySat.jpg')
        plt.clf(); plt.close()
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Date'],SpiceData.loc[:,'Sigma0'],c=SpiceData.loc[:,'Oxy'],vmin=minO, vmax=maxO,cmap=cmo.matter)
        plt.colorbar()
        plt.xticks([])
        # new_locs=[]
        # new_labels=[]
        # xticks_ind=np.arange(0,len(locs), step=len(locs)//num_ticks)
        # for i in xticks_ind:
        #     new_locs=new_locs+[locs[i]]
        #     new_labels=new_labels+[labels[i]]
        # plt.xticks(new_locs,new_labels)
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_Sigma0DateOxy.jpg')
        plt.clf(); plt.close()
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Date'],SpiceData.loc[:,'Sigma0'],c=SpiceData.loc[:,'Spice'],vmin=-2, vmax=2,cmap=cmo.balance)
        plt.colorbar()
        plt.xticks([])
        # new_locs=[]
        # new_labels=[]
        # xticks_ind=np.arange(0,len(locs), step=len(locs)//num_ticks)
        # for i in xticks_ind:
        #     new_locs=new_locs+[locs[i]]
        #     new_labels=new_labels+[labels[i]]
        # plt.xticks(new_locs,new_labels)
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_Sigma0DateSpice.jpg')
        plt.clf(); plt.close()
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Date'],SpiceData.loc[:,'Pres'],c=SpiceData.loc[:,'OxySat'],vmin=minOsat, vmax=maxOsat,cmap=cmo.balance)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_PresDateOxySat.jpg')
        plt.clf(); plt.close()
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Date'],SpiceData.loc[:,'Pres'],c=SpiceData.loc[:,'Oxy'],vmin=minO, vmax=maxO,cmap=cmo.matter)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_PresDateOxy.jpg')
        plt.clf(); plt.close()
        
        plt.figure(figsize=(figx, figy))
        plt.scatter(SpiceData.loc[:,'Date'],SpiceData.loc[:,'Pres'],c=SpiceData.loc[:,'Spice'],vmin=-2, vmax=2,cmap=cmo.balance)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.title('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_PresDateSpice.jpg')
        plt.clf(); plt.close()
        
        # # Calculate oxygen time series along each isopycnal
        unique_dates=SpiceData.loc[:,'Date'].unique()
        oxy_inventory=np.zeros(len(unique_dates))
        oxy_inventory[:]=np.NaN
        
        for dd in np.arange(len(unique_dates)):
            prof_date=unique_dates[dd]
            # Get pressure,density, and oxygen slices
            SubSet_Data=SpiceData.loc[SpiceData.loc[:,'Date']==prof_date,:]
            pres=SubSet_Data.loc[:,'Pres'].to_numpy()*10000
            dense=SubSet_Data.loc[:,'Sigma0'].to_numpy()+1000
            oxy=SubSet_Data.loc[:,'Oxy'].to_numpy()
            
            depth=pres/dense/9.81 # m 
            
            if len(depth)>1:
                ox_in_t=np.zeros(len(depth)-1)
                ox_in_t[:]=np.NaN
                for j in np.arange(len(depth)-1):
                    deltah=depth[j+1]-depth[j]
                    # Use average of 2 oxygen values
                    oxy_val=np.nanmean((oxy[j],oxy[j+1]))
                    dense_val=np.nanmean((dense[j],dense[j+1]))
                    
                    ox_in_t[j]=deltah*oxy_val*dense_val # m * µmol/kg * kg/m^3
                
                oxy_inventory[dd]=np.nansum(ox_in_t)*10**-6

        
        plt.figure(figsize=(figx, figy))
        plt.plot(unique_dates,oxy_inventory)
        plt.xticks([])
        plt.ylabel('Oxygen Inventory (mol/m2)')
        plt.title('Oxygen Inventory Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_OxygenInventory.jpg')
        plt.clf(); plt.close()
        
        MeanData=SpiceData.groupby(by='Date').mean()
        # Spice maps
        
        ## Oxygen ##
        fig, axs = plt.subplots(2,3,figsize=(10,6))
        
        ax=plt.subplot(2,3,1,projection=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
        cm_cbr=ax.scatter(MeanData.loc[:,'Lon'], MeanData.loc[:,'Lat'], c=MeanData.loc[:,'Temp'], vmin=minT, vmax=maxT, marker ='o', s=2,cmap=cmo.thermal)  #cmo.balance
        ax.set_title('Temperature')
        fig.colorbar(cm_cbr)
        
        ax=plt.subplot(2,3,2,projection=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
        cm_cbr=ax.scatter(MeanData.loc[:,'Lon'], MeanData.loc[:,'Lat'], c=MeanData.loc[:,'Oxy'], vmin=minO, vmax=maxO, marker ='o', s=2,cmap=cmo.matter)  #cmo.balance
        ax.set_title('Oxygen')
        fig.colorbar(cm_cbr)
        
        ax=plt.subplot(2,3,3,projection=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
        cm_cbr=ax.scatter(MeanData.loc[:,'Lon'], MeanData.loc[:,'Lat'], c=MeanData.loc[:,'Pres'], vmin=0, vmax=2000, marker ='o', s=2,cmap=cmo.haline)  #cmo.balance
        ax.set_title('Pressure')
        fig.colorbar(cm_cbr)
        
        ax=plt.subplot(2,3,4,projection=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
        cm_cbr=ax.scatter(MeanData.loc[:,'Lon'], MeanData.loc[:,'Lat'], c=MeanData.loc[:,'Sal'], vmin=minS, vmax=minS, marker ='o', s=2,cmap=cmo.haline)  #cmo.balance
        ax.set_title('Salinity')
        fig.colorbar(cm_cbr)
        
        ## Oxy Sat
        ax=plt.subplot(2,3,5,projection=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
        cm_cbr=ax.scatter(MeanData.loc[:,'Lon'], MeanData.loc[:,'Lat'], c=MeanData.loc[:,'OxySat'], vmin=minOsat, vmax=maxOsat, marker ='o', s=2,cmap=cmo.balance)  #cmo.balance
        ax.set_title('Oxygen Saturation')
        fig.colorbar(cm_cbr)
        
        ax=plt.subplot(2,3,6,projection=ccrs.PlateCarree())
        ax.coastlines('50m')
        ax.set_extent([lab_W, lab_E, lab_S, lab_N], ccrs.PlateCarree())
        cm_cbr=ax.scatter(MeanData.loc[:,'Lon'], MeanData.loc[:,'Lat'], c=MeanData.loc[:,'Spice'], vmin=-2, vmax=2, marker ='o', s=2,cmap=cmo.balance)  #cmo.balance
        ax.set_title('Spiciness')
        fig.colorbar(cm_cbr)
        
        
        plt.suptitle('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Maps_Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_SpiceOxyP.jpg')
        plt.clf(); plt.close()
        
        ## Plot things versus Spice
        fig, axs = plt.subplots(1,2,figsize=(10,8))
        ax=plt.subplot(1,2,1)
        ax.scatter(SpiceData.loc[:,'Spice'],SpiceData.loc[:,'Oxy'])
        ax.set_xlabel('Spice')
        ax.set_ylabel('Oxygen (µmol/kg)')
        
        ax=plt.subplot(1,2,2)
        ax.scatter(SpiceData.loc[:,'Spice'],SpiceData.loc[:,'OxySat'])
        ax.set_xlabel('Spice')
        ax.set_ylabel('Oxygen Saturation (%)')
        
        plt.suptitle('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
        plt.savefig(FigDir+'Sigma0_'+str(lower_bound)+'_'+str(upper_bound)+'_SpicevOxyVals.jpg')
        plt.clf(); plt.close()
        
        toc2=time.time()
        print('Minutes Elapsed: ', (toc2-tic2)/60)

toc1=time.time()
print('\nTOTAL TIME ELAPSED (mins): ', (toc1-tic1)/60)

SpiceData=pd.read_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/Spice.csv')
SpiceData.loc[SpiceData.loc[:,'Oxy']==0,'Oxy']=np.NaN
SpiceData=SpiceData.dropna()

fig, axs = plt.subplots(1,2,figsize=(10,8))
ax=plt.subplot(1,2,1)
ax.scatter(SpiceData.loc[:,'Spice'],SpiceData.loc[:,'Oxy'])
ax.set_xlabel('Spice')
ax.set_ylabel('Oxygen (µmol/kg)')

ax=plt.subplot(1,2,2)
ax.scatter(SpiceData.loc[:,'Spice'],SpiceData.loc[:,'OxySat'])
ax.set_xlabel('Spice')
ax.set_ylabel('Oxygen Saturation (%)')

plt.suptitle('Sigma0 Range: '+str(lower_bound)+'-'+str(upper_bound))
plt.savefig(FigDir+'All_SpicevOxyVals.jpg')
plt.clf(); plt.close()