#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:02:41 2021

@author: Ellen
"""

import pandas as pd
import gsw
import numpy as np
import cmocean.cm as cmo
import matplotlib.pyplot as plt

OutDir='/Users/Ellen/Desktop/GOSHIP_Data/NAtlantic_Sorted/'
GOSHIPData=pd.read_csv(OutDir+'QCFilteredData.csv')

FigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/GOSHIP_Compare/'

# Calculate density

SA=gsw.SA_from_SP(GOSHIPData.loc[:,'SAL'],GOSHIPData.loc[:,'PRES'],GOSHIPData.loc[:,'LONGITUDE'],GOSHIPData.loc[:,'LATITUDE'])
CT=gsw.CT_from_t(SA, GOSHIPData.loc[:,'TEMP'], GOSHIPData.loc[:,'PRES'])

Sigma=np.zeros(len(SA))
Sigma[:]=np.NaN

for i in np.arange(len(Sigma)):
    Sigma[i]=gsw.sigma0(SA[i], CT[i])
    
G_Spicy=gsw.spiciness0(SA,CT)
GOSHIPData['Sigma0']=Sigma
GOSHIPData['Spice']=G_Spicy
# Load Argo Data
ArgoFile='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/Spice.csv'
ArgoData=pd.read_csv(ArgoFile)

var_list=ArgoData.columns.to_list()
check=0
for i in var_list:
    if i == 'Spice':
        check =1

if check == 0:
    SA=gsw.SA_from_SP(ArgoData.loc[:,'Sal'],ArgoData.loc[:,'Pres'],ArgoData.loc[:,'Lon'],ArgoData.loc[:,'Lat'])
    CT=gsw.CT_from_t(SA,ArgoData.loc[:,'Temp'],ArgoData.loc[:,'Pres'])
    Spicy=gsw.spiciness0(SA,CT)
    ArgoData['Spice']=Spicy
    ArgoData.to_csv(ArgoFile)
    
minsig=27.8
maxP=ArgoData.loc[:,'Pres'].max()
GOSHIPData.loc[GOSHIPData.loc[:,'Sigma0']<minsig,'Sigma0']=np.NaN
GOSHIPData.loc[GOSHIPData.loc[:,'PRES']>maxP,'PRES']=np.NaN
GOSHIPData=GOSHIPData.dropna()
ArgoData.loc[ArgoData.loc[:,'Sigma0']<minsig,'Sigma0']=np.NaN
ArgoData=ArgoData.dropna()

minO=250
maxO=300

fig, axs = plt.subplots(1,2,figsize=(8,5))
ax=plt.subplot(1,2,1)
cm_cbr=ax.scatter(ArgoData.loc[:,'Temp'], ArgoData.loc[:,'Sal'], c=ArgoData.loc[:,'Oxy'], vmin=minO, vmax=maxO, marker ='o', s=2,cmap=cmo.matter)  #cmo.balance
ax.set_title('Argo Data')
ax.set_xlim((2.4,3.5))
ax.set_ylim((34.850,34.95))
fig.colorbar(cm_cbr)

ax=plt.subplot(1,2,2)
cm_cbr=ax.scatter(GOSHIPData.loc[:,'TEMP'], GOSHIPData.loc[:,'SAL'], c=GOSHIPData.loc[:,'OXY'], vmin=minO, vmax=maxO, marker ='o', s=2,cmap=cmo.matter)  #cmo.balance
ax.set_title('GO-SHIP Data')
fig.colorbar(cm_cbr)
ax.set_xlim((2.4,3.5))
ax.set_ylim((34.850, 34.95))
plt.savefig(FigDir+'Compare_GOSHIP_TSO_Sigma_27.8_Oxy.jpg')
plt.close()


## Oxy vs. Time
fig, axs = plt.subplots(1,2,figsize=(8,3))
ax=plt.subplot(1,2,1)
cm_cbr=ax.scatter(ArgoData.loc[:,'Date'],ArgoData.loc[:,'Pres'],c=ArgoData.loc[:,'Oxy'],vmin=minO, vmax=maxO,s=2,cmap=cmo.matter)
fig.colorbar(cm_cbr)
fig.gca().invert_yaxis()
ax.set_xticks([])

ax=plt.subplot(1,2,2)
cm_cbr=ax.scatter(GOSHIPData.loc[:,'DATE'],GOSHIPData.loc[:,'PRES'],c=GOSHIPData.loc[:,'OXY'],vmin=minO, vmax=maxO,s=2,cmap=cmo.matter)
fig.colorbar(cm_cbr)
fig.gca().invert_yaxis()
ax.set_xticks([])
plt.close()

## Just Argo
plt.figure(figsize=(10,8))
cm_cbr=plt.scatter(ArgoData.loc[:,'Date'],ArgoData.loc[:,'Pres'],c=ArgoData.loc[:,'Oxy'],vmin=minO, vmax=maxO,s=2,cmap=cmo.matter)
plt.colorbar(cm_cbr)
plt.gca().invert_yaxis()
plt.xticks([])
plt.savefig(FigDir+'Sigma_27.8_DatePresOxy.jpg')
plt.close()

plt.figure(figsize=(10,8))
cm_cbr=plt.scatter(ArgoData.loc[:,'Temp'],ArgoData.loc[:,'Sal'],c=ArgoData.loc[:,'Oxy'],vmin=minO, vmax=maxO,s=2,cmap=cmo.matter)
plt.colorbar(cm_cbr)
plt.savefig(FigDir+'Sigma_27.8_TSOxy.jpg')
plt.close()

plt.show()


