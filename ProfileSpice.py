#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:32:13 2021

@author: Ellen
"""

import xarray as xr
from datetime import datetime#, timedelta, timezone
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gsw
import RandomFxns as RF
import BCClassFxns as BCF
import time
import gsw
import cmocean.cm as cmo
from scipy import interpolate
import RandomFxns as RF

Argofiles='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/AdjDataFlags.csv'

SectionFigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Sections/'

ArgoDataFrames=pd.read_csv(Argofiles)
ArgoDataFrames=ArgoDataFrames.dropna()
new_ind=np.arange(ArgoDataFrames.shape[0])
ArgoDataFrames=ArgoDataFrames.set_index(pd.Index(list(new_ind)))
    
ArgoWMO=ArgoDataFrames.loc[:,'FloatWMO']
FileNames=ArgoDataFrames.loc[:,'FileName']

good_QC=[1,2,5,8]
lat_N=70.000
lat_S= 48.00
lon_E= -45.00
lon_W= -70.00

figx=10
figy=8

minT=0
maxT=12

minS=34
maxS=36

minO=250
maxO=400

minOsat=80
maxOsat=120

mindense=27
maxdense=28
dense_ss=0.05
dense_bins=np.arange(mindense, maxdense+dense_ss,dense_ss)

step_size=10
pres_range=np.arange(0,2001,step_size)

refdate=datetime(1990,1,1)

TSOsat=pd.DataFrame({'WMO': [np.NaN],'Date':[np.NaN],'Lat':[np.NaN],'Lon': [np.NaN],'Pres':[np.NaN],'Temp':[np.NaN],'Sal':[np.NaN],'Oxy':[np.NaN],'OxySat':[np.NaN],'Sigma0':[np.NaN]})
tic1=time.time()
for i in np.arange(len(ArgoWMO)):
#for i in [43]:
    tic2=time.time()
    
    dac=FileNames[i].split('/')[0]
    WMO=ArgoWMO[i]
    
    print('\n%%% ',WMO,' %%%\n')
    print(i, ' Floats completed; ', len(ArgoWMO)-i,' Floats Left')
    f = RF.ArgoDataLoader(DAC=dac, WMO=WMO)
    
    # Make sure float measures oxygen
    float_vars=list(f.keys())
    float_vars=np.array(float_vars)
    oxy_check = np.where(float_vars=='DOXY')
    
    flux_profcount=0
    
    if oxy_check[0].size != 0:

        # Load float data and determine if use adjusted or not adjusted data
        [pres, pres_QC, pres_flag]=RF.DetermineAdjusted(raw_data=f.PRES.values,raw_data_QC=f.PRES_QC.values,a_data=f.PRES_ADJUSTED.values, a_data_QC=f.PRES_ADJUSTED_QC.values)
        [temp, temp_QC, temp_flag]=RF.DetermineAdjusted(raw_data=f.TEMP.values, raw_data_QC= f.TEMP_QC.values,a_data=f.TEMP_ADJUSTED.values, a_data_QC=f.TEMP_ADJUSTED_QC.values)
        [sal, sal_QC, sal_flag]=RF.DetermineAdjusted(raw_data=f.PSAL.values, raw_data_QC=f.PSAL_QC.values,a_data=f.PSAL_ADJUSTED.values,a_data_QC=f.PSAL_ADJUSTED_QC.values)
        [doxy, doxy_QC, doxy_flag]=RF.DetermineAdjusted(raw_data=f.DOXY.values, raw_data_QC=f.DOXY_QC.values,a_data=f.DOXY_ADJUSTED.values,a_data_QC=f.DOXY_ADJUSTED_QC.values)
        
        ## Quality control float data ##
        pres=RF.ArgoQC(Data=pres, Data_QC=pres_QC, goodQC_flags=good_QC)
        temp=RF.ArgoQC(Data=temp, Data_QC=temp_QC, goodQC_flags=good_QC)
        sal=RF.ArgoQC(Data=sal, Data_QC=sal_QC, goodQC_flags=good_QC)
        doxy=RF.ArgoQC(Data=doxy, Data_QC=doxy_QC, goodQC_flags=good_QC)
        
        lat=f.LATITUDE.values
        lon=f.LONGITUDE.values
        
        dates=f.JULD.values
        
        date_reform=[[]]*dates.shape[0]
        bad_index=[]
        
        for k in np.arange(len(date_reform)):
            if np.isnat(dates[k]) == False:
                date_reform[k]=datetime.fromisoformat(str(dates[k])[:-3])
            else:
                date_reform[k]=dates[k]
                bad_index=bad_index+[k]
        
        # For each profile, determine if data point is in BC, Gyre, or N/A
        for j in np.arange(len(lat)):
            
            # Make sure data point is not a bad index 
            bad_check = 0
            if len(bad_index)>0:
                for b_i in bad_index:
                    if b_i == j:
                        bad_check =1
             
            if bad_check == 0:          
                # Make sure profile is in the generally correct region and a valid position
                if (lat[j]<=lat_N and lat[j]>=lat_S and lon[j]>=lon_W and lon[j]<=lon_E and np.isnan(lat[j]) == False and np.isnan(lon[j]) == False):
                    # Save profile information 
                    a=1
                else:
                    # Replace values with nan
                    lat[j]=np.NaN
                    lon[j]=np.NaN
                    pres[j,:]=np.NaN
                    temp[j,:]=np.NaN
                    sal[j,:]=np.NaN
                    doxy[j,:]=np.NaN
        
        if np.sum(np.isnan(lat)) != len(lat):
            
            if np.sum(np.isnan(doxy)) != doxy.shape[0]*doxy.shape[1]:
                
                print('\n Making figures...')
                date_reform=np.array(date_reform)
                date_reform_sub=date_reform-refdate
                date_reform_num=np.zeros(len(date_reform))
                date_reform_num[:]=np.NaN
                
                for l in np.arange(len(date_reform)):
                    date_reform_num[l]=date_reform_sub[l].total_seconds()
                
                # Only make sections if there is oxygen data
                
                # Interpolate data
                T_P=np.zeros((temp.shape[0],len(pres_range)))
                T_P[:]=np.NaN
                
                S_P=np.zeros((temp.shape[0],len(pres_range)))
                S_P[:]=np.NaN
                
                O_P=np.zeros((temp.shape[0],len(pres_range)))
                O_P[:]=np.NaN
                
                T_P=RF.PresInterpolation(OriginalData=temp, OriginalPressure=pres, NewPressure=pres_range, Pres_StepSize=step_size, NewData=T_P)
                S_P=RF.PresInterpolation(OriginalData=sal, OriginalPressure=pres, NewPressure=pres_range, Pres_StepSize=step_size, NewData=S_P)
                O_P=RF.PresInterpolation(OriginalData=doxy, OriginalPressure=pres, NewPressure=pres_range, Pres_StepSize=step_size, NewData=O_P)

                # Interpolate by day
                new_dates=pd.date_range(date_reform[0],date_reform[-1], freq='1D')
                new_dates=new_dates.to_pydatetime()
                new_dates_sub=new_dates-refdate
                new_dates_num=np.zeros(len(new_dates))
                new_dates_num[:]=np.NaN

                for l in np.arange(len(new_dates)):
                    new_dates_num[l]=new_dates_sub[l].total_seconds()
                
                T_PD=np.zeros((len(pres_range),len(new_dates)))
                T_PD[:]=np.NaN
                
                S_PD=np.zeros((len(pres_range),len(new_dates)))
                S_PD[:]=np.NaN
                
                O_PD=np.zeros((len(pres_range),len(new_dates)))
                O_PD[:]=np.NaN
                
                T_PD=RF.TimeInterpolatation(PresInterpData=T_P, PresInterpTime=date_reform_num, NewTime=new_dates_num, NewData=T_PD)
                S_PD=RF.TimeInterpolatation(PresInterpData=S_P, PresInterpTime=date_reform_num, NewTime=new_dates_num, NewData=S_PD)
                O_PD=RF.TimeInterpolatation(PresInterpData=O_P, PresInterpTime=date_reform_num, NewTime=new_dates_num, NewData=O_PD)
                
                X, Y = np.meshgrid(new_dates, pres_range)
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,T_PD,vmin=minT, vmax=maxT, cmap=cmo.thermal)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Temperature Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Interp_Temperature.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,S_PD,vmin=minS, vmax=maxS, cmap=cmo.haline)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Salinity Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Interp_Salinity.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,O_PD,vmin=minO, vmax=maxO, cmap=cmo.matter)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Oxygen Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Interp_Oxygen.jpg')
                plt.clf(); plt.close()
                
                # Calculate oxygen saturation
                lat_array_pd=np.zeros(O_PD.shape)
                lat_array_pd[:]=np.NaN
                lon_array_pd=np.zeros(O_PD.shape)
                lon_array_pd[:]=np.NaN
                
                ## Interpolate location
                lat_interp_time=interpolate.interp1d(date_reform_num, lat)
                lat_interp=lat_interp_time(new_dates_num)
                lon_interp_time=interpolate.interp1d(date_reform_num, lon)
                lon_interp=lon_interp_time(new_dates_num)
                
                for l in np.arange(lat_array_pd.shape[0]):
                    # rows: Pressure, columns dates 
                    lat_array_pd[l,:]=lat_interp.T
                    lon_array_pd[l,:]=lon_interp.T
                    
                P_PD=np.zeros(T_PD.shape)
                for l in np.arange(len(new_dates)):
                    P_PD[:,l]=pres_range
                    
                SA=gsw.SA_from_SP(S_PD,P_PD,lon_array_pd,lat_array_pd)
                CT=gsw.CT_from_t(SA, T_PD, P_PD)
                oxy_eq=gsw.O2sol(SA, CT, P_PD, lon_array_pd, lat_array_pd)
                OS_PD=O_PD/oxy_eq*100
                
                D_PD=np.zeros(T_PD.shape)
                D_PD[:]=np.NaN
                
                for r in np.arange(D_PD.shape[0]):
                    for c in np.arange(D_PD.shape[1]):
                        if np.isnan(SA[r,c]) == False and np.isnan(CT[r,c]) == False:
                            D_PD[r,c]=gsw.sigma0(SA[r,c], CT[r,c])
                        
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,OS_PD ,vmin=minOsat, vmax=maxOsat, cmap=cmo.haline)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Oxygen Saturation Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Interp_OxygenSaturation.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,D_PD,vmin=mindense, vmax=maxdense, cmap=cmo.dense)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Sigma 0 Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Interp_Density.jpg')
                plt.clf(); plt.close()
                
                lat_array=np.zeros(pres.shape)
                lat_array[:]=np.NaN
                lon_array=np.zeros(pres.shape)
                lon_array[:]=np.NaN
                
                for l in np.arange(lat_array.shape[0]):
                    lat_array[l,:]=lat[l]
                    lon_array[l,:]=lon[l]
                
                SA=gsw.SA_from_SP(sal,pres,lon_array,lat_array)
                CT=gsw.CT_from_t(SA, temp, pres)
                oxy_eq=gsw.O2sol(SA, CT, pres, lon_array, lat_array)
                
                oxy_sat=doxy/oxy_eq*100
                
                # Calculate density 
                density=np.zeros(temp.shape)
                for r in np.arange(sal.shape[0]):
                    for c in np.arange(sal.shape[1]):
                        if np.isnan(SA[r,c]) == False and np.isnan(CT[r,c]) == False:
                            density[r,c]=gsw.sigma0(SA[r,c], CT[r,c])
            
                # plot cross sections
                #plt.plot.scatter(x=date_reform, y=pres, c=sal)
                profs=list(f.CYCLE_NUMBER.values)
                levels=np.arange(pres.shape[1])
                
                date_list=np.array(list(date_reform),dtype=np.datetime64)
                date_list=np.stack((date_list,np.array(list(date_reform),dtype=np.datetime64)),axis=1)
                
                a=np.array(list(date_reform),dtype=np.datetime64)
                a=np.stack((a,np.array(list(date_reform),dtype=np.datetime64)),axis=1)
                for l in np.arange(1,len(levels)):
                    date_list=np.concatenate((date_list,a),axis=1)
                
                date_list=date_list[:,:len(levels)]
                ds = xr.Dataset(
                                data_vars=dict(
                                    temperature=(["x", "y"], temp),
                                    pressure=(["x", "y"], pres),
                                    salinity=(["x", "y"], sal),
                                    oxygen=(["x", "y"], doxy),
                                    oxygen_saturation=(["x", "y"], oxy_sat),
                                    dates=(["x", "y"], date_list),
                                    density=(["x", "y"], density),
                                ),
                                coords=dict(
                                    profs=(["x"], profs),
                                    levels=(["y"], levels),
                                    lon=lon,
                                    lat=lat,
                                    time=date_reform,
                                ),
                                attrs=dict(description="Weather related data."),
                                )
                
            #     # Salinity
                plt.figure(figsize=(figx, figy))
                ds.plot.scatter(x='dates',y='pressure',hue='salinity',vmin=minS, vmax=maxS, cmap=cmo.haline,marker='s')
                plt.gca().invert_yaxis()
                plt.xticks(rotation=45)
                plt.title('Salinity Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Salinity.jpg')
                plt.clf(); plt.close()
                
                # Temperature
                plt.figure(figsize=(figx, figy))
                ds.plot.scatter(x='dates',y='pressure',hue='temperature', vmin=minT, vmax=maxT, cmap=cmo.thermal,marker='s')
                plt.gca().invert_yaxis()
                plt.xticks(rotation=45)
                plt.title('Temperature Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Temperature.jpg')
                plt.clf(); plt.close()
            
            # # if np.sum(np.isnan(doxy)) != doxy.shape[0]*doxy.shape[1]:
            #     # Oxygen
                plt.figure(figsize=(figx, figy))
                ds.plot.scatter(x='dates',y='pressure',hue='oxygen', vmin=minO, vmax=maxO, cmap=cmo.matter,marker='s')
                plt.gca().invert_yaxis()
                plt.xticks(rotation=45)
                plt.title('Oxygen Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Oxygen.jpg')
                plt.clf(); plt.close()
                
            #     # Oxygen saturation
                plt.figure(figsize=(figx, figy))
                ds.plot.scatter(x='dates',y='pressure',hue='oxygen_saturation', vmin=minOsat, vmax=maxOsat, cmap=cmo.balance,marker='s')
                plt.gca().invert_yaxis()
                plt.xticks(rotation=45)
                plt.title('Oxygen Saturation Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_OxygenSat.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                ds.plot.scatter(x='dates',y='pressure',hue='density', vmin=mindense, vmax=maxdense, cmap=cmo.dense,marker='s')
                plt.gca().invert_yaxis()
                plt.xticks(rotation=45)
                plt.title('Density Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_Density.jpg')
                plt.clf(); plt.close()
                
            #     # Make T-S-O plot!
                plt.figure(figsize=(figx, figy))
                ds.plot.scatter(x='temperature',y='salinity',hue='oxygen_saturation', vmin=minOsat, vmax=maxOsat, cmap=cmo.haline,marker='s')
                plt.savefig(SectionFigDir+str(WMO)+'_TSOxySat.jpg')
                plt.clf(); plt.close()
                
                # Bin along density axis
                binned_T=np.zeros((len(dense_bins), len(profs)))
                binned_T[:]=np.NaN
                binned_S=np.zeros((len(dense_bins), len(profs)))
                binned_S[:]=np.NaN
                binned_O=np.zeros((len(dense_bins), len(profs)))
                binned_O[:]=np.NaN
                binned_Osat=np.zeros((len(dense_bins), len(profs)))
                binned_Osat[:]=np.NaN
                
                for l in np.arange(len(profs)):
                    dense_slice=density[l,:]
                    t_slice=temp[l,:]
                    s_slice=sal[l,:]
                    o_slice=doxy[l,:]
                    osat_slice=oxy_sat[l,:]
                    
                    T_bins=[[]]*len(dense_bins)
                    S_bins=[[]]*len(dense_bins)
                    O_bins=[[]]*len(dense_bins)
                    Osat_bins=[[]]*len(dense_bins)
                    
                    for k in np.arange(len(dense_slice)):
                        d=dense_slice[k]
                        
                        if d>=maxdense:
                            # Data goes into last bin
                            T_bins[-1]=T_bins[-1]+[t_slice[k]]
                            S_bins[-1]=S_bins[-1]+[s_slice[k]]
                            O_bins[-1]=O_bins[-1]+[o_slice[k]]
                            Osat_bins[-1]=Osat_bins[-1]+[osat_slice[k]]
                        elif d<=mindense:
                            # Data goes in first bin
                            T_bins[0]=T_bins[0]+[t_slice[k]]
                            S_bins[0]=S_bins[0]+[s_slice[k]]
                            O_bins[0]=O_bins[0]+[o_slice[k]]
                            Osat_bins[0]=Osat_bins[0]+[osat_slice[k]]
                        else:
                            new_range=d-mindense
                            d_i=int(new_range//dense_ss)
                            T_bins[d_i]=T_bins[d_i]+[t_slice[k]]
                            S_bins[d_i]=S_bins[d_i]+[s_slice[k]]
                            O_bins[d_i]=O_bins[d_i]+[o_slice[k]]
                            Osat_bins[d_i]=Osat_bins[d_i]+[osat_slice[k]]
                    
                    for m in np.arange(len(dense_bins)):
                        # Average each bin
                        if len(T_bins[m]) != 0:
                            binned_T[m,l]=np.nanmean(np.array(T_bins[m])) 
                        if len(S_bins[m]) != 0:
                            binned_S[m,l]=np.nanmean(np.array(S_bins[m]))
                        if len(O_bins[m]) != 0:
                            binned_O[m,l]=np.nanmean(np.array(O_bins[m]))
                        if len(Osat_bins[m]) != 0:
                            binned_Osat[m,l]=np.nanmean(np.array(Osat_bins[m]))
                
                
                x, y = np.meshgrid(date_reform, dense_bins)       
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(x,y,binned_T,vmin=minT, vmax=maxT,cmap=cmo.thermal)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.ylabel('Sigma0')
                plt.title('Temperature Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_DensityAxis_Temperature.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(x,y,binned_S,vmin=minS, vmax=maxS,cmap=cmo.haline)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.ylabel('Sigma0')
                plt.title('Salinity Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_DensityAxis_Salinity.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(x,y,binned_O,vmin=minO, vmax=maxO,cmap=cmo.matter)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.ylabel('Sigma0')
                plt.title('Oxygen Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_DensityAxis_Oxygen.jpg')
                plt.clf(); plt.close()
                        
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(x,y,binned_Osat,vmin=minOsat, vmax=maxOsat,cmap=cmo.balance)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.ylabel('Sigma0')
                plt.title('Oxygen Saturation Float: '+str(WMO))
                plt.savefig(SectionFigDir+str(WMO)+'_DensityAxis_OxygenSat.jpg')
                plt.clf(); plt.close()
                
                wmo_list=np.zeros(len(date_list.flatten()))
                wmo_list[:]=WMO
                # Save all T-S-O-Osat data
                df_temp=pd.DataFrame({'WMO':wmo_list,'Date':date_list.flatten(),'Lat':lat_array.flatten(), 'Lon':lon_array.flatten(),'Pres':pres.flatten(),'Temp':temp.flatten(),'Sal':sal.flatten(),'Oxy':doxy.flatten(),'OxySat':oxy_sat.flatten(),'Sigma0':density.flatten()})
                df_temp=df_temp.dropna()
                TSOsat=TSOsat.append(df_temp)
                TSOsat=TSOsat.dropna()
    
    toc2=time.time()
    print('Time elapsed (sec): ', toc2-tic2)

TSOsat=TSOsat.dropna()
TSOsat.to_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/Spice.csv')

plt.figure(figsize=(figx, figy))
plt.scatter(TSOsat.loc[:,'Temp'],TSOsat.loc[:,'Sal'],c=TSOsat.loc[:,'OxySat'],vmin=minOsat, vmax=maxOsat,cmap=cmo.balance)
plt.colorbar()
plt.savefig(SectionFigDir+'All_TSO.jpg')
plt.clf(); plt.close()

toc1=time.time()
print('\nTOTAL Time Elapsed (min): ', (toc1-tic1)/60)
                
            




             