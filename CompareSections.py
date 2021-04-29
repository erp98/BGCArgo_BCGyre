#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 07:13:37 2021

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
import cmocean.cm as cmo
import glob

OFiles = glob.glob('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/OverlapFiles_BC/*')
WMODacDict = RF.WMODacPair()

iso = [27.70, 27.75, 27.80,27.85]
# Interpolate to same space and time steps

# Calculate MLD at each point

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
maxS=35.5

minO=250
maxO=400

minOsat=80
maxOsat=120

mindense=27.5
maxdense=28
dense_ss=0.05
dense_bins=np.arange(mindense, maxdense+dense_ss,dense_ss)

step_size=10
pres_range=np.arange(0,2001,step_size)

refdate=datetime(1990,1,1)

for File in OFiles:
    
    o_data = pd.read_csv(File,engine='python')
    
    bc_floats = o_data.loc[:,'BC_Float'].to_numpy()
    gyre_floats = o_data.loc[:,'Gyre_Float'].to_numpy()
    
    for o in np.arange(len(bc_floats)):
        # gyre, bc
        float_pair = [int(gyre_floats[o]), int(bc_floats[o])]
        dac_pair=[WMODacDict[int(gyre_floats[o])],WMODacDict[int(bc_floats[o])]]

    
        bc_float = float_pair[1]
        gyre_float = float_pair[0]
        
        FigDir = '/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/CompareFloats/BC_'+str(bc_float)+'_G_'+str(gyre_float)+'_'
        
        print('\nFloat Pair: ', gyre_float, bc_float)
        for i in np.arange(len(float_pair)):
            dac=dac_pair[i]
            WMO=float_pair[i]
             
            print(WMO)
            
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
                        
                        a = 1
                        
                        # Crop data that is missing
                        good_index= np.where(np.isnan(lat)==0)
                        lat=lat[good_index]
                        lon=lon[good_index]
                        pres=pres[good_index]
                        temp=temp[good_index]
                        sal=sal[good_index]
                        doxy=doxy[good_index]
                        
                        profs=list(f.CYCLE_NUMBER.values)
                        
                        t_p=[]
                        t_d = []
                        
                        for l in np.arange(good_index[0].shape[0]):
                            ll = good_index[0][l]
                            t_d=t_d+[date_reform[ll]]
                            t_p=t_p +[profs[ll]]
                        
                        date_reform=t_d
                        profs=t_p
                        
                        date_reform=np.array(date_reform)
                        date_reform_sub=date_reform-refdate
                        date_reform_num=np.zeros(len(date_reform))
                        date_reform_num[:]=np.NaN
                        
                        for l in np.arange(len(date_reform)):
                            date_reform_num[l]=date_reform_sub[l].total_seconds()
                        
                        # # Only make sections if there is oxygen data
                        
                        # # Interpolate data
                        # T_P=np.zeros((temp.shape[0],len(pres_range)))
                        # T_P[:]=np.NaN
                        
                        # S_P=np.zeros((temp.shape[0],len(pres_range)))
                        # S_P[:]=np.NaN
                        
                        # O_P=np.zeros((temp.shape[0],len(pres_range)))
                        # O_P[:]=np.NaN
                        
                        # T_P=RF.PresInterpolation(OriginalData=temp, OriginalPressure=pres, NewPressure=pres_range, Pres_StepSize=step_size, NewData=T_P)
                        # S_P=RF.PresInterpolation(OriginalData=sal, OriginalPressure=pres, NewPressure=pres_range, Pres_StepSize=step_size, NewData=S_P)
                        # O_P=RF.PresInterpolation(OriginalData=doxy, OriginalPressure=pres, NewPressure=pres_range, Pres_StepSize=step_size, NewData=O_P)
                        
                        # start_date = datetime(date_reform[0].year, date_reform[0].month, date_reform[0].day)
                        # end_date=datetime(date_reform[0].year, date_reform[-1].month, date_reform[-1].day)
                        
                        # # Interpolate by day
                        # new_dates=pd.date_range(date_reform[0],date_reform[-1], freq='1D')
                        # new_dates=new_dates.to_pydatetime()
                        # new_dates_sub=new_dates-refdate
                        # new_dates_num=np.zeros(len(new_dates))
                        # new_dates_num[:]=np.NaN
        
                        # for l in np.arange(len(new_dates)):
                        #     new_dates_num[l]=new_dates_sub[l].total_seconds()
                        
                        # T_PD=np.zeros((len(pres_range),len(new_dates)))
                        # T_PD[:]=np.NaN
                        
                        # S_PD=np.zeros((len(pres_range),len(new_dates)))
                        # S_PD[:]=np.NaN
                        
                        # O_PD=np.zeros((len(pres_range),len(new_dates)))
                        # O_PD[:]=np.NaN
                        
                        # T_PD=RF.TimeInterpolatation(PresInterpData=T_P, PresInterpTime=date_reform_num, NewTime=new_dates_num, NewData=T_PD)
                        # S_PD=RF.TimeInterpolatation(PresInterpData=S_P, PresInterpTime=date_reform_num, NewTime=new_dates_num, NewData=S_PD)
                        # O_PD=RF.TimeInterpolatation(PresInterpData=O_P, PresInterpTime=date_reform_num, NewTime=new_dates_num, NewData=O_PD)
                        
                        # X, Y = np.meshgrid(new_dates, pres_range)
                        
                        # plt.figure(figsize=(figx, figy))
                        # plt.pcolormesh(X,Y,T_PD,vmin=minT, vmax=maxT, cmap=cmo.thermal)
                        # plt.gca().invert_yaxis()
                        # plt.colorbar()
                        # plt.xticks(rotation=45)
                        # plt.title('Temperature Float: '+str(WMO))
                        # # plt.savefig(SectionFigDir+str(WMO)+'_Interp_Temperature.jpg')
                        # # plt.clf(); plt.close()
                        
                        # plt.figure(figsize=(figx, figy))
                        # plt.pcolormesh(X,Y,S_PD,vmin=minS, vmax=maxS, cmap=cmo.haline)
                        # plt.gca().invert_yaxis()
                        # plt.colorbar()
                        # plt.xticks(rotation=45)
                        # plt.title('Salinity Float: '+str(WMO))
                        # # plt.savefig(SectionFigDir+str(WMO)+'_Interp_Salinity.jpg')
                        # # plt.clf(); plt.close()
                        
                        # plt.figure(figsize=(figx, figy))
                        # plt.pcolormesh(X,Y,O_PD,vmin=minO, vmax=maxO, cmap=cmo.matter)
                        # plt.gca().invert_yaxis()
                        # plt.colorbar()
                        # plt.xticks(rotation=45)
                        # plt.title('Oxygen Float: '+str(WMO))
                        # # plt.savefig(SectionFigDir+str(WMO)+'_Interp_Oxygen.jpg')
                        # # plt.clf(); plt.close()
                        
                        # # Calculate oxygen saturation
                        # lat_array_pd=np.zeros(O_PD.shape)
                        # lat_array_pd[:]=np.NaN
                        # lon_array_pd=np.zeros(O_PD.shape)
                        # lon_array_pd[:]=np.NaN
                        
                        # ## Interpolate location
                        # lat_interp_time=interpolate.interp1d(date_reform_num, lat)
                        # lat_interp=lat_interp_time(new_dates_num)
                        # lon_interp_time=interpolate.interp1d(date_reform_num, lon)
                        # lon_interp=lon_interp_time(new_dates_num)
                        
                        # for l in np.arange(lat_array_pd.shape[0]):
                        #     # rows: Pressure, columns dates 
                        #     lat_array_pd[l,:]=lat_interp.T
                        #     lon_array_pd[l,:]=lon_interp.T
                            
                        # P_PD=np.zeros(T_PD.shape)
                        # for l in np.arange(len(new_dates)):
                        #     P_PD[:,l]=pres_range
                            
                        # SA=gsw.SA_from_SP(S_PD,P_PD,lon_array_pd,lat_array_pd)
                        # CT=gsw.CT_from_t(SA, T_PD, P_PD)
                        # oxy_eq=gsw.O2sol(SA, CT, P_PD, lon_array_pd, lat_array_pd)
                        # OS_PD=O_PD/oxy_eq*100
                        
                        # D_PD=np.zeros(T_PD.shape)
                        # D_PD[:]=np.NaN
                        
                        # for r in np.arange(D_PD.shape[0]):
                        #     for c in np.arange(D_PD.shape[1]):
                        #         if np.isnan(SA[r,c]) == False and np.isnan(CT[r,c]) == False:
                        #             D_PD[r,c]=gsw.sigma0(SA[r,c], CT[r,c])
                                
                        # plt.figure(figsize=(figx, figy))
                        # plt.pcolormesh(X,Y,OS_PD ,vmin=minOsat, vmax=maxOsat, cmap=cmo.haline)
                        # plt.gca().invert_yaxis()
                        # plt.colorbar()
                        # plt.xticks(rotation=45)
                        # plt.title('Oxygen Saturation Float: '+str(WMO))
                        # # plt.savefig(SectionFigDir+str(WMO)+'_Interp_OxygenSaturation.jpg')
                        # # plt.clf(); plt.close()
                        
                        # plt.figure(figsize=(figx, figy))
                        # plt.pcolormesh(X,Y,D_PD,vmin=mindense, vmax=maxdense, cmap=cmo.dense)
                        # plt.gca().invert_yaxis()
                        # plt.colorbar()
                        # plt.xticks(rotation=45)
                        # plt.title('Sigma 0 Float: '+str(WMO))
                        # # plt.savefig(SectionFigDir+str(WMO)+'_Interp_Density.jpg')
                        # # plt.clf(); plt.close()
                        
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
                        
                        # Calculate MLD
                        mld=np.zeros(sal.shape[0])
                        mld[:]=np.NaN
                        
                        for r in np.arange(sal.shape[0]):
                            
                            mld_pres = RF.MLD(Pres=pres[r,:], Temp=temp[r,:], Sal=sal[r,:], Lat=lat[r], Lon=lon[r])
                            mld[r]=mld_pres
                            
                            for c in np.arange(sal.shape[1]):
                                if np.isnan(SA[r,c]) == False and np.isnan(CT[r,c]) == False:
                                    density[r,c]=gsw.sigma0(SA[r,c], CT[r,c])
                    
                        
                        
                        # Get isopycnal valus
                        a=1
                        
                        i1 = np.zeros(density.shape[0])
                        i1[:]=np.NaN
                        gi=np.where(np.round(density,2) == iso[0])
                        i1[gi[0]]=pres[gi[0],gi[1]]
                        
                        i2 = np.zeros(density.shape[0])
                        i2[:]=np.NaN
                        gi=np.where(np.round(density,2) == iso[1])
                        i2[gi[0]]=pres[gi[0],gi[1]]
                        
                        i3 = np.zeros(density.shape[0])
                        i3[:]=np.NaN
                        gi=np.where(np.round(density,2) == iso[2])
                        i3[gi[0]]=pres[gi[0],gi[1]]
                        
                        i4 = np.zeros(density.shape[0])
                        i4[:]=np.NaN
                        gi=np.where(np.round(density,2) == iso[3])
                        i4[gi[0]]=pres[gi[0],gi[1]]
                        
                        levels=np.arange(pres.shape[1])
                        
                        date_list=np.array(list(date_reform),dtype=np.datetime64)
                        date_list=np.stack((date_list,np.array(list(date_reform),dtype=np.datetime64)),axis=1)
                        
                        a=np.array(list(date_reform),dtype=np.datetime64)
                        a=np.stack((a,np.array(list(date_reform),dtype=np.datetime64)),axis=1)
                        for l in np.arange(1,len(levels)):
                            date_list=np.concatenate((date_list,a),axis=1)
                        
                        date_list=date_list[:,:len(levels)]
                        
                        if i == 0:
                            ds_gyre = xr.Dataset(
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
                                                lon=(["x"],lon),
                                                lat=(["x"],lat),
                                                MLD=(["x"],mld),
                                                I1=(["x"],i1),
                                                I2=(["x"],i2),
                                                I3=(["x"],i3),
                                                I4=(["x"],i4),
                                                time=(["x"],date_reform)
                                            ),
                                            attrs=dict(description="Weather related data."),
                                            )
                        elif i == 1:
                            ds_bc = xr.Dataset(
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
                                                lon=(["x"],lon),
                                                lat=(["x"],lat),
                                                MLD=(["x"],mld),
                                                I1=(["x"],i1),
                                                I2=(["x"],i2),
                                                I3=(["x"],i3),
                                                I4=(["x"],i4),
                                                time=(["x"],date_reform)
                                            ),
                                            attrs=dict(description="Weather related data."),
                                            )
                        
                        # ds_interp = xr.Dataset(
                        #                 data_vars=dict(
                        #                     temperature=(["x", "y"], T_PD),
                        #                     pressure=(["x", "y"], pres),
                        #                     salinity=(["x", "y"], S_PD),
                        #                     oxygen=(["x", "y"], O_PD),
                        #                     oxygen_saturation=(["x", "y"], OS_PD),
                        #                     dates=(["x", "y"], date_list),
                        #                     density=(["x", "y"], D_PD),
                        #                 ),
                        #                 coords=dict(
                        #                     profs=(["x"], profs),
                        #                     levels=(["y"], levels),
                        #                     lon=lon,
                        #                     lat=lat,
                        #                     time=date_reform,
                        #                 ),
                        #                 attrs=dict(description="Weather related data."),
                        #                 )
        
        print('\n Making figures')
        a=1
        # Get start and end 
        start_window=np.nanmax([np.nanmin(ds_gyre.time.values), np.nanmin(ds_bc.time.values)])
        end_window = np.nanmin([np.nanmax(ds_gyre.time.values), np.nanmax(ds_bc.time.values)])
        
        
        # Crop data to same time window
        ds_gyre = ds_gyre.where(ds_gyre.time >= start_window, drop = True)
        ds_gyre=ds_gyre.where(ds_gyre.time <= end_window, drop = True)
        
        ds_bc = ds_bc.where(ds_bc.time >= start_window, drop = True)
        ds_bc=ds_bc.where(ds_bc.time <= end_window, drop = True)
        
        #maxP = np.nanmin([np.nanmax(ds_gyre.pressure.values),np.nanmax(ds_bc.pressure.values)])
        
        if ds_gyre.profs.values.shape[0] > 0 and ds_bc.profs.values.shape[0] > 0:
            
            # Salinity
            figs, axs = plt.subplots(2,1, figsize=(figx, figy))
            ds_gyre.plot.scatter(x='dates',y='pressure',hue='salinity',vmin=minS, vmax=maxS, cmap=cmo.haline,marker='s', ax=axs[0])   
            ds_bc.plot.scatter(x='dates',y='pressure',hue='salinity',vmin=minS, vmax=maxS, cmap=cmo.haline,marker='s', ax=axs[1])
            
            axs[0].plot(ds_gyre.time.values, ds_gyre.MLD.values, 'k-', label='MLD')
            axs[0].plot(ds_gyre.time.values, ds_gyre.I1.values, 'w-',label=str(iso[0]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I2.values, 'w-',label=str(iso[1]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I3.values, 'w-',label=str(iso[2]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I4.values, 'w-',label=str(iso[3]))
            axs[0].legend()
            axs[1].plot(ds_bc.time.values, ds_bc.MLD.values,'k-', label ='MLD')
            axs[1].plot(ds_bc.time.values, ds_bc.I1.values, 'w-',label=str(iso[0]))
            axs[1].plot(ds_bc.time.values, ds_bc.I2.values, 'w-',label=str(iso[1]))
            axs[1].plot(ds_bc.time.values, ds_bc.I3.values, 'w-',label=str(iso[2]))
            axs[1].plot(ds_bc.time.values, ds_bc.I4.values, 'w-',label=str(iso[3]))
            axs[1].legend()
            
            for i in np.arange(len(axs)):
                #axs[i].set_ylim([0,maxP])
                axs[i].tick_params(axis='x', rotation=45) 
                axs[i].invert_yaxis()
            
            axs[0].set_title('Gyre')
            axs[1].set_title('Boundary Current')
            figs.subplots_adjust(hspace=1)
            plt.savefig(FigDir+'Salinity.jpg')
            plt.clf(); plt.close()
            
            # Temperature
            figs, axs = plt.subplots(2,1, figsize=(figx, figy))
            ds_gyre.plot.scatter(x='dates',y='pressure',hue='temperature', vmin=minT, vmax=maxT, cmap=cmo.thermal,marker='s', ax=axs[0])   
            ds_bc.plot.scatter(x='dates',y='pressure',hue='temperature', vmin=minT, vmax=maxT, cmap=cmo.thermal,marker='s', ax=axs[1])
            
            axs[0].plot(ds_gyre.time.values, ds_gyre.MLD.values, 'k-', label='MLD')
            axs[0].plot(ds_gyre.time.values, ds_gyre.I1.values, 'w-',label=str(iso[0]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I2.values, 'w-',label=str(iso[1]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I3.values, 'w-',label=str(iso[2]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I4.values, 'w-',label=str(iso[3]))
            axs[0].legend()
            axs[1].plot(ds_bc.time.values, ds_bc.MLD.values,'k-', label ='MLD')
            axs[1].plot(ds_bc.time.values, ds_bc.I1.values, 'w-',label=str(iso[0]))
            axs[1].plot(ds_bc.time.values, ds_bc.I2.values, 'w-',label=str(iso[1]))
            axs[1].plot(ds_bc.time.values, ds_bc.I3.values, 'w-',label=str(iso[2]))
            axs[1].plot(ds_bc.time.values, ds_bc.I4.values, 'w-',label=str(iso[3]))
            axs[1].legend()
            for i in np.arange(len(axs)):
                #axs[i].set_ylim([0,maxP])
                axs[i].tick_params(axis='x', rotation=45) 
                axs[i].invert_yaxis()
            
            axs[0].set_title('Gyre')
            axs[1].set_title('Boundary Current')
            figs.subplots_adjust(hspace=1)
            plt.savefig(FigDir+'Temperature.jpg')
            plt.clf(); plt.close()
            
            # Oxygen
            figs, axs = plt.subplots(2,1, figsize=(figx, figy))
            ds_gyre.plot.scatter(x='dates',y='pressure',hue='oxygen', vmin=minO, vmax=maxO, cmap=cmo.matter,marker='s', ax=axs[0])   
            ds_bc.plot.scatter(x='dates',y='pressure',hue='oxygen', vmin=minO, vmax=maxO, cmap=cmo.matter,marker='s', ax=axs[1])
            
            axs[0].plot(ds_gyre.time.values, ds_gyre.MLD.values, 'k-', label='MLD')
            axs[0].plot(ds_gyre.time.values, ds_gyre.I1.values, 'w-',label=str(iso[0]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I2.values, 'w-',label=str(iso[1]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I3.values, 'w-',label=str(iso[2]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I4.values, 'w-',label=str(iso[3]))
            axs[0].legend()
            axs[1].plot(ds_bc.time.values, ds_bc.MLD.values,'k-', label ='MLD')
            axs[1].plot(ds_bc.time.values, ds_bc.I1.values, 'w-',label=str(iso[0]))
            axs[1].plot(ds_bc.time.values, ds_bc.I2.values, 'w-',label=str(iso[1]))
            axs[1].plot(ds_bc.time.values, ds_bc.I3.values, 'w-',label=str(iso[2]))
            axs[1].plot(ds_bc.time.values, ds_bc.I4.values, 'w-',label=str(iso[3]))
            axs[1].legend()
            for i in np.arange(len(axs)):
                #axs[i].set_ylim([0,maxP])
                axs[i].tick_params(axis='x', rotation=45) 
                axs[i].invert_yaxis()
            
            axs[0].set_title('Gyre')
            axs[1].set_title('Boundary Current')
            figs.subplots_adjust(hspace=1)
            plt.savefig(FigDir+'Oxygen.jpg')
            plt.clf(); plt.close()
            
            # Oxygen saturation
            figs, axs = plt.subplots(2,1, figsize=(figx, figy))
            ds_gyre.plot.scatter(x='dates',y='pressure',hue='oxygen_saturation', vmin=minOsat, vmax=maxOsat, cmap=cmo.balance,marker='s', ax=axs[0])   
            ds_bc.plot.scatter(x='dates',y='pressure',hue='oxygen_saturation', vmin=minOsat, vmax=maxOsat, cmap=cmo.balance,marker='s', ax=axs[1])
            
            axs[0].plot(ds_gyre.time.values, ds_gyre.MLD.values, 'k-', label='MLD')
            axs[0].plot(ds_gyre.time.values, ds_gyre.I1.values, 'w-',label=str(iso[0]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I2.values, 'w-',label=str(iso[1]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I3.values, 'w-',label=str(iso[2]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I4.values, 'w-',label=str(iso[3]))
            axs[0].legend()
            axs[1].plot(ds_bc.time.values, ds_bc.MLD.values,'k-', label ='MLD')
            axs[1].plot(ds_bc.time.values, ds_bc.I1.values, 'w-',label=str(iso[0]))
            axs[1].plot(ds_bc.time.values, ds_bc.I2.values, 'w-',label=str(iso[1]))
            axs[1].plot(ds_bc.time.values, ds_bc.I3.values, 'w-',label=str(iso[2]))
            axs[1].plot(ds_bc.time.values, ds_bc.I4.values, 'w-',label=str(iso[3]))
            axs[1].legend()
            for i in np.arange(len(axs)):
                #axs[i].set_ylim([0,maxP])
                axs[i].tick_params(axis='x', rotation=45) 
                axs[i].invert_yaxis()
            
            axs[0].set_title('Gyre')
            axs[1].set_title('Boundary Current')
            figs.subplots_adjust(hspace=1)
            plt.savefig(FigDir+'OxygenSat.jpg')
            plt.clf(); plt.close()
            
            # Density 
            figs, axs = plt.subplots(2,1, figsize=(figx, figy))
            ds_gyre.plot.scatter(x='dates',y='pressure',hue='density', vmin=mindense, vmax=maxdense, cmap=cmo.dense,marker='s', ax=axs[0])   
            ds_bc.plot.scatter(x='dates',y='pressure',hue='density', vmin=mindense, vmax=maxdense, cmap=cmo.dense,marker='s', ax=axs[1])
            
            axs[0].plot(ds_gyre.time.values, ds_gyre.MLD.values, 'k-', label='MLD')
            axs[0].plot(ds_gyre.time.values, ds_gyre.I1.values, 'w-',label=str(iso[0]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I2.values, 'w-',label=str(iso[1]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I3.values, 'w-',label=str(iso[2]))
            axs[0].plot(ds_gyre.time.values, ds_gyre.I4.values, 'w-',label=str(iso[3]))
            axs[0].legend()
            axs[1].plot(ds_bc.time.values, ds_bc.MLD.values,'k-', label ='MLD')
            axs[1].plot(ds_bc.time.values, ds_bc.I1.values, 'w-',label=str(iso[0]))
            axs[1].plot(ds_bc.time.values, ds_bc.I2.values, 'w-',label=str(iso[1]))
            axs[1].plot(ds_bc.time.values, ds_bc.I3.values, 'w-',label=str(iso[2]))
            axs[1].plot(ds_bc.time.values, ds_bc.I4.values, 'w-',label=str(iso[3]))
            axs[1].legend()
            for i in np.arange(len(axs)):
                #axs[i].set_ylim([0,maxP])
                axs[i].tick_params(axis='x', rotation=45) 
                axs[i].invert_yaxis()
            
            axs[0].set_title('Gyre')
            axs[1].set_title('Boundary Current')
            figs.subplots_adjust(hspace=1)
            plt.savefig(FigDir+'Density.jpg')
            plt.clf(); plt.close()
            plt.show()
            
            # Compare time-series along isopycnals
            I1_bc = ds_bc.where(ds_bc.density >= iso[0], drop = True)
            I1_bc=I1_bc.where(I1_bc.density <= iso[1], drop = True)
            I1_g = ds_gyre.where(ds_gyre.density >= iso[0], drop = True)
            I1_g=I1_g.where(I1_g.density <= iso[1], drop = True)
            
            I2_bc = ds_bc.where(ds_bc.density >= iso[1], drop = True)
            I2_bc=I2_bc.where(I2_bc.density <= iso[2], drop = True)
            I2_g = ds_gyre.where(ds_gyre.density >= iso[1], drop = True)
            I2_g=I2_g.where(I2_g.density <= iso[2], drop = True)
            
            I3_bc = ds_bc.where(ds_bc.density >= iso[2], drop = True)
            I3_bc=I3_bc.where(I3_bc.density <= iso[3], drop = True)
            I3_g = ds_gyre.where(ds_gyre.density >=iso[2], drop = True)
            I3_g=I3_g.where(I3_g.density <= iso[3], drop = True)
            
            plt.figure()
            
            ## ISO # 1
            figs, axs = plt.subplots(3,3, figsize=(10, 8))
            ax = plt.subplot(3,3,1)
            ax.errorbar(I1_bc.time.values,np.nanmean(I1_bc.temperature.values, axis=1), yerr=np.nanstd(I1_bc.temperature.values, axis=1), label='BC')
            ax.errorbar(I1_g.time.values,np.nanmean(I1_g.temperature.values, axis=1), yerr=np.nanstd(I1_g.temperature.values, axis=1),label='G')
            ax.legend()
            ax.set_xticklabels([], rotation=45)
            ax.set_ylabel('Temperature')
            ax.set_title('ISO1')
            
            ax = plt.subplot(3,3,2)
            ax.errorbar(I1_bc.time.values,np.nanmean(I1_bc.salinity.values, axis=1), yerr=np.nanstd(I1_bc.salinity.values, axis=1), label='BC')
            ax.errorbar(I1_g.time.values,np.nanmean(I1_g.salinity.values, axis=1), yerr=np.nanstd(I1_g.salinity.values, axis=1),label='G')
            ax.legend()
            ax.set_xticklabels([], rotation=45)
            ax.set_ylabel('Salinity')
            ax.set_title('ISO1')
            
            ax = plt.subplot(3,3,3)
            ax.errorbar(I1_bc.time.values,np.nanmean(I1_bc.oxygen.values, axis=1), yerr=np.nanstd(I1_bc.oxygen.values, axis=1), label='BC')
            ax.errorbar(I1_g.time.values,np.nanmean(I1_g.oxygen.values, axis=1), yerr=np.nanstd(I1_g.oxygen.values, axis=1),label='G')
            ax.legend()
            ax.set_xticklabels([], rotation=45)
            ax.set_ylabel('Oxygen')
            ax.set_title('ISO1')
            
                
            ## ISO # 2
            ax = plt.subplot(3,3,4)
            ax.errorbar(I2_bc.time.values,np.nanmean(I2_bc.temperature.values, axis=1), yerr=np.nanstd(I2_bc.temperature.values, axis=1), label='BC')
            ax.errorbar(I2_g.time.values,np.nanmean(I2_g.temperature.values, axis=1), yerr=np.nanstd(I2_g.temperature.values, axis=1),label='G')
            ax.legend()
            ax.set_xticklabels([], rotation=45)
            ax.set_ylabel('Temperature')
            ax.set_title('ISO2')
            
            ax = plt.subplot(3,3,5)
            ax.errorbar(I2_bc.time.values,np.nanmean(I2_bc.salinity.values, axis=1), yerr=np.nanstd(I2_bc.salinity.values, axis=1), label='BC')
            ax.errorbar(I2_g.time.values,np.nanmean(I2_g.salinity.values, axis=1), yerr=np.nanstd(I2_g.salinity.values, axis=1),label='G')
            ax.legend()
            ax.set_xticklabels([], rotation=45)
            ax.set_ylabel('Salinity')
            ax.set_title('ISO2')
            
            ax = plt.subplot(3,3,6)
            ax.errorbar(I2_bc.time.values,np.nanmean(I2_bc.oxygen.values, axis=1), yerr=np.nanstd(I2_bc.oxygen.values, axis=1), label='BC')
            ax.errorbar(I2_g.time.values,np.nanmean(I2_g.oxygen.values, axis=1), yerr=np.nanstd(I2_g.oxygen.values, axis=1),label='G')
            ax.legend()
            ax.set_xticklabels([], rotation=45)
            ax.set_ylabel('Oxygen')
            ax.set_title('ISO2')
            
            
            ## ISO # 3
            ax = plt.subplot(3,3,7)
            ax.errorbar(I3_bc.time.values,np.nanmean(I3_bc.temperature.values, axis=1), yerr=np.nanstd(I3_bc.temperature.values, axis=1), label='BC')
            ax.errorbar(I3_g.time.values,np.nanmean(I3_g.temperature.values, axis=1), yerr=np.nanstd(I3_g.temperature.values, axis=1),label='G')
            ax.legend()
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_ylabel('Temperature')
            ax.set_title('ISO3')
            
            ax = plt.subplot(3,3,8)
            ax.errorbar(I3_bc.time.values,np.nanmean(I3_bc.salinity.values, axis=1), yerr=np.nanstd(I3_bc.salinity.values, axis=1), label='BC')
            ax.errorbar(I3_g.time.values,np.nanmean(I3_g.salinity.values, axis=1), yerr=np.nanstd(I3_g.salinity.values, axis=1),label='G')
            ax.legend()
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_ylabel('Salinity')
            ax.set_title('ISO3')
            
            ax = plt.subplot(3,3,9)
            ax.errorbar(I3_bc.time.values,np.nanmean(I3_bc.oxygen.values, axis=1), yerr=np.nanstd(I3_bc.oxygen.values, axis=1), label='BC')
            ax.errorbar(I3_g.time.values,np.nanmean(I3_g.oxygen.values, axis=1), yerr=np.nanstd(I3_g.oxygen.values, axis=1),label='G')
            ax.legend()
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
            ax.set_ylabel('Oxygen')
            ax.set_title('ISO3')
            
            title_str = 'BC Float: '+str(bc_float)+' Gyre Float: '+str(gyre_float)
            plt.suptitle(title_str)
            plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
            plt.savefig(FigDir+'IsopycnalAverages.jpg')
        else:
            print('Not enough data to make figs')
    
    
