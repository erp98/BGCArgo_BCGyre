 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:02:40 2021

@author: Ellen
"""

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
import scipy.integrate as integrate

Argofiles='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/BC_Bath/AdjDataFlags.csv'

SectionFigDir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/Figures/Sections1m/'

ArgoDataFrames=pd.read_csv(Argofiles)
ArgoDataFrames=ArgoDataFrames.dropna()
new_ind=np.arange(ArgoDataFrames.shape[0])
ArgoDataFrames=ArgoDataFrames.set_index(pd.Index(list(new_ind)))
    
ArgoWMO=ArgoDataFrames.loc[:,'FloatWMO']
FileNames=ArgoDataFrames.loc[:,'FileName']

good_QC=[1,2,5,8]
lat_N=65.000
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

step_size=1
interp_pres=100
depth_range=np.arange(0,interp_pres+step_size,step_size)

refdate=datetime(1990,1,1)

O2Inv=pd.DataFrame({'WMO': [np.NaN],'Prof': [np.NaN], 'Date':[np.NaN],'Lat': [np.NaN],'Lon':[np.NaN],'O2Inv':[np.NaN],'O2EqInv': [np.NaN]})
tic1=time.time()
for i in np.arange(len(ArgoWMO)):
#for i in [16]:
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
                
                # Convert from pressure to depth
                z_values=np.zeros(pres.shape)
                z_values[:]=np.NaN
                
                for pp in np.arange(pres.shape[0]):
                    z_values[pp,:]=gsw.z_from_p(pres[pp,:], lat[pp])
                    
                z_values=z_values*-1
                # Only make sections if there is oxygen data
                
                # Interpolate data
                T_P=np.zeros((temp.shape[0],len(depth_range)))
                T_P[:]=np.NaN
                
                S_P=np.zeros((temp.shape[0],len(depth_range)))
                S_P[:]=np.NaN
                
                O_P=np.zeros((temp.shape[0],len(depth_range)))
                O_P[:]=np.NaN
                
                T_P=RF.PresInterpolation1m(OriginalData=temp, OriginalPressure=pres, NewPressure=depth_range, Pres_StepSize=step_size, NewData=T_P)
                S_P=RF.PresInterpolation1m(OriginalData=sal, OriginalPressure=pres, NewPressure=depth_range, Pres_StepSize=step_size, NewData=S_P)
                O_P=RF.PresInterpolation1m(OriginalData=doxy, OriginalPressure=pres, NewPressure=depth_range, Pres_StepSize=step_size, NewData=O_P)

                # Interpolate by day
                new_dates_sub=date_reform-refdate
                new_dates_num=np.zeros(len(new_dates_sub))
                new_dates_num[:]=np.NaN

                for l in np.arange(len(new_dates_sub)):
                    new_dates_num[l]=new_dates_sub[l].total_seconds()
                
                T_PD=T_P.T
                S_PD=S_P.T
                O_PD=O_P.T
                
                X, Y = np.meshgrid(date_reform, depth_range)
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,T_PD,vmin=minT, vmax=maxT, cmap=cmo.thermal)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Temperature Float: '+str(WMO))
                plt.savefig(SectionFigDir+'Pres_'+str(interp_pres)+'_'+str(WMO)+'_Interp_Temperature.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,S_PD,vmin=minS, vmax=maxS, cmap=cmo.haline)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Salinity Float: '+str(WMO))
                plt.savefig(SectionFigDir+'Pres_'+str(interp_pres)+'_'+str(WMO)+'_Interp_Salinity.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,O_PD,vmin=minO, vmax=maxO, cmap=cmo.matter)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Oxygen Float: '+str(WMO))
                plt.savefig(SectionFigDir+'Pres_'+str(interp_pres)+'_'+str(WMO)+'_Interp_Oxygen.jpg')
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
                for l in np.arange(len(new_dates_num)):
                    P_PD[:,l]=depth_range
                    
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
                plt.savefig(SectionFigDir+'Pres_'+str(interp_pres)+'_'+str(WMO)+'_Interp_OxygenSaturation.jpg')
                plt.clf(); plt.close()
                
                plt.figure(figsize=(figx, figy))
                plt.pcolormesh(X,Y,D_PD,vmin=mindense, vmax=maxdense, cmap=cmo.dense)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.xticks(rotation=45)
                plt.title('Sigma 0 Float: '+str(WMO))
                plt.savefig(SectionFigDir+'Pres_'+str(interp_pres)+'_'+str(WMO)+'_Interp_Density.jpg')
                plt.clf(); plt.close()
                
                # plot cross sections
                #plt.plot.scatter(x=date_reform, y=pres, c=sal)
                profs=list(f.CYCLE_NUMBER.values)
                
                wmo_list=np.zeros(len(profs))
                wmo_list[:]=WMO
                
                # Calculate O2 inventory
                oxy_inventory=np.zeros(O_PD.shape[1])
                oxy_inventory[:]=np.NaN
                
                oxyeq_inventory=np.zeros(O_PD.shape[1])
                oxyeq_inventory[:]=np.NaN
                
                print('\n Calculating Oxygen Inventory...')
                good_prof=0
                
                oxy_m3=O_PD*(D_PD+1000)
                oxyeq_m3=oxy_eq*(D_PD+1000)
                
                for d in np.arange(O_PD.shape[1]):
                    
                    if np.sum(np.isnan(O_PD[:,d])) == 0:
                        good_prof=good_prof+1
                        # Means there are O2 values for 0-2000 m 
                        # ox_in_t=np.zeros(O_PD.shape[0]-1)
                        # ox_in_t[:]=np.NaN
                        
                        # ox_eq_int=np.zeros(O_PD.shape[0]-1)
                        # ox_eq_int[:]=np.NaN
                        
                        # for l in np.arange(O_PD.shape[0]-1):
                        #     delta_h=1 
                        #     oxy_val=np.nanmean((O_PD[l+1,d],O_PD[l,d]))
                        #     dense_val=np.nanmean((D_PD[l+1,d],D_PD[l,d]))+1000
                            
                        #     ox_in_t[l]=delta_h*oxy_val*dense_val # m * µmol/kg * kg/m^3
                            
                        #     oxyeq_val=np.nanmean((oxy_eq[l+1,d],oxy_eq[l,d]))
                        #     ox_eq_int[l]=delta_h*oxyeq_val*dense_val
                        
                        # oxy_inventory[d]=np.nansum(ox_in_t)*10**-6
                        # oxyeq_inventory[d]=np.nansum(ox_eq_int)*10**-6
                        
                        oxy_inventory[d]=integrate.trapz(oxy_m3[:,d],dx=step_size)*10**-6
                        oxyeq_inventory[d]=integrate.trapz(oxyeq_m3[:,d],dx=step_size)*10**-6
                        
                print('\n',good_prof,' of ', int(len(profs)),' used to calculate inventory')
                            
                #Save all T-S-O-Osat data
                df_temp=pd.DataFrame({'WMO': wmo_list,'Prof': profs, 'Date':date_reform,'Lat': lat,'Lon':lon,'O2Inv':oxy_inventory, 'O2EqInv':oxyeq_inventory})
                df_temp=df_temp.dropna()
                O2Inv=O2Inv.append(df_temp)
    
    toc2=time.time()
    print('Time elapsed (sec): ', toc2-tic2)

O2Inv=O2Inv.dropna()
O2Inv.to_csv('/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/CSVFiles/'+'Pres_'+str(interp_pres)+'_O2WaterColumn_1m.csv')

# plt.figure(figsize=(figx, figy))
# plt.scatter(TSOsat.loc[:,'Temp'],TSOsat.loc[:,'Sal'],c=TSOsat.loc[:,'OxySat'],vmin=minOsat, vmax=maxOsat,cmap=cmo.balance)
# plt.colorbar()
# plt.savefig(SectionFigDir+'All_TSO.jpg')
# plt.clf(); plt.close()

toc1=time.time()
print('\nTOTAL Time Elapsed (min): ', (toc1-tic1)/60)
                
            




             