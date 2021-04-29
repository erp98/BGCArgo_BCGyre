#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:09:53 2020

@author: Ellen
"""
import glob
import pandas as pd
import numpy as np
import csv
import time

# Load and save all GO-SHIP data in a specific region in a more
# accessible format

LatN=64
LatS=48
LonE=-45
LonW=-65

BottleDataFiles=glob.glob('/Users/Ellen/Desktop/GOSHIP_Data/cchdo_search_results/*')
BottleDataFiles=sorted(BottleDataFiles)
OutFileDir='/Users/Ellen/Desktop/GOSHIP_Data/NAtlantic_Sorted/Reformat/'

df_counter=0

tic=time.time()

# If there are bottle files and/or CTD files
# Open them, find dates, lat-lon, values

# CSV bottle files
saved_files=0

for i in np.arange(len(BottleDataFiles)):
    
    botfname=BottleDataFiles[i]
    rcon=0
    
    # Open files and read in JUST the data, exclude the notes
    r=0
    
    if botfname != '/Users/Ellen/Desktop/GOSHIP_Data/cchdo_search_results/81_06AQ20050122_hy1.csv':
        with open(botfname) as csvfile:
            spamreader = csv.reader(csvfile)
            data=list(spamreader)
            for row in data:
                a=1
                if len(row)>0:
                    if len(row[0]) > 0:
                        #print(row[0][0])
        
                        if (row[0][0] == '#' or row[0]=='BOTTLE' or r==0):
                            rcon=rcon+1
                
                r=r+1
    else:
        rcon=35

    BottleData = pd.read_csv(botfname, skiprows=rcon)
    
    # Remove first line that has units
    BottleData=BottleData.iloc[1:,:]
    
    # Make sure lat/lon values are numbers (not strings)
    BottleData.loc[:,'LATITUDE']=BottleData.loc[:,'LATITUDE'].astype('float64')
    BottleData.loc[:,'LONGITUDE']=BottleData.loc[:,'LONGITUDE'].astype('float64')
    
    # Check to make sure data is in the region of interest
    BottleData.loc[BottleData.loc[:,'LATITUDE']>LatN,:]=np.NaN
    BottleData.loc[BottleData.loc[:,'LATITUDE']<LatS,:]=np.NaN
    BottleData.loc[BottleData.loc[:,'LONGITUDE']>LonE,:]=np.NaN
    BottleData.loc[BottleData.loc[:,'LONGITUDE']<LonW,:]=np.NaN
    BottleData=BottleData.dropna()
    
    if BottleData.shape[0]>0:
        new_ind=np.arange(BottleData.shape[0])
        BottleData=BottleData.set_index(pd.Index(list(new_ind)))
        
        # Save data as new csv file
        fname=botfname.split('/')[-1]
        BottleData.to_csv(OutFileDir+fname)
        saved_files=saved_files+1
        
        # Save notes file as text file 
        tname=fname.split('.')[0]
        if botfname != '/Users/Ellen/Desktop/GOSHIP_Data/cchdo_search_results/81_06AQ20050122_hy1.csv':
            Notes=data[:rcon]
            with open(OutFileDir+'FileNotes/'+tname+'.txt','w') as f:
                    for ele in Notes:
                        ele_j=' '.join(ele)
                        f.write(ele_j+'\n')
        else:
            test=pd.read_csv(botfname, skipfooter=BottleData.shape[0]-rcon+1, engine = 'python')
            test=test.values

            with open(OutFileDir+'FileNotes/'+tname+'.txt','w') as f:
                for i in np.arange(test.shape[0]): 
                    line=test[i,:]
                    Line=[]
                    for sec in line:
                        if type(sec) == str:
                            Line=Line+[sec]
                            
                    ele_j=' '.join(Line)
                    f.write(ele_j+'\n')
            
            
            
toc=time.time()

print(toc-tic, 'sec elapsed')
print((toc-tic)/60, 'min elapsed')
print('\n', saved_files, ' files in region (N: ', LatN,', S: ', LatS,', E: ', LonE,', W: ', LonW,')')


