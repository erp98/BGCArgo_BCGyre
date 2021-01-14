#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:29:07 2020

@author: Ellen
"""
import numpy as np
import time

####
# Change latitude and longitude values in lines 21-24
# Can modify code for any part of the world.
# Change ocean letter (A,P,I) in line 118
####

##### Things To Update ######
# 1. Output an index file containing floats of interest
# 2. Out float IDs by specific BGC variables

Dir='/Users/Ellen/Documents/GitHub/BGCArgo_BCGyre/TextFiles/'

BGCfiles='/Users/Ellen/Desktop/ArgoGDAC/argo_synthetic-profile_index.txt';

tic = time.time()

lat_N=80.000
lat_S= 40.00
lon_E= 0.00
lon_W= -80.00

count = 0

goodArgo=0
good_fname=[] # Maybe should run to find number then initialize size ?
total_index=[]

# Extract file names and important info from txt file

# Important Info
file=[]
date=[]
lat=[]
lon=[]
ocean=[]
prof_type=[]
param=[]
param_datamode=[]
date_update=[]

#imp_val=[file,date,lat,lon,ocean,prof_type,param,param_datamode,date_update]
#imp_val=['file','date','lat','lon','ocean','prof_type','param','param_datamode','date_update']

print('Start of file reading...')
with open(BGCfiles) as fp:
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        #print("Line{}: {}".format(count,line.strip()))
        #print(line.strip())
        x=line.strip()
        #print(len(x))
        #print('new line')
        x_split=x.split(',')
#         0 filename
#         1 date=''.join(date)
#         2 lat=''.join(lat)
#         3 lon=''.join(lon)
#         4 ocean=''.join(ocean)
#         5 prof_type=''.join(prof_type)
#         6 inst=''.join(inst)
#         7 param=''.join(param)
#         8 param_datamode=''.join(param_datamode)
#         9 date_update=''.join(date_update)
        #print('This is:', x_split)

        if x_split[0][0] != '#' and x_split[0][0] != 'f':
            ## Check if in the right ocean (ocean == A) (Can be A, P, or I)
            # Loop through data to get to the ocean...

            # Check if ocean is atlantic ocean
            #print('Starting ocean check')
            ocean=x_split[4]
            #print(ocean)

            if ocean == 'A':
                # Float is in Atlantic Ocean
                # Narrow search to geographic region (lat, lon)

                # Check 1: is lat in desired range?

                lat=x_split[2]
                lon=x_split[3]

                #print('Start lat check')

                if ((float(lat) >= lat_S) and (float(lat)<=lat_N)):
                    # Float is in the correct latitude range
                    # If Check 1 == Yes --> Check 2: is lon in desired range?

                    if ((float(lon)<= lon_E) and (float(lon)>= lon_W)):
                        #print('Start lon check')
                        # Float is in the correct lat & lon region
                        # Store data
                        #print('\n ** good float ** \n')
                        file=x_split[0]
                        good_fname=good_fname+[file]
                        total_index=total_index+[x]

                        goodArgo = goodArgo +1

                #else:
                    #print("Float in correct lat but wrong lon")

            #else:
                #print('Float not in correct lat region')


       # else:
            # Not in Atlantic stop search
            #print('Not in Atlantic')

print(total_index)
print('File reading completed')
toc = time.time()
print(toc-tic, ' sec elapsed')

print('\n There are', goodArgo,'good profiles')

#print(good_fname)



# From goodArgo float file names extract ArgoFloat numbers
# float_type(?)/float#/profiles/float#_profile#.nc

total_float_num=[]
total_file_name=[]

good_aoml = []
good_bodc = []
good_coriolis = []
good_csio = []
good_csiro = []
good_incois = []
good_jma = []
good_kma = []
good_kordi = []
good_meds = []
good_nmdis = []

dac_count=[0,0,0,0,0,0,0,0,0,0,0]

print('\nStart float number extraction...')
tic=time.time()

for x in np.arange(len(good_fname)):

    fname=good_fname[x]
    fname_split=fname.split('/')
    # 0 GDAC
    # 1 float number
    # 2 profiles
    # 3 SRfloatnumber_prof#.nc


    float_num=fname_split[1]
    found = 0
    j = 0
    while j <= len(total_float_num)-1:

        if total_float_num[j] == float_num:
            found = 1
        j=j+1

    if found == 0:
        total_float_num=total_float_num+[float_num]
        total_file_name=total_file_name+[fname_split[0]+'/'+float_num]

        if fname_split[0]=='aoml':
            good_aoml=good_aoml+[fname_split[0]+'/'+float_num]
            dac_count[0]=1
        elif fname_split[0]=='bodc':
            good_bodc=good_bodc+[fname_split[0]+'/'+float_num]
            dac_count[1]=1
        elif fname_split[0]=='coriolis':
            good_coriolis=good_coriolis+[fname_split[0]+'/'+float_num]
            dac_count[2]=1
        elif fname_split[0]=='csio':
            good_csio=good_csio+[fname_split[0]+'/'+float_num]
            dac_count[3]=1
        elif fname_split[0]=='csiro':
            good_csiro=good_csiro+[fname_split[0]+'/'+float_num]
            dac_count[4]=1
        elif fname_split[0]=='incois':
            good_incois=good_incois+[fname_split[0]+'/'+float_num]
            dac_count[5]=1
        elif fname_split[0]=='jma':
            good_jma=good_jma+[fname_split[0]+'/'+float_num]
            dac_count[6]=1
        elif fname_split[0]=='kma':
            good_kma=good_kma+[fname_split[0]+'/'+float_num]
            dac_count[7]=1
        elif fname_split[0]=='kordi':
            good_kordi=good_kordi+[fname_split[0]+'/'+float_num]
            dac_count[8]=1
        elif fname_split[0]=='meds':
            good_meds=good_meds+[fname_split[0]+'/'+float_num]
            dac_count[9]=1
        elif fname_split[0]=='nmdis':
            good_nmdis=good_nmdis+[fname_split[0]+'/'+float_num]
            dac_count[10]=1


print('End float number extraction')
toc=time.time()
print(toc-tic, 'sec elapsed')

print('\n There are', len(total_float_num),'good floats')
#print(total_file_name)

outfname1=Dir+'DacWMO_NAtlantic.txt' #'goodArgo_small.txt'
outfname2=Dir+'WMO_NAtlantic.txt'
outfname3=Dir+'Index_NAtlantic.txt'

with open(outfname1,'w') as f:
    for ele in total_file_name:
        f.write(ele+'\n')
    #f.write('\n'.join(total_file_name))

with open(outfname2,'w') as f:
    for ele in total_float_num:
        f.write(ele+'\n')
    #f.write('\n'.join(total_float_num))

with open(outfname3,'w') as f:
    for ele in total_index:
        f.write(ele+'\n')

for i in np.arange(len(dac_count)):

    if dac_count[i]==1:
        if i==0:
            dacdata=good_aoml
            dac='aoml'
        if i==1:
            dacdata=good_bodc
            dac='bodc'
        if i==2:
            dacdata=good_coriolis
            dac='coriolis'
        if i==3:
            dacdata=good_csio
            dc='csio'
        if i==4:
            dacdata==good_csiro
            dac='csiro'
        if i ==5:
            dacdata==good_incois
            dac='incois'
        if i==6:
            dacdata=good_jma
            dac='jma'
        if i==7:
            dacdata=good_kma
            dac='kma'
        if i ==8:
            dacdata==good_kordi
            dac='kordi'
        if i==9:
            dacdata=good_meds
            dac='meds'
        if i==10:
            dacdata==good_nmdis
            dac='nmdis'

        with open(Dir+'good'+dac+'.txt','w') as f:
            for ele in dacdata:
                f.write(ele+'\n')
