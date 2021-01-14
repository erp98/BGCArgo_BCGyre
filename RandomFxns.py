#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:10:59 2021

@author: Ellen
"""
import numpy as np

def DetermineAdjusted(raw_data,a_data):
    # raw_data = not adjusted BGC Argo data
    # a_data = adjuasted BGC Argo data
    search_flag=0
    j=0
    adj_flag=np.NaN
    
    raw_ss=raw_data[0,:]
    a_ss=a_data[0,:]
    
    while (j < len(raw_ss) and search_flag == 0):
        
        if (np.isnan(raw_ss[j]) != True and np.isnan(a_ss[j]) != True):
            search_flag=1
            adj_flag=1
        elif (np.isnan(raw_ss[j]) != True and np.isnan(a_ss[j]) == True):
            search_flag=1
            adj_flag=0
        elif (np.isnan(raw_ss[j]) == True and np.isnan(a_ss[j]) == True):
            j=j+1
    
    if adj_flag == 1:
        output_data = a_data
    elif adj_flag == 0:
        output_data = raw_data
    elif np.isnan(adj_flag) == True:
        print('ERROR: Using raw data')
        output_data=raw_data
        
    return output_data, adj_flag
                
            
