#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 08:55:39 2021

@author: Ellen
"""

import xarray as xr
import matplotlib.pyplot as plt
import RandomFxns as RF

dac= 'meds'
WMO = 4901141

f = RF.ArgoDataLoader(DAC=dac, WMO=WMO)

plt.figure()
f.plot.scatter(x='JULD',y='PRES_ADJUSTED',hue='DOXY_ADJUSTED')
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.show()