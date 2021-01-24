#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:09:10 2021

@author: Ellen
"""

import gasex.airsea as AS

#L13(C,u10,SP,pt,*,slp=1.0,gas=None,rh=1.0,chi_atm=None)
[Fd, Fc, Fp, Deq, k] = AS.L13(C=0.01410,u10=5,SP=35,pt=10,slp=1,gas='Ar',rh=0.9)

#[Fd, Fc, Fp, Deq, k] = fas_N11(C,u10,S,T,slp,gas,rh)
#[Fd, Fc, Fp, Deq, k] = fas_N11(0.01410,5,35,10,1,'Ar',0.9)
#[Fd_N16, Fc_N16, Fp_N16, Deq_N16, k__N16]=AS.N16(C=0.01410,u10=5, SP=35, pt=10, slp=1, gas='Ar', rh=1)