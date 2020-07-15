#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:38:11 2020

@author: egorkozlov
"""

import numpy as onp
import jax.numpy as jnp
np = onp

def compare_arrays(a0,a1,name='',verbose=True,diff=True):
    stats = [np.mean,np.max,np.min,np.median,
             lambda x : np.quantile(x,0.25),
             lambda x : np.quantile(x,0.75)]
            
    descs = ['avg','max','min','med','q25','q75']
    
    sh = ('a1-a0' if diff else 'a1, a0')
    print('comparing {}, {}: '.format(name,sh),end=' ')
    for stat, desc in zip(stats,descs):
        v0 = stat(a0)
        v1 = stat(a1)
        
        if diff:
            print('{} {:.2f}'.format(desc,v1-v0),end = ' ')
        else:
            print('{} are {:.2f}, {:.2f}'.format(desc,v0,v1),' ')
    print('')