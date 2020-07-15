#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:13:57 2019

@author: egorkozlov
"""

import jax.numpy as jnp
import numpy as onp
np = jnp

# this is a gridded linear interpolant that uses fast numpy interpolation

class VecOnGrid(object):
    def __init__(self,grid,values,trim=True,fix_w=True):
        # this assumes grid is strictly increasing o/w unpredictable
        self.val = values
        self.val_trimmed = np.clip(values,grid[0],grid[-1])
        self.grid = grid
        
        i, wnext = interp(self.grid,self.val,return_wnext=True,trim=trim)
        
        
        i_neg = (i<0)
        assert np.allclose(wnext[i_neg],1.0)
        i = i*(~i_neg) # + 0*(i_neg)
        wnext = wnext*(~i_neg) # + 0.0*i_neg
        
        
            
            
        if fix_w:
            i_fix_w = (np.isclose(wnext,1.0)) & (i < (self.grid.size-2))
            self.i = i*(~i_fix_w) + (i+1)*(i_fix_w)
            self.wnext = wnext*(~i_fix_w) # + 0.0*(i_fix_w)
        else:
            self.i = i
            self.wnext = wnext
            
        self.wnext = self.wnext.astype(grid.dtype)
        
        self.n = self.i.size
        
        self.one = np.array(1).astype(grid.dtype) # sorry
        
        self.wthis = self.one-self.wnext
        self.trim = trim
        
        i_next_closest = (self.wnext>0.5) 
        self.i_closest = self.i + i_next_closest
        
        
        
    def roll(self,shocks=None):
        # this draws a random vector of grid poisitions such that probability
        # of self.i is self.wthis and probability of self.i+1 is self.wnext
        
        if shocks is None:
            print('Warning: fix the seed please')
            shocks = np.random.random_sample(self.val.shape)
            
        out = self.i
        out[shocks>self.wthis] += 1
        return out


def interp(grid,xnew,return_wnext=True,trim=False):    
    # this finds grid positions and weights for performing linear interpolation
    # this implementation uses numpy
    
    if trim: xnew = np.clip(xnew,grid[0],grid[-1])
    
    j = np.minimum( np.searchsorted(grid,xnew,side='left')-1, grid.size-2 )
    wnext = (xnew - grid[j])/(grid[j+1] - grid[j])
    
    return j, (wnext if return_wnext else 1-wnext) 