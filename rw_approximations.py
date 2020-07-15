#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:38:02 2020

@author: egorkozlov
"""

import jax.numpy as jnp
import numpy as onp
from jax import ops

np = jnp


def rouw_nonst_jax(T,sigma_persistent,sigma_init,npts):
    sd_z = sd_rw(T,sigma_persistent,sigma_init)
    assert(npts>=2)
    Pi = list()
    X = list()
    
    for t in range(0,T+1):
        nsd = np.sqrt(npts-1)
        if t <= T-1: X = X + [np.linspace(-nsd*sd_z[t],nsd*sd_z[t],num=npts)]
        gen = rouw_nonst_one #jit(rouw_nonst_one,static_argnums=[2])
        if t >= 1: Pi = Pi + [gen(sd_z[t-1],sd_z[t],npts)]
            
    #Pi = Pi + [None] # last matrix is not defined
    
    return X, Pi

def sd_rw(T,sigma_persistent,sigma_init):
    return np.sqrt(sigma_init**2 + np.arange(0,T+1)*(sigma_persistent**2))
  

def rouw_nonst_one(sd0,sd1,npts):
    # this generates one-period Rouwenhorst transition matrix
    assert(npts>=2)
    pi0 = 0.5*(1+(sd0/sd1))
    Pi = np.array([[pi0,1-pi0],[1-pi0,pi0]])
    #assert(pi0<1)
    #assert(pi0>0)
    for n in range(3,npts+1):
        Z = np.zeros((n,n))
        s0 = slice(0,n-1)
        s1 = slice(1,n)
        if np is jnp:
            A = ops.index_update(Z,(s0,s0),Pi)
            B = ops.index_update(Z,(s0,s1),Pi)
            C = ops.index_update(Z,(s1,s1),Pi)
            D = ops.index_update(Z,(s1,s0),Pi)
            Pi0 = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
            Pi = ops.index_update(Pi0,(slice(1,n-1),None),0.5*Pi0[(slice(1,n-1),None)])
        else:
            A = Z.copy()
            A[(s0,s0)] = Pi
            B = Z.copy()
            B[(s0,s1)] = Pi
            C = Z.copy()
            C[(s1,s1)] = Pi
            D = Z.copy()
            D[(s1,s0)] = Pi
            Pi = pi0*A + (1-pi0)*B + pi0*C + (1-pi0)*D
            Pi[1:(n-1),:] = 0.5*Pi[1:(n-1),:]
        assert(np.all(np.abs(np.sum(Pi,axis=1)-1)<1e-5 ))
    
    return Pi


def combine_matrices(a,b,Pia,Pib,check=True):
    # this combines INDEPENDENT transition matrices Pia and Pib
    grid = mat_combine(a,b)
    
    Pi = np.kron(Pia,Pib)
    
    if check:
        assert(all(abs(np.sum(Pi,axis=1)-1)<1e-5))
    
    return grid, Pi


def mat_combine(a,b):
    # this gets combinations of elements of a and b
    
    a = a[:,np.newaxis] if a.ndim == 1 else a
    b = b[:,np.newaxis] if b.ndim == 1 else b
    
    assert a.ndim==2 and b.ndim==2
    
    l_a = a.shape[0]
    l_b = b.shape[0]
    
    w_a = a.shape[1]
    w_b = b.shape[1]
    
    
    grid = np.zeros((0,w_a+w_b))
    for ia in range(l_a):
        grid_add = np.hstack((a[ia:ia+1,:]*np.ones((l_b,1)),b))
        grid = np.vstack((grid,grid_add))
        
    return grid

