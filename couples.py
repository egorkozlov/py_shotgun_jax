#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:45:56 2020

@author: egorkozlov
"""

import jax.numpy as np
from  jax.scipy.special import logsumexp as lse
from jax import jit


def iteration_couples(model,t,Vnext):
    s = model.setup
    sig = s.sigma
    bet = s.beta
    
    # compute EV
    
    ti = t if t < model.T else model.T-1
    EV = dot_3d(Vnext,s.zfzmpsi_mat[ti].T)
    
    i, wn, wt, sgrid = s.v_sgrid_c.i, s.v_sgrid_c.wnext, s.v_sgrid_c.wthis, s.v_sgrid_c.val

    wf = np.exp(s.zfzmpsi[ti][:,0])
    wm = np.exp(s.zfzmpsi[ti][:,1])
    
    
    money = s.R*s.agrid_c[:,None] + wf[None,:] + wm[None,:]
    umult = s.u_mult(s.theta_grid_coarse)
    
    #print(EV.shape)
    solver = jit(solve,static_argnums=[2,3,4,5,6,7,8])
    V = solver(money,EV,umult,sig,bet,i,wn,wt,sgrid)
    
    return V


def solve(money,EV,umult,sigma,beta,i,wn,wt,sgrid):
    
    ts = 0.1
    # dim is (na,ns,nexo,ntheta)
    EV_stretch = (wt[:,None,None]*EV[i,:,:] + wn[:,None,None]*EV[i+1,:,:])[None,:,:,:]
    consumption = money[:,None,:] - sgrid[None,:,None]
    consumption_negative = (consumption <= 0)
    uc = (np.maximum(consumption,1e-6))**(1-sigma)/(1-sigma)
    utility = umult[None,None,None,:]*(uc[:,:,:,None]) - 1e6*consumption_negative[:,:,:,None]
    
    mega_matrix = utility + beta*EV_stretch
    #print(mega_matrix.shape)
    
    V = ts*lse(mega_matrix/ts,axis=1)
    #print(V.mean())
    #print(V.max())
    #print(V.min())
    return V



def dot_3d(V,M):
    # this computes array Q, such that
    # Q[:,:,i] = V[:,:,i]*M for each i in V.shape[2]
    
    q_shape = (V.shape[0],M.shape[1],V.shape[2])
    Q = np.zeros(q_shape[:-1]+(0,),dtype=V.dtype)
    
    for k in range(V.shape[-1]):
        Q = np.concatenate((Q,np.dot(V[:,:,k],M)[:,:,None]),axis=2)
    return Q