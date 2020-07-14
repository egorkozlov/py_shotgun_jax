#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:45:56 2020

@author: egorkozlov
"""

import jax.numpy as np
from  jax.scipy.special import logsumexp as lse
from  jax.nn import softmax as sm
from jax import jit


def iteration_couples(model,t,Vnext,MUnext):
    s = model.setup
    sig = s.sigma
    bet = s.beta
    R = s.R
    # compute EV
    
    ti = t if t < model.T else model.T-1
    
    dot = jit(dot_3d)
    EV = dot(Vnext,s.zfzmpsi_mat[ti].T)
    EMU = dot(MUnext,s.zfzmpsi_mat[ti].T)
    
    
    
    agrid = s.agrid_c
    
    
    i, wn, wt, sgrid = s.v_sgrid_c.i, s.v_sgrid_c.wnext, s.v_sgrid_c.wthis, s.v_sgrid_c.val

    wf = np.exp(s.zfzmpsi[ti][:,0])
    wm = np.exp(s.zfzmpsi[ti][:,1])
    psi = s.zfzmpsi[ti][:,2][None,:,None]
    
    
    li = wf[None,:] + wm[None,:]
    money = R*s.agrid_c[:,None] + li
    
    umult = s.u_mult(s.theta_grid_coarse)
    
    
    
    #print(EV.shape)
    #solver = jit(solve,static_argnums=[2,3,4,5,6,7,8])
    V_vfi, s_vfi = solve_vfi(money,EV,umult,sig,bet,i,wn,wt,sgrid,psi)    
    
    last = (t==model.T)
    V, s, MU = solve_egm(EV,EMU,li,umult,agrid,sig,bet,R,i,wn,wt,psi,last) 
    print((V.mean(),V_vfi.mean()))
    print((s.mean(),s_vfi.mean()))
    
    return V, MU, s


def solve_egm(EV,EMU,li,umult,agrid,sigma,beta,R,i,wn,wt,psi,last):
    
    if not last:
        c_prescribed = (beta*R*EMU / umult[None,None,:])**(-1/sigma) 
        m_implied = c_prescribed + agrid[:,None,None]
        a_implied = (1/R)*(m_implied - li[:,:,None])
        
        a_i_min = a_implied[0,...]
        a_i_max = a_implied[-1,...]
        
        
        j, wn = interp_manygrids(a_implied,agrid,axis=0,trim=True)
        s_egm = agrid[j]*(1-wn) + agrid[j+1]*wn
        EV_egm = np.take_along_axis(EV,j,axis=0)*(1-wn) + \
                 np.take_along_axis(EV,j+1,axis=0)*(wn)
        
        agrid_r = agrid.reshape((agrid.size,) + a_i_min.ndim*(1,))
        i_above = (agrid_r >= a_i_max)
        i_below = (agrid_r <= a_i_min)
        i_egm = (~i_above) & (~i_below)
        
        s_below = 0.0
        s_above = agrid[-1]
        
        EV_below = EV[:1, ...]
        EV_above = EV[-1:,...]
        
        
        s = s_egm*i_egm + s_above*i_above # + 0.0*i_below
        EV_int = EV_egm*i_egm + EV_above*i_above + EV_below*i_below
        c = R*agrid_r + li[:,:,None] - s
        
        assert np.all(s>=s_below)
        assert np.all(s<=s_above)
        assert np.all(c>=0)
        
    else:
        
        c = R*agrid[:,None,None] + li[:,:,None]
        s = 0.0*c
        EV_int = EV # it is 0 anyways
        
    
    MUc = umult[None,None,:]*(c**(-sigma))
    
    # this EV should be interpolated as well
    # 
    V = umult[None,None,:]*(c**(1-sigma)/(1-sigma)) + psi + beta*EV_int
    return V, s, MUc


def interp(grid,xnew,return_wnext=True,trim=False):    
    # this finds grid positions and weights for performing linear interpolation
    # this implementation uses numpy
    
    if trim: xnew = np.clip(xnew,grid[0],grid[-1])
    
    j = np.minimum( np.searchsorted(grid,xnew,side='left')-1, grid.size-2 )
    wnext = (xnew - grid[j])/(grid[j+1] - grid[j])
    
    return j, (wnext if return_wnext else 1-wnext) 



def interp_manygrids(grids,xs,axis=0,return_wnext=True,trim=False):
    # this routine interpolates xs on many grids, defined along 
    # the axis in an array grids. (so for axis=0 grids are 
    #grids[:,i,j,k] for all i, j, k)
    
    
    assert np.all(np.diff(grids,axis=axis) > 0)
    
    '''
    if trim: xs = np.clip(xs[:,None,None],
                          grids.min(axis=axis,keepdims=True),
                          grids.max(axis=axis,keepdims=True))
    '''
    
    # this requires everything to be sorted
    mat = grids[...,None] < xs[(None,)*grids.ndim + (slice(None),)]
    ng  = grids.shape[axis]
    j = np.clip(np.sum(mat,axis=axis)[None,...]-1,0,ng-2)
    j = np.swapaxes(j,-1,axis).squeeze(axis=-1)
    grid_j = np.take_along_axis(grids,j,axis=axis)
    grid_jp = np.take_along_axis(grids,j+1,axis=axis)
    xs_r = xs.reshape((1,)*(axis-1) + (xs.size,) + (1,)*(grids.ndim - 1 - axis))
    wnext = (xs_r - grid_j)/(grid_jp - grid_j)
    return j, (wnext if return_wnext else 1-wnext)




def solve_vfi(money,EV,umult,sigma,beta,i,wn,wt,sgrid,psi):
    
    #ts = 0.01
    # dim is (na,ns,nexo,ntheta)
    EV_stretch = (wt[:,None,None]*EV[i,:,:] + wn[:,None,None]*EV[i+1,:,:])[None,:,:,:]
    consumption = money[:,None,:] - sgrid[None,:,None]
    consumption_negative = (consumption <= 0)
    uc = (np.maximum(consumption,1e-8))**(1-sigma)/(1-sigma)
    utility = umult[None,None,None,:]*(uc[:,:,:,None]) - 1e8*consumption_negative[:,:,:,None]
    
    mega_matrix = utility + beta*EV_stretch
    #print(mega_matrix.shape)
    
    ind_s = mega_matrix.argmax(axis=1)
    V = np.take_along_axis(mega_matrix,ind_s[:,None,:,:],1).squeeze(axis=1) + psi
    s = sgrid[ind_s]
    return V, s






def dot_3d(V,M):
    # this computes array Q, such that
    # Q[:,:,i] = V[:,:,i]*M for each i in V.shape[2]
    
    q_shape = (V.shape[0],M.shape[1],V.shape[2])
    Q = np.zeros(q_shape[:-1]+(0,),dtype=V.dtype)
    
    for k in range(V.shape[-1]):
        Q = np.concatenate((Q,np.dot(V[:,:,k],M)[:,:,None]),axis=2)
    return Q