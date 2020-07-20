#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:49:35 2020

@author: egorkozlov
"""

import jax.numpy as np

def iteration_singles(model,t,Vnext,MUnext,female):
    s = model.setup
    sig = s.sigma
    bet = s.beta
    R = s.R
    
    # compute EV
    
    ti = t if t < model.T else model.T-1
    M = s.zf_mat[ti].T if female else s.zm_mat[ti].T
    
    wage = np.exp((s.zf_grid[ti] if female else s.zm_grid[ti]))
    
    agrid = s.agrid_s
    sgrid = s.sgrid_s
    vsgrid = s.v_sgrid_s
    
    
    i, wn, wt, sgrid = vsgrid.i, vsgrid.wnext, vsgrid.wthis, vsgrid.val
    
    
    
    money = R*agrid[:,None] + wage[None,:]
    
    EV_in = np.dot(Vnext,M)
    
    V_vfi, s_vfi = solve_singles_vfi(EV_in,money,sig,bet,i,wn,wt,sgrid)
    
    
    
    EMU = np.dot(MUnext,M)
    
    last = (t==model.T)
    li = wage[None,:]
    
    V, s, MU = \
                solve_singles_egm(EV_in,EMU,li,agrid,sig,bet,R,i,wn,wt,last) 
    
    
    #print('V egm {}, vfi {}'.format(V.mean(),V_vfi.mean()))
    #print('s egm {}, vfi {}'.format(s.mean(),s_vfi.mean()))
    return V, MU, s


def solve_singles_vfi(EV_in,money,sigma,beta,i,wn,wt,sgrid):
    EV = wt[:,None]*EV_in[i,:] + wn[:,None]*EV_in[i+1,:]
    
    consumption = money[:,None,:] - sgrid[None,:,None]
    
    consumption_negative = (consumption <= 0)
    
    utility = (np.maximum(consumption,1e-8))**(1-sigma)/(1-sigma) - \
                                                    1e9*consumption_negative
    
    
    
    mega_matrix = utility + beta*EV
    #print(mega_matrix.shape)
    
    ind_s = mega_matrix.argmax(axis=1)
    V = np.take_along_axis(mega_matrix,ind_s[:,None,:],1).squeeze(axis=1) 
                                                
                                                
    s = sgrid[ind_s]
    c = money - s
    
    V_check = (c**(1-sigma)/(1-sigma)) + beta*np.take_along_axis(EV,ind_s,0)
                
    assert np.allclose(V_check,V,atol=1e-5)
    
    return V, s

from couples import interp_manygrids
def solve_singles_egm(EV,EMU,li,agrid,sigma,beta,R,i,wn,wt,last):
    
    if not last:
        
        c_prescribed = (beta*R*EMU)**(-1/sigma) 
        m_implied = c_prescribed + agrid[:,None]
        a_implied = (1/R)*(m_implied - li)
        
        a_i_min = a_implied[0,...]
        a_i_max = a_implied[-1,...]
        
        
        j, wn = interp_manygrids(a_implied,agrid)
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
        
        c = R*agrid_r + li - s
        
        assert np.all(s>=s_below)
        assert np.all(s<=s_above)
        assert np.all(c>=0)
        
    else:
        
        c = R*agrid[:,None] + li
        s = 0.0*c
        EV_int = EV # it is 0 anyways
    
    
    MUc = (c**(-sigma))
    
    # this EV should be interpolated as well
    
    V = (c**(1-sigma)/(1-sigma)) + beta*EV_int
    
    return V, s, MUc
