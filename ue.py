#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:59:25 2020

@author: egorkozlov
"""

import numpy as onp
import jax.numpy as jnp
np = jnp
from jax.experimental import loops
from jax.lax import cond
from jax import vmap, jit



def upper_envelope_vmap(bEV,a_implied,c_implied,agrid,li,um,R,sig):
    uvm = vmap(upper_envelope_one,in_axes=(1,1,1,None,0,0,None,None),out_axes=(1,1))
    return uvm(bEV,a_implied,c_implied,agrid,li,um,R,sig)



def upper_envelope_one(bEV,a_implied,c_implied,agrid,li,um,R,sig):
    
        
    def u(c):
        return c**(1-sig)/(1-sig)
    
    na = bEV.shape[0]

    ap = a_implied
    Wp = bEV
    cp = c_implied
    
    ai_min = ap.min()
    ai_max = ap.max()
    
    
    # compute slopes, low and high poitnts
    
    ai_low = ap[:-1]
    ai_hig = ap[ 1:]
    ci_low = cp[:-1]
    ci_hig = cp[ 1:]
    Wi_low = Wp[:-1]
    Wi_hig = Wp[ 1:]
    
    
    
    ci_slope = (ci_hig - ci_low) / (ai_hig - ai_low)
    Wi_slope = (Wi_hig - Wi_low) / (ai_hig - ai_low)
    
    
    with loops.Scope() as s:
        
        s.c_list = np.zeros((na,))
        s.V_list = np.zeros((na,))
        
        
        
        
        for ia in s.range(na):
            cthis = 0.0
            Vthis = -np.inf
            
            # true upper envelope
            # brute force with precomputed points and slopes
            athis = agrid[ia]
                            
            for iap in s.range(na-1):
                
               
                
                
                da0 = ai_low[iap] - athis
                da1 = athis - ai_hig[iap]
                prod = da0*da1
                c = (prod >= 0)
                
                for _ in s.cond_range(c):
                    c_guess = ci_low[iap] + ci_slope[iap] * (athis - ai_low[iap])
                    W_guess = Wi_low[iap] + Wi_slope[iap] * (athis - ai_low[iap])
                    V_guess = um*u(c_guess) + W_guess
                    Vout = np.where(V_guess > Vthis,V_guess,Vthis)
                    cout = np.where(V_guess > Vthis,c_guess,cthis)
                    cthis, Vthis = cout, Vout
                    s.c_list = s.c_list.at[ia].set(cthis)
                    s.V_list = s.V_list.at[ia].set(Vthis)
                
    
                        
            for _ in s.cond_range(agrid[ia] < ai_min):
                cthis = R*agrid[ia] + li
                Vthis = um*u(cthis) + bEV[0]
                s.c_list = s.c_list.at[ia].set(cthis)
                s.V_list = s.V_list.at[ia].set(Vthis)
    
            for _ in s.cond_range( agrid[ia] > ai_max):
                cthis = R*agrid[ia] + li - agrid[-1]
                Vthis = um*u(cthis) + bEV[-1]
                s.c_list = s.c_list.at[ia].set(cthis)
                s.V_list = s.V_list.at[ia].set(Vthis)
                
            
                        
        #print(s.c_list)
        #print(s.V_list)
        
    return s.c_list, s.V_list




def upper_envelope(bEV,a_implied,c_implied,agrid,li,um,R,sig):
    
    
    
    na, no = bEV.shape
    dt = c_implied.dtype
    c_out = np.zeros((na,0),dtype=dt)
    V_out = np.zeros((na,0),dtype=dt)
    
    def u(c):
        return c**(1-sig)/(1-sig)
    
    
    
    
    for io in range(no):
        
        ap = a_implied[:,io]
        Wp = bEV[:,io]
        cp = c_implied[:,io]
        
        ai_min = ap.min()
        ai_max = ap.max()
        
        
        # compute slopes, low and high poitnts
        
        ai_low = ap[:-1]
        ai_hig = ap[ 1:]
        ci_low = cp[:-1]
        ci_hig = cp[ 1:]
        Wi_low = Wp[:-1]
        Wi_hig = Wp[ 1:]
        
        
        
        ci_slope = (ci_hig - ci_low) / (ai_hig - ai_low)
        Wi_slope = (Wi_hig - Wi_low) / (ai_hig - ai_low)
        
        
        with loops.Scope() as s:
            
            s.c_list = np.zeros((na,))
            s.V_list = np.zeros((na,))
            
            
            
            
            for ia in s.range(na):
                cthis = 0.0
                Vthis = -np.inf
                
                # true upper envelope
                # brute force with precomputed points and slopes
                athis = agrid[ia]
                                
                for iap in s.range(na-1):
                    
                   
                    
                    
                    da0 = ai_low[iap] - athis
                    da1 = athis - ai_hig[iap]
                    prod = da0*da1
                    c = (prod >= 0)
                    
                    for _ in s.cond_range(c):
                        c_guess = ci_low[iap] + ci_slope[iap] * (athis - ai_low[iap])
                        W_guess = Wi_low[iap] + Wi_slope[iap] * (athis - ai_low[iap])
                        V_guess = um[io]*u(c_guess) + W_guess
                        Vout = np.where(V_guess > Vthis,V_guess,Vthis)
                        cout = np.where(V_guess > Vthis,c_guess,cthis)
                        cthis, Vthis = cout, Vout
                        s.c_list = s.c_list.at[ia].set(cthis)
                        s.V_list = s.V_list.at[ia].set(Vthis)
                    
        
                            
                for _ in s.cond_range(agrid[ia] < ai_min):
                    cthis = R*agrid[ia] + li[io]
                    Vthis = um[io]*u(cthis) + bEV[0,io]
                    s.c_list = s.c_list.at[ia].set(cthis)
                    s.V_list = s.V_list.at[ia].set(Vthis)
        
                for _ in s.cond_range( agrid[ia] > ai_max):
                    cthis = R*agrid[ia] + li[io] - agrid[-1]
                    Vthis = um[io]*u(cthis) + bEV[-1,io]
                    s.c_list = s.c_list.at[ia].set(cthis)
                    s.V_list = s.V_list.at[ia].set(Vthis)
                    
                
                            
            print(s.c_list)
            print(s.V_list)
            #assert False
            c_out = np.hstack((c_out,s.c_list[:,None].copy()))
            V_out = np.hstack((V_out,s.V_list[:,None].copy()))
    
    return c_out, V_out


jit_here = lambda f : jit(f,static_argnums=[3,6,7])
@jit_here
def upper_envelope_matrix(bEV,a_implied,c_implied,agrid,li,um,R,sig):
    
    sl = (slice(None,-1),slice(None))
    su = (slice(1,None),slice(None))
    ai_low = a_implied[sl]
    ai_high = a_implied[su]
    da = ai_high - ai_low
    ci_low = c_implied[sl]
    ci_high = c_implied[su]
    Wi_low = bEV[sl]
    Wi_high = bEV[su]
    ci_slope = (ci_high - ci_low)/da
    Wi_slope = (Wi_high - Wi_low)/da
    
    
    def u(c):
        return np.maximum(c,1e-8)**(1-sig)/(1-sig)
    
    
    da = agrid[:,None,None] - ai_low[None,:,:]
    da_p = ai_high[None,:,:] - agrid[:,None,None]
    
    c_allint = ci_low[None,:,:] + ci_slope[None,:,:]*da
    W_allint = Wi_low[None,:,:] + Wi_slope[None,:,:]*da
    V_allint = um[None,None,:]*u(c_allint) + W_allint
    
    i_outside = (da*da_p < 0)
    
    
    V_check = V_allint - 1e20*i_outside
    i_best = V_check.argmax(axis=1)[:,None,:]
    
    
    
    c_egm = np.take_along_axis(c_allint,i_best,1).squeeze(axis=1)
    V_egm = np.take_along_axis(V_check,i_best,1).squeeze(axis=1)
    
    c_below = R*agrid[:,None] + li[None,:]
    V_below = um[None,:]*u(c_below) + bEV[0:1,:]
    c_above = R*agrid[:,None] + li[None,:] - R*agrid[-1]
    V_above = um[None,:]*u(c_above) + bEV[-1:,:] # this can be a bad number
    
    aimin = a_implied.min(axis=0)
    aimax = a_implied.max(axis=0)
    
    i_below = (agrid[:,None] <= aimin[None,:])
    i_above = (agrid[:,None] >= aimax[None,:])
    i_egm = (~i_below) & (~i_above)
    
    c_out = i_egm*c_egm + i_below*c_below + i_above*c_above
    V_out = i_egm*V_egm + i_below*V_below + i_above*V_above
    
    
    return c_out, V_out

