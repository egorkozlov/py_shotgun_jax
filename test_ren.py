#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:04:20 2020

@author: egorkozlov
"""

import numpy as np

vf_in = np.array([-5.054238 , -2.8541722, -2.4162447, -2.1846287, -2.0306628,
             -1.916129 , -1.8248256, -1.7484286, -1.682069 , -1.6225789,
             -1.5676547, -1.51535  , -1.463579 , -1.4090296, -1.3393142], dtype=np.float32)
vm_in = np.array([-1.3393142, -1.4090296, -1.4635789, -1.5153499, -1.5676547,
             -1.622579 , -1.682069 , -1.7484286, -1.8248256, -1.9161291,
             -2.0306628, -2.1846287, -2.4162445, -2.8541725, -5.054239 ],dtype=np.float32)
vf_out = np.array([-4.401132],dtype=np.float32)
vm_out = np.array([-1.3826733],dtype=np.float32)


def ren_divorce_one(vf_in,vm_in,vf_out,vm_out,ts):
    
    # this takes one row and returns transition matrix over theta
    
    def L(x):
        return (1 + np.exp(-x/ts))**(-1)
    
    nt = vf_in.shape[0]
    
    M = np.minimum(vf_in - vf_out, vm_in - vm_out)
    M_ebs = np.max(M)
    i_ebs =  np.argmax(M)
    p_div = 1-L(M_ebs)
    p_sq = L(M)
    M_dext = np.concatenate((np.array([-np.inf]),M))
    M_uext = np.concatenate((M,np.array([-np.inf])))
    L_up = np.maximum( L(M_dext[1:]) - L(M_dext[:-1]), 0.0)
    L_up_cs = np.cumsum(L_up)
    L_down = np.maximum( L(M_uext[:-1]) - L(M_uext[1:]), 0.0)
    L_down_cs = np.cumsum(L_down[::-1])[::-1]
    
    #print(L_up)
    #print(L_down)
    
    
    
    p_stay = np.sum(L_up)
    #assert np.allclose(p_stay,np.sum(L_down))
    
    Mout = np.zeros((nt,nt),dtype=np.float32)
    
    for ir in range(nt):
        if ir < i_ebs:
            Mout[ir,ir] = L_up_cs[ir]
            Mout[ir,(ir+1):(i_ebs+1)] = L_up[(ir+1):(i_ebs+1)]
        elif ir > i_ebs:
            Mout[ir,ir] = L_down_cs[ir]
            Mout[ir,(i_ebs):ir] = L_down[(i_ebs):ir]
        elif ir == i_ebs:
            Mout[ir,ir] = p_stay
         
    ir = np.arange(nt)        
    
    it_c = ir[:,None]
    it_r = ir[None,:]
    iebs_c = i_ebs*np.ones((1,nt),dtype=np.int16)
    #iebs_r = i_ebs*np.ones((nt,1),dtype=np.int16)
    
    i_irtoebs = (it_r <= iebs_c) & (it_r > it_c)
    i_ebstoir = (it_r >= iebs_c) & (it_r < it_c)
    i_diagbeloweq = (it_r <= iebs_c) & (it_r == it_c)
    i_diagabove = (it_r > iebs_c) & (it_r == it_c)
    
    Mout2 = i_irtoebs*L_up[None,:] + i_ebstoir*L_down[None,:] + \
            i_diagbeloweq*L_up_cs[None,:] + \
            i_diagabove*L_down_cs[None,:]
    
    
    #Mout2 = Mout
    print(Mout)
    assert np.allclose(Mout2,Mout)
    
    return Mout
    
ts = 0.1
M = ren_divorce_one(vf_in,vm_in,vf_out,vm_out,ts)