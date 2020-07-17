#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:42:24 2020

@author: egorkozlov
"""

from couples import iteration_couples
from singles import iteration_singles
import jax.numpy as jnp
import numpy as onp
from timeit import default_timer as dt
np = jnp





from setup import Setup
class Model(object):
    def __init__(self,**kwagrs):
        self.setup = Setup(**kwagrs)
        self.t0 = dt()
        s = self.setup
        self.T = self.setup.T
        self.v_couple_shape = [(s.na_c,s.nexo_t[t],s.ntheta_coarse) for t in range(self.T)]
        self.v_sf_shape = [(s.na_s,s.n_zf) for t in range(self.T)]
        self.v_sm_shape = [(s.na_s,s.n_zm) for t in range(self.T)]
        print('setup created, time {}'.format(dt() - self.t0))
        self.solve()
        
    def solve(self):
        self.VC = []
        self.sc = []
        self.MUC = []
        
        self.VSF = []
        self.ssf = []
        self.MUSF = []
        
        self.VSM = []
        self.ssm = []
        self.MUSM = []
        
        for t in range(self.T,0,-1):
            # couples:
            
            print('solving for t = {}'.format(t))
            try:
                Vnext  = self.VC[0]  + (self.VSF[0], self.VSM[0] )
                MUnext = (self.MUC[0],) + (self.MUSF[0],self.MUSM[0])
            except:
                zz = (np.zeros(self.v_sf_shape[-1]),
                      np.zeros(self.v_sm_shape[-1]))
                
                Vnext = 3*(np.zeros(self.v_couple_shape[-1]),) + zz
               
                MUnext = (np.zeros(self.v_couple_shape[-1]),) + zz
                
            
            
            Vthis, MUthis, s = iteration_couples(self,t,Vnext,MUnext)
            self.sc = [s] + self.sc
            self.VC = [Vthis] + self.VC
            self.MUC = [MUthis] + self.MUC
            print('couples done, time {}'.format(dt() - self.t0))
            
            
            # singles:
            
            try:
                Vnext = self.VSF[0] 
                MUnext = self.MUSF[0] 
            except:
                Vnext = np.zeros(self.v_sf_shape[-1])
                MUnext = np.zeros(self.v_sf_shape[-1])
            
            Vthis, MUthis, s = iteration_singles(self,t,Vnext,MUnext,True)
            self.ssf = [s] + self.ssf
            self.VSF = [Vthis] + self.VSF
            self.MUSF = [MUthis] + self.MUSF
            print('single female done, time {}'.format(dt() - self.t0))
            
            
            try:
                Vnext = self.VSM[0] 
                MUnext = self.MUSM[0] 
            except:
                Vnext = np.zeros(self.v_sm_shape[-1])
                MUnext = np.zeros(self.v_sm_shape[-1])
            
            Vthis, MUthis, s = iteration_singles(self,t,Vnext,MUnext,False)
            self.ssm = [s] + self.ssm
            self.VSM = [Vthis] + self.VSM
            self.MUSM = [MUthis] + self.MUSM
            print('single male done, time {}'.format(dt() - self.t0))
            
            
            
    
t0 = dt()
#q = Model(beta=0.95,T=10).ssf[0].mean()
from jax import value_and_grad, jacfwd, jacrev




ff = lambda x : Model(sig_zf=x[0],sig_zm=x[1],sig_zf_init=x[2],sig_zm_init=x[3],T=10).sc[0].mean()

pt = np.array([0.2,0.2,0.4,0.4])

'''
val0 = ff(pt)
print('value computed for {} sec'.format(dt() - t0))
print('value 0 is {}'.format(val0))
t0 = dt()
'''




val, grad = value_and_grad(ff)(pt)
print('value and grad computed for {} sec'.format(dt() - t0))
print('value is {}, grad is {}'.format(val,grad))


'''
t0 = dt()
H = jacrev(jacrev(ff))(pt)
print('hessian computed for {} sec'.format(dt() - t0))
print('hessian is {}'.format(H))
'''


