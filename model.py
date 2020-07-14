#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:42:24 2020

@author: egorkozlov
"""

from couples import iteration_couples
import jax.numpy as np
from timeit import default_timer as dt



from setup import Setup
class Model(object):
    def __init__(self,**kwagrs):
        self.setup = Setup(**kwagrs)
        self.t0 = dt()
        s = self.setup
        self.T = self.setup.T
        self.v_couple_shape = [(s.na_c,s.nexo_t[t],s.ntheta_coarse) for t in range(self.T)]
        print('setup created, time {}'.format(dt() - self.t0))
        self.solve()
        
    def solve(self):
        self.V = []
        self.s = []
        self.MU = []
        
        for t in range(self.T,0,-1):
            # dtype - ?
            print('solving for t = {}'.format(t))
            try:
                Vnext = self.V[0] 
                MUnext = self.MU[0] 
            except:
                Vnext = 3*(np.zeros(self.v_couple_shape[-1]),)
                MUnext = np.zeros(self.v_couple_shape[-1])
            
            
            Vthis, MUthis, s = iteration_couples(self,t,Vnext,MUnext)
            self.s = [s] + self.s
            self.V = [Vthis] + self.V
            self.MU = [MUthis] + self.MU
            print('min s: {}, max s: {}, mean s: {}'.format(s.min(),s.max(),s.mean()))
            print('done, time {}'.format(dt() - self.t0))
      
q = Model(beta=0.95,T=10).s[0].mean()
#from jax import grad

#ds = grad(lambda x : Model(beta=x,T=10).s[0].mean())(0.95)


