#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:15:16 2020

@author: egorkozlov
"""

import jax.numpy as jnp
import numpy as onp
np = jnp
from gridvec import VecOnGrid

class Setup(object):
    
    
    def __init__(self,**kwargs):
        '''
        this sets basic parameters in arbitrary formats
        passing additional kwargs like "key = value" replaces values in
        self.pars
        '''
        
        # below this line define parameters 
        beta = 0.98
        R = 0.98
        sigma = 1.5
        couple_rts = 0.23 
        A = 1.0
        T = 55
        T_ret = 40        
        pmeet_21 = 0.1
        pmeet_28 = 0.2
        pmeet_35 = 0.1        
        u_lost_divorce = 0.1
        
        n_zf = 4
        n_zm = 3
        n_psi = 3
        sig_zf = 0.2
        sig_zf_init = 0.4
        sig_zm = 0.2
        sig_zm_init = 0.4
        sig_psi = 0.1
        sig_psi_init = 0.8
        
        
        
        
        
        # this part just packs what is defined above to a dict
        l = locals()
        self.pars = dict()      
        for q in l:
            if q == 'self' or q == 'kwargs': continue
            self.pars[q] = l[q]
        
        # this replaces variables in dict with kwagrs entries
        for key in kwargs:
            assert (key in self.pars), '{} not found!'.format(key)
            self.pars[key] = kwargs[key]
            
        self.get_derivative_params() # this simple things derived from pars
        self.get_grids()        
        self.check_conflicts()
        
        
    def all_indices(self,ind_or_inds=None):
        # this is a tool to convert indices of zf, zm and psi into iexo
        
        if ind_or_inds is None: 
            ind_or_inds = np.array(range(self.nexo_t[0]))
        
        if isinstance(ind_or_inds,tuple):
            izf,izm,ipsi = ind_or_inds
            ind = izf*self.n_zm*self.n_psi + izm*self.n_psi + ipsi
        else:
            ind = ind_or_inds
            izf = ind // (self.n_zm*self.n_psi)
            izm = (ind - izf*self.n_zm*self.n_psi) // self.n_psi
            ipsi = ind - izf*self.n_zm*self.n_psi - izm*self.n_psi
            
        return ind, izf, izm, ipsi
        
        
        
    def get_derivative_params(self):
        self.derivative_pars = dict()
        self.derivative_pars['beta_t'] = [self['beta']]*self['T']
    
    
    def get_grids(self):
        self.get_asset_grids()
        self.get_singles_grids()
        self.get_couples_grid()
        self.get_theta_grid()
    
    
    
       
    def get_asset_grids(self):
        self.agrids = dict()
        g = self.agrids
        factor_c = 2.0
        pwr = 0.0
        pivot = 0.1
        
        g['na_s'] = 40
        g['na_c'] = 50
        g['amin'] = 0.0
        g['amax_s'] = 40.0
        g['amax_c'] = factor_c*g['amax_s']
        
        
        def flex_space(amin,amax,na):
            if pwr == 0.0:
                return -pivot + np.exp(np.linspace(np.log(amin+pivot),np.log(amax+pivot),na))
            else:
                return -pivot + (np.linspace((amin+pivot)**pwr,(amax+pivot)**pwr,na))**(1/pwr)
        
        g['agrid_s'] = flex_space(g['amin'],g['amax_s'],g['na_s'])
        g['agrid_c'] = flex_space(g['amin'],g['amax_c'],g['na_c'])
        
        n_between = 7
        da_min = 0.01
        da_max = 0.25
        
        g['sgrid_s'] = build_s_grid(g['agrid_s'],n_between,da_min,da_max)
        g['sgrid_c'] = build_s_grid(g['agrid_c'],n_between,da_min,da_max)
        
        g['v_sgrid_s'] = VecOnGrid(g['agrid_s'],g['sgrid_s'])
        g['v_sgrid_c'] = VecOnGrid(g['agrid_c'],g['sgrid_c'])
        
    def get_singles_grids(self):
        
        
        from rw_approximations import rouw_nonst_jax
        gen = rouw_nonst_jax
        
        
        self.female_grid = dict()
        sig_zf_init = 0.4
        sig_zf = 0.16
        zf, zf_mat = gen(self.T,sig_zf,sig_zf_init,self.n_zf)
        self.female_grid['zf_grid'] = zf
        self.female_grid['zf_mat'] = zf_mat
        
        
        self.male_grid = dict()
        sig_zm_init = 0.4
        sig_zm = 0.16
        zm, zm_mat = gen(self.T,sig_zm,sig_zm_init,self.n_zm)
        self.male_grid['zm_grid'] = zm
        self.male_grid['zm_mat'] = zm_mat
        
    
    def get_couples_grid(self):
        
        self.couples_grid = dict()
        from rw_approximations import rouw_nonst_jax as gen
        psi, psi_mat = gen(self.T,self.sig_psi,self.sig_psi_init,self.n_psi)
        
        self.couples_grid['psi'] = psi
        self.couples_grid['psi_mat'] = psi_mat
        
        from rw_approximations import combine_matrices
        
        self.couples_grid['zfzmpsi'] = []
        self.couples_grid['zfzmpsi_mat'] = []
        self.couples_grid['nexo_t'] = []
        
        
        
        for t in range(self.T):
            
            zfzm, zfzm_mat = combine_matrices(self.zf_grid[t],self.zm_grid[t],
                                              self.zf_mat[t],self.zm_mat[t])
            
            zzp, zzp_mat = combine_matrices(zfzm,psi[t],zfzm_mat,psi_mat[t])
            
            self.couples_grid['zfzmpsi'].append(zzp)
            self.couples_grid['zfzmpsi_mat'].append(zzp_mat)
            
            self.couples_grid['nexo_t'].append(zzp.shape[0])
        
        
    
    def get_theta_grid(self):
        
        self.theta_grid = dict()
        tg = self.theta_grid
        ntheta = 7
        ntheta_fine = 121 # preliminary
        theta_min = 0.01
        theta_max = 0.99
        
        tg['theta_grid_coarse'] = np.linspace(theta_min,theta_max,ntheta)
        tg['ntheta_coarse'] = ntheta

        tfine = np.unique(np.concatenate((np.linspace(theta_min,theta_max,ntheta_fine),tg['theta_grid_coarse'])))
        
        tg['theta_gird_fine'] = tfine
        tg['ntheta_fine'] = tfine.size
        tg['v_theta'] = VecOnGrid(tg['theta_grid_coarse'],tg['theta_gird_fine'])
        
        
    
    def u_mult(self,theta):
        assert np.all(theta > 0) and np.all(theta < 1)
        powr = (1+self.couple_rts)/(self.couple_rts+self.sigma)
        tf = theta
        tm = 1-theta
        ces = (tf**powr + tm**powr)**(1/powr)
        umult = (self.A**(1-self.sigma))*ces
        
        return umult
    
    
    def c_mult(self,theta):
        
        assert np.all(theta > 0) and np.all(theta < 1)
        powr = (1+self.couple_rts)/(self.couple_rts+self.sigma)
        irho = 1/(1+self.couple_rts)
        irs  = 1/(self.couple_rts+self.sigma)
        tf = theta
        tm = 1-theta
        bottom = (tf**(powr) + tm**(powr))**irho 
        
        kf = self.A*(tf**(irs))/bottom
        km = self.A*(tm**(irs))/bottom
        
        assert kf.shape == theta.shape
        assert km.shape == theta.shape
        
        return kf, km
        
        
        
    def check_conflicts(self):
        # this ensures no parameter is defined twice
        sources = self.__parameter_sources__()
        
        for src0 in sources:
            for key in src0:
                for src1 in sources:
                    if src1 is src0: continue
                    if key in src1:
                        print('Warning: {} is defined repeatedly')


    def __parameter_sources__(self):
        # this may be a bit inefficient as it calls dir all the time but 
        names = ['pars','derivative_pars','agrids','female_grid',
                 'male_grid','couples_grid','theta_grid']
        sdir = self.__dir__()
        sources = [getattr(self,name) for name in names if name in sdir]
        return sources
    
    
    def __getitem__(self,key):
        # this allows to call self[key] instead of remembering where 
        # the parameter is hidden
        # make sure this function runs ok it can cause unpredictable behavior
        # if there is something 
        
        
        sources = self.__parameter_sources__()
        
        for source in sources:
            try:
                return source[key]
            except KeyError:
                continue
        raise KeyError('{} is not found anywhere'.format(key))
    
    def __getattr__(self,key):
        # this allows calling self.key instead of self[key]
        try:
            return self.__getitem__(key)
        except:
            raise AttributeError('{} not found'.format(key))
        
    
    
def build_s_grid(agrid,n_between,da_min,da_max):
    sgrid = np.array([0.0],agrid.dtype)
    for j in range(agrid.size-1):
        step = (agrid[j+1] - agrid[j])/n_between
        if step >= da_min and step <= da_max:
            s_add = np.linspace(agrid[j],agrid[j+1],n_between)[:-1]
        elif step < da_min:
            s_add = np.arange(agrid[j],agrid[j+1],da_min)
        elif step > da_max:
            s_add = np.arange(agrid[j],agrid[j+1],da_max)
        sgrid = np.concatenate((sgrid,s_add))
    
    sgrid = np.concatenate((sgrid,np.array([agrid[-1]])))
            
    if sgrid[0] == sgrid[1]: 
        sgrid = sgrid[1:]
        
    return sgrid
        