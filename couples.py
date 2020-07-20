#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:45:56 2020

@author: egorkozlov
"""

import jax.numpy as jnp
from jax import jit, vmap
import numpy as onp
from gridvec import VecOnGrid
np = jnp
from jax.nn import sigmoid as logit


from timeit import default_timer as dt
taste_shock = 0.05
#def jit(f): return f
 
from utils import compare_arrays, logit_expectation_conditional_above

def iteration_couples(model,t,Vnext_list,MUnext_list):
    s = model.setup
    sig = s.sigma
    bet = s.beta
    R = s.R
    # compute EV
    
    ti = t if t < model.T else model.T-1
    
    
    
    VCnext, VFnext, VMnext, VFsingle, VMsingle = Vnext_list
    
    MUnext, MUnext_sf, MUnext_sm = MUnext_list
    
    EVCR, EVC, EVF, EVM, EMU = \
    naive_divorce(model,Vnext_list,MUnext_list,s.zfzmpsi_mat[ti].T)
    
    '''
    EVC, EVF, EVM, EMU  = [dot(x,s.zfzmpsi_mat[ti].T) \
                           for x in (VCnext,VFnext,VMnext,MUnext)]
    
    '''
    EV = (EVCR, EVC, EVF, EVM)
    
    
    
    agrid = s.agrid_c
    
    
    i, wn, wt, sgrid = \
    s.v_sgrid_c.i, s.v_sgrid_c.wnext, s.v_sgrid_c.wthis, s.v_sgrid_c.val

    wf = np.exp(s.zfzmpsi[ti][:,0])
    wm = np.exp(s.zfzmpsi[ti][:,1])
    psi = s.zfzmpsi[ti][:,2][None,:,None]
    
    
    li = wf[None,:] + wm[None,:]
    
    umult = s.u_mult(s.theta_grid_coarse)
    kf, km = s.c_mult(s.theta_grid_coarse)
    
    

    t0 = dt()
    #segm = jit(solve_egm,static_argnums=(3,4,5,6,7,8,9,10,11,12,13,14))
    segm = solve_egm
    
    last = (t==model.T)
    V, VF, VM, s, MU = \
                segm(EV,EMU,li,umult,kf,km,agrid,sig,bet,R,i,wn,wt,psi,last) 
                
    
    print('egm time: {}'.format(dt() - t0))
                   
    return (V,VF,VM), MU, s

from ue import upper_envelope_matrix
    


def solve_egm(EV_list,EMU,li,umult,kf,km,agrid,sigma,beta,R,i,wn,wt,psi,last):
    
    if not last:
        c_implied = (beta*R*EMU / umult[None,None,:])**(-1/sigma) 
        m_implied = c_implied + agrid[:,None,None]
        a_implied = (1/R)*(m_implied - li[:,:,None])
        bEV = beta*EV_list[0]
        
        
        
        shp = (a_implied.shape[0],a_implied.size // a_implied.shape[0])
           
        a_implied_r = a_implied.reshape(shp)
        c_implied_r = c_implied.reshape(shp)
        m_implied_r = m_implied.reshape(shp)
        bEV_r = bEV.reshape(shp)
        li_r = np.broadcast_to(li.squeeze()[:,None],a_implied.shape[1:]).reshape(shp[-1])
        um_r = np.broadcast_to(umult[None,:],a_implied.shape[1:]).reshape(shp[-1])
        
        
        
        dm = np.diff(m_implied_r,axis=0)
        
        i_where = np.any(dm<=0,axis=0)
        ind = np.where(i_where)[0]
        
        
        # this inwokes arbitrary size dimensionality...
        
        
        # first do everything as if the grid is monotonic
        # the fix few entities where it is not
        # simple interpolation
        #print('no upper envelope required: simple algorithm')
        
        a_i_min = a_implied_r.min(axis=0,keepdims=True)
        a_i_max = a_implied_r.max(axis=0,keepdims=True)           
        j_egm, wn_egm = interp_manygrids(a_implied_r,agrid,axis=0,trim=True)
        s_egm = agrid[j_egm]*(1-wn_egm) + agrid[j_egm+1]*wn_egm
        i_above = (agrid[:,None] >= a_i_max)
        i_below = (agrid[:,None] <= a_i_min)
        i_egm = (~i_above) & (~i_below)            
        
        s_r = s_egm*i_egm + agrid[-1]*i_above
        c_r = R*agrid[:,None] + li_r - s_r
        
        # yeah this runs things again, compare with j_egm
        j_r, wn_r = interp(agrid,s_r)
        #assert np.all(j_r[i_egm]==j_egm[i_egm])
            
        # this makes the function un-jittable
        if np.any(dm<=0):
            print('upper envelope required: invoking the algorithm')    
            print('{:.2f} % cases non-monotonic'.format(100*np.mean(i_where)))
            
            bEV_r_ue, a_implied_r_ue, c_implied_r_ue,  = \
                [x[:,ind] for x in (bEV_r,a_implied_r,c_implied_r)]
            li_r_ue, um_r_ue = [x[ind] for x in (li_r,um_r)]
            
            
            c_r_ue, V_r_ue = upper_envelope_matrix(bEV_r_ue,a_implied_r_ue,c_implied_r_ue,agrid,li_r_ue,um_r_ue,R,sigma)
            #assert np.allclose(V_r2,V_r)
            #assert np.allclose(c_r2,c_r)
            
            s_r_ue = li_r_ue + R*agrid[:,None] - c_r_ue
            j_r_ue, wn_r_ue = interp(agrid,s_r_ue) # correspodning indices
            
            c_r, s_r, j_r, wn_r = [x.at[:,ind].set(y) for x,y
                                    in zip((c_r, s_r, j_r, wn_r),
                                         (c_r_ue, s_r_ue, j_r_ue, wn_r_ue))]
        
        
        c, s, j, wn = [x.reshape(a_implied.shape)
                                for x in (c_r, s_r, j_r, wn_r)]
        
        
        EV_int_list = [(np.take_along_axis(x,j,axis=0)*(1-wn) + \
                        np.take_along_axis(x,j+1,axis=0)*(wn))
                            for x in EV_list]
           
    else:
        
        c = R*agrid[:,None,None] + li[:,:,None]
        s = 0.0*c
        EV_int_list = EV_list # it is 0 anyways
        
        
    
    MUc = umult[None,None,:]*(c**(-sigma))
    
    # this EV should be interpolated as well
    # 
    
    EVR_int, EV_int, EVF_int, EVM_int = EV_int_list
    V = umult[None,None,:]*(c**(1-sigma)/(1-sigma)) + psi + beta*EV_int
    VF = ((kf[None,None,:]*c)**(1-sigma)/(1-sigma)) + psi + beta*EVF_int
    VM = ((km[None,None,:]*c)**(1-sigma)/(1-sigma)) + psi + beta*EVM_int
    return V, VF, VM, s, MUc




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
    
    
    #assert np.all(np.diff(grids,axis=axis) > 0)
    
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
    
    xs_r = xs.reshape(
            (1,)*(axis-1) + (xs.size,) + (1,)*(grids.ndim - 1 - axis)
                     )
    #assert False
    wnext = (xs_r - grid_j)/(grid_jp - grid_j)
    #assert wnext.shape == j.shape
    return j, (wnext if return_wnext else 1-wnext)
        

def naive_divorce(model,Vlist,MUlist,M,div_costs=0.0):
    # this performs integration with a naive divorce
    
    s = model.setup
    
    VCnext, VFnext, VMnext, VFsingle, VMsingle = Vlist
    
    MUnext, MUnext_sf, MUnext_sm = MUlist
    
    
    assets_divorce_fem = 0.5*s.agrid_c
    assets_divorce_mal = 0.5*s.agrid_c
    
    v_a_fem = VecOnGrid(s.agrid_s,assets_divorce_fem)
    i_f, wn_f, wt_f = v_a_fem.i, v_a_fem.wnext, v_a_fem.wthis
    
    v_a_mal = VecOnGrid(s.agrid_s,assets_divorce_mal)
    i_m, wn_m, wt_m = v_a_mal.i, v_a_mal.wnext, v_a_mal.wthis
    
    ie, izf, izm, ipsi = s.all_indices()
    
    VF_div = (wt_f[:,None]*VFsingle[i_f,:] + wn_f[:,None]*VFsingle[i_f+1,:])[:,izf][:,:,None] - div_costs
    VM_div = (wt_m[:,None]*VMsingle[i_m,:] + wn_m[:,None]*VMsingle[i_m+1,:])[:,izm][:,:,None] - div_costs
    
    MUF_div = (wt_f[:,None]*(0.5*MUnext_sf)[i_f,:] + wn_f[:,None]*(0.5*MUnext_sf)[i_f+1,:])[:,izf][:,:,None]
    MUM_div = (wt_m[:,None]*(0.5*MUnext_sm)[i_m,:] + wn_m[:,None]*(0.5*MUnext_sm)[i_m+1,:])[:,izm][:,:,None]
    
    t = s.theta_grid_coarse[None,None,:]
    VC_div = t*VF_div + (1-t)*VM_div
    MU_div = t*MUF_div + (1-t)*MUM_div

    
    # get transition probability over theta
    trans_mat_theta, r_factor, e_shock = ren_divorce_mat(VFnext,VMnext,VF_div,VM_div,taste_shock,s.theta_grid_coarse)
    p_stay = trans_mat_theta.sum(axis=3)
    p_divorce = 1.0-p_stay
    
    # transition for divorce
    ee = lambda q : np.einsum('ijk,ijlk->ijl',q,trans_mat_theta)
    
    ee_r = lambda q : np.einsum('ijk,ijlk->ijl',q,r_factor*trans_mat_theta)
    # adds rescaling 
    
    es = e_shock[:,:,None]
    
    VCR = ee_r(VCnext) + p_divorce*(VC_div+es)
    
    VC, VF, VM, MU = [ee(x) + p_divorce*y for x,y in
                                  zip((VCnext,VFnext,VMnext,MUnext),
                                      (VC_div+es,VF_div+es,VM_div+es,MU_div))]
    
    # transition over exogenous state
    dd = lambda x : np.einsum('ijk,jl->ilk',x,M)
    
    EVCR, EVC, EVF, EVM, EMU = [dd(x) for x in [VCR, VC, VF, VM, MU]]
    
    #assert np.all(np.diff(EVC,axis=0)>=0)
    
    return EVCR, EVC, EVF, EVM, EMU




def solve_vfi(money,EV_list,umult,kf,km,sigma,beta,i,wn,wt,sgrid,psi):
    
    #ts = 0.01
    # dim is (na,ns,nexo,ntheta)
    
    
    
    EV_stretch_list = [(wt[:,None,None]*x[i,:,:] + \
                              wn[:,None,None]*x[i+1,:,:]) for x in EV_list]
    
    
    
    consumption = money[:,None,:] - sgrid[None,:,None]
    consumption_negative = (consumption <= 0)
    uc = (np.maximum(consumption,1e-8))**(1-sigma)/(1-sigma)
    utility = umult[None,None,None,:]*(uc[:,:,:,None]) - \
                                1e9*consumption_negative[:,:,:,None]
    
    
    EVRs, EVs, EVFs, EVMs = EV_stretch_list
    
    mega_matrix = utility + beta*EVRs[None,:,:,:] 
    #print(mega_matrix.shape)
    
    ind_s = mega_matrix.argmax(axis=1)
    #V0 = np.take_along_axis(mega_matrix,ind_s[:,None,:,:],1)\
    #                                            .squeeze(axis=1) + psi
                                                
    # this is spoiled V: it has participation constraints things
                                          
    s = sgrid[ind_s]
    c = money[:,:,None] - s
    
    
    V = umult[None,None,:]*(c**(1-sigma)/(1-sigma)) + \
                            psi + beta*np.take_along_axis(EVs,ind_s,0)
                
    VF = ((kf[None,None,:]*c)**(1-sigma)/(1-sigma)) + \
                            psi + beta*np.take_along_axis(EVFs,ind_s,0)
    VM = ((km[None,None,:]*c)**(1-sigma)/(1-sigma)) + \
                            psi + beta*np.take_along_axis(EVMs,ind_s,0)
                
                
    #assert np.allclose(V_check,V,atol=1e-5)
    
    return V, VF, VM, s



jit_rdm = lambda f : jit(f,static_argnums=[4,5])
@jit_rdm
def ren_divorce_mat(vf_in,vm_in,vf_out,vm_out,ts,thetagrid):    
    
    # for each [ia,ie] this generates transition matrix over possible future
    # values of theta (itp) given the current one (it). The indexing is
    # [ia,ie,it,itp]. The transition matrix does not sum to 1 --- there is a
    # probability the the couple divorces. Instead, it sums to 1-p_divorce 
    # over the last dimension.
    
    
    
    def L(x): return logit(x/ts)
    
    na, ne, nt = vf_in.shape
    
    M = np.minimum(vf_in - vf_out, vm_in - vm_out)
    i_ebs =  np.argmax(M,axis=2)
    M_ebs = np.max(M,axis=2)
    M_dext = np.concatenate((-np.inf*np.ones((na,ne,1)),M),axis=2)
    M_uext = np.concatenate((M,-np.inf*np.ones((na,ne,1))),axis=2)
    L_up = np.maximum( L(M_dext[:,:,1:]) - L(M_dext[:,:,:-1]), 0.0)
    L_up_cs = np.cumsum(L_up,axis=2)
    L_down = np.maximum( L(M_uext[:,:,:-1]) - L(M_uext[:,:,1:]), 0.0)
    L_down_cs = np.cumsum(L_down[:,:,::-1],axis=2)[:,:,::-1] # inverse order cumcum
    
    p_divorce = (1 - L(M.max(axis=2)))[:,:,None]
    #p_stay = np.sum(L_up)
    #assert np.allclose(p_stay,np.sum(L_down))
    
    ir = np.broadcast_to(np.arange(nt)[None,None,:],M.shape)    
    
    it_c = ir[:,:,:,None]
    it_r = ir[:,:,None,:]
    iebs_c = i_ebs[:,:,None,None]
    #iebs_r = i_ebs*np.ones((nt,1),dtype=np.int16)
    
    i_irtoebs = (it_r <= iebs_c) & (it_r > it_c)
    i_ebstoir = (it_r >= iebs_c) & (it_r < it_c)
    i_diagbeloweq = (it_r <= iebs_c) & (it_r == it_c)
    i_diagabove = (it_r > iebs_c) & (it_r == it_c)
    
    Mout  = i_irtoebs*L_up[:,:,None,:] + i_ebstoir*L_down[:,:,None,:] + \
            i_diagbeloweq*L_up_cs[:,:,None,:] + \
            i_diagabove*L_down_cs[:,:,None,:]
    
    Mout = (1-p_divorce[...,None])*(Mout / np.maximum(Mout.sum(axis=3,keepdims=True),1e-8)) # numerical fix...
    

    # here we build the rescale factor corresponding the the grid
    # this allows to compute correct value function for couples problem:
    # we should not allow decision weights to fall
    tt = thetagrid
    ttc = tt[:,None]
    ttr = tt[None,:]
    factor = np.maximum(ttc/ttr,(1-ttc)/(1-ttr))[None,None,:,:]
    
    # finally we need expected value of the shock for divorce
    E_shock = logit_expectation_conditional_above(ts,M_ebs)
    
    return Mout, factor, E_shock

