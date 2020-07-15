#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:45:56 2020

@author: egorkozlov
"""

import jax.numpy as jnp
from jax import jit
import numpy as onp
from gridvec import VecOnGrid
np = jnp


from timeit import default_timer as dt
#def jit(f): return f
 
from utils import compare_arrays

def iteration_couples(model,t,Vnext_list,MUnext_list):
    s = model.setup
    sig = s.sigma
    bet = s.beta
    R = s.R
    # compute EV
    
    ti = t if t < model.T else model.T-1
    
    
    
    VCnext, VFnext, VMnext, VFsingle, VMsingle = Vnext_list
    
    MUnext, MUnext_sf, MUnext_sm = MUnext_list
    
    EVC, EVF, EVM, EMU, i_div = \
    naive_divorce(model,Vnext_list,MUnext_list,s.zfzmpsi_mat[ti].T)
    
    '''
    EVC, EVF, EVM, EMU  = [dot(x,s.zfzmpsi_mat[ti].T) \
                           for x in (VCnext,VFnext,VMnext,MUnext)]
    
    '''
    EV = (EVC, EVF, EVM)
    
    
    
    agrid = s.agrid_c
    
    
    i, wn, wt, sgrid = \
    s.v_sgrid_c.i, s.v_sgrid_c.wnext, s.v_sgrid_c.wthis, s.v_sgrid_c.val

    wf = np.exp(s.zfzmpsi[ti][:,0])
    wm = np.exp(s.zfzmpsi[ti][:,1])
    psi = s.zfzmpsi[ti][:,2][None,:,None]
    
    
    li = wf[None,:] + wm[None,:]
    money = R*s.agrid_c[:,None] + li
    
    umult = s.u_mult(s.theta_grid_coarse)
    kf, km = s.c_mult(s.theta_grid_coarse)
    
    
    
    t0 = dt()
    #print(EV.shape)
    #solver = jit(solve,static_argnums=[2,3,4,5,6,7,8])
    V_vfi, VF_vfi, VM_vfi, s_vfi = \
                    solve_vfi(money,EV,umult,kf,km,sig,bet,i,wn,wt,sgrid,psi)    
    
    print('vfi time: {}'.format(dt() - t0))
    t0 = dt()
    
    
    
    #segm = jit(solve_egm,static_argnums=(3,4,5,6,7,8,9,10,11,12,13,14))
    segm = solve_egm
    
    last = (t==model.T)
    V, VF, VM, s, MU = \
                segm(EV,EMU,li,umult,kf,km,agrid,sig,bet,R,i,wn,wt,psi,last) 
                
    
    print('egm time: {}'.format(dt() - t0))
                   
    #assert np.abs(s_vfi.max() - s.max()) < 1         
    desc = ['V', 'VF','VM','s']
    v0 = (V, VF, VM, s)
    v1 = (V_vfi, VF_vfi, VM_vfi, s_vfi)
    
    for d, a0, a1 in zip(desc,v0,v1):
        name = 'egm vs vfi, {}'.format(d)
        compare_arrays(a0,a1,name)
    
    
    return (V,VF,VM), MU, s

from ue import upper_envelope_matrix, upper_envelope, upper_envelope_vmap
    
def solve_egm(EV_list,EMU,li,umult,kf,km,agrid,sigma,beta,R,i,wn,wt,psi,last):
    
    if not last:
        c_implied = (beta*R*EMU / umult[None,None,:])**(-1/sigma) 
        m_implied = c_implied + agrid[:,None,None]
        a_implied = (1/R)*(m_implied - li[:,:,None])
        bEV = beta*EV_list[0]
        
        dm = np.diff(m_implied,axis=0)
        
        if not np.all(dm>0):
            
            print('upper envelope required')    
            
            shp = (a_implied.shape[0],a_implied.size // a_implied.shape[0])
           
            a_implied_r = a_implied.reshape(shp)
            c_implied_r = c_implied.reshape(shp)
            bEV_r = bEV.reshape(shp)
            li_r = np.broadcast_to(li.squeeze()[:,None],a_implied.shape[1:]).reshape(shp[-1])
            um_r = np.broadcast_to(umult[None,:],a_implied.shape[1:]).reshape(shp[-1])
            
            
            # split
            
            
            #print(inds)
            #s = jit(upper_envelope_matrix,static_argnums=(3,5,6,7))
            #c_r, V_r = s(bEV_r,a_implied_r,c_implied_r,agrid,li_r,um_r,R,sigma)
            
            c_r, V_r = upper_envelope_vmap(bEV_r,a_implied_r,c_implied_r,agrid,li_r,um_r,R,sigma)
            
            #assert np.allclose(V_r2,V_r)
            #assert np.allclose(c_r2,c_r)
            
            s_r = li_r + R*agrid[:,None] - c_r
            
            
            j_r, wn_r = interp(agrid,s_r) # correspodning indices
            
            
            c, s, V, j, wn = [x.reshape(a_implied.shape)
                                    for x in (c_r, s_r, V_r, j_r, wn_r)]
        
        else:
            
            # simple interpolation
            print('no upper envelope required')
            
            a_i_min = a_implied[0,...]
            a_i_max = a_implied[-1,...]            
            j_egm, wn_egm = interp_manygrids(a_implied,agrid,axis=0,trim=True)
            s_egm = agrid[j_egm]*(1-wn_egm) + agrid[j_egm+1]*wn_egm
            
            i_above = (agrid[:,None,None] >= a_i_max)
            i_below = (agrid[:,None,None] <= a_i_min)
            i_egm = (~i_above) & (~i_below)
            
            
            s = s_egm*i_egm + agrid[-1]*i_above
            c = R*agrid[:,None,None] + li[:,:,None] - s
            V = None # this V is not needed 
            
            # yeah this runs things again, compare with j_egm
            j, wn = interp(agrid,s)
            assert np.all(j[i_egm]==j_egm[i_egm])
            
            
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
    
    EV_int, EVF_int, EVM_int = EV_int_list
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
    
    xs_r = xs.reshape(
            (1,)*(axis-1) + (xs.size,) + (1,)*(grids.ndim - 1 - axis)
                     )
    
    wnext = (xs_r - grid_j)/(grid_jp - grid_j)
    return j, (wnext if return_wnext else 1-wnext)




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
    
    
    EVs, EVFs, EVMs = EV_stretch_list
    
    mega_matrix = utility + beta*EVs[None,:,:,:] 
    #print(mega_matrix.shape)
    
    ind_s = mega_matrix.argmax(axis=1)
    V = np.take_along_axis(mega_matrix,ind_s[:,None,:,:],1)\
                                                .squeeze(axis=1) + psi
                                                
                                                
    s = sgrid[ind_s]
    c = money[:,:,None] - s
    
    V_check = umult[None,None,:]*(c**(1-sigma)/(1-sigma)) + \
                            psi + beta*np.take_along_axis(EVs,ind_s,0)
                
    VF = ((kf[None,None,:]*c)**(1-sigma)/(1-sigma)) + \
                            psi + beta*np.take_along_axis(EVFs,ind_s,0)
    VM = ((km[None,None,:]*c)**(1-sigma)/(1-sigma)) + \
                            psi + beta*np.take_along_axis(EVMs,ind_s,0)
                
                
    assert np.allclose(V_check,V,atol=1e-5)
    
    return V, VF, VM, s


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
    
    VF_div = (wt_f[:,None]*VFsingle[i_f,:] + wn_f[:,None]*VFsingle[i_f+1,:])[:,izf][:,:,None]
    VM_div = (wt_m[:,None]*VMsingle[i_m,:] + wn_m[:,None]*VMsingle[i_m+1,:])[:,izm][:,:,None]
    
    MUF_div = (wt_f[:,None]*(0.5*MUnext_sf)[i_f,:] + wn_f[:,None]*(0.5*MUnext_sf)[i_f+1,:])[:,izf][:,:,None]
    MUM_div = (wt_m[:,None]*(0.5*MUnext_sm)[i_m,:] + wn_m[:,None]*(0.5*MUnext_sm)[i_m+1,:])[:,izm][:,:,None]
    
    i_stay = (VFnext >= VF_div - div_costs) & (VMnext >= VM_div - div_costs)
    i_div = ~i_stay
    
    t = s.theta_grid_coarse[None,None,:]
    VC_div = t*VF_div + (1-t)*VM_div
    MU_div = t*MUF_div + (1-t)*MUM_div
    
    VC, VF, VM, MU = [i_stay*x + (i_div)*y for x,y in
                                  zip((VCnext,VFnext,VMnext,MUnext),
                                      (VC_div,VF_div,VM_div,MU_div))]
    
    dot = jit(dot_3d)
    
    EVC, EVF, EVM, EMU = [dot(x,M) for x in [VC, VF, VM, MU]]
    return EVC, EVF, EVM, EMU, i_div
    
    

def dot_3d(V,M):
    # this computes array Q, such that
    # Q[:,:,i] = V[:,:,i]*M for each i in V.shape[2]
    
    q_shape = (V.shape[0],M.shape[1],V.shape[2])
    Q = np.zeros(q_shape[:-1]+(0,),dtype=V.dtype)
    
    for k in range(V.shape[-1]):
        Q = np.concatenate((Q,np.dot(V[:,:,k],M)[:,:,None]),axis=2)
    return Q
