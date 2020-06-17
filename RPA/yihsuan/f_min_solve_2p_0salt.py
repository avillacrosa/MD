# RPA model (no FH) for 2 overall neutral charge sequences   
# No salt. No conterions.
# Constant permittivity 
# solute sizes are not tunable

# ver Git.1 Apr 14, 2020
# Upload to github
# Rewrite data structure: from class to dict
# Rewrtie the code for calculating S(k): from matrix product to linear summation

# ver 2 May 14, 2017

import numpy as np
import scipy.optimize as sco

import thermal_2p_0salt as tt

invT = 1e6

# All functions in this module rely on the HP dict generated by
# thermal_2p_0salt.twoProteins
#========== Minimize system energy to solve coexistence points ===========

# total free energy
def f_total_v( phi_12_a_v, HP, u, phis_ori ): 
    phi1ori = phis_ori[0] 
    phi2ori = phis_ori[1]  

    phi1a = phi_12_a_v[0]
    phi2a = phi_12_a_v[1]
    v     = phi_12_a_v[2]
    phi1b = (phi1ori-v*phi1a)/(1-v)
    phi2b = (phi2ori-v*phi2a)/(1-v)
 
    fa = tt.feng(HP, phi1a, phi2a, u)
    fb = tt.feng(HP, phi1b, phi2b, u)
    
    #print(phi1a ,phi2a, phi1b,phi2b, v)
    return invT*( v*fa + (1-v)*fb ) 

#Jacobian of total free energy
def J_f_total_v( phi_12_a_v, HP, u, phis_ori ):
    phi1ori = phis_ori[0] 
    phi2ori = phis_ori[1]  

    phi1a = phi_12_a_v[0]
    phi2a = phi_12_a_v[1]
    v     = phi_12_a_v[2]
    phi1b = (phi1ori-v*phi1a)/(1-v)
    phi2b = (phi2ori-v*phi2a)/(1-v)
    #print(phi1a ,phi2a, phi1b,phi2b, v)   


    fa = tt.feng(HP, phi1a, phi2a, u)
    fb = tt.feng(HP, phi1b, phi2b, u)
    df1a, df2a = tt.dfeng(HP, phi1a, phi2a, u)
    df1b, df2b = tt.dfeng(HP, phi1b, phi2b, u)
    
    J = np.empty(3)
    J[0] = invT*( v*( df1a - df1b )  )
    J[1] = invT*( v*( df2a - df2b )  )
    J[2] = invT*( fa - fb + (phi1b-phi1a)*df1b + (phi2b-phi2a)*df2b )
    
    return J

# Constraint functions
# 0 < phi1a < 1
# 0 < phi2a < 1
# 0 < phi1b < 1   : v < phi1ori/phi1a && v < (1-phi1ori)/(1-phi1a)
# 0 < phi2b < 1   : v < phi2ori/phi2a && v < (1-phi2ori)/(1-phi2a)
# phi1a+phi2a < 1
# phi1b+phi2b < 1 : v < (1-phi1ori-phi2ori)/(1-phi1a-phi2a)
# 0 < v < 1
def vmin(phi_12_a_v, phis_ori):
    phi1ori = phis_ori[0] 
    phi2ori = phis_ori[1]  

    phi1a = phi_12_a_v[0]
    phi2a = phi_12_a_v[1]
    #v     = phi_12_a_v[2]

    return min(1, phi1ori/phi1a, (1-phi1ori)/(1-phi1a), \
                  phi2ori/phi2a, (1-phi2ori)/(1-phi2a), \
                  (1-phi1ori-phi2ori)/(1-phi1a-phi2a)     )   


def ps_bi_solve( HP, u, phis_ori ,r_vini, useJ ): 
    err = tt.phi_min_sys

    phi1ori = phis_ori[0] 
    phi2ori = phis_ori[1]  

    # for pappu30+pappu1
    #phi_ini = [phi1ori*0.5, phi2ori*1.5 ]
    
    # for pappu28+pappu24
    phi_ini = [phi1ori*0.5, phi2ori*0.5 ]

    vini = [r_vini*vmin(phi_ini, phis_ori)]
    inis = phi_ini + vini
    
    #print(inis)
    cons_all = ( {'type':'ineq', 'fun': lambda x:   x[0]-err }, \
                 {'type':'ineq', 'fun': lambda x: 1-x[0]-err }, \
                 {'type':'ineq', 'fun': lambda x:   x[1]-err }, \
                 {'type':'ineq', 'fun': lambda x: 1-x[1]-err }, \
                 {'type':'ineq', 'fun': lambda x: 1-x[0]-x[1]-err }, \
                 {'type':'ineq', 'fun': lambda x:   x[2]-err }, \
                 {'type':'ineq', 'fun': lambda x: vmin(x,phis_ori)-x[2]-err } \
               )


    if useJ:
        result = sco.minimize( f_total_v, inis,         \
                               args = (H, u, phis_ori), \
                               method = 'SLSQP', \
                               jac = J_f_total_v, \
                               constraints = cons_all, \
                               tol = tt.phi_min_sys   )
                               #options = {'maxiter':300,'ftol':1e-10})
    else:
        result = sco.minimize(  f_total_v, inis,\
                                args = (sig1, sig2, u, phis_ori), \
                                #method = 'trust-constr', \
                                method = 'COBYLA', 
                                constraints = cons_all, \
                                tol = err )
                                #options = {'maxiter':1000, 'catol':0.5*err } )#'ftol':err})   


    #print(result.x)
    phi1a = result.x[0]
    phi2a = result.x[1]
    v     = result.x[2]
    phi1b = (phi1ori-v*phi1a)/(1-v)
    phi2b = (phi2ori-v*phi2a)/(1-v)
 
    return [phi1a, phi2a, phi1b, phi2b]


def bisolve( HP, u, phi_oris ):

    phi1 = phi_oris[0]
    phi2 = phi_oris[1]
    #if ddf_calc(sig1, sig2, u, phi1, phi2 ) > 0 :
    #    return [phi1, phi2, phi1, phi2]

    r_vini1 = 0.5
    phiall = ps_bi_solve( HP, u, phi_oris, r_vini1 , 0 )
    phi_test = np.array(phiall)
    try_max = 20
    try_i = 0
    while np.isnan(sum(phi_test) ) \
          and np.array(np.where(((0<phi_test) & (phi_test<1)))).size != 4 \
          and try_i <= try_max:
        phiall = ps_bi_solve( HP, u, phi_oris, np.random.rand() , 1)
        phi_test = np.array(phiall)
        try_i = try_i+1        
        #print(try_i)
    return phiall

def ddf_calc(HP, phi1, phi2, u):
    return tt.ddfeng(HP, phi1, phi2, u)[3]

