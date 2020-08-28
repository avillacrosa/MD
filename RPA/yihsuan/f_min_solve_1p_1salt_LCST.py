# RPA+FH model for single charge sequence
# Short-range interaction contributes to only the k=0 FH term
# The FH term ehs parameters follow the definition in the PRL paper
# 
# Code for LCST. Of course, one needs negative ehs[0] and positive ehs[1]
#
# ver Git.1 Apr 14, 2020
# Upload to github

# ver2: Aug 27, 2019:
# - Allow concentration-dependent permittivity
# - Rewrite in terms of dict instead of class; simplify the structure.

# ver0: May 27, 2018

import numpy as np
import scipy.optimize as sco


import global_vars as gv
import thermal_1p_1salt as tt

invT = 1e2

# All functions in this module rely on the HP dict generated by
# thermal_1p_1salt.RPAFH
#====================== Critical point calculation =======================
   

def cri_calc( HP, phis, in2=1e-4):
    #Find a phi where there is a change of phase.
    phi_max = (1-2*phis*gv.r_sal)/(gv.r_res+gv.r_con*HP['pc'])
    # Scan phi values until we find one where there is a change of sign.
    while True:
        ddf2 = cri_u_solve(in2, HP, phis)
        if ddf2 is None:
            if in2>phi_max:
                print("Could not find a phi with Phase Separation")
                return
            else:
                in2 *= 1.25
        else: break
    step = 0.9
    # Then find to neighbour points where there is still a change of sign
    in1 = in2*step
    while True:
        ddf1 = cri_u_solve(in1, HP, phis)
        if ddf1 is None:
            if in1<gv.phi_min_sys:
                print("Could not find in1")
                return
            else:
                step = np.sqrt(step)
                in1 = in2*step
        else: break
    step = 1.1
    in3 = in2*step
    while True:
        ddf3 = cri_u_solve(in3, HP, phis)
        if ddf3 is None:
            if in3>phi_max:
                print("Could not find in3")
                return
            else:
                step = np.sqrt(step)
                in3 = in2*step
        else: break
    print('ddf1={:.4f}, in1={:4f}'.format(ddf1,in1))
    print('ddf2={:.4f}, in1={:4f}'.format(ddf2,in2))
    print('ddf3={:.4f}, in1={:4f}'.format(ddf3,in3))
    # Finally, we can optimize these 3 points so that there is a maximum
    # u has a maximum in LCST (as it is inverse temperature)
    while not((ddf1<ddf2) and (ddf3<ddf2)):
        if ddf1 >= ddf2:
            in1 *= 0.999
            ddf1 = cri_u_solve(in1, HP, phis)
        else: # ddf3 >= ddf2
            ddf4 = cri_u_solve(in3*0.99, HP, phis)
            if ddf4 < ddf3: 
                in3 *= 1.1
                ddf3 = cri_u_solve(in3, HP, phis)
            else:
                in2 += (in3-in2)/2
                ddf2 = cri_u_solve(in2, HP, phis)

    result = sco.brent(cri_u_solve_max, \
                   args = ( HP, phis), \
                   brack= (in1,in2,in3), full_output=1 )
    phicr =  result[0]
    ucr   = -result[1]

    return phicr, ucr


def cri_u_solve( phi, HP, phis ):
    def cri_u_solve( phi, HP, phis ):
    """
    Try to bracket the u more strictly as going too high may lead 
    to a reentrant behaviour.
    """
    umin = 0.01
    if ddf_u (umin,phi, HP, phis)>0:
        print("Problem ddf_u is positive even al low u. Maybe ehs not for LCST?")
        print("phi= {:.4f}, u = {:.4f}".format(phi, umin))
        return
    umax = umin*1.25
    while True:
        if ddf_u(umax,phi, HP, phis)>0:
            break
        else:
            umax *= 1.25
        if umax > 10:
            print("Problem ddf_u is negative even al high u. Maybe ehs not for LCST?")
            print("phi= {:.4f}, u = {:.4f}".format(phi, umax))
            return
    result = sco.brenth(ddf_u, umin, umax, args=(phi, HP, phis ) )
    return result

def cri_u_solve_max( phi, HP, phis ):
    """
    just a change of sign of cri_u_solve_max
    I could have changed sign in the code, but these was simpler
    to follow
    """
    result = cri_u_solve( phi, HP, phis )
    return -result

# Function handle for cri_u_solve
def ddf_u(u, phi, HP, phis ):
    ddf = tt.ddfeng(HP, phi, phis, u)
    #print(u, phi, ddf, flush=True)
    return ddf



#========================= Solve spinodal points =========================
def ps_sp_solve( HP, phis, u, phi_ori ):
    err = gv.phi_min_sys
    phi_max = (1-2*phis*gv.r_sal)/(gv.r_res+gv.r_con*HP['pc'])-err
    sp1 = sco.brenth(ddf_phi, err, phi_ori, args=(phis, u, HP ) )
    sp2 = sco.brenth(ddf_phi, phi_ori, phi_max, args=(phis, u, HP ) )
    return sp1, sp2

# Function handle for ps_sp_solve
def ddf_phi(phi, phis, u, HP ) :
    return tt.ddfeng(HP, phi, phis, u)

#========== Minimize system energy to solve coexistence points ===========
def ps_bi_solve( HP, phis, u, phi_sps, phi_ori=None):
    err = gv.phi_min_sys
    phi_max = (1-2*phis*gv.r_sal)/(gv.r_res+gv.r_con*HP['pc'])-err
    sps1, sps2 = phi_sps
    
    phi_all_ini = [ phi_sps[0]*0.9, phi_sps[1]*1.1]
    if phi_ori == None:
        phi_ori = (sps1+sps2)/2

    fori = tt.feng(HP, phi_ori, phis, u)

    result = sco.minimize( Eng_all, phi_all_ini, \
                           args = (HP, phis, u, phi_ori, fori), \
                           method = 'L-BFGS-B', \
                           jac = J_Eng_all, \
                           bounds = ((err,sps1-err), (sps2+err,phi_max-err)), \
                           options={'ftol':1e-20, 'gtol':1e-20, 'eps':1e-20} )

    bi1 = min(result.x)
    bi2 = max(result.x)
    return bi1, bi2
  

#==================== System energy for minimization =====================
def Eng_all( phi_all, HP, phis, u, phi_ori, f_ori ):
    phi1 = phi_all[0];
    phi2 = phi_all[1];
    v =( phi2 - phi_ori )/( phi2 - phi1 )

    f1 = tt.feng(HP, phi1, phis, u )
    f2 = tt.feng(HP, phi2, phis, u )

    return invT*(v*f1 + (1-v)*f2  - f_ori )

#Jacobian of Eng_all
def J_Eng_all( phi_all, HP, phis, u, phi_ori, f_ori ):

    phi1 = phi_all[0];
    phi2 = phi_all[1];
    v =( phi2 - phi_ori )/( phi2 - phi1 )

    f1 =  tt.feng(HP, phi1, phis, u )
    f2 =  tt.feng(HP, phi2, phis, u )
    df1 = tt.dfeng(HP, phi1, phis, u )
    df2 = tt.dfeng(HP, phi2, phis, u )

    J = np.empty(2)

    J[0] = v*( (f1-f2)/(phi2-phi1) + df1 )
    J[1] = (1-v)*( (f1-f2)/(phi2-phi1) + df2 )

    return invT*J;







