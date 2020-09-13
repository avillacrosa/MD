# import multiprocessing as mp
import pathos.multiprocessing as mp

import numpy as np
import sys
import time

import seq_list as sl
import scipy.optimize as sco
import freef_fg as ff
from numpy import pi


class fgRPA_lcst:
    def __init__(self, seqname, ehs, phis_mM, find_cri, umin, pH=7, zc=1, zs=1, epsfun=False, eps_modify_ehs=False, parallel=False,
                 **kwargs):
        self.du = kwargs.get('du', 0.1)
        self.zc = zc
        self.zs = zs
        self.epsfun = epsfun
        self.eps_modify_ehs = eps_modify_ehs
        self.ehs = ehs
        self.umin = umin
        self.parallel = parallel
        self.mimics = kwargs.get("mimics", None)

        protein_data = sl.get_the_charge(seqname, pH=pH)
        self.seqname = seqname
        self.sigma = protein_data[0]
        self.N = protein_data[1]
        self.name = kwargs.get('name',None)
        self.cri_only = kwargs.get('cri_only', False)
        self.sequence = protein_data[2]

        self.nt = kwargs.get('nt', 100)
        self.ionic_strength = kwargs.get('ionic_strength', 100e-3)

        # Flory parameters !
        self.phis_mM = phis_mM
        self.phis = phis_mM * 0.001 / (1000. / 18.)

        self.HP = self.Heteropolymer()
        self.find_cri = find_cri

        # Minimization helpers
        self.phi_min_calc = 1e-12
        self.invT = 1e4

        self.r_res = 1
        self.r_con = 1
        self.r_sal = 1
        self.eta = 1

        self.phi_min_sys = 1e-12

    def run(self):
        # use  phi-dependent permittivity or not
        ehs = self.ehs
        seqname = self.seqname
        sequence = self.sequence
        N = self.N
        phis = self.phis
        umin = self.umin

        print('Seq:', seqname, '=', sequence, '\nphi_s=', phis,
              'r_res =', self.r_res, ', r_con =', self.r_con,
              ', r_sal =', self.r_sal, ', eta =', self.eta,
              '\nehs :', ehs[0], ehs[1])

        t0 = time.time()

        # critical_point
        phi_cri, u_cri = self.cri_calc()
        if self.cri_only:
            sp_file = '/home/adria/perdiux/prod/lammps/final/RPA/fg_ucst/c_only_' + self.name + '.txt'
            print(f"Saving criticial point only at {sp_file}")
            with open(sp_file, 'a+') as fin:
                fin.write(f"{phi_cri} {u_cri} {self.mimics} \n")
            return

        print('Critical point found in', time.time() - t0, 's')
        print('u_cri =', '{:.4e}'.format(u_cri),
              'T*_cri = {:.4f}'.format(1 / u_cri),
              ', phi_cri =', '{:.8e}'.format(phi_cri))
        if umin is None:
            sys.exit()
        # ============================ Set up u range =============================
        # ddu = du / 10
        # umax = (np.ceil(u_cri / ddu) - 1) * ddu
        # uclose = (np.ceil(u_cri / du) - 2) * du
        #
        # if umin > u_cri:
        #     umin = u_cri / 1.5
        # if uclose < umin:
        #     uclose = umin

        umax = u_cri * 1.5
        hel = np.linspace(u_cri, umax, 10)
        du = hel[1] - hel[0]
        ddu = du / 10
        umin = (np.floor(u_cri / ddu) + 1) * ddu
        uclose = (np.floor(u_cri / du) + 2) * du

        uall = np.append(np.arange(umax, uclose, -ddu), np.arange(uclose, umin - du, -du))

        print(uall, flush=True)

        # ==================== Parallel calculate multiple u's ====================

        def bisp_parallel(u):
            sp1, sp2 = self.ps_sp_solve(u, phi_cri)
            print(u, sp1, sp2, 'sp done!', flush=True)
            bi1, bi2 = self.ps_bi_solve(u, [sp1, sp2], phi_cri)
            print(u, bi1, bi2, 'bi done!', flush=True)

            return sp1, sp2, bi1, bi2

        # Sequential, for testing:
        # sp1ss = []
        # sp2ss = []
        # bi1ss = []
        # bi2ss = []
        # for  u_i in uall:
        #    print("u_i=", u_i)
        #    result = bisp_parallel(u_i)
        #    sp1ss.append(result[0])
        #    sp2ss.append(result[1])
        #    bi1ss.append(result[2])
        #    bi2ss.append(result[3])

        if self.parallel:
            pool = mp.Pool(processes=4)
            sp1ss, sp2ss, bi1ss, bi2ss = zip(*pool.map(bisp_parallel, uall))
        else:
            sp1ss, sp2ss, bi1ss, bi2ss = zip(*map(bisp_parallel, uall))

        ind_slc = np.where(np.array(bi1ss) > self.phi_min_sys)[0]
        unew = uall[ind_slc]
        sp1s, sp2s = np.array(sp1ss)[ind_slc], np.array(sp2ss)[ind_slc]
        bi1s, bi2s = np.array(bi1ss)[ind_slc], np.array(bi2ss)[ind_slc]
        new_umax = np.max(unew)
        nnew = ind_slc.shape[0]

        sp_out, bi_out = np.zeros((2 * nnew + 1, 2)), np.zeros((2 * nnew + 1, 2))
        sp_out[:, 0] = np.append(np.append(sp1s[::-1], phi_cri), sp2s)
        sp_out[:, 1] = np.append(np.append(unew[::-1], u_cri), unew)
        bi_out[:, 0] = np.append(np.append(bi1s[::-1], phi_cri), bi2s)
        bi_out[:, 1] = sp_out[:, 1]

        print(sp_out)
        print(bi_out)

        monosize = str(self.r_res) + '_' + str(self.r_con) + '_' + str(self.r_sal)
        ehs_str = '_'.join(str(x) for x in self.ehs)

        calc_info = '_RPAFH_N{}_phis_{:.5f}_{}_eh{:.2f}_es{:.2f}_umax{:.2f}_du{:.2f}_ddu{:.2f}.txt'.format(
            N, phis, seqname, ehs[0], ehs[1], new_umax, du, ddu)

        if self.name is None:
            sp_file = '/home/adria/perdiux/prod/lammps/final/RPA/rg_ucst/sp' + calc_info
            bi_file = '/home/adria/perdiux/prod/lammps/final/RPA/rg_ucst/bi' + calc_info
        else:
            sp_file = '/home/adria/perdiux/prod/lammps/final/RPA/rg_ucst/sp_' + self.name + '.txt'
            bi_file = '/home/adria/perdiux/prod/lammps/final/RPA/rg_ucst/bi_' + self.name + '.txt'

        print(sp_file)
        print(bi_file)

        cri_info = "u_cri= " + str(u_cri) + " , phi_cri= " + str(phi_cri)

        np.savetxt(sp_file, sp_out, fmt='%.8e', header=cri_info)
        np.savetxt(bi_file, bi_out, fmt='%.8e', header=cri_info)

    def Heteropolymer(self, eps_a=18.931087269965023, eps_b=84.51003476887941):
        # sequence parameters
        sig = self.sigma
        N = self.N  # sequence length
        pc = np.abs(np.sum(sig)) / N  # prefactor for counterions
        Q = np.sum(sig * sig) / N  # fraction of charged residues (sig=+/-1)

        # linear summation for S(k)
        mel = np.kron(sig, sig).reshape((N, N))
        Tel = np.array([np.sum(mel.diagonal(n) + mel.diagonal(-n)) for n in range(N)])
        Tel[0] /= 2
        L = np.arange(N)

        HP = {'sig': sig,
              'N': N,
              'pc': pc,
              'Q': Q,
              'L': L,
              'Tel': Tel,
              'ehs': self.ehs
              }

        if self.epsfun:
            a, b = eps_a, eps_b
            HP['eps0'] = b
            flinear = lambda x: a * x + b * (1 - x)
            HP['epsx'] = lambda x: b / flinear(x)
            HP['depsx'] = lambda x: -b * (a - b) / (flinear(x)) ** 2
            HP['ddepsx'] = lambda x: 2 * b * (a - b) * (a - b) / (flinear(x)) ** 3
        else:
            HP['eps0'] = 1
            HP['epsx'] = lambda x: 1 * (x == x)
            HP['depsx'] = lambda x: 0 * x
            HP['ddepsx'] = lambda x: 0 * x

            # using eps_r=eps0 to rescale ehs (assuming the input ehs is of eps_r=1)
        if self.eps_modify_ehs:
            self.ehs[0] *= HP['eps0']

        return HP

    def cri_calc(self, in1=1e-4, in2=1e-3, in3=1e-2):
        # Find a phi where there is a change of phase.
        HP = self.HP
        phi_max = (1 - 2 * self.phis * self.r_sal) / (self.r_res + self.r_con * HP['pc'])
        # Scan phi values until we find one where there is a change of sign.
        while True:
            ddf2 = self.cri_u_solve(in2)
            if ddf2 is None:
                if in2 > phi_max:
                    print("Could not find a phi with Phase Separation")
                    return
                else:
                    in2 *= 1.25
            else:
                break
        step = 0.9
        # Then find to neighbour points where there is still a change of sign
        in1 = in2 * step
        while True:
            ddf1 = self.cri_u_solve(in1)
            if ddf1 is None:
                if in1 < self.phi_min_sys:
                    print("Could not find in1")
                    return
                else:
                    step = np.sqrt(step)
                    in1 = in2 * step
            else:
                break
        step = 1.1
        in3 = in2 * step
        while True:
            ddf3 = self.cri_u_solve(in3)
            if ddf3 is None:
                if in3 > phi_max:
                    print("Could not find in3")
                    return
                else:
                    step = np.sqrt(step)
                    in3 = in2 * step
            else:
                break
        print('ddf1={:.4f}, in1={:4f}'.format(ddf1, in1))
        print('ddf2={:.4f}, in1={:4f}'.format(ddf2, in2))
        print('ddf3={:.4f}, in1={:4f}'.format(ddf3, in3))
        # Finally, we can optimize these 3 points so that there is a maximum
        # u has a maximum in LCST (as it is inverse temperature)
        while not ((ddf1 < ddf2) and (ddf3 < ddf2)):
            if ddf1 >= ddf2:
                in1 *= 0.999
                ddf1 = self.cri_u_solve(in1)
            else:  # ddf3 >= ddf2
                ddf4 = self.cri_u_solve(in3 * 0.99)
                if ddf4 < ddf3:
                    in3 *= 1.1
                    ddf3 = self.cri_u_solve(in3)
                else:
                    in2 += (in3 - in2) / 2
                    ddf2 = self.cri_u_solve(in2)

        result = sco.brent(self.cri_u_solve_max, brack=(in1, in2, in3), full_output=True)
        phicr = result[0]
        ucr = -result[1]

        return phicr, ucr

    def cri_u_solve(self, phi):
        """
        Try to bracket the u more strictly as going too high may lead
        to a reentrant behaviour.
        """

        umin = 0.01
        if self.ddf_u(umin, phi) > 0:
            print("Problem ddf_u is positive even al low u. Maybe ehs not for LCST?")
            print("phi= {:.4f}, u = {:.4f}".format(phi, umin))
            return
        umax = umin * 1.25
        while True:
            if self.ddf_u(umax, phi) > 0:
                break
            else:
                umax *= 1.25
            if umax > 10:
                print("Problem ddf_u is negative even al high u. Maybe ehs not for LCST?")
                print("phi= {:.4f}, u = {:.4f}".format(phi, umax))
                return
        result = sco.brenth(self.ddf_u, umin, umax, args=phi)
        return result

    def cri_u_solve_max(self, phi,):
        """
        just a change of sign of cri_u_solve_max
        I could have changed sign in the code, but these was simpler
        to follow
        """
        result = self.cri_u_solve(phi)
        return -result

    def ddf_u(self, u, phi):
        HP = self.HP
        phis = self.phis
        ddf = ff.ddfeng(HP, phi, phis, u)
        # print(u, phi, ddf, flush=True)
        return ddf

    def ps_sp_solve(self, u, phi_ori):
        err = self.phi_min_sys
        HP = self.HP
        phis = self.phis
        phi_max = (1 - 2 * phis * self.r_sal) / (self.r_res + self.r_con * HP['pc']) - err
        sp1 = sco.brenth(self.ddf_phi, err, phi_ori, args=u)
        sp2 = sco.brenth(self.ddf_phi, phi_ori, phi_max, args=u)
        return sp1, sp2

    def ddf_phi(self, phi, u):
        HP = self.HP
        phis = self.phis
        return ff.ddfeng(HP, phi, phis, u)

    def ps_bi_solve(self, u, phi_sps, phi_ori=None):
        HP = self.HP
        err = self.phi_min_sys
        phis = self.phis

        phi_max = (1 - 2 * self.phis * self.r_sal) / (self.r_res + self.r_con * HP['pc']) - err
        sps1, sps2 = phi_sps

        phi_all_ini = [phi_sps[0] * 0.9, phi_sps[1] * 1.1]
        if phi_ori == None:
            phi_ori = (sps1 + sps2) / 2

        fori = ff.feng(HP, phi_ori, phis, u)

        result = sco.minimize(self.Eng_all, phi_all_ini,
                              args=(u, phi_ori, fori),
                              method='L-BFGS-B',
                              jac=self.J_Eng_all,
                              bounds=((err, sps1 - err), (sps2 + err, phi_max - err)),
                              options={'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20})

        bi1 = min(result.x)
        bi2 = max(result.x)
        return bi1, bi2

    def Eng_all(self, phi_all, u, phi_ori, f_ori):
        HP = self.HP
        phis = self.phis
        phi1 = phi_all[0]
        phi2 = phi_all[1]
        v = (phi2 - phi_ori) / (phi2 - phi1)

        f1 = ff.feng(HP, phi1, phis, u)
        f2 = ff.feng(HP, phi2, phis, u)

        return self.invT * (v * f1 + (1 - v) * f2 - f_ori)

    def J_Eng_all(self, phi_all, u, phi_ori, f_ori):
        HP = self.HP
        phis = self.phis
        phi1 = phi_all[0]
        phi2 = phi_all[1]
        v = (phi2 - phi_ori) / (phi2 - phi1)

        f1 = ff.feng(HP, phi1, phis, u)
        f2 = ff.feng(HP, phi2, phis, u)
        df1 = ff.dfeng(HP, phi1, phis, u)
        df2 = ff.dfeng(HP, phi2, phis, u)

        J = np.empty(2)

        J[0] = v * ((f1 - f2) / (phi2 - phi1) + df1)
        J[1] = (1 - v) * ((f1 - f2) / (phi2 - phi1) + df2)

        return self.invT * J


if __name__ == '__main__':
    seqname = sys.argv[1]
    find_cri = sys.argv[2]
    if find_cri == 'True':
        print("NONNING")
        umin = None
    else:
        umin = float(find_cri)
    phis = float(sys.argv[3])
    eh = float(sys.argv[4])
    es = float(sys.argv[5])
    fgRPA_lcst(seqname=seqname, find_cri=find_cri, phis_mM=phis, ehs=[eh, es], umin=umin, parallel=True).run()
