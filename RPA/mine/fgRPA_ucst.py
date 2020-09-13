# import multiprocessing as mp
import pathos.multiprocessing as mp

import numpy as np
import sys
import time

import seq_list as sl
import scipy.optimize as sco
import freef_fg as ff
from numpy import pi


class fgRPA_ucst:
    def __init__(self, seqname, ehs, phis_mM, find_cri, pH=7, zc=1, zs=1, eps0=80, epsfun=False, eps_modify_ehs=False,
                 parallel=False, **kwargs):
        self.du = kwargs.get('du', 0.05)
        self.zc = zc
        self.zs = zs
        self.eps0 = eps0
        self.epsfun = epsfun
        self.eps_modify_ehs = eps_modify_ehs
        self.ehs = ehs
        self.parallel = parallel
        self.name = kwargs.get('name',None)
        self.cri_only = kwargs.get('cri_only', False)
        self.mimics = kwargs.get("mimics", None)

        protein_data = sl.get_the_charge(seqname, pH=pH, mimics=self.mimics)
        self.seqname = seqname
        self.sigma = protein_data[0]
        self.N = protein_data[1]
        self.sequence = protein_data[2]

        self.nt = kwargs.get('nt', 100)
        self.ionic_strength = kwargs.get('ionic_strength', 100e-3)

        # Flory parameters !
        self.phis_mM = phis_mM
        self.phis = phis_mM * 0.001 / (1000. / 18.)

        self.HP = self.Heteropolymer()
        self.find_cri = find_cri

        # Minimization helpers
        self.phi_min_calc = 0
        self.invT = 1e4

        self.r_res = 1
        self.r_con = 1
        self.r_sal = 1
        self.eta = 1

        self.phi_min_sys = 1e-12

    def run(self):
        # ehs must be a 2-element list: the first for entropy and the latter enthalpy
        ehs = self.ehs
        seq_name = self.seqname
        the_seq = self.sequence
        du = self.du
        phis = self.phis
        # ======================= Calculate critical point ========================

        print('Seq:', seq_name, '=', the_seq, '\nphi_s=', phis,
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

        # ============================ Set up u range =============================

        umax = u_cri * 1.5
        hel = np.linspace(u_cri,umax,10)
        du = hel[1]-hel[0]
        ddu = du / 10
        umin = (np.floor(u_cri / ddu) + 1) * ddu
        uclose = (np.floor(u_cri / du) + 2) * du

        print(du, umax)
        if uclose > umax:
            uclose = umax

        uall = np.append(np.arange(umin, uclose, ddu),
                         np.arange(uclose, umax + du, du))

        print(uall, flush=True)

        # ==================== Parallel calculate multiple u's ====================

        def bisp_parallel(u):
            sp1, sp2 = self.ps_sp_solve(u, phi_cri)
            # print(u, sp1, sp2, 'sp done!', flush=True)
            bi1, bi2 = self.ps_bi_solve(u, [sp1, sp2], phi_cri)
            # print(u, bi1, bi2, 'bi done!', flush=True)

            return sp1, sp2, bi1, bi2

        if self.parallel:
            pool = mp.Pool(processes=4)
            sp1ss, sp2ss, bi1ss, bi2ss = zip(*pool.map(bisp_parallel, uall))
        else:
            sp1ss, sp2ss, bi1ss, bi2ss = zip(*map(bisp_parallel, uall))

        ind_slc = np.where(np.array(bi1ss) > -10000000)[0]
        # ind_slc = np.where(np.array(bi1ss) > 1e12)[0]
        unew = uall[ind_slc]
        sp1s, sp2s = np.array(sp1ss)[ind_slc], np.array(sp2ss)[ind_slc]
        bi1s, bi2s = np.array(bi1ss)[ind_slc], np.array(bi2ss)[ind_slc]
        nnew = ind_slc.shape[0]

        sp_out, bi_out = np.zeros((2 * nnew + 1, 2)), np.zeros((2 * nnew + 1, 2))
        sp_out[:, 0] = np.append(np.append(sp1s[::-1], phi_cri), sp2s)
        sp_out[:, 1] = np.append(np.append(unew[::-1], u_cri), unew)
        bi_out[:, 0] = np.append(np.append(bi1s[::-1], phi_cri), bi2s)
        bi_out[:, 1] = sp_out[:, 1]

        print(bi_out)
        print(sp_out)

        monosize = str(self.r_res) + '_' + str(self.r_con) + '_' + str(self.r_sal)
        ehs_str = '_'.join(str(x) for x in ehs)

        # calc_info = '_RPAFH_N{}_phis_{:.5f}_{}_eh{:.2f}_es{:.2f}_umax{:.2f}_du{:.2f}_ddu{:.2f}.txt'.format(
        #     N, phis, seq_name, ehs[0], ehs[1], new_umax, du, ddu)
        calc_info = '_RPAFH_phis_{:.5f}_{}_eh{:.2f}_es{:.2f}.txt'.format(
            phis, seq_name, ehs[0], ehs[1])
        if self.name is None:
            sp_file = '/home/adria/perdiux/prod/lammps/final/RPA/fg_ucst/sp' + calc_info
            bi_file = '/home/adria/perdiux/prod/lammps/final/RPA/fg_ucst/bi' + calc_info
        else:
            sp_file = '/home/adria/perdiux/prod/lammps/final/RPA/fg_ucst/sp_' + self.name + '.txt'
            bi_file = '/home/adria/perdiux/prod/lammps/final/RPA/fg_ucst/bi_' + self.name + '.txt'

        print(sp_file)
        print(bi_file)

        cri_info = "u_cri= " + str(u_cri) + " , phi_cri= " + str(phi_cri)

        np.savetxt(sp_file, sp_out, fmt='%.8e', header=cri_info)
        np.savetxt(bi_file, bi_out, fmt='%.8e', header=cri_info)

        return uall, sp1ss, sp2ss, bi1ss, bi2ss

    def Heteropolymer(self, eps_a=18.931087269965023, eps_b=84.51003476887941):
        # epsfun=False   : constant permittivity
        # epsfun=True    : linear phi-dependent permittivity, eps_a, eps_b are used
        # eps_modify_ehs : Ture if using eps_r=eps0 to rescale ehs
        #                  (assuming the input ehs is of eps_r=1)
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
            HP['eps0'] = self.eps0
            HP['epsx'] = lambda x: self.eps0 * (x == x)
            HP['depsx'] = lambda x: 0 * x
            HP['ddepsx'] = lambda x: 0 * x

            # using eps_r=eps0 to rescale ehs (assuming the input ehs is of eps_r=1)
        if self.eps_modify_ehs:
            self.ehs[0] *= HP['eps0']

        return HP

    def cri_calc(self, in1=1e-4, in2=1e-3, in3=1e-2):
        HP = self.HP
        phis = self.phis
        phi_max = (1 - 2 * phis * self.r_sal) / (self.r_res + self.r_con * HP['pc'])
        # in1, in2, in3 = 1e-4, 1e-2,  phi_max*1/5
        ddf1 = self.cri_u_solve(in1)
        ddf2 = self.cri_u_solve(in2)
        ddf3 = self.cri_u_solve(in3)
        while not ((ddf1 > ddf2) and (ddf3 > ddf2)):
            if ddf1 <= ddf2:
                in1 /= 2
                ddf1 = self.cri_u_solve(in1)
            else:  # ddf3 <= ddf2
                ddf4 = self.cri_u_solve(in3 * 0.999)
                if ddf4 > ddf3:
                    in3 *= 1.1
                    ddf3 = self.cri_u_solve(in3)
                else:
                    in2 += (in3 - in2) / 2
                    ddf2 = self.cri_u_solve(in2)

        result = sco.brent(self.cri_u_solve, brack=(in1, in2, in3), full_output=True)
        phicr = result[0]
        ucr = result[1]

        return phicr, ucr

    def cri_u_solve(self, phi):
        result = sco.brenth(self.ddf_u, 0.01, 5000, args=phi)
        return result

    def ddf_u(self, u, phi):
        ddf = ff.ddfeng(self.HP, phi, self.phis, u)
        # print(phi, u, ddf, flush=True)
        return ddf

    # ========================= Solve spinodal points =========================
    def ps_sp_solve(self, u, phi_ori):
        HP = self.HP
        phis = self.phis
        err = self.phi_min_sys
        phi_max = (1 - 2 * phis * self.r_sal) / (self.r_res + self.r_con * HP['pc']) - err

        sp1 = sco.brenth(self.ddf_phi, err, phi_ori, args=u)
        sp2 = sco.brenth(self.ddf_phi, phi_ori, phi_max, args=u)
        return sp1, sp2

    # Function handle for ps_sp_solve
    def ddf_phi(self, phi, u):
        return ff.ddfeng(self.HP, phi, self.phis, u)

    # ========== Minimize system energy to solve coexistence points ===========
    def ps_bi_solve(self, u, phi_sps, phi_ori=None):
        HP = self.HP
        err = self.phi_min_sys
        phis = self.phis
        phi_max = (1 - 2 * phis * self.r_sal) / (self.r_res + self.r_con * HP['pc']) - err
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

    # ==================== System energy for minimization =====================
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
        phis = self.phis
        HP = self.HP

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
        umax = None
    else:
        umax = float(find_cri)
    phis = float(sys.argv[3])
    eh = float(sys.argv[4])
    es = float(sys.argv[5])
    fgRPA_ucst(seqname=seqname, find_cri=find_cri, phis_mM=phis, ehs=[eh, es], umax=umax, parallel=True).run()