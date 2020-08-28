import numpy as np
from math import pi
import seq_list as sl
import scipy.optimize as sco
import freef_rg as ff
import pathos.multiprocessing as mp
import time
import sys


class rgRPA_ucst:
    def __init__(self, seqname, find_cri, phis, eh, es, umax, fgRPA=False, pH=None, parallel=False, **kwargs):
        self.zc = kwargs.get('zc', 1)
        self.zs = kwargs.get('zs', 1)
        self.find_cri = find_cri
        self.cri_pre_calc = kwargs.get('cri_pre_calc', False)
        self.phi_min_calc = ff.phi_min_calc
        self.invT = 1e2
        self.wtwo_ratio = kwargs.get('wtwo_ratio', None)
        self.parallel = parallel
        self.umax = umax

        protein_data = sl.get_the_charge(seqname, pH=pH)
        self.seqname = seqname
        self.sigma = protein_data[0]
        self.N = protein_data[1]
        self.sequence = protein_data[2]

        self.phis = float(phis)
        self.eh = float(eh)
        self.es = float(es)
        self.fgRPA = fgRPA

        self.HP = self.Heteropolymer()

    def run(self):
        HP = self.HP
        seqname = self.seqname  # 'Ddx4_N1'
        phis = self.phis
        ff.eh = self.eh
        ff.es = self.es
        umax = self.umax

        duD = int(2)
        du = 10 ** (-duD)

        try:
            pHvalue = float(sys.argv[6])
            sig, N, the_seq = sl.get_the_charge(seqname, pH=pHvalue)
        except:
            sig, N, the_seq = sl.get_the_charge(seqname)

        if self.fgRPA:
            ff.useleff = False

        # ----------------------- Calculate critical point -----------------------

        print('Seq:', seqname, '=', the_seq)

        print('w2=', HP['w2'])
        print('phis=', phis)
        print('eh=' + str(ff.eh) + ' , es=' + str(ff.es))

        t0 = time.time()
        phi_cri, u_cri = self.cri_calc()
        print('Critical point found in', time.time() - t0, 's')

        print('u_cri =', '{:.8e}'.format(u_cri), ', phi_cri =', '{:.8e}'.format(phi_cri))
        if umax is None:
            sys.exit()

        # ---------------------------- Set up u range ----------------------------
        ddu = du / 10
        umin = (np.floor(u_cri / ddu) + 1) * ddu
        uclose = (np.floor(u_cri / du) + 2) * du
        if umax < u_cri:
            umax = np.floor(u_cri * 1.5)
        if uclose > umax:
            uclose = umax
        uall = np.append(np.arange(umin, uclose, ddu), np.arange(uclose, umax + du, du))

        # -------------------- Parallel calculate multiple u's -------------------

        def bisp_parallel(u):
            sp1, sp2 = self.ps_sp_solve(u, phi_cri)
            print(u, sp1, sp2, 'sp done!', flush=True)
            bi1, bi2 = self.ps_bi_solve(u, (sp1, sp2), phi_cri)
            print(u, bi1, bi2, 'bi done!', flush=True)

            return sp1, sp2, bi1, bi2
        if self.parallel:
            pool = mp.Pool(processes=4)
            sp1, sp2, bi1, bi2 = zip(*pool.map(bisp_parallel, uall))
        else:
            sp1, sp2, bi1, bi2 = zip(*map(bisp_parallel, uall))

        # ---------------------- Prepare for output ----------------------

        ind = np.where(np.array(bi1) > self.phi_min_calc)[0]
        nout = ind.shape[0]

        sp1t = np.array([sp1[i] for i in ind])
        sp2t = np.array([sp2[i] for i in ind])
        bi1t = np.array([bi1[i] for i in ind])
        bi2t = np.array([bi2[i] for i in ind])
        ut = np.array([round(uu, duD + 1) for uu in uall[ind]])
        new_umax = np.max(ut)

        sp_out = np.zeros((2 * nout + 1, 2))
        bi_out = np.zeros((2 * nout + 1, 2))

        sp_out[:, 1] = np.concatenate((ut[::-1], [u_cri], ut))
        sp_out[:, 0] = np.concatenate((sp1t[::-1], [phi_cri], sp2t))
        bi_out[:, 1] = sp_out[:, 1]
        bi_out[:, 0] = np.concatenate((bi1t[::-1], [phi_cri], bi2t))

        print(sp_out)
        print(bi_out)

        calc_info = seqname + '_N' + str(N) + '_TwoFields' + \
                    '_phis' + str(phis) + \
                    '_w2r' + str(self.wtwo_ratio) + \
                    '_eh' + str(ff.eh) + '_es' + str(ff.es) + \
                    '_umax' + str(round(new_umax, duD)) + \
                    '_du' + str(du) + '_ddu' + str(ddu) + \
                    '.txt'

        sp_file = '../results/sp_' + calc_info
        bi_file = '../results/bi_' + calc_info

        print(sp_file)
        print(bi_file)

        cri_info = "u_cri= " + str(u_cri) + " , phi_cri= " + str(phi_cri) + \
                   "\n------------------------------------------------------------"

        np.savetxt(sp_file, sp_out, fmt='%.10e', header=cri_info)
        np.savetxt(bi_file, bi_out, fmt='%.10e', header=cri_info)

    def Heteropolymer(self, w2=4 * pi / 3, wd=1 / 6, FH_funs=None):
        if self.fgRPA:
            w2 = 0
        elif self.wtwo_ratio:
            w2 *= self.wtwo_ratio

        sig = self.sigma
        N = self.N
        zc = self.zc
        zs = self.zs
        eh = self.eh
        es = self.es
        pc = np.abs(np.sum(sig) / N)
        Q = np.sum(sig * sig) / N
        IN = np.eye(N)

        mel = np.kron(sig, sig).reshape((N, N))
        Tel = np.array([np.sum(mel.diagonal(n) + mel.diagonal(-n)) for n in range(N)])
        Tel[0] /= 2
        Tex = 2 * np.arange(N, 0, -1)
        Tex[0] /= 2
        mlx = np.kron(sig, np.ones(N)).reshape((N, N))
        Tlx = np.array([np.sum(mlx.diagonal(n) + mlx.diagonal(-n)) for n in range(N)])
        Tlx[0] /= 2

        L = np.arange(N)
        L2 = L * L

        HP = {'sig': sig,
              'zs': zs,
              'zc': zc,
              'w2': w2,
              'wd': wd,
              'N': N,
              'pc': pc,
              'Q': Q,
              'IN': IN,
              'L': L,
              'L2': L2,
              'Tel': Tel,
              'Tex': Tex,
              'Tlx': Tlx
              }

        # Default is a Flory-Huggins model
        if FH_funs is None:
            HP['FH'] = lambda phi, u: (w2 / 2 - eh * u - es) * phi * phi
            HP['dFH'] = lambda phi, u: (w2 - 2 * eh * u - 2 * es) * phi
            HP['ddFH'] = lambda phi, u: (w2 - 2 * eh * u - 2 * es)
        else:
            HP['FH'] = lambda phi, u: FH_fun[0](phi, u) + w2 / 2 * phi * phi
            HP['dFH'] = lambda phi, u: FH_fun[1](phi, u) + w2 * phi
            HP['ddFH'] = lambda phi, u: FH_fun[2](phi, u) + w2

        return HP

    def cri_calc(self, ini1=1e-4, ini3=1e-1, ini2=2e-1):
        HP = self.HP
        phis = self.phis
        phi_max = (1 - 2 * self.phis) / (1 + HP['pc'])
        # ini1, ini3, ini2 = 1e-6, 1e-2, phi_max*2/3

        if self.cri_pre_calc:

            u1 = self.cri_u_solve(ini1, phis)
            u2 = self.cri_u_solve(ini2, phis)
            u3 = self.cri_u_solve(ini3, phis)

            while min(u1, u2, u3) != u3:
                if u1 >= u2:
                    ini3 = (ini2 + ini3) / 2
                else:
                    ini3 = (ini1 + ini3) / 2

            u3 = self.cri_u_solve(ini3, 0)
        result = sco.brent(self.cri_u_solve, args=(phis,), brack=(ini1, ini3, ini2), full_output=True)

        phicr, ucr = result[0], result[1]

        return phicr, ucr

    def cri_u_solve(self, phi, phis):
        return sco.brenth(self.ddf_u, 0.0001, 1000, args=(phi, phis))

    # Function handle for cri_u_solve
    def ddf_u(self, u, phi, phis):
        ddf = ff.ddf_eng(self.HP, phi, phis, u)[0]
        print(phi, u, ddf, flush=True)
        return ddf

        # --------------------- Solve salt-free spinodal points ---------------------

    def ps_sp_solve(self, u, phi_ori):
        HP = self.HP
        err = self.phi_min_calc
        phis = self.phis
        phi_max = (1 - 2 * self.phis) / (1 + HP['pc']) - err

        phi1 = sco.brenth(self.ddf_phi, err, phi_ori, args=u)
        phi2 = sco.brenth(self.ddf_phi, phi_ori, phi_max, args=u)
        return phi1, phi2

    def ddf_phi(self, phi, u):
        HP = self.HP
        phis = self.phis

        return ff.ddf_eng(HP, phi, phis, u)[0]

    # -------------------- Solve salt-free coexistence curve --------------------
    def ps_bi_solve(self, u, phi_sps, phi_ori=None):
        err = self.phi_min_calc
        phis = self.phis
        HP = self.HP
        phi_max = (1 - 2 * phis) / (1 + HP['pc']) - err
        sps1, sps2 = phi_sps

        phi_all_ini = [sps1 * 0.9, sps2 * 1.1]
        if phi_ori is None:
            phi_ori = (sps1 + sps2) / 2

        f_ori = ff.f_eng(HP, phi_ori, 0, u)

        result = sco.minimize(self.Eng_all, phi_all_ini,
                              args=(u, phi_ori, f_ori),
                              method='L-BFGS-B',
                              jac=self.J_Eng_all,
                              bounds=((err, sps1 - err), (sps2 + err, phi_max)),
                              options={'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20})
        bi1 = min(result.x)
        bi2 = max(result.x)
        return bi1, bi2

    # -------------------- System energy for minimization --------------------
    def Eng_all(self, phi_all, u, phi0, f0):
        HP = self.HP
        phis = self.phis
        invT = self.invT

        phi1 = phi_all[0]
        phi2 = phi_all[1]
        v = (phi2 - phi0) / (phi2 - phi1)

        f1 = ff.f_eng(HP, phi1, phis, u)
        f2 = ff.f_eng(HP, phi2, phis, u)

        fall = invT * (v * f1 + (1 - v) * f2 - f0)

        return fall

    def J_Eng_all(self, phi_all, u, phi0, f0):
        HP = self.HP
        phis = self.phis
        invT = self.invT

        phi1 = phi_all[0]
        phi2 = phi_all[1]
        v = (phi2 - phi0) / (phi2 - phi1)

        xeff1 = ff.x_eff(HP, phi1, phis, u) if ff.useleff else 1
        f1 = ff.f_eng(HP, phi1, phis, u, x=xeff1)
        df1 = ff.df_eng(HP, phi1, phis, u, x=xeff1)[0]

        xeff2 = ff.x_eff(HP, phi2, phis, u) if ff.useleff else 1
        f2 = ff.f_eng(HP, phi2, phis, u, x=xeff2)
        df2 = ff.df_eng(HP, phi2, phis, u, x=xeff2)[0]

        J = np.empty(2)

        J[0] = v * ((f1 - f2) / (phi2 - phi1) + df1)
        J[1] = (1 - v) * ((f1 - f2) / (phi2 - phi1) + df2)

        return invT * J


if __name__ == '__main__':
    seqname = sys.argv[1]
    find_cri = sys.argv[2]
    if find_cri == 'True':
        umax = None
    else:
        umax = float(find_cri)
    phis = float(sys.argv[3])
    eh = float(sys.argv[4])
    es = float(sys.argv[5])
    rgRPA_ucst(seqname=seqname, find_cri=find_cri, phis=phis, eh=eh, es=es, umax=umax, parallel=True).run()
