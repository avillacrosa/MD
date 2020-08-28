# import multiprocessing as mp
import pathos.multiprocessing as mp

import numpy as np

import seq_list as sl
import scipy.optimize as sco
import free_f as ff
from numpy import pi


class rgRPA_ps:
    def __init__(self, u, seqname, phiTop, phisTop, phiBot=None, phisBot=None, zc=1, zs=1, **kwargs):
        self.u = u
        self.du = kwargs.get('du', 0.1)
        self.zc = zc
        self.zs = zs

        protein_data = sl.get_the_charge(seqname)
        self.seqname = seqname
        self.sigma = protein_data[0]
        self.N = protein_data[1]
        self.sequence = protein_data[2]

        self.nt = kwargs.get('nt', 100)
        self.ionic_strength = kwargs.get('ionic_strength', 100e-3)

        # Flory parameters !
        self.eh, self.es = 0, 0
        self.phiTop, self.phiBot = phiTop, phiBot
        self.phisTop, self.phisBot = phisTop, phisBot

        self.HP = self.Heteropolymer()

        # Minimization helpers
        self.phi_min_calc = 1e-12
        self.invT = 1e4

    def run_salt(self):
        # pool = mp.Pool(processes=2)
        # phi0, phis0, phia, phisa, phib, phisb, v = zip(*pool.map(self.salt_parallel, self.pp))
        pp = self.get_phis()
        phi0, phis0, phia, phisa, phib, phisb, v = zip(*map(self.salt_parallel, pp))
        output = np.zeros((pp.shape[0], 7))

        output[:, 0] = np.array(phi0)
        output[:, 1] = np.array(phis0)
        output[:, 2] = np.array(phia)
        output[:, 3] = np.array(phisa)
        output[:, 4] = np.array(phib)
        output[:, 5] = np.array(phisb)
        output[:, 6] = np.array(v)

        head = ' u=' + str(self.u) + ' , phiTop=' + str(self.phiTop) + ' , phisTop=' + str(self.phisTop) + \
               ' , phiBot=' + str(self.phiBot) + ' , phisBot=' + str(self.phisBot) + '\n' + \
               '  [phiori, phisori]  [phia, phisa]  [phib, phisb], v \n' + \
               '--------------------------------------------------------------------------------------------'
        fname = 'saltdept_zc' + str(self.HP['zc']) + '_zs' + str(self.HP['zs']) + '_' \
                + self.seqname + '_u' + str(self.u) + '_w2_4.189.txt'
        np.savetxt(fname, output, header=head)
        print(head)
        print(output)
        print(f"Saving to {fname}")

    def Heteropolymer(self, w2=4 * pi / 3, wd=1 / 6, FH_funs=None):
        sigma = self.sigma
        zc = self.zc
        zs = self.zs

        sig = np.array(sigma)
        N = sig.shape[0]
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
        eh, es = self.eh, self.es
        if FH_funs is None:
            HP['FH'] = lambda phi, u: (w2 / 2 - eh * u - es) * phi * phi
            HP['dFH'] = lambda phi, u: (w2 - 2 * eh * u - 2 * es) * phi
            HP['ddFH'] = lambda phi, u: (w2 - 2 * eh * u - 2 * es)
        else:
            HP['FH'] = lambda phi, u: FH_fun[0](phi, u) + w2 / 2 * phi * phi
            HP['dFH'] = lambda phi, u: FH_fun[1](phi, u) + w2 * phi
            HP['ddFH'] = lambda phi, u: FH_fun[2](phi, u) + w2
        return HP

    def salt_parallel(self, p):
        try:
            x = self.bisolve(p)
            print("PX", p, x)
            return p[0], p[1], x[0], x[1], x[2], x[3], x[4]
        except:
            for test in range(5):
                p[0] = p[0] * (0.95 + 0.1 * np.random.rand())
                try:
                    x = self.bisolve(p)
                    print("PX", p, x)
                    return p[0], p[1], x[0], x[1], x[2], x[3], x[4]
                except:
                    pass
            return p[0], p[1], -1, -1, -1, -1, -1

    def bisolve(self, phiPS_ori):
        r_vini1 = 0.5
        phiall = self.ps_bi_solve(phiPS_ori, r_vini1, 1)
        phi_test = np.array(phiall)
        try_max = 20
        try_i = 0
        while np.isnan(sum(phi_test)) \
                and np.array(np.where(((0 < phi_test) & (phi_test < 1)))).size != 4 \
                and try_i <= try_max:
            phiall = self.ps_bi_solve(phiPS_ori, np.random.rand(), 1)
            phi_test = np.array(phiall)
            try_i = try_i + 1
        return phiall

    def ps_bi_solve(self, phiPS_ori, r_vini, useJ):
        # Constraint functions
        # 0 < phia < 1
        # 0 < phisa < 1
        # 0 < phib < 1   : v < phiori/phia   && v < (1-phiori)/(1-phia)
        # 0 < phisb < 1  : v < phisori/phisa && v < (1-phisori)/(1-phisa)
        # phia+phisa < 1
        # phib+phisb < 1 : v < (1-phiori-phisori)/(1-phia-phisa)
        # 0 < v < 1

        def vmin(phiPS_a_v, phiPS_ori):
            phiori = phiPS_ori[0]
            phisori = phiPS_ori[1]

            phia = phiPS_a_v[0]
            phisa = phiPS_a_v[1]
            # v     = phi_12_a_v[2]

            return min(1, phiori / phia, (1 - phiori) / (1 - phia),
                       phisori / phisa, (1 - phisori) / (1 - phisa),
                       (1 - phiori - phisori) / (1 - phia - phisa))

        err = self.phi_min_calc

        phiori = phiPS_ori[0]
        phisori = phiPS_ori[1]

        f0 = ff.f_eng(self.HP, phiori, phisori, self.u)

        phi_ini = [phiori * 0.5, phisori * 0.95]

        vini = [r_vini * vmin(phi_ini, phiPS_ori)]
        inis = phi_ini + vini

        cons_all = ({'type': 'ineq', 'fun': lambda x: x[0] - err},
                    {'type': 'ineq', 'fun': lambda x: 1 - x[0] - err},
                    {'type': 'ineq', 'fun': lambda x: x[1] - err},
                    {'type': 'ineq', 'fun': lambda x: 1 - x[1] - err},
                    {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1] - err},
                    {'type': 'ineq', 'fun': lambda x: x[2] - err},
                    {'type': 'ineq', 'fun': lambda x: vmin(x, phiPS_ori) - x[2] - err})

        if useJ:
            result = sco.minimize(self.f_total_v, inis,
                                  args=(phiPS_ori, f0),
                                  method='SLSQP',
                                  jac=self.J_f_total_v,
                                  constraints=cons_all,
                                  tol=err / 100,
                                  options={'ftol': err, 'eps': err})
        else:
            result = sco.minimize(self.f_total_v, inis,
                                  args=(phiPS_ori, f0),
                                  method='COBYLA',
                                  constraints=cons_all,
                                  tol=err / 100)

        phia = result.x[0]
        phisa = result.x[1]
        v = result.x[2]
        phib = (phiori - v * phia) / (1 - v)
        phisb = (phisori - v * phisa) / (1 - v)
        if phia > phib:
            t1, t2 = phib, phisb
            phib, phisb = phia, phisa
            phia, phisa = t1, t2
            v = 1 - v

        return [phia, phisa, phib, phisb, v]

    def get_phis(self):
        """
        Converts the phiTop and phiBot to a linspace with a small displacement which I don't know why.
        Same for the salt case
        Then collapses both linspaces to a (nt, 2) array where each element is the phi of salt and
        monomer for a given range.
        :return:
        """
        nt = self.nt
        phiTop = self.phiTop
        phisTop = self.phisTop
        phiBot = self.phiBot
        phisBot = self.phisBot
        if phiBot is not None and phisBot is not None:
            phiall = np.linspace(phiBot + (phiTop - phiBot) * 0.001, phiTop - (phiTop - phiBot) * 0.001, nt)
            phisall = np.linspace(phisBot + (phisTop - phisBot) * 0.001, phisTop - (phisTop - phisBot) * 0.001, nt)
        else:
            phiall = np.linspace(0.1, phiTop, nt)
            phisall = np.linspace(phisTop * 0.01, phisTop * 0.999, nt)
        pp = np.array([[phiall[i], phisall[i]] for i in range(nt)])
        return pp

    def f_total_v(self, phiPS_a_v, phiPS_ori, f0):
        HP = self.HP
        u = self.u

        phiori = phiPS_ori[0]
        phisori = phiPS_ori[1]

        phia = phiPS_a_v[0]
        phisa = phiPS_a_v[1]
        v = phiPS_a_v[2]
        phib = (phiori - v * phia) / (1 - v)
        phisb = (phisori - v * phisa) / (1 - v)

        fa = ff.f_eng(HP, phia, phisa, u)
        fb = ff.f_eng(HP, phib, phisb, u)

        # print(phia ,phisa, phib,phisb, v)
        return self.invT * (v * fa + (1 - v) * fb - f0)

    def J_f_total_v(self, phiPS_a_v, phiPS_ori, f0=0):
        HP = self.HP
        u = self.u

        phiori = phiPS_ori[0]
        phisori = phiPS_ori[1]

        phia = phiPS_a_v[0]
        phisa = phiPS_a_v[1]
        v = phiPS_a_v[2]
        phib = (phiori - v * phia) / (1 - v)
        phisb = (phisori - v * phisa) / (1 - v)

        xeffa = ff.x_eff(HP, phia, phisa, u)
        fa = ff.f_eng(HP, phia, phisa, u, x=xeffa)
        dfa, dfsa = ff.df_eng(HP, phia, phisa, u, x=xeffa)

        xeffb = ff.x_eff(HP, phib, phisb, u)
        fb = ff.f_eng(HP, phib, phisb, u, x=xeffb)
        dfb, dfsb = ff.df_eng(HP, phib, phisb, u, x=xeffb)

        J = np.empty(3)
        J[0] = v * (dfa - dfb)
        J[1] = v * (dfsa - dfsb)
        J[2] = fa - fb + (phib - phia) * dfb + (phisb - phisa) * dfsb

        return self.invT * J