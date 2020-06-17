from math import log, sqrt, pi, exp
import numpy as np
import scipy.integrate as integrate
import seq_list
import scipy.optimize as sco


class fgRPA:
    def __init__(self, phi_m):
        self.T_star = 0.286
        self.eta = 1

        # res_list gives other info also !
        seq_data = seq_list.get_the_charge('Ddx4_N1_m')
        # seq_data = seq_list.get_the_charge('alt_DdX4')
        self.sigmas = seq_data[0]
        self.N = seq_data[1]

        self.phi_m = phi_m
        self.phi_c = phi_m*np.abs(np.sum(self.sigmas))/self.N
        self.phi_s = 0
        # Assuming r_i = 0 for all i
        self.phi_w = 1 - self.phi_m - self.phi_c - self.phi_s

    ################ Entropies and derivatives ##################

    def s(self):
        # Eq. 13
        phi_m = self.phi_m
        phi_w = self.phi_w
        phi_c = self.phi_c
        N = self.N
        s = phi_m*log(phi_m)/N + phi_w*log(phi_w) + phi_c*log(phi_c)
        # s = phi_m*log(phi_m)/N + phi_w*log(phi_w)
        s = -s
        return s

    def ds(self):
        N = self.N
        phi_m = self.phi_m
        r = log(phi_m)/N
        r += 1 / N
        r = -r
        return r

    ################## Gks and derivatives ####################

    def Gk(self, k):
        phi_c = self.phi_c
        phi_m = self.phi_m
        phi_s = self.phi_s
        N = self.N
        Gm = 0
        # Eq. 26
        for i, sigma_i in enumerate(self.sigmas):
            for j, sigma_j in enumerate(self.sigmas):
                gauss = exp(-k**2*abs(i-j)/6)
                Gm += sigma_i * gauss * sigma_j
        # Eq. 40
        G = 4*pi*(2*phi_s+phi_c+phi_m*Gm/N)/(k**2*(1+k**2)*self.T_star)
        return G

    def dGk(self, k):
        phi_c = self.phi_c
        phi_s = self.phi_s
        N = self.N
        Gm = 0
        # Eq. 26
        for i, sigma_i in enumerate(self.sigmas):
            for j, sigma_j in enumerate(self.sigmas):
                gauss = exp(-k**2*abs(i-j)/6)
                Gm += sigma_i * gauss * sigma_j
        # Eq. 40
        G = 4*pi*(2*phi_s+phi_c+Gm/N)/(k**2*(1+k**2)*self.T_star)
        return G

    ##################### Free energies #######################

    def fel(self):
        eta = self.eta

        # Eq. 39
        def integrand(k):
            Gk = self.Gk(k)
            int = (log(1+eta*Gk)/eta-Gk)*k**2/(4*pi**2)
            return int

        integral = integrate.quad(integrand, 0, np.inf, limit=1000)*2
        return integral[0]

    def dfel(self):
        eta = self.eta

        # Eq. 39
        def integrand(k):
            Gk = self.Gk(k)
            dGk = self.dGk(k)
            int = 1/(1+eta*Gk)*(eta*dGk)-dGk
            int *= k**2/(4*pi**2)
            return int

        integral = integrate.quad(integrand, 0, np.inf, limit=1000)*2
        return integral[0]

    def f(self):
    # def f(self, phi_m):
    #     self.phi_m = phi_m
        s = self.s()
        f_int = self.fel()
        return -s + f_int

    def df(self,):
    # def df(self, phi_m):
    #     self.phi_m = phi_m
        ds = self.ds()
        df_int = self.dfel()
        return -ds + df_int

    ###################### Utilities #####################

    def find_binodals(self):
        # Eq 46a
        def cond1(phis):
            phi_a = phis[0]
            phi_b = phis[1]
            df_phi_a = self.df(phi_a)
            df_phi_b = self.df(phi_b)
            eq49a = df_phi_a - df_phi_b
            eq49b = self.f(phi_a) - self.f(phi_b) - phi_a*df_phi_a + phi_b*df_phi_b
            return [eq49a, eq49b]

        res = sco.fsolve(cond1, x0=[0.01, 0.05])
        return res

    def find_spinodals(self):
        pass

    def Gk_sf(self, k, sigma, L, n):
        z = exp(-k ** 2 / (6 * sigma))
        N = self.N
        phi_s = self.phi_s
        phi_c = self.phi_c
        phi_m = self.phi_m

        quo_h = (1 - z ** (L * sigma)) / (1 + z ** (L * sigma))

        g1 = sigma * (1 + z) / (1 - z)
        g1 += 4 * z / (L * (1 - z) ** 2) * quo_h
        g1 *= N
        g2 = 2 * z * quo_h / (1 - z) ** 2 * (1 - (-1) ** n * z ** (N * sigma))
        Gm = g1 + g2
        G = 4 * pi * (2 * phi_s + phi_c + phi_m * Gm / N) / (k ** 2 * (1 + k ** 2) * self.T_star)
        return G