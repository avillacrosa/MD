from math import log, sqrt, pi, exp
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as sco
import seq_list


class rgRPA:
    def __init__(self, phi_m):
        self.T_star = 0.286
        self.eta = 1

        self.lb = 1
        self.l = 1
        self.debye = 0.1

        self.zs = 1
        self.zc = 1
        self.v2 = 1

        # res_list gives other info also !
        seq_data = seq_list.get_the_charge('Ddx4_N1_m')
        # seq_data = seq_list.get_the_charge('alt_DdX4')
        self.sigmas = seq_data[0]
        self.N = seq_data[1]

        self.phi_m = phi_m
        self.phi_c = phi_m * np.abs(np.sum(self.sigmas)) / self.N
        self.phi_s = 0
        # Assuming r_i = 0 for all i
        self.phi_w = 1 - self.phi_m - self.phi_c - self.phi_s

    def s(self):
        phi_m = self.phi_m
        phi_s = self.phi_s
        phi_c = self.phi_c
        phi_w = 1 - phi_m - phi_s - phi_c
        # s = phi_m/self.N*log(phi_m) + phi_s*log(phi_s) + phi_c*log(phi_c) + phi_w*log(phi_w)
        s = phi_m/self.N*log(phi_m) + phi_c*log(phi_c) + phi_w*log(phi_w)
        s = -s
        return s

    def fp(self):
        l = self.l
        x = self.get_x()
        phi_m = self.phi_m

        def integrand(k):
            ntg = 1 + phi_m*((self.xi(k, x)/self.nu(k))+self.v2*self.g(k,x))
            ntg += self.v2/self.nu(k)*phi_m**2*(self.xi(k, x)*self.g(k, x)-self.zeta(k, x)**2)
            return ntg

        int_res = integrate.quad(integrand, 0, np.inf, limit=100)

        result = l**3/2/(2*pi)**3*int_res[0]
        return result

    def fion(self):
        k = self.debye
        l = self.l
        f = log(1 + k*l)-k*l+0.5*(k*l)**2
        f /= -4*pi
        return f

    def f0(self):
        phi_m = self.phi_m
        v2 = self.v2
        f = 0.5*v2*phi_m**2
        return f

    def f(self, phi_m):
        self.phi_m = phi_m
        s = self.s()
        f_ion = self.fion()
        f_p = self.fp()
        f_0 = self.f0()
        return -s + f_p + f_ion + f_0

    def get_x(self):
        phi_m = self.phi_m
        N = self.N
        l = self.l

        def delta(k, x):
            one_four = (self.nu(k) + phi_m*self.xi(k,x)) * (1/self.v2 + phi_m*self.g(k, x))
            two_three = (phi_m * self.xi(k,x))**2
            return one_four - two_three

        def int_xi(k,x):
            xi1 = self.xi(k,x,hat=True) / self.v2
            xi2 = self.nu(k)*self.g(k,x, hat=True)
            xi3 = phi_m*(self.xi(k,x,hat=True) * self.g(k,x) + self.xi(k,x) * self.g(k,x, hat=True) - 2*self.xi(k,x)*self.xi(k,x,hat=True))
            return xi1 + xi2 + xi3

        def integrand(k, x):
            return k**2*int_xi(k, x)/(2*pi)**3/delta(k,x)

        def eq_min(x):
            integral = integrate.quad(integrand, 0, np.inf, limit=100, args=x)[0]*2
            integral *= N*l**2/(18*(N-1))
            return 1 - 1/x - integral

        min = sco.fsolve(eq_min,x0=1)
        return min[0]

    def nu(self, k):
        zs = self.zs
        zc = self.zc
        phi_s = self.phi_s
        phi_c = self.phi_c
        nu1 = k**2/(4*pi*self.lb)
        nu2 = zs**2*phi_s + zc**2*phi_c
        return nu1 + nu2

    def xi(self, k, x, hat=False):
        xi = 0
        l = self.l
        for tau, sigma_i in enumerate(self.sigmas):
            for mu, sigma_j in enumerate(self.sigmas):
                gauss = exp(-(k * l) ** 2 * x * abs(tau - mu) / 6)
                if hat:
                    gauss = gauss * abs(tau-mu)**2
                xi += sigma_i * gauss * sigma_j
        return xi/self.N

    def zeta(self, k, x, hat=False):
        l = self.l
        zeta = 0
        for tau, sigma_i in enumerate(self.sigmas):
            for mu, sigma_j in enumerate(np.ones_like(self.sigmas)):
                gauss = exp(-(k * l) ** 2 * x * abs(tau - mu) / 6)
                if hat:
                    gauss = gauss * abs(tau-mu)**2
                zeta += sigma_i * gauss * sigma_j
        return zeta / self.N

    def g(self, k, x, hat=False):
        g = 0
        l = self.l
        for tau, sigma_i in enumerate(np.ones_like(self.sigmas)):
            for mu, sigma_j in enumerate(np.ones_like(self.sigmas)):
                gauss = exp(-(k * l) ** 2 * x * abs(tau - mu) / 6)
                if hat:
                    gauss = gauss * abs(tau-mu)**2
                g += sigma_i * gauss * sigma_j
        return g / self.N


