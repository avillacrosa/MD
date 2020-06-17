"""
Implementation of the model here
https://www.sciencedirect.com/science/article/pii/S0167732216322449
"""

import numpy as np
from numpy import log, pi

class RPA():
    def __init__(self, b, N, sigma, L, n, phi_s = 0.,):
        self.N = N
        self.b = b
        self.phi_c = np.sum(sigma)/V#Cert?? perquè no tenim volum enlloc...
        self.phi_s = phi_s
        # L and n determine sigma for symetric ampholites. See Fig 1.


    def s(self, phi_m):
        """
        Entropic part of free energy Eq.
        """
        rw = 1.
        rm = 1.
        rc = 1.
        rs = 1.
        phi_s = self.phi_s
        phi_c = self.phi_c
        N = self.N
        phi_w = (1-rm*phi_m-rc*phi_c-2*rs*phi_s)/rw
        s = phi_m/N*log(phi_m)+phi_c*log(phi_c)+2*phi_s*log(phi_s)+phi_w*log(phi_w)
        s = -s
        return s
    def f_el(self, phi_m, Ts):
        """
        Interaction potential. f_int=f_el in Eq.(1)
        Defined by Eq. 39 and Eq. 40
        """
        def G(k, phi_m, Ts):
            """
            Eq. 40
            """
            def sGms_salt_free(k):
                """
                Eq. 52
                Es cas analitic per seq amb simetria
                de fet ho podem usar per comprovar que sGms_s dona be.
                """
                N = self.N
                sigma = self.sigma
                n = self.N
                L = self.L
                z = np.exp(-(k*b)**2/(6*sigma))
                quo1 = (1-z**(L*sigma))/(1+z**(L*sigma))
                func = N*(sigma*(1+z)/(1-z)-4*z/(L*(1-z)**2)*quo1) +
                       2*z/(1-z)**2 *(quo1)**2 *(1-(-1)**n*z**(N*sigma))
                return func
            def sGms_s(k):
                """
                Eq. 26
                """
                N = self.N
                sigma = self.sigma
                b = self.b
                sGms = 0
                #Slow. May need optimization
                for i in np.arange(sigma.size):
                    for j in np.arange(sigma.size):
                        sGms += sigma[i]*np.exp(-(k*b)**2*np.abs(i-j)/6) *
                                sigma[j]
                return sGms
            G = (2*self.phi_s+self.phi_c+phi_m/N*sGms(k))
            G *= 4*pi/(k*k*(1+k*k)*Ts*eps_const)

        nu = 1.0
        integrand = log(1+nu*G(k))/nu
        integrand -= G(k)
        integrand *= k*k/(4*pi*pi)
        return np.integrate(integrand, -np.inf, np.inf)

    def f_el_eps(self, phi_m, Ts):
        """
        Interaction potential. f_int=f_el in Eq.(1)
        Defined by Eq. 69 (dielectric dependent)
        """
        def G1(k):
            """
            Eq. 69a
            """

            def sGms_s(k):
                """
                Eq. 26
                """
                N = self.N
                sigma = self.sigma
                b = self.b
                sGms = 0
                for i in np.arange(sigma.size):
                    for j in np.arange(sigma.size):
                        sGms += sigma[i]*np.exp(-(k*b)**2*np.abs(i-j)/6) *
                                sigma[j]
                return sGms

            G1 = (2*self.phi_s+self.phi_c+phi_m/N*sGms(k))
            G1 *= 4*pi/(k*k*(1+k*k)*Ts*eps(phi_m))
            return G1
        def G2(k, phi_m):
            """
            Eq. 69b
            """
            G2 = (2*self.phi_s+self.phi_c+phi_m/N*np.sum(sigma))
            G2 *= 4*pi/(k*k*(1+k*k)*Ts*eps(phi_m))
        nu = 1.0
        integrand = log(1+nu*G1(k))/nu
        integrand -= G2(k)
        integrand *= k*k/(4*pi*pi)
        return np.integrate(integrand, -np.inf, np.inf)


    def f(self, phi_m):
        """
        Total free energy
        """
        return -s(self, phi_m)+f_ef(self, phi_m, Ts)

