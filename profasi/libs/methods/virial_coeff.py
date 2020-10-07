import sys
sys.path.insert(1, '/home/adria/scripts/profasi/libs/')

import utils
import math
import numpy as np
import datetime
import time
from numba import jit
import scipy.integrate as integrate
import scipy.constants as cnt


def find_kd(idp_a, idp_b, temp, ionic_strength=0.165, order=2, verbose=True):
    """
    Virial coefficient calculation for 2 charged IDP's based on their charge pattern as from the paper: arXiv:1910.11194
    """
    ionic_strength = ionic_strength * 10 ** (-27) * 10 ** 3
    lb = cnt.e * cnt.e / (4 * math.pi * cnt.epsilon_0 * 80 * cnt.Boltzmann * temp) * 10 ** 9
    kappa = np.sqrt(8 * math.pi * lb * ionic_strength * cnt.Avogadro)
    kuhn = 0.3768

    q_idp_a = idp_a[:, 0].sum()
    q_idp_b = idp_b[:, 0].sum()

    ti = time.time()

    @jit(nopython=True)
    def integrand(x, a, b):
        v = (x / (x ** 2.0 + kappa ** 2)) ** 2.0
        s = 0
        for qa1 in range(a.shape[0]):
            for qa2 in range(a.shape[0]):
                for qb1 in range(b.shape[0]):
                    for qb2 in range(b.shape[0]):
                        s += a[qa1][0] * a[qa2][0] * b[qb1][0] * b[qb2][0] * math.exp(
                            -(1.0 / 6.0) * (abs(a[qa1][1] - a[qa2][1]) + abs(b[qb1][1] - b[qb2][1])) * (
                                        x * kuhn) ** 2.0)
        return v * s

    b2_1 = 4.0 * math.pi * lb * q_idp_a * q_idp_b / (kappa ** 2.0)
    b2_2 = 0

    if order == 2:
        integral = integrate.quad(integrand,
                                  a=0,
                                  b=np.inf,
                                  args=(idp_a, idp_b),
                                  epsabs=0,
                                  epsrel=1.49e-10)
        b2_2 = - 4.0 * (integral[0]) * (lb ** 2.0)
    if verbose:
        print("="*50)
        print(f"Time used :{datetime.timedelta(seconds=time.time()-ti)}")
        print("="*50)

    b2 = (b2_1 + b2_2) * 10 ** (-27)  # In m**3
    if b2 != 0:
        kd = -(1 / (b2 * cnt.Avogadro)) * 10 ** (-3)  # In L
    else:
        kd = 0
    return kd
