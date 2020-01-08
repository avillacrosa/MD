import math
import numpy as np
import numba
from numba import jit
import time
import scipy.integrate as integrate
import scipy.constants as cnt
from scipy.optimize import minimize
from decimal import Decimal
import multiprocessing as mp

h1 = {
    "seq": "TENSTSAPAAKPKRAKASKKSTDHPKYSDMIVAAIQAEKNRAGSSRQSIQKYIKSHYKVGENADSQIKLSIKRLVTTGVLKQTKGVGASGSFRLAKSDEPKKSVAFKKTKKEIKKVATPKKASKPKKAASKAPTKKPKATPVKKAKKKLAATPKKAKKPKTVKAKPVKASKPKKAKPVKPKAKSSAKRAGKKK",
    "qseq": [],
    "q": +53
}
protalpha = {
    "seq": "GPSDAAVDTSSEITTKDLKEKKEVVEEAENGRDAPANGNAENEENGEQEADNEVDEEEEEGGEEEEEEEEGDGEEEDGDEDEEAESATGKRAAEDDEDDDVDTKKQKTDEDD",
    "qseq": [],
    "q": -44
}


def findO2factors(protA, protB):
    @jit(nopython=True)
    def loop_speedup(qseqA, qseqB):
        intfactors = []
        efactors = []
        for qa1 in range(qseqA.shape[0]):
            for qa2 in range(qseqA.shape[0]):
                for qb1 in range(qseqB.shape[0]):
                    for qb2 in range(qseqA.shape[0]):
                        intfactors.append(qseqA[qa1][0] * qseqA[qa2][0] * qseqA[qb1][0] * qseqA[qb2][0])
                        efactors.append(abs(qseqA[qa1][1] - qseqA[qa2][1]) + abs(qseqA[qb1][1] - qseqA[qb2][1]))
        return np.array(intfactors), np.array(efactors)

    charge_dict = {'D': -1, 'E': -1, 'R': +1, 'K': +1}  # This dict reproduces data from the paper
    for prot in [protA, protB]:
        for i, d in enumerate(prot["seq"]):
            if charge_dict.get(d, None) is not None:
                prot["qseq"].append([charge_dict[d], i + 1])
        prot["qseq"] = np.array(prot["qseq"])

    qseqA = protA["qseq"]
    qseqB = protB["qseq"]

    return loop_speedup(qseqA, qseqB)


def findKd(protA, protB, lb, kappa, intfactors, efactors, kuhn, ranges, order=1, ranget=1e8, mpi=True):
    results = []

    def integrand(x, kuhn, kappa, efactor):
        v = (x / (x ** 2 + kappa ** 2)) ** 2
        s = math.exp(-(1.0 / 6.0) * efactor * (x * kuhn) ** 2.0)
        return v * s

    def integrate(a, b, kuhn, kappa, efactors, ranget, range_mpi):
        for i in range(range_mpi[0], range_mpi[1]):
            integral = integrate.quad(integrand,
                                      a=0,
                                      b=ranget,
                                      args=(a["qseq"], b["qseq"], kuhn, kappa, efactors[i]),
                                      epsabs=0)

    def collect_result(result):
        global results
        results.append(result)

    B2_O1 = 4.0 * math.pi * lb * protA["q"] * protB["q"] / (kappa ** 2.0)
    B2_O2 = 0

    if order == 2:
        for i in range(1, 13):
            old_range = np.sum(ranges[0:i])
            new_range = np.sum(ranges[0:i + 1])
            results.append(pool.apply_async(integrate,
                                            args=(protA, protB, kuhn, kappa, efactors, ranget, [old_range, new_range]),
                                            callback=collect_result))

        pool.close()
        pool.join()

        results = [r.get()[1] for r in results]

        B2_O2 = - 4.0 * lb ** 2.0 * np.sum(np.array(results))

    B2 = B2_O1 + B2_O2
    kD = (-1 / (B2 * cnt.Avogadro))
    return kD, results, B2_O1, B2_O2

pool = mp.Pool(mp.cpu_count())

expkD0 = np.array([142, 189, 223, 257, 300]) * 10 ** -6
expkD1 = np.array([3.41, 5.09, 6.46, 7.94, 9.95]) * 10 ** -6

c     = np.array([165, 220, 260, 300, 350]) * 10 ** -3
I     = c[0]                                                                        # Salt Concentration in Molar
lb    = cnt.e * cnt.e / (4 * math.pi * cnt.epsilon_0 * 80.2 * cnt.Boltzmann * 300)  # Bjerrum Length (at T=300K)
kappa = np.sqrt(4 * math.pi * lb * (2 * I * cnt.Avogadro))                          # Debye Wavenumber
kuhn  = 3.5e-10                                                                     # Kuhn Length

h1["qseq"]        = []
protalpha["qseq"] = []
intfactors, efactors = findO2factors(h1, protalpha)

ranges = [0,1908858,1908858,1908858,1908858,1908859,1908859,1908859,1908859,1908859,1908859,1908859,1908859]

Kd = findKd(h1, protalpha, lb, kappa, intfactors, efactors, kuhn, ranges, order=2, ranget=np.inf)