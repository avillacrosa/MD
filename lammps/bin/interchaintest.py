import analysis
import time

sr = analysis.Analysis(oliba_wd='/home/adria/data/prod/lammps/7D_CPEB4x50/REX', max_frames=4)
ti = time.time()
ampi_res = sr.async_inter_distance_map(temperature=0)
print("Time spent", time.time()-ti)




import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit


def hps(x, lamb, sigm):
    LJ = 4 * hps_eps * (sigm ** 12 / x ** 12 - sigm ** 6 / x ** 6)
    if x <= 2 ** (1 / 6) * sigm:
        return LJ + (1 - lamb) * hps_eps
    else:
        return lamb * LJ


def ljfitter(sigma):
    def lj612(x, C12, C6):
        x6 = (sigma / x) ** 6
        LJ = 4 * hps_eps * (C12 * x6 ** 2 + C6 * x6)
        return LJ

    return lj612


hps_eps = 0.2 * 4.184

ls, ss = [], []

lambdas = {'ALA': 0.72973, 'ARG': 0.0, 'ASN': 0.432432, 'ASP': 0.378378, 'CYS': 0.594595, 'GLU': 0.459459,
           'GLN': 0.513514, 'GLY': 0.648649, 'HIS': 0.513514, 'ILE': 0.972973, 'LEU': 0.972973, 'LYS': 0.513514,
           'MET': 0.837838, 'PHE': 1.0, 'PRO': 1.0, 'SER': 0.594595, 'THR': 0.675676, 'TRP': 0.945946, 'TYR': 0.864865,
           'VAL': 0.891892}
sigmas = {'ALA': 5.04, 'ARG': 6.56, 'ASN': 5.68, 'ASP': 5.58, 'CYS': 5.48, 'GLN': 6.02, 'GLU': 5.92, 'GLY': 4.5,
          'HIS': 6.08, 'ILE': 6.18, 'LEU': 6.18, 'LYS': 6.36, 'MET': 6.18, 'PHE': 6.36, 'PRO': 5.56, 'SER': 5.18,
          'THR': 5.62, 'TRP': 6.78, 'TYR': 6.46, 'VAL': 5.86}

# DICT TO ARRAY
for k in lambdas:
    ls.append(lambdas[k])
    ss.append(sigmas[k] / 10)

pairs_l, pairs_s = [], []

# PAIRS
for l in range(len(ls)):
    for m in range(l, len(ls)):
        pairs_l.append((ls[l] + ls[m]) / 2)
        pairs_s.append((ss[l] + ss[m]) / 2)

rmin, rmax = 5, 20
r = np.linspace(rmin / 10, rmax / 10, 2000)
hps_data, fits = [], []
fits_lse = []

for i in range(len(pairs_l)):
    hps_data = []
    for rr in r:
        hps_data.append(hps(rr, pairs_l[i], pairs_s[i]))
    fit = curve_fit(ljfitter(pairs_s[i]), r, hps_data)
    fits.append([fit[0][0], fit[0][1]])

plt.figure(figsize=(8, 6))
plt.title("$C_6$")
plt.xlabel('λ', fontsize=14)
plt.ylabel('$-C_6/4\epsilon\sigma^6$', fontsize=14)
plt.plot(pairs_l, -np.array(fits)[:, 1], '.')
line = np.polyfit(np.array(pairs_l), -ndfits[:, 1], 1)
print(line)
plt.plot(pairs_l, line[1] + line[0] * np.array(pairs_l))