import numpy as np
import sys
sys.path.insert(1, '../libs')
from methods.EFit import MultiFitAlphas

import utils

t1 = 329.6184330000084515
t2 = 335.3144689999861612

tidx1 = 4
tidx2 = 3


dir = ['/home/adria/perdiux/prod/integrase/integrase-og']

#   t1 to t2
scale = t2/t1
alpha = t1/t2
print(alpha)
alphas = np.ones(shape=8)
alphas[:] = alpha
f1 = MultiFitAlphas(dir, equil=10000, beta=tidx1)
weights = f1.get_w(alphas)
rg = 0
for w in weights:
    rg = np.dot(f1.rg[0][tidx1], w)
print("Original RG", f1.rg[0][tidx1].mean(), "Target RG", f1.rg[0][tidx2].mean(), "Reweighted RG", rg, "Error", rg - f1.rg[0][tidx2].mean(), "Relative error", (rg - f1.rg[0][tidx2].mean())/f1.rg[0][tidx2].mean())

#   t2 to t1
scale = t1/t2
alpha = t2/t1
print(alpha)
alphas = np.ones(shape=8)
alphas[:] = alpha
f2 = MultiFitAlphas(dir, equil=10000, beta=tidx2)
weights = f2.get_w(alphas)
rg = 0
for w in weights:
    rg = np.dot(f2.rg[0][tidx2], w)
print("Original RG", f1.rg[0][tidx2].mean(), "Target RG", f2.rg[0][tidx1].mean(), "Reweighted RG", rg, "Error", rg - f2.rg[0][tidx1].mean(), "Relative error", (rg - f2.rg[0][tidx1].mean())/f2.rg[0][tidx1].mean())

