#!/usr/bin/env python

import numpy as np
import sys
sys.dont_write_bytecode = True
sys.path.insert(1, '/home/adria/scripts/depured_profasi/libs/')
from methods.multi_rg_reweight import MultiFitAlphas

paths = ['/home/adria/data/asyn37h', '/home/adria/data/integrase']
f = MultiFitAlphas(paths, equil=10000, beta=2)
f.n_eff(np.ones(8))
rg_exps = []

for rg in f.rg:
    rg_exp = rg[2].mean() * 1.1  #expanded rg
    rg_exps.append(rg_exp)

alphas = np.ones(8)
alphas = f.fit(rg_target=rg_exps, i_alphas=alphas, regulator=2, verbose=True)
# f.maxent_fit(rg_target=rg_exp)
