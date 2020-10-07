#!/usr/bin/env python

"""
Do some plots fitting at different Rg to check alphas evolution and n_eff
Need to work with larger sample
Check equilibration time
Check also temperature and actual experimental Rg
"""

import numpy as np
import sys
import math
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import os

sys.path.insert(1, '../libs')
from methods.EFit import MultiFitAlphas


def alphas_from_reg(dir, alphas=np.ones(8), beta=2, normalize=False, rg_save=None, scales=(0.9, 1.1)):
    iterations = 50

    f = MultiFitAlphas(dir, equil=10000, beta=beta)
    rg_exps = []
    for rg in f.rg:
        rg_exps.append(rg[beta].mean())

    full_rgs = []
    full_alphas = []
    full_effs = []

    for idx, rg_scale in enumerate(scales):
        rg = np.array(rg_exps) * rg_scale
        rg_error = []
        all_effs = []
        all_alphas = []
        all_rgs = []
        for r in np.logspace(5, 0, iterations):
            alphas = f.fit(rg_target=rg, i_alphas=alphas, regulator=r, verbose=False, method="Nelder-Mead")
            rg_reweighted = []
            for k, weight in enumerate(f.get_w(alphas)):
                rg_reweighted.append(np.dot(weight, f.rg[k][f.temperature_index]))
            rg_error.append(rg - rg_reweighted)
            all_effs.append(f.n_eff(alphas))
            all_rgs.append(rg_reweighted)
            all_alphas.append(alphas)
            if rg_save is not None:
                # if rg_exps[0] * 1.05 <= rg_reweighted[0] <= rg_exps[0] * 1.06:
                if rg_save * 0.995 <= rg_reweighted[0] <= rg_save * 1.005:
                    print("Beta", beta, "RG-rewe", rg_reweighted, "rg exp", rg_exps, "r", r, "rg_save", rg_save)
                    print("alphas", alphas, "\n")
        all_alphas = np.array(all_alphas)
        all_rgs = np.array(all_rgs)
        all_effs = np.array(all_effs)
        if rg_scale < 1:
            all_alphas = all_alphas[::-1]
            all_rgs = all_rgs[::-1]
            all_effs = all_effs[::-1]
        np.flip(all_alphas)
        np.flip(all_rgs)
        if normalize:
            all_rgs = all_rgs - np.array(rg_exps)

        full_effs.append(all_effs)
        full_rgs.append(all_rgs)
        full_alphas.append(all_alphas)

    full_alphas = np.array(full_alphas)
    full_rgs = np.array(full_rgs)
    full_alphas = np.array(full_alphas).reshape(len(scales) * iterations, 8)

    if len(dir) == 1:
        full_rgs = np.array(full_rgs).reshape(len(scales) * iterations)
        full_effs = np.array(full_effs).reshape(len(scales) * iterations)
    else:
        full_rgs = np.array(full_rgs).reshape(len(scales) * iterations, len(dir))
        full_effs = np.array(full_effs).reshape(len(scales) * iterations, len(dir))

    return full_alphas, full_rgs, full_effs


def alphas_from_rg(dir, alphas=np.ones(8),  beta=2, regularizer=110, normalize=False):

    f = MultiFitAlphas(dir, equil=10000, beta=beta)
    rg_exps = []
    for rg in f.rg:
        rg_exps.append(rg[beta].mean())
    rg_error = []
    all_effs = []
    all_alphas = []
    all_rgs = []
    # TODO HARD CODED RANGES; SHOULD BE DYNAMIC IF POSSIBLE!!!!!!!!!!!!!!
    for rg in np.logspace(math.log(5), math.log(30), 250, base=math.e):
        # TODO !!!!!!!!!!!!!!
        rg = [rg]
        alphas = f.fit(rg_target=rg, i_alphas=alphas, regulator=regularizer, verbose=False, method="Nelder-Mead")
        rg_reweighted = []
        for k, weight in enumerate(f.get_w(alphas)):
            rg_reweighted.append(np.dot(weight, f.rg[k][f.temperature_index]))
        rg_error.append(rg[0] - rg_reweighted)
        all_effs.append(f.n_eff(alphas))
        all_rgs.append(rg_reweighted)
        all_alphas.append(alphas)

    all_alphas = np.array(all_alphas)
    all_rgs = np.array(all_rgs)

    if normalize:
        all_rgs = all_rgs - np.array(rg_exps)

    if len(dir) == 1:
        all_rgs = all_rgs.reshape((all_rgs.shape[0],))

    return all_alphas, all_rgs, all_effs


def alphas_from_maxent(dir, alphas=np.ones(8), beta=2, normalize=False, rg_save=None):
    f = MultiFitAlphas(dir, equil=10000, beta=beta)
    rg_exps = []
    for rg in f.rg:
        rg_exps.append(rg[beta].mean())
    rg_error = []
    all_effs = []
    all_alphas = []
    all_rgs = []
    # TODO !!!!!!!!!!!!!!
    for rg in np.logspace(math.log(12), math.log(22), 250, base=math.e):
        # TODO !!!!!!!!!!!!!!
        alphas = f.maxent_fit(rg_target=rg, i_alphas=alphas, verbose=False)
        rg_reweighted = []
        for k, weight in enumerate(f.get_w(alphas)):
            rg_reweighted.append(np.dot(weight, f.rg[k][f.temperature_index]))
        rg_error.append(rg - rg_reweighted)
        all_effs.append(f.n_eff(alphas))
        all_rgs.append(rg_reweighted)
        all_alphas.append(alphas)

    all_alphas = np.array(all_alphas)
    all_rgs = np.array(all_rgs)

    if normalize:
        all_rgs = all_rgs - np.array(rg_exps)

    if len(dir) == 1:
        all_rgs = all_rgs.reshape((all_rgs.shape[0],))

    return all_alphas, all_rgs, all_effs
