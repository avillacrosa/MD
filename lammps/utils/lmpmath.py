import numpy as np
import scipy.constants as cnt


def debye_length(I, eps_rel=80, T=300, angstrom=False):
    kappa = 1 / (np.sqrt(2 * I * 10 ** 3 * cnt.Avogadro / (cnt.epsilon_0 * eps_rel * cnt.Boltzmann * T)) * cnt.e)
    if angstrom:
        kappa = kappa
    return kappa


def I_from_debye(kappas, eps_rel=80, T=300, from_angst=False):
    Is = []
    for kappa in kappas:
        kappa = 1/kappa
        if from_angst:
            kappa = kappa * 10 ** (-10)
        I = cnt.epsilon_0 * eps_rel * cnt.Boltzmann * T / (kappa*kappa*cnt.e*cnt.e*2*10**3*cnt.Avogadro)
        Is.append(I)
    return Is


def flory_scaling(x, flory, r0):
    return r0 * (x ** flory)

# def flory_scaling(x, flory):
#     return 10. * (x ** flory)
