#!/usr/bin/env python

"""
Reweight Profasi ensembles to fit different Rg.
Compare MaxEnt with direct fit
Use a class to define the functions
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os.path import basename

import sys

sys.dont_write_bytecode = True
# Don't create __pycache__
sys.path.insert(1, '../libs')

import utils

plt.ion()


class MultiFitAlphas():
    def __init__(self, pathnames, equil, beta):
        """
        pathname: where the profasi results directory resides
        equil: number of equilibration steps to ignore
        """
        self.pathnames = pathnames
        self.equil = equil
        self.ene_fact = 1.342 * 1000.
        self.energies, self.rg = self.get_energies()
        self.temperature_index = None
        self.beta = None
        self.headers = None
        self.set_beta(index=beta)
        self.get_headers()

    def get_energies(self):
        """
        Get the different energy components and the Rg
        Get result for all temperatures
        It reads the base directory of the profasi bin (where the n* directories are)
        Return a dictionary of numpy arrays with energy components and an array with Rg
               at different temperatures
        """
        all_energies = []
        all_rgs = []
        for prot in self.pathnames:
            data = np.genfromtxt(join(prot, 'results', 'rt'))
            # Now sort the data according to the temperature
            temperatures = np.unique(data[:, 2]).astype(np.int)
            energies = {}
            rg = {}
            data = data[data[:, 1] > self.equil, :]
            for t in temperatures:
                indices = data[:, 2] == t
                energies[t] = data[indices, 4:12] * self.ene_fact
                rg[t] = data[indices, -1]
            all_energies.append(energies)
            all_rgs.append(rg)

        return all_energies, all_rgs

    def set_beta(self, index):
        """
        Choose a temperature index from the filename.
        Temperatures should be in Kelvin
        """
        filename = join(self.pathnames[0], 'results', 'temperature.info')
        temperatures = np.genfromtxt(filename)
        temp = temperatures[index, 2]  # Third column in kelvin
        r = 1.9872159  # cal/K/mol
        self.beta = 1. / (r * temp)
        self.temperature_index = index

    def get_headers(self):
        """
        Get name of each component of the energy
        """
        obs = utils.get_obs_names(self.pathnames[0])
        e_names = obs[4:12]
        self.headers = e_names

    def get_w(self, alphas):
        """
        return the new weights when changing the alphas (energy term coefficients)
        """
        if not self.beta:
            print('temperature not set. Call set_beta')
            return
        weights = []
        for energy in self.energies:
            energies = energy[self.temperature_index]
            E0 = np.sum(energies, axis=1)
            E = np.dot(energies, alphas)
            E -= E0
            E -= E.min()
            w = np.exp(-self.beta * E)
            w /= np.sum(w)
            weights.append(w)
        return weights

    def rg_error(self, alphas, rg_target, reg=0.0):
        """
        Calculate the error between the reweighted rg and a target rg
        reg is a regularization term.
        """
        if not self.beta:
            print('temperature not set. Call set_beta')
            return

        err = 0
        w = self.get_w(alphas)
        for idx, weight in enumerate(w):
            rg = self.rg[idx][self.temperature_index]
            rg = np.dot(rg, weight)
            err += (rg - rg_target[idx]) ** 2
        err += reg * np.sum((alphas - 1) ** 2)
        return err

    def get_energy_average(self, alphas, obs):
        if type(obs) is int:
            e_i = obs
            print(self.headers)
        else:
            e_i = self.headers.index(obs)
        e = []
        w = self.get_w(alphas)
        for idx, weight in enumerate(w):
            e_component = self.energies[idx][self.temperature_index][:, e_i]
            e_component = np.dot(e_component, weight)
            e.append(e_component)
        print(e)

    def rg_error_fixed_T(self, alphas, rg_target, reg=0.0):
        """
        Calculate the error between the reweighted rg and a target rg
        reg is a regularization term.
        """
        if not self.beta:
            print('temperature not set. Call set_beta')
            return
        w = self.get_w(alphas)
        rg = self.rg[self.temperature_index]
        rg = np.dot(rg, w)
        k = 10000000.
        return (rg - rg_target) ** 2 + reg * np.sum((alphas - 1) ** 2) + k * (np.mean(alphas) - 1) ** 2

    def n_eff(self, alphas, kish=False):
        """
        Return the normalized effective sample size
        if kiss: Kish n_effective size
        else: relative entropy based
        """
        w = self.get_w(alphas)
        n_eff = []
        if kish:
            for weight in w:
                n_eff.append(1. / np.sum(weight ** 2) / weight.size)
        else:
            for weight in w:
                n_eff.append(np.exp(-np.sum(weight * np.log(weight * weight.size))))
        return n_eff

    def get_maxent_w(self, rg_target, sigma, theta):
        """
        return the weights from applying MaxEnt with a given theta
        to fit the experimental rg_target
        """
        import bme_reweight as bme
        if not self.beta:
            print('temperature not set. Call set_beta')
            return
        bmea = bme.Reweight(verbose=False)
        with open('../default_output/exp_file.dat', 'wt') as exp_file:
            exp_file.write(f'# DATA=SAXS PRIOR=GAUSS\nrg {rg_target}  {sigma}\n')
        with open('../default_output/calc_file.dat', 'wt') as calc_file:
            calc_file.write('# frame   rg\n')
            # for i, rg in enumerate(self.rg[self.temperature_index]):
            # TODO CAREFUL HERE; ONLY CONSIDERING 1D MINIMIZATION...
            for i, rg in enumerate(self.rg[0][self.temperature_index]):
                calc_file.write(f'{i}  {rg:.3f}\n')
        bmea.load('../default_output/exp_file.dat', '../default_output/calc_file.dat')
        bmea.optimize(theta=theta)
        return bmea.get_weights()

    def w_residuals(self, alphas, w_target):
        """
        Residuals of the fit to the weights to be called by least_squares
        """
        w = self.get_w(alphas)
        w = w[0]
        return w - w_target

    def w_error(self, alphas, w_target):
        """
        Residuals of the fit to the weights to be called by least_squares
        """
        w = self.get_w(alphas)
        return np.sum((w - w_target) ** 2)

    def fit(self, rg_target, i_alphas, regulator=0, verbose=True, method="BFGS"):
        from scipy.optimize import minimize

        result = minimize(self.rg_error, i_alphas, args=(rg_target, regulator), method=method)
        fitted_alphas = result.x

        if verbose is True:
            rg_0, rg_reweighted = [], []
            weights = self.get_w(fitted_alphas)
            for idx, rg in enumerate(self.rg):
                rg_0.append(rg[self.temperature_index].mean())
                rg_reweighted.append(np.dot(weights[idx], rg[self.temperature_index]))
            rg_0 = np.array(rg_0)
            rg_reweighted = np.array(rg_reweighted)

            print("=" * 100)
            if regulator == 0:
                print("Fitting without regulator")
            else:
                print("Fitting with regulator " + str(regulator))
            for idx, prot in enumerate(self.pathnames):
                name = basename(prot)
                print(len(name))
                print("=" * int(((100-len(name))/2)), name, "=" * int(((100-len(name))/2)))
                print(f"Sucessful minimization: {result.success}")
                print(f'Initial Rg = {rg_0[idx]:.2f} --- Target Rg = {rg_target[idx]:.2f} --- Reweighted Rg = {rg_reweighted[idx]:.2f}')
                print("Alphas:")
                with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
                    print(fitted_alphas)
                print(f'Efective sample size: {self.n_eff(fitted_alphas)[idx]:.3f}')

        return fitted_alphas

    def maxent_fit(self, rg_target, i_alphas, verbose=True):
        from scipy.optimize import least_squares

        my_sigma = 0.5
        my_theta = 1.0

        #TODO CAREFUL; ONLY WILL WORK ON 1D

        w_target = self.get_maxent_w(rg_target, my_sigma, my_theta)
        result = least_squares(self.w_residuals, i_alphas, args=(w_target,))
        fitted_alphas = result.x
        rg_0 = self.rg[0][self.temperature_index].mean()
        rg_reweighted = np.dot(self.get_w(fitted_alphas), self.rg[0][self.temperature_index])

        if verbose is True:
            print("=" * 100)
            print('Using MaxEnt')
            print("=" * 100)
            print(f"Sucessful minimization: {result.success}")
            print(f'Initial Rg = {rg_0:.2f} --- Target Rg = {rg_target:.2f} --- Reweighted Rg = {rg_reweighted:.2f}')
            print("Alphas:")
            with np.printoptions(formatter={'float': '{: 0.3f}'.format}):
                print(fitted_alphas)
            print(f'Efective sample size: {self.n_eff(fitted_alphas):.3f}')

        return fitted_alphas
