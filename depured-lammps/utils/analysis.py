# TODO: Solve import deficiencies (might help Pycharm)
import lmp
import itertools
import mdtraj as md
import math
import numpy as np
import os
import re
import time
import datetime
import multiprocessing as mp
import glob
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from pathlib import Path

from numba import jit


class Analysis(lmp.LMP):
    def __init__(self, **kw):
        # TODO DO TEMPER HANDLE AT START
        self.contacts = None
        self.equilibration = 3e6
        super(Analysis, self).__init__(**kw)

    def distance_map(self, use='default', contacts=False):
        # print("="*20, f"Calculating contact map using {use}", "="*20)
        pdbs = self.make_initial_frame()
        topo = pdbs[0]
        contact_maps = []
        if self.temper:
            xtcs = self._temper_trj_reorder()
            # xtcs = glob.glob(os.path.join(self.o_wd, '*.xtc*'))
        else:
            xtcs = []
            for xtc in Path(self.o_wd).rglob('*.xtc'):
                xtcs.append(os.fspath(xtc))
            xtcs = sorted(xtcs)
        for xtc in xtcs:
            traj = md.load(xtc, top=topo)
            # TODO: TEST FIRST PART OF IF
            if use == 'jit':
                dframe = []
                for frame in range(traj.n_frames):
                    tframe = traj[frame]
                    dijf = jit_contact_calc(tframe.n_atoms, tframe.xyz)
                    dframe.append(dijf)
                dframe = np.array(dframe)
                contact_map = dframe.mean(0)
            elif use == 'md':
                pairs = list(itertools.product(range(traj.top.n_atoms - 1), range(1, traj.top.n_atoms)))
                d = md.compute_distances(traj, pairs)
                d = md.geometry.squareform(d, pairs)
                if contacts:
                    dcop = np.copy(d)
                    dcop[d < 0.6] = 1.
                    dcop[d > 0.6] = 0.
                    contact_map = dcop.mean(0)
                else:
                    contact_map = d.mean(0)
            else:
                pool = mp.Pool()
                ranges = np.linspace(0, traj.n_frames, mp.cpu_count() + 1, dtype='int')
                for i in range(1, len(ranges)):
                    t_r = traj[ranges[i-1]:ranges[i]]
                    pool.apply_async(mpi_contact_calc,
                                     args=(t_r.n_atoms, t_r.xyz),
                                     callback=lambda x: mpi_results.append(x))
                pool.close()
                pool.join()
                contact_map = np.mean(mpi_results, axis=0)
            contact_map = contact_map * 10
            contact_maps.append(contact_map)
        self.contacts = np.array(contact_maps)
        # print("="*20, "CONTACT MAP CALCULATION FINISHED", "="*20)
        return contact_maps

    def ij_from_contacts(self, use='default', contacts=None):
        # print("="*20, f"Calculating ij from contact map using {use}", "="*20)
        if contacts is None:
            self.distance(use=use)
        else:
            self.contacts = contacts
        tot_means = []
        tot_ijs = []
        for contact in self.contacts:
            ijs = []
            means = []
            for d in range(contact.shape[0]):
                a = np.diagonal(contact, offset=int(d))
                means.append(a.mean())
                ijs.append(d)
            tot_ijs.append(ijs)
            tot_means.append(means)
        # print("="*20, "D_IJ FROM CONTACT MAP CALCULATION FINISHED", "="*20)
        return np.array(tot_ijs), np.array(tot_means)

    def flory_scaling_fit(self, r0=None, use='default', ijs=None):
        # print("="*20, f"Starting flory exponent calculation for R0 = {r0} using {use}", "="*20)
        if ijs is None:
            tot_ijs, tot_means = self.ij_from_contacts(use=use)
        else:
            tot_ijs, tot_means = ijs[0], ijs[1]
        florys, r0s = [], []
        for i, ij in enumerate(tot_ijs):
            mean = tot_means[i]
            if r0 is None:
                fit, fitv = curve_fit(full_flory_scaling, ij, mean)
                fit_flory = fit[0]
                fit_r0 = fit[1]
            else:
                fit, fitv = curve_fit(flory_scaling(r0), ij, mean)
                fit_flory = fit[0]
                fit_r0 = r0
            florys.append(fit_flory)
            r0s.append(fit_r0)
        # print("="*20, "FLORY EXPONENT CALCULATION FINISHED", "="*20)
        return np.array(florys), np.array(r0s)

    def distance_map_by_residue(self):
        contacts = self.contacts
        dict_res_translator = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                               'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
        saver = np.zeros(shape=(20, 20))
        dicti = {}
        for k in dict_res_translator.keys():
            for l in dict_res_translator.keys():
                dicti[k, l] = []
        for i in range(contacts.shape[0]):
            for j in range(contacts.shape[0]):
                iaa = self.sequence[i]
                jaa = self.sequence[j]
                dicti[iaa, jaa].append(contacts[i][j])

        for key in dicti.keys():
            saver[dict_res_translator[key[0]]][dict_res_translator[key[1]]] = np.array(dicti[key]).mean()

        return saver, dict_res_translator

    def rg(self, use='lmp'):
        if self.temper:
            data = self.get_lmp_temper_data()
        else:
            data = self.get_lmp_data()
        rgs = []
        xtcs = self._temper_trj_reorder()
        # TODO: DEFINIETELY BROKEN
        if use == 'lmp':
            data = self.data
            for run in range(data.shape[0]):
                rg_frame = data[run, data[run, :, 0] > self.equilibration, 4]
                rgs.append(rg_frame)
        if use == 'md':
            tloads = []
            topo = self.make_initial_frame()[0]
            for xtc in xtcs:
                tload = md.load(xtc, top=topo)
                rg = md.compute_rg(tload)*10.
                rgs.append(rg)
            # for dir in self.lmp_drs:
            #     traj = os.path.join(dir, 'hps_traj.xtc')
            #     topo = os.path.join(os.path.dirname(traj), 'hps_trj.pdb')
            #     tload = md.load(traj, top=topo)
            #     rg = md.compute_rg(tload)
            #     rgs.append(rg)
            #     tloads.append(tload)
        return np.array(rgs)

    def get_rg_distribution(self, scikit=False):
        rgs = self.rg_from_lmp()
        rgs = rgs[0]
        kde_scipy = gaussian_kde(rgs)
        kde_scikit = KernelDensity(kernel='gaussian').fit(rgs[:, np.newaxis])

        x = np.linspace(20, 40, 1000)
        xscikit = np.linspace(20, 40, rgs.shape[0])[:, np.newaxis]
        if scikit:
            return kde_scikit, xscikit
        return kde_scipy, x

    def get_rg_acf(self):
        rgs = self.rg_from_lmp()
        rgs = rgs[0][0]
        acf = sm.tsa.stattools.acf(rgs, nlags=1000)
        return rgs, acf


mpi_results = []


def full_flory_scaling(x, flory, r0):
    return r0 * (x ** flory)


def flory_scaling(r0):
    def flory(x, flory):
        return r0 * (x ** flory)

    return flory


@jit(nopython=True)
def jit_contact_calc(natoms, positions):
    dij = np.zeros(shape=(natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            d = 0
            for r in range(3):
                d += (positions[0, i, r] - positions[0, j, r]) ** 2
            dij[i, j] = math.sqrt(d)
    return dij


def mpi_contact_calc(natoms, positions):
    d_fr = []
    for fr in range(positions.shape[0]):
        dij = np.zeros(shape=(natoms, natoms))
        for i in range(natoms):
            for j in range(natoms):
                d = 0
                for r in range(3):
                    d += (positions[fr, i, r] - positions[fr, j, r]) ** 2
                dij[i, j] = math.sqrt(d)
        d_fr.append(dij)
    return np.array(d_fr).mean(axis=0)
