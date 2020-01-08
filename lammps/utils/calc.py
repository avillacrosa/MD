import numpy as np
import mdtraj as md
import time
import math
import os
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy import signal
import datetime
from numba import jit

import conversion
import definitions
import lmpmath


def charged_positions(seq):
    charged_plus = []
    charged_minus = []
    for i, aa in enumerate(seq):
        if definitions.residues[aa]["q"] < 0:
            charged_minus.append(i)
        if definitions.residues[aa]["q"] > 0:
            charged_plus.append(i)
    return charged_plus, charged_minus


def contact_map(traj_path, topo):
    @jit(nopython=True)
    def contact_calc(natoms, positions):
        dij = np.zeros(shape=(natoms, natoms))
        for i in range(natoms):
            for j in range(natoms):
                d = 0
                for r in range(3):
                    d += (positions[0, i, r] - positions[0, j, r]) ** 2
                dij[i, j] = math.sqrt(d)
        return dij
    traj = md.load(traj_path, top=topo)
    traj = traj[2000:5000]
    dframe = []
    for frame in range(traj.n_frames):
        tframe = traj[frame]
        dijf = contact_calc(tframe.n_atoms, tframe.xyz)
        dframe.append(dijf)
    dframe = np.array(dframe)
    contact_map = dframe.mean(0) * 10
    return contact_map


def ij_from_contacts(contacts):
    means = []
    ijs = []
    for d in range(contacts.shape[0]):
        a = np.diagonal(contacts, offset=int(d))
        means.append(a.mean())
        ijs.append(d)
    return ijs, means


def res_contact_map(traj_path, topo, seq):
    dict_res_translator = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                           'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
    saver = np.zeros(shape=(20, 20))
    dicti = {}
    for k in dict_res_translator.keys():
        for l in dict_res_translator.keys():
            dicti[k, l] = []
    contacts = contact_map(traj_path, topo)
    for i in range(contacts.shape[0]):
        for j in range(contacts.shape[0]):
            iaa = seq[i]
            jaa = seq[j]
            dicti[iaa, jaa].append(contacts[i][j])

    for key in dicti.keys():
        saver[dict_res_translator[key[0]]][dict_res_translator[key[1]]] = np.array(dicti[key]).mean()

    return saver, dict_res_translator


def rg_from_lmp(dir, equil=3655000):
    data, kappas = conversion.getLMPthermo(dir)
    rgs = []
    for run in range(data.shape[0]):
        rg_frame = data[run, data[run, :, 0] > equil, 4]
        rgs.append(rg_frame)
    Is = lmpmath.I_from_debye(kappas, eps_rel=80, T=300, from_angst=True)
    Is = np.array(Is) * 10 ** 3
    return rgs, Is


def rg_from_mdtraj(dir):
    dirs = conversion.getLMPdirs([dir])
    data, kappas = conversion.getLMPthermo(dirs[0])
    dirs.sort()
    rgs, tloads = [], []
    ti = time.time()
    dirs.pop(0)
    for dir in dirs:
        traj = os.path.join(dir, 'hps_traj.xtc')
        conversion.getPDBfromLAMMPS(os.path.join(os.path.dirname(traj), 'hps'))
        topo = os.path.join(os.path.dirname(traj), 'hps_trj.pdb')
        tload = md.load(traj, top=topo)
        rg = md.compute_rg(tload)
        rgs.append(rg)
        tloads.append(tload)
    print(f'Time expended : {datetime.timedelta(seconds=time.time() - ti)}')
    Is = lmpmath.I_from_debye(kappas, eps_rel=80, T=300, from_angst=True)
    Is = np.array(Is) * 10 ** 3
    return Is, rgs
