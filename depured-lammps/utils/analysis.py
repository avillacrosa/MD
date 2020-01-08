import lmp
import mdtraj as md
import math
import numpy as np
import os
import time
import datetime
from numba import jit


class Analysis(lmp.LMP):
    def __init__(self, equil=3e6, **kw):
        self.contacts = None
        self.equilibration = equil
        super(Analysis, self).__init__(**kw)

    def contact_map(self):
        pdbs = self.make_initial_frames()
        topo = pdbs[0]



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
        self.contacts = contact_map
        return contact_map

    def contact_map_by_residue(self):
        contacts = self.contacts
        dict_res_translator = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
                               'L': 10,
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

    def ij_from_contacts(self):
        if self.contacts is None:
            self.contact_map()
        means = []
        ijs = []
        for d in range(self.contacts.shape[0]):
            a = np.diagonal(self.contacts, offset=int(d))
            means.append(a.mean())
            ijs.append(d)
        return ijs, means

    def rg_from_lmp(self):
        data = self.get_lmp_data(dir)
        rgs = []
        for run in range(data.shape[0]):
            rg_frame = data[run, data[run, :, 0] > self.equilibration, 4]
            rgs.append(rg_frame)
        return rgs

    def rg_from_mdtraj(self):
        rgs, tloads = [], []
        ti = time.time()
        self.make_initial_frames()
        for dir in self.lmp_drs:
            traj = os.path.join(dir, 'hps_traj.xtc')
            topo = os.path.join(os.path.dirname(traj), 'hps_trj.pdb')
            tload = md.load(traj, top=topo)
            rg = md.compute_rg(tload)
            rgs.append(rg)
            tloads.append(tload)
        print(f'Time expended : {datetime.timedelta(seconds=time.time() - ti)}')
        return rgs

    def flory_scaling(x, flory, r0):
        return r0 * (x ** flory)
