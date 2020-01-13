# import sys
# import importlib
# sys.path.insert(1,'../utils')
# import lmp
# import lmpsetup
# import analysis
# import numpy as np
# import math
# importlib.reload(lmp);
# importlib.reload(lmpsetup);
# importlib.reload(analysis);
#
# def contact_calc(natoms, positions):
#     dij = np.zeros(shape=(natoms, natoms))
#     for i in range(natoms):
#         for j in range(natoms):
#             d = 0
#             for r in range(3):
#                 d += (positions[0, i, r] - positions[0, j, r]) ** 2
#             dij[i, j] = math.sqrt(d)
#     print(dij)
#     print("HELLO?!")
#     return dij
#
#
# def collect_result(result):
#     global results
#     results.append(result)
#
# ana_dir = '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.051'
# analyzer = analysis.Analysis(oliba_wd = ana_dir)
# analyzer.contact_map(use_jit=False)

import mdtraj as md
import math
import numpy as np
import os
import time
import datetime
import multiprocessing as mp
import glob
from numba import jit


def contact_calc(natoms, positions):
    dij = np.zeros(shape=(natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            d = 0
            for r in range(3):
                d += (positions[0, i, r] - positions[0, j, r]) ** 2
            dij[i, j] = math.sqrt(d)
    print(dij)
    return dij


def collect_result(result):
    global results
    results.append(result)

class Dipshit():
    def __init__(self):
        self.wtf = 0

    def test(self):

        topo = '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.051/hps_trj.pdb'

        xtcs = glob.glob(os.path.join('/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.051', '*xtc'))
        xtc = xtcs[0]

        traj = md.load(xtc, top=topo)
        traj_mp = traj.n_frames / mp.cpu_count()

        pool = mp.Pool(mp.cpu_count())

        results = []
        ranges = np.linspace(0, traj.n_frames, mp.cpu_count() + 1, dtype='int')
        for i in range(1, len(ranges)):
            tframed = traj[ranges[i - 1]:ranges[i]]
            results.append(pool.apply_async(contact_calc,
                                            args=(tframed.n_atoms, tframed.xyz), callback=collect_result))
        pool.close()
        pool.join()

a = Dipshit()
a.test()