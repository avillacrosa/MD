# ! /usr/bin/env python3

import mdtraj as md
import glob
import MDAnalysis
from DEERpredict.PREPrediction5 import PREPrediction
import sys
import time
from multiprocessing import Pool
from joblib import Parallel, delayed

# filelist = glob.glob('/home/adria/data/asyn/pre/asyn100k/frames/traj100k.xtc')
# traj = md.load(filelist, top='/home/adria/data/asyn/pre/asyn100k/frames/frame_0000000.pdb')
# trj1 = traj[::2]
# trj2 = traj[1::2]
# trj1.save("PREs100/traj1.xtc")
# trj2.save("PREs100/traj2.xtc")
# # print(traj)
# # traj.save("/home/adria/data/asyn/pre/asyn1k/frames/traj1k.xtc")

# CALCULATE INITAL PREs
label_list = (18, 20, 24, 42, 62, 76, 85, 87, 90, 103, 120, 140)
#
def run_PREPrediction(lab):
    proteinStructure = MDAnalysis.Universe('/home/adria/data/asyn/pre/asyn100k/frames/frame_0000000.pdb',
                                           'PREs100/traj1.xtc')
    profile_analysis = PREPrediction(proteinStructure, lab, plotting_delta=0, replicas=1,
                                     output_prefix='PREs100/1_1000-res',
                                     save_file='PREs100/save-1_1000-res-{}.pkl'.format(lab),
                                     tau_c=1e-9, tau_t=10e-12,
                                     selection='N', optimize=False, idp=True, wh=800)

    proteinStructure = MDAnalysis.Universe('/home/adria/data/asyn/pre/asyn100k/frames/frame_0000000.pdb',
                                           'PREs100/traj2.xtc')
    profile_analysis = PREPrediction(proteinStructure, lab, plotting_delta=0, replicas=1,
                                     output_prefix='PREs100/2_1000-res',
                                     save_file='PREs100/save-2_1000-res-{}.pkl'.format(lab),
                                     tau_c=1e-9, tau_t=10e-12,
                                     selection='N', optimize=False, idp=True, wh=800)

    # Here we cannot return profile_analysis because it can not be picked (and this gives an error)
    return None
#
#
# # n_procs = 12
# # p = Pool(n_procs)
t0 = time.time()
for label in label_list:
    run_PREPrediction(label)
# # p_out = p.map(run_PREPrediction, [ label for label in label_list])
print('Timing {:.3f}'.format(time.time()-t0))
