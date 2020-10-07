#! /usr/bin/env python3

# SETUP

import sys
import time
from multiprocessing import Pool
from joblib import Parallel, delayed

# CALCULATE INITAL PREs



label_list = (18,20,24,42,62,76,85,87,90,103,120,140)
 

import MDAnalysis
from DEERpredict.PREPrediction5 import PREPrediction
#
#These 2 work but this one seems to be slower, on a very short tests I did
# But the second one gives pickling errors erratically. 
#
#def run_PREPrediction(lab):
#    proteinStructure = MDAnalysis.Universe('./run1/trajs/A-syn_HPS_0_50000_frames_all.pdb',
#                                       './run1/trajs/A-syn_HPS_0_50000_frames_all.xtc')
#    profile_analysis = PREPrediction(proteinStructure, lab, plotting_delta=0, replicas=1,
#                                     output_prefix='./PREs/0_50000-res',
#                                     save_file='./PREs/save-0_50000-res-{}.pkl'.format(lab),
#                                     tau_c=1e-9, tau_t=10e-12,
#                                     selection='N', optimize=False, idp=True, wh=800)

#    proteinStructure = MDAnalysis.Universe('./run1/trajs/A-syn_HPS_1_50000_frames_all.pdb',
#                                       './run1/trajs/A-syn_HPS_1_50000_frames_all.xtc')
#    profile_analysis = PREPrediction(proteinStructure, lab, plotting_delta=0, replicas=1,
#                                     output_prefix='./PREs/1_50000-res',
#                                     save_file='./PREs/save-1_50000-res-{}.pkl'.format(lab),
#                                     tau_c=1e-9, tau_t=10e-12,
#                                     selection='N', optimize=False, idp=True, wh=800)
#
    #return profile_analysis

#t0=time.time()
#Parallel(n_jobs=len(label_list))(delayed(run_PREPrediction) (label) for label in label_list)
#print('Timing {:.3f}'.format(time.time()-t0))

#So I'l use this one (which is the same used when parallelizing Puchra

def run_PREPrediction(lab):
    import MDAnalysis
    from DEERpredict.PREPrediction5 import PREPrediction
    proteinStructure = MDAnalysis.Universe('./run1/trajs/A-syn_HPS_0_50000_frames_all.pdb',
                                       './run1/trajs/A-syn_HPS_0_50000_frames_all.xtc')
    profile_analysis = PREPrediction(proteinStructure, lab, plotting_delta=0, replicas=1,
                                     output_prefix='./PREs/0_50000-res',
                                     save_file='./PREs/save-0_50000-res-{}.pkl'.format(lab),
                                     tau_c=1e-9, tau_t=10e-12,
                                     selection='N', optimize=False, idp=True, wh=800)

    proteinStructure = MDAnalysis.Universe('./run1/trajs/A-syn_HPS_1_50000_frames_all.pdb',
                                       './run1/trajs/A-syn_HPS_1_50000_frames_all.xtc')
    profile_analysis = PREPrediction(proteinStructure, lab, plotting_delta=0, replicas=1,
                                     output_prefix='./PREs/1_50000-res',
                                     save_file='./PREs/save-1_50000-res-{}.pkl'.format(lab),
                                     tau_c=1e-9, tau_t=10e-12,
                                     selection='N', optimize=False, idp=True, wh=800)

    

    #Here we cannot return profile_analysis because it can not be picked (and this gives an error)
    return None
    
n_procs=12
p = Pool(n_procs)
t0=time.time()
p_out = p.map(run_PREPrediction, [ label for label in label_list])
print('Timing {:.3f}'.format(time.time()-t0))
