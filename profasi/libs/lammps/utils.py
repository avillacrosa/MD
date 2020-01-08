"""
Objects here are system independent. They do not need the definitions in the data directory
"""
import numpy as np
from subprocess import run, PIPE
import pandas as pd
#import xarray as xr
from os.path import join
import io

# Missing
def get_lammps_energies(source, from_file_name=False):
    """
    Get the energies of a re-run LAMMPS output.
    Input: source: a str with the output or a file name with the output.
    from_file_name: If true, source is a string corresponding to the name of a file
    """
    #Choose function to open depending on whether 'source' is a StringIO or a file name.
    if from_file_name:
        openfunc = open
    else:
        openfunc = io.StringIO

    with openfunc(source) as filein:
        reading = False
        energies = []
        for line in filein:
            if 'Loop time of' in line: #end of energy output
                reading = False
           
            if reading:
                if line.startswith('WARNING'): #this warning does not affect the final energy
                    continue
                else:
                    ene = line.split()[1]
                    energies.append(ene) 
           
            if 'Step PotEng KinEng Temp' in line: #start of energy output
                reading = True 
    energies = np.array(energies, dtype=np.float)
    return energies


# Got
def calc_weights(new_E, old_E):
    """
    Calculate new weights from new energies.
    Input: new_E, array of new energies in kT units.    
    Input: old_E, array of old energies in kT units (that generated current structures).    
    Return: array of w (normalized)
    """
    delta_E = new_E-old_E
    w = np.exp(-delta_E)
    w /= w.sum()
    return w

# Missing
def run_lammps():
    """
    Run a LAMMPS job

    """
    filename = 'hps.data'
    command = '/home/adria/local/lammps/bin/lmp -in ' + filename
    out = run(command.split(), stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return out


# Got
def read_exp_data(basedir, labels, filename):
    """
    Reads experimental PREs (data in one single column)
    input: basedir: directory where the PREs .dat files are located. 
           Their names should be "{}{}.dat".format(filename,label)
    labels: lists of labels (of type int)
    return: a pandas DataFrame with the labels as columns and the residues as indices
    """
    data = {}
    for lab in labels:
        name = join(basedir, '{}{}.dat'.format(filename,lab))
        data[lab]=pd.read_csv(name,
                              header=None,
                              names=[lab])
        data[lab].index += 1 #start residue number from 1 (bio-like)

    #Merge all data frames
    data = pd.concat([data[lab] for lab in data], axis=1)
    data.rename_axis('residue', inplace=True)
    data.rename_axis('label', axis='columns',inplace=True)
    return data


# Got
def read_calc_data(basedir, labels, filename):
    """
    Reads calculated PREs (data files have 2 columns as output from DEERpredict)
    input: basedir: directory where the PREs .dat files are located. 
           Their names should be "{}{}.dat".format(filename,label)
    labels: lists of labels (of type int)
    return: a pandas DataFrame with the labels as columns and the residues as indices
    """
    data = {}
    for lab in labels:
        name = join(basedir, '{}{}.dat'.format(filename,lab))
        data[lab]=pd.read_csv(name,
                              header=None, sep='\s+', usecols=[1	],
                              names=[lab])
        data[lab].index += 1 #start residue number from 1 (bio-like)

    #Merge all data frames
    data = pd.concat([data[lab] for lab in data], axis=1)
    data.rename_axis('residue', inplace=True)
    data.rename_axis('label', axis='columns',inplace=True)
    return data

# Got
def fit(data_exp, data_calc, label=None):
    """
    Return the Mean Square Deviation between experimental and calculated PRE
    input: data_exp: a pandas dataframe with the experimental PRE
    input: data_calc: a pandas dataframe with the calculated PRE
    input: label, if not None, calculate only for a givel label
    """
    if label is not None:
        rmsd = ((data_calc[label]-data_exp[label])**2).mean()
    else:
        rmsd = ((data_calc-data_exp)**2).mean().mean()
    return rmsd

# Got
def _run_pulchra(t):
    """
    This is an auxiliary function of reconstruct pulchra. It has to be here, and not inside pulchra
    because it raises an error if it is there (it cannot be pickled by Pool).
    See details here:
    https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
    An alternative would be to use pathos.multiprocessing.
    """
    import mdtraj as md
    import os
    i, frame=t
    name = '/dev/shm/ramon/input-p-{}.pdb'.format(i)
    frame.save(name)
    #call pulchra
    call_line ='/lindorffgrp-isilon/wyong/CACBGo_TIM910GGC/RUN/SAXStools/pulchra304/bin/linux/pulchra {}'.format(name)
    call_line = call_line.split()
    run(call_line)
    outname = '/dev/shm/ramon/input-p-{}.rebuilt.pdb'.format(i)
    trajtemp = md.load(outname)
    os.remove(name)
    os.remove(outname)
    if i==0:
        return trajtemp
    else:
        return trajtemp.xyz

# Got
def reconstruct_pulchra(name, n_procs=8):
    """
    Reconstruct an all atom trajectory from a calpha trajectory.
    Input: name: basename of the trajectory .xtc (or .dcd) and the corresponding pdb file.
           That is, the trajectory and the pdb should be called '{}.xtc'.format(name) and
           '{}.pdb'.format(name)
           n_procs: number of processors to use in parallel.
    Return: A reconstructed mdtraj trajectory.
    """
    import mdtraj as md
    import os
    from multiprocessing import Pool
    import time


    trajin = md.load('{}.xtc'.format(name), top='{}.pdb'.format(name))
    #Superpose frames to avoid large coordinates arising from diffusion out of the box
    trajin = trajin.superpose(trajin, frame=0)    
    
    try:
        os.mkdir('/dev/shm/ramon')
    except FileExistsError:
        pass
    print("Starting reconstruction of {} frames".format(trajin.n_frames))
    t0 = time.time()
    p = Pool(n_procs)

    p_out = p.map(_run_pulchra, [(i, frame) for i, frame in enumerate(trajin)])

    trajout = p_out[0]
    trajout.xyz = np.vstack([p_out[0].xyz, *p_out[1:]])
    #Manually define time (otherwise saving fails)
    trajout.time = np.arange(trajout.n_frames)*trajin.timestep
    print("Done! Execution time: {:.1f} s".format(time.time()-t0))
    return trajout
    #Save final trajectory
    #trajout.save('{}_all.xtc'.format(name))
    #trajout[0].save('{}_all.pdb'.format(name))



