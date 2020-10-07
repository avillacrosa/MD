import numpy as np
import pandas as pd
import re
import mdtraj as md
import glob
import os
from os.path import join
from pathlib import Path
from subprocess import run
import MDAnalysis
from DEERpredict.PREPrediction5 import PREPrediction

prf_dict = {
    "rank": {"name": "Replica", "unit": ""},
    "time": {"name": "t", "unit": "(cyc)"},
    "temperature": {"name": "T", "unit": "K"},
    "etot": {"name": "E", "unit": "prf"},
    "exvol": {"name": "Excluded Volume Energy", "unit": "prf"},
    "locexvol": {"name": "Local Excluded Volume Energy", "unit": "prf"},
    "bias": {"name": "Local Electrostatic Energy", "unit": "prf"},
    "torsionterm": {"name": "Torsion Angle Potential Energy", "unit": "prf"},
    "hbmm": {"name": "Backbone-Backbone H-Bond Energy", "unit": "prf"},
    "hbms": {"name": "Backbone-Sidechain H-Bond Energy", "unit": "prf"},
    "hydrophobicity": {"name": "Hydrophobic Energy (non-polar SC)", "unit": "prf"},
    "chargedscinteraction": {"name": "Energy for Charged Sidechains", "unit": "prf"},
    "helixcontent": {"name": "Helix Content", "unit": ""},
    "betastrandcontent": {"name": "Beta Content", "unit": ""},
    "rg": {"name": "Rg", "unit": "nm"},
    "E": {"name": "Energy components", "unit": "prf"}
}


def convert_to_kelvin(dir, rt=None):
    """
    Converts rt data from PROFASI from prf units to Kelvin. Returns a list of the Kelvin Temperatures
    """
    filename = join(dir, 'results', 'temperature.info')
    temperatures = np.genfromtxt(filename)
    if len(temperatures.shape) == 1:
        temperatures = np.array([temperatures])
    temp = temperatures[:, [0, 2]]
    if rt is not None:
        for i, temp_index in enumerate(rt[:, 2]):
            rt[i][2] = temp[int(rt[i][2])][1]
    # No need to return rt due to python not copying arrays on assigns
    return temp


def get_obs_names(dir, energies=False):
    """
    Get observable names from PROFASI's rtkey file
    """
    observables = []
    rt_keys = open(join(dir, 'results', 'rtkey'), "r").readlines()
    rt_keys.pop(0)
    for key in rt_keys:
        o = re.findall(r'([a-zA-Z]+)', key)[0].lower()
        observables.append(o)
    if energies:
        observables = observables[4:12]
    return observables


def get_max_temperature(dir):
    """
    Self explicatory
    """
    rt = np.genfromtxt(join(dir, 'results', 'rt'))
    return int(np.max(rt[:, 2]))


def get_all_profasi_dirs(dir_paths):
    """
    Gets and returns all directories where a settings.cnf file is found under the dir_paths list
    """
    dirs = []
    for dir in dir_paths:
        for filename in Path(dir).rglob('*.cnf'):
            dirs.append(os.path.dirname(filename))
    return dirs


def charge_pattern(seq, zeros=False):
    """
    Find the charge patterning of a protein sequence in integers (/e). Returns such value and the position relative
    to the full sequence
    """
    charge_dict = {'D': -1, 'E': -1, 'R': +1, 'K': +1}
    qseq = []
    for i, d in enumerate(seq):
        if charge_dict.get(d, None) is not None:
            qseq.append([charge_dict[d], i + 1])
        elif zeros:
            qseq.append([0, i + 1])
    return np.array(qseq)


# TODO : Complete function, maybe not possible...
def run_all_profasi_dirs(dir_paths):
    dirs = get_all_profasi_dirs(dir_paths)


def build_xtc_traj(frames_dir, outdir):
    """
    Build xtc type of trajectory from frames (PDB's)
    """
    paths = join(frames_dir, "frame_*pdb")
    filelist = glob.glob(paths)
    traj = md.load(filelist)
    trj1 = traj[::2]
    trj2 = traj[1::2]
    traj.save(join(outdir, "full_traj.xtc"))
    trj1.save(join(outdir, "even_traj.xtc"))
    trj2.save(join(outdir, "odd_traj.xtc"))
    return traj


def calc_pre(traj, topology, labels, output):
    for lab in labels:
        proteinStructure = MDAnalysis.Universe(topology,
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


def read_pre_alphas(basename, label):
    name = join(basename, 'alphas-{}-000.dat'.format(label))
    if os.path.isfile(name):
        alphas = np.genfromtxt(name)
        return alphas
    else:
        return np.ones(8)


def get_profasi_energies(source, max_frames):
    """
    Get the energies of a re-run PROFASI rt output.
    Input: source: a str with the output or a file name with the output.
    from_file_name: If true, source is a string corresponding to the name of a file
    """
    energies = np.genfromtxt(source)
    energies = energies[energies[:, 1] > 10000, :]
    energies = energies[0:max_frames, :]
    energies = energies[:, 4: 12]
    return energies


def calc_weights(new_E, old_E):
    """
    Calculate new weights from new energies.
    Input: new_E, array of new energies in kT units.
    Input: old_E, array of old energies in kT units (that generated current structures).
    Return: array of w (normalized)
    """
    delta_E = new_E - old_E
    w = np.exp(-delta_E)
    w /= w.sum()
    return w


def read_pre_exp_data(basedir, labels, filename):
    """
    Reads experimental PREs (data in one single column)
    input: basedir: directory where the PREs .dat files are located.
           Their names should be "{}{}.dat".format(filename,label)
    labels: lists of labels (of type int)
    return: a pandas DataFrame with the labels as columns and the residues as indices
    """
    data = {}
    for lab in labels:
        name = join(basedir, '{}{}.dat'.format(filename, lab))
        data[lab] = pd.read_csv(name,
                                header=None,
                                names=[lab])
        data[lab].index += 1  # start residue number from 1 (bio-like)

    # Merge all data frames
    data = pd.concat([data[lab] for lab in data], axis=1)
    data.rename_axis('residue', inplace=True)
    data.rename_axis('label', axis='columns', inplace=True)
    return data


def read_calc_pre(basedir, labels, filename):
    """
    Reads calculated PREs (data files have 2 columns as output from DEERpredict)
    input: basedir: directory where the PREs .dat files are located.
           Their names should be "{}{}.dat".format(filename,label)
    labels: lists of labels (of type int)
    return: a pandas DataFrame with the labels as columns and the residues as indices
    """
    data = {}
    for lab in labels:
        name = join(basedir, '{}{}.dat'.format(filename, lab))
        data[lab] = pd.read_csv(name,
                                header=None, sep='\s+', usecols=[1],
                                names=[lab])
        data[lab].index += 1  # start residue number from 1 (bio-like)

    # Merge all data frames
    data = pd.concat([data[lab] for lab in data], axis=1)
    data.rename_axis('residue', inplace=True)
    data.rename_axis('label', axis='columns', inplace=True)
    return data


def fit_pre(data_exp, data_calc, label=None):
    """
    Return the Mean Square Deviation between experimental and calculated PRE
    input: data_exp: a pandas dataframe with the experimental PRE
    input: data_calc: a pandas dataframe with the calculated PRE
    input: label, if not None, calculate only for a givel label
    """
    if label is not None:
        rmsd = ((data_calc[label] - data_exp[label]) ** 2).mean()
    else:
        rmsd = ((data_calc - data_exp) ** 2).mean().mean()
    return rmsd


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
    i, frame = t
    name = '/dev/shm/yihsuan/input-p-{}.pdb'.format(i)
    frame.save(name)
    # call pulchra
    call_line = '/lindorffgrp-isilon/wyong/CACBGo_TIM910GGC/RUN/SAXStools/pulchra304/bin/linux/pulchra {}'.format(name)
    call_line = call_line.split()
    run(call_line)
    outname = '/dev/shm/yihsuan/input-p-{}.rebuilt.pdb'.format(i)
    trajtemp = md.load(outname)
    os.remove(name)
    os.remove(outname)
    if i == 0:
        return trajtemp
    else:
        return trajtemp.xyz


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
    # Superpose frames to avoid large coordinates arising from diffusion out of the box
    trajin = trajin.superpose(trajin, frame=0)

    try:
        os.mkdir('/dev/shm/yihsuan')
    except FileExistsError:
        pass
    print("Starting reconstruction of {} frames".format(trajin.n_frames))
    t0 = time.time()
    p = Pool(n_procs)

    p_out = p.map(_run_pulchra, [(i, frame) for i, frame in enumerate(trajin)])

    trajout = p_out[0]
    trajout.xyz = np.vstack([p_out[0].xyz, *p_out[1:]])
    # Manually define time (otherwise saving fails)
    trajout.time = np.arange(trajout.n_frames) * trajin.timestep
    print("Done! Execution time: {:.1f} s".format(time.time() - t0))
    return trajout
