import os
import numpy as np
from pathlib import Path


def getPDBfromLAMMPS(lammps_data):
    lammps2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'
    os.system(lammps2pdb+' '+lammps_data)
    fileout = lammps_data+'_trj.pdb'
    return fileout


def getLMPdirs(dir_paths):
    dirs = []
    for dir in dir_paths:
        for filename in Path(dir).rglob('*lmp'):
            dirs.append(os.path.dirname(filename))
    dirs.sort()
    return dirs


def getLMPthermo(dir):
    lmp_dirs = getLMPdirs([dir])
    data, kappas = [], []
    for dir in lmp_dirs:
        log_lmp = open(os.path.join(dir, 'log.lammps'), 'r')
        lines = log_lmp.readlines()
        data_start = 0
        data_end = 0
        for i, line in enumerate(lines):
            if "Step" in line:
                data_start = i+1
            if "Loop" in line:
                data_end = i
            if data_end and data_start != 0:
                break
        if data_end == 0:
            data_end = len(lines)
        kappas.append(os.path.basename(dir))
        data.append(np.loadtxt(os.path.join(dir, 'log.lammps'), skiprows=data_start, max_rows=data_end - data_start))
    data = np.array(data)
    # kappas = np.array(kappas, dtype='float')
    kappas = [1]
    return data, kappas
