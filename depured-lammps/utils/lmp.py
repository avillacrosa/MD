import numpy as np
import os
import definitions
import glob
import multiprocessing as mp
from subprocess import run, PIPE
import pathlib


class LMP:
    def __init__(self, oliba_wd, temper=False):
        self.sequence = None
        self.res_dict = definitions.residues

        self.temper = temper

        self.o_wd = oliba_wd
        self.p_wd = oliba_wd.replace('/perdiux', '')

        self.lmp = '/home/adria/local/lammps/bin/lmp'
        self.lmp_drs = self.get_lmp_dirs()
        self.n_drs = len(self.lmp_drs)
        # if temper:
        #     self.data = self.get_lmp_temper_data()
        # else:
        # self.data = self.get_lmp_data()
        self.data = None
        self.lmp2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'

    def get_lmp_dirs(self):
        dirs = []
        for filename in pathlib.Path(self.o_wd).rglob('*lmp'):
            dirs.append(os.path.dirname(filename))
        dirs.sort()
        return dirs

    def make_initial_frame(self, dirs=None):
        pdb_paths = []
        if dirs is None:
            dirs = self.lmp_drs
        for dir in dirs:
            files = glob.glob(os.path.join(dir, '*.data'))
            file = os.path.basename(files[0])
            file = file.replace('.data', '')
            lammps2pdb = self.lmp2pdb
            os.system(lammps2pdb + ' ' + os.path.join(dir, file))
            fileout = os.path.join(dir, file) + '_trj.pdb'
            pdb_paths.append(fileout)
        return pdb_paths

    def get_lmp_data(self):
        data = []
        for d in self.lmp_drs:
            log_lmp = open(os.path.join(d, 'log.lammps'), 'r')
            lines = log_lmp.readlines()
            data_start = 0
            data_end = 0
            for i, line in enumerate(lines):
                if "Step" in line:
                    data_start = i + 1
                if "Loop" in line:
                    data_end = i
                if data_end and data_start != 0:
                    break
            if data_end == 0:
                data_end = len(lines)
            data.append(
                np.loadtxt(os.path.join(d, 'log.lammps'), skiprows=data_start, max_rows=data_end - data_start))
        data = np.array(data)
        return data

    def get_lmp_temper_data(self, lmp_directories=None):
        data = []
        for d in self.lmp_drs:
            if lmp_directories is None:
                lmps = glob.glob(os.path.join(d, "log.lammps.*"))
            else:
                lmps = lmp_directories
            for lmp in lmps:
                log_lmp = open(os.path.join(d, lmp), 'r')
                lines = log_lmp.readlines()
                data_start = 0
                data_end = 0
                for i, line in enumerate(lines):
                    if "Step" in line:
                        data_start = i + 1
                    if "Loop" in line:
                        data_end = i
                    if data_end and data_start != 0:
                        break
                if data_end == 0:
                    data_end = len(lines)
                data.append(
                    np.loadtxt(os.path.join(d, lmp), skiprows=data_start, max_rows=data_end - data_start))
        data = np.array(data)
        return data

    def set_sequence(self, seq, from_file=False):
        if from_file:
            print("TODO")
        self.sequence = seq

    def decode_seq_from_hps(self):
        for lmp_dir in self.lmp_drs:
            data_paths = glob.glob(os.path.join(lmp_dir, "*.data"))
            data_path = data_paths[0]
            with open(data_path, 'r') as data_file:
                lines = data_file.readlines()
                mass_range, atom_range = [0, 0], [0, 0]
                reading_masses, reading_atoms = False, False
                # TODO Not too smart
                for i, line in enumerate(lines):

                    if "Atoms" in line:
                        reading_masses = False
                        # TODO Funny how if I dont substract 1 it doesnt work for RAMONS case but does for mine
                        mass_range[1] = i - 1

                    if reading_masses and line != '\n' and mass_range[0] == 0:
                        mass_range[0] = i

                    if "Masses" in line:
                        reading_masses = True

                    if "Bonds" in line:
                        reading_atoms = False
                        atom_range[1] = i - 1

                    if reading_atoms and line != '\n' and atom_range[0] == 0:
                        atom_range[0] = i

                    if "Atoms" in line:
                        reading_atoms = True
            print(mass_range, lines[mass_range[1]])
            masses = np.loadtxt(data_path, skiprows=mass_range[0], max_rows=mass_range[1] - mass_range[0])
            atoms = np.loadtxt(data_path, skiprows=atom_range[0], max_rows=atom_range[1] - atom_range[0])
            mass_dict = {}
            for mass in masses:
                for res in self.res_dict:
                    if self.res_dict[res]["mass"] == mass[1]:
                        mass_dict[int(mass[0])] = res
                        break
            seq = []
            for atom in atoms:
                seq.append(mass_dict[atom[2]])
            seq_str = ''.join(seq)
            print(
                "Finding used sequence. Note that Isoleucine and Leucine have the same exact HPS parameters. Therefore, they are not distinguishable.")
            return seq_str

    def run(self, file, n_cores=1):
        f_name = os.path.join(self.o_wd, file)
        if n_cores > mp.cpu_count():
            raise SystemExit(f'Desired number of cores exceed available cores on this machine ({mp.cpu_count()})')
        if n_cores > 1:
            command = f'mpirun -n {n_cores} {self.lmp} -in {f_name}'
        elif n_cores == 1:
            command = f'{self.lmp} -in {f_name}'
        else:
            raise SystemExit('Invalid core number')
        old_wd = os.getcwd()
        os.chdir(self.o_wd)
        out = run(command.split(), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        os.chdir(old_wd)
        return out

# TODO Handle Temperatures or Ionic strengths...
