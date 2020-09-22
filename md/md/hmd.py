import glob
import os
import re
import mdtraj as md
import math
import numpy as np
import definitions
import analysis
import pathlib
from scipy.optimize import curve_fit


class HMD(analysis.Analysis):
    def __init__(self, md_dir, low_mem=False, **kwargs):
        self.md_dir = md_dir
        self.in_files = self._get_hmd_files()
        self.residue_dict = dict(definitions.residues)

        self.low_mem = low_mem
        self.every_frames, self.last_frames = kwargs.get('every', 10),  kwargs.get('total', None)
        self.equil_frames = 300

        self.slab = self._is_slab()
        self.data = self.get_data()
        self.box = self.get_box()
        self.get_sigmas()
        self.temperatures = self.get_temperatures()
        self.sequence = self._get_sequence()
        self.chain_atoms = len(self.sequence)
        self.chains = self.get_n_chains()
        self.masses = self._get_masses()
        self.structures = self.get_structures(every=self.every_frames, total=self.last_frames)
        self.protein = self.get_protein_from_sequence()
        super().__init__(framework='HMD')

    def _get_hmd_files(self):
        hmd = glob.glob(os.path.join(self.md_dir, '*.py'))
        hmd.sort()
        hmd_files = []
        for h_file in hmd:
            with open(os.path.join(self.md_dir, h_file), 'r') as lmp_file:
                hmd_files.append(lmp_file.readlines())
        return hmd_files

    def _get_masses(self):
        m = []
        for chain in range(self.chains):
            for aa in self.sequence:
                m.append(self.residue_dict[aa]["mass"])
        return np.array(m)

    def _is_slab(self):
        for file in self.in_files:
            for line in file:
                if "box_resize" in line:
                    return True
        return False

    def get_data(self):
        logs = glob.glob(os.path.join(self.md_dir, 'log*log'))
        logs.sort()
        log_data = []
        for logfile in logs:
            log_data.append(np.genfromtxt(logfile, skip_header=2))
        return np.array(log_data)

    def get_box(self):
        dict = {}
        if not self.slab:
            for line in self.in_files[0]:
                if "l = " in line:
                    dict["x"] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])) * 10.
                    dict["y"] = dict["x"]
                    dict["z"] = dict["x"]
        else:
            # Todo : strong assumption
            for line in self.in_files[0]:
                if "h_c" in line and "linear_interp" in line:
                    dict["x"] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))*10.
                    dict["y"] = dict["x"]
                if "h_e" in line and "linear_interp" in line:
                    dict["z"] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))*10.
        if "x" not in dict:
            dict["x"] = self.data[0][:, -3].min(axis=0)*10.
        if "y" not in dict:
            dict["y"] = self.data[0][:, -2].min(axis=0)*10.
        if "z" not in dict:
            dict["z"] = self.data[0][:, -1].min(axis=0)*10.
        return dict

    def _get_sequence(self):
        sequence = None
        for file in self.in_files:
            for line in file:
                if 'sequence' in line:
                    sequence = line.split()[-1].replace("\"", '')
                    break
        return sequence

    def get_n_chains(self):
        """
        Get the number of chains of the simulation
        :return: [int nchains, int atoms_per_chain]
        """
        n_chains = None
        for file in self.in_files:
            for line in file:
                if 'chains' in line:
                    n_chains = int(re.findall(r'\d+', line)[0])
                    break
        return n_chains

    def get_temperatures(self):
        temperatures = []
        for h_file in self.in_files:
            for line in h_file:
                if "temperature" in line:
                    temperatures.append(re.findall(r'\d+', line)[0])
                    break
        return np.array(temperatures, dtype='float')

    def get_structures(self, every=1, total=None):
        dcds = glob.glob(os.path.join(self.md_dir, '*.dcd'))
        if len(dcds) > 1:
            dcds = sorted(dcds, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        structures = []
        if os.path.exists(os.path.join(self.md_dir, 'topo.pdb')):
            topo = os.path.join(self.md_dir, 'topo.pdb')
        else:
            return SystemError("Topology not found, but only looked for a file named topo.pdb")
        n_frames = []
        for dcd in dcds:
            tr = md.load(dcd, top=topo)
            tr.xyz = tr.xyz * 10. * 10.
            tr = tr[self.equil_frames::self.every_frames]
            tr = tr[:total]
            structures.append(tr)
            n_frames.append(tr.n_frames)
        print(f"> Taking frames every {every} for a total of {n_frames} to avoid strong correlations")
        return structures

    def save_frames(self, name, frames=100):
        movie_dir = '/home/adria/Movies/HMD'
        movie_dir = os.path.join(movie_dir, name)
        pathlib.Path(movie_dir).mkdir(parents=True, exist_ok=True)
        for T, struct in enumerate(self.structures):
            evr = int(struct.n_frames/100)
            struct = struct[::evr]
            struct = struct[:frames]
            struct.save_netcdf(os.path.join(movie_dir, f'{self.protein}_x{self.chains}-T{self.temperatures[T]:.0f}.netcdf'))
            self.save_lammpstrj(struct, os.path.join(movie_dir, f'{self.protein}_x{self.chains}-T{self.temperatures[T]:.0f}.lammpstrj'))
        print(f"Saving at {movie_dir}")

    def get_seq_from_hmd(self):
        f = self.in_files[0]
        for line in f:
            if "sequence = " in line:
                sequence = line.split()[-1]
                sequence = str(sequence)
        sequence = sequence.replace('\"', '')
        return sequence

    def get_protein_from_sequence(self):
        """
        Search on the sequences directory for a file matching the used sequence in the LMP, obtained by translating data
        :return: Name (string) of the protein
        """
        sequence = self.get_seq_from_hmd()
        stored_seqs = glob.glob(os.path.join(definitions.hps_data_dir, 'sequences/*.seq'))
        for seq_path in stored_seqs:
            with open(seq_path, 'r') as seq_file:
                seq = seq_file.readlines()[0]
                # print(sequence, seq)
                if sequence.replace('L', 'I') == seq.replace('L', 'I'):
                    protein = os.path.basename(seq_path).replace('.seq', '')
                    return protein

    def get_sigmas(self):
        for key in self.residue_dict:
            for sig_key in definitions.sigmas:
                if self.residue_dict[key]["name"] == sig_key:
                    self.residue_dict[key]["sigma"] = definitions.sigmas[sig_key]

    def save_lammpstrj(self, traj, fname, center=False):
        """
        Save a trajectory (it's coordinates) in a custom LAMMPS format, since mdtraj seems to do something weird
        :param traj: mdtraj trajectory
        :param T: temperature index
        :return: Nothing
        """
        save_step = int(self.data[0][1, 0] - self.data[0][0, 0])
        f = ''
        for frame in range(traj.n_frames):
            frame_center = traj.xyz[frame, :, :].mean(axis=0)
            f += f'ITEM: TIMESTEP \n'
            f += f'{frame * save_step} \n'
            f += f'ITEM: NUMBER OF ATOMS \n'
            f += f'{traj.n_atoms} \n'
            f += f'ITEM: BOX BOUNDS pp pp pp \n'
            f += f'-{self.box["x"]/2} {self.box["x"]/2} \n'
            f += f'-{self.box["y"]/2} {self.box["y"]/2} \n'
            f += f'-{self.box["z"]/2} {self.box["z"]/2} \n'
            # for key in self.box:
            #     f += f'{self.box[key][0]} {self.box[key][1]} \n'
            f += f'ITEM: ATOMS id type chain x y z sigma charge \n'
            for c in range(self.chains):
                for i, aa in enumerate(self.sequence):
                    xyz = traj.xyz[frame, i + self.chain_atoms * c, :]
                    xyzf = xyz.copy()
                    if center:
                        xyzf = xyzf - frame_center * 10
                    f += f'{(c + 1) * (i + 1)} {self.residue_dict[aa]["id"]} {c} {xyzf[0]:.5f} {xyzf[1]:.5f} {xyzf[2]:.5f} {self.residue_dict[aa]["sigma"]:.2f} {self.residue_dict[aa]["q"]:.2f} \n'
        with open(fname, 'tw') as lmpfile:
            lmpfile.write(f)