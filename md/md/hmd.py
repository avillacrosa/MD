import glob
import os
import re
import mdtraj as md
import math
import numpy as np
import definitions
import analysis
from scipy.optimize import curve_fit


class HMD(analysis.Analysis):
    def __init__(self, md_dir, low_mem=False, **kwargs):
        self.md_dir = md_dir
        self.in_files = self._get_hmd_files()
        self.residue_dict = dict(definitions.residues)

        self.low_mem = low_mem
        self.equil_frames = kwargs.get('equil_frames', 300)

        self.slab = self._is_slab()
        self.data = self.get_data()
        self.box = self.get_box()
        self.temperatures = self.get_temperatures()
        self.sequence = self._get_sequence()
        self.chain_atoms = len(self.sequence)
        self.chains = self.get_n_chains()
        self.masses = self._get_masses()
        self.structures = self.get_structures()

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
            log_data.append(np.genfromtxt(logfile))
        return np.array(log_data)

    def get_box(self):
        dict = {}
        if not self.slab:
            for line in self.in_files[0]:
                if "l = " in line:
                    dict["x"] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])) * 10.
                    dict["y"] = dict["x"]
                    dict["z"] = dict["x"]
                    return dict
        else:
            # Todo : strong assumption
            for line in self.in_files[0]:
                if "h_c" in line and "linear_interp" in line:
                    dict["x"] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))*10.
                    dict["y"] = dict["x"]
                if "h_e" in line and "linear_interp" in line:
                    dict["z"] = abs(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))*10.
            return dict
        if "x" not in dict:
            dict["x"] = self.data[:, :, -3].min(axis=1)
        if "y" not in dict:
            dict["y"] = self.data[:, :, -2].min(axis=1)
        if "z" not in dict:
            dict["z"] = self.data[:, :, -1].min(axis=1)
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

    def get_structures(self):
        dcds = glob.glob(os.path.join(self.md_dir, '*.dcd'))
        if len(dcds) > 1:
            dcds = sorted(dcds, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        structures = []
        if os.path.exists(os.path.join(self.md_dir, 'topo.pdb')):
            topo = os.path.join(self.md_dir, 'topo.pdb')
        else:
            return SystemError("Topology not found, but only looked for a file named topo.pdb")
        for dcd in dcds:
            tr = md.load(dcd, top=topo)
            tr.xyz = tr.xyz * 10. * 10.
            structures.append(tr[self.equil_frames:])
        return structures

