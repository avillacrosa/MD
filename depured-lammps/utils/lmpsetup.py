import definitions
import scipy.constants as cnt
import numpy as np
from collections import OrderedDict
import shutil
import os
import decimal
from string import Template


class LMPSetup:
    def __init__(self, out_dir, seq):
        self.o_wd = out_dir
        self.p_wd = self.o_wd.replace('/oliba', '')
        self.sequence = seq

        self.temperature = 300
        self.ionic_strength = 100e-3
        self.debye_wv = 1/self.debye_length()

        self.seq_charge = None
        self.residue_dict = definitions.residues
        self.key_ordering = list(self.residue_dict.keys())

        self.hps_epsilon = 0.2
        self.hps_pairs = None

        self.topo_file_dict = {}
        self.lmp_file_dict = {}
        self.qsub_file_dict = {}

        self.job_name = 'hps'
        self.processors = 8

    def del_missing_aas(self):
        missing, pos = [], []
        for key in self.residue_dict:
            if key not in self.sequence:
                # if key != 'L' and key != 'I':
                pos.append(self.residue_dict[key]["id"])
                missing.append(key)

        for m in missing:
            del self.residue_dict[m]

        pos.reverse()

        for p in pos:
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] > p:
                    self.residue_dict[key]["id"] += -1
        self.key_ordering = list(self.residue_dict.keys())

    def lammps_ordering(self):
        id = 1
        done_aas = []
        for aa in self.sequence:
            if aa not in done_aas:
                self.residue_dict[aa]["id"] = id
                done_aas.append(aa)
                id += 1
        ordered_keys = sorted(self.residue_dict, key=lambda x: (self.residue_dict[x]['id']))
        self.key_ordering = ordered_keys

    def get_charge_seq(self):
        charged_plus = []
        charged_minus = []
        for i, aa in enumerate(self.sequence):
            if self.residue_dict[aa]["q"] < 0:
                charged_minus.append(i)
            if self.residue_dict[aa]["q"] > 0:
                charged_plus.append(i)
        return charged_plus, charged_minus

    def get_hps_params(self):
        for key in self.residue_dict:
            for lam_key in definitions.lambdas:
                if self.residue_dict[key]["name"] == lam_key:
                    self.residue_dict[key]["lambda"] = definitions.lambdas[lam_key]
            for sig_key in definitions.sigmas:
                if self.residue_dict[key]["name"] == sig_key:
                    self.residue_dict[key]["sigma"] = definitions.sigmas[sig_key]

    def get_hps_pairs(self, from_file=None):
        lines = ['pair_coeff          *       *       0.000000   0.000    0.000000   0.000   0.000\n']

        if from_file:
            lambda_gen = np.genfromtxt(from_file)
            count = 0
        print(len(self.key_ordering))
        print(self.key_ordering)
        for i in range(len(self.key_ordering)):
            for j in range(i, len(self.key_ordering)):
                res_i = self.residue_dict[self.key_ordering[i]]
                res_j = self.residue_dict[self.key_ordering[j]]
                lambda_ij = (res_i["lambda"] + res_j["lambda"])/2
                sigma_ij = (res_i["sigma"] + res_j["sigma"])/2
                if from_file:
                    lambda_ij = lambda_gen[count]
                    count += 1
                if res_i["q"] != 0 and res_j["q"] != 0:
                    cutoff = 35.00
                else:
                    cutoff = 0.0
                line = 'pair_coeff         {:2d}      {:2d}       {:.6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                    i + 1, j + 1, self.hps_epsilon, sigma_ij, lambda_ij, 3 * sigma_ij, cutoff)
                lines.append(line)
            self.hps_pairs = lines

        self.hps_pairs = lines

    def write_hps_files(self, output_dir='default'):

        if output_dir == 'default':
            output_dir = self.o_wd

        self._generate_lmp_input()
        self._generate_qsub()
        self._generate_topo_input()

        topo_temp_file = open('../templates/topo_template.data')
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(self.topo_file_dict)

        lmp_temp_file = open('../templates/input_template.lmp')
        lmp_template = Template(lmp_temp_file.read())
        lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)

        qsub_temp_file = open('../templates/qsub_template.tmp')
        qsub_template = Template(qsub_temp_file.read())
        qsub_subst = qsub_template.safe_substitute(self.qsub_file_dict)

        with open(f'../default_output/hps.data', 'tw') as fileout:
            fileout.write(topo_subst)
        with open(f'../default_output/{self.job_name}.qsub', 'tw') as fileout:
            fileout.write(qsub_subst)
        with open(f'../default_output/hps.lmp', 'tw') as fileout:
            fileout.write(lmp_subst)

        if output_dir is not None:
            shutil.copyfile(f'../default_output/hps.data', os.path.join(output_dir, 'data.data'))
            shutil.copyfile(f'../default_output/{self.job_name}.qsub', os.path.join(output_dir, f'{self.job_name}.qsub'))
            shutil.copyfile(f'../default_output/hps.lmp', os.path.join(output_dir, 'lmp.lmp'))

    def _generate_lmp_input(self):
        if self.hps_pairs is None:
            self.get_hps_pairs()
        self.lmp_file_dict["pair_coeff"] = ''.join(self.hps_pairs)
        self.lmp_file_dict["debye"] = round(self.debye_wv*10**-10, 1)
        self.lmp_file_dict["v_seed"] = 494211
        self.lmp_file_dict["langevin_seed"] = 451618
        self.lmp_file_dict["temp"] = self.temperature

    def _generate_topo_input(self, nchains=1):
        masses = []
        for i in range(1, len(self.residue_dict)+1):
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] == i:
                    masses.append(f'           {i:2d}  {self.residue_dict[key]["mass"]} \n')

        atoms, bonds, coords = [], [], []
        k = 1

        for chain in range(1, nchains + 1):
            coords = [-240., -240 + chain * 20, -240]
            for aa in self.sequence:
                coords[0] += definitions.bond_length
                atoms.append(f'       {k :3d}          {chain}    '
                             f'      {self.residue_dict[aa]["id"]:2d}   '
                             f'    {self.residue_dict[aa]["q"]: .2f}'
                             f'    {coords[0]: .3f}'
                             f'    {coords[1]: .3f}'
                             f'    {coords[2]: .3f} \n')
                if k != chain * (len(self.sequence)):
                    bonds.append(f'       {k:3d}       1       {k:3d}       {k + 1:3d}\n')
                k += 1

        self.topo_file_dict["natoms"] = nchains * len(self.sequence)
        self.topo_file_dict["nbonds"] = nchains * (len(self.sequence) - 1)
        self.topo_file_dict["atom_types"] = len(self.residue_dict)
        self.topo_file_dict["masses"] = ''.join(masses)
        self.topo_file_dict["atoms"] = ''.join(atoms)
        self.topo_file_dict["bonds"] = ''.join(bonds)

    def _generate_qsub(self):
        self.qsub_file_dict["work_dir"] = self.p_wd
        self.qsub_file_dict["command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -in {input}"
        self.qsub_file_dict["np"] = self.processors
        self.qsub_file_dict["jobname"] = self.job_name

    def debye_length(self):
        kappa = 1 / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro / (cnt.epsilon_0 * 80 * cnt.Boltzmann * self.temperature)) * cnt.e)
        return kappa

    def I_from_debye(kappas, eps_rel=80, T=300, from_angst=False):
        Is = []
        for kappa in kappas:
            kappa = 1 / kappa
            if from_angst:
                kappa = kappa * 10 ** (-10)
            I = cnt.epsilon_0 * eps_rel * cnt.Boltzmann * T / (
                        kappa * kappa * cnt.e * cnt.e * 2 * 10 ** 3 * cnt.Avogadro)
            Is.append(I)
        return Is

    def set_sequence(self, seq):
        self.sequence = seq
