import definitions
import lmp
import scipy.constants as cnt
import numpy as np
import multiprocessing as mp
from subprocess import run, PIPE
import shutil
import os
import glob
import mdtraj as md
from string import Template


class LMPSetup(lmp.LMP):
    def __init__(self, seq, **kw):
        super(LMPSetup, self).__init__(**kw)
        self.p_wd = self.oliba_wd.replace('/oliba', '')
        self.sequence = seq

        self.temperature = 300
        self.ionic_strength = 100e-3
        self.debye_wv = 1/self.debye_length()
        self.dt = 10.
        self.t = 10000000
        self.xyz = None

        self.seq_charge = None
        self.residue_dict = dict(definitions.residues)
        self.key_ordering = list(self.residue_dict.keys())

        self.lmp = '/home/adria/local/lammps/bin/lmp'
        self.box_size = 2500
        self.hps_scale = 1.0

        self.hps_epsilon = 0.2
        self.hps_pairs = None

        self.topo_file_dict = {}
        self.lmp_file_dict = {}
        self.qsub_file_dict = {}

        self.v_seed = 494211
        self.langevin_seed = 451618
        self.save = 50000

        self.job_name = 'hps'
        self.processors = 8
        # self.temperatures = np.linspace(150.0, 600.0, self.processors)
        self.temperatures = '150.0 170.0 192.5 217.5 247.5 280.0 320.0 362.5 410.0 467.5 530.0 600.0'
        self.swap_every = 1000

    def del_missing_aas(self):
        missing, pos = [], []
        for key in self.residue_dict:
            if key not in self.sequence:
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
                lambda_ij = lambda_ij * self.hps_scale
                line = 'pair_coeff         {:2d}      {:2d}       {:.6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                    i + 1, j + 1, self.hps_epsilon, sigma_ij, lambda_ij, 3 * sigma_ij, cutoff)
                lines.append(line)
        self.hps_pairs = lines

    def write_hps_files(self, output_dir='default'):
        if output_dir == 'default':
            output_dir = self.oliba_wd

        self._generate_lmp_input()
        self._generate_qsub()
        self._generate_topo_input()

        topo_temp_file = open('../templates/topo_template.data')
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(self.topo_file_dict)

        if self.temper:
            lmp_temp_file = open('../templates/replica/input_template.lmp')
        else:
            lmp_temp_file = open('../templates/general/input_template.lmp')
        lmp_template = Template(lmp_temp_file.read())
        lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)

        qsub_temp_file = open('../templates/qsub_template.tmp')
        qsub_template = Template(qsub_temp_file.read())
        qsub_subst = qsub_template.safe_substitute(self.qsub_file_dict)

        with open(f'../default_output/data.data', 'tw') as fileout:
            fileout.write(topo_subst)
        with open(f'../default_output/{self.job_name}.qsub', 'tw') as fileout:
            fileout.write(qsub_subst)
        with open(f'../default_output/lmp.lmp', 'tw') as fileout:
            fileout.write(lmp_subst)

        if output_dir is not None:
            shutil.copyfile(f'../default_output/data.data', os.path.join(output_dir, 'data.data'))
            shutil.copyfile(f'../default_output/{self.job_name}.qsub', os.path.join(output_dir, f'{self.job_name}.qsub'))
            shutil.copyfile(f'../default_output/lmp.lmp', os.path.join(output_dir, 'lmp.lmp'))

    def get_equilibration_xyz(self):
        lmp2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'
        meta_maker = LMPSetup(oliba_wd='../default_output', seq=self.sequence)
        meta_maker.t = 100000
        meta_maker.del_missing_aas()
        meta_maker.get_hps_params()
        meta_maker.get_hps_pairs()
        meta_maker.write_hps_files(output_dir=None)
        meta_maker.run('lmp.lmp', n_cores=8)

        os.chdir('/home/adria/scripts/depured-lammps/default_output')
        file = '../default_output/data'
        os.system(lmp2pdb + ' ' + file)
        fileout = file + '_trj.pdb'

        traj = md.load('hps_traj.xtc', top=fileout)
        self.xyz = traj[-1].xyz*10
        mx = np.abs(self.xyz).max()
        self.box_size = int(mx*3)

        files = glob.glob('*')
        for file in files:
            os.remove(file)

    def get_pdb_xyz(self, pdb):
        struct = md.load_pdb(pdb)
        self.xyz = struct.xyz*10

    def _generate_lmp_input(self):
        if self.hps_pairs is None:
            self.get_hps_pairs()
        self.lmp_file_dict["t"] = self.t
        self.lmp_file_dict["dt"] = self.dt
        self.lmp_file_dict["pair_coeff"] = ''.join(self.hps_pairs)
        self.lmp_file_dict["debye"] = round(self.debye_wv*10**-10, 1)
        self.lmp_file_dict["v_seed"] = self.v_seed
        self.lmp_file_dict["langevin_seed"] = self.langevin_seed
        self.lmp_file_dict["temp"] = self.temperature
        self.lmp_file_dict["temperatures"] = self.temperatures
        self.lmp_file_dict["swap_every"] = self.swap_every
        self.lmp_file_dict["save"] = self.save
        # TODO this sucks but it is what it is
        self.lmp_file_dict["replicas"] = np.array2string(np.linspace(0, self.processors-1, self.processors, dtype='int'))[1:-1]

    def _generate_topo_input(self, nchains=1):
        masses = []
        for i in range(1, len(self.residue_dict)+1):
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] == i:
                    masses.append(f'           {i:2d}  {self.residue_dict[key]["mass"]} \n')

        atoms, bonds = [], []
        k = 1
        spaghetti = False

        for chain in range(1, nchains + 1):
            if self.xyz is None:
                xyz = [-240., -240 + chain * 20, -240]
                spaghetti = True
            for aa in self.sequence:
                if spaghetti:
                    xyz[0] += definitions.bond_length
                else:
                    xyz = self.xyz[0, k-1, :]
                atoms.append(f'      {k :3d}          {chain}    '
                             f'      {self.residue_dict[aa]["id"]:2d}   '
                             f'    {self.residue_dict[aa]["q"]: .2f}'
                             f'    {xyz[0]: .3f}'
                             f'    {xyz[1]: .3f}'
                             f'    {xyz[2]: .3f} \n')
                if k != chain * (len(self.sequence)):
                    bonds.append(f'       {k:3d}       1       {k:3d}       {k + 1:3d}\n')
                k += 1

        self.topo_file_dict["natoms"] = nchains * len(self.sequence)
        self.topo_file_dict["nbonds"] = nchains * (len(self.sequence) - 1)
        self.topo_file_dict["atom_types"] = len(self.residue_dict)
        self.topo_file_dict["masses"] = ''.join(masses)
        self.topo_file_dict["atoms"] = ''.join(atoms)
        self.topo_file_dict["bonds"] = ''.join(bonds)
        self.topo_file_dict["box_size"] = self.box_size

    def _generate_qsub(self):
        self.qsub_file_dict["work_dir"] = self.p_wd
        if self.temper:
            self.qsub_file_dict["command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -partition {self.processors}x1 -in lmp.lmp"
        else:
            self.qsub_file_dict["command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -in lmp.lmp"
        self.qsub_file_dict["np"] = self.processors
        self.qsub_file_dict["jobname"] = self.job_name

    def debye_length(self):
        kappa = 1 / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro / (cnt.epsilon_0 * 80 * cnt.Boltzmann * self.temperature)) * cnt.e)
        return kappa

    #TODO GET THIS FROM DATA ?
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
