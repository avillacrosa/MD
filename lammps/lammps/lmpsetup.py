import definitions
import lmp
import scipy.constants as cnt
import numpy as np
import shutil
import math
import os
import mdtraj as md
import pathlib
import multiprocessing as mp
from string import Template
from subprocess import run, PIPE


class LMPSetup:
    def __init__(self, oliba_wd, protein, temper, chains=1):
        self.o_wd = oliba_wd
        self.p_wd = oliba_wd.replace('/perdiux', '')
        self.temper = temper

        self.lmp = '/home/adria/local/lammps/bin/lmp'

        self.processors = 12
        with open(os.path.join('../data/sequences', f'{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        self.chains = chains
        self.hps_epsilon = 0.2
        self.hps_pairs = None

        # ---- LMP PARAMETERS
        self.temperature = 300
        self.temperatures = [300, 320, 340, 360, 380, 400]
        self.ionic_strength = 100e-3
        self.dt = 10.
        self.t = 100000000
        self.xyz = None
        self.protein = protein

        self.box_size = 2500
        self.water_perm = 80.
        self.hps_scale = 1.0

        self.seq_charge = None
        self.residue_dict = dict(definitions.residues)
        self.key_ordering = list(self.residue_dict.keys())

        self.v_seed = 494211
        self.langevin_seed = 451618
        self.save = 50000
        self.langevin_damp = 10000

        self.debye_wv = 1 / self.debye_length()
        self.swap_every = 1000

        self.rerun_skip = 0
        # ----

        self.topo_file_dict = {}
        self.lmp_file_dict = {}
        self.qsub_file_dict = {}
        self.slurm_file_dict = {}
        self.rst_file_dict = {}

        self.rerun_dump = None

        # TODO : GRACEFULLY HANDLE THIS...
        self.base_dir = f'{self.hps_scale:.1f}ls-{self.ionic_strength * 1e3:.0f}I-{self.water_perm:.0f}e'
        # self.out_dir = os.path.join(self.o_wd, self.base_dir)
        self.out_dir = self.o_wd
        self.job_name = f'x{self.chains}-{self.protein}_{self.base_dir}'

        pathlib.Path(self.out_dir).mkdir(parents=True, exist_ok=True)

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
        idd = 1
        done_aas = []
        for aa in self.sequence:
            if aa not in done_aas:
                self.residue_dict[aa]["id"] = idd
                done_aas.append(aa)
                idd += 1
        ordered_keys = sorted(self.residue_dict, key=lambda x: (self.residue_dict[x]['id']))
        self.key_ordering = ordered_keys

    def get_equilibration_xyz(self, save=False, t=100000):
        lmp2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'
        meta_maker = LMPSetup(oliba_wd='../temp', protein=self.protein, temper=self.temper)
        meta_maker.t = t
        meta_maker.get_hps_pairs()
        meta_maker.write_hps_files(output_dir='', equil=True)
        meta_maker.run('lmp.lmp', n_cores=8)

        os.chdir('/home/adria/scripts/depured-lammps/temp')
        file = '../temp/data'
        os.system(lmp2pdb + ' ' + file)
        fileout = file + '_trj.pdb'

        traj = md.load('xtc_traj.xtc', top=fileout)
        self.xyz = traj[-1].xyz * 10
        mx = np.abs(self.xyz).max()
        self.box_size = int(mx * 3)

        if save:
            print(f"-> Saving equilibration pdb at {os.path.join(self.out_dir, 'equilibration.pdb')}")
            traj[-1].save_pdb(os.path.join(self.out_dir, 'equilibration.pdb'))
        else:
            print(f"-> Saving equilibration pdb at {os.path.join('../temp', 'equilibration.pdb')}")
            traj[-1].save_pdb(os.path.join('../temp', 'equilibration.pdb'))

    def get_pdb_xyz(self, pdb, padding=0.55):
        struct = md.load_pdb(pdb)
        struct.center_coordinates()
        rg = md.compute_rg(struct)
        d = rg[0] * self.chains ** (1 / 3) * 8
        struct.unitcell_lengths = np.array([[d, d, d]])

        if self.chains == 1:
            self.xyz = struct.xyz * 10
            self.box_size = d * 10
        else:
            # TEST
            n_cells = int(math.ceil(self.chains ** (1 / 3)))
            unitcell_d = d / n_cells

            def _build_box():
                c = 0
                for z in range(n_cells):
                    for y in range(n_cells):
                        for x in range(n_cells):
                            if c == self.chains:
                                return adder
                            c += 1
                            dist = np.array(
                                [unitcell_d * (x + 1 / 2), unitcell_d * (y + 1 / 2), unitcell_d * (z + 1 / 2)])
                            dist -= padding * (dist - d / 2)

                            struct.xyz[0, :, :] = struct.xyz[0, :, :] + dist
                            if x + y + z == 0:
                                adder = struct[:]
                            else:
                                adder = adder.stack(struct)
                            struct.xyz[0, :, :] = struct.xyz[0, :, :] - dist
                return adder

            system = _build_box()

            self.box_size = d * 10
            self.xyz = system.xyz * 10

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

    # TODO : THIS CAN BE BETTER
    def write_hps_files(self, output_dir='default', equil=False, rerun=False, data=True, qsub=True, lmp=True,
                        slurm=True, readme=True, rst=True):
        if output_dir == 'default':
            output_dir = self.out_dir

        self._generate_lmp_input()
        self._generate_qsub()
        self._generate_data_input()
        self._generate_slurm()
        if readme:
            self._generate_README()
        if self.xyz is not None:
            self._generate_pdb()

        if self.temper:
            lmp_temp_file = open('../data/templates/replica/input_template.lmp')
        elif equil:
            lmp_temp_file = open('../data//templates/equilibration/input_template.lmp')
        elif rerun:
            lmp_temp_file = open('../data//templates/rerun/input_template.lmp')
        else:
            lmp_temp_file = open('../data//templates/general/input_template.lmp')

        lmp_template = Template(lmp_temp_file.read())
        lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)
        topo_temp_file = open('../data//templates/topo_template.data')
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(self.topo_file_dict)

        rst_temp_file = open('../data//templates/restart/input_template.lmp')
        rst_template = Template(rst_temp_file.read())
        rst_subst = rst_template.safe_substitute(self.lmp_file_dict)

        qsub_temp_file = open('../data//templates/qsub_template.tmp')
        qsub_template = Template(qsub_temp_file.read())
        qsub_subst = qsub_template.safe_substitute(self.qsub_file_dict)

        slurm_temp_file = open('../data//templates/slurm_template.tmp')
        slurm_template = Template(slurm_temp_file.read())
        slurm_subst = slurm_template.safe_substitute(self.slurm_file_dict)

        # TODO: USELESS SAVE IN TEMP FOLDER...
        if data:
            with open(f'../temp/data.data', 'tw') as fileout:
                fileout.write(topo_subst)
        if qsub:
            with open(f'../temp/{self.job_name}.qsub', 'tw') as fileout:
                fileout.write(qsub_subst)
        if lmp:
            with open(f'../temp/lmp.lmp', 'tw') as fileout:
                fileout.write(lmp_subst)
        if rst:
            with open(f'../temp/rst.lmp', 'tw') as fileout:
                fileout.write(rst_subst)
        if slurm:
            with open(f'../temp/{self.job_name}.slm', 'tw') as fileout:
                fileout.write(slurm_subst)

        if os.path.abspath(output_dir) != os.path.abspath('../temp'):
            if data:
                shutil.copyfile(f'../temp/data.data', os.path.join(output_dir, 'data.data'))
                os.remove(f'../temp/data.data')
            if qsub:
                shutil.copyfile(f'../temp/{self.job_name}.qsub', os.path.join(output_dir, f'{self.job_name}.qsub'))
                os.remove(f'../temp/{self.job_name}.qsub')
            if slurm:
                shutil.copyfile(f'../temp/{self.job_name}.slm', os.path.join(output_dir, f'{self.job_name}.slm'))
                os.remove(f'../temp/{self.job_name}.slm')
            if lmp:
                shutil.copyfile(f'../temp/lmp.lmp', os.path.join(output_dir, 'lmp.lmp'))
                os.remove(f'../temp/lmp.lmp')
            if rst:
                shutil.copyfile(f'../temp/rst.lmp', os.path.join(output_dir, 'rst.lmp'))
                os.remove(f'../temp/rst.lmp')

    def assert_build(self):
        asserter = lmp.LMP(oliba_wd=self.o_wd)
        assert int(asserter.get_eps()) == int(self.water_perm)
        assert asserter.get_temperatures() == list(map(str, self.temperatures))
        assert asserter.get_ionic_strength() == self.ionic_strength
        assert asserter.get_seq_from_hps().replace('L', 'I') == self.sequence.replace('L', 'I')
        print("Assertion complete")

    def debye_length(self):
        # TODO : INCORPORATE SELF.TEMPERATURES
        l = np.sqrt(cnt.epsilon_0 * self.water_perm * cnt.Boltzmann * 300)
        l = l / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro) * cnt.e)
        return l

    def get_hps_pairs(self, from_file=None):
        self._get_hps_params()
        lines = ['pair_coeff          *       *       0.000000   0.000    0.000000   0.000   0.000\n']
        if from_file:
            lambda_gen = np.genfromtxt(from_file)
            count = 0
        else:
            self._get_hps_params()
        for i in range(len(self.key_ordering)):
            for j in range(i, len(self.key_ordering)):
                res_i = self.residue_dict[self.key_ordering[i]]
                res_j = self.residue_dict[self.key_ordering[j]]
                lambda_ij = (res_i["lambda"] + res_j["lambda"]) / 2
                sigma_ij = (res_i["sigma"] + res_j["sigma"]) / 2
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

    def _get_hps_params(self):
        for key in self.residue_dict:
            for lam_key in definitions.lambdas:
                if self.residue_dict[key]["name"] == lam_key:
                    self.residue_dict[key]["lambda"] = definitions.lambdas[lam_key]
            for sig_key in definitions.sigmas:
                if self.residue_dict[key]["name"] == sig_key:
                    self.residue_dict[key]["sigma"] = definitions.sigmas[sig_key]

    # TODO : THIS ONLY IF WE HAVE self.xyz... Maybe I can make it general
    # TODO : THIS SEEMS TO BREAK DOWN IF WE SPAGHETTI...
    # TODO : ALSO THIS IS NOT CENTERED FOR SINGLE CHAIN!!!!
    def _generate_pdb(self, display=None):
        abc = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']
        header = f'CRYST1     {self.box_size:.0f}     {self.box_size:.0f}     {self.box_size:.0f}     90     90     90   \n'
        xyz = ''
        c = 0
        for n in range(self.chains):
            for i, aa in enumerate(self.sequence):
                coords = self.xyz[0, c, :]
                if display:
                    if self.residue_dict[aa]["q"] > 0:
                        res_name = 'C' if display == 'charged' else 'P'
                    elif self.residue_dict[aa]["q"] < 0:
                        res_name = 'C' if display == 'charged' else 'M'
                    else:
                        res_name = 'N'
                else:
                    res_name = aa
                xyz += f'ATOM  {c + 1:>5} {res_name:>4}   {res_name} {abc[n]} {i + 1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      PROT \n'
                c += 1
            xyz += 'TER \n'
        bonds = ''
        for i in range(len(self.sequence) * self.chains):
            if (i + 1) % len(self.sequence) != 0: bonds += 'CONECT{:>5}{:>5} \n'.format(i + 1, i + 2)
        bottom = 'END \n'
        with open(os.path.join(self.o_wd, f'topo.pdb'), 'w+') as f:
            f.write(header + xyz + bonds + bottom)

    # TODO : THIS STUFF MIGHT BE A BIT USELESS... MAYBE DOABLE AT INIT
    def _generate_lmp_input(self):
        if self.hps_pairs is None:
            self.get_hps_pairs()
        self.lmp_file_dict["t"] = self.t
        self.lmp_file_dict["dt"] = self.dt
        self.lmp_file_dict["pair_coeff"] = ''.join(self.hps_pairs)
        self.lmp_file_dict["debye"] = round(1 / self.debye_length() * 10 ** -10, 3)
        self.lmp_file_dict["v_seed"] = self.v_seed
        self.lmp_file_dict["langevin_seed"] = self.langevin_seed
        self.lmp_file_dict["temp"] = self.temperature
        self.lmp_file_dict["temperatures"] = ' '.join(map(str, self.temperatures))
        self.lmp_file_dict["water_perm"] = self.water_perm
        self.lmp_file_dict["swap_every"] = self.swap_every
        self.lmp_file_dict["save"] = self.save
        self.lmp_file_dict["rerun_skip"] = self.rerun_skip
        if int(self.t / 10000) != 0:
            self.lmp_file_dict["restart"] = int(self.t / 10000)
        else:
            self.lmp_file_dict["restart"] = 500
        # TODO this sucks but it is what it is, better option upstairs..
        ntemps = len(self.temperatures)
        self.lmp_file_dict["replicas"] = ' '.join(map(str, np.linspace(0, ntemps - 1, ntemps, dtype='int')))
        self.lmp_file_dict["rerun_dump"] = self.rerun_dump
        self.lmp_file_dict["langevin_damp"] = self.langevin_damp

    def _generate_data_input(self):
        masses = []
        for i in range(1, len(self.residue_dict) + 1):
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] == i:
                    masses.append(f'           {i:2d}  {self.residue_dict[key]["mass"]} \n')

        atoms, bonds = [], []
        k = 1
        spaghetti = False
        xyzs = []
        for chain in range(1, self.chains + 1):
            # TODO CENTER SPAGHETTI BETTER...
            if self.xyz is None:
                xyz = [240., 240 + chain * 20, 240]
                spaghetti = True
            for aa in self.sequence:
                if spaghetti:
                    xyz[0] += definitions.bond_length
                    xyzs.append(xyz)
                else:
                    xyz = self.xyz[0, k - 1, :]
                atoms.append(f'      {k :3d}          {chain}    '
                             f'      {self.residue_dict[aa]["id"]:2d}   '
                             f'    {self.residue_dict[aa]["q"]: .2f}'
                             f'    {xyz[0]: .3f}'
                             f'    {xyz[1]: .3f}'
                             f'    {xyz[2]: .3f} \n')
                if k != chain * (len(self.sequence)):
                    bonds.append(f'       {k:3d}       1       {k:3d}       {k + 1:3d}\n')
                k += 1
        if spaghetti:
            self.xyz = np.array([xyzs])
        self.topo_file_dict["natoms"] = self.chains * len(self.sequence)
        self.topo_file_dict["nbonds"] = self.chains * (len(self.sequence) - 1)
        self.topo_file_dict["atom_types"] = len(self.residue_dict)
        self.topo_file_dict["masses"] = ''.join(masses)
        self.topo_file_dict["atoms"] = ''.join(atoms)
        self.topo_file_dict["bonds"] = ''.join(bonds)
        self.topo_file_dict["box_size"] = int(self.box_size)

    def _generate_qsub(self):
        self.qsub_file_dict["work_dir"] = self.p_wd
        if self.temper:
            self.qsub_file_dict[
                "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -partition {self.processors}x1 -in lmp.lmp"
        else:
            self.qsub_file_dict[
                "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -in lmp.lmp"
        self.qsub_file_dict["np"] = self.processors
        self.qsub_file_dict["jobname"] = self.job_name

    def _generate_slurm(self):
        if self.temper:
            self.slurm_file_dict["command"] = f"srun `which lmp` -in lmp.lmp -partition {self.processors}x1"
        else:
            self.slurm_file_dict["command"] = f"srun `which lmp` -in lmp.lmp"
        self.slurm_file_dict["np"] = self.processors
        self.slurm_file_dict["jobname"] = self.job_name

    def _generate_README(self):
        with open(os.path.join(self.o_wd, 'README.txt'), 'tw') as readme:
            readme.write(f'HPS Scale : {self.hps_scale} \n')
            readme.write(f'Ionic Strength : {self.ionic_strength} \n')
            readme.write(f'Medium Permittivity : {self.water_perm} \n')
            readme.write(f'Protein : {self.protein} \n')
            readme.write(f'Temperatures : {self.temperatures} \n')
            readme.write(f'Temper : {self.temper} \n')
            readme.write(f'Number of chains : {self.chains} \n')
            readme.write(f'Langevin Damp : {self.langevin_damp} \n')
