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
import statsmodels.tsa.stattools
import psutil
from string import Template
from subprocess import run, PIPE


class LMPSetup:
    """
    LMPSetup serves as a way to create easily LAMMPS input files (.lmp) as well as LAMMPS topology files (.data)
    """
    # TODO : Allow to pass parameters as kwargs
    def __init__(self, oliba_wd, protein, temper, chains=1, equil_start=True):
        """
        :param oliba_wd: string, path where our run is
        :param protein: string, protein of choice. Its sequence must exist in ../data/sequences/{protein}.seq
        :param temper: bool, make it a REX or not
        :param chains: int, number of chaing we want to simulate
        :param equil_start: ???
        """
        self.o_wd = oliba_wd
        self.temper = temper

        self.lmp = definitions.lmp

        self.processors = 12
        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        self.chains = chains
        self.hps_epsilon = 0.2
        self.hps_pairs = None

        # ---- LMP PARAMETERS
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
        self.save = 5000
        self.langevin_damp = 10000

        self.debye_wv = None
        self.swap_every = 1000

        self.rerun_skip = 0
        self.rerun_start = 0
        self.rerun_stop = 0
        # ----

        self.topo_file_dict = {}
        self.lmp_file_dict = {}
        self.qsub_file_dict = {}
        self.slurm_file_dict = {}
        self.rst_file_dict = {}

        self.rerun_dump = None
        self.equil_start = equil_start

        # TODO : GRACEFULLY HANDLE THIS...
        self.job_name = f'x{self.chains}-{self.protein}'

    def del_missing_aas(self):
        """
        Do not include aminoacids neither in the pair_coeff entry in lmp.lmp and also not in data.data. It might be
        useful when debugging.
        :return: Nothing
        """
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
        """
        Aminoacids are ordered by alphabetical index by default. This method aminoacid indexing to a LAMMPS ordering,
        where each aminoacid is indexed by its order of appearence from the start of the chain
        :return: Nothing
        """
        idd = 1
        done_aas = []
        for aa in self.sequence:
            if aa not in done_aas:
                self.residue_dict[aa]["id"] = idd
                done_aas.append(aa)
                idd += 1
        ordered_keys = sorted(self.residue_dict, key=lambda x: (self.residue_dict[x]['id']))
        self.key_ordering = ordered_keys

    def make_equilibration_traj(self, t=500000):
        """
        Generate an equilibration trajectory
        :param t: How many steps do we want to run
        :return: Nothing
        """
        lmp2pdb = definitions.lmp2pdb
        meta_maker = LMPSetup(oliba_wd=os.path.join(definitions.module_dir, 'temp'), protein=self.protein, temper=False, equil_start=False)
        meta_maker.t = t
        meta_maker.save=int(t/10)
        meta_maker.write_hps_files(equil=True)
        meta_maker.run('lmp.lmp', n_cores=8)

        os.chdir(os.path.join(definitions.module_dir, 'temp'))
        file = os.path.join(definitions.module_dir, 'temp/data')
        os.system(lmp2pdb + ' ' + file)
        fileout = file + '_trj.pdb'

        traj = md.load('xtc_traj.xtc', top=fileout)
        self.xyz = traj[-1].xyz * 10
        mx = np.abs(self.xyz).max()
        self.box_size = int(mx * 3)

        print(f"-> Saving equilibration pdb at {os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb')}")
        traj[-1].save_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))

    def get_pdb_xyz(self, pdb=None, padding=0.55):
        """
        Get the xyz of a pdb. If we're demanding more than 1 chain, then generate xyz for every other chain
        so that they are in a cubic setup using the xyz from the single case.
        :param pdb: string, path leading to the pdb we want to use
        :param padding: float, how close we want the copies in the multichain case
        :return: Nothing
        """
        if pdb is not None:
            struct = md.load_pdb(pdb)
        else:
            if not os.path.exists(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb')):
                self.make_equilibration_traj()
            struct = md.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
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
        """
        Run a LAMMPS job
        :param file: string, path to the .lmp file to run LAMMPS
        :param n_cores: int, number of cores we wish to use
        :return: stdout of the LAMMPS run
        """
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
        # for proc in psutil.process_iter():
        #     if proc.name() == 'lmp':
        #         proc.kill()
        return out

    # TODO : KWARGS ?
    def write_hps_files(self, equil=False, rerun=False, data=True, qsub=True, lmp=True,
                        slurm=False, readme=False, rst=True, pdb=True, silent=False):
        """
        Write the files for the corresponding LAMMPS parameters (I, T...). All function parameters are booleans to
        choose whether we want to write files of that extension or not
        :param equil: ??
        :param rerun:
        :param data:
        :param qsub:
        :param lmp:
        :param slurm: .slm file to run at CSUC
        :param readme:
        :param rst: .lmp file to run LAMMPS restarts
        :param pdb:
        :return:
        """
        pathlib.Path(self.o_wd).mkdir(parents=True, exist_ok=True)

        if not self.temper:
            for T in range(len(self.temperatures)):
                if len(self.temperatures)==1:
                    p = self.o_wd
                else:
                    p = os.path.join(self.o_wd, f"T{T}")
                    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
                # TODO : KWARGS ?
                self._write(output_path=p, T=T, equil=equil, rerun=rerun, data=data, qsub=qsub, lmp=lmp, slurm=slurm,
                            readme=readme, rst=rst, pdb=pdb)
        else:
            # TODO : KWARGS ?
            self._write(output_path=self.o_wd, equil=equil, rerun=rerun, data=data, qsub=qsub, lmp=lmp, slurm=slurm,
                            readme=readme, rst=rst, pdb=pdb)
        if not silent:
            self._success_message()

    def assert_build(self):
        """
        Check whether the demanded parameters match those actually written. Depends on the lmp library
        :return: Nothing
        """
        asserter = lmp.LMP(oliba_wd=self.o_wd)
        assert int(asserter.get_eps()) == int(self.water_perm)
        assert asserter.get_temperatures() == list(map(str, self.temperatures))
        assert asserter.get_ionic_strength() == self.ionic_strength
        assert asserter.get_seq_from_hps().replace('L', 'I') == self.sequence.replace('L', 'I')
        print("Assertion complete")

    def debye_length(self, T=None):
        """
        Calculate debye Length for the given system. If T is None (in the REX chase for example) then assume 300K
        :return: float, debvye length
        """
        # TODO : INCORPORATE SELF.TEMPERATURES
        if T is None:
            l = np.sqrt(cnt.epsilon_0 * self.water_perm * cnt.Boltzmann * 300)
        else:
            T = self.temperatures[T]
            l = np.sqrt(cnt.epsilon_0 * self.water_perm * cnt.Boltzmann * T)
        l = l / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro) * cnt.e)
        return l

    def get_hps_pairs(self, from_file=None):
        """
        Build pair_coeff from HPS parameters. If from_file is None, they are automatically generated from those in
        ../../data/hps
        :param from_file: string, path to read the pairs from
        :return: Nothing
        """
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

    def get_correlation_time(self):
        """
        Obtaing the correlation time from the radius of gyration autocorrelation function
        :return: list[T], list for the autocorrelation time for every temperature
        """
        save_period = self.data[0, 1, 0]
        rg = self.rg()
        corr_times_T = []
        for T in range(len(self.get_temperatures())):
            ac = statsmodels.tsa.stattools.acf(rg[T,:], fft=False, nlags=80)
            corr_t = ac[np.abs(ac) < 0.5][0]
            corr_time_T.append(save_period*np.where(ac==corr_t))
        return corr_times_T

    def _get_hps_params(self):
        """
        Get the HPS parameters to a python dict
        :return:
        """
        for key in self.residue_dict:
            for lam_key in definitions.lambdas:
                if self.residue_dict[key]["name"] == lam_key:
                    self.residue_dict[key]["lambda"] = definitions.lambdas[lam_key]
            for sig_key in definitions.sigmas:
                if self.residue_dict[key]["name"] == sig_key:
                    self.residue_dict[key]["sigma"] = definitions.sigmas[sig_key]

    def _write(self, output_path, T=None, equil=False, rerun=False, data=True, qsub=True, lmp=True,
                        slurm=False, readme=False, rst=True, pdb=True):
        """
        Write desired parameters to disk. Essentially behavior write_hps_files. Here we use python templates to actually
        generate the files. Such templates can be found at ../templates/ It is important to check the templates out
        in case we want to add any further features
        :param output_path:
        :param T:
        :param equil:
        :param rerun:
        :param data:
        :param qsub:
        :param lmp:
        :param slurm:
        :param readme:
        :param rst:
        :param pdb:
        :return:
        """
        if self.equil_start:
            self.get_pdb_xyz()

        if self.temper:
            self.processors = len(self.temperatures)

        self._generate_lmp_input(T)
        self._generate_qsub(perdiu_dir=output_path.replace('/perdiux', ''))
        self._generate_data_input()
        self._generate_slurm()

        if readme:
            self._generate_README()
        if self.xyz is not None and pdb:
            pdb = self._generate_pdb()
            with open(os.path.join(output_path, f'topo.pdb'), 'w+') as f:
                f.write(pdb)

        if self.temper:
            lmp_temp_file = open(os.path.join(definitions.module_dir, 'templates/replica/input_template.lmp'))
        elif equil:
            lmp_temp_file = open(os.path.join(definitions.module_dir, 'templates/equilibration/input_template.lmp'))
        elif rerun:
            lmp_temp_file = open(os.path.join(definitions.module_dir, 'templates/rerun/input_template.lmp'))
        else:
            lmp_temp_file = open(os.path.join(definitions.module_dir, 'templates/general/input_template.lmp'))

        lmp_template = Template(lmp_temp_file.read())
        lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)

        topo_temp_file = open(os.path.join(definitions.module_dir, 'templates/topo_template.data'))
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(self.topo_file_dict)

        rst_temp_file = open(os.path.join(definitions.module_dir, 'templates/restart/input_template.lmp'))
        rst_template = Template(rst_temp_file.read())
        rst_subst = rst_template.safe_substitute(self.lmp_file_dict)

        qsub_temp_file = open(os.path.join(definitions.module_dir, 'templates/qsub_template.tmp'))
        qsub_template = Template(qsub_temp_file.read())
        qsub_subst = qsub_template.safe_substitute(self.qsub_file_dict)

        slurm_temp_file = open(os.path.join(definitions.module_dir, 'templates/slurm_template.tmp'))
        slurm_template = Template(slurm_temp_file.read())
        slurm_subst = slurm_template.safe_substitute(self.slurm_file_dict)

        if os.path.abspath(output_path):
            if data:
                with open(os.path.join(output_path, 'data.data'), 'tw') as fileout:
                    fileout.write(topo_subst)
            if qsub:
                with open(os.path.join(output_path, f'{self.job_name}.qsub'), 'tw') as fileout:
                    fileout.write(qsub_subst)
            if slurm:
                with open(os.path.join(output_path, f'{self.job_name}.slm'), 'tw') as fileout:
                    fileout.write(slurm_subst)
            if lmp:
                with open(os.path.join(output_path, 'lmp.lmp'), 'tw') as fileout:
                    fileout.write(lmp_subst)
            if rst:
                with open(os.path.join(output_path, 'rst.lmp'), 'tw') as fileout:
                    fileout.write(rst_subst)

    # TODO : THIS ONLY IF WE HAVE self.xyz... Maybe I can make it general
    # TODO : ALSO THIS IS NOT CENTERED FOR SINGLE CHAIN!!!!
    def _generate_pdb(self, display=None):
        """
        Generate an HPS pdb, since LAMMPS and MDTraj are unable to do it
        :param display: string, switch to print charged aminoacids vs non charged (display=anything except "charged"/None) ; or charged+ vs charged- vs noncharged (display=charged)
        :return: Nothing
        """
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
                # Empty space is chain ID, which seems to not be strictly necessary...
                xyz += f'ATOM  {c + 1:>5} {res_name:>4}   {res_name} {" "} {i + 1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      PROT \n'
                c += 1
            xyz += 'TER \n'
        bonds = ''
        for i in range(len(self.sequence) * self.chains):
            if (i + 1) % len(self.sequence) != 0: bonds += 'CONECT{:>5}{:>5} \n'.format(i + 1, i + 2)
        bottom = 'END \n'
        return header + xyz + bonds + bottom

    # TODO : THIS STUFF MIGHT BE A BIT USELESS... MAYBE DOABLE AT INIT
    def _generate_lmp_input(self, T):
        """
        Generate a python dict containing all the chosen parameters. The python dict is necessary for the python
        Template substituion, otherwise it would be useless
        :param T: int, temperature index we wish to write. If coming from a REX, T is None
        :return: Nothing
        """
        if self.hps_pairs is None:
            self.get_hps_pairs()
        self.lmp_file_dict["t"] = self.t
        self.lmp_file_dict["dt"] = self.dt
        self.lmp_file_dict["pair_coeff"] = ''.join(self.hps_pairs)
        self.lmp_file_dict["debye"] = round(1 / self.debye_length(T) * 10 ** -10, 3)
        self.lmp_file_dict["v_seed"] = self.v_seed
        self.lmp_file_dict["langevin_seed"] = self.langevin_seed
        if T is None and self.temper:
            self.lmp_file_dict["temperatures"] = ' '.join(map(str, self.temperatures))
        else:
            self.lmp_file_dict["temp"] = self.temperatures[T]
        self.lmp_file_dict["water_perm"] = self.water_perm
        self.lmp_file_dict["swap_every"] = self.swap_every
        self.lmp_file_dict["save"] = self.save
        self.lmp_file_dict["rerun_skip"] = self.rerun_skip
        self.lmp_file_dict["rerun_start"] = self.rerun_start
        self.lmp_file_dict["rerun_stop"] = self.rerun_stop
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
        """
        Generate the topology dict containing all chosen parameters. The dict is necessary for Python Template
        substitution
        :return:
        """
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
                    xyzs.append(xyz.copy())
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

    def _generate_qsub(self, perdiu_dir):
        """
        Generate a qsub file to run at perdiux's
        :return:
        """
        self.qsub_file_dict["work_dir"] = perdiu_dir
        if self.temper:
            self.qsub_file_dict[
                "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -partition {self.processors}x1 -in lmp.lmp"
        else:
            self.qsub_file_dict[
                "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -in lmp.lmp"
        self.qsub_file_dict["np"] = self.processors
        self.qsub_file_dict["jobname"] = self.job_name

    def _generate_slurm(self):
        """
        Generate a slm file to run at CSUC
        :return:
        """
        if self.temper:
            self.slurm_file_dict["command"] = f"srun `which lmp` -in lmp.lmp -partition {self.processors}x1"
        else:
            self.slurm_file_dict["command"] = f"srun `which lmp` -in lmp.lmp"
        self.slurm_file_dict["np"] = self.processors
        self.slurm_file_dict["jobname"] = self.job_name

    def _generate_README(self):
        """
        Generate a custom README file containing the most essential parameters of an HPS LAMMPS run
        :return:
        """
        with open(os.path.join(self.o_wd, 'README.txt'), 'tw') as readme:
            readme.write(f'HPS Scale : {self.hps_scale} \n')
            readme.write(f'Ionic Strength : {self.ionic_strength} \n')
            readme.write(f'Medium Permittivity : {self.water_perm} \n')
            readme.write(f'Protein : {self.protein} \n')
            readme.write(f'Temperatures : {self.temperatures} \n')
            readme.write(f'Temper : {self.temper} \n')
            readme.write(f'Number of chains : {self.chains} \n')
            readme.write(f'Langevin Damp : {self.langevin_damp} \n')

    def _success_message(self, padding=5, section_padding=3, param_padding=6):
        """
        Print a recoplitaory message of the run
        :param padding: int, General padding
        :param section_padding: int, PARAMETERS section padding
        :param param_padding: int, parameters padding
        :return: Nothing
        """
        title = f"Input files created at {self.o_wd} for {self.protein}"
        l = len(title)
        out = ''
        out += f'╔{"═" * (l + padding * 2)}╗\n'
        out += f'║{" " * padding}{title:<{l}}{" " * padding}║\n'
        out += f'║{"-"*(len(title)+padding*2)}║\n'
        out += f'║{" " * section_padding}{"PARAMETERS":<{l}}{" " * (padding + padding - section_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Chains = {self.chains}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Ionic Strength (mM) = {self.ionic_strength}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Medium Permittivity = {self.water_perm}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Temperatures (K) = {self.temperatures}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - HPS Scale = {self.hps_scale}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'╚{"═" * (l + padding * 2)}╝'
        print(out)
