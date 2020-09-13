import definitions
import lmp
import scipy.constants as cnt
import numpy as np
import stat
import decimal as dec
import math
import pandas as pd
import os
import mdtraj as md
import pathlib
import multiprocessing as mp
import statsmodels.tsa.stattools
from string import Template
from subprocess import run, PIPE


class LMPSetup:
    """
    LMPSetup serves as a way to create easily LAMMPS input files (.lmp) as well as LAMMPS topology files (.data)
    """
    # TODO : Allow to pass parameters as kwargs
    def __init__(self, oliba_wd, protein, temper=False, chains=1, equil_start=True, model='HPS-T', slab=False, use_temp_eps=False, **kwargs):
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
        self.slab = slab

        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        self.this = os.path.dirname(os.path.dirname(__file__))
        self.hps_pairs = None
        self.equil_start = equil_start
        self.fix_region = kwargs.get('fix_region', None)
        self.fixed_atoms = None

        # --- EXPLICIT RUN PARAMETERS --- #
        self.t = kwargs.get('t', 100000000)
        self.dt = kwargs.get('dt', 10.)
        self.chains = chains
        self.ionic_strength = kwargs.get('ionic_strength', 100e-3) # In Molar
        self.temperatures = kwargs.get('temperatures', [300, 320, 340, 360, 380, 400])
        self.box_size = {"x": 2500, "y": 2500, "z": 2500}
        self.water_perm = kwargs.get('water_perm', 80.)
        self.use_temp_eps = use_temp_eps
        self.v_seed = kwargs.get('v_seed', 494211)
        self.langevin_seed = kwargs.get('langevin_seed', 451618)
        self.save = kwargs.get('save', 10000)
        self.langevin_damp = kwargs.get('langevin_damp', 10000)
        self.processors = kwargs.get('processors', 1)
        self.charge_scale = kwargs.get('charge_scale', 1)
        # ------------------------------- #

        # ---- LMP PARAMETERS ---- #
        self.hps_epsilon = kwargs.get('hps_epsilon', 0.2)
        self.hps_scale = kwargs.get('hps_scale', 1.0)
        # ------------------------ #

        # ---- gHPS PARAMETERS ---- #
        self.a6 = kwargs.get('a6', 0.8)
        # ------------------------- #

        # ---- SLAB PARAMETERS ---- #
        self.slab_t = kwargs.get('slab_t', 200000)
        # self.final_slab_volume = self.box_size["x"]/4
        self.slab_dimensions = {}
        droplet_zlength = 500
        # self.slab_dimensions["x"] = 1.3*(self.chains*4*math.pi/3/droplet_zlength*self.rw_rg()**3)**0.5
        # self.slab_dimensions["x"] = 1.5*(self.chains*4*math.pi/3/droplet_zlength*self.rw_rg()**3)**0.5
        self.slab_dimensions["x"] = 1.3*(self.chains * 4 * math.pi / 3 / droplet_zlength * self.rw_rg() ** 3) ** 0.5
        self.slab_dimensions["y"] = self.slab_dimensions["x"]
        self.slab_dimensions["z"] = 5*droplet_zlength
        # self.slab_dimensions["z"] = 2800

        self.final_slab_volume = self.box_size["x"]/4
        self.deformation_ts = kwargs.get('deformation_ts', 1)
        # ------------------------- #

        # ---------- KH ----------- #
        self.kh_alpha = 0.228
        self.kh_eps0 = -1

        self.use_random = kwargs.get('use_random', False)

        self.xyz = None
        self.protein = protein
        self.seq_charge = None
        self.residue_dict = dict(definitions.residues)
        self.key_ordering = list(self.residue_dict.keys())

        self.debye_wv = kwargs.get('debye', None)
        self.swap_every = 1000

        self.avg_sigma = 0
        for s in definitions.sigmas:
            self.avg_sigma += definitions.sigmas[s]
        self.avg_sigma = self.avg_sigma/len(definitions.sigmas)

        self.rerun_skip = 0
        self.rerun_start = 0
        self.rerun_stop = 0
        self.host = kwargs.get('host', "")

        self.topo_file_dict = {}
        self.lmp_file_dict = {}
        self.qsub_file_dict = {}
        self.slurm_file_dict = {}
        self.rst_file_dict = {}

        self.rerun_dump = None
        self.model = model

        self.job_name = kwargs.get('job_name', f's{self.hps_scale}_x{self.chains}-{self.protein}') 

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

    def make_equilibration_traj(self, t=100000):
        """
        Generate an equilibration trajectory
        :param t: How many steps do we want to run
        :return: Nothing
        """
        print(f"-> Equilibrating structure with a short LAMMPS run...")
        meta_maker = LMPSetup(oliba_wd=os.path.join(self.this, 'temp'), protein=self.protein, temper=False, equil_start=False)
        meta_maker.t = t
        meta_maker.save = int(t/10)
        meta_maker.temperatures = [300]
        meta_maker.write_hps_files(equil=True, qsub=False, silent=True)
        meta_maker.run('lmp0.lmp', n_cores=8)
        traj = md.load(os.path.join(self.this, 'temp/dcd_traj_0.dcd'), top=os.path.join(self.this, 'temp/topo.pdb'))
        self.xyz = traj[-1].xyz * 10
        mx = np.abs(self.xyz).max()
        self.box_size["x"] = int(mx * 3)
        self.box_size["y"] = int(mx * 3)
        self.box_size["z"] = int(mx * 3)

        print(f"-> Saving equilibration pdb at {os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb')}")
        traj[-1].save_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))

    def get_pdb_xyz(self, pdb=None, padding=0.8):
        """
        Get the xyz of a pdb. If we're demanding more than 1 chain, then generate xyz for every other chain
        so that they are in a cubic setup using the xyz from the single case.
        :param use_random:
        :param pdb: string, path leading to the pdb we want to use
        :param padding: float, how close we want the copies in the multichain case
        :return: Nothing
        """
        if self.xyz is not None:
            return self.xyz

        if pdb is not None:
            struct = md.load_pdb(pdb)
        else:
            if not os.path.exists(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb')):
                self.make_equilibration_traj()
            struct = md.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
        struct.center_coordinates()
        struct.xyz = struct.xyz*10.
        d = 4.0 * self.rw_rg(monomer_l=5.5) * (self.chains * 4 * math.pi / 3) ** (1 / 3)
        struct.unitcell_lengths = np.array([[d, d, d]])

        self.xyz = struct.xyz
        if self.chains > 1:
            def _build_cubic_box():
                c = 0
                n_cells = int(math.ceil(self.chains ** (1 / 3)))
                u_d = d / n_cells
                for z in range(n_cells):
                    for y in range(n_cells):
                        for x in range(n_cells):
                            if c == self.chains:
                                return adder
                            c += 1
                            dist = np.array([u_d * (x + 1/2), u_d * (y + 1/2), u_d * (z + 1/2)])
                            dist -= d / 2
                            dist *= padding

                            struct.xyz[0, :, :] = struct.xyz[0, :, :] + dist
                            if x + y + z == 0:
                                adder = struct[:]
                            else:
                                adder = adder.stack(struct)
                            struct.xyz[0, :, :] = struct.xyz[0, :, :] - dist
                return adder

            def build_random():
                adder = struct[:]
                dist = np.random.uniform(-d/2, d/2, 3)
                # dist = dist - 0.6 * (dist-d/2)
                adder.xyz[0, :, :] += dist
                chain = 1
                centers_save = [dist]
                rg_ext = self.rw_rg()

                def find_used(dist_d, centers_save_d):
                    for i, center in enumerate(centers_save_d):
                        # Dynamic Rg scale based on equilibration Rg ?
                        if np.linalg.norm(dist_d - center) < rg_ext*3.0:
                            return True
                        else:
                            continue
                    return False

                while chain < self.chains:
                    # TODO : move to gaussian ? Uniform is not uniform in 3d ?
                    dist = np.random.uniform(-d/2, d/2, 3)
                    # TODO: REMOVE THIS QUICK DIRTY TRICK
                    dist -= 0.2 * (dist)

                    used = find_used(dist, centers_save)
                    if not used:
                        struct.xyz[0, :, :] = struct.xyz[0, :, :] + dist
                        adder = adder.stack(struct)
                        struct.xyz[0, :, :] = struct.xyz[0, :, :] - dist
                        chain += 1
                        centers_save.append(dist)
                return adder

            if self.use_random:
                system = build_random()
            else:
                system = _build_cubic_box()
            self.xyz = system.xyz
        self.box_size["x"] = d
        self.box_size["y"] = d
        self.box_size["z"] = d

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
                p = self.o_wd
                pathlib.Path(p).mkdir(parents=True, exist_ok=True)
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
            if self.use_temp_eps:
                epsilon = self._eps(T)
            else:
                epsilon = self.water_perm
            l = np.sqrt(cnt.epsilon_0 * epsilon * cnt.Boltzmann * T)
        l = l / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro) * cnt.e)
        return l

    def get_hps_pairs(self, T, from_file=None):
        """
        Build pair_coeff from HPS parameters. If from_file is None, they are automatically generated from those in
        ../../data/hps
        :param from_file: string, path to read the pairs from
        :return: Nothing
        """

        # def good_round(num, decs=6):
        #     num_decs = abs(dec.Decimal(str(num)).as_tuple().exponent)
        #     if num_decs > decs:
        #         d = int(list(str(num))[-2])
        #         str_dec = "0." + "0" * (decs - 1) + "1"
        #         if d < 5:
        #             num = dec.Decimal(num).quantize(dec.Decimal(str_dec), dec.ROUND_DOWN)
        #         elif d >= 5:
        #             num = dec.Decimal(num).quantize(dec.Decimal(str_dec), dec.ROUND_UP)
        #         return float(num)
        #     else:
        #         return round(num,6)

        if T is not None:
            temperature_params = self.temperatures[T]
        else:
            temperature_params = None
        self._get_hps_params(temp=temperature_params)
        # if self.model != 'gHPS' or self.model != 'gHPS-T':
        if self.model == 'HPS' or self.model == 'HPS-T':
            lines = ['pair_coeff          *       *       0.000000   0.000    0.000000   0.000   0.000\n']
        else:
            lines = ['pair_coeff          *       *       0.000000   0.000    0.0000   0.000\n']
        d = ""
        if from_file:
            lambda_gen = np.genfromtxt(from_file)
            count = 0
        for i in range(len(self.key_ordering)):
            for j in range(i, len(self.key_ordering)):
                res_i = self.residue_dict[self.key_ordering[i]]
                res_j = self.residue_dict[self.key_ordering[j]]
                lambda_ij = (res_i["lambda"] + res_j["lambda"]) / 2
                d += f"{lambda_ij} {round(lambda_ij,7)} \n"
                # d += f"{testttt} \n"
                sigma_ij = (res_i["sigma"] + res_j["sigma"]) / 2
                if from_file:
                    lambda_ij = lambda_gen[count]
                    count += 1
                if res_i["q"] != 0 and res_j["q"] != 0:
                    cutoff = 35.00
                else:
                    cutoff = 0.0
                lambda_ij = lambda_ij * self.hps_scale
                if self.model == 'gHPS' or self.model == 'gHPS-T':
                    C6 = -4*self.hps_epsilon*sigma_ij**6*(1+self.a6*(lambda_ij-1))
                    C12 = 4*self.hps_epsilon*sigma_ij**12*(1+(1-self.a6)*(lambda_ij-1))
                    print(-C12/C6 < 0, 1+self.a6*(lambda_ij-1))
                    new_sigma = (-C12/C6)**(1/6)
                    new_eps = 0.25*C6**2/C12
                    line = 'pair_coeff         {:2d}      {:2d}      {:.6f}   {:.3f}    {:6.3f}  {:6.3f}\n'.format(
                        i + 1, j + 1, new_eps, new_sigma, 3 * sigma_ij, cutoff)
                elif self.model.lower() == 'kh':
                    ks = pd.read_csv('/home/adria/scripts/data/hps/kh_f.dat', sep=" ", index_col=0)
                    res_i = self.key_ordering[i]
                    res_j = self.key_ordering[j]
                    eps_ij = ks[res_j][res_i]
                    line = 'pair_coeff         {:2d}      {:2d}       {: .6f}   {:.3f}    {:6.3f} {:6.3f}\n'.format(
                        i + 1, j + 1, eps_ij, sigma_ij, 3 * sigma_ij, cutoff)
                elif self.model.lower() == 'kh-hps':
                    ks = pd.read_csv('/home/adria/scripts/data/hps/kh.dat', sep=" ", index_col=0)
                    res_i = self.key_ordering[i]
                    res_j = self.key_ordering[j]
                    eps_ij = ks[res_j][res_i]
                    epss = eps_ij/self.kh_alpha + self.kh_eps0
                    if epss <= self.kh_eps0:
                        lambda_ij = 1
                    else:
                        lambda_ij = -1
                    eps_ij = abs(eps_ij)
                    line = 'pair_coeff         {:2d}      {:2d}      {: .6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                        i + 1, j + 1, eps_ij, sigma_ij, lambda_ij, 3 * sigma_ij, cutoff)
                else:
                    # if res_i["name"] == "ARG" or res_j["name"] == "ARG":
                    #     ks = pd.read_csv('/home/adria/scripts/data/hps/kh.dat', sep=" ", index_col=0)
                    #     epsilon = ks[self.key_ordering[j]][self.key_ordering[i]]
                    #     epss = epsilon / self.kh_alpha + self.kh_eps0
                    #     if epss <= self.kh_eps0:
                    #         lambda_ij = 1
                    #     else:
                    #         lambda_ij = -1
                    #     epsilon = abs(epsilon)
                    # else:
                    #     epsilon = self.hps_epsilon
                    line = 'pair_coeff         {:2d}      {:2d}      {: .6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                        i + 1, j + 1, self.hps_epsilon, sigma_ij, lambda_ij, 3 * sigma_ij, cutoff)
                lines.append(line)
        self.hps_pairs = lines
        with open('/home/adria/scripts/data/hps/asyn_coeffs.txt', 'w+') as coeff_data:
            coeff_data.write(d)

    def get_lambda_seq(self, window=9):
        """
        Get the windowed charged sequence of the sequence
        :param window: int window to calculate the charge
        :return: ndarray windowed charge of the sequence, ndarray of positive values, ndarray of negative values
        """
        self._get_hps_params(temp=300)
        win = np.zeros(len(self.sequence))
        rr = int((window-1)/2)
        if rr >= 0:
            for i in range(len(self.sequence)):
                c = 0
                for w in range(-rr, rr+1):
                    if len(self.sequence) > i+w > 0:
                        jaa = self.sequence[i+w]
                        c += 1
                        win[i] += self.residue_dict[jaa]["lambda"]
                    else:
                        continue
                win[i] /= window
                # win[i] += self.residue_dict[jaa]["q"]
        else:
            for i in range(len(self.sequence)):
                for j in range(-rr, rr+1):
                    if len(self.sequence) > i+j > 0:
                        jaa = self.sequence[i+j]
                        win[i] += self.residue_dict[jaa]["lambda"]
                        #CORRECTOR
                        # if 1 > abs(self.residue_dict[jaa]["q"]) > 0:
                        #     win[i] += 1 - self.residue_dict[jaa]["q"]
                win[i] /= window
        plus = np.copy(win)
        minus = np.copy(win)
        plus[plus < 0.] = 0
        minus[minus > 0.] = 0
        return win, plus, minus

    # TODO : BROKEN
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
            corr_times_T.append(save_period*np.where(ac==corr_t))
        return corr_times_T

    def _get_hps_params(self, temp):
        """
        Get the HPS parameters to a python dict
        :return:
        """
        def convert_to_HPST(aa):
            l = 0
            if aa["type"].lower() == "hydrophobic":
                l = aa["lambda"] - 25.475 + 0.14537*temp - 0.00020059*temp**2
            elif aa["type"].lower() == "aromatic":
                l = aa["lambda"] - 26.189 + 0.15034*temp - 0.00020920*temp**2
            elif aa["type"].lower() == "other":
                l = aa["lambda"] + 2.4580 - 0.014330*temp + 0.000020374*temp**2
            elif aa["type"].lower() == "polar":
                l = aa["lambda"] + 11.795 - 0.067679*temp + 0.000094114*temp**2
            elif aa["type"].lower() == "charged":
                l = aa["lambda"] + 9.6614 - 0.054260*temp + 0.000073126*temp**2
            else:
                t = aa["type"]
                raise SystemError(f"We shouldn't be here...{t}")
            return l

        for key in self.residue_dict:
            for lam_key in definitions.lambdas:
                if self.residue_dict[key]["name"] == lam_key:
                    self.residue_dict[key]["lambda"] = definitions.lambdas[lam_key]
                    if self.model.upper() == 'HPS-T' or self.model.upper() == 'GHPS-T':
                        self.residue_dict[key]["lambda"] = convert_to_HPST(self.residue_dict[key])

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

        if T is None:
            T_str = ""
        else:
            T_str = str(T)

        self._generate_lmp_input(T)
        self._generate_qsub(perdiu_dir=output_path.replace('/perdiux', ''), T=T_str)
        self._generate_data_input()
        self._generate_slurm(T=T_str)

        if readme:
            self._generate_README()
        if self.xyz is not None and pdb:
            pdb = self._generate_pdb()
            with open(os.path.join(output_path, f'topo.pdb'), 'w+') as f:
                f.write(pdb)

        if self.temper:
            lmp_temp_file = open(os.path.join(self.this, 'templates/lmp/replica/input_template.lmp'))
            lmp_template = Template(lmp_temp_file.read())
            lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)
        else:
            preface_file = open(os.path.join(self.this, 'templates/lmp/general/hps_preface.lmp'))
            preface_template = Template(preface_file.read())
            preface_subst = preface_template.safe_substitute(self.lmp_file_dict)
            if self.slab:
                lmp_temp_file = open(os.path.join(self.this, 'templates/lmp/general/hps_slab.lmp'))
            # elif equil:
            #     lmp_temp_file = open(os.path.join(self.this, 'templates/general/hps_equil.lmp'))
            elif rerun:
                lmp_temp_file = open(os.path.join(self.this, 'templates/lmp/general/hps_rerun.lmp'))
            else:
                lmp_temp_file = open(os.path.join(self.this, 'templates/lmp/general/hps_general.lmp'))

            lmp_template = Template(lmp_temp_file.read())
            lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)

            # TODO : NOT CLEAN...
            if self.fix_region is not None:
                helper = []
                for c in range(self.chains):
                    d = np.array(self.fix_region)
                    d = d + len(self.sequence)*c
                    helper.append(f'{d[0]:.0f}:{d[1]:.0f}')
                helper = ' '.join(helper)
                self.lmp_file_dict["fix_group"] = helper
                fixed = open(os.path.join(self.this, 'templates/lmp/general/hps_fix_rigid.lmp'))
                fixed_template = Template(fixed.read())
                fixed_subst = fixed_template.safe_substitute(self.lmp_file_dict)
            else:
                fixed_subst = ""
            lmp_subst = preface_subst + fixed_subst + lmp_subst

        topo_temp_file = open(os.path.join(self.this, 'templates/lmp/hps_data.data'))
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(self.topo_file_dict)

        if os.path.abspath(output_path):
            if data:
                with open(os.path.join(output_path, 'data.data'), 'tw') as fileout:
                    fileout.write(topo_subst)
            if qsub:
                qsub_temp_file = open(os.path.join(self.this, 'templates/qsub_template.qsub'))
                qsub_template = Template(qsub_temp_file.read())
                qsub_subst = qsub_template.safe_substitute(self.qsub_file_dict)
                with open(os.path.join(output_path, f'{self.job_name}_{T_str}.qsub'), 'tw') as fileout:
                    fileout.write(qsub_subst)
                run_free = True
                if os.path.exists(os.path.join(output_path, f'run.sh')):
                    with open(os.path.join(output_path, f'run.sh'), 'r') as runner:
                        if f'qsub {self.job_name}_{T_str}.qsub \n' in runner.readlines():
                            run_free = False
                if run_free:
                    with open(os.path.join(output_path, f'run.sh'), 'a+') as runner:
                        runner.write(f'qsub {self.job_name}_{T_str}.qsub \n')
                st = os.stat(os.path.join(output_path, f'run.sh'))
                os.chmod(os.path.join(output_path, f'run.sh'), st.st_mode | stat.S_IEXEC)
            if slurm:
                slurm_temp_file = open(os.path.join(self.this, 'templates/slurm_template.slm'))
                slurm_template = Template(slurm_temp_file.read())
                slurm_subst = slurm_template.safe_substitute(self.slurm_file_dict)
                with open(os.path.join(output_path, f'{self.job_name}_{T_str}.slm'), 'tw') as fileout:
                    fileout.write(slurm_subst)
                run_free = True
                if os.path.exists(os.path.join(output_path, f'run.sh')):
                    with open(os.path.join(output_path, f'run.sh'), 'r') as runner:
                        if f'sbatch {self.job_name}_{T_str}.slm \n' in runner.readlines():
                            run_free = False
                if run_free:
                    with open(os.path.join(output_path, f'run.sh'), 'a+') as runner:
                        runner.write(f'sbatch {self.job_name}_{T_str}.slm \n')
                st = os.stat(os.path.join(output_path, f'run.sh'))
                os.chmod(os.path.join(output_path, f'run.sh'), st.st_mode | stat.S_IEXEC)
            if lmp:
                with open(os.path.join(output_path, f'lmp{T_str}.lmp'), 'tw') as fileout:
                    fileout.write(lmp_subst)
            # if rst:
            # rst_temp_file = open(os.path.join(self.this, 'templates/restart/input_template.lmp'))
            # rst_template = Template(rst_temp_file.read())
            # rst_subst = rst_template.safe_substitute(self.lmp_file_dict)
            #     with open(os.path.join(output_path, f'rst{T_str}.lmp'), 'tw') as fileout:
            #         fileout.write(rst_subst)

    def rw_rg(self, monomer_l=5.5):
        rg = monomer_l*(len(self.sequence)/6)**0.5
        return rg

    def scramble_Ps(self, p_prot_name, positions=None):
        with open(os.path.join(definitions.hps_data_dir, f'sequences/{p_prot_name}.seq')) as f:
            phospho_sequence = f.readlines()[0]
        total_phosphos = 0
        if positions is None:
            for i in range(len(self.sequence)):
                if self.sequence[i] != phospho_sequence[i]:
                    total_phosphos += 1
            positions = np.random.randint(0, len(self.sequence), size=(total_phosphos,))
            while len(positions) != total_phosphos:
                positions = np.random.randint(0, len(self.sequence), size=(total_phosphos,))
        seq_helper = list(self.sequence)
        for pos in positions:
            seq_helper[pos] = 'D'
        s = ''.join(seq_helper)
        self.sequence = s

    # TODO : THIS ONLY IF WE HAVE self.xyz... Maybe I can make it general
    # TODO : ALSO THIS IS NOT CENTERED FOR SINGLE CHAIN!!!!
    def _generate_pdb(self, display=None):
        """
        Generate an HPS pdb, since LAMMPS and MDTraj are unable to do it
        :param display: string, switch to print charged aminoacids vs non charged (display=anything except "charged"/None) ; or charged+ vs charged- vs noncharged (display=charged)
        :return: Nothing
        """
        header = f'CRYST1     {self.box_size["x"]:.0f}     {self.box_size["y"]:.0f}     {self.box_size["z"]:.0f}     90     90     90   \n'
        xyz = ''
        c = 0
        for n in range(self.chains):
            for i, aa in enumerate(self.sequence):
                coords = self.xyz[0, c, :]
                coords = coords + [self.box_size["x"]/2, self.box_size["y"]/2, self.box_size["z"]/2]
                if display:
                    if self.residue_dict[aa]["q"] > 0:
                        res_name = 'C' if display == 'charged' else 'P'
                    elif self.residue_dict[aa]["q"] < 0:
                        res_name = 'C' if display == 'charged' else 'M'
                    else:
                        res_name = 'N'
                else:
                    res_name = self.residue_dict[aa]["name"]
                # Empty space is chain ID, which seems to not be strictly necessary...
                # xyz += f'ATOM  {c + 1:>5} {res_name:>4}   {res_name} {" "} {i + 1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      PROT \n'
                tag = "CA"
                xyz += f'ATOM  {c + 1:>5}{tag:>4}  {res_name} {chr(65+n)} {i + 1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      C \n'
                c += 1
            # TODO : Ovito stops reading after first TER...
        bonds = ''
        # for i in range(len(self.sequence) * self.chains):
        #     if (i + 1) % len(self.sequence) != 0: bonds += 'CONECT{:>5}{:>5} \n'.format(i + 1, i + 2)
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
        self.get_hps_pairs(T)
        if self.model == 'gHPS' or self.model=='gHPS-T':
            self.lmp_file_dict["pair_potential"] = 'lj/cut/coul/debye'
            if self.debye_wv:
                self.lmp_file_dict["pair_parameters"] = f"{self.debye_wv} 0.0"
            else:
                self.lmp_file_dict["pair_parameters"] = f"{round(1 / self.debye_length(T) * 10 ** -10, 3)} 0.0"
        elif self.model.lower() == 'kh':
            self.lmp_file_dict["pair_potential"] = 'kh/cut/coul/debye'
            self.lmp_file_dict["pair_parameters"] = f"{self.debye_wv} 0.0 35.0"
        else:
            self.lmp_file_dict["pair_potential"] = 'ljlambda'
            if self.debye_wv:
                self.lmp_file_dict["pair_parameters"] = f"{self.debye_wv} 0.0 35.0"
            else:
                self.lmp_file_dict["pair_parameters"] = f"{round(1 / self.debye_length(T, ) * 10 ** -10, 3)} 0.0 35.0"

        if T is None:
            dcd_dump = f"dcd_traj.dcd"
            lammps_dump = f"atom_traj.lammpstrj"
            log_file = f"log.lammps"
        else:
            dcd_dump = f"dcd_traj_{T}.dcd"
            lammps_dump = f"atom_traj_{T}.lammpstrj"
            log_file = f"log_{T}.lammps"

        self.lmp_file_dict["t"] = self.t
        self.lmp_file_dict["dt"] = self.dt
        self.lmp_file_dict["pair_coeff"] = ''.join(self.hps_pairs)
        self.lmp_file_dict["v_seed"] = self.v_seed
        self.lmp_file_dict["langevin_seed"] = self.langevin_seed
        if T is None and self.temper:
            self.lmp_file_dict["temperatures"] = ' '.join(map(str, self.temperatures))
        else:
            self.lmp_file_dict["temp"] = self.temperatures[T]
        # TODO : Remember to add funcionality when we want constant EPS
        # self.lmp_file_dict["water_perm"] = self.water_perm
        if self.use_temp_eps:
            self.lmp_file_dict["water_perm"] = self._eps(self.temperatures[T])
        else:
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
        self.lmp_file_dict["deformation_ts"] = self.deformation_ts

        self.lmp_file_dict["final_slab_x"] = round(self.slab_dimensions["x"]/2, 2)
        self.lmp_file_dict["final_slab_y"] = round(self.slab_dimensions["y"]/2, 2)
        self.lmp_file_dict["final_slab_z"] = round(self.slab_dimensions["z"]/2, 2)

        self.lmp_file_dict["lammps_dump"] = lammps_dump
        self.lmp_file_dict["hps_scale"] = self.hps_scale
        self.lmp_file_dict["dcd_dump"] = dcd_dump
        self.lmp_file_dict["log_file"] = log_file

        self.lmp_file_dict["slab_t"] = self.slab_t

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
                    masses.append(f'          {i:2d}   {self.residue_dict[key]["mass"]}        #{key} \n')

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
                atoms.append(f'     {k :3d}          {chain}    '
                             f'     {self.residue_dict[aa]["id"]:2d}   '
                             f'   {self.residue_dict[aa]["q"]*self.charge_scale: .2f}'
                             f'    {xyz[0]: <6.3f}'
                             f'    {xyz[1]: .3f}'
                             f'    {xyz[2]: .3f}                 #{aa} \n')
                if k != chain * (len(self.sequence)):
                    bonds.append(f'     {k:3d}       1     {k:3d}     {k + 1:3d}\n')
                k += 1
        if spaghetti:
            self.xyz = np.array([xyzs])
        self.topo_file_dict["natoms"] = self.chains * len(self.sequence)
        self.topo_file_dict["nbonds"] = self.chains * (len(self.sequence) - 1)
        self.topo_file_dict["atom_types"] = len(self.residue_dict)
        self.topo_file_dict["masses"] = ''.join(masses)
        self.topo_file_dict["atoms"] = ''.join(atoms)
        self.topo_file_dict["bonds"] = ''.join(bonds)
        self.topo_file_dict["box_size_x"] = int(self.box_size["x"]/2)
        self.topo_file_dict["box_size_y"] = int(self.box_size["y"]/2)
        self.topo_file_dict["box_size_z"] = int(self.box_size["z"]/2)

    def _generate_qsub(self, perdiu_dir, T):
        """
        Generate a qsub file to run at perdiux's
        :return:
        """
        self.qsub_file_dict["work_dir"] = perdiu_dir
        # TODO : ADD STRING INSTEAD
        if self.temper:
            if self.processors == 1:
                self.qsub_file_dict[
                    "command"] = f"/home/adria/local/lammps3/bin/lmp -in lmp{T}.lmp"
            else:
                self.qsub_file_dict[
                    "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps3/bin/lmp -partition {self.processors}x1 -in lmp.lmp"
        else:
            self.qsub_file_dict[
                "command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps3/bin/lmp -in lmp{T}.lmp -log log_{T}.lammps"
        self.qsub_file_dict["np"] = self.processors
        self.qsub_file_dict["host"] = self.host
        self.qsub_file_dict["jobname"] = self.job_name

    def _generate_slurm(self, T):
        """
        Generate a slm file to run at CSUC
        :return:
        """
        if self.temper:
            self.slurm_file_dict["command"] = f"srun `which lmp` -in lmp.lmp -partition {self.processors}x1"
        else:
            self.slurm_file_dict["command"] = f"srun `which lmp` -in lmp{T}.lmp -log log_{T}.lammps"
        self.slurm_file_dict["np"] = self.processors
        self.slurm_file_dict["jobname"] = self.job_name
        self.slurm_file_dict["in_files"] = f"data.data lmp{T}.lmp"
        self.slurm_file_dict["temp"] = T

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
        out += f'║{" " * param_padding}{f" - Model = {self.model}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Chains = {self.chains}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Ionic Strength (mM) = {self.ionic_strength}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Medium Permittivity = {self.water_perm}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Temperatures (K) = {self.temperatures}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - HPS Scale = {self.hps_scale}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'╚{"═" * (l + padding * 2)}╝'
        print(out)

    def _eps(self, temp):
        # Formula is in celsius
        temp = temp - 273.15
        epsilon = 87.740 - 0.4008*temp + 9.398e-4*temp**2-1.410e-6*temp**3
        if temp > 100:
            epsilon = 87.740 - 0.4008*100 + 9.398e-4*100**2-1.410e-6*100**3
        if temp <= 1:
            epsilon = 80
        return epsilon
