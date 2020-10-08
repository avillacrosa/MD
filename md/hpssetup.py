import pandas as pd
import os
import numpy as np
import mdtraj as md
import math
import scipy.constants as cnt


class HPSSetup:

    def __init__(self, md_dir, protein, model, chains, **kwargs):

        #----- MD -----#
        self.t = kwargs.get('t', 100000000)
        self.dt = kwargs.get('dt', 10.)
        self.chains = chains
        self.ionic_strength = kwargs.get('ionic_strength', 100e-3) # In Molar
        self.temperatures = kwargs.get('temperatures', [300, 320, 340, 360, 380, 400])
        self.box_size = {"x": 2500, "y": 2500, "z": 2500}
        self.water_perm = kwargs.get('water_perm', 80.)
        self.use_temp_eps = kwargs.get('use_temp_eps',False)
        self.v_seed = kwargs.get('v_seed', 494211)
        self.langevin_seed = kwargs.get('langevin_seed', 451618)
        self.save = kwargs.get('save', 10000)
        self.langevin_damp = kwargs.get('langevin_damp', 10000)
        self.processors = kwargs.get('processors', 1)
        self.charge_scale = kwargs.get('charge_scale', 1)
        self.debye = kwargs.get('debye', 0.1)
        self.fix_region = kwargs.get('fix_region', None)
        self.fixed_atoms = None
        self.temper = kwargs.get('temper', False)

        #----- HPS -----#
        self.hps_epsilon = kwargs.get('hps_epsilon', 0.2)
        self.hps_scale = kwargs.get('hps_scale', 1.0)

        #----- gHPS ----#
        self.a6 = kwargs.get('a6', 0.8)

        #----- KH -----#
        self.kh_alpha = 0.228
        self.kh_eps0 = -1

        #----- SLAB -----#
        self.slab_t = kwargs.get('slab_t', 200000)
        self.slab_method = kwargs.get('slab_method', 'classical')
        self.contract_t = kwargs.get('contract_t', 100000)
        self.slab_dimensions = {}
        self.slab_dimensions["x"] = 130
        self.slab_dimensions["y"] = self.slab_dimensions["x"]
        self.slab_dimensions["z"] = 2300
        self.final_slab_volume = self.box_size["x"]/4
        self.deformation_ts = kwargs.get('deformation_ts', 1)

        #---- PYTHON ----#
        self.this = os.path.dirname(os.path.dirname(__file__))
        self.residue_dict, self.residue_df = self._get_residue_dict()
        self.key_ordering = list(self.residue_dict.keys())
        with open(os.path.join(self.this, f'md/data/sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]
        self.model = model
        self.md_dir = md_dir
        self.model = model
        self.protein = protein
        self.job_name = kwargs.get('job_name', f's{self.hps_scale}_x{self.chains}-{self.protein}')
        self.lmp = None
        self.slab = kwargs.get('slab', False)
        self.swap_every = 1000
        self.rerun_dump = None
        self.rerun_skip = 0
        self.rerun_start = 0
        self.rerun_stop = 0
        self.host = kwargs.get('host', "")
        self.use_random = kwargs.get('use_random', False)
        self.xyz = None

    # TODO REVIEW...
    def debye_length(self, temp_K=None):
        """
        Calculate debye Length for the given system. If T is None (in the REX chase for example) then assume 300K
        :return: float, debvye length
        """
        # TODO : INCORPORATE SELF.TEMPERATURES
        if temp_K is None:
            l = np.sqrt(cnt.epsilon_0 * self.water_perm * cnt.Boltzmann * 300)
        else:
            if self.use_temp_eps:
                epsilon = self._eps(temp_K)
            else:
                epsilon = self.water_perm
            l = np.sqrt(cnt.epsilon_0 * epsilon * cnt.Boltzmann * temp_K)
        l = l / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro) * cnt.e)
        return l

    def rw_rg(self, monomer_l=5.5):
        rg = monomer_l*(len(self.sequence)/6)**0.5
        return rg

    def _get_topo(self):
        struct = md.load_pdb(os.path.join(self.this, f'md/data/equil/{self.protein}.pdb'))
        struct.xyz = struct.xyz*10.
        struct.center_coordinates()
        d = 4.0 * self.rw_rg(monomer_l=5.5) * (self.chains * 4 * math.pi / 3) ** (1 / 3)
        struct.unitcell_lengths = np.array([[d, d, d]])

        system = struct
        if self.chains > 1:
            def build_random():
                adder = struct[:]
                dist = np.random.uniform(-d / 2, d / 2, 3)
                dist -= 0.4 * dist
                adder.xyz[0, :, :] += dist
                chain = 1
                centers_save = [dist]
                rg_ext = self.rw_rg(monomer_l=5.5)

                def find_used(dist_d, centers_save_d):
                    for i, center in enumerate(centers_save_d):
                        # Dynamic Rg scale based on equilibration Rg ?
                        if np.linalg.norm(dist_d - center) < rg_ext * 3.0:
                            return True
                        else:
                            continue
                    return False

                while chain < self.chains:
                    # TODO : move to gaussian ? Uniform is not uniform in 3d ?
                    dist = np.random.uniform(-d / 2, d / 2, 3)
                    # TODO: REMOVE THIS QUICK DIRTY TRICK
                    dist -= 0.4 * dist

                    used = find_used(dist, centers_save)
                    if not used:
                        struct.xyz[0, :, :] = struct.xyz[0, :, :] + dist
                        adder = adder.stack(struct)
                        struct.xyz[0, :, :] = struct.xyz[0, :, :] - dist
                        chain += 1
                        centers_save.append(dist)
                return adder

            system = build_random()
        self.xyz = system.xyz
        self.box_size["x"] = d
        self.box_size["y"] = d
        self.box_size["z"] = d
        system.unitcell_lengths = np.array([[d, d, d]])

    def _get_sigmas(self, sigmas_file=None):
        rd = self.residue_dict
        if sigmas_file is None:
            sigmas_file = os.path.join(self.this, 'md/data/hps/sigmas.dat')
        sigmas_data = pd.read_csv(sigmas_file, sep=" ", header=0, index_col=0).drop("RES",1)
        sigmas_df = pd.DataFrame(np.zeros(shape=(len(rd), len(rd))), index=rd.keys(), columns=rd.keys())
        for aa_i in rd:
            for aa_j in rd:
                sigmas_df[aa_i][aa_j] = (sigmas_data["SIGMA"][aa_i] + sigmas_data["SIGMA"][aa_j]) / 2
        return sigmas_df

    def _get_HPS_params(self, temp_K, lambdas_file=None):
        """
        Get the HPS parameters to a python dict
        :return:
        """

        if lambdas_file is None:
            lambdas_file = os.path.join(self.this, 'md/data/hps/lambdas.dat')
        lambdas_data = pd.read_csv(lambdas_file, sep=" ", header=0, index_col=0).drop("RES",1)
        def convert_to_HPST(key):
            if bd[key]["type"].lower() == "hydrophobic":
                l = lambdas_data["LAMBDA"][key] - 25.475 + 0.14537 * temp_K - 0.00020059 * temp_K ** 2
            elif bd[key]["type"].lower() == "aromatic":
                l = lambdas_data["LAMBDA"][key] - 26.189 + 0.15034 * temp_K - 0.00020920 * temp_K ** 2
            elif bd[key]["type"].lower() == "other":
                l = lambdas_data["LAMBDA"][key] + 2.4580 - 0.014330 * temp_K + 0.000020374 * temp_K ** 2
            elif bd[key]["type"].lower() == "polar":
                l = lambdas_data["LAMBDA"][key] + 11.795 - 0.067679 * temp_K + 0.000094114 * temp_K ** 2
            elif bd[key]["type"].lower() == "charged":
                l = lambdas_data["LAMBDA"][key] + 9.6614 - 0.054260 * temp_K + 0.000073126 * temp_K ** 2
            else:
                t = bd[key]["type"]
                raise SystemError(f"We shouldn't be here...{t}")
            return l
        bd = self.residue_dict
        lambdas_df = pd.DataFrame(np.zeros(shape=(len(bd), len(bd))), index=bd.keys(), columns=bd.keys())
        if self.model.lower() == 'hps-t' or self.model.lower() == 'ghps-t':
            for key in self.residue_dict:
                # self.residue_dict[key]["lambda"] = convert_to_HPST(self.residue_dict[key])
                lambdas_data["LAMBDA"][key] = convert_to_HPST(key)
        for aa_i in bd:
            for aa_j in bd:
                lambdas_df[aa_i][aa_j] = (lambdas_data["LAMBDA"][aa_i] + lambdas_data["LAMBDA"][aa_j]) / 2

        epsilons = pd.DataFrame(np.ones(shape=(len(bd), len(bd)))*self.hps_epsilon, index=bd.keys(), columns=bd.keys())

        return lambdas_df, epsilons

    def _get_KH_params(self):
        miya_jern = pd.read_csv(os.path.join(self.this,'md/data/kh_f.dat', sep=" ", index_col=0))

        bd = self.residue_dict
        lambdas = pd.DataFrame(np.zeros(shape=(len(bd), len(bd))), index=bd.keys(), columns=bd.keys())
        epsilons = pd.DataFrame(np.zeros(shape=(len(bd), len(bd))), index=bd.keys(), columns=bd.keys())
        for aa_i in bd:
            for aa_j in bd:
                epss = miya_jern[aa_i][aa_j] / self.kh_alpha + self.kh_eps0
                lambda_ij = 1 if epss <= self.kh_eps0 else -1
                lambdas[aa_i][aa_j] = lambda_ij
        epsilons.values = np.abs(epsilons.values)
        return lambdas, epsilons

    def _get_interactions(self, temp_K):
        if self.model.lower() == 'hps' or self.model.lower() == 'hps-t':
            lambdas, epsilons = self._get_HPS_params(temp_K)
        elif self.model.lower() == 'kh' or self.model.lower() == 'kh-hps':
            lambdas, epsilons = self._get_KH_params()
        else:
            raise SystemError("Unknown model")
        return lambdas, epsilons

    def _get_residue_dict(self, residue_file=None, lambdas_file=None, sigmas_file=None):
        if residue_file is None:
            residue_file = os.path.join(self.this, 'md/data/general/residues.txt')
        try:
            residues = pd.read_csv(residue_file, sep=" ", header=0, index_col=0)
        except:
            raise SystemError("Could not load lambda, sigma or residue text file")

        residues.columns = ["name", "mass", "q", "type"]
        residues.insert(0, "id", np.arange(1, len(residues.index) + 1))
        return residues.to_dict(orient='index'), residues

    def _write_pdb(self, display=None):
        header = f'CRYST1     {self.box_size["x"]:.0f}     {self.box_size["y"]:.0f}     {self.box_size["z"]:.0f}        90     90     90   \n'
        xyz = ''
        c = 0
        for n in range(self.chains):
            for i, aa in enumerate(self.sequence):
                coords = self.xyz[0, c, :]
                coords = coords + [self.box_size["x"] / 2, self.box_size["y"] / 2, self.box_size["z"] / 2]
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
                xyz += f'ATOM  {c + 1:>5}{tag:>4}  {res_name} {chr(65 + n)} {i + 1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      C \n'
                c += 1
            # TODO : Ovito stops reading after first TER...
        bonds = ''
        # for i in range(len(self.sequence) * self.chains):
        #     if (i + 1) % len(self.sequence) != 0: bonds += 'CONECT{:>5}{:>5} \n'.format(i + 1, i + 2)
        bottom = 'END \n'
        pdb = header + xyz + bonds + bottom
        with open(os.path.join(self.md_dir, f'topo.pdb'), 'w+') as f:
            f.write(pdb)

    def get_lambda_seq(self, window=9):
        """
        Get the windowed hydrophobicity of the sequence
        :param window: int window to calculate the charge
        :return: ndarray windowed charge of the sequence, ndarray of positive values, ndarray of negative values
        """
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

    def scramble_Ps(self, p_prot_name, positions=None):
        with open(os.path.join(self.this, f'md/data/sequences/{p_prot_name}.seq')) as f:
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

    def _generate_README(self):
        """
        Generate a custom README file containing the most essential parameters of an HPS LAMMPS run
        :return:
        """
        with open(os.path.join(self.md_dir, 'README.txt'), 'tw') as readme:
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
        Print a recopilatory message of the run
        :param padding: int, General padding
        :param section_padding: int, PARAMETERS section padding
        :param param_padding: int, parameters padding
        :return: Nothing
        """
        title = f"Input files created at {self.md_dir} for {self.protein}"
        l = len(title)
        out = ''
        out += f'╔{"═" * (l + padding * 2)}╗\n'
        out += f'║{" " * padding}{title:<{l}}{" " * padding}║\n'
        out += f'║{"-"*(len(title)+padding*2)}║\n'
        out += f'║{" " * section_padding}{"PARAMETERS":<{l}}{" " * (padding + padding - section_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Model = {self.model}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Chains = {self.chains}":<{l}}{" " * (padding + padding - param_padding)}║\n'
        out += f'║{" " * param_padding}{f" - Debye length = {self.debye}":<{l}}{" " * (padding + padding - param_padding)}║\n'
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

