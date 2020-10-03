import numpy as np
import mdtraj as md
import os
import inspect
import definitions
import pathlib
import stat
import shutil
import glob
import math
from string import Template


def HPS_potential(r, rmin, rmax, eps, lambd, sigma):
    V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F = 4 * eps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    if r <= 2 ** (1 / 6) * sigma:
        V = V + (1 - lambd) * eps
    else:
        V = lambd * V
        F = lambd * F
    return (V, F)


def HPS_pi_potential(r, rmin, rmax, eps_hps, eps_pi, lambd, sigma):
    V = 4 * eps_hps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F = 4 * eps_hps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    V_pi = eps_pi * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F_pi = eps_pi / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    if r <= 2 ** (1 / 6) * sigma:
        V = V + (1 - lambd) * eps_hps + V_pi
        F = F + F_pi
    else:
        V = lambd * V + V_pi
        F = lambd * F + F_pi
    return (V, F)


class HMDSetup:
    def __init__(self, oliba_wd, protein, chains=1, **kwargs):
        self.model = kwargs.get('model', 'HPS')
        self.temperature = kwargs.get('temperature', 300)
        self.protein = protein
        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]
        self.particles, self.particle_types = self._get_HPS_particles()
        self.chains = chains
        self.o_wd = oliba_wd
        self.this = os.path.dirname(os.path.dirname(__file__))
        self.processors = kwargs.get('processors', None)

        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        # Helpers
        self.xyz = None
        self.eps_pi = kwargs.get('eps_pi', 3)
        self.temp_dict = {}
        self.aux_dict = {}
        self.slurm_file_dict = {}
        self.save = kwargs.get('save', 10000)
        self.water_perm = kwargs.get('water_perm', 80)
        self.topo_path = kwargs.get('topo_path', None)
        self.t = kwargs.get('t', 1e7)

        # HPS Parameters ; HOOMD Units : Energy = kJ/mol, distance = nm, mass = amu
        self.eps = 0.2*4184*10**-3
        self.hps_scale = kwargs.get("hps_scale", 1.)
        self.debye = kwargs.get("debye", 1.)

        # Thermodynamical properties
        # self.box_size = kwargs.get('box_size', 200)
        self.box_size = {}
        self.kT = 0.00831445986144858*self.temperature
        self.ionic_strength = kwargs.get('ionic_strength', 300)
        if self.processors is not None:
            self.context = ""
        else:
            self.context = f"--gpu={len(glob.glob(os.path.join(self.o_wd, '*.py')))}"
        self.residue_dict = dict(definitions.residues)

        # Slab parameters
        # self.final_slab_volume = self.box_size["x"]/
        self.slab = kwargs.get('slab', False)
        self.slab_method = kwargs.get('slab_method', 'classical')
        self.slab_dimensions = {}
        droplet_zlength = 60
        ## 1.2 factor to give space for more expanded than usual configurations
        # self.slab_dimensions["x"] = 1.5*(self.chains * 4 * math.pi / 3 / droplet_zlength * self.rw_rg() ** 3) ** 0.5
        # self.slab_dimensions["x"] = 1.0*(self.chains * 4 * math.pi / 3 / droplet_zlength * self.rw_rg() ** 3) ** 0.5
        self.slab_dimensions["x"] = 13
        self.slab_dimensions["y"] = self.slab_dimensions["x"]
        self.slab_dimensions["z"] = 250
        self.contract_t = kwargs.get('contract_t', 100000)
        self.slab_t = kwargs.get('slab_t', 1000000)
        self.final_slab_volume = 400 / 4
        self.job_name = kwargs.get('job_name', f's{self.hps_scale}_x{self.chains}-{self.protein}')

    def _get_HPS_particles(self):
        def convert_to_HPST(aa):
            l = 0
            temp = self.temperature
            if aa["type"].lower() == "hydrophobic":
                l = aa["lambda"] - 25.475 + 0.14537 * temp - 0.00020059 * temp ** 2
            elif aa["type"].lower() == "aromatic":
                l = aa["lambda"] - 26.189 + 0.15034 * temp - 0.00020920 * temp ** 2
            elif aa["type"].lower() == "other":
                l = aa["lambda"] + 2.4580 - 0.014330 * temp + 0.000020374 * temp ** 2
            elif aa["type"].lower() == "polar":
                l = aa["lambda"] + 11.795 - 0.067679 * temp + 0.000094114 * temp ** 2
            elif aa["type"].lower() == "charged":
                l = aa["lambda"] + 9.6614 - 0.054260 * temp + 0.000073126 * temp ** 2
            else:
                t = aa["type"]
                raise SystemError(f"We shouldn't be here...{t}")
            return l
        residues = dict(definitions.residues)

        for key in residues:
            for lam_key in definitions.lambdas:
                if residues[key]["name"] == lam_key:
                    residues[key]["lambda"] = definitions.lambdas[lam_key]

            for sig_key in definitions.sigmas:
                if residues[key]["name"] == sig_key:
                    residues[key]["sigma"] = definitions.sigmas[sig_key]

        if self.model.lower() == 'hps-t':
            for key in residues:
                residues[key]["lambda"] = convert_to_HPST(residues[key])

        return residues, list(residues.keys())

    def get_topo(self):
        struct = md.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
        struct.center_coordinates()
        # struct.xyz = struct.xyz
        d = 4.0 * self.rw_rg(monomer_l=0.55) * (self.chains * 4 * math.pi / 3) ** (1 / 3)
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
                rg_ext = self.rw_rg(monomer_l=0.55)

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

    def rw_rg(self, monomer_l=0.55):
        rg = monomer_l*(len(self.sequence)/6)**0.5
        return rg

    def write_hps_files(self, slurm=False):
        pathlib.Path(self.o_wd).mkdir(parents=True, exist_ok=True)
        # if self.topo_path is None:
        #     self.topo_path = self.get_topo()
        # else:
        #     shutil.copyfile(self.topo_path, os.path.join(self.o_wd, 'topo.pdb'))
        self.get_topo()
        self._generate_slurm()
        self.topo_path = 'topo.pdb'
        pdb = self._generate_pdb()
        self._build_temp_dict()

        if self.model.lower() == 'hps-cat':
            preface_file = open(os.path.join(self.this, 'templates/hmd/preface_pi.pyt'))
        else:
            preface_file = open(os.path.join(self.this, 'templates/hmd/preface.pyt'))
        preface_template = Template(preface_file.read())
        preface_subst = preface_template.safe_substitute(self.temp_dict)

        if self.slab is True:
            hmd_temp_file = open(os.path.join(self.this, 'templates/hmd/slab.pyt'))
        elif self.slab_method.lower() == 'npt':
            hmd_temp_file = open(os.path.join(self.this, 'templates/hmd/slab_npt.pyt'))
        else:
            hmd_temp_file = open(os.path.join(self.this, 'templates/hmd/general.pyt'))
        hmd_template = Template(hmd_temp_file.read())
        hmd_subst = hmd_template.safe_substitute(self.temp_dict)

        hmd_file = preface_subst + hmd_subst
        # hmd_aux_file = open(os.path.join(self.this, 'templates/aux.pyt'))
        # hmd_template_aux = Template(hmd_aux_file.read())
        # hmd_subst_aux = hmd_template_aux.safe_substitute(self.aux_dict)

        with open(os.path.join(self.o_wd, f'topo.pdb'), 'w+') as f:
            f.write(pdb)
        with open(os.path.join(self.o_wd, f'hmd_hps_{self.temperature:.0f}.py'), 'tw') as fileout:
            fileout.write(hmd_file)
        # with open(os.path.join(self.o_wd, f'aux.py'), 'tw') as aux:
        #     aux.write(hmd_subst_aux)
        with open(os.path.join(self.o_wd, f'run.sh'), 'a+') as runner:
            runner.write(f'python3 hmd_hps_{self.temperature:.0f}.py > run_{self.temperature:.0f}.log & \n')
        st = os.stat(os.path.join(self.o_wd, f'run.sh'))
        os.chmod(os.path.join(self.o_wd, f'run.sh'), st.st_mode | stat.S_IEXEC)
        if slurm:
            slurm_temp_file = open(os.path.join(self.this, 'templates/hmd_slurm_template.slm'))
            slurm_template = Template(slurm_temp_file.read())
            slurm_subst = slurm_template.safe_substitute(self.slurm_file_dict)
            with open(os.path.join(self.o_wd, f'{self.job_name}_{self.temperature:.0f}.slm'), 'tw') as fileout:
                fileout.write(slurm_subst)
            run_free = True
            if os.path.exists(os.path.join(self.o_wd, f'run.sh')):
                with open(os.path.join(self.o_wd, f'run.sh'), 'r') as runner:
                    if f'sbatch {self.job_name}_{self.temperature:.0f}.slm \n' in runner.readlines():
                        run_free = False
            if run_free:
                with open(os.path.join(self.o_wd, f'run.sh'), 'a+') as runner:
                    runner.write(f'sbatch {self.job_name}_{self.temperature:.0f}.slm \n')
            st = os.stat(os.path.join(self.o_wd, f'run.sh'))
            os.chmod(os.path.join(self.o_wd, f'run.sh'), st.st_mode | stat.S_IEXEC)

    def _generate_pdb(self, display=None):
        """
        Generate an HPS pdb, since LAMMPS and MDTraj are unable to do it
        :param display: string, switch to print charged aminoacids vs non charged (display=anything except "charged"/None) ; or charged+ vs charged- vs noncharged (display=charged)
        :return: Nothing
        """
        header = f'CRYST1     {self.box_size["x"]*10.:.0f}     {self.box_size["y"]*10.:.0f}     {self.box_size["z"]*10.:.0f}     90     90     90   \n'
        xyz = ''
        c = 0
        for n in range(self.chains):
            for i, aa in enumerate(self.sequence):
                coords = self.xyz[0, c, :]
                coords = coords + [self.box_size["x"]/2, self.box_size["y"]/2, self.box_size["z"]/2]
                coords = coords*10.
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
                xyz += f'ATOM  {c + 1:>5} {res_name:>4}   {res_name} {" "} {i + 1:>3}    {coords[0]:>8.4f}{coords[1]:>8.4f}{coords[2]:>8.4f}  1.00  0.00      PROT \n'
                c += 1
            # TODO : Ovito stops reading after first TER...
        bonds = ''
        # for i in range(len(self.sequence) * self.chains):
        #     if (i + 1) % len(self.sequence) != 0: bonds += 'CONECT{:>5}{:>5} \n'.format(i + 1, i + 2)
        bottom = 'END \n'
        return header + xyz + bonds + bottom

    def _build_temp_dict(self):
        self.temp_dict["temperature"] = self.temperature
        print(self.temperature, self.temp_dict["temperature"])
        self.temp_dict["box_size"] = self.box_size["x"]
        self.temp_dict["particles"] = self.particles
        self.temp_dict["particle_types"] = self.particle_types
        self.temp_dict["chains"] = self.chains
        self.temp_dict["protein"] = self.protein
        self.temp_dict["sequence"] = self.sequence
        self.temp_dict["topo_path"] = self.topo_path
        self.temp_dict["explicit_potential_code"] = inspect.getsource(HPS_potential)
        self.temp_dict["explicit_pi_potential_code"] = inspect.getsource(HPS_pi_potential)
        self.temp_dict["save"] = self.save
        self.temp_dict["t"] = self.t
        self.temp_dict["context"] = self.context
        self.temp_dict["water_perm"] = self.water_perm
        self.temp_dict["contract_t"] = self.contract_t
        self.temp_dict["slab_t"] = self.slab_t
        self.temp_dict["final_slab_x"] = round(self.slab_dimensions["x"], 2)
        self.temp_dict["final_slab_y"] = round(self.slab_dimensions["y"], 2)
        self.temp_dict["final_slab_z"] = round(self.slab_dimensions["z"], 2)
        self.temp_dict["hps_scale"] = self.hps_scale
        self.temp_dict["debye"] = self.debye

        self.aux_dict["particles"] = self.particles
        self.aux_dict["particle_types"] = self.particle_types
        self.aux_dict["xyz"] = self.xyz.tolist()

    # Deprecated
    def get_LJ(self, nl):
        hps_table = md.pair.lj(r_cut=2.5, nlist=nl)
        for i in range(len(self.particle_types)):
            aa_i = self.particle_types[i]
            for j in range(i, len(self.particle_types)):
                aa_j = self.particle_types[j]
                hps_table.pair_coeff.set(aa_i, aa_j, epsilon=0.2*4.184, sigma=0.65)
        return hps_table

    def get_dipole_pair_table(self, nl):
        dipole_table = md.pair.dipole(r_cut=35.0, nlist=nl)
        for i in range(len(self.particle_types)):
            aa_i = self.particle_types[i]
            for j in range(i, len(self.particle_types)):
                aa_j = self.particle_types[j]
                if self.particles[aa_i]["q"] != 0 and self.particles[aa_j]["q"] != 0:
                    dipole_table.pair_coeff.set(aa_i, aa_j, mu=0.0, kappa=0.1, A=1.0)
                else:
                    dipole_table.pair_coeff.set(aa_i, aa_j, mu=0.0, kappa=100, A=0.0)

    def get_HPS_pair_table(self, nl):
        hps_table = md.pair.table(width=len(self.sequence), nlist=nl)
        for i in range(len(self.particle_types)):
            aa_i = self.particle_types[i]
            for j in range(i, len(self.particle_types)):
                aa_j = self.particle_types[j]
                lambd = (self.particles[aa_i]["lambda"] + self.particles[aa_j]["lambda"]) / 2
                sigma = (self.particles[aa_i]["sigma"] + self.particles[aa_j]["sigma"]) / 2
                print(lambd*self.hps_scale)
                hps_table.pair_coeff.set(aa_i, aa_j, func=HPS_potential,
                                         rmin=2,
                                         rmax=3,
                                         coeff=dict(eps=0.2*4.184, lambd=lambd, sigma=sigma/10))
        return hps_table

    def _generate_slurm(self):
        """
        Generate a slm file to run at CSUC
        :return:
        """
        if self.processors is not None:
            self.slurm_file_dict["command"] = f"python hmd_hps_{self.temperature:.0f}.py"
            self.slurm_file_dict["tasktype"] = f"--ntasks={self.processors}"
            self.slurm_file_dict["queue"] = "std"
            self.slurm_file_dict["activate"] = "hoomd_cpu"
        else:
            self.slurm_file_dict["command"] = f"mpirun -np 1 python hmd_hps_{self.temperature:.0f}.py  --mode=gpu"
            self.slurm_file_dict["tasktype"] = "--gres=gpu:1"
            self.slurm_file_dict["queue"] = "gpu"
            self.slurm_file_dict["activate"] = "hoomd_cpu"

        self.slurm_file_dict["jobname"] = self.job_name+f"-{self.temperature:.0f}"
