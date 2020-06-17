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


class HMDSetup:
    def __init__(self, oliba_wd, protein, chains=50, **kwargs):

        self.protein = protein
        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]
        self.particles, self.particle_types = self._get_HPS_particles()
        self.chains = chains
        self.o_wd = oliba_wd
        self.this = os.path.dirname(os.path.dirname(__file__))

        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        # Helpers
        self.xyz = None
        self.temp_dict = {}
        self.save = kwargs.get('save', 5000)
        self.water_perm = kwargs.get('water_perm', 80)
        self.topo_path = kwargs.get('topo_path', None)
        self.t = kwargs.get('t', 1e7)

        # HPS Parameters ; HOOMD Units : Energy = kJ/mol, distance = nm, mass = amu
        self.eps = 0.2*4184*10**-3

        # Thermodynamical properties
        self.box_size = kwargs.get('box_size', 200)
        self.temperature = kwargs.get('temperature', 300)
        self.kT = 0.00831445986144858*self.temperature
        self.ionic_strength = kwargs.get('ionic_strength', 300)
        self.context = f"--gpu={len(glob.glob(os.path.join(self.o_wd, '*.py')))}"

        # Slab parameters
        self.slab_t = kwargs.get('slab_t', 200000)
        # self.final_slab_volume = self.box_size["x"]/4
        self.slab_dimensions = {}
        droplet_zlength = 500
        self.slab_dimensions["x"] = (self.chains * 4 * math.pi / 3 / droplet_zlength * self.rw_rg() ** 3) ** 0.5
        self.slab_dimensions["y"] = (self.chains * 4 * math.pi / 3 / droplet_zlength * self.rw_rg() ** 3) ** 0.5
        self.slab_dimensions["z"] = 5 * droplet_zlength
        self.contract_t = 100000
        self.slab_t = kwargs.get('slab_t', 200000)
        self.final_slab_volume = 400 / 4

    def _get_HPS_particles(self):
        residues = dict(definitions.residues)
        for key in residues:
            for lam_key in definitions.lambdas:
                if residues[key]["name"] == lam_key:
                    residues[key]["lambda"] = definitions.lambdas[lam_key]

            for sig_key in definitions.sigmas:
                if residues[key]["name"] == sig_key:
                    residues[key]["sigma"] = definitions.sigmas[sig_key]
        return residues, list(residues.keys())

    def get_topo(self):
        struct = md.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
        struct.center_coordinates()
        rg = md.compute_rg(struct)
        d = rg[0] * self.chains ** (1 / 3) * 8
        struct.unitcell_lengths = np.array([[d, d, d]])

        system = struct
        if self.chains > 1:
            def build_random():
                adder = struct[:]
                dist = np.random.uniform(-d/2, d/2, 3)
                dist -= 0.2 * dist
                adder.xyz[0, :, :] += dist
                chain = 1
                centers_save = [dist]

                def find_used(dist_d, centers_save_d):
                    for i, center in enumerate(centers_save_d):
                        if np.linalg.norm(dist_d - center) < 9:
                            return True
                        else:
                            continue
                    return False

                while chain < self.chains:
                    # TODO : move to gaussian ? Uniform is not uniform in 3d ?
                    dist = np.random.uniform(-d/2, d/2, 3)
                    dist -= 0.2 * (dist)

                    used = find_used(dist, centers_save)
                    if not used:
                        struct.xyz[0, :, :] = struct.xyz[0, :, :] + dist
                        adder = adder.stack(struct)
                        struct.xyz[0, :, :] = struct.xyz[0, :, :] - dist
                        chain += 1
                        centers_save.append(dist)
                return adder

            system = build_random()
        struct.xyz = system.xyz
        struct.unitcell_lengths = np.array([[d, d, d]])
        p = os.path.join(self.o_wd, 'topo.pdb')
        struct.save_pdb(p)
        return 'topo.pdb'

    def rw_rg(self, monomer_l=5.5):
        rg = monomer_l*(len(self.sequence)/6)**0.5
        return rg

    def write_hps_files(self):
        pathlib.Path(self.o_wd).mkdir(parents=True, exist_ok=True)
        if self.topo_path is None:
            self.topo_path = self.get_topo()
        else:
            shutil.copyfile(self.topo_path, os.path.join(self.o_wd, 'topo.pdb'))
            self.topo_path = 'topo.pdb'
        self._build_temp_dict()
        hmd_temp_file = open(os.path.join(self.this, 'templates/general.pyt'))
        hmd_template = Template(hmd_temp_file.read())
        hmd_subst = hmd_template.safe_substitute(self.temp_dict)
        with open(os.path.join(self.o_wd, f'hmd_hps_{self.temperature:.0f}.py'), 'tw') as fileout:
            fileout.write(hmd_subst)
        with open(os.path.join(self.o_wd, f'run.sh'), 'a+') as runner:
            runner.write(f'python3 hmd_hps_{self.temperature:.0f}.py > run_{self.temperature:.0f}.log & \n')
        st = os.stat(os.path.join(self.o_wd, f'run.sh'))
        os.chmod(os.path.join(self.o_wd, f'run.sh'), st.st_mode | stat.S_IEXEC)

    def _build_temp_dict(self):
        self.temp_dict["temperature"] = self.temperature
        self.temp_dict["box_size"] = self.box_size
        self.temp_dict["particles"] = self.particles
        self.temp_dict["particle_types"] = self.particle_types
        self.temp_dict["chains"] = self.chains
        self.temp_dict["protein"] = self.protein
        self.temp_dict["sequence"] = self.sequence
        self.temp_dict["topo_path"] = self.topo_path
        self.temp_dict["explicit_potential_code"] = inspect.getsource(HPS_potential)
        self.temp_dict["save"] = self.save
        self.temp_dict["t"] = self.t
        self.temp_dict["context"] = self.context
        self.temp_dict["water_perm"] = self.water_perm
        self.temp_dict["contract_t"] = self.contract_t
        self.temp_dict["slab_t"] = self.slab_t
        self.temp_dict["final_slab_x"] = round(self.slab_dimensions["x"] / 2, 2)
        self.temp_dict["final_slab_y"] = round(self.slab_dimensions["y"] / 2, 2)
        self.temp_dict["final_slab_z"] = round(self.slab_dimensions["z"] / 2, 2)

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
                hps_table.pair_coeff.set(aa_i, aa_j, func=self.HPS_potential,
                                         rmin=2,
                                         rmax=3,
                                         coeff=dict(eps=0.2*4.184, lambd=lambd, sigma=sigma/10))
        return hps_table
