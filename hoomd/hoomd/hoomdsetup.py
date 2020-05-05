import numpy as np
import hoomd
from hoomd import md
import os
import definitions
import scipy.constants as cnt
import mdtraj


class HPS():
    def __init__(self, protein, **kwargs):
        self.protein = protein
        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]
        self.particles, self.particle_types = self._get_HPS_particles()
        self.chains = kwargs.get('chains', 1)
        self.eps = 0.2*4184*10**-3

    def HPS_potential(self, r, rmin, rmax, eps, lambd, sigma):
        V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
        F = 4 * eps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
        if r <= 2 ** (1 / 6) * sigma:
            V = V + (1 - lambd) * eps
        else:
            V = lambd * V
            F = lambd * F
        return (V, F)

    def build_spaghetti_pos(self, l):
        n = len(self.sequence)
        pos = np.zeros((n, 3))
        pos[:, 0] = np.linspace(-l / 2 * 0.5, l / 2 * 0.5, n)
        return pos

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

    def get_LJ(self, nl):
        hps_table = md.pair.lj(r_cut=3.0, nlist=nl)
        for i in range(len(self.particle_types)):
            aa_i = self.particle_types[i]
            for j in range(i, len(self.particle_types)):
                aa_j = self.particle_types[j]
                hps_table.pair_coeff.set(aa_i, aa_j, epsilon=0.2*4.184, sigma=6.5/10.)
        return hps_table

    def get_HPS_pair_table(self, nl):
        hps_table = md.pair.table(width=len(self.sequence), nlist=nl)
        for i in range(len(self.particle_types)):
            aa_i = self.particle_types[i]
            for j in range(i, len(self.particle_types)):
                aa_j = self.particle_types[j]
                lambd = (self.particles[aa_i]["lambda"] + self.particles[aa_j]["lambda"]) / 2
                sigma = (self.particles[aa_i]["sigma"] + self.particles[aa_j]["sigma"]) / 2
                hps_table.pair_coeff.set(aa_i, aa_j, func=self.HPS_potential,
                                         rmin=0.4,
                                         rmax=3*sigma/10,
                                         coeff=dict(eps=0.2*4.184, lambd=lambd, sigma=sigma/10))
        return hps_table

    def get_pdb_xyz(self):
        """
        Get the xyz of a pdb. If we're demanding more than 1 chain, then generate xyz for every other chain
        so that they are in a cubic setup using the xyz from the single case.
        :param use_random:
        :param pdb: string, path leading to the pdb we want to use
        :param padding: float, how close we want the copies in the multichain case
        :return: Nothing
        """

        struct = md.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
        struct.center_coordinates()
        rg = md.compute_rg(struct)
        d = rg[0] * self.chains ** (1 / 3) * 8
        struct.unitcell_lengths = np.array([[d, d, d]])

        def build_random():
            adder = struct[:]
            dist = np.random.uniform(-d/2, d/2, 3)
            dist -= 0.2 * (dist)
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
        return system.xyz *10.

    def get_pos(self):
        struct = mdtraj.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
        struct.center_coordinates()
        return struct.xyz[0, :, :]*10.

    def build_bonds(self, snap):
        bond_arr = []
        for chain in range(self.chains):
            for i, aa in enumerate(self.sequence):
                j = i + chain*len(self.sequence)
                snap.particles.typeid[j] = self.particles[aa]["id"]-1
                snap.particles.mass[j] = self.particles[aa]["mass"]

                if self.particles[aa]["q"] != 0:
                    snap.particles.charge[j] = self.particles[aa]["q"]
                bond_arr.append([j, j + 1])
            del bond_arr[-1]
        snap.bonds.resize((len(self.sequence) - 1)*self.chains)
        snap.bonds.group[:] = bond_arr

    def temperature_to_kT(self, T):
        k = cnt.Boltzmann*cnt.Avogadro/4184
        return k * T

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
