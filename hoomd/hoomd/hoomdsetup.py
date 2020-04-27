import numpy as np
import hoomd
from hoomd import md
import os
import definitions
import mdtraj


class HPS():
    def __init__(self, protein):
        self.protein = protein
        with open(os.path.join(definitions.hps_data_dir, f'sequences/{protein}.seq')) as f:
            self.sequence = f.readlines()[0]
        self.particles, self.particle_types = self.get_HPS_particles()
        self.rmin = 3.8
        self.rmax=35

    def HPS_potential(self, r, rmin, rmax, eps, lambd, sigma):
        V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
        F = 4 * eps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
        if r <= 2 ** (1 / 6) * sigma:
            V = V + (1 - lambd) * eps
        else:
            V = lambd * V
            F = lambd * F
        return (V, F)

    def get_HPS_pair_table(self, nl):
        hps_table = md.pair.table(width=len(self.sequence), nlist=nl)
        for i in range(len(self.particle_types)):
            aa_i = self.particle_types[i]
            for j in range(i, len(self.particle_types)):
                aa_j = self.particle_types[j]
                lambd = (self.particles[aa_i]["lambda"] + self.particles[aa_j]["lambda"]) / 2
                sigma = (self.particles[aa_i]["sigma"] + self.particles[aa_j]["sigma"]) / 2
                hps_table.pair_coeff.set(aa_i, aa_j, func=self.HPS_potential,
                                         rmin=self.rmin,
                                         rmax=self.rmax,
                                         coeff=dict(eps=0.1, lambd=lambd, sigma=sigma))
        return hps_table

    def get_pos(self):
        struct = mdtraj.load_pdb(os.path.join(definitions.hps_data_dir, f'equil/{self.protein}.pdb'))
        struct.center_coordinates()
        return struct.xyz[0, :, :]

    def build_bonds(self, snap):
        bond_arr = []
        for i, aa in enumerate(self.sequence):
            snap.particles.typeid[i] = self.particles[aa]["id"]
            bond_arr.append([i, i + 1])
        del bond_arr[-1]
        snap.bonds.resize(len(self.sequence) - 1);
        snap.bonds.group[:] = bond_arr[:]

    def get_sequence(self):
        return self.sequence

    def get_HPS_particles(self):
        residues = dict(definitions.residues)
        for key in residues:
            for lam_key in definitions.lambdas:
                if residues[key]["name"] == lam_key:
                    residues[key]["lambda"] = definitions.lambdas[lam_key]

            for sig_key in definitions.sigmas:
                if residues[key]["name"] == sig_key:
                    residues[key]["sigma"] = definitions.sigmas[sig_key]
        return residues, list(residues.keys())
