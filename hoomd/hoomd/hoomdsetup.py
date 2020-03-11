import definitions
import hoomd
import numpy as np
from hoomd import md

class HOOMDSetup():
    def __init__(self, oliba_wd, protein, temper, chains=1):
        self.o_wd = oliba_wd
        self.temper = temper
        self.chains = chains
        self.residue_dict = dict(definitions.residues)
        self.residue_types = list(self.residue_dict.keys())

        with open(os.path.join('../data/sequences', f'{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        self.xyz = None
        self.snap = None

    def build_snapshot(self, box):
        snap = hoomd.data.make_snapshot(len(self.sequence), box=box, bond_types=['harmonic'], particle_types=self.residue_types)
        self.get_pdb_xyz(pdb='/home/adria/scripts/hoomd/data/equil/CPEB4.pdb')
        snap.particles.position[:] = self.xyz
        self.snap = snap
        return snap

    def build_bonds(self, snap):
        bond_arr = []
        for i, aa in enumerate(seq):
            snap.particles.typeid[i] = self.residue_dict[aa]["id"]
            bond_arr.append([i, i+1])
        # TODO Use residue to cut last
        del bond_arr[-1]
        snap.bonds.resize(len(seq)-1)
        snap.bonds.group[:] = bond_arr[:]
        return snap

    def build_hps_pairs(self):
        nl = md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=3.0,nlist=nl)

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

    def build_HPS(self):
        print("TODO")

    def build_HPS(self):
        print("TODO")

    def replica(self):
        print("TODO")

    def HPS(self, r, rmin, rmax, eps, lambd, sigma):
        V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
        F = 4 * eps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
        if r <= 2**(1/6)*sigma:
            V = V + (1-lambd)*eps
        else:
            V = lambd*V
            F = lambd*F
        return (V, F)