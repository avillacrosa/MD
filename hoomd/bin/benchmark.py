import hmdsetup
import hoomd
from hoomd import md
import mdtraj
import definitions
import os


def HPS_potential(r, rmin, rmax, eps, lambd, sigma):
    V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F = 4 * eps / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)
    if r <= 2 ** (1 / 6) * sigma:
        V = V + (1 - lambd) * eps
    else:
        V = lambd * V
        F = lambd * F
    return (V, F)


def _get_HPS_particles():
    residues = dict(definitions.residues)
    for key in residues:
        for lam_key in definitions.lambdas:
            if residues[key]["name"] == lam_key:
                residues[key]["lambda"] = definitions.lambdas[lam_key]

        for sig_key in definitions.sigmas:
            if residues[key]["name"] == sig_key:
                residues[key]["sigma"] = definitions.sigmas[sig_key]
    return residues, list(residues.keys())

temperature = 300.
l = 200
chains = 50
particles, particle_types = _get_HPS_particles()
protein = 'CPEB4'
chains = 50
sequence = 'MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE'

hoomd.context.initialize("")
box = hoomd.data.boxdim(Lx=l, Ly=l, Lz=l)
snap = hoomd.data.make_snapshot(N=chains*len(sequence),
                                box=box,
                                bond_types=['harmonic'],
                                particle_types=particle_types)

poss = mdtraj.load_pdb("/home/adria/xoriguer/adria/prod/topox50.pdb")
poss.center_coordinates()
snap.particles.position[:] = poss.xyz

bond_arr = []
for chain in range(chains):
    for i, aa in enumerate(sequence):
        j = i + chain*len(sequence)
        snap.particles.typeid[j] = particles[aa]["id"]-1
        snap.particles.mass[j] = particles[aa]["mass"]
        snap.particles.diameter[j] = particles[aa]["sigma"]

        if particles[aa]["q"] != 0:
            snap.particles.charge[j] = particles[aa]["q"]
        bond_arr.append([j, j + 1])
    del bond_arr[-1]
snap.bonds.resize((len(sequence) - 1)*chains)
snap.bonds.group[:] = bond_arr

hoomd.init.read_snapshot(snap)

harmonic = md.bond.harmonic()

nl = md.nlist.cell()

harmonic.bond_coeff.set('harmonic', k=10.*4.184, r0=0.38)

hps_table = md.pair.table(width=len(sequence), nlist=nl)
for i in range(len(particle_types)):
    aa_i = particle_types[i]
    for j in range(i, len(particle_types)):
        aa_j = particle_types[j]
        lambd = (particles[aa_i]["lambda"] + particles[aa_j]["lambda"]) / 2
        sigma = (particles[aa_i]["sigma"] + particles[aa_j]["sigma"]) / 2
        hps_table.pair_coeff.set(aa_i, aa_j, func=HPS_potential,
                                 rmin=0.1,
                                 rmax=3*sigma/10,
                                 coeff=dict(eps=0.2*4.184, lambd=lambd, sigma=sigma/10))

hoomd.analyze.log(filename="log.log",
                  quantities=['potential_energy', 'temperature'],
                  period=5000,
                  overwrite=True)
hoomd.dump.gsd(filename=f"trajectory_{temperature}.gsd", period=5000, group=hoomd.group.all(), overwrite=True)
hoomd.dump.dcd(filename="trajectory.dcd", period=5000, group=hoomd.group.all(), overwrite=True)

# dt in picosecond units (10**-12) while dt in femtoseconds in lammps (10**-15). And we typically use 10fs in lammps
md.integrate.mode_standard(dt=0.01)

# hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=hps.temperature_to_kT(temperature), seed=4)
print(particles, particle_types)
# hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=0.593, seed=4)
hoomd.md.integrate.nve(group=hoomd.group.all())
hoomd.run(10e4)