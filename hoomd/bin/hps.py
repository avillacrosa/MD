import numpy as np
import hoomd
from hoomd import md

def HPS(r, rmin, rmax, eps, lambd, sigma):
    V = 4*eps*((sigma/r)**12 - (sigma/r)**6)
    F = 4*eps/r*(12*(sigma/r)**12 - 6*(sigma/r)**6)
    if r <= 2**(1/6)*sigma:
        V = V + (1-lambd)*eps
    else:
        V = lambd*V
        F = lambd*F
    return (V, F)

particles = {'A': {'id': 1, 'name': 'ALA', 'mass': 71.0800018, 'q': 0.0, 'lambda': 0.72973, 'sigma': 5.04}, 'R': {'id': 2, 'name': 'ARG', 'mass': 156.10111, 'q': 1.0, 'lambda': 0.0, 'sigma': 6.56}, 'N': {'id': 3, 'name': 'ASN', 'mass': 114.099998, 'q': 0.0, 'lambda': 0.432432, 'sigma': 5.68}, 'D': {'id': 4, 'name': 'ASP', 'mass': 115.099998, 'q': -1.0, 'lambda': 0.378378, 'sigma': 5.58}, 'C': {'id': 5, 'name': 'CYS', 'mass': 103.00919, 'q': 0.0, 'lambda': 0.594595, 'sigma': 5.48}, 'E': {'id': 6, 'name': 'GLU', 'mass': 129.100006, 'q': -1.0, 'lambda': 0.459459, 'sigma': 5.92}, 'Q': {'id': 7, 'name': 'GLN', 'mass': 128.100006, 'q': 0.0, 'lambda': 0.513514, 'sigma': 6.02}, 'G': {'id': 8, 'name': 'GLY', 'mass': 57.0499992, 'q': 0.0, 'lambda': 0.648649, 'sigma': 4.5}, 'H': {'id': 9, 'name': 'HIS', 'mass': 137.100006, 'q': 0.5, 'lambda': 0.513514, 'sigma': 6.08}, 'I': {'id': 10, 'name': 'ILE', 'mass': 113.199997, 'q': 0.0, 'lambda': 0.972973, 'sigma': 6.18}, 'L': {'id': 11, 'name': 'LEU', 'mass': 113.199997, 'q': 0.0, 'lambda': 0.972973, 'sigma': 6.18}, 'K': {'id': 12, 'name': 'LYS', 'mass': 128.199997, 'q': 1.0, 'lambda': 0.513514, 'sigma': 6.36}, 'M': {'id': 13, 'name': 'MET', 'mass': 131.199997, 'q': 0.0, 'lambda': 0.837838, 'sigma': 6.18}, 'F': {'id': 14, 'name': 'PHE', 'mass': 147.199997, 'q': 0.0, 'lambda': 1.0, 'sigma': 6.36}, 'P': {'id': 15, 'name': 'PRO', 'mass': 97.1200027, 'q': 0.0, 'lambda': 1.0, 'sigma': 5.56}, 'S': {'id': 16, 'name': 'SER', 'mass': 87.0800018, 'q': 0.0, 'lambda': 0.594595, 'sigma': 5.18}, 'T': {'id': 17, 'name': 'THR', 'mass': 101.099998, 'q': 0.0, 'lambda': 0.675676, 'sigma': 5.62}, 'W': {'id': 18, 'name': 'TRP', 'mass': 186.07931, 'q': 0.0, 'lambda': 0.945946, 'sigma': 6.78}, 'Y': {'id': 19, 'name': 'TYR', 'mass': 163.199997, 'q': 0.0, 'lambda': 0.864865, 'sigma': 6.46}, 'V': {'id': 20, 'name': 'VAL', 'mass': 99.0699997, 'q': 0.0, 'lambda': 0.891892, 'sigma': 5.86}}
particle_types = list(particles.keys())
l = 200
seq     = 'MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE'

hoomd.context.initialize("")

box = hoomd.data.boxdim(Lx=l, Ly=l, Lz=l)
snap = hoomd.data.make_snapshot(N=len(seq),
                                box=box,
                                bond_types=['harmonic'],
                                particle_types=particle_types)
bond_arr = []
for i, aa in enumerate(seq):
    snap.particles.typeid[i]=particles[aa]["id"]
    bond_arr.append([i,i+1])
del bond_arr[-1]
snap.bonds.resize(len(seq)-1);
snap.bonds.group[:] = bond_arr[:]
    
hoomd.init.read_snapshot(snap);

harmonic = md.bond.harmonic()
harmonic.bond_coeff.set('harmonic', k=100.0, r0=3.8)

# Specify Lennard-Jones interactions between particle pairs
nl = md.nlist.cell()
# debye = md.charge.pppm(group='all', nlist=nl)
hps_table = md.pair.table(width=len(seq), nlist=nl);
for i in range(len(particle_types)):
    aa_i = particle_types[i]
    for j in range(i, len(particle_types)):
        aa_j = particle_types[j]
        lambd = (particles[aa_i]["lambda"] + particles[aa_j]["lambda"])/2
        sigma = (particles[aa_i]["sigma"] + particles[aa_j]["sigma"])/2
        hps_table.pair_coeff.set(aa_i, aa_j, func=HPS, 
                                 rmin=3.8, 
                                 rmax=35,
                                 coeff=dict(eps=0.1, lambd=lambd, sigma=sigma))


# lj = md.pair.lj(r_cut=3.0, nlist=nl)
# lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

# Integrate at constant temperature
md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.2, seed=4)

# Run for 10,00 time steps
hoomd.run(10e2)
