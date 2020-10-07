import mdtraj
import hoomd
from hoomd import md
import math
import pandas as pd
import scipy.constants as cnt

temperature = $temperature
l = $box_size
water_perm = $water_perm

particles_df = pd.read_csv('residues.txt', sep=" ", header=0, index_col=0)
particle_types = list(particles_df.index)

lambdas = pd.read_csv(f'lambdas_{temperature:.0f}.txt', sep=" ", header=0, index_col=0)
epsilons = pd.read_csv('epsilons.txt', sep=" ", header=0, index_col=0)
sigmas = pd.read_csv('sigmas.txt', sep=" ", header=0, index_col=0)

protein = "$protein"
chains = $chains
sequence = "$sequence"
topo_path = "$topo_path"
q_hmd_factor = math.sqrt(4*math.pi*cnt.epsilon_0*10**-9*10**3/cnt.Avogadro)
boltzmann_hmd = cnt.Boltzmann/1000.*cnt.Avogadro

# HPS PARAMETERS FROM https://pubs.acs.org/doi/10.1021/acscentsci.9b00102 (Temperature-Controlled Liquid–Liquid Phase...)
spring_k = 9.6     # kcal/(mol * Ang)
spring_r0 = 3.8   # Ang

# Convert to HOOMD units: energy = kJ/mol, distance = nm,
spring_k = spring_k * 4.184 * 100. * 2.
spring_r0 = spring_r0 / 10

$explicit_potential_code

hoomd.context.initialize("$context")

box = hoomd.data.boxdim(Lx=l, Ly=l, Lz=l)
snap = hoomd.data.make_snapshot(N=chains*len(sequence),
                                box=box,
                                bond_types=['harmonic'],
                                particle_types=particle_types)

poss = mdtraj.load_pdb(topo_path)
# poss.center_coordinates()
# MDtraj always divides by 10 when reading a pdb
snap.particles.position[:] = poss.xyz - l / 2

bond_arr = []
for chain in range(chains):
    for i, aa in enumerate(sequence):
        j = i + chain*len(sequence)
        snap.particles.typeid[j] = particles_df["id"][aa]-1
        snap.particles.mass[j] = particles_df["mass"][aa]
        snap.particles.diameter[j] = sigmas[aa][aa]/10.
        snap.particles.charge[j] = particles_df["q"][aa]*cnt.e/q_hmd_factor
        bond_arr.append([j, j + 1])
    del bond_arr[-1]
snap.bonds.resize((len(sequence) - 1)*chains)
snap.bonds.group[:] = bond_arr

sys = hoomd.init.read_snapshot(snap)
harmonic = md.bond.harmonic()
nl = md.nlist.cell(r_buff=0.35)

harmonic.bond_coeff.set('harmonic', k=spring_k, r0=spring_r0)

hps_table = md.pair.table(width=10000, nlist=nl)
for i in range(len(particle_types)):
    aa_i = particle_types[i]
    for j in range(i, len(particle_types)):
        aa_j = particle_types[j]
        lambd = $hps_scale * lambdas[aa_i][aa_j]
        sigma = sigmas[aa_i][aa_j]
        eps = epsilons[aa_i][aa_j]
        hps_table.pair_coeff.set(aa_i, aa_j, func=HPS_potential,
                                 rmin=0.2,
                                 rmax=3*sigma/10,
                                 coeff=dict(eps=eps*4.184, lambd=lambd, sigma=sigma/10))

yukawa = hoomd.md.pair.yukawa(r_cut=3.5, nlist=nl)
for i in range(len(particle_types)):
    aa_i = particle_types[i]
    for j in range(i, len(particle_types)):
        aa_j = particle_types[j]
        cutoff=3.5
        qi = particles_df["q"][aa_i] * cnt.e
        qj = particles_df["q"][aa_j] * cnt.e
        if particles_df["q"][aa_i] == 0 or  particles_df["q"][aa_j] == 0:
            cutoff=0
            qiqj = 0.
        else:
            cutoff = 3.5
            qiqj = qi*qj/(4*math.pi*cnt.epsilon_0*water_perm)
            qiqj *= 10**-3*cnt.Avogadro*10**9
        yukawa.pair_coeff.set(aa_i, aa_j, kappa=$debye, epsilon=qiqj, r_cut=cutoff, r_on=cutoff)


hoomd.analyze.log(filename=f"log_{temperature:.0f}.log", quantities=['potential_energy', 'bond_harmonic_energy', 'pair_yukawa_energy', 'pair_table_energy', 'temperature', 'lx', 'ly', 'lz'], period=$save, overwrite=True)
hoomd.dump.gsd(filename=f"trajectory_{temperature:.0f}.gsd", period=$save, group=hoomd.group.all(), overwrite=True)
hoomd.dump.dcd(filename=f"trajectory_{temperature:.0f}.dcd", period=$save, group=hoomd.group.all(), overwrite=True, unwrap_full=True)

# dt in picosecond units (10**-12) while dt in femtoseconds in lammps (10**-15). And we typically use 10fs in lammps
# which is 0.01
md.integrate.mode_standard(dt=0.01)

lang = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=boltzmann_hmd*temperature, seed=4)
for i in range(len(particle_types)):
    aa_i = particle_types[i]
    mass = particles_df["mass"][aa_i]
    lang.set_gamma(aa_i, mass/100)
    lang.set_gamma_r(aa_i, mass/100)
