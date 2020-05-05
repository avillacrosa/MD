import hoomdsetup
import hoomd
from hoomd import md
import mdtraj
import numpy as np

temperature = 300.
l = 200
chains = 50

hps = hoomdsetup.HPS(protein='CPEB4', chains=chains)

hoomd.context.initialize("")
box = hoomd.data.boxdim(Lx=l, Ly=l, Lz=l)
snap = hoomd.data.make_snapshot(N=chains*len(hps.sequence),
                                box=box,
                                bond_types=['harmonic'],
                                particle_types=hps.particle_types)

poss = mdtraj.load_pdb("/home/adria/xoriguer/adria/prod/topox50.pdb")
poss.center_coordinates()
snap.particles.position[:] = poss.xyz

hps.build_bonds(snap)

hoomd.init.read_snapshot(snap)

harmonic = md.bond.harmonic()

nl = md.nlist.cell()

# harmonic.bond_coeff.set('harmonic', k=10., r0=3.8)
harmonic.bond_coeff.set('harmonic', k=10.*4.184, r0=0.38)
hps_table = hps.get_HPS_pair_table(nl)
# hps_table = hps.get_LJ(nl)
# dipole_table = hps.get_dipole_pair_table(nl)

hoomd.analyze.log(filename="log.log",
                  quantities=['potential_energy', 'temperature'],
                  period=5000,
                  overwrite=True)
hoomd.dump.gsd(filename=f"trajectory_{temperature}.gsd", period=5000, group=hoomd.group.all(), overwrite=True)
hoomd.dump.dcd(filename="trajectory.dcd", period=5000, group=hoomd.group.all(), overwrite=True)

# dt in picosecond units (10**-12) while dt in femtoseconds in lammps (10**-15). And we typically use 10fs in lammps
md.integrate.mode_standard(dt=0.01)

# hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=hps.temperature_to_kT(temperature), seed=4)

# hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=0.593, seed=4)
hoomd.md.integrate.nve(group=hoomd.group.all())
hoomd.run(10e4)