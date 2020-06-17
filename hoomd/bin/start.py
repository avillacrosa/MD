import hmdsetup
import hoomd
from hoomd import md


l = 400
temperature = 300

hps = hmdsetup.HPS(protein='CPEB4')
hoomd.context.initialize("")
box = hoomd.data.boxdim(Lx=l, Ly=l, Lz=l)
snap = hoomd.data.make_snapshot(N=len(hps.sequence),
                                box=box,
                                bond_types=['harmonic'],
                                particle_types=hps.particle_types)

snap.particles.position[:] = hps.get_pos()

hps.build_bonds(snap)

hoomd.init.read_snapshot(snap)

harmonic = md.bond.harmonic()
harmonic.bond_coeff.set('harmonic', k=9.6, r0=3.8)

nl = md.nlist.cell()
hps_table = hps.get_HPS_pair_table(nl)

hoomd.analyze.log(filename="log.log",
                  quantities=['potential_energy', 'temperature'],
                  period=5000,
                  overwrite=True)
hoomd.dump.gsd(filename="trajectory.gsd", period=5000, group=hoomd.group.all(), overwrite=True)
hoomd.dump.dcd(filename="trajectory.dcd", period=5000, group=hoomd.group.all(), overwrite=True)
md.integrate.mode_standard(dt=0.005)

hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=hps.temperature_to_kT(temperature), seed=4)

# Run for 10,00 time steps
hoomd.run(10e2)
