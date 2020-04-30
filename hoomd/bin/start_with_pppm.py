import hoomdsetup
import hoomd
from hoomd import md


l = 400
temperature = 300

hps = hoomdsetup.HPS(protein='CPEB4')
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
nl = md.nlist.cell()

pppm = md.charge.pppm(group=hoomd.group.charged(), nlist=nl)
pppm.set_params(Nx=64, Ny=64, Nz=64, order=6, rcut=2.0, alpha=0.1)

harmonic.bond_coeff.set('harmonic', k=9.6, r0=3.8)
hps_table = hps.get_HPS_pair_table(nl)
print("CHARGED!", hoomd.group.charged())

hoomd.analyze.log(filename="log.log",
                  quantities=['potential_energy', 'temperature'],
                  period=5000,
                  overwrite=True)

hoomd.dump.gsd(filename="trajectory.gsd", period=5000, group=hoomd.group.all(), overwrite=True)
hoomd.dump.dcd(filename="trajectory.dcd", period=5000, group=hoomd.group.all(), overwrite=True)
md.integrate.mode_standard(dt=0.005)

hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=hps.temperature_to_kT(temperature), seed=4)

# Run for 10,00 time steps
hoomd.run(10e5)
