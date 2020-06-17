import numpy as np
import hoomd
import hoomd.md
import hoomd.deprecated

hoomd.context.initialize("--gpu=1")

n = 450
Lx = 900
snapshot = hoomd.data.make_snapshot(N=n,
                                    box=hoomd.data.boxdim(Lx=Lx, Ly=10, Lz=5),
                                    particle_types=['A', 'B'],
                                    bond_types=['polymer']);

pos = np.zeros((n, 3))
pos[:, 0] = np.linspace(-Lx / 2 * 0.99, Lx / 2 * 0.99, n)

# pos = np.zeros((n,3))
# pos[:, 0] = np.arange(0, 450, 1)-Lx/2

snapshot.particles.position[:] = pos

for i in range(n):
    if i % 5 == 0:
        snapshot.particles.typeid[i] = 0
    else:
        snapshot.particles.typeid[i] = 1
# create bonds:
bonds = []
for i in range(n - 1):
    bonds.append([i, i + 1])

snapshot.bonds.resize(n - 1)
snapshot.bonds.group[:] = bonds

snapshot.replicate(1, 5, 10)
system = hoomd.init.read_snapshot(snapshot)

nl = hoomd.md.nlist.cell(r_buff=0.575)

lj = hoomd.md.pair.lj(r_cut=2.5 * 2.0, nlist=nl)

lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=2.0, r_cut=2.5 * 1)
lj.pair_coeff.set('A', 'B', epsilon=np.sqrt(3), sigma=1.5, r_cut=2.5 * 1.5)
lj.pair_coeff.set('B', 'B', epsilon=0.5, sigma=1.0, r_cut=2.5 * 1)

harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('polymer', k=300.0, r0=2.0)
all = hoomd.group.all()
# Initial equilibration. Compress the box
integrator = hoomd.md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=all, kT=0.3, seed=42)
equil_steps = 1e5
updater = hoomd.update.box_resize(Lx=hoomd.variant.linear_interp([(0, Lx), (equil_steps, 200)]))
traj = hoomd.dump.gsd("resizing-b.gsd", period=equil_steps // 100, group=all, overwrite=True)
hoomd.run(equil_steps)
traj.disable()

# Sampling
# integrator.set_params(dt=0.005)
updater.disable()  # stop compressing the box
sampling_steps = 1e8
hoomd.analyze.log(filename="log-output-b.log",
                  quantities=['potential_energy', 'temperature'],
                  period=sampling_steps // 1000,
                  overwrite=True)
hoomd.dump.gsd("trajectory-b.gsd", period=sampling_steps // 1000, group=all, overwrite=True)
hoomd.run(sampling_steps, profile=False)
