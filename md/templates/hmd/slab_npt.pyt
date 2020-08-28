lang.disable()

npt_f = hoomd.md.integrate.npt(group=hoomd.group.all(), kT=boltzmann_hmd*150, tau=1, P=1, tauP=1)
hoomd.run(100000)

npt_f.disable()
lang.enable()

npt_snap = sys.take_snapshot(all=True)
h_e = hoomd.variant.linear_interp([(0, npt_snap.box.Lz), ($slab_t, $final_slab_z)])
expansion_z = hoomd.update.box_resize(Lz=h_e, scale_particles=False)

hoomd.run($slab_t)
expansion_z.disable()

hoomd.run($t)
