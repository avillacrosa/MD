lang.disable()

npt_f = hoomd.md.integrate.npt(group=hoomd.group.all(), kT=boltzmann_hmd*150, tau=1, P=1, tauP=1)
hoomd.run(400000)

npt_f.disable()
lang.enable()
npt_snap = sys.take_snapshot(all=True)

h_c = hoomd.variant.linear_interp([(0, npt_snap.box.Lx), ($contract_t, $final_slab_x)])
collapse_r = hoomd.update.box_resize(Lx=h_c, Ly=h_c, Lz=h_c, scale_particles=True)

hoomd.run($contract_t)
collapse_r.disable()

h_e = hoomd.variant.linear_interp([(0, $final_slab_x), ($slab_t, $final_slab_z)])
expansion_z = hoomd.update.box_resize(Lz=h_e, scale_particles=False)

hoomd.run($slab_t)
expansion_z.disable()

hoomd.run($t)
