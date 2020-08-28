h_c = hoomd.variant.linear_interp([(0, l), ($contract_t, $final_slab_x)])
collapse_r = hoomd.update.box_resize(Lx=h_c, Ly=h_c, Lz=h_c, scale_particles=False)

hoomd.run($contract_t)
collapse_r.disable()

h_e = hoomd.variant.linear_interp([(0, $final_slab_x), ($slab_t, $final_slab_z)])
expansion_z = hoomd.update.box_resize(Lz=h_e, scale_particles=False)
hoomd.run($slab_t)
expansion_z.disable()

hoomd.run($t)
