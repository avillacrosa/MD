h_c = hoomd.variant.linear_interp([(0, l), ($contract_t, $final_slab_x)])
collapse_r = hoomd.update.box_resize(Lx=h_c, Ly=h_c, Lz=h_c, scale_particles=True)
lang.disable()

lang_cont = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=boltzmann_hmd*600, seed=4)
for i in range(len(particle_types)):
    aa_i = particle_types[i]
    mass = particles[aa_i]["mass"]
    lang.set_gamma(aa_i, mass/100)
    lang.set_gamma_r(aa_i, mass/100)

hoomd.run($contract_t)
collapse_r.disable()
lang_cont.disable()
lang.enable()


h_e = hoomd.variant.linear_interp([(0, $final_slab_x), ($slab_t, $final_slab_z)])
expansion_z = hoomd.update.box_resize(Lz=h_e, scale_particles=False)
hoomd.run($slab_t)
expansion_z.disable()

hoomd.run($t)
