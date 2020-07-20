from analyse import *
import hoomd
import hoomd.md
import time
import os
import sys
​
def simulate(residues,name,prot):
    residues = residues.set_index('one') # DataFrame with sigma, lambda, MWs
    hoomd.context.initialize("--mode=cpu");
    hoomd.option.set_notice_level(1)
    types = list(np.unique(prot.fasta)) # fasta sequence
    snapshot = hoomd.data.make_snapshot(N=len(prot.fasta),
                                    box=hoomd.data.boxdim(Lx=100, Ly=100, Lz=100),
                                    particle_types=types,
                                    bond_types=['polymer']);
    snapshot.particles.position[:] = [[0,0,(i-len(prot.fasta)/2.)*.38] for i,_ in enumerate(prot.fasta)];
    ids = [types.index(a) for a in prot.fasta] # convert types to int
    for i,idi in enumerate(ids):
        snapshot.particles.typeid[i] = idi
        snapshot.particles.mass[i] = residues.loc[types[idi]].MW
    snapshot.bonds.resize(len(prot.fasta)-1);
    snapshot.bonds.group[:] = [[i,i+1] for i in range(len(prot.fasta)-1)];
    hoomd.init.read_snapshot(snapshot);
    nl = hoomd.md.nlist.cell();
    kT = 8.3145*prot.temp*1e-3
    lj = hoomd.md.pair.lj(r_cut=4.0, nlist=nl)
    yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
    pairs, lj_eps, lj_sigma, _, _ = genParamsLJ(residues,name,prot) #  generate lj eps and sigma
    yukawa_eps, yukawa_kappa = genParamsDH(residues,name,prot) # generate yukawa eps and kappa
    for a,b in pairs:
        lj.pair_coeff.set(a, b, epsilon=lj_eps.loc[a,b], sigma=lj_sigma.loc[a,b])
        yukawa.pair_coeff.set(a, b, epsilon=yukawa_eps.loc[a,b], kappa=yukawa_kappa)
    harmonic = hoomd.md.bond.harmonic();
    harmonic.bond_coeff.set('polymer', k=8033.0, r0=0.38);
    integrator_mode = hoomd.md.integrate.mode_standard(dt=0.01);
    integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=kT,seed=np.random.randint(100));
    for a in types:
        integrator.set_gamma(a, residues.loc[a].MW/100)
        integrator.set_gamma_r(a, residues.loc[a].MW/100)
    #nl.tune()
    hoomd.dump.dcd(prot.path+"/hoomd_{:s}.dcd".format(name), period=5e2, group=hoomd.group.all(), overwrite=True, unwrap_full=True);
    hoomd.analyze.log(filename=prot.path+"/{:s}.log".format(name),
              quantities=['pair_lj_energy', 'temperature'],
              period=5e2,
              overwrite=True);
    hoomd.run(1e7)
