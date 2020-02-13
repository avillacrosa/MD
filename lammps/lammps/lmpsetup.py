import definitions
import lmp
import scipy.constants as cnt
import numpy as np
import multiprocessing as mp
from subprocess import run, PIPE
import shutil
import math
import random
import os
import glob
import mdtraj as md
from string import Template


class LMPSetup(lmp.LMP):
    # TODO : GENERAL, INCLUDE BACKTESTING ?
    def __init__(self, protein, chains=1, **kw):
        super(LMPSetup, self).__init__(**kw)

        with open(os.path.join('../data/sequences', f'{protein}.seq')) as f:
            self.sequence = f.readlines()[0]

        self.temperature = 300
        self.ionic_strength = 100e-3
        self.dt = 10.
        self.t = 100000000
        self.xyz = None
        self.n_chains = chains
        self.protein = protein

        self.seq_charge = None
        self.residue_dict = dict(definitions.residues)
        self.key_ordering = list(self.residue_dict.keys())

        self.lmp = '/home/adria/local/lammps/bin/lmp'
        self.box_size = 2500
        self.water_perm = 80.
        self.hps_scale = 1.0

        self.topo_file_dict = {}
        self.lmp_file_dict = {}
        self.qsub_file_dict = {}

        self.rerun_dump = None
        # TODO : GRACEFULLY INCORPORATE THIS ?
        self.base_dir = f'{self.hps_scale:.1f}ls-{self.ionic_strength*1e3:.0f}I-{self.water_perm:.0f}e'
        # TODO: MEH
        # self.o_wd = os.path.join(self.o_wd, self.base_dir)

        self.p_wd = self.o_wd.replace('/perdiux', '')

        self.v_seed = 494211
        self.langevin_seed = 451618
        self.save = 50000
        self.langevin_damp = 10000

        self.debye_wv = 1/self.debye_length()

        #TODO : Fancy job name ?
        self.job_name = f'hps_{os.path.basename(self.o_wd)}'
        self.processors = 12
        # self.temperatures = np.linspace(150.0, 600.0, self.processors)
        # TODO HARD CODED... ALSO, AS AN ARRAY ?
        self.temperatures = [300, 320, 340, 360, 380, 400]
        self.swap_every = 1000

    def del_missing_aas(self):
        missing, pos = [], []
        for key in self.residue_dict:
            if key not in self.sequence:
                pos.append(self.residue_dict[key]["id"])
                missing.append(key)

        for m in missing:
            del self.residue_dict[m]

        pos.reverse()

        for p in pos:
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] > p:
                    self.residue_dict[key]["id"] += -1
        self.key_ordering = list(self.residue_dict.keys())

    def lammps_ordering(self):
        id = 1
        done_aas = []
        for aa in self.sequence:
            if aa not in done_aas:
                self.residue_dict[aa]["id"] = id
                done_aas.append(aa)
                id += 1
        ordered_keys = sorted(self.residue_dict, key=lambda x: (self.residue_dict[x]['id']))
        self.key_ordering = ordered_keys

    def get_equilibration_xyz(self, save=False, t=100000):
        lmp2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'
        meta_maker = LMPSetup(oliba_wd='../default_output', seq=self.sequence)
        meta_maker.t = t
        # meta_maker.del_missing_aas()
        meta_maker.get_hps_pairs()
        meta_maker.write_hps_files(output_dir=None, equil=True)
        meta_maker.run('lmp.lmp', n_cores=8)

        os.chdir('/home/adria/scripts/depured-lammps/default_output')
        file = '../default_output/data'
        os.system(lmp2pdb + ' ' + file)
        fileout = file + '_trj.pdb'

        #TODO ONLY DO RUN IF XTC NOT PRESENT...
        traj = md.load('xtc_traj.xtc', top=fileout)
        self.xyz = traj[-1].xyz*10
        mx = np.abs(self.xyz).max()
        self.box_size = int(mx*3)

        print(traj[-1].unitcell_lengths)
        if save:
            print(f"-> Saving equilibration pdb at {os.path.join(self.oliba_wd, 'equilibration.pdb')}")
            traj[-1].save_pdb(os.path.join(self.oliba_wd, 'equilibration.pdb'))
        else:
            print(f"-> Saving equilibration pdb at {os.path.join('../default_output', 'equilibration.pdb')}")
            traj[-1].save_pdb(os.path.join('../default_output', 'equilibration.pdb'))

    def get_pdb_xyz(self, pdb, padding=0):
        struct = md.load_pdb(pdb)
        struct.center_coordinates()
        rg = md.compute_rg(struct)
        # TODO : Arbitrary numbers currently... make dis a function of padding strength
        # d = rg[0] * self.n_chains ** (1 / 3) * 20
        d = rg[0] * self.n_chains ** (1 / 3) * 8
        struct.unitcell_lengths = np.array([[d, d, d]])

        if self.n_chains == 1:
            self.xyz = struct.xyz*10
            self.box_size = d*10
            struct.save_pdb('../default_output/centered.pdb')
        # TODO : Include padding ?!
        else:
            # TEST
            n_cells = int(math.ceil(self.n_chains ** (1 / 3)))
            unitcell_d = d / n_cells

            def _build_box():
                c = 0
                # TODO : CORRECT PADDING !
                # padding = unitcell_d/4
                for z in range(n_cells):
                    for y in range(n_cells):
                        for x in range(n_cells):
                            if c == self.n_chains:
                                return adder
                            c += 1
                            dist = np.array([unitcell_d * (x + 1 / 2), unitcell_d * (y + 1 / 2), unitcell_d * (z + 1 / 2)])
                            dist -= 0.55*(dist-d/2)

                            struct.xyz[0, :, :] = struct.xyz[0, :, :] + dist
                            if x + y + z == 0:
                                adder = struct[:]
                            else:
                                adder = adder.stack(struct)
                            struct.xyz[0, :, :] = struct.xyz[0, :, :] - dist
                return adder
            system = _build_box()

            self.box_size = d*10
            self.xyz = system.xyz*10
            system.save_pdb('../default_output/double_eq.pdb')

    def write_hps_files(self, output_dir='default', equil=False, rerun=False, data=True, qsub=True, lmp=True):
        self.base_dir = f'{self.hps_scale:.1f}ls-{self.ionic_strength*1e3:.0f}I-{self.water_perm:.0f}e'
        self.o_wd = os.path.join(self.o_wd, f'x{self.n_chains}')
        if not os.path.isdir(self.o_wd):
            os.mkdir(self.o_wd)
            self.o_wd = os.path.join(self.o_wd, self.base_dir)
            if not os.path.isdir(self.o_wd):
                os.mkdir(self.o_wd)
                print(f"Directory does not exist. Creating dir : {self.o_wd}")
            else:
                self.o_wd = os.path.join(self.o_wd, self.base_dir)
        else:
            self.o_wd = os.path.join(self.o_wd, self.base_dir)
            if not os.path.isdir(self.o_wd):
                os.mkdir(self.o_wd)
                print(f"Directory does not exist. Creating dir : {self.o_wd}")

        if output_dir == 'default':
            output_dir = self.o_wd

        self._generate_lmp_input()
        self._generate_qsub()
        self._generate_data_input()
        self._generate_README()
        if self.xyz is not None:
            self._generate_pdb()

        topo_temp_file = open('../templates/topo_template.data')
        topo_template = Template(topo_temp_file.read())
        topo_subst = topo_template.safe_substitute(self.topo_file_dict)

        if self.temper:
            lmp_temp_file = open('../templates/replica/input_template.lmp')
        elif equil:
            lmp_temp_file = open('../templates/equilibration/input_template.lmp')
        elif rerun:
            lmp_temp_file = open('../templates/rerun/input_template.lmp')
        else:
            lmp_temp_file = open('../templates/general/input_template.lmp')
        lmp_template = Template(lmp_temp_file.read())
        lmp_subst = lmp_template.safe_substitute(self.lmp_file_dict)

        qsub_temp_file = open('../templates/qsub_template.tmp')
        qsub_template = Template(qsub_temp_file.read())
        qsub_subst = qsub_template.safe_substitute(self.qsub_file_dict)

        with open(f'../default_output/data.data', 'tw') as fileout:
            fileout.write(topo_subst)
        with open(f'../default_output/{self.job_name}.qsub', 'tw') as fileout:
            fileout.write(qsub_subst)
        with open(f'../default_output/lmp.lmp', 'tw') as fileout:
            fileout.write(lmp_subst)

        if output_dir is not None:
            if data:
                shutil.copyfile(f'../default_output/data.data', os.path.join(output_dir, 'data.data'))
            if qsub:
                shutil.copyfile(f'../default_output/{self.job_name}.qsub', os.path.join(output_dir, f'{self.job_name}.qsub'))
            if lmp:
                shutil.copyfile(f'../default_output/lmp.lmp', os.path.join(output_dir, 'lmp.lmp'))

    def assert_build(self):
        asserter = lmp.LMP(oliba_wd=self.o_wd, temper=True)
        assert int(asserter.get_eps()) == int(self.water_perm)
        assert asserter.get_temperatures() == list(map(str,self.temperatures))
        assert asserter.get_ionic_strength() == self.ionic_strength
        assert asserter.get_seq_from_hps().replace('L', 'I') == self.sequence.replace('L', 'I')
        print("Assertion complete")

    def debye_length(self):
        l = np.sqrt(cnt.epsilon_0 * self.water_perm * cnt.Boltzmann * self.temperature) / (np.sqrt(2 * self.ionic_strength * 10 ** 3 * cnt.Avogadro) * cnt.e)
        return l

    def _generate_pdb(self, display=None):
        abc = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
               'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        header = f'CRYST1     {self.box_size:.0f}     {self.box_size:.0f}     {self.box_size:.0f}     90     90     90   \n'
        xyz = ''
        c = 0
        for n in range(self.n_chains):
            for i, aa in enumerate(self.sequence):
                coords = self.xyz[0, c, :]
                if display:
                    if self.residue_dict[aa]["q"] > 0:
                        # TODO: BETTER WAY FOR THIS ?
                        res_name = 'C' if display == 'charged' else 'P'
                    elif self.residue_dict[aa]["q"] < 0:
                        res_name = 'C' if display == 'charged' else 'M'
                    else:
                        res_name = 'N'
                else:
                    res_name = aa
                # xyz += f'ATOM  {c+1:>5} {aa+str(i+1):>4}   {res_name} {abc[n]} {i+1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      PROT \n'
                xyz += f'ATOM  {c+1:>5} {res_name:>4}   {res_name} {abc[n]} {i+1:>3}    {coords[0]:>8.2f}{coords[1]:>8.2f}{coords[2]:>8.2f}  1.00  0.00      PROT \n'
                c += 1
            xyz += 'TER \n'
        bottom = 'END \n'
        with open('/home/adria/scripts/lammps/default_output/test.pdb', 'w+') as f:
            f.write(header + xyz + bottom)
        shutil.copyfile('/home/adria/scripts/lammps/default_output/test.pdb', os.path.join(self.o_wd, f'topo.pdb'))

    def _generate_lmp_input(self):
        if self.hps_pairs is None:
            self.get_hps_pairs()
        self.lmp_file_dict["t"] = self.t
        self.lmp_file_dict["dt"] = self.dt
        self.lmp_file_dict["pair_coeff"] = ''.join(self.hps_pairs)
        self.lmp_file_dict["debye"] = round(1/self.debye_length()*10**-10, 3)
        self.lmp_file_dict["v_seed"] = self.v_seed
        self.lmp_file_dict["langevin_seed"] = self.langevin_seed
        self.lmp_file_dict["temp"] = self.temperature
        self.lmp_file_dict["temperatures"] = ' '.join(map(str, self.temperatures))
        self.lmp_file_dict["water_perm"] = self.water_perm
        self.lmp_file_dict["swap_every"] = self.swap_every
        self.lmp_file_dict["save"] = self.save
        if int(self.t/10000) != 0:
            self.lmp_file_dict["restart"] = int(self.t/10000)
        else:
            self.lmp_file_dict["restart"] = 500
        # TODO this sucks but it is what it is, better option upstairs..
        self.lmp_file_dict["replicas"] = np.array2string(np.linspace(0, self.processors-1, self.processors, dtype='int'))[1:-1]
        self.lmp_file_dict["rerun_dump"] = self.rerun_dump
        self.lmp_file_dict["langevin_damp"] = self.langevin_damp

    def _generate_data_input(self):
        masses = []
        for i in range(1, len(self.residue_dict)+1):
            for key in self.residue_dict:
                if self.residue_dict[key]["id"] == i:
                    masses.append(f'           {i:2d}  {self.residue_dict[key]["mass"]} \n')

        atoms, bonds = [], []
        k = 1
        spaghetti = False

        for chain in range(1, self.n_chains + 1):
            #TODO CENTER SPAGHETTI BETTER...
            if self.xyz is None:
                xyz = [240., 240 + chain * 20, 240]
                spaghetti = True
            for aa in self.sequence:
                if spaghetti:
                    xyz[0] += definitions.bond_length
                else:
                    xyz = self.xyz[0, k-1, :]
                atoms.append(f'      {k :3d}          {chain}    '
                             f'      {self.residue_dict[aa]["id"]:2d}   '
                             f'    {self.residue_dict[aa]["q"]: .2f}'
                             f'    {xyz[0]: .3f}'
                             f'    {xyz[1]: .3f}'
                             f'    {xyz[2]: .3f} \n')
                if k != chain * (len(self.sequence)):
                    bonds.append(f'       {k:3d}       1       {k:3d}       {k + 1:3d}\n')
                k += 1

        self.topo_file_dict["natoms"] = self.n_chains * len(self.sequence)
        self.topo_file_dict["nbonds"] = self.n_chains * (len(self.sequence) - 1)
        self.topo_file_dict["atom_types"] = len(self.residue_dict)
        self.topo_file_dict["masses"] = ''.join(masses)
        self.topo_file_dict["atoms"] = ''.join(atoms)
        self.topo_file_dict["bonds"] = ''.join(bonds)
        self.topo_file_dict["box_size"] = int(self.box_size)

    def _generate_qsub(self):
        self.qsub_file_dict["work_dir"] = self.p_wd
        if self.temper:
            self.qsub_file_dict["command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -partition {self.processors}x1 -in lmp.lmp"
        else:
            self.qsub_file_dict["command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {self.processors} /home/adria/local/lammps/bin/lmp -in lmp.lmp"
        self.qsub_file_dict["np"] = self.processors
        self.qsub_file_dict["jobname"] = self.job_name

    # TODO : !
    def _generate_restart(self):
        print("TODO")

    # TODO : !
    def _generate_slurm(self):
        print("TODO")

    def _generate_README(self):
        with open(os.path.join(self.o_wd, 'README.txt'), 'tw') as readme:
            readme.write(f'HPS Scale : {self.hps_scale} \n')
            readme.write(f'Ionic Strength : {self.ionic_strength} \n')
            readme.write(f'Medium Permittivity : {self.water_perm} \n')
            readme.write(f'Protein : {self.protein} \n')
            readme.write(f'Temperatures : {self.temperatures} \n')
            readme.write(f'Temper : {self.temper} \n')
            readme.write(f'Number of chains : {self.n_chains} \n')
            readme.write(f'Langevin Damp : {self.langevin_damp} \n')
