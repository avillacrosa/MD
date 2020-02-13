import numpy as np
import os
import definitions
import re
import random
import shutil
import mdtraj as md
import glob
import multiprocessing as mp
from subprocess import run, PIPE
import pathlib
import scipy.constants as cnt


class LMP:
    def __init__(self, oliba_wd, force_reorder=False):
        self.o_wd = oliba_wd

        self.lmp2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'
        self.lmp = '/home/adria/local/lammps/bin/lmp'

        # Even if named "files" they are actually the lines of the files...
        self.data_file = self._get_data_file()
        self.lmp_file = self._get_lmp_file()
        self.temper = self._is_temper()
        self.log_files = self._get_log_files()
        self.topology_path = self._get_initial_frame()

        self.residue_dict = dict(definitions.residues)

        self.force_reorder = force_reorder

        self.lmp_drs = self.get_lmp_dirs()
        self.chains, self.chain_atoms = self.get_n_chains()
        self.sequence = self.get_seq_from_hps()
        self.data = self.get_lmp_data()

    def get_lmp_dirs(self, path=None):
        if path is None:
            path = self.o_wd
        dirs = []
        for filename in pathlib.Path(path).rglob('*lmp'):
            dirs.append(os.path.dirname(filename))
        dirs.sort()
        return dirs

    def _get_initial_frame(self, dirs=None):
        if os.path.exists(os.path.join(self.o_wd, 'topo.pdb')):
            return os.path.join(self.o_wd, 'topo.pdb')
        else:
            files = glob.glob(os.path.join(self.o_wd, '*.data'))
            file = os.path.basename(files[0])
            file = file.replace('.data', '')
            lammps2pdb = self.lmp2pdb
            os.system(lammps2pdb + ' ' + os.path.join(self.o_wd, file))
            fileout = os.path.join(self.o_wd, file) + '_trj.pdb'
            return fileout

    # TODO : SHOULD I REORDER THIS???
    def get_lmp_data(self, progress=False):
        data, data_ends, prog = [], [], []
        dstart = 0
        for log_data in self.log_files:
            data_start = 0
            data_end = 0
            for i, line in enumerate(log_data):
                if progress:
                    if "temper" in line or "run" in line:
                        prog.append(int(re.findall(r'\d+', line)[0]))
                if "Step" in line:
                    data_start = i + 1
                if "Loop" in line:
                    data_end = i
                if data_end and data_start != 0:
                    break
            if data_end == 0:
                data_end = len(log_data)
            data_ends.append(data_end)
            dstart = data_start
        data_end_min = np.array(data_ends).min()
        for log_data in self.log_files:
            dat = np.genfromtxt(log_data[dstart:data_end_min])
            data.append(dat)
        data = np.array(data)
        ran_steps = int(data[:, :, 0].mean(axis=0).max())
        if progress:
            print(f"Run Completed at {ran_steps/np.array(prog).mean()*100:.2f} %")
        return data

    def get_lmp_E(self):
        E = self.data[:, :, [1, 2]]
        return E

    def get_eps(self):
        eps = 0
        for line in self.lmp_file:
            if "dielectric" in line:
                eps = re.findall(r'\d+', line)[0]
                break
        return eps

    def get_ionic_strength(self):
        ionic_strength = 0
        for line in self.lmp_file:
            if "ljlambda" in line:
                debye = re.findall(r'\d+\.?\d*', line)[0]
                unroundedI = self.get_I_from_debye(float(debye), eps=float(self.get_eps()))
                if unroundedI >= 0.1:
                    ionic_strength = round(unroundedI, 1)
                    break
                else:
                    ionic_strength = round(unroundedI, 3)
                    break
        return ionic_strength

    def get_temperatures(self):
        if not self.temper:
            print("TODO IF NOT TEMPER")
            return
        T = []
        for line in self.lmp_file:
            if "variable T world" in line:
                T = re.findall(r'\d+\.?\d*', line)
                break
        return np.array(T, dtype=float)

    def get_n_chains(self):
        unit_atoms, n_atoms, reading_atoms = 0, 0, False
        for line in self.data_file:
            if 'atoms' in line:
                n_atoms = int(re.findall(r'\d+', line)[0])
            if reading_atoms and line != '\n':
                nums = re.findall(r'\d+', line)
                if nums:
                    if int(nums[1]) == 1:
                        unit_atoms += 1
                    if int(nums[1]) != 1:
                        reading_atoms = False
                        break
            if 'Atoms' in line:
                reading_atoms = True
            if 'Bonds' in line:
                break
        n_chains = int(n_atoms/unit_atoms)
        return n_chains, unit_atoms

    # TODO : Maybe this at LMPSETUP???
    def charge_scramble(self, sequence, keep_positions=True, shuffles=1):
        charges = self.get_charge_seq(sequence=sequence)[0]
        old_charges = charges[:]
        seqs, deltas, scds = [sequence], [self.pappu_delta(sequence)], [self.chan_scd(sequence)]
        for i in range(shuffles):
            seq = list(sequence)
            if keep_positions:
                random.shuffle(charges)
            else:
                charges = np.random.randint(0, len(seq), len(charges))
            for k in range(len(charges)):
                seq[old_charges[k]], seq[charges[k]] = seq[charges[k]], seq[old_charges[k]]
            seqs.append(''.join(seq))
            deltas.append(self.pappu_delta(seq))
            scds.append(self.chan_scd(seq))
        deltas = np.array(deltas)
        seqs = np.array(seqs)
        scds = np.array(scds)

        deltas = deltas/deltas.max()

        idx = deltas.argsort()
        return seqs[idx][::-1], deltas[idx][::-1], scds[idx][::-1]

    # TODO : Maybe this at LMPSETUP???
    def get_charge_seq(self, sequence, window=9):
        win = np.zeros(len(sequence))
        rr = int((window-1)/2)
        if rr >= 0:
            for i, aa in enumerate(sequence):
                jaa = sequence[i]
                win[i] += self.residue_dict[jaa]["q"]
        else:
            for i, aa in enumerate(sequence):
                for j in range(-rr, rr+1):
                    if len(sequence) > i+j > 0:
                        jaa = sequence[i+j]
                        win[i] += self.residue_dict[jaa]["q"]
                        #CORRECTOR
                        # if 1 > abs(self.residue_dict[jaa]["q"]) > 0:
                        #     win[i] += 1 - self.residue_dict[jaa]["q"]
                win[i] /= window
        plus = np.copy(win)
        minus = np.copy(win)
        plus[plus < 0.] = 0
        minus[minus > 0.] = 0
        return win, plus, minus

    # TODO : PROBABLY BROKEN
    def pappu_delta(self, sequence, g=5):
        # TODO: ONLY FOR PROLINE-LESS SEQUENCES! What the hell ? Check ref
        N = len(sequence) - (g - 1)
        itot, iplus, iminus = self.get_charge_seq(sequence)
        f_plus = np.count_nonzero(iplus > 0)/len(itot)
        f_minus = np.count_nonzero(iminus < 0)/len(itot)
        sigma = (f_plus - f_minus) ** 2 / (f_plus + f_minus)
        delta = 0
        for i in range(N):
            blob = sequence[i:i + g]
            total, plus, minus = self.get_charge_seq(blob, window=1)
            if len(plus) != 0 or len(minus) != 0:
                f_plus_i = np.count_nonzero(iplus > 0) / g
                f_minus_i = np.count_nonzero(iminus < 0) / g
                # print(np.count_nonzero(iplus > 0))
                # print(f_minus_i)
                sigma_i = (f_plus_i - f_minus_i) ** 2 / (f_plus_i + f_minus_i)
                delta += (sigma_i - sigma) ** 2 / N
        return delta

    # TODO : PROBABLY BROKEN
    def chan_scd(self, sequence):
        total, plus, minus = self.get_charge_seq(sequence)
        scd = 0
        # TODO DUMB....
        for i in range(len(total)):
            for j in range(i, len(total)):
                sigma_i = 0
                sigma_j = 0
                if total[i] in plus:
                    sigma_i = +1
                if total[i] in minus:
                    sigma_i = -1
                if total[j] in plus:
                    sigma_j = +1
                if total[j] in minus:
                    sigma_j = -1
                scd += sigma_i*sigma_j*np.sqrt(np.abs(total[i]-total[j]))
        return scd/(len(sequence))

    # TODO : MULTICHAIN CASE ?
    def get_seq_from_hps(self):
        mass_range, atom_range = [0, 0], [0, 0]
        reading_masses, reading_atoms = False, False
        # TODO Not too smart
        for i, line in enumerate(self.data_file):
            if "Atoms" in line:
                reading_masses = False
                mass_range[1] = i
            if reading_masses and line != '\n' and mass_range[0] == 0:
                mass_range[0] = i
            if "Masses" in line:
                reading_masses = True
            if "Bonds" in line:
                reading_atoms = False
                atom_range[1] = i
            if reading_atoms and line != '\n' and atom_range[0] == 0:
                atom_range[0] = i
            if "Atoms" in line:
                reading_atoms = True
        masses = np.genfromtxt(self.data_file[mass_range[0]:mass_range[1]])
        atoms = np.genfromtxt(self.data_file[atom_range[0]:atom_range[1]])
        mass_dict = {}
        for mass in masses:
            for res in self.residue_dict:
                if self.residue_dict[res]["mass"] == mass[1]:
                    mass_dict[int(mass[0])] = res
                    break
        seq = []
        for atom in atoms:
            seq.append(mass_dict[atom[2]])
        seq_str = ''.join(seq)
        return seq_str

    def run(self, file, n_cores=1):
        f_name = os.path.join(self.o_wd, file)
        if n_cores > mp.cpu_count():
            raise SystemExit(f'Desired number of cores exceed available cores on this machine ({mp.cpu_count()})')
        if n_cores > 1:
            command = f'mpirun -n {n_cores} {self.lmp} -in {f_name}'
        elif n_cores == 1:
            command = f'{self.lmp} -in {f_name}'
        else:
            raise SystemExit('Invalid core number')
        old_wd = os.getcwd()
        os.chdir(self.o_wd)
        out = run(command.split(), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        os.chdir(old_wd)
        return out

    def get_I_from_debye(self, debye_wv, eps=80, temperature=300):
        sqrtI = np.sqrt(cnt.epsilon_0 * eps * cnt.Boltzmann * temperature) / ((
                    np.sqrt(2 * 10 ** 3 * cnt.Avogadro) * cnt.e)*(1/debye_wv)*10**-10)
        return sqrtI**2

    def get_structures(self):
        if self.temper:
            dcds = self._temper_trj_reorder()
        else:
            dcds = glob.glob(os.path.join(self.o_wd, '*.dcd'))
            dcds = sorted(dcds, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        structures = []
        for dcd in dcds:
            structures.append(md.load(dcd, top=self.topology_path))
        return structures

    def _get_lmp_file(self):
        lmp = glob.glob(os.path.join(self.o_wd, '*.lmp'))[0]
        with open(os.path.join(self.o_wd, lmp), 'r') as lmp_file:
            return lmp_file.readlines()

    def _get_data_file(self):
        data = glob.glob(os.path.join(self.o_wd, '*.data'))[0]
        with open(os.path.join(self.o_wd, data), 'r') as data_file:
            return data_file.readlines()

    def _get_log_files(self):
        logs = glob.glob(os.path.join(self.o_wd, 'log.lammps*'))
        if self.temper:
            logs.remove(os.path.join(self.o_wd, 'log.lammps'))
            logs = sorted(logs, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        log_data_store = []
        for log in logs:
            with open(os.path.join(self.o_wd, log), 'r') as log_file:
                log_data_store.append(log_file.readlines())
        return log_data_store

    def _is_temper(self):
        temper = False
        for line in self.lmp_file:
            if "temper" in line:
                temper = True
                break
        return temper

    # TODO : SLOW, PARALLELIZE ?
    def _temper_trj_reorder(self):
        def _get_temper_switches():
            # Temper runs are single file
            log_lmp = open(os.path.join(self.o_wd, 'log.lammps'), 'r')
            lines = log_lmp.readlines()
            T_start = 0
            T_end = len(lines)
            for x, line in enumerate(lines):
                if 'Step' in line:
                    T_start = x + 1
                    break
            in_temp_log = np.loadtxt(os.path.join(self.o_wd, 'log.lammps'), skiprows=T_start, max_rows=T_end - T_start,
                               dtype='int')
            return in_temp_log

        if not self.force_reorder:
            if glob.glob(os.path.join(self.o_wd, '*reorder*')):
                print("Omitting temper reordering (reorder files already present)")
                xtcs = glob.glob(os.path.join(self.o_wd, '*reorder*'))
                xtcs = sorted(xtcs, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
                return xtcs

        temp_log = _get_temper_switches()

        # !!!!!!!!! Assuming dump frequency is the same as thermo frequence !!!!!!!!!
        self.data = self.get_lmp_temper_data(progress=False)

        T_swap_rate = temp_log[1, 0] - temp_log[0, 1]
        traj_save_rate = int(self.data[0, 1, 0] - self.data[0, 0, 0])
        if traj_save_rate < T_swap_rate:
            trj_frame_incr = int(T_swap_rate/traj_save_rate)
            runner = 1
        else:
            trj_frame_incr = 1
            runner = int(traj_save_rate/T_swap_rate)

        top = self.make_initial_frame()[0]

        xtcs = glob.glob(os.path.join(self.o_wd, '*.dcd*'))
        if not xtcs:
            xtcs = glob.glob(os.path.join(self.o_wd, '*temper*.xtc*'))
        if not xtcs:
            raise SystemError("No trajectory files to read (attempted .xtc and .dcs)")
        xtcs = sorted(xtcs, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

        reordered_data = np.zeros_like(self.data)

        trajs, trajs_xyz = [], []
        nmin = []
        for xtc in xtcs:
            tr = md.load(xtc, top=top)
            nmin.append(tr.n_frames)
            trajs.append(tr)
            trajs_xyz.append(tr.xyz)
        # IN CASE SIMULATION HAS NOT FINISHED, FORCE SAME NUMBER OF FRAMES BETWEEN REPLICAS
        for k, tr in enumerate(trajs):
            trajs[k] = tr[:np.array(nmin).min()]
        reordered_trajs = np.zeros(shape=(len(trajs), trajs[0].n_frames, trajs[0].n_atoms, 3))
        c = 0
        # TODO: Time consuming...
        for i in range(0, temp_log.shape[0], runner):
            print(f'Swapping progress : {i/temp_log.shape[0]*100.:.2f} %', end='\r')
            if runner != 1:
                cr = slice(c, c+1, 1)
                c += 1
            else:
                #TODO MIGHT BLOW UP ?
                cr = slice(trj_frame_incr*(i), trj_frame_incr*(i+1), 1)
            for Ti, T in enumerate(temp_log[i, 1:]):
                reordered_data[T, cr, :] = self.data[Ti, cr, :]
                reordered_trajs[T, cr, :, :] = trajs[Ti][cr].xyz

        print('\r', end='\r')
        for i, rtrj in enumerate(reordered_trajs):
            trajs[i].xyz = rtrj

        xtc_paths = []
        for k in range(len(trajs)):
            f = f'../default_output/reorder-{k}.xtc'
            xtc_paths.append(os.path.abspath(f))
            # trajs[k].save_xtc(f)
            trajs[k].save_lammpstrj(f)
            # shutil.copyfile(f, os.path.join(self.o_wd, f'reorder-{k}.xtc'))
            shutil.copyfile(f, os.path.join(self.o_wd, f'reorder-{k}.lammpstrj'))
        self.data = reordered_data
        print(reordered_data.shape)
        np.savetxt('../default_output/datareorder.lammps', reordered_data[0,:,:])
        return xtc_paths
