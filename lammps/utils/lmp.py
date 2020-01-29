import numpy as np
import os
import definitions
import re
import random
import math
import shutil
import mdtraj as md
import glob
import multiprocessing as mp
from subprocess import run, PIPE
import pathlib
from numba import jit


class LMP:
    def __init__(self, oliba_wd, temper=False, force_reorder=False):
        # TODO DO TEMPER HANDLE AT START
        self.res_dict = definitions.residues
        # TODO seq from file ?
        self.sequence = None
        self.residue_dict = dict(definitions.residues)

        self.temper = temper
        self.force_reorder = force_reorder

        self.o_wd = oliba_wd
        self.p_wd = oliba_wd.replace('/perdiux', '')

        self.lmp = '/home/adria/local/lammps/bin/lmp'
        self.lmp_drs = self.get_lmp_dirs()
        self.n_drs = len(self.lmp_drs)

        self.hps_epsilon = 0.2
        self.hps_pairs = None
        # if temper:
        #     self.data = self.get_lmp_temper_data()
        # else:
        # self.data = self.get_lmp_data()
        self.data = None
        self.lmp2pdb = '/home/adria/perdiux/src/lammps-7Aug19/tools/ch2lmp/lammps2pdb.pl'
        self.re_reorder = '/home/adria/local/bin/reorder_remd_traj.py'

    def get_lmp_dirs(self, path=None):
        if path is None:
            path = self.o_wd
        dirs = []
        for filename in pathlib.Path(path).rglob('*lmp'):
            dirs.append(os.path.dirname(filename))
        dirs.sort()
        return dirs

    def make_initial_frame(self, dirs=None):
        pdb_paths = []
        if dirs is None:
            dirs = self.lmp_drs
        for dir in dirs:
            files = glob.glob(os.path.join(dir, '*.data'))
            if glob.glob(os.path.join(dir, '*.pdb')):
                pdb_paths.append(glob.glob(os.path.join(dir, '*.pdb'))[0])
                continue
            file = os.path.basename(files[0])
            file = file.replace('.data', '')
            lammps2pdb = self.lmp2pdb
            os.system(lammps2pdb + ' ' + os.path.join(dir, file))
            fileout = os.path.join(dir, file) + '_trj.pdb'
            pdb_paths.append(fileout)
        return pdb_paths

    def get_lmp_data(self, progress=False):
        data = []
        prog = 0
        for d in self.lmp_drs:
            log_lmp = open(os.path.join(d, 'log.lammps'), 'r')
            lines = log_lmp.readlines()
            data_start = 0
            data_end = 0
            for i, line in enumerate(lines):
                if progress:
                    if "run" in line:
                        prog = int(re.findall(r'\d+', line)[0])
                if "Step" in line:
                    data_start = i + 1
                if "Loop" in line:
                    data_end = i
                if data_end and data_start != 0:
                    break
            if data_end == 0:
                data_end = len(lines)
            dat = np.loadtxt(os.path.join(d, 'log.lammps'), skiprows=data_start, max_rows=data_end - data_start)
            data.append(dat)
            print(f"Run Completed at {dat[:, 0].max()/prog*100:.2f} %")
        data = np.array(data)
        return data

    def get_lmp_temper_data(self, lmp_directories=None, progress=False):
        data = []
        prog = []
        dends, dstarts = [], 0
        for d in self.lmp_drs:
            if lmp_directories is None:
                lmps = glob.glob(os.path.join(d, "log.lammps.*"))
            else:
                lmps = lmp_directories
            lmps = sorted(lmps, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
            for lmp in lmps:
                log_lmp = open(os.path.join(d, lmp), 'r')
                lines = log_lmp.readlines()
                data_start = 0
                data_end = 0
                for i, line in enumerate(lines):
                    if progress:
                        if "temper" in line:
                            if int(re.findall(r'\d+', line)[0]) > 100000:
                                prog.append(int(re.findall(r'\d+', line)[0]))
                    if "Step" in line:
                        data_start = i + 1
                    if "Loop" in line:
                        data_end = i
                    if data_end and data_start != 0:
                        break
                if data_end == 0:
                    data_end = len(lines)
                dends.append(data_end)
                dstarts = data_start
        dmin = np.array(dends).min()
        for i, lmp in enumerate(lmps):
            dat = np.loadtxt(os.path.join(d, lmp), skiprows=dstarts, max_rows=dmin - dstarts)
            data.append(dat)
            step_max = dat[:, 0].max()
        if progress:
            print(f"Run Completed at {step_max / np.array(prog).mean() * 100:.2f} %")
        data = np.array(data)
        return data

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

    def get_charge_seq(self, sequence, window=9):
        charged_plus = []
        charged_minus = []
        # for i, aa in enumerate(sequence):
        #     if self.residue_dict[aa]["q"] < 0:
        #         charged_minus.append(i)
        #     if self.residue_dict[aa]["q"] > 0:
        #         charged_plus.append(i)
        win = np.zeros(len(sequence))
        rr = int((window-1)/2)
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
            # if self.residue_dict[aa]["q"] < 0:
            #     charged_minus.append(i)
            # if self.residue_dict[aa]["q"] > 0:
            #     charged_plus.append(i)
        return sorted(charged_plus + charged_minus), charged_plus, charged_minus

    def pappu_delta(self, sequence, g=5):
        # TODO: ONLY FOR PROLINE-LESS SEQUENCES! What the hell ?
        N = len(sequence) - (g - 1)
        itot, iplus, iminus = self.get_charge_seq(sequence)
        f_plus = len(iplus)/len(itot)
        f_minus = len(iminus)/len(itot)
        sigma = (f_plus - f_minus) ** 2 / (f_plus + f_minus)
        delta = 0
        for i in range(N):
            blob = sequence[i:i + g]
            total, plus, minus = self.get_charge_seq(blob)
            if len(plus) != 0 or len(minus) != 0:
                f_plus_i = len(plus) / (len(plus) + len(minus))
                f_minus_i = len(minus) / (len(plus) + len(minus))
                sigma_i = (f_plus_i - f_minus_i) ** 2 / (f_plus_i + f_minus_i)
                delta += (sigma_i - sigma) ** 2 / N
        return delta

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

    def set_sequence(self, seq, from_file=False):
        # TODO !!
        if from_file:
            print("TODO")
        self.sequence = seq

    def decode_seq_from_hps(self):
        for lmp_dir in self.lmp_drs:
            data_paths = glob.glob(os.path.join(lmp_dir, "*.data"))
            data_path = data_paths[0]
            with open(data_path, 'r') as data_file:
                lines = data_file.readlines()
                mass_range, atom_range = [0, 0], [0, 0]
                reading_masses, reading_atoms = False, False
                # TODO Not too smart
                for i, line in enumerate(lines):
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
            masses = np.loadtxt(data_path, skiprows=mass_range[0], max_rows=mass_range[1] - mass_range[0])
            atoms = np.loadtxt(data_path, skiprows=atom_range[0], max_rows=atom_range[1] - atom_range[0])
            mass_dict = {}
            for mass in masses:
                for res in self.res_dict:
                    if self.res_dict[res]["mass"] == mass[1]:
                        mass_dict[int(mass[0])] = res
                        break
            seq = []
            for atom in atoms:
                seq.append(mass_dict[atom[2]])
            seq_str = ''.join(seq)
            print(
                "Finding used sequence. Note that Isoleucine and Leucine have the same exact HPS parameters. Therefore, they are not distinguishable.")
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

    def get_hps_params(self):
        for key in self.residue_dict:
            for lam_key in definitions.lambdas:
                if self.residue_dict[key]["name"] == lam_key:
                    self.residue_dict[key]["lambda"] = definitions.lambdas[lam_key]
            for sig_key in definitions.sigmas:
                if self.residue_dict[key]["name"] == sig_key:
                    self.residue_dict[key]["sigma"] = definitions.sigmas[sig_key]

    def get_hps_pairs(self, from_file=None):
        lines = ['pair_coeff          *       *       0.000000   0.000    0.000000   0.000   0.000\n']

        if from_file:
            lambda_gen = np.genfromtxt(from_file)
            count = 0
        for i in range(len(self.key_ordering)):
            for j in range(i, len(self.key_ordering)):
                res_i = self.residue_dict[self.key_ordering[i]]
                res_j = self.residue_dict[self.key_ordering[j]]
                lambda_ij = (res_i["lambda"] + res_j["lambda"])/2
                sigma_ij = (res_i["sigma"] + res_j["sigma"])/2
                if from_file:
                    lambda_ij = lambda_gen[count]
                    count += 1
                if res_i["q"] != 0 and res_j["q"] != 0:
                    cutoff = 35.00
                else:
                    cutoff = 0.0
                lambda_ij = lambda_ij * self.hps_scale
                line = 'pair_coeff         {:2d}      {:2d}       {:.6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                    i + 1, j + 1, self.hps_epsilon, sigma_ij, lambda_ij, 3 * sigma_ij, cutoff)
                lines.append(line)
        self.hps_pairs = lines

    # TODO : SLOW, PARALLELIZE ?
    def _temper_trj_reorder(self):
        def _get_temper_switches():
            # Temper runs are single file
            log_lmp = open(os.path.join(self.o_wd, 'log.lammps'), 'r')
            lines = log_lmp.readlines()
            T_start = 0
            T_end = len(lines)
            for i, line in enumerate(lines):
                if 'Step' in line:
                    T_start = i + 1
                    break
            T_log = np.loadtxt(os.path.join(self.o_wd, 'log.lammps'), skiprows=T_start, max_rows=T_end - T_start,
                               dtype='int')
            return T_log

        # TODO : !!!!!!!!!!
        if not self.force_reorder:
            if glob.glob(os.path.join(self.o_wd, '*reorder*')):
                print("Omitting temper reordering (reorder files already present)")
                xtcs = glob.glob(os.path.join(self.o_wd, '*reorder*'))
                xtcs = sorted(xtcs, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
                return xtcs

        T_log = _get_temper_switches()

        # !!!!!!!!! Assuming dump frequency is the same as thermo frequence !!!!!!!!!
        self.data = self.get_lmp_temper_data(progress=False)

        T_swap_rate = T_log[1, 0] - T_log[0, 1]
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
        print(runner)
        for i in range(0, T_log.shape[0], runner):
            print(f'Swapping progress : {i/T_log.shape[0]*100.:.2f} %', end='\r')
            if runner != 1:
                cr = slice(c, c+1, 1)
                c += 1
            else:
                #TODO MIGHT BLOW UP ?
                cr = slice(trj_frame_incr*(i), trj_frame_incr*(i+1), 1)
            for Ti, T in enumerate(T_log[i, 1:]):
                reordered_data[T, cr, :] = self.data[Ti, cr, :]
                reordered_trajs[T, cr, :, :] = trajs[Ti][cr].xyz

        print('\r', end='\r')
        for i, rtrj in enumerate(reordered_trajs):
            trajs[i].xyz = rtrj

        xtc_paths = []
        for k in range(len(trajs)):
            f = f'../default_output/reorder-{k}.xtc'
            xtc_paths.append(os.path.abspath(f))
            trajs[k].save_xtc(f)
            shutil.copyfile(f, os.path.join(self.o_wd, f'reorder-{k}.xtc'))
        self.data = reordered_data
        print(reordered_data.shape)
        np.savetxt('../default_output/datareorder.lammps', reordered_data[0,:,:])
        return xtc_paths

# TODO Handle Temperatures or Ionic strengths...
