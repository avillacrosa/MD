import numpy as np
import os
import definitions
import re
import random
import shutil
import time
import mdtraj as md
import glob
import multiprocessing as mp
from subprocess import run, PIPE
import pathlib
import scipy.constants as cnt


class LMP:
    """
    LMP is a base class to translate parameters of a finished or running LAMMPS run to python.
    It can also perform some basic transformations like reorder a replica exchange
    """
    def __init__(self, oliba_wd, force_reorder=False, equil_frames=300, silent=False):
        self.o_wd = oliba_wd
        self.this = os.path.dirname(os.path.dirname(__file__))

        self.silent = silent

        self.lmp2pdb = definitions.lmp2pdb
        self.lmp = definitions.lmp
        self.residue_dict = dict(definitions.residues)

        self.force_reorder = force_reorder
        self.temper = None
        self.every_frames, self.last_frames = None, None
        if self.o_wd is not None:
            self.data_file = self._get_data_file()
            self.lmp_files = self._get_lmp_files()
            self.temper = self._is_temper()
            self.rerun = self._is_rerun()
            self.log_files = self._get_log_files()
            self.topology_path = self._get_initial_frame()

            self.chains, self.chain_atoms = self.get_n_chains()
            self.box = self.get_box()
            self.protein = self.get_protein_from_sequence()
            self.sequence = self.get_seq_from_hps()
            self.data = self.get_lmp_data()
            self.structures = None
            self.equil_frames = equil_frames
            self.temperatures = self.get_temperatures()

    def get_lmp_dirs(self, path=None):
        """
        Find all directories containing a .lmp file below self.o_wd or path
        :param path: Path where to get all lmp directories
        :return: List of all lmp directories
        """
        if path is None:
            path = self.o_wd
        dirs = []
        for filename in pathlib.Path(path).rglob('lmp*.lmp'):
            if os.path.dirname(filename) not in dirs:
                dirs.append(os.path.dirname(filename))
        dirs.sort()
        return dirs

    def get_protein_from_sequence(self):
        """
        Search on the sequences directory for a file matching the used sequence in the LMP, obtained by translating data
        :return: Name (string) of the protein
        """

        sequence = self.get_seq_from_hps()
        stored_seqs = glob.glob(os.path.join(definitions.hps_data_dir, 'sequences/*.seq'))
        for seq_path in stored_seqs:
            with open(seq_path, 'r') as seq_file:
                seq = seq_file.readlines()[0]
                if sequence.replace('L', 'I') == seq.replace('L', 'I'):
                    protein = os.path.basename(seq_path).replace('.seq', '')
                    return protein

    def get_lmp_data(self, progress=True):
        """
        Save the thermo data generated by LMP (in log.lammps.T usually) in a numpy array
        :param progress:  Whether we want or not to display the progress of the run (% of completion)
        :return: ndarray with shape (T,frames,Ncols) where Ncols is the number of columns in thermo output (log.lammps)
        """
        data, data_ends, prog = [], [], []
        dstart = 0
        for log_data in self.log_files:
            data_start = 0
            data_end = 0
            for i, line in enumerate(log_data):
                if progress:
                    if "temper" in line or "run" in line:
                        if "rerun" not in line:
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
        if progress and prog and not self.rerun and not self.silent:
            print(f"> Run Completed at {ran_steps/np.array(prog).mean()*100:.2f}% for {self.protein}. Ran {ran_steps:.0f} steps for a total of {data.shape[1]} frames ")
        return data

    def get_eps(self):
        """
        Get medium permittivity from .lmp file
        :return: float Medium permittivity
        """
        eps = []
        for lmp_file in self.lmp_files:
            for line in lmp_file:
                if "dielectric" in line:
                    eps.append(re.findall(r'\d+', line)[0])
                    break
        return eps

    def get_ionic_strength(self):
        """
        Get the ionic_strength from .lmp file (by inverting Debye length)
        :return: float Ionic strength
        """
        ionic_strength = []
        for i, lmp_file in enumerate(self.lmp_files):
            for line in lmp_file:
                if "ljlambda" in line:
                    debye = re.findall(r'\d+\.?\d*', line)[0]
                    unroundedI = self.get_I_from_debye(float(debye), eps=float(self.get_eps()[i]))
                    if unroundedI >= 0.1:
                        ionic_strength.append(round(unroundedI, 1))
                        break
                    else:
                        ionic_strength.append(round(unroundedI, 3))
                        break
        return ionic_strength

    def get_temperatures(self):
        """
        Get the temperatures used on the simulation
        :return: ndarray with shape (T) containing temperatures
        """
        T = []
        if not self.temper:
            for lmp_file in self.lmp_files:
                for line in lmp_file:
                    if "velocity" in line:
                        T.append(re.findall(r'\d+', line)[0])
                        break
        else:
            for line in self.lmp_files[0]:
                if "variable T world" in line:
                    T = re.findall(r'\d+\.?\d*', line)
                    break
        T.sort()
        return np.array(T, dtype=float)

    def get_n_chains(self):
        """
        Get the number of chains of the simulation
        :return: [int nchains, int atoms_per_chain]
        """
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
        """
        Generate a new sequence with the same total charge but the positions of each q moved
        :param sequence: string of aminoacids with no spaces or linespaces
        :param keep_positions: boolean to choose whether we keep charged positions or we also move them
        :param shuffles: int how many shuffles we want to generate
        :return: ndarray of the sequences, ndarray of kappas each sequence, ndarray of scds
        """
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
    def get_charge_seq(self, window=9):
        """
        Get the windowed charged sequence of the sequence
        :param window: int window to calculate the charge
        :return: ndarray windowed charge of the sequence, ndarray of positive values, ndarray of negative values
        """
        win = np.zeros(len(self.sequence))
        rr = int((window-1)/2)
        if rr >= 0:
            for i in range(self.chain_atoms):
                jaa = self.sequence[i]
                win[i] += self.residue_dict[jaa]["q"]
        else:
            for i in range(self.chain_atoms):
                for j in range(-rr, rr+1):
                    if len(self.sequence) > i+j > 0:
                        jaa = self.sequence[i+j]
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
        """
        Calculate the delta_i to calculate the kappa as Pappu
        :param sequence: string with aminoacidic sequence
        :param g: int blob size as defined by Pappu
        :return: float delta_i, that is, not normalized
        """
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
                sigma_i = (f_plus_i - f_minus_i) ** 2 / (f_plus_i + f_minus_i)
                delta += (sigma_i - sigma) ** 2 / N
        return delta

    def chan_scd(self, sequence):
        """
        CURRENTLY BROKEN
        :param sequence:
        :return:
        """
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

    def save_movies(self, frames=100, T=None, center=True):
        structures = self.structures
        if frames == 'all':
            frames = structures[0].n_frames
        if T is not None:
            structures = [self.structures[T]]
        for T, struct in enumerate(structures):
            frame_step = int(struct.n_frames/frames) if int(struct.n_frames/frames) != 0 else 1
            small_struct = struct[::frame_step]
            small_struct = small_struct[:frames]
            self.save_lammpstrj(small_struct, fname=(os.path.join(definitions.movie_dir, f'{self.protein}x{self.chains}-T{self.temperatures[T]:.0f}.lammpstrj')), center=center)
            small_struct.save_xtc(os.path.join(definitions.movie_dir, f'{self.protein}x{self.chains}-T{self.temperatures[T]:.0f}.xtc'))
            small_struct.save_netcdf(os.path.join(definitions.movie_dir, f'{self.protein}x{self.chains}-T{self.temperatures[T]:.0f}.netcdf'))
            if os.path.exists(os.path.join(self.o_wd, 'topo.pdb')):
                shutil.copyfile(os.path.join(self.o_wd, 'topo.pdb'), os.path.join(definitions.movie_dir, f'{self.protein}x{self.chains}-T{self.temperatures[T]:.0f}.pdb'))
            else:
                pdb = glob.glob(os.path.join(self.o_wd, '*.pdb'))[0]
                shutil.copyfile(pdb, os.path.join(definitions.movie_dir, f'{self.protein}x{self.chains}-T{self.temperatures[T]:.0f}.pdb'))

    def save_last_frame(self):
        for T, struct in enumerate(self.structures):
            struct[-1].save_pdb(os.path.join(definitions.movie_dir, f'{self.protein}x{self.chains}-T{self.temperatures[T]:.0f}.pdb'))

    def get_box(self):
        dict = {}
        for line in self.data_file:
            if "xlo" and "xhi" in line:
                dict["x"] = re.findall(r'-?\d+', line)
            if "ylo" and "yhi" in line:
                dict["y"] = re.findall(r'-?\d+', line)
            if "zlo" and "zhi" in line:
                dict["z"] = re.findall(r'-?\d+', line)
        return dict

    # TODO : MULTICHAIN CASE ?
    def get_seq_from_hps(self):
        """
        Get the sequence from the data.data file considering it belongs to an HPS dynamic
        :return: string of the sequence
        """
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
        return seq_str[:self.chain_atoms]

    def run(self, file, n_cores=1):
        """
        Run a lammps job
        :param file: string, lmp file to run lammps
        :param n_cores: int, number of cores to use for mpirun
        :return: standard ouput ?
        """
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

    def get_I_from_debye(self, debye_wv, eps=80., temperature=300):
        """
        Static method. Returns the ionic strength in molar given a debye wavevector, a temperature and an epsilon
        :param debye_wv: float, debye's wavevector
        :param eps: float, permittivity of the medium
        :param temperature:
        :return: float, ionic strength (M)
        """
        sqrtI = np.sqrt(cnt.epsilon_0 * eps * cnt.Boltzmann * temperature) / ((
                    np.sqrt(2 * 10 ** 3 * cnt.Avogadro) * cnt.e)*(1/debye_wv)*10**-10)
        return sqrtI**2

    def get_structures(self, total_frames=1000, every=1):
        """
        Get mdtraj objects for every unique trajectory generated by a LAMMPS run
        :param total_frames: int, maximum frames that we wish to load
        :param every: int, take 1 of "every" frames
        :return: list[T], of mdtraj trajectories for each temperature
        """
        if self.temper:
            dcds = self.temper_reorder()
        else:
            dcds = glob.glob(os.path.join(self.o_wd, '*.dcd'))
            if len(dcds) > 1:
                dcds = sorted(dcds, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        structures = []
        n_frames = []
        for dcd in dcds:
            tr = md.load(dcd, top=self.topology_path)
            if not self.rerun:
                if total_frames:
                    if int(tr.n_frames/every) < total_frames:
                        every = int(tr.n_frames/total_frames)-1
                        if every <= 0: every = 1
                tr = tr[self.equil_frames::every]
                tr = tr[-total_frames:]
            n_frames.append(tr.n_frames)
            structures.append(tr)
        minn = np.array(n_frames).min()
        if not self.temper:
            for i in range(len(structures)):
                structures[i] = structures[i][:minn]
        if total_frames is not None and not self.rerun and not self.silent:
            print(f"> Taking frames every {every} for a total of {total_frames} to avoid strong correlations")
            self.last_frames = total_frames
            self.every_frames = every
        return structures

    def temper_reorder(self, save_lammpstrj=True):
        """
        Reorder the trajectories generated by LAMMPS so that temperature remains constant. First it will attempt to read
        DCD, then XTC, then fail if there's nothing more. If succeeded, it will generate a file for each temperature
        called reorder-T.dcd
        :param save_lammpstrj: Whether we want to save a .lammpstrj reordered file (useful for Ovito / Reruns)
        :return: list[T], list of strings containing the paths to the newly generated reordered trajectories.
        """
        self.data = self._data_reorder()

        # TODO : ! :(
        # print(glob.glob(os.path.join(self.o_wd, '[!_reorder][*.dcd]')))
        # TODO : ! :(
        if not self.force_reorder:
            if glob.glob(os.path.join(self.o_wd, '*reorder*.dcd')):
                files = glob.glob(os.path.join(self.o_wd, '*reorder*.dcd'))
                files = sorted(files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
                return files
        temp_log = self._get_temper_switches()
        runner, trj_frame_incr = self._get_swap_rates()

        files = glob.glob(os.path.join(self.o_wd, '*.dcd*'))
        if not files:
            files = glob.glob(os.path.join(self.o_wd, '*.xtc*'))
        if not files:
            raise SystemError("No trajectory files to read (attempted .xtc and .dcs)")
        files = sorted(files, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))

        trajs, trajs_xyz = [], []
        nmin = []

        # We are not using self.structures here because we need the full trajectories
        for f in files:
            tr = md.load(f, top=self._get_initial_frame())
            nmin.append(tr.n_frames)
            trajs.append(tr)
        for k, tr in enumerate(trajs):
            trajs[k] = tr[:np.array(nmin).min()]
            trajs_xyz.append(trajs[k].xyz)

        trajs_xyz = np.array(trajs_xyz)
        xyzcop = trajs_xyz.copy()

        c = 0
        for i in range(0, temp_log.shape[0], runner):
            if runner != 1:
                cr = slice(c, c + 1, 1)
                c += 1
            else:
                cr = slice(trj_frame_incr * (i), trj_frame_incr * (i + 1), 1)
            trajs_xyz[temp_log[i, 1:], cr, :] = xyzcop[:, cr, :]
            print(f'Swapping progress for {self.protein} : {i / temp_log.shape[0] * 100.:.2f} %', end='\r')
        print(" " * 40, end='\r')

        for i, rtrj in enumerate(trajs_xyz):
            trajs[i].xyz = trajs_xyz[i, :, :, :]
        xtc_paths = []
        for k in range(len(trajs)):
            dcd = os.path.join(self.o_wd, f'reorder-{k}.dcd')
            xtc_paths.append(os.path.abspath(dcd))
            trajs[k].save_dcd(dcd)
            if save_lammpstrj:
                # Time sink is here
                self.save_lammpstrj(trajs[k], os.path.join(self.o_wd, f'reorder-{k}.lammpstrj'))
        return xtc_paths

    def save_lammpstrj(self, traj, fname, center=False):
        """
        Save a trajectory (it's coordinates) in a custom LAMMPS format, since mdtraj seems to do something weird
        :param traj: mdtraj trajectory
        :param T: temperature index
        :return: Nothing
        """
        save_step = int(self.data[0, 1, 0] - self.data[0, 0, 0])
        f = ''
        for frame in range(traj.n_frames):
            frame_center = traj.xyz[frame, :, :].mean(axis=0)
            f += f'ITEM: TIMESTEP \n'
            f += f'{frame*save_step} \n'
            f += f'ITEM: NUMBER OF ATOMS \n'
            f += f'{traj.n_atoms} \n'
            f += f'ITEM: BOX BOUNDS pp pp pp \n'
            for key in self.box:
                f += f'{self.box[key][0]} {self.box[key][1]} \n'
            f += f'ITEM: ATOMS id type chain x y z \n'
            for c in range(self.chains):
                for i, aa in enumerate(self.sequence):
                    xyz = traj.xyz[frame, i+self.chain_atoms*c, :]*10
                    xyzf = xyz.copy()
                    if center:
                        xyzf = xyzf - frame_center*10
                    f += f'{(c+1)*(i+1)} {self.residue_dict[aa]["id"]} {c} {xyzf[0]:.5f} {xyzf[1]:.5f} {xyzf[2]:.5f} \n'
        with open(fname, 'tw') as lmpfile:
            lmpfile.write(f)

    def _get_initial_frame(self):
        """
        Create the pdb containing the same information as the "data" topology file from LAMMPS
        :return: string, path pointing to the generated pdb
        """
        if os.path.exists(os.path.join(self.o_wd, 'topo.pdb')):
            return os.path.join(self.o_wd, 'topo.pdb')
        else:
            files = glob.glob(os.path.join(self.o_wd, 'data.data'))
            file = os.path.basename(files[0])
            file = file.replace('.data', '')
            lammps2pdb = self.lmp2pdb
            os.system(lammps2pdb + ' ' + os.path.join(self.o_wd, file))
            fileout = os.path.join(self.o_wd, file) + '_trj.pdb'
            return fileout

    def _get_hps_params(self):
        """
        Read and add the lambdas to the dict (self.residue_dict) containing all HPS aminoacids
        :return: nothing
        """
        for key in self.residue_dict:
            for lam_key in definitions.lambdas:
                if self.residue_dict[key]["name"] == lam_key:
                    self.residue_dict[key]["lambda"] = definitions.lambdas[lam_key]
            for sig_key in definitions.sigmas:
                if self.residue_dict[key]["name"] == sig_key:
                    self.residue_dict[key]["sigma"] = definitions.sigmas[sig_key]

    def _get_lmp_files(self):
        """
        Get the content of the main LAMMPS script (lmp.lmp) file
        :return: list[N_lines], list containing all lines of the main .lmp file
        """
        lmp = glob.glob(os.path.join(self.o_wd, 'lmp*.lmp'))
        lmp_files = []
        for l_file in lmp:
            with open(os.path.join(self.o_wd, l_file), 'r') as lmp_file:
                lmp_files.append(lmp_file.readlines())
        return lmp_files

    def _get_data_file(self):
        """
        Get the content of the LAMMPS topology (data.data) file
        :return: list[N_lines], list containing all lines of the topology file .data
        """
        data = glob.glob(os.path.join(self.o_wd, '*.data'))[0]
        with open(os.path.join(self.o_wd, data), 'r') as data_file:
            return data_file.readlines()

    def _get_log_files(self):
        """
        Get the content of the log.lammps.T files. If it is a temper run, discard log.lammps, since it should include
        the switch rates
        :return: list[T, N_lines], list containing all lines on every log.lammps.T file
        """
        logs = glob.glob(os.path.join(self.o_wd, 'log*.lammps*'))
        if self.temper:
            logs.remove(os.path.join(self.o_wd, 'log.lammps'))
            logs = sorted(logs, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        log_data_store = []
        for log in logs:
            with open(os.path.join(self.o_wd, log), 'r') as log_file:
                log_data_store.append(log_file.readlines())
        return log_data_store

    def _is_temper(self):
        """
        Get if the run is a REX or not
        :return: bool
        """
        temper = False
        for line in self.lmp_files[0]:
            if "temper" in line:
                temper = True
                break
        return temper

    def _is_rerun(self):
        """
        Get if the run is a rerun or not
        :return: bool
        """
        rerun = False
        for line in self.lmp_files[0]:
            if "rerun" in line:
                rerun = True
                break
        return rerun

    def _data_reorder(self):
        """
        Reorder the data extracted from the log.lammps.T files so that temperature is constant (it is not on the
        log.lammps.T files)
        :return: ndarray[T,frames,observables], where observables is the number of columns in a log.lammps.T file
        """
        temp_log = self._get_temper_switches()
        runner, trj_frame_incr = self._get_swap_rates()
        c = 0
        d = self.data.copy()
        for i in range(0, temp_log.shape[0], runner):
            if runner != 1:
                cr = slice(c, c+1, 1)
                c += 1
            else:
                cr = slice(trj_frame_incr*(i), trj_frame_incr*(i+1), 1)
            self.data[temp_log[i, 1:], cr, :] = d[:, cr, :]
        return self.data.copy()

    def _get_temper_switches(self):
        """
        Get the temperatures of each trajectory at a given step, that is, read the log.lammps file if we're in the
        temper case
        :return:
        """
        if not self.temper:
            return
        with open(os.path.join(self.o_wd, 'log.lammps'), 'r') as temp_switches:
            lines = temp_switches.readlines()
            T_start, T_end = 0, len(lines)
            for x, line in enumerate(lines):
                if 'Step' in line:
                    T_start = x + 1
                    break
            in_temp_log = np.genfromtxt(lines[T_start:T_end], dtype='int')
        return in_temp_log

    def _get_swap_rates(self):
        """
        Get how many trajectory frames we have for every attempted REX change
        :return: float, float, runner = every how many T changes we must read, trj_frame_incr = how many ts between switch in trj
        """
        # TODO : INCOHERENT AMB LA RESTA DEL CODI... (La resta definim un self.something i l'usem no l'anem cridant dins les funcions)
        temp_log = self._get_temper_switches()
        T_swap_rate = temp_log[1, 0] - temp_log[0, 1]
        traj_save_rate = int(self.data[0, 1, 0] - self.data[0, 0, 0])
        if traj_save_rate < T_swap_rate:
            trj_frame_incr = int(T_swap_rate / traj_save_rate)
            runner = 1
        else:
            trj_frame_incr = 1
            runner = int(traj_save_rate / T_swap_rate)
        # TODO: one is the inverse of the other ?
        return runner, trj_frame_incr
