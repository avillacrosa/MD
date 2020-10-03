import itertools
import shutil
import mdtraj as md
import math
import scipy.constants as cnt
import numpy as np
import os
import scipy
import time
import pathlib
import definitions
import lmp
import lmpsetup
import pathos.multiprocessing as mp
import multiprocessing as mpp
import statsmodels.tsa.stattools
from scipy.optimize import curve_fit, fsolve
import MDAnalysis.analysis as mda
import scipy.integrate as integrate
from MDAnalysis.analysis import contacts as compute_contacts
from numba import jit
import pandas as pd
import glob


class Analysis:
    """
    Not really a necessary tool, but just a summary of interesting analytical observables for HPS
    :param max_frames: int, how many frames are to be analyzed
    :param kw: kwargs
    """

    def __init__(self, framework):
        self.c_rg = None
        self.framework = framework

    def multi_chain_contact_map(self, temperature=None, contact_cutoff=6):
        inter_maps = []
        intra_maps = []
        if temperature is None:
            temp_range = range(len(self.temperatures))
        else:
            temp_range = [temperature]
        for T in temp_range:
            struct = self.structures[T]
            coords = struct.xyz
            contacts_frame = np.empty(shape=(coords.shape[1], coords.shape[1]))
            # n_frames = coords.shape[0]
            n_frames = 100
            for frame in range(n_frames):
                print(f"Doing frame : {frame}/{n_frames}", end='\r')
                cf = coords[-frame, :, :]
                contacts_frame += mda.distances.contact_matrix(cf, cutoff=contact_cutoff)
            contacts_frame /= n_frames
            dd = np.zeros(shape=(self.chains, self.chains, self.chain_atoms, self.chain_atoms)) - 1

            for c1 in range(self.chains):
                for c2 in range(self.chains):
                    c1_slice = slice(c1 * self.chain_atoms, (c1 + 1) * self.chain_atoms)
                    c2_slice = slice(c2 * self.chain_atoms, (c2 + 1) * self.chain_atoms)
                    dd[c1, c2] = contacts_frame[c1_slice, c2_slice]

            intra = np.diagonal(dd, axis1=0, axis2=1).mean(axis=2)
            inter = np.zeros(shape=(self.chain_atoms, self.chain_atoms))
            for chain in range(self.chains - 1):
                inter += np.diagonal(dd, axis1=0, axis2=1, offset=chain + 1).mean(axis=2)
                inter += inter.T / 2

            inter_maps.append(inter)
            intra_maps.append(intra)
        inter = np.array(inter_maps)
        intra = np.array(intra_maps)
        return intra, inter

    def intra_chain_contact_map(self, temperature=None, contact_cutoff=6):
        intra_maps = []
        if temperature is None:
            temp_range = range(len(self.temperatures))
        elif len(temperature) > 1:
            temp_range = temperature
        else:
            temp_range = [temperature]
        for T in temp_range:
            struct = self.structures[T]
            coords = struct.xyz
            contacts_frame = np.zeros(shape=(coords.shape[1], coords.shape[1]))
            n_frames = struct.n_frames
            for frame in range(n_frames):
                # print(f"Doing frame : {frame}/{n_frames}", end='\r')
                cf = coords[-frame, :, :]
                contacts_frame += mda.distances.contact_matrix(cf, cutoff=contact_cutoff)
            contacts_frame /= n_frames
            intra_maps.append(contacts_frame)
        intra = np.array(intra_maps)
        return intra

    def map_to_residue(self, contacts=None, normed=False, contact_cutoff=6, T=0):
        if contacts is None:
            if self.chains != 1:
                contacts = self.multi_chain_contact_map(temperature=T, contact_cutoff=contact_cutoff)[1]
            else:
                contacts = np.array([self.intra_chain_contact_map(temperature=T, contact_cutoff=contact_cutoff)[0]])

        seq_to_id = []
        for res in self.sequence:
            seq_to_id.append(self.residue_dict[res]["id"])

        aa_map = np.zeros(shape=(20, 20))

        for i, id1 in enumerate(seq_to_id):
            for j, id2 in enumerate(seq_to_id):
                aa_map[id1 - 1, id2 - 1] += contacts[0, i, j]

        if normed:
            counter = {}
            for res in self.residue_dict:
                counter[self.residue_dict[res]["id"] - 1] = self.sequence.count(res)
            print(counter)
            for i in range(len(self.residue_dict)):
                for j in range(len(self.residue_dict)):
                    aa_map[i, j] /= counter[i] * counter[j]
        return aa_map

    def intra_distance_map(self, contacts=False, temperature=None, normed=False):
        """
        Calculate intrachain distances using MDTraj
        :param contacts: bool, compute contacts instead of distances
        :param temperature: int, temperature index if we wish to compute a single T (if temper)
        :return: ndarray[T, frames, n_atoms, n_atoms], distances per frame, to get uncertainities
        """
        structures = self.structures.copy()
        contact_maps = []
        if normed:
            d = np.zeros(shape=(448, 448))
            for i in range(448):
                for j in range(i, 448):
                    d[i][j] = 1.12 * abs(i - j) ** 0.5
                    d[j][i] = d[i][j]
        if temperature is not None:
            structures = [structures[temperature]]
        for traj in structures:
            pairs = list(itertools.product(range(traj.top.n_atoms - 1), range(1, traj.top.n_atoms)))
            d = md.compute_distances(traj, pairs).astype('float16')
            d = md.geometry.squareform(d, pairs)
            contact_map = d
            if contacts:
                contact_map = mda.contacts.contact_matrix(contact_map, radius=6)
            if normed:
                contact_map -= d
            contact_maps.append(contact_map)
        if self.low_mem:
            self.structures = None
        return np.array(contact_maps, dtype='float32')

    def async_inter_distance_map(self, contacts=False, temperature=None):
        print("PARALLEL CALC")
        """
        Calculate both intrachain and interchain distances using MDTraj for the multichain case
        :param contacts: bool, compute contacts instead of distances
        :param temperature: int, temperature index if we wish to compute a single T (if temper)
        :return: ndarray[T, n_atoms, n_atoms], distances per temperature (saving per frame is too costly)
        """

        def parallel_calc(initial_frame, final_frame, i):
            inter_chain = np.zeros(shape=(self.chains, self.chain_atoms, self.chain_atoms), dtype='float64')
            intra_chain = np.zeros(shape=(self.chains, self.chain_atoms, self.chain_atoms), dtype='float64')
            traj = structures[i]
            for frame in range(initial_frame, final_frame):
                for c1 in range(chain_runner):
                    inter, intra = [], []
                    for c2 in range(chain_runner):
                        pairs = list(itertools.product(range(c1 * self.chain_atoms, (c1 + 1) * self.chain_atoms),
                                                       range(c2 * self.chain_atoms, (c2 + 1) * self.chain_atoms)))
                        d = md.compute_distances(traj[-frame], pairs)
                        d = md.geometry.squareform(d, flat_pairs) * 10
                        d = d[0]
                        if contacts:
                            d = mda.contacts.contact_matrix(d, radius=6)
                        if c1 != c2:
                            inter.append(d)
                        elif c1 == c2:
                            intra.append(d)
                    inter = np.array(inter)
                    intra = np.array(intra)
                    inter_chain[c1, :, :] += inter.sum(axis=0)
                    intra_chain[c1, :, :] += intra[0, :, :]
            inter_chain /= final_frame - initial_frame
            inter_chain = inter_chain.sum(axis=0)
            intra_chain = intra_chain.mean(axis=0)
            intra_chain = intra_chain / (final_frame - initial_frame)
            return inter_chain, intra_chain

        if temperature is not None:
            structures = [self.structures[temperature]]
        if self.chains == 1:
            raise SystemError("Demanded interchain distances but only a single chain is present!")

        structures = self.structures
        if temperature is not None:
            structures = [self.structures[temperature]]
        chain_runner = self.chains

        flat_pairs = list(itertools.product(range(self.chain_atoms), range(self.chain_atoms)))

        nthreads = 4
        p = mp.Pool()
        obj = self
        r_objs = []
        final_results = []
        for i in range(len(structures)):
            frange = np.linspace(0, structures[i].n_frames, nthreads + 1, dtype='int')
            frange = np.array([frange, np.roll(frange, -1)]).T
            frange = frange[:-1, :]
            for ra in frange:
                # r_objs.append(p.apply_async(parallel_calc, [ra[0], ra[1], i]))
                r_objs.append(p.apply_async(top_parallel_calc, [ra[0], ra[1], i, obj, flat_pairs]))
            print(" " * 40, end='\r')
            results = [r.get() for r in r_objs]
            p.close()
            p.join()
            results = np.array(results).sum(axis=0) / np.array(results).shape[0]
            final_results.append(results)
        final_results = np.array(final_results)
        return final_results[:, 0], final_results[:, 1]

    def inter_distance_map(self, contacts=False, temperature=None):
        """
        Calculate both intrachain and interchain distances using MDTraj for the multichain case
        :param contacts: bool, compute contacts instead of distances
        :param temperature: int, temperature index if we wish to compute a single T (if temper)
        :return: ndarray[T, n_atoms, n_atoms], distances per temperature (saving per frame is too costly)
        """
        if self.chains == 1:
            raise SystemError("Demanded interchain distances but only a single chain is present!")
        structures = self.structures.copy()
        inter_cmap, intra_cmap = [], []
        if temperature is not None:
            structures = [structures[temperature]]
        chain_runner = self.chains
        flat_pairs = list(itertools.product(range(self.chain_atoms), range(self.chain_atoms)))
        for traj in structures:
            inter_chain = np.zeros(shape=(self.chains, self.chain_atoms, self.chain_atoms), dtype='float64')
            intra_chain = np.zeros(shape=(self.chains, self.chain_atoms, self.chain_atoms), dtype='float64')
            for frame in range(traj.n_frames):
                print(f"{frame}/{traj.n_frames}", end='\r')
                for c1 in range(chain_runner):
                    # TODO : MAYBE ONLY UPPER OR LOWER TRIANGLE ?
                    inter, intra = [], []
                    for c2 in range(chain_runner):
                        pairs = list(itertools.product(range(c1 * self.chain_atoms, (c1 + 1) * self.chain_atoms),
                                                       range(c2 * self.chain_atoms, (c2 + 1) * self.chain_atoms)))
                        d = md.compute_distances(traj[-frame], pairs)
                        d = md.geometry.squareform(d, flat_pairs) * 10
                        d = d[0]
                        if contacts:
                            d = mda.contacts.contact_matrix(d, radius=6)
                        if c1 != c2:
                            inter.append(d)
                        elif c1 == c2:
                            intra.append(d)
                    inter = np.array(inter)
                    intra = np.array(intra)
                    inter_chain[c1, :, :] += inter.sum(axis=0)
                    intra_chain[c1, :, :] += intra[0, :, :]
            inter_chain /= traj.n_frames
            inter_chain = inter_chain.sum(axis=0)
            inter_cmap.append(inter_chain)
            intra_chain = intra_chain.mean(axis=0)
            intra_cmap.append(intra_chain / traj.n_frames)
        return np.array(inter_cmap), np.array(intra_cmap)

    def ij_from_contacts(self, contacts=None):
        """
        Calculate average distance between 2 residues separated by i-j residues from the distance maps generated
        by intra_distance_map or inter_distance_map functions
        :param contacts: bool, find contacts instead of distances
        :return: list, list, list ; abs(i-j) for plotting purposes, d(i-j), and error
        """
        if contacts is None:
            contacts = self.intra_distance_map()
        d_ijs, ijs = [], np.arange(0, contacts.shape[2])

        for d in range(contacts.shape[2]):
            diag = np.diagonal(contacts, offset=int(d), axis1=2, axis2=3)
            d_ijs.append(diag.mean(axis=2))
        d_ijs = np.array(d_ijs)
        # TODO : THERE MIGHT BE AN ERROR HERE...
        err = self.block_error(np.reshape(d_ijs, newshape=(d_ijs.shape[1], d_ijs.shape[2], d_ijs.shape[0])))
        d_ijs = d_ijs.mean(axis=2)
        d_ijs = d_ijs.T
        return ijs, d_ijs, err

    def flory_scaling_fit(self, r0=5.5, ijs=None):
        """
        Perform a flory scaling fit (Rij = r0*(i-j)**0.5) for a given r0. If r0 is None, then also fit for r0
        :param r0: Kuhn length's used for the fit. Typically 5.5 amstrongs for IDPs
        :param ijs: [ijs, dijs, err] in case we want to fit a particular data set
        :return: ndarray[T], ndarray[T], ndarray[T], flory exponent, kuhn lengths and fit errors
        """
        if ijs is None:
            tot_ijs, tot_means, err = self.ij_from_contacts()
        else:
            tot_ijs, tot_means, err = ijs[:,0,:], ijs[:,1,:], ijs[:,2,:]

        florys, r0s, covs = [], [], []
        for T in range(ijs.shape[0]):
            ij = tot_ijs[T]
            mean = tot_means[T]
            if r0 is None:
                fit, fitv = curve_fit(full_flory_scaling, ij, mean, sigma=err[T])
                fit_flory = fit[0]
                fit_r0 = fit[1]
            else:
                fit, fitv = curve_fit(flory_scaling(r0), ij, mean, sigma=err[T])
                fit_flory = fit[0]
                fit_r0 = r0
            florys.append(fit_flory)
            r0s.append(fit_r0)
            covs.append(np.sqrt(np.diag(fitv)))
        # final_err = np.sqrt((err/(tot_means*np.log(tot_ijs)))**2 + (np.array(covs)[:, 0])**2)
        return np.array(florys), np.array(r0s), np.array(covs)[:, 0]

    def flory_scaling_formula(self, gamma=1.1615, b=5.5):

        def rg_f(x, rg):
            nu = x
            g = gamma
            N = len(self.sequence)
            prefactor = g * (g + 1) / (2 * (g + 2 * nu) * (g + 2 * nu + 1))
            prefactor = prefactor ** 0.5
            return rg - prefactor * b * N ** nu

        self.rg()
        rg = self.c_rg[0]
        print(rg)
        nus = []
        for T in range(len(self.temperatures)):
            ret = fsolve(rg_f, x0=0.5, args=rg[T])
            nus.append(ret[0])
        return nus

    def pair_energies(self, model='HPS'):
        def HPS_potential(r, eps, lambd, sigma):
            V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
            if r <= 2 ** (1 / 6) * sigma:
                V += (1 - lambd) * eps
            else:
                V *= lambd
            return V

        def KH_potential(r, eps, sigma):
            V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
            if r <= 2 ** (1 / 6) * sigma:
                V += 2 * eps
            else:
                V *= -1
            return V

        self._get_hps_params()
        df = pd.DataFrame(index=self.residue_dict.keys(), columns=self.residue_dict.keys())
        for aa1 in self.residue_dict:
            for aa2 in self.residue_dict:
                sigma = (self.residue_dict[aa1]["sigma"] + self.residue_dict[aa2]["sigma"]) / 2
                lambd = (self.residue_dict[aa1]["lambda"] + self.residue_dict[aa2]["lambda"]) / 2
                if model == 'KH':
                    ks = pd.read_csv('/home/adria/scripts/data/hps/kh.dat', sep=" ", index_col=0)
                    eps_ij = ks[aa1][aa2]
                    kh_alpha = 0.228
                    kh_eps0 = -1
                    epss = eps_ij / kh_alpha + kh_eps0
                    if epss <= kh_eps0:
                        lambd = 1
                    else:
                        lambd = -1
                    eps_ij = abs(eps_ij)
                    df[aa1][aa2] = HPS_potential(2 ** (1 / 6) * sigma, eps_ij, lambd, sigma)
                else:
                    df[aa1][aa2] = HPS_potential(2 ** (1 / 6) * sigma, 0.2, lambd, sigma)
        return df

    def pair_potential(self, aminoacid, model='HPS', energies=False):
        """
        Utility for getting the HPS potential for the given current parameters for plotting
        :return:
        """

        def HPS_potential(r, eps, lambd, sigma):
            V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
            close_cond = np.where(r <= 2 ** (1 / 6) * sigma)
            far_cond = np.where(r > 2 ** (1 / 6) * sigma)
            V[close_cond] = V[close_cond] + (1 - lambd) * eps
            V[far_cond] = lambd * V[far_cond]
            return V

        def KH_potential(r, eps, sigma):
            V = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
            close_cond = np.where(r <= 2 ** (1 / 6) * sigma)
            far_cond = np.where(r > 2 ** (1 / 6) * sigma)
            V[close_cond] = V[close_cond] + 2 * eps
            V[far_cond] = -V[far_cond]
            return V

        r = np.linspace(5, 10, 200)
        aa_pot = {}
        aa = aminoacid
        if model == 'HPS':
            lambd = self.residue_dict[aa]["lambda"]
            sigma = self.residue_dict[aa]["sigma"]
            return r, HPS_potential(np.array([2 ** (1 / 6) * sigma]), 0.2, lambd, sigma)
        if model == 'KH':
            ks = pd.read_csv('/home/adria/scripts/data/hps/kh.dat', sep=" ", index_col=0)
            eps_ij = ks[aa][aa]
            print(eps_ij)
            kh_alpha = 0.228
            kh_eps0 = -1
            # epss = eps_ij / kh_alpha + kh_eps0
            epss = eps_ij / kh_alpha + kh_eps0
            sigma = self.residue_dict[aa]["sigma"]
            if epss <= kh_eps0:
                lambd = 1
            else:
                lambd = -1
            print("KH_KH")
            # return r, HPS_potential(r, eps_ij, lambd, sigma)
            return r, KH_potential(r, eps_ij, sigma)
        return r, aa_pot

    # TODO : UNTESTED
    def distance_map_by_residue(self, contacts):
        """
        Unstable. Translate the contact/distance map from intra/inter_chain_distance_map to a contact map by aminoacid
        :param contacts: ???
        :return: ???
        """
        dict_res_translator = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10,
                               'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
        saver = np.zeros(shape=(20, 20))
        dicti = {}
        for k in dict_res_translator.keys():
            for l in dict_res_translator.keys():
                dicti[k, l] = []
        for i in range(contacts.shape[0]):
            for j in range(contacts.shape[0]):
                iaa = self.sequence[i]
                jaa = self.sequence[j]
                dicti[iaa, jaa].append(contacts[i][j])

        for key in dicti.keys():
            saver[dict_res_translator[key[0]]][dict_res_translator[key[1]]] = np.array(dicti[key]).mean()

        return saver, dict_res_translator

    def rw_rg(self, monomer_l=5.5):
        rg = monomer_l * (self.chain_atoms / 6) ** 0.5
        return rg

    def dig_rg(self, monomer_l=5.5, gamma=1.1615):
        nu = 0.5
        b = monomer_l
        g = gamma
        N = len(self.sequence)
        prefactor = g * (g + 1) / (2 * (g + 2 * nu) * (g + 2 * nu + 1))
        prefactor = prefactor ** 0.5
        return prefactor * b * N ** nu

    def rg(self, use='md', full=False, save=None):
        """
        Calculate rg from the LAMMPS trajectories simulations
        :param use: string, either lmp or md. If lmp get rg's from the log.lammps.T, if md from MDTraj
        :param full: whether we want rg for every frame or its average
        :param save: text file where output will be saved
        :return: ndarray[T,frames], rg's
        """
        if self.c_rg is not None:
            return self.c_rg
        rgs, err = [], []
        if use == 'lmp':
            if self.chains != 1:
                raise SystemError("LMP method not reliable for Multichain systems")
            data = self.get_lmp_data()
            for run in range(data.shape[0]):
                # TODO INCORPORATE EQUILIBRATION EVERYWHERE
                rg_frame = data[run, :, 4]
                rgs.append(rg_frame)
        if use == 'md':
            for traj in self.structures:
                if self.chains != 1:
                    rg = 0
                    # TODO : PARALLELIZE THIS (split frames into block and do it!)
                    for chain in range(self.chains):
                        tr = traj[:]
                        atom_slice = np.linspace(chain * self.chain_atoms, (chain + 1) * self.chain_atoms,
                                                 self.chain_atoms, endpoint=False, dtype=int)
                        tr = tr.atom_slice(atom_slice)
                        rg += md.compute_rg(tr)
                    rg /= self.chains
                else:
                    rg = md.compute_rg(traj)
                rgs.append(rg)
        for i in range(len(rgs)):
            err.append(self.block_error(observable=np.array([rgs[i]]))[0])
            if full is False:
                rgs[i] = rgs[i].mean()
        self.c_rg = [np.array(rgs), np.array(err)]
        if self.low_mem:
            self._clean_memory()
            rgs = []
            err = []
        if save is not None and full is False:
            base_dir = '/home/adria/FINAL_DATA'
            pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(base_dir, save)
            np.savetxt(path, np.array([np.array(rgs), np.array(err)]))
        return np.array(rgs), np.array(err)

    def phospho_rew(self, T, savefile, iters=10000, n_ps=7, temp_dir='/home/adria/P-rw', debye=0.1, total_frames=100000,
                    model='HPS-T', scale=0.75):
        its = []
        pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
        rgi = self.rg(full=True)[0][T]

        shutil.copyfile(os.path.join(self.md_dir, f'topo.pdb'),
                        os.path.join(temp_dir, f'topo.pdb'))

        if self.last_frames is None:
            rgi = rgi[-total_frames:]

        def gen_ps():
            ps = list(np.random.choice(range(len(self.sequence)), n_ps, replace=False))
            if ps not in its:
                its.append(list(ps))
                return ps
            else:
                gen_ps()

        for it in range(iters):
            ps = gen_ps()
            og_time = int(np.max(self.data[T, :, 0]))
            og_save = int(self.data[T, 1, 0])

            seq_to_l = np.array(list(self.sequence))
            seq_to_l[ps] = 'D'
            sequence = ''.join(seq_to_l)
            # sequence = self.sequence

            shutil.copyfile(os.path.join(self.md_dir, f'atom_traj_{T}.lammpstrj'),
                            os.path.join(temp_dir, f'atom_traj_P_rw.lammpstrj'))

            every = self.every_frames if self.every_frames is not None else 1
            last = self.last_frames if self.last_frames is not None else total_frames

            Ei = self.data[T, :, 1]
            Ei = Ei[self.equil_frames::every]
            Ei = Ei[-last:]

            rerun = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein='CPEB4',
                                      debye=debye, temperatures=[self.temperatures[T]])
            rerun.rerun_dump = f'atom_traj_P_rw.lammpstrj'
            rerun.sequence = sequence
            rerun.debye = 0.1
            rerun.temperature = self.temperatures[T]
            rerun.rerun_skip = every
            rerun.model = model
            # in timesteps
            rerun.rerun_start = og_time + og_save - Ei.shape[0] * og_save * every
            # in timesteps
            rerun.rerun_stop = og_time
            rerun.hps_scale = scale

            rerun.write_hps_files(rerun=True, data=True, qsub=False, slurm=False, readme=False, rst=False, pdb=False,
                                  silent=True)
            rerun.run(file=os.path.join(temp_dir, 'lmp0.lmp'), n_cores=1)

            rerun_analysis = lmp.LMP(md_dir=temp_dir, every=1)

            Ef = rerun_analysis.data[0, :, 1]
            weights = rerun_analysis.weights(Ei=Ei, Ef=Ef, T=self.temperatures[T])
            weights = weights / np.sum(weights)
            n_eff = self.n_eff_calc(weights)
            reweight_rg = np.dot(rgi, weights.T)
            with open(savefile, 'a+') as fout:
                fout.write(f"{reweight_rg.mean():.3f} {n_eff:.3f} {rgi.mean():.3f} {ps}\n")

    # TODO : TDP USEFUL ONLY
    def topo_minimize(self, T, new_seq, temp_dir='/home/adria/TDP'):
        """
        TDP USABLE ONLY. Might be universal later on. Basically perform a rerun on topology changes
        :param T:
        :param new_seq:
        :param temp_dir:
        :return:
        """
        lseq = list(self.sequence)

        og_time = int(np.max(self.data[T, :, 0]))
        og_save = int(self.data[T, 1, 0])

        lnewseq = list(new_seq)
        lseq[289:373] = lnewseq[:]
        sequence = ''.join(lseq)
        rerun = lmpsetup.LMPSetup(md_dir=temp_dir, protein="TDP43", temper=False, equil_start=False)
        rerun.rerun_dump = f'atom_traj_tdp.lammpstrj'
        rerun.sequence = sequence

        rerun.rerun_skip = int(self.every_frames)
        rerun.rerun_start = int(og_time + og_save - self.every_frames * self.last_frames * og_save)
        rerun.rerun_stop = og_time

        # rerun.box_size = 2500
        rerun.temperatures = [self.get_temperatures()[T]]
        rerun.write_hps_files(rerun=True, data=True, qsub=False, slurm=False, readme=False, rst=False, pdb=False,
                              silent=True)
        rerun.run(file=os.path.join(temp_dir, 'lmp0.lmp'), n_cores=4)

        os.system(self.lmp2pdb + ' ' + os.path.join(temp_dir, 'data'))
        r_analysis = Analysis(md_dir=temp_dir)
        rrg = r_analysis.rg()[0]
        lmprg = r_analysis.data[0, :, -1].mean()
        og_data = self.data[T, self.equil_frames::self.every_frames, :]
        og_data = og_data[-self.last_frames:, :]
        weights = self.weights(Ei=og_data[:, 1], Ef=r_analysis.data[0, :, 1], T=self.get_temperatures()[T])
        weights = weights / np.sum(weights)
        n_eff = self.n_eff(weights)
        rew_rg = np.dot(weights, self.rg()[T, :])
        return rrg.mean(), lmprg, rew_rg, n_eff

    def maximize_charge(self, above_obj, below_obj, T, l0, eps0, method=None,
                        temp_dir='/home/adria/OPT', total_frames=100000,
                        savefile='/home/adria/results.txt', model='HPS-T',
                        weight_cost_mean=1):
        """
        Optimize the radius of gyration difference between 2 independent LAMMPS run. The cost function is essentially
        c = -(RgA - RgB)**2 + weight_cost_mean*((RgA+RgB)/2 - <Rg>)**2
        :param a_dir: string, path leading to the run with RgA > RgB
        :param b_dir: string, path leading to the run with RgB > RgA
        :param T: int, temperature index where we optimize
        :param method: string, "sto" or "opt" : if "opt" use scipy, if "sto" use custom ""random walk"" algorithm
        :param temp_dir: string, path leading to a temporal directory where the rerun will be performed
        :param I0: float, initial value for optimization over ionic strength in M. If None, don't optimize around it and assume I = 100 mM
        :param l0: float, initial value for optimization over HPS scale. If None, don't optimize around it and assume HPS Scale = 1
        :param eps0: float, initial value for optimization over medium permittivity. If None, don't optimize around it and assume a value of 80
        :param savefile: string, path leading to a file where the result of the optimization will be saved
        :param weight_cost_mean: float, weight given to the "distance from mean" term of the cost
        :return: float, list, list: best cost, best values that give cost, neff of the best fit
        """
        print("RUNNING REWEIGHTING")
        pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)
        x0 = np.array([l0, eps0])
        above_tc, below_tc = above_obj, below_obj
        rgiA, rgiB = above_tc.c_rg[0][T].mean(), below_tc.c_rg[0][T].mean()

        def rg_rerun(T, eps, ls, prot):
            data_dir = prot.md_dir
            og_time = int(np.max(prot.data[T, :, 0]))
            og_save = int(prot.data[T, 1, 0])

            every = prot.every_frames if prot.every_frames is not None else 1
            last = prot.last_frames if prot.last_frames is not None else total_frames
            Ei = prot.data[T, :, 1]
            Ei = Ei[prot.equil_frames::every]
            Ei = Ei[-last:]
            rgi = prot.c_rg[0][T]
            if prot.last_frames is None:
                rgi = rgi[-total_frames:]
            protein = prot.protein

            shutil.copyfile(os.path.join(data_dir, 'data.data'), os.path.join(temp_dir, f'data.data'))
            if not os.path.exists(os.path.join(temp_dir, f'atom_traj_{protein}_{T}.lammpstrj')):
                shutil.copyfile(os.path.join(data_dir, f'atom_traj_{T}.lammpstrj'),
                                os.path.join(temp_dir, f'atom_traj_{protein}_{T}.lammpstrj'))

            rerun = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein=protein)
            rerun.water_perm = eps
            rerun.hps_scale = ls
            rerun.save = og_save
            rerun.model = model
            rerun.debye_wv = 0.1
            rerun.rerun_dump = f'atom_traj_{protein}_{T}.lammpstrj'
            # in frames
            rerun.rerun_skip = every
            # in timesteps
            rerun.rerun_start = og_time + og_save - Ei.shape[0] * og_save * every
            # in timesteps
            rerun.rerun_stop = og_time
            rerun.temperatures = [prot.temperatures[T]]
            rerun.box_size = {"x": 5000, "y": 5000, "z": 5000}
            rerun.write_hps_files(rerun=True, data=False, qsub=False, rst=False, silent=True)
            rerun.run(file=os.path.join(temp_dir, 'lmp0.lmp'), n_cores=1)
            rerun_analysis = lmp.LMP(md_dir=temp_dir)
            Ef = rerun_analysis.data[0, :, 1]
            weights = rerun_analysis.weights(Ei=Ei, Ef=Ef, T=prot.temperatures[T])
            weights = weights / np.sum(weights)
            n_eff = self.n_eff_calc(weights)
            reweight_rg = np.dot(rgi, weights.T)
            return reweight_rg, n_eff

        def calc_cost(x, T):
            ls = x[0]
            eps = x[1]
            rw_above_rg, neff_a = rg_rerun(T=T, eps=eps, ls=ls, prot=above_tc)
            rw_below_rg, neff_b = rg_rerun(T=T, eps=eps, ls=ls, prot=below_tc)
            global rw_A
            global rw_B
            rw_A, rw_B = rw_above_rg, rw_below_rg
            # c = - (rw_above_rg - rw_below_rg) ** 2 + weight_cost_mean * ((rw_below_rg + rw_above_rg) / 2 - mean_rg) ** 2
            c = - (rw_above_rg - rw_below_rg) ** 2
            print("=" * 80)
            print(f"ls={ls}, eps={eps}, RgA_i={rgiA}, RgB_i={rgiB}")
            # print(f"Rw RgA, {rw_A:.3f}, Rw RgB {rw_B:.3f}, diff {rw_A - rw_B:.2f}, dist to mean {(rw_below_rg + rw_above_rg) / 2 - mean_rg:.2f}, cost {c:.2f}")
            print(f"Rw RgA, {rw_A:.3f}, Rw RgB {rw_B:.3f}, diff {rw_A - rw_B:.2f}, cost {c:.2f}")
            print(f"Neff-A, {neff_a} Neff-B, {neff_b}")
            print("=" * 80)
            if method == 'opt':
                return c
            else:
                return c, rw_A - rw_B, neff_a, neff_b, rw_A, rw_B

        if method is None:
            with open(savefile, 'a+') as writer:
                cost, diff, neff_a, neff_b, rw_A, rw_B = calc_cost([l0, eps0], T)
                return l0, eps0, diff, neff_a, neff_b, rw_A, rw_B
        elif method == 'opt':
            cons_all = ({'type': 'ineq', 'fun': lambda x: 0.95 - x[0]},
                        {'type': 'ineq', 'fun': lambda x: x[0] - 0.85},
                        {'type': 'ineq', 'fun': lambda x: 80 - x[1]},
                        {'type': 'ineq', 'fun': lambda x: x[1] - 60})
            m = scipy.optimize.minimize(fun=calc_cost, x0=np.array(x0), method="COBYLA", constraints=cons_all,
                                        args=(T,))

            print("=" * 80)
            print(
                f"For T={T} and initial parameters I = {I0}, eps = {eps0} with RgA = {above_tc.rg(use='md').mean(axis=1)[T]:.2f}, RgB = {below_tc.rg(use='md').mean(axis=1)[T]:.2f}")
            print(f"Minimization resulted in {m.x[0]}  {m.x[1]} and reweighted Rg's A = {rw_A:.2f} and B = {rw_B:.2f}")
            print("=" * 80)
            result = "=============================================\n"
            result += f'T={T}, I0={I0}, eps={eps0}, rgiAbove={rgiA}, rgiBelow={rgiB} \n'
            result += f'If={m.x[0]}, eps={m.x[1]}, rgAboveF={rw_A}, rgBelowF={rw_B}\n'
            result += "=============================================\n"
            with open(f'minimization_{T}.txt', 'w+') as min:
                min.write(result)

            return m
        elif method == 'sto':
            best, best_params, best_neffs = 0, [], [-1, -1]
            iterations = 50000
            for it in range(iterations):
                current = []
                for x in x0:
                    current.append(x * np.random.normal(1, 0.02, 1)[0])
                cst, neff_a, neff_b = calc_cost(current, args=(T,))
                if cst < best:
                    best = cst
                    best_params = current[:]
                    best_neffs = [neff_a, neff_b]
                    print("Saving at /home/adria/rew_test.txt", f'data/{savefile}')
                    with open(savefile, 'a+') as fout:
                        str_curr = ", ".join(map(str, current))
                        fout.write(
                            f"{it} {cst} {rw_A} {rw_B} {str_curr} {neff_a} {neff_b}\n")
                    x0 = current
            return best, best_params, best_neffs

    def get_correlation_tau(self):
        """
        Find the correlation decay time from the rg autocorrelaition function
        :return: float, maximum correlation decay time among all temperatures
        """
        self.structures = self.get_structures()
        if self.rerun:
            return 2
        save_period = self.data[0, 1, 0]
        corr_tau_T = []
        rg = self.rg(full=True)
        acs = []
        for T in range(len(self.temperatures)):
            ac = statsmodels.tsa.stattools.acf(rg[0][T], nlags=200, fft=False)
            acs.append(ac)
            mi = ac[np.abs(ac) < 0.2][0]
            # corr_tau_T.append(int(save_period * np.where(ac == mi)[0][0]))
            corr_tau_T.append(np.where(ac == mi)[0][0])
        return np.array(acs), np.max(corr_tau_T)

    def n_eff_calc(self, weights, kish=False):
        """
        Return the effective sample size from given probabilites (weights)
        :param weights: ndarray, weights for a given run
        :param kish: ??
        :return: float, effective sample size
        """
        if kish:
            n_eff = 1. / np.sum(weights ** 2) / weights.size
        else:
            n_eff = np.exp(-np.sum(weights * np.log(weights * weights.size)))
        return n_eff

    def weights(self, Ei, Ef, T):
        """
        Get new probabilites for a given structure of a new ensemble, given its new energy and the original energy
        :param Ei: ndarray[frames], initial energy
        :param Ef: ndarray[frames], final energy
        :param T: flaot, current temperature in Kelvin
        :return: ndarray[frames], weights of each structure
        """
        """ ENERGIES IN LMP IS IN KCAL/MOL (1 KCAL = 4184J) """
        kb = cnt.Boltzmann * cnt.Avogadro / 4184
        w = np.exp(-(Ef - Ei) / (kb * T))
        return w

    def B2(self, T):
        g_x = self.density_profile(T=T)
        bins, data = np.histogram(g_x, bins=25)

        def f(x):
            idx_sup = np.where(bins == np.min(bins[bins > x]))[0][0]
            idx_inf = np.where(bins == np.max(bins[bins < x]))[0][0]
            slope = (data[idx_sup] - data[idx_inf]) / (
                    bins[idx_sup] - bins[idx_inf])
            intersect = data[idx_sup] - slope * bins[idx_sup]
            g = slope * x + intersect
            return (1 - g) * x * x

        B2 = integrate.quad(f, 0, np.inf)
        return B2

    def block_error(self, observable, n_blocks=5):
        """
        Find statistical error through block averaging. Observable needs to be in shape (T, frames) (assuming Temper)
        :param observable: ndarray[T, frames], observable data
        :param block_length: int, block size to obtain the errors
        :return:
        """
        mean = observable.mean(axis=1)
        errors = []
        for T in range(0, observable.shape[0]):
            err = 0
            blocks = np.linspace(0, observable.shape[1], n_blocks + 1, dtype='int')
            for r in range(n_blocks):
                rng = slice(blocks[r], blocks[r + 1])
                err += (observable[T, rng].mean() - mean[T]) ** 2
            errors.append(np.sqrt(err / (n_blocks * (n_blocks - 1))))
        return np.array(errors)

    def find_Tc(self, florys=None):
        """
        Get critical temperature from an array of flory exponents, using interpolation
        :param florys: iterable[T], float of florys
        :return: float, critical temperature
        """
        temps = self.get_temperatures()
        if florys is None:
            florys = self.flory_scaling_fit()[0]
        lim_inf = np.max(florys[florys < 0.5])
        lim_sup = np.min(florys[florys > 0.5])
        # TODO : FLOAT CONVERSION IN SELF.GET_TEMPERATURES()?
        T_inf = float(temps[np.where(florys == lim_inf)[0][0]])
        T_sup = float(temps[np.where(florys == lim_sup)[0][0]])
        slope = (lim_sup - lim_inf) / (T_sup - T_inf)
        intersect = lim_sup - slope * T_sup
        Tc = (0.5 - intersect) / slope
        return Tc

    def chain_coms(self):
        """
        Get the center of masses for every chain
        :return: ndarray[T, nchains, frames, 3], center of mass
        """
        sequence = self.get_seq_from_hps()
        m_i = []
        for atom in range(self.chain_atoms):
            aa = sequence[atom]
            m_i.append(self.residue_dict[aa]["mass"])
        m_i = np.array(m_i)
        M = np.sum(m_i)
        coms = np.zeros(shape=(len(self.temperatures), self.chains, self.structures[0].n_frames, 3))
        for T, traj in enumerate(self.structures):
            for c in range(self.chains):
                atidx = np.linspace(c * self.chain_atoms, (c + 1) * self.chain_atoms, self.chain_atoms, endpoint=False,
                                    dtype=int)
                tr = traj.atom_slice(atidx)
                xyz = np.reshape(tr.xyz, (tr.xyz.shape[0], tr.xyz.shape[2], tr.xyz.shape[1])) * 10
                s = np.dot(xyz, m_i)
                com = s / M
                coms[T, c, :] = com
        return coms

    def clusters(self, T=0, cutoff=50):
        """
        Get clusters of chains for each frame. We define a cluster all those chains that are less than "cutoff" amstrongs apart
        from each other
        :param T: int, temperature index to get the clusters
        :param cutoff: float, distance limit where we consider a cluster
        :return: list[frames, clusters], list[frames,3], non regular list containing all clusters per frame, center of masses per frame (essentially chain_coms for a given T)
        """
        if T is None:
            temperatures = range(len(self.structures))
        else:
            temperatures = [T]
        f_xyz = self.chain_coms()

        def run(chain, filling):
            if chain not in taken:
                clusters.append([chain])
                filling = chain
            for i, d in enumerate(d_matrix[chain]):
                if cutoff > d > 0 and i not in taken:
                    taken.append(i)
                    if i != filling:
                        clusters[-1].append(i)
                    run(i, filling)

        nchains = self.chains
        f_clust, T_clust = [], []
        print(temperatures)
        for T in temperatures:
            print(f"Doing temperature {T}", self.structures[T])
            f_clust = []
            for f in range(self.structures[T].n_frames):
                print(f'{f}/{self.structures[T].n_frames}', end='\r')
                xyz = f_xyz[T, :, f, :]
                d_matrix = np.zeros(shape=(nchains, nchains))
                for i, d1 in enumerate(xyz):
                    for j, d2 in enumerate(xyz):
                        d_matrix[i][j] = np.linalg.norm(d1 - d2)

                taken, clusters = [], []

                for chain in range(d_matrix.shape[0]):
                    run(chain, 0)
                f_clust.append(clusters)
            T_clust.append(f_clust)
        return T_clust, f_xyz

    def old_density_profile(self, T=None):
        """
        Density profile for a given frame. The idea is that we first center the system to the largest cluster, and then
        count the number of chains to left and right for a given axis. If we get a flat plot, we are far from PS.
        :param T: int, temperature index for the calculation
        :return: ndarray[T, frames, chains, 3], center of mass of each chain centered around the biggest cluster
        """
        random_displacement = np.random.randint(200, size=self.structures[0].n_frames)

        random_displacement = np.repeat(random_displacement[:, np.newaxis], self.structures[0].n_atoms, axis=1)
        self.structures[0].xyz[:, :, 2] = self.structures[0].xyz[:, :, 2] - random_displacement
        if T is None:
            T = range(len(self.get_temperatures()))
        else:
            T = [T]
        droplet_temp = []
        for temp in T:
            droplet_frame = []
            frame_clusters, coms = self.clusters(T=temp)
            for f, clusters in enumerate(frame_clusters[0]):
                l_old, biggest_cluster = 0, None
                for cluster in clusters:
                    l_new = len(cluster)
                    if l_new > l_old:
                        biggest_cluster = cluster
                        l_old = l_new

                r = 0
                for chain in biggest_cluster:
                    r += coms[temp, chain, f, :]
                r /= len(biggest_cluster)
                droplet_frame.append(r)
            droplet_temp.append(droplet_frame)

        def bin_coms(bin_size, data):
            # bins_f = []
            xbins = np.arange(-self.box["z"], self.box["z"], bin_size)
            bins_f = []
            for frame in range(data.shape[0]):
                bins = []
                tf = data[frame, :]
                for z in xbins:
                    ll = len(tf[np.where((tf > z) & (tf < (z + bin_size)))])
                    bins.append(ll / self.structures[0].n_atoms)
                bins_f.append(bins)
            return np.array(bins_f), np.array(xbins)

        xa, slab_bins = bin_coms(bin_size=10, data=self.structures[0].xyz[:, :, 2])

        rz = np.array(droplet_temp)[:, :, 2][0]
        rrp = np.repeat(rz[:, np.newaxis], self.structures[0].n_atoms, axis=1)

        xshift, slab_bins_shift = bin_coms(bin_size=10, data=self.structures[0].xyz[:, :, 2] - rrp)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 6))
        plt.imshow(xa, extent=[-15, 15, 0, 24])
        plt.figure(figsize=(16, 6))
        plt.imshow(xshift, extent=[-15, 15, 0, 24])

        return np.array(droplet_temp)

    def density_profile_by_pos(self, T=0):

        test = self.structures[T].xyz[:, :, 2]

        def bin_coms(bin_size):
            xbins = np.arange(-self.box["z"] / 2, self.box["z"] / 2, bin_size)
            bins_f = []
            for frame in range(test.shape[0]):
                zz = np.zeros(shape=(self.chain_atoms, self.chains))
                for chain in range(self.chains):
                    for atom in range(self.chain_atoms):
                        zz[atom, chain] = test[frame, atom * (chain + 1)]

                zz = np.array(zz)
                bins_at = []
                for at in range(self.chain_atoms):
                    bins = []
                    zz2 = zz[at, :]
                    for z in xbins:
                        ll = len(zz2[np.where((zz2 > z) & (zz2 < (z + bin_size)))])
                        bins.append(ll / bin_size)
                    bins_at.append(np.array(bins))
                bins_f.append(bins_at)
            return np.array(bins_f), np.array(xbins)

        xa, slab_bins = bin_coms(bin_size=16)
        return xa, slab_bins

    def density_profile_by_aa(self):
        test = self.structures[T].xyz[:, :, 2]

        types = ["hydrophobic", "charged", "polar", "aromatic", "other"]
        seq_to_type = []
        for aa in self.sequence:
            seq_to_type.append(self.residue_dict[aa]["type"])
        seq_to_type = seq_to_type * self.chains

        def bin_coms(bin_size):
            xbins = np.arange(-self.box["z"] / 2, self.box["z"] / 2, bin_size)
            bins_f = []
            for frame in range(test.shape[0]):
                tf = test[frame, :]
                bins_ty = []
                bins = []
                for type in types:
                    zz = tf[np.where(seq_to_type == type)]
                    for z in xbins:
                        ll = len(zz[np.where((zz > z) & (zz < (z + bin_size)))])
                        bins.append(ll / bin_size)
                    bins_ty.append(np.array(bins))
                bins_f.append(bins_ty)
            return np.array(bins_f), np.array(xbins)

        xa, slab_bins = bin_coms(bin_size=16)
        return xa, slab_bins

    # def density_profile_by_type(self, z, rho_z_cent, T=0, ):
    def density_profile_by_type(self, shifts=None, T=0, ):
        if shifts is None:
            _, _, _, _, _, shifts = self.density_profile(T=T)
        test = self.structures[T].xyz[:, :, 2] - shifts[..., np.newaxis]

        types = ["hydrophobic", "charged", "polar", "aromatic", "other"]
        seq_to_type = []
        for aa in self.sequence:
            seq_to_type.append(self.residue_dict[aa]["type"])
        seq_to_type = seq_to_type * self.chains
        seq_to_type = np.array(seq_to_type)

        def bin_coms(bin_size):
            xbins = np.arange(-self.box["z"] / 2, self.box["z"] / 2, bin_size)
            bins_ty = []
            for type in types:
                bins_f = []
                for frame in range(test.shape[0]):
                    tf = test[frame, :]
                    bins = []
                    zz = tf[np.where(seq_to_type == type)]
                    for z in xbins:
                        ll = len(zz[np.where((zz > z) & (zz < (z + bin_size)))])
                        bins.append(ll / bin_size)
                    bins_f.append(bins)
                bins_ty.append(np.array(bins_f))
            return np.array(bins_ty), np.array(xbins)

        xa, slab_bins = bin_coms(bin_size=16)
        return xa, slab_bins

    def density_profile(self, T=0, noise=False):

        if noise:
            random_displacement = np.random.randint(-self.box["z"] / 2 * 0.5, self.box["z"] / 2 * 0.5,
                                                    size=self.structures[0].n_frames)
            random_displacement = np.repeat(random_displacement[:, np.newaxis], self.structures[0].n_atoms, axis=1)
            test = self.structures[T].xyz[:, :, 2] - random_displacement
        else:
            test = self.structures[T].xyz[:, :, 2]

        def bin_coms(bin_size):
            xbins = np.arange(-self.box["z"] / 2, self.box["z"] / 2, bin_size)
            bins_f = []
            for frame in range(test.shape[0]):
                bins = []
                tf = test[frame, :]
                for z in xbins:
                    ll = len(tf[np.where((tf > z) & (tf < (z + bin_size)))])
                    # TODO : STRONG ASSUMPTION
                    bins.append(ll / bin_size)
                bins = np.array(bins)
                bins_f.append(bins)
            return np.array(bins_f), np.array(xbins)

        bin_size = 16
        xa, slab_bins = bin_coms(bin_size=bin_size)
        print(xa.shape)
        n_lags = math.ceil(len(slab_bins) / 2)
        caa = np.zeros(shape=(xa.shape[0], n_lags * 2))
        for lag in range(0, n_lags):
            xa_mod = xa[:, lag:]
            xa_lag = xa[:, :(len(slab_bins) - lag)]
            acf = np.sum((xa_mod - xa.mean()) * (xa_lag - xa.mean()), axis=1)
            # Caa is symmetric
            caa[:, lag + n_lags] = acf / ((len(slab_bins) - lag) * np.var(xa, axis=1))
            caa[:, -lag + n_lags] = acf / ((len(slab_bins) - lag) * np.var(xa, axis=1))
        caa = caa[:, (n_lags * 2 - len(slab_bins)):]

        pa = np.ones_like(caa)
        pa[caa < 0.5] = 0
        pa[:, 0] = 0

        xcc = np.zeros(shape=caa.shape)

        # TODO : REVIEW THESE TWO
        # WORKS
        for lag in range(0, n_lags):
            xa_mod = xa[:, lag:]
            pa_lag = np.flip(pa, axis=1)[:, :(2 * n_lags - lag - (n_lags * 2 - len(slab_bins)))]
            xcc[:, lag] = np.sum(xa_mod * pa_lag, axis=1)

        for lag in range(0, n_lags):
            xa_mod = np.flip(xa, axis=1)[:, lag:]
            pa_lag = pa[:, :(2 * n_lags - lag - (n_lags * 2 - len(slab_bins)))]
            xcc[:, -lag] = np.sum(xa_mod * pa_lag, axis=1)

        nmaxs = np.argmax(xcc, axis=1)
        for f in range(len(nmaxs)):
            if nmaxs[f] > n_lags:
                nmaxs[f] = nmaxs[f] - n_lags * 2 + (n_lags * 2 - len(slab_bins))

        x_shift = np.zeros(shape=xa.shape)
        for frame in range(xa.shape[0]):
            x_shift[frame, :] = np.roll(xa[frame, :], -nmaxs[frame], axis=0)

        z = slab_bins
        rho_z = x_shift
        shifts = nmaxs * bin_size
        return z, rho_z, xa, caa, pa, shifts

    def interface_position(self, z, rho_z, cutoff_c=0.9, cutoff_d=0.1):
        def tanh_fit(x, s, y0, x0):
            y = y0 - y0 * np.tanh((x + x0) * s)
            return y

        def tanh_solve(x, s, y0, x0, cutoff):
            return tanh_fit(x, s, y0, x0) - cutoff

        rho_z_plus = rho_z[np.where(z >= 0)]
        z_plus = z[np.where(z >= 0)]
        i_guess = [0.1, rho_z_plus.max() / 2, -100]

        try:
            popt_plus, pcov_plus = curve_fit(tanh_fit, z_plus, rho_z_plus, p0=i_guess)
        except:
            print("fit Failed")
            return [0, 0], [0, 0], 0, 0

        c_interface = fsolve(tanh_solve, x0=1, args=(*popt_plus, rho_z_plus.max() * cutoff_c))[0]
        d_interface = fsolve(tanh_solve, x0=1, args=(*popt_plus, rho_z_plus.max() * cutoff_d))[0]

        return [-c_interface, c_interface], [-d_interface, d_interface], z, tanh_fit(z, *popt_plus)

    def phase_diagram(self, cutoff_c=0.9, cutoff_d=0.1, full=False, intf_c_cutoff=None, intf_d_cutoff=None,
                      start_T=None, end_T=None):
        dilute_densities = []
        condensed_densities = []
        for T in range(len(self.temperatures[start_T:end_T])):
            z, rho_z, xa, caa, pa, shifts = self.density_profile(T=T, noise=False)
            # z_fit, rho_fit, tanh_fit, interface_pos = self.interface_position(rho_z=rho_z.mean(axis=0), slab_bins=z)
            if intf_c_cutoff is None and intf_d_cutoff is None:
                interface_c, interface_d, z, fit = self.interface_position(rho_z=rho_z.mean(axis=0), z=z,
                                                                           cutoff_c=cutoff_c, cutoff_d=cutoff_d)
            else:
                interface_c = [-intf_c_cutoff, intf_c_cutoff]
                interface_d = [-intf_d_cutoff, intf_d_cutoff]
            print("Interfaces at ", interface_c, interface_d)
            c = self.structures[T].xyz[:, :, 2] - shifts[..., np.newaxis]

            mass_helper = np.zeros_like(c)
            # mass_helper[np.where((-interface_pos < c) & (c < interface_pos))] = 1
            mass_helper[np.where((interface_c[0] < c) & (c < interface_c[1]))] = 1
            # md.append(mass_helper.sum(axis=1))
            mass_condensed = np.dot(mass_helper, self.masses)
            mass_helper = np.zeros_like(c)
            mass_helper[np.where(interface_d[0] > c)] = 1
            mass_helper[np.where(interface_d[1] < c)] = 1
            mass_dilute = np.dot(mass_helper, self.masses)
            volume_condensed = ((interface_c[1] - interface_c[0]) * self.box["x"] * self.box["y"])
            # volume_dilute = self.box["x"] * self.box["y"] * self.box["z"] - volume_condensed
            volume_dilute = self.box["x"] * self.box["y"] * self.box["z"] - (
                    (interface_d[1] - interface_d[0]) * self.box["x"] * self.box["y"])
            # dilute_densities.append(diluted*mass_dilute/volume_dilute)
            if full:
                dilute_densities.append((mass_dilute / volume_dilute))
                condensed_densities.append((mass_condensed / volume_condensed))
            else:
                dilute_densities.append((mass_dilute / volume_dilute).mean())
                condensed_densities.append((mass_condensed / volume_condensed).mean())
            # condensed_densities.append(condensed*mass_condensed/volume_condensed)
        # return np.array(md)
        dilute_densities = np.array(dilute_densities) * 10 ** 27 / cnt.Avogadro
        condensed_densities = np.array(condensed_densities) * 10 ** 27 / cnt.Avogadro

        return dilute_densities, condensed_densities

    def aa_composition(self):
        order = "GAVLMIFYWKRHDESTCNQP"
        dict = {}
        for aa in order:
            dict[aa] = self.sequence.count(aa)
        return dict

    # def find_Tc_from_diagram(self, rho_condensed, rho_solvated, temperatures, err_condensed, err_dilute):
    def find_Tc_from_diagram(self, rho_c, rho_d, temperatures):
        def scaling_coex_densities(x, A, Tc_sc):
            beta = 0.325
            # return A * (x-Tc_sc) ** beta
            return A * (x - Tc_sc) ** beta

        def rectilinear_diametres(x, A2, Tc_rd, rho_c):
            # ref : https://link.springer.com/article/10.1007/BF02847185
            return 2 * rho_c * (1 + A2 * (Tc_rd - x) / Tc_rd)
            # These fits perform good but parameters are bad if points in the left are also bad I guess
            # return 2*(rho_c + A2 * (Tc_rd - x))

        # try:
        #     popt, pcovt = curve_fit(scaling_coex_densities, temperatures, rho_c - rho_d, p0=[150, 270])
        popt, pcovt = curve_fit(scaling_coex_densities, temperatures, rho_c - rho_d, p0=[100, 250])

        popt2, pcovt2 = curve_fit(rectilinear_diametres, temperatures, (rho_c + rho_d),
                                  p0=[0.2, 0.2, 0.2])
        # p0=[0.2, popt[1], 0.2])
        print("PPT1", popt, "PPT2", popt2)
        # print(popt)
        # except:
        #     print("Optimization failed")
        #     return None, None

        # critical_T = [popt[1], popt2[1]]
        # critical_rho = popt2[2]
        critical_T = [popt[1], popt2[1]]
        critical_rho = popt2[2]
        ext_T = np.linspace(np.max(temperatures), critical_T[0], 2000)
        rho_drop_fit = (scaling_coex_densities(ext_T, *popt) + rectilinear_diametres(ext_T, *popt2)) / 2
        rho_solv_fit = (-scaling_coex_densities(ext_T, *popt) + rectilinear_diametres(ext_T, *popt2)) / 2
        fit = np.zeros(shape=(4000, 2))
        fit[:2000, 0] = rho_solv_fit
        fit[2000:, 0] = np.flip(rho_drop_fit)
        fit[:2000, 1] = ext_T
        fit[2000:, 1] = np.flip(ext_T)
        # fit = np.zeros(shape=(2, 4000))
        # fit[0, :2000] = rho_solv_fit
        # fit[0, 2000:] = np.flip(rho_drop_fit)
        # fit[1, :2000] = ext_T
        # fit[1, 2000:] = np.flip(ext_T)
        cr_point = [critical_rho, critical_T[0]]
        return fit[:, :2000], popt

    def _clean_memory(self):
        self.structures = None


def top_parallel_calc(initial_frame, final_frame, obj, i, flat_pairs):
    inter_chain = np.zeros(shape=(obj.chains, obj.chain_atoms, obj.chain_atoms), dtype='float64')
    intra_chain = np.zeros(shape=(obj.chains, obj.chain_atoms, obj.chain_atoms), dtype='float64')
    traj = obj.structures[i]
    traj = traj[:10]
    for frame in range(initial_frame, final_frame):
        for c1 in range(obj.chains):
            inter, intra = [], []
            for c2 in range(obj.chains):
                pairs = list(itertools.product(range(c1 * obj.chain_atoms, (c1 + 1) * obj.chain_atoms),
                                               range(c2 * obj.chain_atoms, (c2 + 1) * obj.chain_atoms)))
                d = md.compute_distances(traj[-frame], pairs)
                d = md.geometry.squareform(d, flat_pairs) * 10
                d = d[0]
                # if contacts:
                if True:
                    d = mda.contacts.contact_matrix(d, radius=6)
                if c1 != c2:
                    inter.append(d)
                elif c1 == c2:
                    intra.append(d)
            inter = np.array(inter)
            intra = np.array(intra)
            inter_chain[c1, :, :] += inter.sum(axis=0)
            intra_chain[c1, :, :] += intra[0, :, :]
    inter_chain /= final_frame - initial_frame
    inter_chain = inter_chain.sum(axis=0)
    intra_chain = intra_chain.mean(axis=0)
    intra_chain = intra_chain / (final_frame - initial_frame)
    return inter_chain, intra_chain


def full_flory_scaling(x, flory, r0):
    """
    Function used for flory fitting
    """
    return r0 * (x ** flory)


def flory_scaling(r0):
    """
    Function used for flory fitting with fixed r0
    """

    def flory(x, flory):
        return r0 * (x ** flory)

    return flory

# def interface_position_old(self, rho_z, slab_bins, cutoff=0.5):
#     def tanh_fit(x, s, y0, x0):
#         y = y0 - y0 * np.tanh((x + x0) * s)
#         return y
#
#     minn_plus = rho_z[np.where(slab_bins >= 0)]
#     minn_minus = rho_z[np.where(slab_bins <= 0)]
#     # TODO ?
#     # x_plus = np.linspace(0, slab_bins.max(), minn_plus.shape[0])
#     print(minn_plus.shape[0])
#     # x_plus = np.linspace(0, 650, minn_plus.shape[0])
#     x_plus = np.arange(0, minn_plus.shape[0])
#     # x_minus = np.linspace(slab_bins.min(), 0, minn_minus.shape[0])
#     # x_minus = np.linspace(-10, 0, minn_minus.shape[0])
#     x_minus = np.linspace(-100, 0, minn_minus.shape[0])
#     try:
#         # popt_plus, pcov_plus = curve_fit(tanh_fit, x_plus, minn_plus)
#         popt_plus, pcov_plus = curve_fit(tanh_fit, x_plus, minn_plus)
#         print("HERE", popt_plus)
#         popt_minus, pcov_minus = curve_fit(tanh_fit, x_minus, minn_minus)
#     except:
#         print("> Interface fit failed, returning 0 (no interface) !!!")
#         return 0, 0
#
#     xvals_plus = np.linspace(0, slab_bins.max(), 10000)
#     fit_plus = tanh_fit(x_plus, *popt_plus)
#     z_fit_plus = slab_bins[slab_bins > 0]
#     yinterp = np.interp(xvals_plus, z_fit_plus, fit_plus)
#     max_tan = yinterp.max()
#     cutoff = max_tan * 0.9
#     idx_interface_plus = np.argmin(np.abs(yinterp - cutoff))
#     print(xvals_plus[idx_interface_plus])
#
#     xvals_minus = np.linspace(slab_bins.min(), 0, 10000)
#     fit_minus = tanh_fit(x_minus, *popt_minus)
#     z_fit_minus = slab_bins[slab_bins < 0]
#     yinterp = np.interp(xvals_minus, z_fit_minus, fit_minus)
#     max_tan = yinterp.max()
#     cutoff = max_tan * 0.9
#     idx_interface_minus = np.argmin(np.abs(yinterp - cutoff))
#
#     return z_fit_plus, minn_plus, fit_plus
#     # return z_fit_minus, minn_minus, fit_minus
#     # return xvals_plus[idx_interface_plus], xvals_minus[idx_interface_minus]
