import lmp
import itertools
import shutil
import mdtraj as md
import math
import scipy.constants as cnt
import numpy as np
import os
import scipy
import time
import definitions
import lmpsetup
import multiprocessing as mp
from scipy.optimize import curve_fit
import MDAnalysis.analysis as mda
from MDAnalysis.analysis import contacts as compute_contacts
from numba import jit


class Analysis(lmp.LMP):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.equilibration = 3e6

        self.ijs = None
        self.rgs = None
        if self.o_wd is not None:
            self.structures = self.get_structures()

    def intra_distance_map(self, use='md', contacts=False, temperature=None):
        structures = self.structures.copy()
        contact_maps = []
        if temperature is not None:
            structures = [structures[temperature]]
        for traj in structures:
            # TODO: TEST FIRST PART OF IF
            pairs = list(itertools.product(range(traj.top.n_atoms - 1), range(1, traj.top.n_atoms)))
            d = md.compute_distances(traj, pairs)
            d = md.geometry.squareform(d, pairs)
            contact_map = d * 10
            if contacts:
                contact_map = mda.contacts.contact_matrix(contact_map, radius=6)
            contact_maps.append(contact_map)
        return np.array(contact_maps)

    def inter_distance_map(self, use='md', contacts=False, temperature=None):
        if self.chains == 1:
            raise SystemError("Demanded interchain distances but only a single chain is present!")
        structures = self.structures.copy()
        inter_cmap, intra_cmap = [], []
        if temperature is not None:
            structures = [structures[temperature]]
        chain_runner, frame_runner = 50, 10
        flat_pairs = list(itertools.product(range(self.chain_atoms), range(self.chain_atoms)))
        for traj in structures:
            inter_chain = np.zeros(shape=(self.chains, self.chain_atoms, self.chain_atoms), dtype='float64')
            intra_chain = np.zeros(shape=(self.chains, self.chain_atoms, self.chain_atoms), dtype='float64')
            for frame in range(frame_runner):
                # if frame != 0:
                #     print("TIME SPENT THIS FRAME", time.time()-ti)
                # ti = time.time()
                print(frame)
                for c1 in range(chain_runner):
                    # TODO : MAYBE ONLY UPPER OR LOWER TRIANGLE ?
                    inter, intra = [], []
                    for c2 in range(chain_runner):
                        pairs = list(itertools.product(range(c1 * self.chain_atoms, (c1 + 1) * self.chain_atoms),
                                                       range(c2 * self.chain_atoms, (c2 + 1) * self.chain_atoms)))
                        d = md.compute_distances(traj[-frame], pairs)
                        d = md.geometry.squareform(d, flat_pairs)*10
                        d = d[0]
                        if contacts:
                            d = mda.contacts.contact_matrix(d, radius=6)
                        if c1 != c2:
                            inter.append(d)
                        elif c1 == c2:
                            intra.append(d)
                    inter = np.array(inter)
                    intra = np.array(intra)
                    # inter_chain[c1, :, :] += inter.mean(axis=0)
                    inter_chain[c1, :, :] += inter.sum(axis=0)
                    intra_chain[c1, :, :] += intra[0, :, :]
            inter_chain /= frame_runner
            # inter_chain = inter_chain.mean(axis=0)
            inter_chain = inter_chain.sum(axis=0)
            inter_cmap.append(inter_chain)
            intra_cmap.append(intra_cmap)
        return np.array(inter_cmap), np.array(intra_cmap)

    def ij_from_contacts(self, use='md', contacts=None):
        if contacts is None:
            contacts = self.intra_distance_map(use=use)
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

    def flory_scaling_fit(self, r0=None, use='md', ijs=None):
        if ijs is None:
            tot_ijs, tot_means, err = self.ij_from_contacts(use=use)
        else:
            tot_ijs, tot_means, err = ijs[0], ijs[1], ijs[2]
        florys, r0s, covs = [], [], []
        for T in range(len(self.get_temperatures())):
            ij = tot_ijs
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

    def distance_map_by_residue(self, contacts):
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

    def rg(self, use='md'):
        rgs = []
        if use == 'lmp':
            if self.chains != 1:
                raise SystemError("LMP method not reliable for Multichain systems")
            data = self.get_lmp_data()
            for run in range(data.shape[0]):
                # TODO INCORPORATE EQUILIBRATION EVERYWHERE
                # rg_frame = data[run, data[run, :, 0] > self.equilibration, 4]
                rg_frame = data[run, :, 4]
                rgs.append(rg_frame)
        if use == 'md':
            """ Only this case seems to work for multichain ! """
            for traj in self.structures:
                if self.chains != 1:
                    rg = 0
                # TODO : PARALLELIZE THIS (split frames into block and do it!)
                    for chain in range(self.chains):
                        tr = traj[:]
                        atom_slice = np.linspace(chain*self.chain_atoms, (chain+1)*self.chain_atoms, self.chain_atoms, endpoint=False, dtype=int)
                        tr = tr.atom_slice(atom_slice)
                        rg += md.compute_rg(tr)
                    rg /= self.chains
                    rg = rg * 10
                else:
                    rg = md.compute_rg(traj)*10.
                rgs.append(rg)
        return np.array(rgs)

    def minimize_I_ls(self, a_dir, b_dir, temp_dir='/home/adria/scripts/lammps/temp', I0=100e-3, l0=1.0, T=0):
        # TODO HANDLE IF B > A
        above_tc = Analysis(oliba_wd=a_dir)
        below_tc = Analysis(oliba_wd=b_dir)

        global rw_A
        global rw_B
        rw_A, rw_B = 0, 0

        def trim_warnings(f):
            with open(f, 'r+') as file:
                lines = file.readlines()
                file.seek(0)
                for line in lines:
                    if "WARNING" not in line:
                        file.write(line)
                file.truncate()

        def rg_rerun(T, ls, I, protein_object, data_dir):
            last = 5000
            ev = 10
            Ei = protein_object.data[T, :, 1][-last::ev]
            rgi = protein_object.rg(use='md')[T, :]
            protein = protein_object.protein
            shutil.copyfile(os.path.join(data_dir, 'data.data'), os.path.join(temp_dir, f'data.data'))
            shutil.copyfile(os.path.join(data_dir, f'reorder-{T}.lammpstrj'), os.path.join(temp_dir, f'atom_traj_{protein}.lammpstrj'))
            rerun = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein=protein, temper=False)
            rerun.ionic_strength = I
            rerun.hps_scale = ls
            rerun.save = int(protein_object.get_lmp_data()[T, 1, 0] - protein_object.get_lmp_data()[T, 0, 0])
            rerun.t = int(np.max(protein_object.get_lmp_data()[T, :, 0]))
            rerun.rerun_dump = f'atom_traj_{protein}.lammpstrj'
            rerun.get_hps_pairs()
            rerun.rerun_skip = int(ev)
            rerun.rerun_start = int(rerun.t+rerun.save - last*rerun.save)
            rerun.rerun_stop = rerun.t
            rerun.temperature = protein_object.get_temperatures()[T]
            rerun.write_hps_files(rerun=True, data=False, qsub=False, slurm=False, readme=False, rst=False)
            rerun.run(file=os.path.join(rerun.out_dir, 'lmp.lmp'), n_cores=2)

            rerun_analysis = Analysis(oliba_wd=temp_dir)
            Ef = rerun_analysis.data[0, :, 1]

            weights = rerun_analysis.weights(Ei=Ei, Ef=Ef, T=protein_object.get_temperatures()[T])
            weights = weights/np.sum(weights)
            reweight_rg = np.dot(rgi, weights.T)
            return reweight_rg

        def cost(x, T):
            T = T[0]
            I, ls = x[0], x[1]
            rw_above_rg = rg_rerun(T=T, ls=ls, I=I, protein_object=above_tc, data_dir=a_dir)
            rw_below_rg = rg_rerun(T=T, ls=ls, I=I, protein_object=below_tc, data_dir=b_dir)
            global rw_A
            global rw_B
            rw_A = rw_above_rg
            rw_B = rw_below_rg
            c = - (rw_above_rg - rw_below_rg)**2 + (5*((rw_below_rg + rw_above_rg)/2 - mean_rg))**2
            print(rw_A, rw_B)
            return c

        # TEST
        rgiA = above_tc.rg(use='md').mean(axis=1)
        rgiB = below_tc.rg(use='md').mean(axis=1)
        print(rgiA[T], rgiB[T])
        mean_rg = (above_tc.rg(use='md').mean(axis=1)+below_tc.rg(use='md').mean(axis=1))/2
        mean_rg = mean_rg[T]
        x0 = np.array([I0, l0])
        m = scipy.optimize.minimize(fun=cost, x0=x0, args=[T])
        print("="*80)
        print(f"For T={T} and initial parameters I = {I0}, ls = {l0} with RgA = {above_tc.rg(use='md').mean(axis=1)[T]:.2f}, RgB = {below_tc.rg(use='md').mean(axis=1)[T]:.2f}")
        print(f"Minimization resulted in {m.x[0]:.2f}  {m.x[1]:.2f} and reweighted Rg's A = {rw_A:.2f} and B = {rw_B:.2f}")
        print("="*80)
        result = "=============================================\n"
        result += f'T={T}, I0={I0}, l0={l0}, rgiAbove={rgiA}, rgiBelow={rgiB} \n'
        result += f'If={m.x[0]:.2f}, lf={m.x[1]:.2f}, rgAboveF={rw_A:.2f}, rgBelowF={rw_B:.2f}\n'
        result += "=============================================\n"
        with open(f'minimization_{T}.txt', 'a+') as min:
            min.write(result)
        return m

    def minimize_I_eps(self, a_dir, b_dir, temp_dir='/home/adria/scripts/lammps/temp', I0=100e-3, eps0=80.0, T=0):
        # TODO HANDLE IF B > A
        above_tc = Analysis(oliba_wd=a_dir)
        below_tc = Analysis(oliba_wd=b_dir)

        global rw_A
        global rw_B
        rw_A, rw_B = 0, 0

        def trim_warnings(f):
            with open(f, 'r+') as file:
                lines = file.readlines()
                file.seek(0)
                for line in lines:
                    if "WARNING" not in line:
                        file.write(line)
                file.truncate()

        def rg_rerun(T, eps, I, protein_object, data_dir):
            last = 5000
            ev = 10
            Ei = protein_object.data[T, :, 1][-last::ev]
            rgi = protein_object.rg(use='md')[T, :]
            protein = protein_object.protein
            shutil.copyfile(os.path.join(data_dir, 'data.data'), os.path.join(temp_dir, f'data.data'))
            shutil.copyfile(os.path.join(data_dir, f'reorder-{T}.lammpstrj'), os.path.join(temp_dir, f'atom_traj_{protein}.lammpstrj'))
            rerun = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein=protein, temper=False)
            rerun.ionic_strength = I
            rerun.water_perm = eps
            rerun.save = int(protein_object.get_lmp_data()[T, 1, 0] - protein_object.get_lmp_data()[T, 0, 0])
            rerun.t = int(np.max(protein_object.get_lmp_data()[T, :, 0]))
            rerun.rerun_dump = f'atom_traj_{protein}.lammpstrj'
            rerun.get_hps_pairs()
            rerun.rerun_skip = int(ev)
            rerun.rerun_start = int(rerun.t+rerun.save - last*rerun.save)
            rerun.rerun_stop = rerun.t
            rerun.temperature = protein_object.get_temperatures()[T]
            rerun.write_hps_files(rerun=True, data=False, qsub=False, slurm=False, readme=False, rst=False)
            rerun.run(file=os.path.join(rerun.out_dir, 'lmp.lmp'), n_cores=2)

            rerun_analysis = Analysis(oliba_wd=temp_dir)
            Ef = rerun_analysis.data[0, :, 1]

            weights = rerun_analysis.weights(Ei=Ei, Ef=Ef, T=protein_object.get_temperatures()[T])
            weights = weights/np.sum(weights)
            reweight_rg = np.dot(rgi, weights.T)
            return reweight_rg

        def cost(x, T):
            T = T[0]
            I, eps = x[0], x[1]
            print(I,eps)
            rw_above_rg = rg_rerun(T=T, eps=eps, I=I, protein_object=above_tc, data_dir=a_dir)
            rw_below_rg = rg_rerun(T=T, eps=eps, I=I, protein_object=below_tc, data_dir=b_dir)
            global rw_A
            global rw_B
            rw_A = rw_above_rg
            rw_B = rw_below_rg
            c = - (rw_above_rg - rw_below_rg)**2 + (5*((rw_below_rg + rw_above_rg)/2 - mean_rg))**2
            return c

        # TEST
        rgiA = above_tc.rg(use='md').mean(axis=1)
        rgiB = below_tc.rg(use='md').mean(axis=1)
        print(rgiA[T], rgiB[T])
        mean_rg = (above_tc.rg(use='md').mean(axis=1)+below_tc.rg(use='md').mean(axis=1))/2
        mean_rg = mean_rg[T]
        x0 = np.array([I0, eps0])
        m = scipy.optimize.minimize(fun=cost, x0=x0, args=[T])
        print(m.x)
        print("="*80)
        print(f"For T={T} and initial parameters I = {I0}, eps = {eps0} with RgA = {above_tc.rg(use='md').mean(axis=1)[T]:.2f}, RgB = {below_tc.rg(use='md').mean(axis=1)[T]:.2f}")
        print(f"Minimization resulted in {m.x[0]}  {m.x[1]} and reweighted Rg's A = {rw_A:.2f} and B = {rw_B:.2f}")
        print("="*80)
        result = "=============================================\n"
        result += f'T={T}, I0={I0}, eps={eps0}, rgiAbove={rgiA[T]}, rgiBelow={rgiB[T]} \n'
        result += f'If={m.x[0]}, eps={m.x[1]}, rgAboveF={rw_A}, rgBelowF={rw_B}\n'
        result += "=============================================\n"
        with open(os.path.join(definitions.module_dir,f'temp/minimization_{T}_{above_tc.protein}_{below_tc.protein}.txt'), 'a+') as min:
            min.write(result)
        return m

    def minimize(self, a_dir, b_dir, T, temp_dir=os.path.join(definitions.module_dir, 'temp'), I0=None, l0=None, eps0=None):
        default = {"I": 100e-3, "ls": 1.0, "eps": 80.0}
        above_tc, below_tc = Analysis(oliba_wd=a_dir), Analysis(oliba_wd=b_dir)
        rgiA, rgiB = above_tc.rg(use='md').mean(axis=1)[T], below_tc.rg(use='md').mean(axis=1)[T]
        mean_rg = (rgiA + rgiB)/2
        x0, x0_dict, args, args_dict = [], {}, [], {}
        if I0 is not None:
            x0.append(I0)
            x0_dict["I"] = len(x0)-1
        if l0 is not None:
            x0.append(l0)
            x0_dict["ls"] = len(x0)-1
        if eps0 is not None:
            x0.append(eps0)
            x0_dict["eps"] = len(x0)-1

        def rg_rerun(T, eps, I, ls, protein_object, data_dir):
            last = 5000
            ev = 10
            Ei = protein_object.data[T, :, 1][-last::ev]
            rgi = protein_object.rg(use='md')[T, :]
            protein = protein_object.protein
            shutil.copyfile(os.path.join(data_dir, 'data.data'), os.path.join(temp_dir, f'data.data'))
            shutil.copyfile(os.path.join(data_dir, f'reorder-{T}.lammpstrj'), os.path.join(temp_dir, f'atom_traj_{protein}.lammpstrj'))
            rerun = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein=protein, temper=False)
            rerun.ionic_strength = I
            rerun.water_perm = eps
            rerun.hps_scale = ls
            rerun.save = int(protein_object.get_lmp_data()[T, 1, 0] - protein_object.get_lmp_data()[T, 0, 0])
            rerun.t = int(np.max(protein_object.get_lmp_data()[T, :, 0]))
            rerun.rerun_dump = f'atom_traj_{protein}.lammpstrj'
            rerun.get_hps_pairs()
            rerun.rerun_skip = int(ev)
            rerun.rerun_start = int(rerun.t+rerun.save - last*rerun.save)
            rerun.rerun_stop = rerun.t
            rerun.temperature = protein_object.get_temperatures()[T]
            rerun.write_hps_files(rerun=True, data=False, qsub=False, slurm=False, readme=False, rst=False)
            rerun.run(file=os.path.join(rerun.out_dir, 'lmp.lmp'), n_cores=4)

            rerun_analysis = Analysis(oliba_wd=temp_dir)
            Ef = rerun_analysis.data[0, :, 1]

            weights = rerun_analysis.weights(Ei=Ei, Ef=Ef, T=protein_object.get_temperatures()[T])
            weights = weights/np.sum(weights)
            print(f"n_eff = {self.n_eff(weights)}")
            reweight_rg = np.dot(rgi, weights.T)
            return reweight_rg

        def cost(x, args):
            T = args[0]
            if "I" in x0_dict:
                I = x[x0_dict["I"]]
            else:
                I = default["I"]
            if "ls" in x0_dict:
                ls = x[x0_dict["ls"]]
            else:
                ls = default["ls"]
            if "eps" in x0_dict:
                eps = x[x0_dict["eps"]]
            else:
                eps = default["eps"]
            print("="*80)
            print(f"I={I}, ls={ls}, eps={eps}")
            rw_above_rg = rg_rerun(T=T, eps=eps, I=I, ls=ls, protein_object=above_tc, data_dir=a_dir)
            rw_below_rg = rg_rerun(T=T, eps=eps, I=I, ls=ls, protein_object=below_tc, data_dir=b_dir)
            global rw_A
            global rw_B
            rw_A, rw_B = rw_above_rg, rw_below_rg
            c = - (rw_above_rg - rw_below_rg)**2 + 0.2*((rw_below_rg + rw_above_rg)/2 - mean_rg)**2
            print(f"Rw RgA, {rw_A:.3f}, Rw RgB {rw_B:.3f}, diff {rw_A - rw_B:.2f}, dist to mean {(rw_below_rg + rw_above_rg)/2 - mean_rg:.2f}, cost {c:.2f}")
            print("="*80)
            return c

        m = scipy.optimize.minimize(fun=cost, x0=np.array(x0), args=[T, default["I"], default["ls"], default["eps"]])

        print("=" * 80)
        print(
            f"For T={T} and initial parameters I = {I0}, eps = {eps0} with RgA = {above_tc.rg(use='md').mean(axis=1)[T]:.2f}, RgB = {below_tc.rg(use='md').mean(axis=1)[T]:.2f}")
        print(f"Minimization resulted in {m.x[0]}  {m.x[1]} and reweighted Rg's A = {rw_A:.2f} and B = {rw_B:.2f}")
        print("=" * 80)
        result = "=============================================\n"
        result += f'T={T}, I0={I0}, eps={eps0}, rgiAbove={rgiA}, rgiBelow={rgiB} \n'
        result += f'If={m.x[0]}, eps={m.x[1]}, rgAboveF={rw_A}, rgBelowF={rw_B}\n'
        result += "=============================================\n"
        with open(f'minimization_{T}.txt', 'a+') as min:
            min.write(result)

        return m

    def n_eff(self, weights, kish=False):
        if kish:
            n_eff = 1. / np.sum(weights ** 2) / weights.size
        else:
            n_eff = np.exp(-np.sum(weights * np.log(weights * weights.size)))
        # if kish:
        #     for weight in weights:
        #         n_eff.append(1. / np.sum(weights ** 2) / weights.size)
        # else:
        #     for weight in weights:
        #         n_eff.append(np.exp(-np.sum(weight * np.log(weight * weight.size))))
        return n_eff

    def weights(self, Ei, Ef, T):
        """ ENERGIES IN LMP IS IN KCAL/MOL (1 KCAL = 4184J) """
        kb = cnt.Boltzmann*cnt.Avogadro/4184
        w = np.exp(-(Ef-Ei)/(kb*T))
        return w

    def block_error(self, observable, block_length=10):
        """
        Observable needs to be in shape (T, frames) (assuming Temper)
        """
        mean = observable.mean(axis=1)
        errors = []
        for T in range(0, observable.shape[0]):
            err = 0
            n_blocks = 0
            for r in range(0, observable.shape[1]-block_length, block_length):
                rng = slice(r, r+block_length)
                err += (observable[T, rng].mean() - mean[T])**2
                n_blocks += 1
            errors.append(np.sqrt(err)/n_blocks)
        return np.array(errors)

    def find_Tc(self, florys=None):
        temps = self.get_temperatures()
        if florys is None:
            florys = self.flory_scaling_fit()[0]
        lim_inf = np.max(florys[florys < 0.5])
        lim_sup = np.min(florys[florys > 0.5])
        # TODO : FLOAT CONVERSION IN SELF.GET_TEMPERATURES()?
        T_inf = float(temps[np.where(florys == lim_inf)[0][0]])
        T_sup = float(temps[np.where(florys == lim_sup)[0][0]])
        slope = (lim_sup-lim_inf)/(T_sup-T_inf)
        intersect = lim_sup-slope*T_sup
        Tc = (0.5 - intersect)/slope
        return Tc

    def chain_coms(self):
        sequence = self.get_seq_from_hps()
        m_i = []
        for atom in range(self.chain_atoms):
            aa = sequence[atom]
            m_i.append(self.residue_dict[aa]["mass"])
        m_i = np.array(m_i)
        M = np.sum(m_i)
        coms = np.zeros(shape=(12, self.chains, self.structures[0].n_frames, 3))
        for T, traj in enumerate(self.structures):
            for c in range(self.chains):
                atidx = np.linspace(c*self.chain_atoms, (c+1)*self.chain_atoms, self.chain_atoms, endpoint=False, dtype=int)
                tr = traj.atom_slice(atidx)
                xyz = np.reshape(tr.xyz, (tr.xyz.shape[0], tr.xyz.shape[2], tr.xyz.shape[1]))*10
                s = np.dot(xyz, m_i)
                com = s/M
                coms[T, c, :] = com
        return coms

    def clusters(self, T, cutoff=50):
        def run(chain, filling):
            if chain not in taken:
                # clusters[chain] = [chain]
                clusters.append([chain])
                filling = chain
            for i, d in enumerate(d_matrix[chain]):
                if cutoff > d > 0 and i not in taken:
                    taken.append(i)
                    if i != filling:
                        clusters[-1].append(i)
                    run(i, filling)

        nchains = self.chains
        f_xyz = self.chain_coms()
        f_xyz = f_xyz[T, :, :]
        f_clust = []

        for f in range(self.structures[T].n_frames):
            xyz = f_xyz[:, f, :]
            d_matrix = np.zeros(shape=(nchains, nchains))
            for i, d1 in enumerate(xyz):
                for j, d2 in enumerate(xyz):
                    d_matrix[i][j] = np.linalg.norm(d1 - d2)

            taken, clusters = [], []

            for chain in range(d_matrix.shape[0]):
                run(chain, 0)
            f_clust.append(clusters)
        return f_clust, f_xyz

    def density_profile(self, T=None):
        if T is None:
            T = range(len(self.get_temperatures()))
        else:
            T = [T]
        coms_temp = []
        for temp in T:
            frame_clusters, coms = self.clusters(T=temp)
            for f, clusters in enumerate(frame_clusters):
                l_old, biggest_cluster = 0, None
                for cluster in clusters:
                    l_new = len(cluster)
                    if l_new > l_old:
                        biggest_cluster = cluster
                        l_old = l_new

                r = 0
                for chain in biggest_cluster:
                    r += coms[chain, f, :]
                r /= len(biggest_cluster)

                coms[:, f, :] -= r
            coms_temp.append(coms)
        return np.array(coms_temp)


mpi_results = []


def full_flory_scaling(x, flory, r0):
    return r0 * (x ** flory)


def flory_scaling(r0):
    def flory(x, flory):
        return r0 * (x ** flory)
    return flory


@jit(nopython=True)
def jit_contact_calc(natoms, positions):
    dij = np.zeros(shape=(natoms, natoms))
    for i in range(natoms):
        for j in range(natoms):
            d = 0
            for r in range(3):
                d += (positions[0, i, r] - positions[0, j, r]) ** 2
            dij[i, j] = math.sqrt(d)
    return dij


def mpi_contact_calc(natoms, positions):
    d_fr = []
    for fr in range(positions.shape[0]):
        dij = np.zeros(shape=(natoms, natoms))
        for i in range(natoms):
            for j in range(natoms):
                d = 0
                for r in range(3):
                    d += (positions[fr, i, r] - positions[fr, j, r]) ** 2
                dij[i, j] = math.sqrt(d)
        d_fr.append(dij)
    return np.array(d_fr).mean(axis=0)
