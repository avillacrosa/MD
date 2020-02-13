import lmp
import itertools
import shutil
import mdtraj as md
import math
import scipy.constants as cnt
import numpy as np
import os
import scipy
import lmpsetup
import multiprocessing as mp
from scipy.optimize import curve_fit
from numba import jit


class Analysis(lmp.LMP):
    def __init__(self, **kw):
        super(Analysis, self).__init__(**kw)
        self.equilibration = 3e6

        self.contacts = None
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
            if use == 'jit':
                dframe = []
                for frame in range(traj.n_frames):
                    tframe = traj[frame]
                    dijf = jit_contact_calc(tframe.n_atoms, tframe.xyz)
                    dframe.append(dijf)
                dframe = np.array(dframe)
                contact_map = dframe.mean(0)
            elif use == 'md':
                pairs = list(itertools.product(range(traj.top.n_atoms - 1), range(1, traj.top.n_atoms)))
                d = md.compute_distances(traj, pairs)
                d = md.geometry.squareform(d, pairs)
                if contacts:
                    dcop = np.copy(d)
                    dcop[d < 0.6] = 1.
                    dcop[d > 0.6] = 0.
                    contact_map = dcop.mean(0)
                else:
                    contact_map = d.mean(0) * 10
            else:
                pool = mp.Pool()
                ranges = np.linspace(0, traj.n_frames, mp.cpu_count() + 1, dtype='int')
                for i in range(1, len(ranges)):
                    t_r = traj[ranges[i-1]:ranges[i]]
                    pool.apply_async(mpi_contact_calc,
                                     args=(t_r.n_atoms, t_r.xyz),
                                     callback=lambda x: mpi_results.append(x))
                pool.close()
                pool.join()
                contact_map = np.mean(mpi_results, axis=0) * 10
            contact_maps.append(contact_map)
        self.contacts = np.array(contact_maps)
        return contact_maps

    def inter_distance_map(self, use='md', contacts=False, temperature=None):
        if self.chains == 1:
            raise SystemError("Demanded interchain distances but only a single chain is present!")
        structures = self.structures.copy()
        contact_maps = []
        if temperature is not None:
            structures = [structures[temperature]]
        for traj in structures:
            # TODO: TEST FIRST PART OF IF
            dmaps_by_chain = np.zeros(
                shape=(self.chains, self.chains - 1, traj.n_frames, self.chain_atoms, self.chain_atoms))
            flat_pairs = list(itertools.product(range(self.chain_atoms), range(self.chain_atoms)))
            # TODO: INCORPORATE CONTACTS!
            for c1 in range(self.chains):
                for c2 in range(self.chains):
                    if c1 != c2:
                        pairs = list(itertools.product(range(c1 * self.chain_atoms, (c1 + 1) * self.chain_atoms),
                                                       range(c2 * self.chain_atoms, (c2 + 1) * self.chain_atoms)))
                        d = md.compute_distances(traj, pairs)
                        d = md.geometry.squareform(d, flat_pairs)
                        if c2 > c1:
                            dmaps_by_chain[c1, c2 - 1, :, :, :] = d
                        else:
                            dmaps_by_chain[c1, c2, :, :, :] = d
            d = dmaps_by_chain.min(axis=1).mean(axis=0)
            if contacts:
                dcop = np.copy(d)
                dcop[d < 0.6] = 1.
                dcop[d > 0.6] = 0.
                contact_map = dcop.mean(0)
            else:
                contact_map = d.mean(0) * 10
            contact_maps.append(contact_map)
        self.contacts = np.array(contact_maps)
        return contact_maps

    def ij_from_contacts(self, use='md', contacts=None):
        if contacts is None:
            self.intra_distance_map(use=use)
        else:
            self.contacts = contacts
        tot_means = []
        tot_ijs = []
        for contact in self.contacts:
            ijs = []
            means = []
            for d in range(contact.shape[0]):
                a = np.diagonal(contact, offset=int(d))
                means.append(a.mean())
                ijs.append(d)
            tot_ijs.append(ijs)
            tot_means.append(means)
        return np.array(tot_ijs), np.array(tot_means)

    def flory_scaling_fit(self, r0=None, use='md', ijs=None):
        if ijs is None:
            tot_ijs, tot_means = self.ij_from_contacts(use=use)
        else:
            # TODO : DANGEROUS...
            tot_ijs, tot_means = ijs[0], ijs[1]
        florys, r0s, covs = [], [], []
        for i, ij in enumerate(tot_ijs):
            mean = tot_means[i]
            if r0 is None:
                fit, fitv = curve_fit(full_flory_scaling, ij, mean)
                fit_flory = fit[0]
                fit_r0 = fit[1]
            else:
                fit, fitv = curve_fit(flory_scaling(r0), ij, mean)
                fit_flory = fit[0]
                fit_r0 = r0
            florys.append(fit_flory)
            r0s.append(fit_r0)
            covs.append(np.sqrt(np.diag(fitv)))
        return np.array(florys), np.array(r0s), np.array(covs)[:, 0]

    def distance_map_by_residue(self):
        contacts = self.contacts
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
                    for chain in range(self.chains):
                        tr = traj[:]
                        atom_slice = slice(chain*tr.n_atoms, (chain+1)*tr.n_atoms)
                        tr.atom_slice(atom_slice)
                        rg += md.compute_rg(tr)
                    rg = rg/self.chains*10.
                else:
                    rg = md.compute_rg(traj)*10.
                rgs.append(rg)
        return np.array(rgs)

    def minimize_I_ls(self, a_dir, b_dir, protein_a, protein_b, temp_dir='/home/adria/test/rerun/min'):
        plus = Analysis(oliba_wd=a_dir, temper=True)
        minus = Analysis(oliba_wd=b_dir, temper=True)

        EiA = plus.get_lmp_E()
        EiB = minus.get_lmp_E()

        def trim_warnings(f):
            with open(f, 'r+') as file:
                lines = file.readlines()
                file.seek(0)
                for line in lines:
                    if "WARNING" not in line:
                        file.write(line)
                file.truncate()

        def cost(x, T):
            EiA = plus.get_lmp_E()

            nframes = 201

            I = x[0]
            ls = x[1]
            shutil.copyfile(os.path.join(a_dir, 'data.data'), os.path.join(temp_dir, 'data.data'))
            shutil.copyfile(os.path.join(a_dir, f'atom_traj_{T}.lammpstrj'), os.path.join(temp_dir, 'atom_traj.lammpstrj'))
            costerA = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein=protein_a)
            costerA.ionic_strength = I
            costerA.hps_scale = ls
            costerA.save = 500
            costerA.box_size = 5000
            costerA.get_hps_params()
            costerA.get_hps_pairs()
            costerA.rerun_dump = 'atom_traj.lammpstrj'
            costerA.get_pdb_xyz(pdb='/home/adria/scripts/lammps/data/equil/12D_CPEB4_D4.pdb', padding=15)
            costerA.write_hps_files(rerun=True, data=False)
            costerA.run(file=os.path.join(temp_dir, 'lmp.lmp'), n_cores=8)
            trim_warnings(os.path.join(temp_dir, 'log.lammps'))
            EA = costerA.get_lmp_E()[:, :nframes, 0]
            a = Analysis(oliba_wd=temp_dir)
            rgA = md.compute_rg(md.load(f'/home/adria/test/rerun/12D4/dcd_traj_{T}.dcd', top='/home/adria/test/rerun/12D4/data_trj.pdb'))
            rgA = np.array([rgA[:nframes]])
            print("PREA", rgA.mean())
            EiA = EiA[0, :EA.shape[1], 0]
            wsA = a.weights(Ei=EiA, Ef=EA, T=300)
            wsA = wsA/np.sum(wsA)
            rgA = np.dot(rgA, wsA.T)

            EiB = minus.get_lmp_E()
            I = x[0]
            ls = x[1]
            shutil.copyfile(os.path.join(b_dir, 'data.data'), os.path.join(temp_dir, 'data.data'))
            shutil.copyfile(os.path.join(b_dir, f'atom_traj_{T}.lammpstrj'), os.path.join(temp_dir, 'atom_traj.lammpstrj'))
            costerB = lmpsetup.LMPSetup(oliba_wd=temp_dir, protein=protein_a)
            costerB.ionic_strength = I
            costerB.hps_scale = ls
            costerB.save = 500
            costerB.box_size = 5000
            costerB.get_hps_params()
            costerB.get_hps_pairs()
            costerB.rerun_dump = 'atom_traj.lammpstrj'
            costerB.get_pdb_xyz(pdb='/home/adria/scripts/lammps/data/equil/CPEB4_D4.pdb', padding=15)
            costerB.write_hps_files(rerun=True, data=True)
            costerB.run(file=os.path.join(temp_dir, 'lmp.lmp'), n_cores=8)
            trim_warnings(os.path.join(temp_dir, 'log.lammps'))
            EB = costerB.get_lmp_E()[:, :nframes, 0]
            b = Analysis(oliba_wd=temp_dir)
            rgB = md.compute_rg(
                md.load(f'/home/adria/test/rerun/D4/dcd_traj_{T}.dcd', top='/home/adria/test/rerun/D4/data_trj.pdb'))
            rgB = np.array([rgB[:nframes]])
            print("PREB", rgB.mean())
            EiB = EiB[0, :EB.shape[1], 0]
            wsB = b.weights(Ei=EiB, Ef=EB, T=300)
            wsB = wsB / np.sum(wsB)
            rgB = np.dot(rgB, wsB.T)

            rgA = rgA[0][0]
            rgB = rgB[0][0]
            c = rgA - rgB + (rgB - 21 + rgA - 21)**2 + ls-1
            print("POST A", rgA, "POST B", rgB)
            print("For ", x)
            return c

        T = (0)
        x0 = np.array([200e-3, 1.0])
        m = scipy.optimize.minimize(fun=cost, x0=x0, args=T)
        return m

    def block_error(self, observable, block_length=5):
        # TODO : Case when not temper
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
            errors.append(math.sqrt(err)/n_blocks)
        return np.array(errors)

    def weights(self, Ei, Ef, T):
        """ ENERGIES IN LMP IS IN KCAL/MOL (1 KCAL = 4184J) """
        kb = cnt.Boltzmann*cnt. Avogadro/4184
        w = np.exp(-(Ef-Ei)/(kb*T))
        return w

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
        xtcs = self._temper_trj_reorder()
        topo = self.make_initial_frame()[0]
        sequence = self.get_seq_from_hps()
        m_i = []
        for aa in sequence:
            m_i.append(self.residue_dict[aa]["mass"])
        m_i = np.array(m_i)
        M = np.sum(m_i)
        # TODO : HARDCODED
        coms = np.zeros(shape=(12, self.chains, 114, 3))
        for T, xtc in enumerate(xtcs):
            traj = md.load(xtc, top=topo)
            for c in range(self.chains):
                atidx = np.linspace(c*self.chain_atoms, (c+1)*self.chain_atoms, self.chain_atoms, endpoint=False, dtype=int)
                tr = traj.atom_slice(atidx)
                m_i_chain = m_i[c*self.chain_atoms:(c+1)*self.chain_atoms]
                xyz = np.reshape(tr.xyz, (tr.xyz.shape[0], tr.xyz.shape[2], tr.xyz.shape[1]))
                s = np.dot(xyz, m_i_chain)
                com = s/M
                coms[T, c, :] = com
        return coms

    # Cutoff in nanometers
    def clusters(self, cutoff=5):
        coms = self.chain_coms()
        clust = np.zeros(shape=(self.chain_atoms, self.chain_atoms))
        clust[0, 0] = 1
        for com1 in coms:
            current_cluster = np.where(clust[com1, :] == 1)[0][0]
            print(current_cluster)
            for com2 in coms:
                distance = np.linalg.norm(com1-com2)
                if distance < cutoff:
                    clust[com2, current_cluster] = 1
        print(clust)


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
