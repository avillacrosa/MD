import analysis
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Plotter(analysis.Analysis):
    def __init__(self, **kw):
        super(Plotter, self).__init__(**kw)
        self.oliba_prod_dir = '/home/adria/data/prod/lammps'
        self.oliba_data_dir = '/home/adria/data/data/lammps'
        self.index = self.make_index()

    def make_index(self):
        lmp_dirs = self.get_lmp_dirs(self.oliba_prod_dir)
        df = pd.DataFrame({
            'Protein': 'none',
            'I': np.zeros(len(lmp_dirs)),
            'Eps': np.zeros(len(lmp_dirs)),
            'Scale': np.zeros(len(lmp_dirs)),
            'Name': 'none',
            'FullPath': 'none'
        })
        prots, Is, epss, ss, paths = [], [], [], [], []
        for i, d in enumerate(lmp_dirs):
            # print(os.path.normpath(d).split(os.sep))
            # TODO Yeah...
            paths.append(str(os.path.basename(d)))
            protein = os.path.normpath(d).split(os.sep)[6]
            prots.append(protein)
            eps, I, scale = None, None, None
            lmp_path = glob.glob(os.path.join(d, '*.lmp'))
            self.get_hps_params()
            if lmp_path:
                log_lmp = open(lmp_path[0], 'r')
                lines = log_lmp.readlines()
                for line in lines:
                    if '#' in line:
                        continue
                    if "dielectric" in line:
                        eps = re.findall(r'\d+', line)[0]
                        epss.append(eps)
                    if "ljlambda" in line:
                        debye = re.findall(r'\d+\.?\d*', line)[0]
                        unroundedI = self.get_I_from_debye(float(debye))
                        if unroundedI >= 0.1:
                            I = round(unroundedI, 1)
                        else:
                            I = round(unroundedI, 3)
                        Is.append(int(I*10**3))
                        # print(I*10**3)
                        # print(I)
                    if "pair_coeff" in line and "1" in line and scale is None:
                        ref = self.residue_dict["A"]["lambda"]
                        test = re.findall(r'\d+\.?\d*', line)[4]
                        scale = float(test)/float(ref)
                        ss.append(round(scale, 2))
                    if eps and I and scale:
                        break
        df["Protein"] = prots
        df["I"] = Is
        df["Eps"] = epss
        df["Scale"] = ss
        df["Name"] = paths
        df["FullPath"] = lmp_dirs
        df.to_csv('../data/index.txt', sep=' ', mode='a')
        return df

    def plot_contacts(self, sequence, B=None):
        contacts = self.distance_map(use='md')
        contacts = np.array(contacts)
        cmap = 'plasma'

        if B:
            B = analysis.Analysis(oliba_wd=B, temper=self.temper)
            B_contacts = B.distance_map(use='md')
            contacts = contacts - B_contacts
            cmap = 'PRGn'

        for T in range(len(contacts)):
            distance_map = contacts[T,:,:]
            fig = plt.figure(num=None, figsize=(16, 12), frameon=False)
            ax = plt.gca()
            # img = ax.imshow(contacts, cmap='PRGn', vmin=-9, vmax=9)
            img = ax.imshow(distance_map, cmap=cmap)
            fig.subplots_adjust(left=0, right=1)
            # fig.tight_layout(pad=0)
            ax.invert_yaxis()

            q_total, q_plus, q_minus = self.get_charge_seq(sequence)
            for i, q in enumerate(q_total):
                if q < 0:
                    ax.axvline(i, alpha=0.15, color='red')
                if q > 0:
                    ax.axvline(i, alpha=0.15, color='blue')

            ax.axvspan(403, 410, color='black', alpha=0.3, label='E4')

            cb = plt.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.08)
            cb.set_label('$\mathregular{d_{ij}}$', fontsize=20)
            cb.ax.tick_params(labelsize=16)
            # img.set_clim(-35, 20)
            plt.legend(loc='upper left')
            fout = f'../default_output/test.png'
            print(f'Saving figure at {fout}')
            # plt.savefig(fout)

    def plot_dijs(self, plot_flory_fit=False, plot_ideal_fit=False):
        ijs, means = self.ij_from_contacts(use='md')
        # TODO: HARDCODED 5.5...
        florys = self.flory_scaling_fit(use='md', r0=5.5, ijs=[ijs, means])[0]

        plt.figure(num=None, figsize=(8, 6), frameon=False)
        plt.xlabel("|i-j|")
        plt.ylabel(r'dij ($\AA$)')
        for i in range(ijs.shape[0]):
            flory = florys[i]
            ij = ijs[i, :]
            mean = means[i, :]
            if plot_flory_fit:
                plt.plot(ij, 5.5 * np.array(ij) ** flory, '--', label=f"Fit to 5.5*N^{flory:.3f}")
                # plt.plot(ij, 5.5 * ij ** flory, '--')
            if plot_ideal_fit:
                plt.plot(ij, np.array(ij) ** 0.5 * 5.5, '--', label="Random Coil Scaling (5.5*N^0.5)")
            plt.plot(ij, mean, label="HPS results")
            plt.legend()
        fout = f'../default_output/test.png'
        plt.savefig(fout)

    def plot_rg(self):
        rg = self.rg(use='md')
        plt.figure(figsize=(8, 6))
        plt.ylabel("Rg (Ang)")
        plt.xlabel("$\mathregular{T_index}$")
        plt.plot(rg.mean(axis=1))


    def plot_aa_map(dir, seq, ref_dir=None):
        pdb = conversion.getPDBfromLAMMPS(os.path.join(dir, 'hps'))
        xtc = os.path.join(dir, 'hps_traj.xtc')
        aa_contacts, translator = calc.res_contact_map(xtc, pdb, seq)

        if ref_dir is not None:
            pdb_ref = conversion.getPDBfromLAMMPS(os.path.join(ref_dir, 'hps'))
            xtc_ref = os.path.join(ref_dir, 'hps_traj.xtc')
            aa_contacts_ref, translator = calc.res_contact_map(xtc_ref, pdb_ref, seq)
            aa_contacts = np.array(aa_contacts) - np.array(aa_contacts_ref)

        plt.figure(num=None, figsize=(16, 12))
        ax = plt.gca()
        img = ax.imshow(aa_contacts, cmap='PRGn', interpolation='nearest')
        ax.invert_yaxis()
        label_list = translator.keys()
        ax.set_xticks(range(0, 20, 1))
        ax.set_yticks(range(0, 20, 1))
        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)

        ax.set_yticklabels(translator.keys())
        cb = plt.colorbar(img)
        cb.set_label(r'd ($\AA$)')
        print("SAVING AT ../default_output/res_map.png")
        plt.savefig('../default_output/res_map.png')
        plt.show()

    def plot_gaussian_kde(dir, equil=3000):
        rgs, Is = calc.rg_from_lmp(dir, equil)
        rgs = rgs[0]
        kde_scipy = gaussian_kde(rgs)
        kde_scikit = KernelDensity(kernel='gaussian').fit(rgs[:, np.newaxis])

        x = np.linspace(20, 40, 1000)
        xsickit = np.linspace(20, 40, rgs.shape[0])[:, np.newaxis]

        plt.hist(rgs, density=True)
        sns.kdeplot(rgs, label='Seaborn')
        plt.plot(x, kde_scipy(x), '--', label='Scipy')
        plt.plot(xsickit, np.exp(kde_scikit.score_samples(xsickit)), label='Scikit')
        plt.legend()
        plt.show()


    def plot_autocorrelation(dir, equil=3655000):
        rgs = calc.rg_from_lmp(dir, equil)
        rgs = rgs[0][0]
        acf = sm.tsa.stattools.acf(rgs, nlags=1000)
        return rgs, acf
