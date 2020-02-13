import analysis
import os
import glob
import re
import pandas as pd
import numpy as np
import pathlib
from scipy import stats
import matplotlib.pyplot as plt


class Plotter(analysis.Analysis):
    # def __init__(self, force_recalc=False, **kw):
    #     super(Plotter, self).__init__(**kw)
    def __init__(self, force_recalc=False, **kw):

        super().__init__(**kw)
        self.oliba_prod_dir = '/home/adria/data/prod/lammps'
        self.oliba_data_dir = '/home/adria/data/data'
        self.index = self.make_index()

        self.figure, self.axis = None, None
        self.protein = None
        self.obs_data = None
        self.label = None
        self.style = None
        self.title_fsize = 20
        self.label_fsize = 16
        self.ticks_fsize = 14
        # TODO HARD CODED...
        self.temperatures = None

        # Observables for convenience
        self.flory = None
        # TODO : SWAP NAMES BETWEEN THIS AND THE PROCESS ?
        self.gyration = None
        self.contact_map = None
        self.distant_map = None

        self.observables = ['rg', 'distance_map', 'contact_map', 'dij', 'flory', 'charge']
        self.force_recalc = force_recalc

    def make_index(self, force_update=True):
        if os.path.exists('../data/index.txt') and not force_update:
            df = pd.read_csv('../data/index.txt', sep=' ', index_col=0)
            print(df)
            return df
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
            lj_lambda_line = None
            lmp_path = glob.glob(os.path.join(d, '*.lmp'))
            self._get_hps_params()
            if lmp_path:
                log_lmp = open(lmp_path[0], 'r')
                lines = log_lmp.readlines()
                for line in lines:
                    if '#' in line:
                        continue
                    if "dielectric" in line:
                        eps = re.findall(r'\d+', line)[0]
                        epss.append(float(eps))
                    if "ljlambda" in line:
                        lj_lambda_line = line
                    if lj_lambda_line and eps:
                        debye = re.findall(r'\d+\.?\d*', lj_lambda_line)[0]
                        unroundedI = self.get_I_from_debye(float(debye), eps=float(eps))
                        if unroundedI >= 0.1:
                            I = round(unroundedI, 1)
                        else:
                            I = round(unroundedI, 3)
                        Is.append(int(I*10**3))
                        lj_lambda_line = None
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
        df.to_csv('../data/index.txt', sep=' ', mode='w')
        return df

    # TODO : TOO MANY ARGUMENTS... think of another possibility ?
    def plot(self, observable, protein, eps, I, ls, protein2=None, sequence=None, temperature=4, label=None, style=None):
        # TODO : SEQUENCE HAS TO BE EXPLICIT, MAYBE CAN BE IMPLICIT
        self.obs_data = None
        if label:
            self.label = label
        else:
            self.label = f'{protein}, I = {I:.0f}, ε = {eps:.0f} HPS = {ls:.1f}'
        self.style = '-'
        if style:
            self.style = style

        # if not self.figure or observable in ['distance_map', 'contact_map']:
        self.figure, self.axis = plt.subplots(figsize=(16, 12), sharex=True)
        if observable not in self.observables:
            print(f"Observable not found. Available ones are : {self.observables}")
            return
        df = self.index
        df2 = df.copy()
        df = df.loc[df['Protein'] == protein]
        df = df.loc[df['Eps'] == eps]
        df = df.loc[df['Scale'] == ls]
        df = df.loc[df['I'] == I]

        fout = f'{ls:.1f}ls-{I:.0f}I-{eps:.0f}e'
        d = os.path.join(self.oliba_data_dir, observable, protein, fout)
        if os.path.exists(d) and self.force_recalc is False:
            print("Requested data already available")
            self.obs_data = np.genfromtxt(os.path.join(d, 'data.txt'))

        # TODO BETTER WAY FOR THIS ???
        # TODO : Maybe pass metaobject to plot ?
        self.o_wd = df.to_numpy()[0][-1]
        # TODO : TOO LATE TO DO THIS ?
        self.temperatures = self.get_temperatures()
        self.protein = protein
        # TODO : IS IT REALLY NECESSARY ?
        self.lmp_drs = self.get_lmp_dirs()

        o_wd2 = None
        if protein2:
            df2 = df2.loc[df2['Protein'] == protein2]
            df2 = df2.loc[df2['Eps'] == eps]
            df2 = df2.loc[df2['Scale'] == ls]
            df2 = df2.loc[df2['I'] == I]
            o_wd2 = df2.to_numpy()[0][-1]
        if observable == 'rg':
            data = self.plot_rg()
        if observable == 'distance_map':
            if o_wd2:
                data = self.plot_distance_map(sequence, temperature=temperature, b=o_wd2, double=True)
            else:
                data = self.plot_distance_map(sequence, temperature=temperature)
        if observable == 'contact_map':
            if o_wd2:
                data = self.plot_distance_map(sequence, contacts=True, temperature=temperature, b=o_wd2, double=True)
            else:
                data = self.plot_distance_map(sequence, contacts=True, temperature=temperature)
        if observable == 'dij':
            data = self.plot_dijs(plot_flory_fit=True, plot_ideal_fit=False)
        if observable == 'flory':
            data = self.plot_flory()
        if observable == 'charge':
            data = self.plot_q_distr(sequence=sequence)
        self.axis.legend(fontsize=self.label_fsize)
        self.axis.xaxis.set_tick_params(labelsize=self.ticks_fsize)
        self.axis.yaxis.set_tick_params(labelsize=self.ticks_fsize)
        if self.obs_data is None:
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
            print("Saving data")
            np.savetxt(os.path.join(d, 'data.txt'), data)
        return

    def plot_distance_map(self, sequence, b=None, contacts=False, temperature=None, double=False, log=False):
        if self.obs_data is not None:
            cont_map = self.obs_data
        else:
            cont_map = self.distance_map(use='md', contacts=contacts, temperature=temperature)
        cont_map = np.array(cont_map)
        cmap = 'plasma'

        if b:
            B = analysis.Analysis(oliba_wd=b, temper=self.temper)
            B_contacts = B.distance_map(use='md', contacts=contacts, temperature=temperature)
            # TODO: FIX! OTHER WAY AROUND...
            if not double:
                if log:
                    cont_map = np.log(cont_map/B_contacts)
                else:
                    cont_map = cont_map - B_contacts
            cmap = 'PRGn'

        if temperature:
            distance_map = cont_map[0]
            # distance_map.max()
            # TODO : BETTER THIS
            if double and b:
                bcont = np.array(B_contacts[0])
                distance_map[np.triu_indices_from(distance_map)] = bcont[np.triu_indices_from(bcont)]
                print(distance_map)
                # distance_map[np.triu_indices_from(distance_map)] = bcont[np.triu_indices_from(bcont)]
                cmap = 'plasma'
                img = self.axis.imshow(distance_map, cmap=cmap)
            elif b:
                cmap = 'PRGn'
                img = self.axis.imshow(distance_map, cmap=cmap, vmin=-np.max(distance_map), vmax=np.max(distance_map))
            else:
                cmap = 'plasma'
                img = self.axis.imshow(distance_map, cmap=cmap)
            self.figure.subplots_adjust(left=0, right=1)
            self.axis.set_adjustable('box')
            self.axis.invert_yaxis()

            q_total, q_plus, q_minus = self.get_charge_seq(sequence)
            for i, q in enumerate(q_total):
                if q < 0:
                    self.axis.axvline(i, alpha=0.15, color='red')
                if q > 0:
                    self.axis.axvline(i, alpha=0.15, color='blue')
            self.axis.axvspan(403, 410, color='black', alpha=0.3, label='E4')
            cb = self.figure.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.08)
            cb.set_label('$\mathregular{d_{ij}}$', fontsize=20)
            cb.ax.tick_params(labelsize=16)
        else:
            for T in range(len(cont_map)):
                distance_map = cont_map[T, :, :]
                if double and b:
                    distance_map[np.triu_indices_from(distance_map)] = B_contacts[T]
                img = self.axis.imshow(distance_map, cmap=cmap, vmin=-np.max(distance_map), vmax=np.max(distance_map))
                # img = self.axis.imshow(distance_map, cmap=cmap, vmin=0., vmax=0.2)
                self.figure.subplots_adjust(left=0, right=1)
                self.axis.invert_yaxis()

                q_total, q_plus, q_minus = self.get_charge_seq(sequence)
                for i, q in enumerate(q_total):
                    if q < 0:
                        self.axis.axvline(i, alpha=0.03, color='red')
                    if q > 0:
                        self.axis.axvline(i, alpha=0.03, color='blue')
                if len(sequence) == 448:
                    self.axis.axvspan(403, 410, color='black', alpha=0.3, label='E4')
                cb = self.figure.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.08)
                cb.set_label('$\mathregular{d_{ij}}$', fontsize=20)
                cb.ax.tick_params(labelsize=16)
        self.axis.set_adjustable('box', True)
        return distance_map

    def plot_dijs(self, r0=5.5, plot_flory_fit=False, plot_ideal_fit=False):
        ijs, means = self.ij_from_contacts(use='md')
        if plot_flory_fit:
            florys = self.flory_scaling_fit(use='md', r0=r0, ijs=[ijs, means])[0]

        self.axis.set_xlabel("|i-j|")
        self.axis.set_ylabel(r'dij ($\AA$)')
        for i in range(ijs.shape[0]):
            ij = ijs[i, :]
            mean = means[i, :]
            if plot_flory_fit:
                flory = florys[i]
                self.axis.plot(ij, r0 * np.array(ij) ** flory, '--', label=f"Fit to {r0:.1f}*N^{flory:.3f}")
            if plot_ideal_fit:
                self.axis.plot(ij, np.array(ij) ** 0.5 * r0, '--', label="Random Coil Scaling (5.5*N^0.5)")
            self.axis.plot(ij, mean, label="HPS results")
        fout = f'../default_output/test.png'
        self.axis.savefig(fout)
        return ijs, means

    def plot_rg(self):
        if self.obs_data is not None:
            rg = self.obs_data
        else:
            rg = self.rg(use='md')
        err_bars = self.block_error(observable=rg)
        self.axis.set_ylabel(r"Rg ($\AA$)", fontsize=self.label_fsize)
        self.axis.set_xlabel("T (K)", fontsize=self.label_fsize)
        self.axis.plot(self.temperatures, rg.mean(axis=1), self.style, label=self.label)
        if self.flory is not None:
            Tc = self.find_Tc(florys=self.flory)
            idx_sup = np.where(self.temperatures == np.min(self.temperatures[self.temperatures > Tc]))[0][0]
            idx_inf = np.where(self.temperatures == np.max(self.temperatures[self.temperatures > Tc]))[0][0]
            slope = (rg[idx_sup, :].mean() - rg[idx_inf, :].mean())/(self.temperatures[idx_sup] - self.temperatures[idx_inf])
            intersect = rg[idx_sup, :].mean() - slope*self.temperatures[idx_sup]
            rgC = slope*Tc + intersect
            phase_params = self.axis.scatter(Tc, rgC, s=80, color='firebrick', alpha=0.5)
            leg = self.axis.legend([phase_params], [f"Phase Transition. Tc = {Tc:.0f}, Rgc = {rgC:.0f}"], loc=4, fontsize=self.label_fsize)
            self.axis.add_artist(leg)
            ylim = self.axis.get_ylim()
            self.axis.set_ylim(ylim)
            self.axis.axhspan(rgC, ylim[1], color="green", alpha=0.08)
            self.axis.axhspan(ylim[0], rgC, color="red", alpha=0.08)
        pline, capline, barline = self.axis.errorbar(self.temperatures, rg.mean(axis=1), yerr=err_bars, uplims=True, lolims=True)
        capline[0].set_marker('_')
        capline[0].set_markersize(10)
        capline[1].set_marker('_')
        capline[1].set_markersize(10)
        self.gyration = rg
        return rg

    def plot_flory(self, r0=5.5):
        if self.obs_data is not None:
            # TODO : RECALCULATE ERR INSTEAD OF SAVING ?
            florys, r0s, err = self.obs_data
        else:
            florys, r0s, err = self.flory_scaling_fit(use='md', r0=r0)
        self.axis.set_ylabel('\u03BD', fontsize=self.label_fsize)
        self.axis.set_xlabel("T (K)", fontsize=self.label_fsize)
        self.axis.set_ylim(0.3, 0.65)
        self.axis.axhspan(0.5, 0.65, color="green", alpha=0.08)
        self.axis.axhspan(0.3, 0.5, color="red", alpha=0.08)
        good_line = self.axis.axhline(3/5, ls='--', color='green')
        bad_line = self.axis.axhline(1/3, ls='--', color="red")
        leg = self.axis.legend([good_line, bad_line], ["Good Solvent", "Poor Solvent"], loc=2, fontsize=self.label_fsize)
        self.axis.plot(self.temperatures, florys, self.style, label=self.label)
        Tc = self.find_Tc(florys=florys)
        self.axis.axvline(Tc, ls='--', label='T Critical', alpha=0.4)

        pline, capline, barline = self.axis.errorbar(self.temperatures, florys, yerr=err, uplims=True, lolims=True)
        capline[0].set_marker('_')
        capline[0].set_markersize(10)
        capline[1].set_marker('_')
        capline[1].set_markersize(10)
        self.axis.add_artist(leg)
        self.flory = florys
        return np.array([florys, r0s, err])

    def plot_q_distr(self, sequence, window=9):
        total, plus, minus = self.get_charge_seq(sequence=sequence, window=window)
        self.axis.stem(plus, markerfmt=' ', use_line_collection=True, linefmt='blue', basefmt='')
        self.axis.stem(minus, markerfmt=' ', use_line_collection=True, linefmt='red', basefmt='')
        self.axis.set_ylim(-0.6, 0.6)
        return total

    def plot_E_ensemble(self):
        # TODO INTEGRATE TO PLOT FUNCTION
        self.figure, self.axis = plt.subplots(figsize=(16, 12), sharex=True)
        data = self.get_lmp_temper_data()
        print(data.shape)
        # kin_E = data[:, :, 2]
        kin_E = 0
        pot_E = data[:, :, 1]
        E = kin_E + pot_E
        # self.axis.hist(E[0], density=True, label=0, bins=80)
        for T in range(E.shape[0]):
            kde_scipy = stats.gaussian_kde(E[T])
            x = np.linspace(min(E[T])-5000, max(E[T]) + 5000, 1000)
            self.axis.fill_between(x=x, y1=np.zeros(shape=x.shape), y2=kde_scipy(x), zorder=2, alpha=0.5)
            self.axis.hist(E[T], density=True, label=f'T = {self.temperatures[T]:.0f}',  bins=80)
            self.axis.plot(x, kde_scipy(x))
        self.axis.legend()

    # TODO !!!! Still need to be translated, but they are pretty niche atm !!!!
    def plot_gaussian_kde(dir, equil=3000):
        rgs, Is = calc.rg_from_lmp(dir, equil)
        rgs = rgs[0]
        kde_scipy = scipy.stats.gaussian_kde(rgs)

        x = np.linspace(20, 40, 1000)

        plt.hist(rgs, density=True)
        plt.plot(x, kde_scipy(x), '--', label='Scipy')
        plt.legend()
        plt.show()

    # TODO !!!! Still need to be translated, but they are pretty niche atm !!!!
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

    # TODO !!!! Still need to be translated, but they are pretty niche atm !!!!
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

    # TODO !!!! Still need to be translated, but they are pretty niche atm !!!!
    def plot_autocorrelation(dir, equil=3655000):
        rgs = calc.rg_from_lmp(dir, equil)
        rgs = rgs[0][0]
        acf = sm.tsa.stattools.acf(rgs, nlags=1000)
        return rgs, acf
