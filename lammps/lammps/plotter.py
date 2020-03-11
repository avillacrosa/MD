import analysis
import os
import glob
import definitions
import re
import pandas as pd
import numpy as np
import scipy
import statsmodels as sm
import pathlib
import matplotlib
from scipy import stats
import matplotlib.pyplot as plt


class Plotter(analysis.Analysis):
    def __init__(self, force_recalc=False):
        super().__init__(oliba_wd=None)
        self.oliba_prod_dir = '/home/adria/data/prod/lammps'
        self.oliba_data_dir = '/home/adria/data/data'
        self.index = self.make_index()

        self.figure, self.axis = None, None
        self.plots = {}
        self.protein = None
        self.obs_data = None
        self.label = None
        self.style = None
        self.color = None
        self.title_fsize = 20
        self.label_fsize = 16
        self.ticks_fsize = 14
        # TODO HARD CODED...
        self.temperatures = None

        # Observables for convenience
        self.florys = {}
        self.gyrations = {}
        self.dijs = {}
        self.contact_maps = {}
        self.obs_data = None
        self.plot_id = None

        self.observables = ['rg', 'distance_map', 'contact_map', 'dij', 'flory', 'charge', 'rho']
        self.force_recalc = force_recalc

        self.rg_Tc_legend_helper = {}
        self.rg_Tc_artist_helper = None

    def make_index(self, force_update=True):
        if os.path.exists(os.path.join(definitions.module_dir, 'index.txt')) and not force_update:
            df = pd.read_csv(os.path.join(definitions.module_dir, 'index.txt'), sep=' ', index_col=0)
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
        df.to_csv(os.path.join(definitions.module_dir, 'index.txt'), sep=' ', mode='w')
        return df

    def plot(self, observable, index, plot_id, **kwargs):
        self.plot_id = plot_id
        if observable not in self.observables:
            print(f"Observable not found. Available ones are : {self.observables}")
            return

        if plot_id not in self.plots:
            self.plots[plot_id] = plt.subplots(figsize=(16, 12), sharex='all')
        self.figure = self.plots[plot_id][0]
        self.axis = self.plots[plot_id][1]

        df = self.index
        protein = df.iloc[index]["Protein"]
        ionic_strength = df.iloc[index]["I"]
        eps = df.iloc[index]["Eps"]
        ls = df.iloc[index]["Scale"]

        self.protein = protein
        if os.path.exists(os.path.join(os.path.join(definitions.hps_data_dir, 'sequences'), f'{self.protein}.seq')):
            with open(os.path.join(os.path.join(definitions.hps_data_dir, 'sequences'), f'{self.protein}.seq')) as f:
                self.sequence = f.readlines()[0]
        else:
            raise SystemError(f"{self.protein}.seq not found")

        self.label = kwargs["label"] if "label" in kwargs else f'{protein}, I = {ionic_strength:.0f}, ε = {eps:.0f} HPS = {ls:.1f}'
        self.style = kwargs["style"] if "style" in kwargs else '-'
        self.color = kwargs["color"] if "color" in kwargs else None

        fout = f'{ls:.1f}ls-{ionic_strength:.0f}I-{eps:.0f}e'
        d = os.path.join(self.oliba_data_dir, observable, protein, fout)
        # if os.path.exists(d) and self.force_recalc is False:
        #     print("Requested data already available")
        #     self.obs_data = np.genfromtxt(os.path.join(d, 'data.txt'))

        self.o_wd = df.iloc[index]["FullPath"]
        # TODO BETTER WAY FOR THIS ???
        # TODO : Maybe pass metaobject to plot ?
        super().__init__(oliba_wd=self.o_wd)
        # TODO : TOO LATE TO DO THIS ?
        self.temperatures = self.get_temperatures()

        if observable == 'rg':
            data = self.plot_rg()
            self.gyrations[protein] = data
        if observable == 'distance_map' or observable == 'contact_map':
            contacts = kwargs["contacts"] if "contacts" in kwargs else False
            temperature = kwargs["temperature"] if "temperature" in kwargs else None
            double = kwargs["double"] if "double" in kwargs else False
            log = kwargs["log"] if "log" in kwargs else False
            frac = kwargs["frac"] if "frac" in kwargs else False
            inter = kwargs["inter"] if "inter" in kwargs else False

            o_wd2 = df.iloc[kwargs["index2"]]["FullPath"] if "index2" in kwargs else None
            data = self.plot_distance_map(contacts=contacts, temperature=temperature, b=o_wd2, double=double, log=log, inter=inter, frac=frac)[0]
            self.contact_maps[protein] = data
        if observable == 'dij':
            data = self.plot_dijs(plot_flory_fit=True, plot_ideal_fit=False)[1]
            self.dijs[protein] = data
        if observable == 'flory':
            data = self.plot_flory()
            if protein not in self.florys:
                self.florys[protein] = data
        if observable == 'charge':
            data = self.plot_q_distr()
        if observable == 'rho':
            temperature = kwargs["temperature"] if "temperature" in kwargs else 0
            data = self.plot_density_profile(T=temperature)
        self.axis.legend(fontsize=self.label_fsize)
        self.axis.xaxis.set_tick_params(labelsize=self.ticks_fsize)
        self.axis.yaxis.set_tick_params(labelsize=self.ticks_fsize)
        if self.obs_data is None:
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
            np.savetxt(os.path.join(d, 'data.txt'), data)
        self.figure.savefig(os.path.join(definitions.module_dir, f'temp/plot_{plot_id}.png'))
        return data

    def plot_distance_map(self, b=None, contacts=False, temperature=None, double=False, log=False, inter=False, frac=False):
        if frac and log:
            raise SystemExit("Can't do both log and frac. Choose one")
        if log and not b or frac and not b:
            raise SystemExit("log/frac plots only available for 2 proteins")

        def get_plot_params(dist_map):
            if log:
                max_map = np.copy(dist_map)
                max_map[max_map == np.inf] = 0.
                max_map[max_map == -np.inf] = 0.
                # vmax_f = np.max(max_map) if np.max(max_map) > -np.min(max_map) else -np.min(max_map)
                # vmin_f = -vmax_f
                # TODO : HARDCODIN'
                vmax_f = +.001
                vmin_f = -.001
                # label_f = '$\mathregular{\log(f_{i}/f_{j})}$' if contacts else '$\mathregular{\log(d_{i}/d_{j})}$'
                label_f = f'log(f({self.protein})/f({b_obj.protein}))' if contacts else '$\log(d_{i}/d_{j})$'
                cmap_f = matplotlib.cm.get_cmap('PRGn')
            elif frac:
                # vmax_f = 0
                # vmin_f = 2
                vmax_f = 0
                vmin_f = 2
                label_f = '$f_{i}/f_{j}$' if contacts else '$d_{i}/d_{j}$'
                cmap_f = matplotlib.cm.get_cmap('PRGn') if b else matplotlib.cm.get_cmap('plasma')
            elif contacts:
                # vmax_f = 1
                vmax_f = .01
                vmin_f = -vmax_f if b else 0
                # label_f = '$\mathregular{f_{i}-f_{j}}$' if b else '$\mathregular{f_{i}}$'
                label_f = f'f({self.protein})-f({b_obj.protein})' if b else '$f_{i}$'
                cmap_f = matplotlib.cm.get_cmap('PRGn') if b else matplotlib.cm.get_cmap('plasma')
            else:
                # vmax_f = np.max(dist_map)
                # vmin_f = -vmax_f if b else 0
                vmax_f = 15
                vmin_f = -vmax_f if b else 0
                label_f = f'd({self.protein})-d({b_obj.protein})' if b else '$\mathregular{d_{i}}$'
                cmap_f = matplotlib.cm.get_cmap('PRGn') if b else matplotlib.cm.get_cmap('plasma')
            return vmin_f, vmax_f, label_f, cmap_f

        if self.obs_data is not None:
            cont_map = self.obs_data
        else:
            # TODO : ONLY TAKING INTERCHAIN INTO ACCOUNT HERE
            cont_map = self.inter_distance_map(use='md', contacts=contacts, temperature=temperature)[0] if inter else self.intra_distance_map(use='md', contacts=contacts, temperature=temperature)
            cont_map = cont_map if inter else cont_map.mean(axis=1)
        if b:
            b_obj = analysis.Analysis(oliba_wd=b)
            b_cont_map = b_obj.inter_distance_map(use='md', contacts=contacts, temperature=temperature) if inter else b_obj.intra_distance_map(use='md', contacts=contacts, temperature=temperature)
            b_cont_map = b_cont_map.mean(axis=1)
            if log:
                cont_map = np.log(cont_map/b_cont_map)
                cont_map[np.isnan(cont_map)] = 0.
            elif frac:
                cont_map = cont_map/b_cont_map
            else:
                cont_map = cont_map - b_cont_map
        runner = [0] if temperature else range(len(cont_map))

        for T in runner:
            distance_map = cont_map[T, :, :]
            vmin, vmax, label, cmap = get_plot_params(distance_map)
            if double and b:
                distance_map[np.triu_indices_from(distance_map)] = b_cont_map[np.triu_indices_from(b_cont_map)]
            img = self.axis.imshow(distance_map, cmap=cmap, vmin=vmin, vmax=vmax)
            cmap.set_bad('red', alpha=0.15)

        q_total, q_plus, q_minus = self.get_charge_seq()
        for i, q in enumerate(q_total):
            if q < 0:
                self.axis.axvline(i, alpha=0.2, color='red')
            if q > 0:
                self.axis.axvline(i, alpha=0.2, color='blue')
        self.figure.subplots_adjust(left=0, right=1)
        self.axis.invert_yaxis()
        if self.chain_atoms == 448:
            self.axis.axvspan(403, 410, color='black', alpha=0.3, label='E4')
        cb = self.figure.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.08)
        cb.set_label(label, fontsize=20)
        cb.ax.tick_params(labelsize=16)
        self.axis.set_adjustable('box', True)
        return cont_map

    def plot_dijs(self, r0=5.5, plot_flory_fit=False, plot_ideal_fit=False):
        # TODO : FIX DOUBLE CALLING FIRST ONE HERE EXPLICIT
        ijs, means, err = self.ij_from_contacts(use='md')
        if plot_flory_fit:
            # TODO : SECOND CALL IS HERE IMPLICIT
            florys = self.flory_scaling_fit(use='md', r0=r0, ijs=[ijs, means, err])[0]

        self.axis.set_xlabel("|i-j|", fontsize=self.label_fsize)
        self.axis.set_ylabel(r'dij ($\AA$)', fontsize=self.label_fsize)
        for i in range(len(self.get_temperatures())):
            mean = means[i, :]
            p = self.axis.plot(ijs, mean, label=f"T={self.get_temperatures()[i]}", zorder=0)
            if plot_flory_fit:
                flory = florys[i]
                # self.axis.plot(ijs, r0 * np.array(ijs) ** flory, '--', label=f"Fit to {r0:.1f}*N^{flory:.3f}", color=self.color)
                self.axis.plot(ijs, r0 * np.array(ijs) ** flory, '--', color=p[0].get_color())
            if plot_ideal_fit:
                self.axis.plot(ijs, np.array(ijs) ** 0.5 * r0, '--', label="Random Coil Scaling (5.5*N^0.5)", color=p[0].get_color(), zorder=2)
            pline, capline, barline = self.axis.errorbar(ijs, mean, yerr=err[i, :], uplims=True, lolims=True, color=p[0].get_color(), zorder=1)
            capline[0].set_marker('_')
            capline[0].set_markersize(10)
            capline[1].set_marker('_')
            capline[1].set_markersize(10)
        return ijs, means

    def plot_rg(self):
        if self.obs_data is not None:
            rg = self.obs_data
        else:
            rg = self.rg(use='md')
        err_bars = self.block_error(observable=rg)
        self.axis.set_ylabel(r"Rg ($\AA$)", fontsize=self.label_fsize)
        self.axis.set_xlabel("T (K)", fontsize=self.label_fsize)
        p = self.axis.plot(self.temperatures, rg.mean(axis=1), self.style, label=self.label, color=self.color)
        if self.protein in self.florys:
            flory = self.florys[self.protein][0, :]
            Tc = self.find_Tc(florys=flory)
            idx_sup = np.where(self.temperatures == np.min(self.temperatures[self.temperatures > Tc]))[0][0]
            idx_inf = np.where(self.temperatures == np.max(self.temperatures[self.temperatures < Tc]))[0][0]
            slope = (rg[idx_sup, :].mean() - rg[idx_inf, :].mean())/(self.temperatures[idx_sup] - self.temperatures[idx_inf])
            intersect = rg[idx_sup, :].mean() - slope*self.temperatures[idx_sup]
            rgC = slope*Tc + intersect
            phase_params = self.axis.scatter(Tc, rgC, s=80, color='firebrick', alpha=0.5)
            # if self.protein not in self.rg_Tc_legend_helper:
            #     self.rg_Tc_legend_helper[self.protein] = [[], []]
            # self.rg_Tc_legend_helper[self.protein][1].append(f"PT {self.protein}. Tc = {Tc:.0f}, Rgc = {rgC:.0f}")
            # self.rg_Tc_legend_helper[self.protein][0].append(phase_params)
            # if self.rg_Tc_artist_helper is not None:
            #     self.rg_Tc_artist_helper.remove()
            # # leg = self.axis.legend(self.rg_Tc_legend_helper[self.protein][0], self.rg_Tc_legend_helper[self.protein][1], loc=4, fontsize=self.label_fsize)
            # # self.rg_Tc_artist_helper = leg
            # self.axis.add_artist(self.rg_Tc_artist_helper)
            # TODO : HARDCODE...
            self.axis.set_ylim(25, 95)
            ylim = self.axis.get_ylim()
            self.axis.axhspan(rgC, ylim[1], color="green", alpha=0.08)
            self.axis.axhspan(ylim[0], rgC, color="red", alpha=0.08)
        pline, capline, barline = self.axis.errorbar(self.temperatures, rg.mean(axis=1), yerr=err_bars, uplims=True, lolims=True, fmt='', color=p[0].get_color())
        capline[0].set_marker('_')
        capline[0].set_markersize(10)
        capline[1].set_marker('_')
        capline[1].set_markersize(10)
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
        p = self.axis.plot(self.temperatures, florys, self.style, label=self.label, color=self.color)
        Tc = self.find_Tc(florys=florys)
        self.axis.axvline(Tc, ls='--', alpha=0.4)

        pline, capline, barline = self.axis.errorbar(self.temperatures, florys, yerr=err, uplims=True, lolims=True, fmt='', color=p[0].get_color())
        capline[0].set_marker('_')
        capline[0].set_markersize(10)
        capline[1].set_marker('_')
        capline[1].set_markersize(10)
        self.axis.add_artist(leg)
        return np.array([florys, r0s, err])

    def plot_density_profile(self, T, axis=0):
        profs = self.density_profile(T=T)
        profs = profs.mean(axis=1)[:, axis]
        bins, data = np.histogram(profs, bins=25)
        cent_bins = (data[:-1]+data[1:])/2
        h = self.axis.hist(profs, bins=50, label=self.label, alpha=0.4)
        self.axis.plot(cent_bins, bins, '.', label=self.label, color=h[0].get_color())
        return profs

    def plot_q_distr(self, window=9):
        total, plus, minus = self.get_charge_seq(sequence=self.sequence, window=window)
        self.axis.stem(plus, markerfmt=' ', use_line_collection=True, linefmt='blue', basefmt='')
        self.axis.stem(minus, markerfmt=' ', use_line_collection=True, linefmt='red', basefmt='')
        self.axis.set_ylim(-0.6, 0.6)
        return total

    def plot_E_ensemble(self):
        # TODO INTEGRATE TO PLOT FUNCTION
        self.figure, self.axis = plt.subplots(figsize=(16, 12), sharex=True)
        data = self.get_lmp_data()
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

    def plot_gaussian_kde(self, data):
        kde_scipy = scipy.stats.gaussian_kde(data)

        x = np.linspace(data.min(), data.max(), 1000)

        self.axis.hist(data, density=True)
        self.axis.plot(x, kde_scipy(x), '--')

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

    def plot_autocorrelation(self, data, nlags=1000):
        acf = sm.tsa.stattools.acf(data, nlags=nlags)
        return acf

    def clean(self):
        self.plots = {}
        self.figure = None
        self.axis = None
