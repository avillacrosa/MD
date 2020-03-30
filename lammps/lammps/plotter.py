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
    def __init__(self, force_recalc=False, max_frames=1000):
        """
        Just a helper for plotting. Many things (paths and what not) are really arbitrary.
        :param force_recalc: bool, Useless atm
        """
        super().__init__(oliba_wd=None, max_frames=max_frames)
        self.oliba_prod_dir = '/home/adria/data/prod/lammps'
        self.oliba_data_dir = '/home/adria/data/data'
        self.index = self.make_index()

        self.figure, self.axis = None, None
        self.figure2, self.axis2 = None, None
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

        self.observables = ['rg', 'distance_map', 'contact_map', 'dij', 'flory', 'charge', 'rho', 'rg_distr']
        self.force_recalc = force_recalc

        self.rg_Tc_legend_helper = {}
        self.rg_Tc_artist_helper = None

    def make_index(self, force_update=True):
        """
        Generate pandas dataframe that serves as an index so that we can see at a glance all available runs and don't
        have to write again and again every path.
        :param force_update: bool, useless atm
        :return: dataframe, contains some parameters of a given run, along its path and what protein
        """
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
            # TODO, get protein name from lmp.protein ?
            protein = os.path.normpath(d).split(os.sep)[6]
            prots.append(protein)
            eps, I, scale = None, None, None
            lj_lambda_line = None
            lmp_path = glob.glob(os.path.join(d, 'lmp.lmp'))
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
                        Is.append(int(I * 10 ** 3))
                        lj_lambda_line = None
                    if "pair_coeff" in line and "1" in line and scale is None:
                        ref = self.residue_dict["A"]["lambda"]
                        test = re.findall(r'\d+\.?\d*', line)[4]
                        scale = float(test) / float(ref)
                        ss.append(round(scale, 2))
                    if eps and I and scale:
                        break
        df["Protein"] = prots
        df["I"] = Is
        df["Eps"] = epss
        df["Scale"] = ss
        # TODO: Remove "Name" ? Seems pretty useless
        df["Name"] = paths
        df["FullPath"] = lmp_dirs
        df.to_csv(os.path.join(definitions.module_dir, 'index.txt'), sep=' ', mode='w')
        return df

    def plot(self, observable, index, plot_id, **kwargs):
        """
        Main function of the class. Essentially given an observable, attempt to plot it for one entry of the index
        dataframe (self.make_index)
        :param observable: string, one of the available observables that can be plotted (check self.observables)
        :param index: int, index of the dataframe containing the desired run
        :param plot_id: free, name that will be given to the current plot. Useful when plotting different things at once
        :param kwargs: kwargs, many things, most of the specific to a given observable
        :return: data, data being plotted, in whatever shape (depends on observable)
        """
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
            with open(os.path.join(os.path.join(definitions.hps_data_dir, 'sequences'), f'CPEB4.seq')) as f:
                self.sequence = f.readlines()[0]

        def_label = f'{protein}, I = {ionic_strength:.0f}, ε = {eps:.0f} HPS = {ls:.1f}'
        self.label = kwargs["label"] if "label" in kwargs else def_label
        self.style = kwargs["style"] if "style" in kwargs else '-'
        self.color = kwargs["color"] if "color" in kwargs else None

        # fout = f'{ls:.1f}ls-{ionic_strength:.0f}I-{eps:.0f}e'
        # d = os.path.join(self.oliba_data_dir, observable, protein, fout)
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
            data = self.plot_rg(**kwargs)
            self.gyrations[protein] = data
        if observable == 'distance_map' or observable == 'contact_map':
            self.figure2, self.axis2 = plt.subplots(figsize=(16, 12), sharex='all')
            data = self.plot_distance_map(**kwargs)[0]
            self.contact_maps[protein] = data
            self.axis2.legend(fontsize=self.label_fsize)
            self.axis2.xaxis.set_tick_params(labelsize=self.ticks_fsize)
            self.axis2.yaxis.set_tick_params(labelsize=self.ticks_fsize)
        if observable == 'dij':
            data = self.plot_dijs(plot_flory_fit=True, plot_ideal_fit=False)[1]
            self.dijs[protein] = data
        if observable == 'flory':
            data = self.plot_flory(**kwargs)
        if observable == 'charge':
            data = self.plot_q_distr(**kwargs)
        if observable == 'rg_distr':
            data = self.plot_rg_distr(T=5, **kwargs)
        if observable == 'rho':
            temperature = kwargs["temperature"] if "temperature" in kwargs else 0
            data = self.plot_density_profile(T=temperature)
        self.axis.legend(fontsize=self.label_fsize)
        self.axis.xaxis.set_tick_params(labelsize=self.ticks_fsize)
        self.axis.yaxis.set_tick_params(labelsize=self.ticks_fsize)
        # if self.obs_data is None:
        #     pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        #     np.savetxt(os.path.join(d, 'data.txt'), data)
        self.figure.savefig(os.path.join(definitions.module_dir, f'temp/plot_{plot_id}.png'))
        return data

    def plot_distance_map(self, **kwargs):
        """
        Plot a distance map using imshow.
        :param kwargs: Defined below
        :return: ndarray[n_atoms, n_atoms]
        """
        contacts = kwargs["contacts"] if "contacts" in kwargs else False
        temperature = kwargs["temperature"] if "temperature" in kwargs else None
        double = kwargs["double"] if "double" in kwargs else False
        log = kwargs["log"] if "log" in kwargs else False
        frac = kwargs["frac"] if "frac" in kwargs else False
        vmin = kwargs["vmin"] if "vmin" in kwargs else None
        vmax = kwargs["vmax"] if "vmax" in kwargs else None
        cmap_label = kwargs["cmap_label"] if "cmap_label" in kwargs else None
        b = self.index.iloc[kwargs["index2"]]["FullPath"] if "index2" in kwargs else None

        def_cmap = 'PRGn' if b else 'plasma'
        cmap = kwargs["cmap"] if "cmap" in kwargs else def_cmap
        cmap = matplotlib.cm.get_cmap(cmap)

        if frac and log:
            raise SystemExit("Can't do both log and frac. Choose one")
        if log and not b or frac and not b:
            raise SystemExit("log/frac plots only available for 2 proteins")

        # TODO : ONLY TAKING INTERCHAIN INTO ACCOUNT HERE
        # cont_map = self.inter_distance_map(contacts=contacts, temperature=temperature)[
        if self.chains != 1:
            # cont_map = self.async_inter_distance_map(contacts=contacts, temperature=temperature)[1]
            cont_map, extra_cont_map = self.inter_distance_map(contacts=contacts, temperature=temperature)
        else:
            cont_map = self.intra_distance_map(contacts=contacts, temperature=temperature)
            cont_map = cont_map.mean(axis=1)

        if b:
            b_obj = analysis.Analysis(oliba_wd=b)
            b_cont_map = b_obj.async_inter_distance_map(contacts=contacts,
                                                  temperature=temperature) if inter else b_obj.intra_distance_map(
                contacts=contacts, temperature=temperature)
            b_cont_map = b_cont_map.mean(axis=1)

            if self.protein == 'CPEB4' and b_obj.protein == 'CPEB4_D4':
                cont_map = np.delete(cont_map, np.arange(402, 410), axis=1)
                cont_map = np.delete(cont_map, np.arange(402, 410), axis=2)

            if self.protein == 'CPEB4_D4' and b_obj.protein == 'CPEB4':
                b_cont_map = np.delete(b_cont_map, np.arange(402, 410), axis=1)
                b_cont_map = np.delete(b_cont_map, np.arange(402, 410), axis=2)
            if log:
                cont_map = np.log(cont_map / b_cont_map)
                cont_map[np.isnan(cont_map)] = 0.
            elif frac:
                cont_map = cont_map / b_cont_map
            else:
                cont_map = cont_map - b_cont_map
        runner = [0] if temperature else range(len(cont_map))

        for T in runner:
            distance_map = cont_map[T, :, :]
            if double and b:
                distance_map[np.triu_indices_from(distance_map)] = b_cont_map[np.triu_indices_from(b_cont_map)]
            img = self.axis.imshow(distance_map, cmap=cmap, vmin=0, vmax=0.005)
            if self.chains != 1:
                extra_distance_map = extra_cont_map[T, :, :]
                img2 = self.axis2.imshow(extra_distance_map, cmap=cmap, vmin=0, vmax=0.005)
            cmap.set_bad('red', alpha=0.15)

        if b:
            if len(b_obj.sequence) < len(self.sequence):
                q_total,  q_plus, q_minus = b_obj.get_charge_seq()
            else:
                q_total, q_plus, q_minus = self.get_charge_seq()
        else:
            q_total, q_plus, q_minus = self.get_charge_seq()

        for i, q in enumerate(q_total):
            if q < 0:
                self.axis.axvline(i, alpha=0.2, color='red')
            if q > 0:
                self.axis.axvline(i, alpha=0.2, color='blue')
        self.figure.subplots_adjust(left=0, right=1)
        self.axis.invert_yaxis()
        if b:
            if self.chain_atoms == 448 and b_obj.chain_atoms == 448:
                self.axis.axvspan(403, 410, color='black', alpha=0.3, label='E4')
        elif self.chain_atoms == 448:
            self.axis.axvspan(403, 410, color='black', alpha=0.3, label='E4')
        cb = self.figure.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.08)
        cb.set_label(cmap_label, fontsize=16)
        cb.ax.tick_params(labelsize=16)
        self.axis.set_adjustable('box', True)

        for i, q in enumerate(q_total):
            if q < 0:
                self.axis2.axvline(i, alpha=0.2, color='red')
            if q > 0:
                self.axis2.axvline(i, alpha=0.2, color='blue')
        self.figure2.subplots_adjust(left=0, right=1)
        self.axis2.invert_yaxis()
        if b:
            if self.chain_atoms == 448 and b_obj.chain_atoms == 448:
                self.axis2.axvspan(403, 410, color='black', alpha=0.3, label='E4')
        elif self.chain_atoms == 448:
            self.axis2.axvspan(403, 410, color='black', alpha=0.3, label='E4')
        cb = self.figure2.colorbar(img2, orientation='horizontal', fraction=0.046, pad=0.08)
        cb.set_label(cmap_label, fontsize=16)
        cb.ax.tick_params(labelsize=16)
        self.axis2.set_adjustable('box', True)
        return cont_map

    def plot_dijs(self, r0=5.5, plot_flory_fit=False, plot_ideal_fit=False):
        """
        Plot the d(i-j) as calculated from the analysis library
        :param r0: float, r0 used to perform the flory scaling fit
        :param plot_flory_fit: bool, plot the fit
        :param plot_ideal_fit: bool, plot also the "ideal chain" scaling (r0*abs(i-j)**0.5)
        :return: list[n_atoms], list[n_atoms], abs(i-j), d(abs(i-j))
        """
        # TODO : FIX DOUBLE CALLING FIRST ONE HERE EXPLICIT
        ijs, means, err = self.ij_from_contacts()
        if plot_flory_fit:
            # TODO : SECOND CALL IS HERE IMPLICIT
            florys = self.flory_scaling_fit(r0=r0, ijs=[ijs, means, err])[0]

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
                self.axis.plot(ijs, np.array(ijs) ** 0.5 * r0, '--', label="Random Coil Scaling (5.5*N^0.5)",
                               color=p[0].get_color(), zorder=2)
            pline, capline, barline = self.axis.errorbar(ijs, mean, yerr=err[i, :], uplims=True, lolims=True,
                                                         color=p[0].get_color(), zorder=1)
            capline[0].set_marker('_')
            capline[0].set_markersize(10)
            capline[1].set_marker('_')
            capline[1].set_markersize(10)
        return ijs, means

    def plot_rg(self, **kwargs):
        """
        Plot rg against temperature for a given LAMMPS run.
        :param kwargs:
        :return:
        """
        # if self.obs_data is not None:
        #     rg = self.obs_data
        # else:
        rg = self.rg(use='md')
        err_bars = self.block_error(observable=rg)
        self.axis.set_ylabel(r"Rg ($\AA$)", fontsize=self.label_fsize)
        self.axis.set_xlabel("T (K)", fontsize=self.label_fsize)
        p = self.axis.plot(self.temperatures, rg.mean(axis=1), self.style, label=self.label, color=self.color)
        if self.protein in self.florys:
            flory = self.florys[self.protein][0, :]
            if min(flory) < 0.5 and max(flory) > 0.5:
                Tc = self.find_Tc(florys=flory)
                idx_sup = np.where(self.temperatures == np.min(self.temperatures[self.temperatures > Tc]))[0][0]
                idx_inf = np.where(self.temperatures == np.max(self.temperatures[self.temperatures < Tc]))[0][0]
                slope = (rg[idx_sup, :].mean() - rg[idx_inf, :].mean()) / (
                            self.temperatures[idx_sup] - self.temperatures[idx_inf])
                intersect = rg[idx_sup, :].mean() - slope * self.temperatures[idx_sup]
                rgC = slope * Tc + intersect
                # TODO : HARDCODE...
                self.axis.set_ylim(25, 95)
                ylim = self.axis.get_ylim()
                self.axis.axhspan(rgC, ylim[1], color="green", alpha=0.08)
                self.axis.axhspan(ylim[0], rgC, color="red", alpha=0.08)
        pline, capline, barline = self.axis.errorbar(self.temperatures, rg.mean(axis=1), yerr=err_bars, uplims=True,
                                                     lolims=True, fmt='', color=p[0].get_color())
        capline[0].set_marker('_')
        capline[0].set_markersize(10)
        capline[1].set_marker('_')
        capline[1].set_markersize(10)
        return rg

    def plot_rg_distr(self, T, **kwargs):
        """
        Plot the radius of gyration distribution for a given T index
        :param T: int, T index
        :param kwargs:
        :return: ndarray[T, frames], radius of gyration
        """
        bins = kwargs["bins"] if "bins" in kwargs else 'auto'
        alpha = kwargs["alpha"] if "alpha" in kwargs else 0.5
        rg = self.rg(use='md')[T, :]
        print(rg.shape)

        kde_scipy = scipy.stats.gaussian_kde(rg)

        x = np.linspace(rg.min(), rg.max(), 1000)

        p = self.axis.plot(x, kde_scipy(x), '--', color=self.color)
        self.axis.hist(rg, bins=bins, alpha=alpha, density=True, label=self.label, color=p[0].get_color())
        self.axis.set_ylabel("P(Rg)", fontsize=self.label_fsize)
        self.axis.set_xlabel("Rg", fontsize=self.label_fsize)
        self.axis.set_xlim(0,200)
        return rg

    def plot_flory(self, r0=5.5, **kwargs):
        """
        Plot the flory scaling exponents against temperature
        :param r0: float, kuhn's length to be used for flory scaling fit
        :param kwargs: kwargs, currently unused
        :return: ndarray[T, florys, r0s, err]
        """
        # if self.obs_data is not None:
        #     # TODO : RECALCULATE ERR INSTEAD OF SAVING ?
        #     florys, r0s, err = self.obs_data
        # else:
        florys, r0s, err = self.flory_scaling_fit(use='md', r0=r0)
        self.axis.set_ylabel('\u03BD', fontsize=self.label_fsize)
        self.axis.set_xlabel("T (K)", fontsize=self.label_fsize)
        self.axis.set_ylim(0.3, 0.65)
        self.axis.axhspan(0.5, 0.65, color="green", alpha=0.08)
        self.axis.axhspan(0.3, 0.5, color="red", alpha=0.08)
        good_line = self.axis.axhline(3 / 5, ls='--', color='green')
        bad_line = self.axis.axhline(1 / 3, ls='--', color="red")
        leg = self.axis.legend([good_line, bad_line], ["Good Solvent", "Poor Solvent"], loc=2,
                               fontsize=self.label_fsize)
        p = self.axis.plot(self.temperatures, florys, self.style, label=self.label, color=self.color)
        if min(florys) < 0.5 and max(florys) > 0.5:
            Tc = self.find_Tc(florys=florys)
            self.axis.axvline(Tc, ls='--', alpha=0.4)

        pline, capline, barline = self.axis.errorbar(self.temperatures, florys, yerr=err, uplims=True, lolims=True,
                                                     fmt='', color=p[0].get_color())
        capline[0].set_marker('_')
        capline[0].set_markersize(10)
        capline[1].set_marker('_')
        capline[1].set_markersize(10)
        self.axis.add_artist(leg)
        return np.array([florys, r0s, err])

    def plot_density_profile(self, T, axis=0):
        """
        Plot the density profile along a given axis (0=x, 1=y, 2=z) and do the mean for all frames. The process is
        basically constructing an histogram and plotting it along such axis
        :param T: int, temperature index
        :param axis: int, 0 1 or 2 as described on the summary
        :return: ndarray[T,frames,chains,3], center of mass for each chain
        """
        profs = self.density_profile(T=T)
        profs = profs.mean(axis=1)[:, axis]
        bins, data = np.histogram(profs, bins=25)
        cent_bins = (data[:-1] + data[1:]) / 2
        h = self.axis.hist(profs, bins=50, label=self.label, alpha=0.4)
        self.axis.plot(cent_bins, bins, '.', label=self.label, color=h[0].get_color())
        return profs

    def plot_q_distr(self, window=9):
        """
        Currently untested. The idea is to plot the "windowed" charge distribution
        :param window:
        :return:
        """
        total, plus, minus = self.get_charge_seq(sequence=self.sequence, window=window)
        self.axis.stem(plus, markerfmt=' ', use_line_collection=True, linefmt='blue', basefmt='')
        self.axis.stem(minus, markerfmt=' ', use_line_collection=True, linefmt='red', basefmt='')
        self.axis.set_ylim(-0.6, 0.6)
        return total

    def plot_E_ensemble(self):
        """
        Currently broken
        :return:
        """
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
            x = np.linspace(min(E[T]) - 5000, max(E[T]) + 5000, 1000)
            self.axis.fill_between(x=x, y1=np.zeros(shape=x.shape), y2=kde_scipy(x), zorder=2, alpha=0.5)
            self.axis.hist(E[T], density=True, label=f'T = {self.temperatures[T]:.0f}', bins=80)
            self.axis.plot(x, kde_scipy(x))
        self.axis.legend()

    # TODO !!!! Still need to be translated, but they are pretty niche atm !!!!
    def plot_aa_map(dir, seq, ref_dir=None):
        """
        Currently broken
        :param seq:
        :param ref_dir:
        :return:
        """
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

    def clean(self):
        """
        This has to be deprecated as it shouldn't be needed, but it currently is due to bad coding...
        :return:
        """
        self.plots = {}
        self.figure = None
        self.axis = None
