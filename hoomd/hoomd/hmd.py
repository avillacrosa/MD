import glob
import os
import re
import mdtraj as md
import math
import numpy as np
import definitions
from scipy.optimize import curve_fit


class HMD:
    def __init__(self, md_dir, **kwargs):
        self.md_dir = md_dir
        self.in_files = self._get_hmd_files()
        self.residue_dict = dict(definitions.residues)

        self.equil_frames = kwargs.get('equil_frames', 300)

        self.slab = self._is_slab()
        self.box = self.get_box()
        self.temperatures = self.get_temperatures()
        self.sequence = self._get_sequence()
        self.chain_atoms = len(self.sequence)
        self.chains = self.get_n_chains()
        self.masses = self._get_masses()
        self.structures = self.get_structures()

    def _get_hmd_files(self):
        hmd = glob.glob(os.path.join(self.md_dir, '*.py'))
        hmd.sort()
        hmd_files = []
        for h_file in hmd:
            with open(os.path.join(self.md_dir, h_file), 'r') as lmp_file:
                hmd_files.append(lmp_file.readlines())
        return hmd_files

    def _get_masses(self):
        m = []
        for chain in range(self.chains):
            for aa in self.sequence:
                m.append(self.residue_dict[aa]["mass"])
        return np.array(m)

    def _is_slab(self):
        for file in self.in_files:
            for line in file:
                if "box_resize" in line:
                    return True
        return False

    def get_box(self):
        dict = {}
        if not self.slab:
            pass
        else:
            # Todo : strong assumption
            for line in self.in_files[0]:
                if "h_c" in line and "linear_interp" in line:
                    dict["x"] = abs(float(re.findall(r'-?\d+', line)[-1]))*10.
                    dict["y"] = dict["x"]
                if "h_e" in line and "linear_interp" in line:
                    dict["z"] = abs(float(re.findall(r'-?\d+', line)[-1]))*10.
        return dict

    def _get_sequence(self):
        sequence = None
        for file in self.in_files:
            for line in file:
                if 'sequence' in line:
                    sequence = line.split()[-1].replace("\"", '')
                    break
        return sequence

    def get_n_chains(self):
        """
        Get the number of chains of the simulation
        :return: [int nchains, int atoms_per_chain]
        """
        n_chains = None
        for file in self.in_files:
            for line in file:
                if 'chains' in line:
                    n_chains = int(re.findall(r'\d+', line)[0])
                    break
        return n_chains

    def get_temperatures(self):
        temperatures = []
        for h_file in self.in_files:
            for line in h_file:
                if "temperature" in line:
                    temperatures.append(re.findall(r'\d+', line)[0])
                    break
        return temperatures

    def get_structures(self):
        dcds = glob.glob(os.path.join(self.md_dir, '*.dcd'))
        if len(dcds) > 1:
            dcds = sorted(dcds, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        structures = []
        if os.path.exists(os.path.join(self.md_dir, 'topo.pdb')):
            topo = os.path.join(self.md_dir, 'topo.pdb')
        else:
            return SystemError("Topology not found, but only looked for a file named topo.pdb")
        for dcd in dcds:
            tr = md.load(dcd, top=topo)
            structures.append(tr[self.equil_frames:])
        print(structures)
        return structures

    def density_profile(self, T=0, noise=False):

        if noise:
            random_displacement = np.random.randint(-self.box["z"] / 2 * 0.5, self.box["z"] / 2 * 0.5,
                                                    size=self.structures[0].n_frames)
            # random_displacement = np.random.randint(50, 500, size=self.structures[0].n_frames)
            # random_displacement = np.random.randint(0, self.box["z"]/2, size=self.structures[0].n_frames)
            random_displacement = np.repeat(random_displacement[:, np.newaxis], self.structures[0].n_atoms, axis=1)
            test = self.structures[T].xyz[:, :, 2] * 10. * 10. - random_displacement
        else:
            test = self.structures[T].xyz[:, :, 2] * 10. * 10.

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

        # xa, slab_bins = bin_coms(bin_size=16)
        xa, slab_bins = bin_coms(bin_size=16)

        n_lags = math.ceil(len(slab_bins) / 2)
        caa = np.zeros(shape=(xa.shape[0], n_lags * 2))
        for lag in range(0, n_lags):
            xa_mod = xa[:, lag:]
            xa_lag = xa[:, :(len(slab_bins) - lag)]
            acf = np.sum((xa_mod - xa.mean()) * (xa_lag - xa.mean()), axis=1)
            # Caa is symmetric
            caa[:, lag + n_lags] = acf / ((len(slab_bins) - lag) * np.var(xa, axis=1))
            caa[:, -lag + n_lags] = acf / ((len(slab_bins) - lag) * np.var(xa, axis=1))
        caa = caa[:, (n_lags*2-len(slab_bins)):]

        pa = np.ones_like(caa)
        pa[caa < 0.5] = 0
        pa[:, 0] = 0

        xcc = np.zeros(shape=caa.shape)

        # TODO : REVIEW THESE TWO
        # WORKS
        for lag in range(0, n_lags):
            xa_mod = xa[:, lag:]
            pa_lag = np.flip(pa, axis=1)[:, :(2 * n_lags - lag - (n_lags*2-len(slab_bins)))]
            xcc[:, lag] = np.sum(xa_mod * pa_lag, axis=1)

        for lag in range(0, n_lags):
            xa_mod = np.flip(xa, axis=1)[:, lag:]
            pa_lag = pa[:, :(2 * n_lags - lag - (n_lags*2-len(slab_bins)))]
            xcc[:, -lag] = np.sum(xa_mod * pa_lag, axis=1)

        nmaxs2 = []
        for frame in range(self.structures[T].n_frames):
            xcf = xcc[frame, :]
            # print(int(np.where(xcf == xcf.max())[0][0]))
            nmaxs2.append(int(np.where(xcf == xcf.max())[0].mean()))

        nmaxs = np.argmax(xcc, axis=1)

        x_shift = np.zeros(shape=xa.shape)
        for frame in range(xa.shape[0]):
            x_shift[frame, :] = np.roll(xa[frame, :], -nmaxs[frame], axis=0)

        z = slab_bins
        rho_z = x_shift
        return z, rho_z

    def interface_position(self, z, rho_z, cutoff=0.9):
        def tanh_fit(x, s, y0, x0):
            y = y0 - y0 * np.tanh((x + x0) * s)
            return y
        rho_z_plus = rho_z[np.where(z >= 0)][:-20]
        rho_z_minus = rho_z[np.where(z <= 0)][:-30]

        try:
            popt_plus, pcov_plus = curve_fit(tanh_fit, np.arange(0, rho_z_plus.shape[0]), rho_z_plus)
            rho_fit = tanh_fit(np.arange(0, rho_z_plus.shape[0]), *popt_plus)
            rho_interp = np.interp(np.linspace(0, rho_z_plus.shape[0], 10000),
                                   np.arange(0, rho_z_plus.shape[0]),
                                   rho_fit)
            # intf_cutoff = rho_interp.max() * 0.1
            intf_cutoff = rho_interp.max() * cutoff
            interface_idx = np.argmin(np.abs(rho_interp - intf_cutoff))
            interface_z_plus = np.linspace(0, z.max(), 10000)[interface_idx]

            int_dil_cutoff = rho_interp.max() * (1 - cutoff)
            interface_idx_dil = np.argmin(np.abs(rho_interp - int_dil_cutoff))
            interface_z_plus_dil = np.linspace(0, z.max(), 10000)[interface_idx_dil]
        except:
            interface_z_plus = 0
            interface_z_plus_dil = 0
            print("> Interface fit failed for positive z, returning 0 (no interface) !!!")
        # try:
        #     popt_minus, pcov_minus = curve_fit(tanh_fit, np.arange(0, rho_z_minus.shape[0]), np.flip(rho_z_minus))
        #     rho_fit = tanh_fit(np.arange(0, rho_z_minus.shape[0]), *popt_minus)
        #     rho_interp = np.interp(np.linspace(0, rho_z_minus.shape[0], 10000),
        #                            np.arange(0, rho_z_minus.shape[0]),
        #                            rho_fit)
        #     intf_cutoff = rho_interp.max() * cutoff
        #     interface_idx = np.argmin(np.abs(rho_interp - intf_cutoff))
        #     interface_z_minus = np.linspace(0, z.max(), 10000)[interface_idx]
        #     interface_z_minus = -interface_z_minus
        # except:
        #     interface_z_minus = 0
        #     print("> Interface fit failed for negative z, returning 0 (no interface) !!!")

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(z, rho_z)
        # plt.axvline(interface_z_plus)
        # plt.axvline(-interface_z_plus)

        return [-interface_z_plus, interface_z_plus], [-interface_z_plus_dil, interface_z_plus_dil]

    def phase_diagram(self, cutoff=0.9, full=False):
        dilute_densities = []
        condensed_densities = []
        for T in range(len(self.temperatures)):
            z, rho_z = self.density_profile(T=T, noise=False)
            # z_fit, rho_fit, tanh_fit, interface_pos = self.interface_position(rho_z=rho_z.mean(axis=0), slab_bins=z)
            interface_c, interface_d = self.interface_position(rho_z=rho_z.mean(axis=0), z=z, cutoff=cutoff)
            # diluted = ((-interface_pos > c) + (interface_pos < c)).sum(axis=1)
            # condensed = ((-interface_pos <= c) & (interface_pos >= c)).sum(axis=1)
            c = self.structures[T].xyz[:, :, 2] * 10. * 10.

            mass_helper = np.zeros_like(c)
            # mass_helper[np.where((-interface_pos < c) & (c < interface_pos))] = 1
            mass_helper[np.where((interface_c[0] < c) & (c < interface_c[1]))] = 1
            mass_condensed = np.dot(mass_helper, self.masses)

            mass_helper = np.zeros_like(c)
            mass_helper[np.where(interface_d[0] > c)] = 1
            mass_helper[np.where(interface_d[1] < c)] = 1
            mass_dilute = np.dot(mass_helper, self.masses)

            volume_condensed = ((interface_c[1] - interface_c[0]) * self.box["x"] * self.box["y"])
            # volume_dilute = self.box["x"] * self.box["y"] * self.box["z"] - volume_condensed
            volume_dilute = self.box["x"] * self.box["y"] * self.box["z"] - ((interface_d[1] - interface_d[0]) * self.box["x"] * self.box["y"])
            # dilute_densities.append(diluted*mass_dilute/volume_dilute)
            if full:
                dilute_densities.append((mass_dilute / volume_dilute))
                condensed_densities.append((mass_condensed / volume_condensed))
            else:
                dilute_densities.append((mass_dilute / volume_dilute).mean())
                condensed_densities.append((mass_condensed / volume_condensed).mean())
            # condensed_densities.append(condensed*mass_condensed/volume_condensed)

        dilute_densities = np.array(dilute_densities)
        condensed_densities = np.array(condensed_densities)

        return dilute_densities, condensed_densities
