#!/home/adria/anaconda3/bin/python

import numpy as np
import math
import sys
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy import signal
import seaborn as sns
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf
import statsmodels
import matplotlib.pyplot as plt
import os

sys.path.insert(1, '../utils/')
import conversion
import lmpmath
import calc
import definitions

xtc = '/home/adria/perdiux/prod/lammps/dignon/cpeb4d4/A-syn_HPS_100000_frames.atom'
nrg = '/home/adria/perdiux/prod/lammps/dignon/cpeb4d4/obs.data'


def plot_rgs(dir, equil=4000000):
    rg_frame, Is = calc.rg_from_lmp(dir, equil)
    rg_frame = np.array(rg_frame)
    rgs = rg_frame.mean(1)
#     block_l = 5000
#     for k in range(1, 5):
#         # print(block_l*k)
#         # rgs = []
#         if block_l*k > rg_frame.shape[1]:
#             rg_bloc = rg_frame[:, block_l*(k-1):rg_frame.shape[1]]
#  #           rg_bloc = rg_bloc[:,::10]
#             plt.plot(Is, rg_bloc.mean(1), label=f'{block_l * (k - 1)} to {rg_frame.shape[1]}')
#         else:
#             rg_bloc = rg_frame[:, block_l*(k-1):block_l*k]
# #            rg_bloc = rg_bloc[:,::10]
#             plt.plot(Is, rg_bloc.mean(1), label=f'{block_l*(k-1)} to {block_l*(k)}')
#         print(rg_bloc.shape)
            # rgs.append(rg_bloc)
        # print(np.array(rgs).shape)
        # plt.plot(Is, np.array(rgs).mean(1))
    # plt.plot(Is, rgs, '.-')
    return Is, rgs


def plot_fit(dir, ref_dir=None, delta4=True, charged=True):
    pdb = conversion.getPDBfromLAMMPS(os.path.join(dir, 'hps'))
    xtc = os.path.join(dir, 'hps_traj.xtc')
    contacts = calc.contact_map(xtc, pdb)
    ijs, means = calc.ij_from_contacts(contacts)
    cb_label = r'd ($\AA$)'

    fit, fitv = curve_fit(lmpmath.flory_scaling, ijs, means)
    fit_flory = fit[0]
    fit_r0 = fit[1]
    # fit_r0 = 15

    plt.figure(num=None, figsize=(8, 6), frameon=False)
    plt.xlabel("|i-j|")
    plt.ylabel(r'dij ($\AA$)')
    plt.plot(ijs, fit_r0 * np.array(ijs) ** fit_flory, '--', label=f"Fit to {fit_r0:.1f}*N^{fit_flory:.3f}")
    plt.plot(ijs, np.array(ijs) ** 0.5 * 3.8, '--', label="Random Coil Scaling (3.8*N^0.5)")
    plt.plot(ijs, means, label="HPS results")
    plt.legend()
    fout = f'../default_output/refit-fit-4.png'
    Iplot = lmpmath.I_from_debye([float(os.path.basename(dir))], from_angst=True)[0]*10**3
    Iplot = int(round(Iplot, 0))
    plt.title(f"I = {Iplot} mM", fontsize=26)
    plt.savefig(fout)
    return fit_flory, Iplot


def plot_contact_map(dir, ref_dir=None, delta4=True, charged=True):
    pdb = conversion.getPDBfromLAMMPS(os.path.join(dir, 'hps'))
    xtc = os.path.join(dir, 'hps_traj.xtc')
    contacts = calc.contact_map(xtc, pdb)
    ijs, means = calc.ij_from_contacts(contacts)
    cb_label = r'd ($\AA$)'

    if ref_dir is not None:
        pdb_ref = conversion.getPDBfromLAMMPS(os.path.join(ref_dir, 'hps'))
        xtc_ref = os.path.join(ref_dir, 'hps_traj.xtc')
        ref_contacts = calc.contact_map(xtc_ref, pdb_ref)
        contacts = np.array(contacts) - np.array(ref_contacts)
        cb_label = r'd_ij - d_ij(l=0,q=0) ($\AA$)'

    fit, fitv = curve_fit(lmpmath.flory_scaling, ijs, means)
    fit_flory = fit[0]
    fit_r0 = fit[1]
    # fit_r0 = 15

    plt.figure(num=None, figsize=(8, 6), frameon=False)
    plt.xlabel("|i-j|")
    plt.ylabel(r'dij ($\AA$)')
    plt.plot(ijs, fit_r0 * np.array(ijs) ** fit_flory, '--', label=f"Fit to {fit_r0:.1f}*N^{fit_flory:.3f}")
    plt.plot(ijs, np.array(ijs) ** 0.5 * 3.8, '--', label="Random Coil Scaling (3.8*N^0.5)")
    plt.plot(ijs, means, label="HPS results")
    plt.legend()
    fout = f'../default_output/RE-fit-WT-{os.path.basename(dir)}-aas_map.png'
    Iplot = lmpmath.I_from_debye([float(os.path.basename(dir))], from_angst=True)[0]*10**3
    Iplot = int(round(Iplot, 0))
    plt.title(f"I = {Iplot} mM", fontsize=26)
    plt.savefig(fout)

    fig = plt.figure(num=None, figsize=(16, 12), frameon=False)
    ax = plt.gca()
    # img = ax.imshow(contacts, cmap='PRGn', vmin=-9, vmax=9)
    img = ax.imshow(contacts, cmap='PRGn')
    fig.subplots_adjust(left=0, right=1)
    # fig.tight_layout(pad=0)
    ax.invert_yaxis()

    if charged:
        charged_plus, charged_minus = calc.charged_positions(cpeb47d_seq)
        labelled = False
        for site in charged_plus:
            if not labelled:
                ax.axvline(site, alpha=0.35, color='red', label='+')
                labelled = True
            else:
                ax.axvline(site, alpha=0.35, color='red')
            # ax.axhline(site, alpha=0.3, color='red')
        labelled = False
        for site in charged_minus:
            if not labelled:
                ax.axvline(site, alpha=0.35, color='blue', label='-')
                labelled = True
            else:
                ax.axvline(site, alpha=0.35, color='blue')
            # ax.axhline(site, alpha=0.3, color='blue')

    if delta4:
        ax.axvspan(definitions.cpeb4d4_sites[0], definitions.cpeb4d4_sites[1], color='orange', alpha=0.5, label='E4')
        # ax.axhspan(definitions.cpeb4d4_sites[0], definitions.cpeb4d4_sites[1], color='orange', alpha=0.7)

    cb = plt.colorbar(img, orientation='horizontal', fraction=0.046, pad=0.08)
    cb.set_label(cb_label)
    cb.ax.tick_params(labelsize=16)
    # img.set_clim(-35, 20)
    plt.legend(loc='upper left')
    fout = f'../default_output/RE-WT-{os.path.basename(dir)}-aas_map.png'
    print(f'Saving figurat {fout}')
    plt.title(f"I = {Iplot} mM", fontsize=26)
    plt.savefig(fout)
    return fit_flory, Iplot


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


wd = '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4'
contact_wd = '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.051'
rc_wd = '/home/adria/perdiux/prod/lammps/dignon/REFS/RC/r6'
cpeb4_seq = 'MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE'

cpeb47d    = [18, 38, 40, 97, 252, 255, 353]
cpeb412d   = [18, 38, 40, 97, 252, 255, 326, 330, 332, 353, 359, 364]

cpeb47d_seq = []
cpeb412d_seq = []
for i, aa in enumerate(cpeb4_seq):
    if i+1 in cpeb47d:
        cpeb47d_seq.append('D')
    else:
        cpeb47d_seq.append(aa)
    if i+1 in cpeb412d:
        cpeb412d_seq.append('D')
    else:
        cpeb412d_seq.append(aa)

cpeb47d_seq = ''.join(cpeb47d_seq)
cpeb412d_seq = ''.join(cpeb412d_seq)
# refIs, refrgs = plot_rgs('/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4')
# Is, rgs = plot_rgs('/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4')
# plt.plot(refIs, refrgs, '-', label='100ns run')
# plt.plot(Is, rgs, '-', label='25ns run')
# plt.legend()
# plt.ylabel('Rg')
# plt.xlabel('I (mM)')
# plt.title('Lambda_ij = 0')
# plt.savefig('../default_output/double_l0_rg.png')
# plt.show()
# Is, rgs = plot_rgs('/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4')
# plt.plot(Is_ref, rgs_ref, '.-', label='100ns (19269 frames)')
# plt.plot(Is, rgs, '.-', label='25ns (4269 frames)')
# plt.xlabel('I (mM)')
# plt.ylabel(r'Rg ($\AA$)')
# plt.title('Lambda_ij = 0')
# plt.legend()
# plt.savefig('../default_output/l0.png')
# plt.show()

# dirs = conversion.getLMPdirs(['/home/adria/perdiux/prod/lammps/dignon/I_cpeb4'])
wt_wd = ['/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.051',
        '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.103',
        '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.136',
        '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.162',
        '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.178',
        '/home/adria/perdiux/prod/lammps/dignon/I_cpeb4/0.205']

l0_wds = ['/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.051',
        '/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.103',
        '/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.136',
        '/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.162',
        '/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.178',
        '/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.205']

d7_wds = ['/home/adria/perdiux/prod/lammps/dignon/7D-I_cpeb4/0.051',
        '/home/adria/perdiux/prod/lammps/dignon/7D-I_cpeb4/0.103',
        '/home/adria/perdiux/prod/lammps/dignon/7D-I_cpeb4/0.136',
        '/home/adria/perdiux/prod/lammps/dignon/7D-I_cpeb4/0.162',
        '/home/adria/perdiux/prod/lammps/dignon/7D-I_cpeb4/0.178',
        '/home/adria/perdiux/prod/lammps/dignon/7D-I_cpeb4/0.205']

d12_wds = ['/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.051',
        '/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.103',
        '/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.136',
        '/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.162',
        '/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.178',
        '/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.205']

rc_wd_REF = '/home/adria/perdiux/prod/lammps/dignon/REFS/RC/r6'
q0l0_wd_REF = '/home/adria/perdiux/prod/lammps/dignon/REFS/q0l0-I_cpeb4/0.051'
# l0_wds_REF = [
#         '/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4/0.051', 
#         '/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4/0.103',
#         '/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4/0.136',
#         '/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4/0.162',
#         '/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4/0.178',
#         '/home/adria/perdiux/prod/lammps/dignon/REFS/l0-I_cpeb4/0.205'
# ]

# plot_aa_map('/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.103', cpeb412d_seq, wt_wd[1])
plot_aa_map('/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.103', cpeb412d_seq)
# plot_aa_map('/home/adria/perdiux/prod/lammps/dignon/l0-I_cpeb4/0.103', cpeb4_seq)
# plot_contact_map('/home/adria/perdiux/prod/lammps/dignon/12D-I_cpeb4/0.103')
#
# flory = []
# Is = []
#
# for i, dir in enumerate(wt_wd):
#     print("="*40)
#     print(f"DOING DIR {dir}")
#     print("="*40)
#     floris, I = plot_fit(dir, rc_wd_REF)
#     flory.append(floris)
#     Is.append(I)
# plt.plot(Is, flory, '.-', label='WT')
# flory = []
# for i, dir in enumerate(l0_wds):
#     print("="*40)
#     print(f"DOING DIR {dir}")
#     print("="*40)
#     floris, I = plot_fit(dir, q0l0_wd_REF)
#     flory.append(floris)
# plt.plot(Is, flory, '.-', label='WT with HPS l=0')
# flory = []
# for i, dir in enumerate(d7_wds):
#     print("="*40)
#     print(f"DOING DIR {dir}")
#     print("="*40)
#     floris, I = plot_fit(dir, wt_wd[i])
#     flory.append(floris)
# plt.plot(Is, flory, '.-', label='7D')
# flory = []
# for i, dir in enumerate(d12_wds):
#     print("="*40)
#     print(f"DOING DIR {dir}")
#     print("="*40)
#     floris, I = plot_fit(dir, wt_wd[i])
#     flory.append(floris)
# plt.plot(Is, flory, '.-', label='12D')
# plt.xlabel('Flory scaling')
# plt.ylabel('I (mM)')
# plt.legend()
# plt.savefig('../default_output/multi_floris.png')

# plt.xlabel('I (mM)')
# plt.ylabel("Flory Scaling")
# plt.plot(Is, flory, '.-')
# fiout = '../default_output/wtflory_is.png'
# print(f'SAVING AT {fiout}')
# plt.savefig(fiout)
# plot_contact_map(ref_contact_wd)
# plot_aa_map(contact_wd, cpeb4_seq)
# plot_rgs(wd)
# plot_gaussian_kde(contact_wd)
# rgs, Is = calc.rg_from_lmp(ref_contact_wd)
# print(np.array(rgs).mean())
# print(3.8/math.sqrt(6)*448**0.5)


# ------------------------------------------------- AUTOCORR -----------------------------------------------------------
# dirs = ['/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr_l0/dt1', '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr_l0/dt100',
#         '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr_l0/dt1000', '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr_l0/dt2000',
#         '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr_l0/dt5000', '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr_l0/dt10000']
# plt.figure(num=None, figsize=(8, 6))
# for dir in dirs:
#     rgs, acf = plot_autocorrelation(dir, equil=0)
#     plt.plot(acf, label=os.path.basename(dir))
# plt.show()
# plt.legend()
# plt.xlabel('Lags')
# plt.ylabel('Autocorrelation')
# plt.title('ACF for different timesteps for l=0')
# plt.savefig('../default_output/l0_autocorr.png')

#-----------------------------------------------------------------------------------------------------------------------
# dirs = conversion.getLMPdirs(['/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr'])
# dirs.sort()
# dirs = ['/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr/dt1', '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr/dt100',
#         '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr/dt1000', '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr/dt2000',
#         '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr/dt5000', '/home/adria/perdiux/prod/lammps/dignon/TEST/autocorr/dt10000']
#
# plt.figure(num=None, figsize=(8, 6))
# for dir in dirs:
#     rgs, acf = plot_autocorrelation(dir, equil=0)
#     plt.plot(acf, label=os.path.basename(dir))
# plt.xlabel('Lags')
# plt.ylabel('Autocorrelation')
# plt.title('ACF(Rgs) for different dt\'s')
# plt.legend()
# plt.savefig('../default_output/acf_rg.png')
# plt.show()
