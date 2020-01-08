#! /usr/bin/env python3
import sys

if '/home/adria/scripts/profasi/libs/' not in sys.path:
    sys.path.insert(1, '/home/adria/scripts/profasi/libs/')

import utils as pc
import os
import random
import numpy as np
import pandas as pd
import MDAnalysis
from DEERpredict.PREPrediction5 import PREPrediction

# @oliba
fits_out_dir = '/home/adria/data/pres/asyn/fits'
alph_out_dir = '/home/adria/data/pres/asyn/alphas'
exp_data_dir = '/home/adria/data/pres/asyn/exp_pre'
calc_data_dir = '/home/adria/data/pres/asyn/calc_pre'
frames_dir = '/home/adria/data/pres/asyn/frames'
sim_data_dir = '/home/adria/perdiux/prod/profasi/asyn/pre/asyn100k'
max_frames = 100000

# READING THE EXPERIMENTAL DATA
labels = [18, 20, 24, 42, 62, 76, 85, 87, 90, 103, 120, 140]
data_exp = pc.read_pre_exp_data(exp_data_dir, labels, 'exp')

# GET THE INITIAL ENERGIES FROM THE OUTPUT FILE (FROM THE INITIAL TRAJECTORY)
ene_0 = pc.get_profasi_energies(source=os.path.join(sim_data_dir, "results", "rt"), max_frames=max_frames)
ene_0 = np.sum(ene_0, axis=1)

ene_0_train = ene_0[0::2]
ene_0_val_1 = ene_0[1::2]
data_calc_0 = pc.read_calc_pre(calc_data_dir, labels, '1_100000-res-')
data_calc_1 = pc.read_calc_pre(calc_data_dir, labels, '2_100000-res-')

repetitions = 100
max_iter = 200
kT = 1.9872e-3 * 300

for repe in range(repetitions):
    for val_label in labels:
        print("="*50)
        print('-> Doing label', val_label, 'Repetition', repe, )
        print("="*50)

        old_tau_c = new_tau_c = 1e-9
        w_1 = np.ones(100000)/100000                # 100k = 50k even + 50k odd

        train_labels = labels.copy()
        train_labels.remove(val_label)              # Reduced labels

        data_train = data_calc_0[train_labels]      # Even Calc PRE's with reduced labels (TRAINING SET)
        data_exp_train = data_exp[train_labels]     # Experimental PRE's with reduced labels (EXP TRAINING SET)
        data_val_1 = data_calc_1[train_labels]      # Odd  Calc PRE's with reduced labels (VALIDATION 1)
        data_val_2 = data_calc_0[[val_label]]       # Even Calc PRE's with excluded label (VALIDATION 2)
        data_exp_val_2 = data_exp[[val_label]]      # Experimental PRE with excluded label (EXP VALIDATION 2)

        old_alphas = np.ones(8)                     # !!
        new_alphas = old_alphas.copy()

        fit_old_train = pc.fit_pre(data_exp_train, data_train)      # Fit Exp Training Set to Calculated Training Set
        fit_old_val_1 = pc.fit_pre(data_exp_train, data_val_1)      # Fit Exp Training Set to Val Set 1 (REDUCED LABELS)
        fit_old_val_2 = pc.fit_pre(data_exp_val_2, data_val_2)      # Fit Exp Val Set to Val Set 2 (EXCLUDED LABELS)

        fit_train_history = []
        fit_val_1_history = []
        fit_val_2_history = []
        neff_history = []
        tauc_history = []

        for i in range(1, max_iter + 1):
            change_alphas = random.choice([True, True, True, False])
            if change_alphas:   # Randomly change an alpha
                # Most impactful alphas for synuclein according to profasi is 4th if starting from 0 (HBMM)
                j = np.random.randint(0, len(old_alphas))           # Take 1 random alpha
                new_alphas[j] += np.random.normal(0, 0.01)       # Add small gaussian noise to alpha
                ene_1_all = pc.get_profasi_energies(source=os.path.join(sim_data_dir, "results", "rt"), max_frames=max_frames)
                ene_1 = []
                for k in range(ene_1_all.shape[0]):
                    ene_1.append(np.dot(ene_1_all[k], new_alphas))
                ene_1 = np.array(ene_1)
                w_1 = pc.calc_weights(ene_1/kT, ene_0/kT)
            else:   # If no lambda change
                new_tau_c = old_tau_c * np.random.normal(1., 0.02)  # Increase time (timestep) ? ; Maybe not needed ????
            proteinStructure = MDAnalysis.Universe(os.path.join(frames_dir, 'frame_0000000.pdb'))

            # RECALCULATE PRE's
            for lab in labels:
                PREPrediction(proteinStructure, lab, plotting_delta=0,
                              replicas=1,
                              load=os.path.join(calc_data_dir, 'save-1_100000-res-{}.pkl'.format(lab)),
                              weights=w_1[0::2],
                              output_prefix=os.path.join(calc_data_dir, '1_100000-res'),
                              save_file=os.path.join(calc_data_dir, 'save-1_100000-res-{}.pkl'.format(lab)),
                              tau_c=new_tau_c, tau_t=10e-12,
                              selection='N', optimize=False,
                              idp=True, wh=800)

                PREPrediction(proteinStructure, lab, plotting_delta=0,
                              replicas=1,
                              load=os.path.join(calc_data_dir, 'save-2_100000-res-{}.pkl'.format(lab)),
                              weights=w_1[1::2],
                              output_prefix=os.path.join(calc_data_dir, '2_100000-res'),
                              save_file=os.path.join(calc_data_dir, 'save-2_100000-res-{}.pkl'.format(lab)),
                              tau_c=new_tau_c, tau_t=10e-12,
                              selection='N', optimize=False,
                              idp=True, wh=800)

            # Read new PRE's with possibly modified weights
            new_data_calc_1 = pc.read_calc_pre(calc_data_dir, labels, '1_100000-res-')
            new_data_calc_2 = pc.read_calc_pre(calc_data_dir, labels, '2_100000-res-')

            new_data_train = new_data_calc_1[train_labels]      # Training set only with PRE's reduced labels
            new_data_val_1 = new_data_calc_2[train_labels]      # Validation set only with PRE's reduced labels
            new_data_val_2 = new_data_calc_1[[val_label]]       # Validation set with excluded labels

            fit_new_train = pc.fit_pre(data_exp_train, new_data_train)              # New fits
            fit_new_val_1 = pc.fit_pre(data_exp_train, new_data_val_1)
            fit_new_val_2 = pc.fit_pre(data_exp_val_2, new_data_val_2)

            n_eff = np.exp(-np.sum(w_1 * np.log(w_1 * w_1.size)))                   # N_eff

            fit_val_1_history.append(fit_old_val_1)                                 # Save
            fit_val_2_history.append(fit_old_val_2)
            fit_train_history.append(fit_old_train)
            tauc_history.append(old_tau_c)
            neff_history.append(n_eff)
            print(f'Iter={i:3d}  old_train={fit_old_train:.4f}  new_train={fit_new_train:.4f} '
                  f' old_val1_1={fit_old_val_1:.4f}  new_val_1={fit_new_val_1:.4f}  old_val_2={fit_old_val_2:.4f} '
                  f' new_val_2={fit_new_val_2:.4f}  n_eff={n_eff:.3f}  tau_c={new_tau_c:.4g}')
            # If TRAINING fit is better
            if fit_new_train < fit_old_train:
                fit_old_train = fit_new_train
                # If BOTH VALIDATION fits are better
                if fit_new_val_1 < fit_old_val_1 and fit_new_val_2 < fit_old_val_2:
                    fit_old_val_1 = fit_new_val_1
                    fit_old_val_2 = fit_new_val_2
                    old_alphas = new_alphas.copy()
                    old_tau_c = new_tau_c
                else:
                    break
            else:
                print("Rejected")
                new_alphas = old_alphas.copy()
                new_tau_c = old_tau_c

        # Save fits to files
        filename = os.path.join(fits_out_dir, f'fits-{val_label:03d}.dat')
        fits = pd.DataFrame({'repe': repe, 'train': fit_train_history, 'val1': fit_val_1_history,
                             'val2': fit_val_2_history, 'neff': neff_history, 'tau': tauc_history})
        fits.astype({'repe': 'int32'})
        fits.to_csv(filename,
                    sep=' ',
                    mode='a',
                    header=False,
                    index=False)

        filename = os.path.join(alph_out_dir, f'alphas-{val_label:03d}.dat')
        alphas = pd.DataFrame({'repe': repe,
                               'exvol': [old_alphas[0]],
                               'locexvol': [old_alphas[1]],
                               'bias': [old_alphas[2]],
                               'torsionterm': [old_alphas[3]],
                               'hbmm': [old_alphas[4]],
                               'hbms': [old_alphas[5]],
                               'hydrophobicity': [old_alphas[6]],
                               'chargedscinteraction': [old_alphas[7]]
                               })
        alphas.astype({'repe': 'int32'})
        alphas.to_csv(filename,
                      sep=' ',
                      mode='a',
                      header=False,
                      index=False)
