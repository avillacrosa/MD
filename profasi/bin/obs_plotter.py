#!/usr/bin/env python

import sys

sys.path.insert(1, '/home/adria/scripts/profasi/libs/')
# sys.path.insert(1, '/home/adria/mounts/oliba/scripts/profasi/libs/')

import numpy as np
import argparse as prs
import matplotlib.pyplot as plt
import utils
import seaborn as sns
import os

plt.figure(num=None, figsize=(8, 6))

if __name__ == "__main__":

    # current_palette = sns.color_palette()
    # sns.set_palette("Dark2")

    parser = prs.ArgumentParser()
    parser.add_argument("-d", "--results_dir", dest="directories", help="Path to PROFASI's result folder", nargs='*')
    parser.add_argument("-x", "--x_values", dest="x", help="Observables to plot on the x-axis (Etot,Rg...)", nargs='?')
    parser.add_argument("-y", "--y_values", dest="y", help="Observables to plot on the y-axis (Etot,Rg...). Choose"
                                                           "E to plot all energies", nargs="*")
    parser.add_argument("-r", "--replica", dest="replica", help="System rank or replica")
    parser.add_argument("-T", "--temperature", dest="temperature", nargs="*",
                        help="Take only systems at the temperature index", default=None)
    parser.add_argument("-no_avg", "--no_average", dest="average", default=True, action='store_false',
                        help="Switch. Show average over x-values")
    parser.add_argument("-obs", "--show_obs", dest="display_obs", default=False, action="store_true",
                        help="Display available observables")
    parser.add_argument("-of", "--output_folder", dest="output_folder", help="Name of the default_output file",
                        default="../default_output")
    parser.add_argument("-i", "--input_file", dest="input_file",
                        help="Include an external data file to plot on top of the generated plot", default=[],
                        nargs="*")
    parser.add_argument("-n", "--normalize", dest="normalize",
                        help="Plot y-values around the mean and not in absolute values", default=False,
                        action='store_true')
    parser.add_argument("-eq", "--equilibration", dest="equilibration",
                        help="Use such number of cycles as equilibration", default=10000, action='store_true')

    prf_dict = utils.prf_dict

    args = parser.parse_args()
    if args.x is None:
        args.x = "temperature"
    if args.y is None:
        args.y = ["helixcontent"]
    args.directories = ["/home/adria/perdiux/prod/profasi/integrase/integrase-og",
                        "/home/adria/perdiux/prod/profasi/integrase/modifiedFF/integrase-shift-v5"]
    #                     # "/home/adria/perdiux/prod/profasi/mod_src/integrase",
    #                     "/home/adria/perdiux/prod/profasi/mod_src/integrase_shift",
    #                     # "/home/adria/perdiux/prod/profasi/mod_src/integrase2",
    #                     # "/home/adria/perdiux/prod/profasi/mod_src/integrase_shift2",
    #                     "/home/adria/perdiux/prod/profasi/mod_src/integrase_shift3"]
    # args.directories = ["/home/adria/data/prod/profasi/asyn/asyn++"]

    args.directories = ['/home/adria/data/prod/profasi/asyn/asyn++',
                        '/home/adria/data/prod/profasi/integrase/integrase-og',
                        '/home/adria/data/prod/profasi/l-repressor',
                        '/home/adria/data/prod/profasi/prothymosin-c/prothymosin-c',
                        '/home/adria/data/prod/profasi/prothymosin-n/prothymosin-n'
    ]

    # args.input_file = [
        # '/home/adria/ExpData/wuttke_data/integrase.csv',
        # '/home/adria/ExpData/wuttke_data/cspm.csv',
        # '/home/adria/ExpData/wuttke_data/l-repressor.csv',
        # '/home/adria/ExpData/wuttke_data/protac.csv',
        # '/home/adria/ExpData/wuttke_data/protan.csv'
    # ]

    args.average = True

    dirs = utils.get_all_profasi_dirs(args.directories)

    full_data = []

    names = []
    for y in args.y:
        names.append(prf_dict[y]["name"])
    plot_title = ', '.join(names) + ' vs. ' + prf_dict[args.x]["name"] + ' for '

    for d, directory in enumerate(dirs):

        available_obs = utils.get_obs_names(dir=directory)
        prot_name = os.path.basename(directory)
        plot_title += prot_name + ", "
        file_names = []
        labels = []

        if not os.path.exists(args.output_folder):
            print("Output folder not found. Data and figure will not be saved")
            args.output_folder = None

        if args.display_obs:
            print("The available observables are : ")
            print(available_obs)
            raise SystemExit

        if args.x not in available_obs or not any(y in args.y for y in available_obs) and args.y[0] != 'E':
            print("x or y observable name unknown. The available options are :")
            print(available_obs)
            raise SystemExit

        x_key, y_key = 0, []

        for idx, obs in enumerate(available_obs):
            if args.x == obs:
                x_key = idx
            if obs in args.y:
                y_key.append(idx)

        indices = np.append(x_key, y_key)

        if args.y == ['E']:
            y_key = np.linspace(4, 11, 8, dtype=int)
            indices = np.append([x_key], y_key)

        rt = np.genfromtxt(os.path.join(directory, 'results', 'rt'))
        T_key = utils.convert_to_kelvin(rt=rt, dir=directory)
        rt = rt[rt[:, 1] > args.equilibration, :]

        if args.average:
            data = []
            for entry in np.unique(rt[:, x_key]):
                data.append(rt[rt[:, x_key] == entry][:, indices])
            data = np.array(data)
            ranked_data = np.mean(data, axis=1)
            full_data = [ranked_data]
            file_names.append('_'.join(args.y) + '_' + args.x + "_" + prot_name + '.txt')
        elif args.replica is not None:
            for replica in args.replicas:
                ranked_data = rt[rt[:, 0] == int(replica)][:, indices]
                full_data.append(ranked_data)
                file_names.append("r" + replica + '_'.join(args.y) + '_' + args.x + "_" + prot_name + '.txt')
                labels.append("Replica : " + replica)
        elif args.temperature is not None:
            for temp in args.temperature:
                kelvin_T = T_key[int(temp)][1]
                ranked_data = rt[rt[:, 2] == kelvin_T][:, indices]
                full_data.append(ranked_data)
                file_names.append("T" + str(kelvin_T) + '_'.join(args.y) + '_' + args.x + "_" + prot_name + '.txt')
                labels.append("Temperature : " + str(kelvin_T))
        else:
            print(
                "No replica, temperature or average selected. Choose either -r RANK -T TEMPERATURE_INDEX "
                "or do not use the -no_avg option")
            raise SystemExit

        if args.normalize:
            for data in full_data:
                for y in range(1, len(indices)):
                    data[:, y] = data[:, y] - data[:, y].mean()

        for i, data in enumerate(full_data):
            for r in range(0, data.shape[1] - 1):
                if args.y == ["E"]:
                    label = utils.get_obs_names(directory, energies=True)[r]
                else:
                    if labels:
                        label = prot_name + labels[i] + " " + args.y[r]
                    else:
                        label = prot_name + " " + args.y[r]
                # label = labels[d]
                plt.plot(data[:, 0], data[:, r + 1], linestyle="-", label=label)
                # ==================================================REMOVE==============================================
                # if d == 0:
                #     plt.plot(0.96*data[:, 0], data[:, r + 1], linestyle="--", label="0.96*Original Integrase")
                #     plt.plot(1.04*data[:, 0], data[:, r + 1], linestyle="--", label="1.04*Original Integrase")
                # ==================================================REMOVE==============================================

        if args.output_folder is not None:
            for i, prot in enumerate(full_data):
                print("Saving data at : " + os.path.join(args.output_folder, file_names[i]))
                np.savetxt(os.path.join(args.output_folder, file_names[i]), prot)

    if args.input_file is not None:
        for k, in_file in enumerate(args.input_file):
            # OG line. Why did I do that ???
            # data = np.genfromtxt(os.path.join(in_file, "results", "rt"))
            data = np.genfromtxt(os.path.join(in_file))
            label = os.path.basename(os.path.splitext(in_file)[0]).capitalize()
            # plt.plot(data[:, 0], data[:, 1], marker="o", markersize=3, linestyle="", color="orange", label=label)
            # label="Experimental Results"
            plt.plot(data[:, 0], data[:, 1],  linestyle="-", label=label)

    x_label = args.x
    y_label = ' ,'.join(args.y)

    plt.xlabel(f'{prf_dict[x_label]["name"]}  ({prf_dict[x_label]["unit"]})')
    plt.ylabel(f'{prf_dict[y_label]["name"]}  ({prf_dict[y_label]["unit"]})')
    plt.title(plot_title)
    if len(args.directories) + len(args.y) + len(args.input_file) != 1:
        plt.legend(loc="upper right")

    if args.output_folder is not None:
        plot_name = '_'.join(args.y) + '_' + args.x + '.png'
        print("Saving figure at : " + os.path.join(args.output_folder, plot_name))
        plt.savefig(os.path.join(args.output_folder, plot_name))

    plt.show()
