#!/usr/bin/env python

import argparse as prs
import os
import sys
from os.path import basename
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# sys.path.insert(1, '/home/adria/mounts/puput/scripts/depured_profasi/libs/')
sys.path.insert(1, '/home/adria/scripts/profasi/libs/')

import utils
from methods.rg_E_fit import *

# dirs = ['/home/adria/data/asyn37h', '/home/adria/data/integrase', '/home/adria/data/l-repressor']
# dirs = ['/home/adria/data/asyn37h', '/home/adria/data/integrase']
# dirs = ['/home/adria/data/cspm34++']
# dirs = ['/home/adria/perdiux/prod/integrase/integrase-og-v2']
dirs = ['/home/adria/perdiux/prod/profasi/integrase/integrase-og']
default_folder = "../default_output"
# beta = 8
beta = 2

parser = prs.ArgumentParser()
parser.add_argument("-d", "--result_dirs", dest="directories", help="Path to PROFASI's result . Either 1 "
                                                                    "path or many", nargs="*", type=str, default=dirs)
parser.add_argument("-T", "--temperature", dest="temperature_index", help="Find alphas only for given temperature",
                    default=beta)
parser.add_argument("-m", "--method", dest="method",
                    help="Choose to minimize by varying the regulator or by varying the radius of Gyration"
                         " (Currently useless)")
parser.add_argument("-err", "--find_errors", dest="error",
                    help="Find errors in alpha calculation, comparing minimization over 1 set agains the minimization "
                         "from multiple sets", default=True, action="store_false")
parser.add_argument("-mp4", "--make_movie", dest="make_movie", help="Record a movie from the different temperatures",
                    action="store_true", default=False)
parser.add_argument("-plt", "--show_plot", dest="show_plot", action="store_true", help="Draw the plots in python way",
                    default=False)
parser.add_argument("-of", "--output_folder", dest="output_folder", default=default_folder,
                    help="Generate default_output files under this path")
parser.add_argument("-n", "--normalize", dest="normalize",
                    help="Plot y-values around the mean and not in absolute values", default=False, action='store_true')
args = parser.parse_args()

args.normalize = True
args.error = False
args.make_movie = True
args.show_plot = False
args.method = "reg"

if args.method == 'rg':
    func = alphas_from_rg
elif args.method == "maxent":
    func = alphas_from_maxent
else:
    func = alphas_from_reg

if len(args.directories) == 1:
    multi_alphas, multi_rgs, multi_effs = func(dir=args.directories, alphas=np.ones(8), normalize=args.normalize,
                                               beta=args.temperature_index)

    if args.show_plot:
        dir = args.directories[0]
        name = os.path.basename(dir)
        plot = plt.subplots()
        plot[0].canvas.set_window_title('alphas')
        if args.normalize:
            plot[1].set_xlabel('Rg - Rg Exp')
        else:
            plot[1].set_xlabel('Rg')
        plot[1].set_ylabel('Alphas')
        title = 'Alphas vs Rg for ' + name
        plot[1].set_title(title)
        for i, label in enumerate(utils.get_obs_names(args.directories[0], energies=True)):
            plot[1].plot(multi_rgs, multi_alphas[:, [i]], label=label)
        plot[1].set_prop_cycle(None)
        plot[1].legend(loc="lower left")
        plt.savefig("../default_output/alphas_" + name + ".png")

        plot = plt.subplots()
        plot[0].canvas.set_window_title('alphas')
        if args.normalize:
            plot[1].set_xlabel('Rg - Rg Exp')
        else:
            plot[1].set_xlabel('Rg')
        plot[1].set_ylabel('n_eff')
        title = 'Ensemble presence ' + name + " while minimizing \n"
        title += os.path.basename(dir) + ' '
        plot[1].set_title(title)
        plot[1].plot(multi_rgs, multi_effs)
        plot[1].set_prop_cycle(None)
        plot[1].legend(loc="lower left")
        plt.savefig("../default_output/ensemble_" + name + ".png")

    if args.output_folder is not None:
        write_data = np.zeros(shape=(multi_alphas.shape[0], multi_alphas.shape[1] + 1))
        write_data[:, 0] = multi_rgs
        write_data[:, 1:multi_alphas.shape[1] + 1] = multi_alphas
        name = os.path.basename(args.directories[0])
        file_path = join(args.output_folder, "alphas_" + name + "T_" + str(int(args.temperature_index)) + ".txt")
        print("Saving alphas and rg to " + file_path + "\n")
        np.savetxt(file_path, write_data,
                   header="rg " + " ".join(utils.get_obs_names(args.directories[0], energies=True)))

    if args.make_movie:
        Tdict = [353, 347, 341, 335, 329, 324, 318, 313, 307, 302, 297, 292, 287, 282, 277, 273]

        headers = utils.get_obs_names(args.directories[0])
        max_t = utils.get_max_temperature(args.directories[0])

        frame_interval = 50
        max_frames = frame_interval * (max_t + 1)


        def init():
            for temp in temps:
                temp.set_data([10, 40], [0.8, 1.2])
            return temps


        def animate(i, rgs=None, alphas=None):
            j = int(i / frame_interval)
            for a, temp in enumerate(temps):
                temp.set_data(rgs[j, :], alphas[j, :, a])
            xmin = rgs[j, :].min()
            xmax = rgs[j, :].max()
            ymin = alphas[:, :, :].min()
            ymax = alphas[:, :, :].max()
            plt.xlim(xmin - xmin / 100, xmax + xmax / 100)
            plt.ylim(ymin - ymin / 100, ymax + ymax / 100)
            # strtext = 'Temperature  = ' + str(Tdict[j]) + '(K) for ' + basename(args.directories[0])
            strtext = 'Temperature  = ' + str(Tdict[j]) + '(K) for integrase'
            ax.set_title(strtext)
            return temps


        fig = plt.figure(num=None, figsize=(8, 6))
        x_range = (multi_rgs.min() - multi_rgs.min() / 8, multi_rgs.max() + multi_rgs.max() / 10)
        y_range = (multi_alphas.min() - multi_alphas.min() / 8, multi_alphas.max() + multi_alphas.max() / 10)
        ax = plt.axes(xlabel='Rg - <Rg> (nm)', ylabel='Energy component scaling (\u03B1)')

        temp_plot, = ax.plot([], [])

        temps = []
        for a in range(0, 8):
            tobj = ax.plot([], [], lw=2, label=utils.prf_dict[headers[a + 4]]["name"])[0]
            temps.append(tobj)

        ax.legend(loc='lower left')

        rgs = []
        alphas = []
        for t in range(0, max_t + 1):
            al, rg, neff = func(dir=args.directories, alphas=np.ones(8), beta=int(t), normalize=args.normalize)
            rgs.append(rg)
            alphas.append(al)
        rgs = np.array(rgs)
        alphas = np.array(alphas)

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=max_frames, interval=frame_interval,
                                       fargs=(rgs, alphas))
        anim.save(basename(args.directories[0]) + '-alpha-T.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show(block=True)

if len(args.directories) > 1:
    multi_alphas, multi_rgs, multi_effs = func(dir=args.directories, alphas=np.ones(8), beta=args.temperature_index)

    if args.show_plot:
        plots = []
        for d in args.directories:
            name = os.path.basename(d)
            plots.append(plt.subplots())
            plots[-1][0].canvas.set_window_title('alphas')
            if args.normalize:
                plots[-1][1].set_xlabel('Rg - Rg Exp')
            else:
                plots[-1][1].set_xlabel('Rg')
            plots[-1][1].set_ylabel('Alphas')
            title = 'Alphas vs Rg for ' + name + " while minimizing \n"
            for wd in args.directories:
                title += os.path.basename(wd) + ' '
            plots[-1][1].set_title(title)

        for k, plot in enumerate(plots):
            for i, label in enumerate(utils.get_obs_names(args.directories[0], energies=True)):
                plot[1].plot(multi_rgs[:, k], multi_alphas[:, [i]], label=label)
            plot[1].set_prop_cycle(None)
            name = "alphas"
            for d in args.directories:
                name += "_" + os.path.basename(d)
            file_name = os.path.join(args.output_folder, name) + ".png"
            print("Saving figure at " + file_name + "\n")
            plt.savefig(file_name)

    if args.error:
        error = []
        figs = []
        rgs = []
        for d in args.directories:
            name = os.path.basename(d)
            al, rg = func(dir=[d], alphas=np.ones(8), beta=args.temperature_index, plot=False)
            rgs.append(rg)
            e = abs(multi_alphas - al)
            error.append(e)
            figs.append(plt.subplots())
            figs[-1][0].canvas.set_window_title('alphas')
            figs[-1][1].set_xlabel('Rgs')
            figs[-1][1].set_ylabel('\u03B1 error')
            figs[-1][1].set_title("error for " + name)

        error = np.array(error)
        for j, err in enumerate(error):
            f_data = np.column_stack((rgs[j], error[j, :, :]))
            f_data = f_data[f_data[:, 0].argsort()]
            figs[j][1].plot(f_data[:, 0], f_data[:, 1:9])
            figs[j][0].savefig(join(args.output_folder, "error_" + basename(args.directories[j]) + ".png"))

    # TODO: CURRENTLY BROKEN...
    if args.make_movie and False:
        plt.figure(num=None, figsize=(8, 6))
        headers = utils.get_obs_names(args.directories[0])
        max_t = utils.get_max_temperature(args.directories[0])

        frame_interval = 50
        max_frames = frame_interval * (max_t + 1)


        def init():
            for temp in temps:
                temp.set_data([], [])
            return temps


        def animate(i, rgs=None, alphas=None, system=0):
            j = int(i / frame_interval)
            for a, temp in enumerate(temps):
                temp.set_data(rgs[j, :], alphas[j, :, a])
            strtext = 'Temperature index = ' + str(j) + ' ' + basename(args.directories[0])
            ax.set_title(strtext)
            return temps


        anims = []

        for l, d in enumerate(args.directories):
            x_range = (multi_rgs.min() - multi_rgs.min() / 8, multi_rgs.max() + multi_rgs.max() / 10)
            y_range = (multi_alphas.min() - multi_alphas.min() / 8, multi_alphas.max() + multi_alphas.max() / 10)
            fig = plt.figure()
            ax = plt.axes(xlim=x_range, ylim=y_range, xlabel='Rg', ylabel='Energy component scaling (\u03B1)')
            temp_plot, = ax.plot([], [])

            temps = []
            for a in range(0, 8):
                tobj = ax.plot([], [], lw=2, label=headers[a])[0]
                temps.append(tobj)

            ax.legend()

            alphas = []
            rgs = []

            for t in range(0, max_t + 1):
                al, rg = func(dir=args.directories, alphas=np.ones(8), beta=int(t), plot=False)
                rgs.append(rg)
                alphas.append(al)
            rgs = np.array(rgs)
            alphas = np.array(alphas)
            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=max_frames, interval=frame_interval,
                                           fargs=(rgs, alphas, l))
            # anim.save(basename(args.directories[0]) + '-alpha-T.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

        # for anim in anims:
        #     anim.save(basename(args.directories[0]) + '-alpha-T.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show(block=True)
