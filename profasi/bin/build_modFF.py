#!/usr/bin/env python

import os
import numpy as np
import sys
import argparse as prs
from numpy.core.defchararray import add


sys.path.insert(1, '/home/adria/scripts/depured_profasi/libs/')
import utils

parser = prs.ArgumentParser()
parser.add_argument("-ap", "--alphas_path", dest="alphas_path", help="Path to the alphas", default=None)
parser.add_argument("-Ti", "--tinfo", dest="temperature_info", help="Path to the temperature.info file",
                    default=None)
parser.add_argument("-T", "--temperatures", dest="temperatures", help="Path to the temperature.info file",
                    default=None)
parser.add_argument("-seq", "--sequence", dest="sequence",
                    help="Path to a file containing the sequence or sequence string", default=None)
parser.add_argument("-f", "--target_folder", dest="target_folder",
                    help="Folder under which the files will be generated")
args = parser.parse_args()

args.target_folder = "/home/adria/perdiux/prod/integrase/modifiedFF/integrase-shift-v4"
args.sequence = open(os.path.join(args.target_folder, "sequence.txt"), "r").read().splitlines()[0]
# args.alphas_path = "/home/adria/perdiux/prod/integrase/modifiedFF/integrase-linear_fit/alphas.txt"
args.alphas_path = os.path.join(args.target_folder, "alphas.txt")
# args.temperature_info = "/home/adria/perdiux/prod/integrase/modifiedFF/integrase-linear_fit/temperature.info"
args.temperature_info = os.path.join(args.target_folder, "temperature.info")
args.temperatures = 8
tinfo = np.genfromtxt(args.temperature_info)

kelvin_temps = tinfo[tinfo[:, 0] == args.temperatures, 2]
data = np.genfromtxt(args.alphas_path)
alphas = data[:, 1:9]
rg = data[:, 0]

#TODO THERE MUST BE A WAY TO DO THIS IN A "NUMPYESC" WAY
rgs = []
for r in rg:
    if r < 0:
        f_rg = "m{0:.2f}".format(-r)
    else:
        f_rg = "{0:.2f}".format(r)
    rgs.append(f_rg)
rgs = np.array(rgs)

if args.temperature_info is None or args.alphas_path is None:
    print("temperature.info or alphas path not given")
    raise SystemExit

ccObs = ("ExVol", "LocExVol", "Bias", "TorsionTerm", "HBMM", "HBMS", "Hydrophobicity", "ChargedSCInteraction")

for k, alpha in enumerate(alphas):
    for j, temp in enumerate(kelvin_temps):
        # out_dir = os.path.join(args.target_folder, str(j + 2) + "T_" + os.path.basename(wd))
        out_dir = os.path.join(args.target_folder, "{0}R_{1}".format(rgs[k], os.path.basename(args.target_folder)))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        file = open(os.path.join(out_dir, "settings.cnf"), "w+")
        template = "add_chain 1 <*" + args.sequence + "*> \n"
        template += "ncycles 1000000 \n"
        template += "conf_write_freq 500 \n"
        template += "nrt 500 \n"
        template += "tmin 273 Kelvin \n"
        template += "tmax 353 Kelvin \n"
        template += "ntmp 16 \n"
        template += "force_field FF08mod=FF08"
        for i, obs in enumerate(ccObs):
            template += "+Extras:ObsEnergy(obs=" + obs + ",scale=" + str(alpha[i] - 1) + ")"
        template += "\n"
        template += "new_obs Rg rg \n"
        file.write(template)
        file.close()









# scales = [0.8, 0.85, 0.9, 0.95, 1.10, 1.15, 1.20]
#
# for j, alpha in enumerate(alphas):
#     out_dir = os.path.join("/home/adria/perdiux/test/collapse/reweighted/TEST", str(scales[j])+"S_" + os.path.basename(dir))
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#     file = open(os.path.join(out_dir, "settings.cnf"), "w+")
#     template = "add_chain 1 <*" + seq + "*> \n"
#     template += "ncycles 1000000 \n"
#     template += "conf_write_freq 500 \n"
#     template += "nrt 500 \n"
#     template += "temperature " + str(329.618433) + " Kelvin \n"
#     template += "force_field FF08s=FF08"
#     for i, o in enumerate(obs):
#         template += "+Extras:ObsEnergy(obs=" + o + ",scale=" + str(alpha[i]-1) + ")"
#     template += "\n"
#     template += "new_obs Rg rg \n"
#     file.write(template)
#     file.close()


# temps = [341.108937, 335.314469, 329.618433, 324.019156, 318.514994, 313.104333, 307.785584, 302.557185, 297.417601,
#          292.365325, 287.398872, 282.516785]
# obs = ["ExVol", "LocExVol", "Bias", "TorsionTerm", "HBMM", "HBMS", "Hydrophobicity", "ChargedSCInteraction"]
