#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse as prs
from os.path import join
from os.path import basename
import re

sys.path.insert(1, '/home/adria/scripts/depured_profasi/libs/')
import utils


def hist_from_file(dir, replica=None):
    n_temperatures = utils.get_max_temperature(dir)
    partemp = open(join(dir, 'results', 'partemp.stat'), 'r')
    lines = partemp.readlines()
    if replica is None:
        hist = np.zeros(shape=(n_temperatures + 1, 2))
        hist[:, 0] = np.linspace(0, n_temperatures, n_temperatures + 1)
        for j, line in enumerate(lines):
            if line[0][0].isdigit():
                values = re.findall(r'(\d+)', line)
                hist[int(values[0])][1] += int(values[1])
    if replica is not None:
        matcher = '(rank ' + str(int(replica)) + ')'
        hist = []
        for i, line in enumerate(lines):
            if re.search(matcher, line) is not None:
                for temp in range(1, n_temperatures + 2):
                    hist.append((lines[i + temp].split()))

    hist = np.array(hist, dtype='int')
    plt.bar(hist[:, 0], hist[:, 1], width=1)
    return hist


def hist_from_rt(dir, replica=0):
    n_temperatures = utils.get_max_temperature(dir)
    rt = np.genfromtxt(join(dir, 'results', 'rt'))
    replica_data = rt[rt[:, 0] == int(replica)][:, 2]
    bins = np.linspace(0, n_temperatures, n_temperatures + 1)
    plt.hist(replica_data, bins=bins)
    full_data = [replica_data, bins]
    return full_data


if __name__ == '__main__':
    dir = '/home/adria/data/cspm34++'
    replica = 16
    file_name = "visits-T.txt"

    parser = prs.ArgumentParser()
    parser.add_argument("-d", "--results_dir", dest="directory", help="Path to PROFASI's result folder", nargs='?',
                        default=dir)
    parser.add_argument("-r", "--replica", dest="replica", help="System rank or replica", default=replica)
    parser.add_argument("-of", "--default_output", dest="default_output", help="Name of the default_output file", default=file_name)
    args = parser.parse_args()

    data = hist_from_file(dir=args.directory, replica=args.replica)

    np.savetxt(args.output, data)

    plt.xlabel("T")
    plt.ylabel("Visits")
    plt.title("Replica " + str(replica) + ' for ' + basename(args.directory))
    plt.show()
