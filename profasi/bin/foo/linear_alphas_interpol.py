import numpy as np
import sys
import math
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import os

sys.path.insert(1, '../../libs')
import utils
from methods.rg_E_fit import *

plot = plt.subplots()

dir = ['/home/adria/data/integrase/integrase-og']
multi_alphas, multi_rgs, multi_effs = alphas_from_reg(dir=dir, alphas=np.ones(8), normalize=True, beta=8,
                                                      scales=[0.6, 1.4])

for i, label in enumerate(utils.get_obs_names(dir[0], energies=True)):
    plot[1].plot(multi_rgs, multi_alphas[:, [i]])
plot[1].set_prop_cycle(None)

# print(multi_alphas.shape, multi_rgs.shape)

data = np.zeros(shape=(100, 9))
data[:, 0] = multi_rgs
data[:, 1:9] = multi_alphas
# data = np.array([multi_alphas, multi_rgs])
# print(data[data[:, 0] < 0.55 and data[:, 0] > 0.45, :])
# print(data)

# TODO HARDCODED... Might be ok since I doubt this will be done again
rg_05 = 0.46665376
alphas_05 = [1.00088736, 0.99906145, 1.00030161, 0.99982653, 1.00804757, 0.99894709, 0.99396269, 0.99952999]

delta = (np.array(alphas_05) - 1) / rg_05
# multi_rgs = multi_rgs[multi_rgs > 0]
interp = np.outer(np.copy(multi_rgs), delta) + np.ones_like(np.outer(np.copy(multi_rgs), delta))

for i, label in enumerate(utils.get_obs_names(dir[0], energies=True)):
    plot[1].plot(multi_rgs, interp[:, [i]], linestyle='--', linewidth=1)
plot[1].set_prop_cycle(None)

i_data = np.zeros(shape=(100, 9))
i_data[:, 0] = multi_rgs
i_data[:, 1:9] = interp

print(i_data)

plt.show(block=True)
