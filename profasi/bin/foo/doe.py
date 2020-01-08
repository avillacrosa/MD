import numpy as np
from scipy.interpolate import interp1d


exp_data = np.genfromtxt("/home/adria/perdiux/test/collapse/reweighted/emp_temp-rg-v2.txt")
f = interp1d(exp_data[:, 0], exp_data[:, 1])
x_sim = np.genfromtxt("/home/adria/perdiux/test/collapse/reweighted/temperature_range.info")
data = np.array([x_sim[:, 0], x_sim[:, 2], f(x_sim[:, 2])])
data = np.transpose(data)
# data = np.array([x_sim[:, 2], f(x_sim[:, 2])])
np.savetxt("exp_interpolated_integrase.txt", data)

