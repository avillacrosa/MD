import os
import sys
import importlib
import matplotlib.pyplot as plt
if '../utils' not in sys.path:
    sys.path.insert(0,'../utils')
import lmp
import lmpsetup
import analysis
import numpy as np

ls055_dir = '/home/adria/perdiux/prod/lammps/dignon/CPEB4/0.55-lS-cpeb4'
ls055 = analysis.Analysis(oliba_wd=ls055_dir, temper=True)
rgs055 = ls055.rg(use='md')
# fs055 = ls055.flory_scaling_fit(r0=5.5, use='md')


plt.figure(figsize=(8,8))
X = np.linspace(0,80,100)
Y = np.linspace(0,80,100)
X,Y = np.meshgrid(X, Y)
Z = X - Y + (Y - 21 + X - 21)**2
# Z = Y-X + 1*(X + Y - 80)**2
cunt = plt.contour(X, Y, Z, levels=200)
plt.colorbar(cunt)