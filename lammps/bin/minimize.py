import analysis
import datetime

I0, e0, l0 = 100e-3, 80, 1.

now = datetime.datetime.now()
wcost = 2.

f_name = f'min_{wcost}-{now.day}-{now.hour}-{now.minute}.txt'
mini = analysis.Analysis(oliba_wd=None)
mini.minimize(a_dir='/home/adria/data/prod/lammps/12D_CPEB4_D4/1.0ls-100I-80e',
              b_dir='/home/adria/data/prod/lammps/CPEB4_D4/1.0ls-100I-80e',
              T=5,
              I0=I0,
              l0=l0,
              eps0=e0,
              savefile=f_name,
              weight_cost_mean=wcost,
              method='sto')
