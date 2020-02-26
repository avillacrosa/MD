import lmp
import plotter
import analysis
import numpy as np
import matplotlib.pyplot as plt
inter_contacts = analysis.Analysis(oliba_wd='/home/adria/data/prod/lammps/CPEB4x50-24')
contacts = inter_contacts.inter_distance_map(use='md', contacts=True, temperature=3)