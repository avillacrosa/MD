import analysis
import time

sr = analysis.Analysis(oliba_wd='/home/adria/data/prod/lammps/7D_CPEB4x50/REX', max_frames=4)
ti = time.time()
ampi_res = sr.async_inter_distance_map(temperature=0)
print("Time spent", time.time()-ti)