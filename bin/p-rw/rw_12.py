import lmp
import hmd

s = lmp.LMP(md_dir='/home/adria/data/real_final/HPS-T/0.75/D4', every=1)
s.phospho_rew(T=0, savefile='/home/adria/p-test-12.txt', scale=0.75, n_ps=12, temp_dir='/home/adria/P-rw-12')
