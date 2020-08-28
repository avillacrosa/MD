import numpy as np
import lmp
import glob
import os

#full rw for HPS-T

l = 0.75
temp_dir = f'/home/adria/OPT_D4-WT/{l}'
base_dir = '/home/adria/scripts/md/bin/wt-d4'


prot_a, prot_b = 'D4', 'WT'

epss = np.linspace(80, 40, 5)
above = lmp.LMP(md_dir=f'/home/adria/data/real_final/HPS-T/{l}/{prot_a}', low_mem=True)
above.rg(full=True)
below = lmp.LMP(md_dir=f'/home/adria/data/real_final/HPS-T/{l}/{prot_b}', low_mem=True)
below.rg(full=True)

temps = [0, 2, 4, 6]
for T in temps:
    lss = [0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75]
    savefile = os.path.join(base_dir, f"rw_{l}_HPST-{T}.txt")
    rgiA, rgiB = above.c_rg[0][T].mean(), below.c_rg[0][T].mean()
    with open(savefile, 'a+') as writer:
        writer.write(f"A = {above.protein}, rgA = {rgiA:.3f} ; B = {below.protein}, rgB = {rgiB:.3f} \n")
        writer.write("lambda, eps, diff,  neff_a,  neff_b, rw_A, rw_B \n")

    for ls in lss:
        for eps in epss:
            calcer = lmp.LMP(None)
            l0, eps0, diff, neff_a, neff_b, rw_A, rw_B = calcer.maximize_charge(above_obj=above,
                                   below_obj=below,
                                   T=T,
                                   l0=ls,
                                   eps0=eps,
                                   model='HPS-T',
                                   temp_dir=temp_dir,
                                   savefile=savefile)
            with open(savefile, 'a+') as writer:
                writer.write(f"{l0:.3f} {eps0:.3f} {diff:.3f} {neff_a:.3f} {neff_b:.3f} {rw_A:.3f} {rw_B:.3f} \n")

