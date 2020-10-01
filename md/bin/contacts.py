import pathlib
import faulthandler
import lmp
import os
import numpy as np

#prots = ["WT", "D4", "12D_D4", "7D_WT"]
prots = ["7D_WT"]
out_dir = '/home/adria/powerlaws'
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
faulthandler.enable()
for prot in prots:
        tr = lmp.LMP(md_dir=f'/home/adria/data/real_final/HPS-T/0.8/{prot}', every=10)
        for T in range(len(tr.temperatures)):
                wt_cmap = tr.intra_distance_map(temperature=T)
                _, dijs, err = tr.ij_from_contacts(contacts=wt_cmap[:,:,:,:])
                print(dijs.shape, err.shape)
                np.savetxt(os.path.join(out_dir, f"flory_dij_{prot}_{T}.txt"), [dijs[0,:], err[0,:]])
        tr = None

