import analysis
import shutil
import os

residues = 'GAVLMIFYWKRHDESTCNQP'
seq = 'GNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNS'

temp_dir = '/home/adria/TDP'
T=3

tdp = analysis.Analysis(oliba_wd='/home/adria/data/prod/lammps/TDP-BIG')
shutil.copyfile(os.path.join(tdp.o_wd, f'reorder-{T}.lammpstrj'), os.path.join(temp_dir, f'atom_traj_tdp.lammpstrj'))

print("iRg", tdp.rg().mean(axis=1)[3])
seqs, rgs, lrgs = [], [], []
for i in range(len(seq)):
    for r in residues:
        new_seq = seq[:i] + r + seq[i + 1:]
        rerun_rg, lmp_rg = tdp.topo_minimize(T=T, new_seq=new_seq, temp_dir=temp_dir)
        rgs.append(rerun_rg)
        seqs.append(new_seq)
        lrgs.append(lmp_rg)
        with open("/home/adria/scripts/data/TDP.txt", 'a+') as seqf:
            seqf.write(f'{rerun_rg} {lmp_rg} {new_seq}\n')
