import fgRPA_ucst as fgu
import numpy as np

poss = []
with open('/home/adria/p-test-12.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = [line.rstrip(']') for line in lines]
    lines = [line.rstrip('[') for line in lines]
    for line in lines:
        str_list = line.split('[')[1].split(',')
        poss.append([int(i) for i in str_list])

find_cri = True
phis_mM = 100
ehs = [0, 0]
umax = 2
seqname = 'CPEB4_D4'
du = 0.001

fgu.fgRPA_ucst(seqname='CPEB4_12D', find_cri=find_cri, phis_mM=phis_mM,
               ehs=ehs, umax=umax, parallel=False, du=du, eps0=80, cri_only=True,
               name="p-test-12").run()

for pos in poss:
    fgu.fgRPA_ucst(seqname=seqname, find_cri=find_cri, phis_mM=phis_mM,
                   ehs=ehs, umax=umax, parallel=False, du=du, eps0=80, cri_only=True,
                   name="p-test-12", mimics=pos).run()