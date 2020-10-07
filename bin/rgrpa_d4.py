import rgRPA_ucst as rgu

find_cri=True
phis_mM = 100
ehs = [0, 0]
seqname='CPEB4_D4'

rgu.rgRPA_ucst(seqname=seqname, find_cri=find_cri, phis=phis_mM, eh=0, es=0, parallel=True, name="d4_eps20").run()

seqname='CPEB4_D12'

rgu.rgRPA_ucst(seqname=seqname, find_cri=find_cri, phis=phis_mM, eh=0, es=0, parallel=True, name="d4_eps20").run()
