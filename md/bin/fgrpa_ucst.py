import fgRPA_ucst as fgu

WT_COL='darkblue'
D7_COL='royalblue'

D4_COL = 'red'
D12_COL = 'orange'

find_cri=True
phis_mM = 100
ehs = [0,0]
umax=None
seqname='CPEB4'

u_wt, wt_sp1, wt_sp2, wt_bi1, wt_bi2 = fgu.fgRPA_ucst(seqname=seqname,
                                            find_cri=find_cri,
                                            phis_mM=phis_mM,
                                            ehs=ehs,
                                            umax=2.5,
                                            parallel=False,
                                            eps0=80).run()