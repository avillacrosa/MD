import sys
import importlib
sys.path.insert(1,'../utils')
import lmp
import lmpsetup

importlib.reload(lmp);
importlib.reload(lmpsetup);

make_dir = '/home/adria/oliba/perdiux/prod/lammps/dignon/TEST/asyn/TEST/test_my_run'
seq = 'MDVFMKGISKAKEGVVAAAEKTKQGVAEAAGKTKEGVIYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQIGKNEEGAPQEGIIEDMPVDPDNEAYEMPSEEGYQDYEPEA'
asyn_maker = lmpsetup.LMPSetup(make_dir, seq)
asyn_maker.del_missing_aas()
asyn_maker.lammps_ordering()
asyn_maker.get_hps_params()
asyn_maker.get_hps_pairs(pairs_from_file='/home/adria/oliba/scripts/depured-lammps/data/lambda_pairs.dat')
# asyn_maker.get_hps_pairs()
asyn_maker.generate_lmp_input()
asyn_maker.generate_topo_input(nchains=1)
asyn_maker.generate_qsub()
asyn_maker.write_hps_files()