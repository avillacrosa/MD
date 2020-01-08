#!/home/adria/anaconda3/bin/python

import sys
sys.path.insert(1, '../utils')
import calc
import numpy as np
import definitions
import lmpsetup
import lmpmath
import shutil
import random
import os

Is = np.array([25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400])*10**(-3)
redIs = np.array([50, 100, 150, 200, 250, 300, 350, 400])*10**(-3) # For faster dynamics
cpeb4 = "MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE"
cpeb4d4 = "MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE"
hnRPA2_CTD = "GRGGNFGFGDSRGGGGNFGPGPGSNFRGGSDGYGSGRGFGDGYNGYGGGPGGGNFGGSPGYGGGRGGYGGGGPGYGNQGGGYGGGYDNYGGGNYGSGNYNDFGNYNQQPSNYGPMKSGNFGGSRNMGGPYGGGNYGPGGSGGSGGYGGRSRY"
asyn = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"

cpeb4d47d  = [18, 38, 40, 97, 252, 255, 353]
cpeb47d    = [18, 38, 40, 97, 252, 255, 353]
cpeb4d412d = [18, 38, 40, 97, 252, 255, 326, 330, 332, 353, 359, 364]
cpeb412d   = [18, 38, 40, 97, 252, 255, 326, 330, 332, 353, 359, 364]


def build_lmp_dI(ionic_strengths, seq, oliba_outdir, pre_id='', lambdas_from='hps'):
    T = 300
    eps_rel_water = 80
    print("PREID", pre_id)

    perdiux_outdir = oliba_outdir.replace('/perdiux','')

    names = []
    for res in definitions.residues:
        names.append(definitions.residues[res]["name"])
    sigmas = [definitions.sigmas[k] for k in names]
    lambdas = [definitions.lambdas[k] for k in names]
    if lambdas_from == 'none':
        print('Using lambda_ij = 0')
        lambdas = np.array(lambdas)
        lambdas.fill(0)
    topo_file = lmpsetup.generate_lammps_topo(seq=seq)
    par_dir = os.path.join(oliba_outdir,f'{pre_id}-I_cpeb4')
    # par_dir = os.path.join(oliba_outdir,f'I')
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)
    for k, I in enumerate(ionic_strengths):
        debye_length = lmpmath.debye_length(I, eps_rel=eps_rel_water, T=T, angstrom=True)
        debye_wv = 1/debye_length
        debye_wv = debye_wv * 10 **(-10)
        wd = os.path.join(par_dir, f'{debye_wv:.3f}')
        if not os.path.exists(wd):
            os.mkdir(wd)
        perdiu_wd = os.path.join(perdiux_outdir, f'{pre_id}-I_cpeb4/{debye_wv:.3f}')
        # perdiu_wd = os.path.join(perdiux_outdir, f'asyn/{debye_wv:.3f}')
        input_file = lmpsetup.generate_lammps_input(lambdas, sigmas, debye=debye_wv, T=T)
        qsub_file = lmpsetup.generate_queue_sub(np=8, input=f'hps{debye_wv:.3f}.lmp', work_dir=perdiu_wd, job_name=f'hpsI-{k}-{pre_id}')
        shutil.copyfile(input_file, os.path.join(wd, os.path.basename(input_file)))
        shutil.copyfile(topo_file, os.path.join(wd, os.path.basename(topo_file)))
        shutil.copyfile(qsub_file, os.path.join(os.path.dirname(wd), os.path.basename(qsub_file)))


def build_lmp_dI_qHist(ionic_strengths, seq=cpeb4):
    charges = [0, 0.5, 0.55, 1]
    for charge in charges:
        definitions.residues["H"]["q"] = charge
        build_lmp_dI(ionic_strengths, seq=seq, pre_id=f'{charge}H')


def build_lmp_dI_q0(ionic_strengths, oliba_outdir, seq=cpeb4):
    for key in definitions.residues.keys():
        definitions.residues[key]["q"] = 0
    build_lmp_dI(ionic_strengths, seq, oliba_outdir, pre_id='q0l0', lambdas_from='none')


def build_lmp_dT(temps, seq, oliba_outdir, pre_id='', lambdas_from='hps'):
    perdiux_outdir = oliba_outdir.replace('/perdiux', '')
    pre_id = 'T'
    names = []
    for res in definitions.residues:
        names.append(definitions.residues[res]["name"])
    sigmas = [definitions.sigmas[k] for k in names]
    lambdas = [definitions.lambdas[k] for k in names]
    if lambdas_from == 'none':
        print('Using lambda_ij = 0')
        lambdas = np.array(lambdas)
        lambdas.fill(0)
    topo_file = lmpsetup.generate_lammps_topo(seq=seq)
    par_dir = os.path.join(oliba_outdir, f'{pre_id}-hnRPA')
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)
    for k, T in enumerate(temps):
        debye_wv = 0.1
        wd = os.path.join(par_dir, f'{T:.1f}')
        if not os.path.exists(wd):
            os.mkdir(wd)
        perdiu_wd = os.path.join(perdiux_outdir, f'{pre_id}-hnRPA/{T:.1f}')
        input_file = lmpsetup.generate_lammps_input(lambdas, sigmas, debye=debye_wv, T=T)
        qsub_file = lmpsetup.generate_queue_sub(np=8, input=f'hps{T:.1f}.lmp', work_dir=perdiu_wd,
                                                job_name=f'hpsI-{k}-{pre_id}')
        shutil.copyfile(input_file, os.path.join(wd, os.path.basename(input_file)))
        shutil.copyfile(topo_file, os.path.join(wd, os.path.basename(topo_file)))
        shutil.copyfile(qsub_file, os.path.join(os.path.dirname(wd), os.path.basename(qsub_file)))


def build_lmp_dI_P(sites, ionic_strengths, seq=cpeb4, mode='random'):
    i_sites = sites
    f_sites = []
    if mode == 'random':
        n = random.randint(1, len(sites))
        for i in range(n):
            rvalue = random.choice(i_sites)
            i_sites.remove(rvalue)
            f_sites.append(rvalue)
    else:
        f_sites = sites
    aseq = list(seq)
    for site in f_sites:
        aseq[site-1] = 'D'
    seq = ''.join(aseq)
    build_lmp_dI(ionic_strengths, seq=seq, pre_id='12D')

# oliba_outdir = '/home/adria/perdiux/prod/lammps/dignon'
oliba_outdir = '/home/adria/perdiux/prod/lammps/dignon/LONG'
# Ts = [150.0, 170.1, 193.0, 218.9, 248.3, 281.7, 319.5, 362.4, 411.1, 466.3, 529.0, 600.0]
# build_lmp_dT(Ts, seq=hnRPA2_CTD, oliba_outdir=oliba_outdir, lambdas_from='hps')

build_lmp_dI([25e-3, 100e-3, 400e-3], seq=cpeb4, oliba_outdir=oliba_outdir, lambdas_from='hps', pre_id='LONG')
build_lmp_dI([25e-3, 100e-3, 400e-3], seq=cpeb4, oliba_outdir=oliba_outdir, lambdas_from='none', pre_id='L0-LONG')
# build_lmp_dI_q0(Is, seq=cpeb4, oliba_outdir=oliba_outdir)
# build_lmp_dI(Is, seq=cpeb4, lambdas_from='hps')
# build_lmp_dI_qHist(redIs, seq=cpeb4)
# build_lmp_dI_P(cpeb412d, Is, seq=cpeb4, mode='definite')
