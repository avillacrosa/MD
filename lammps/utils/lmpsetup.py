"""
Objects here are system independent. They do not need the definitions in the data directory
"""
import numpy as np
from subprocess import run, PIPE
from string import Template
import random
import sys
import definitions
import io


def generate_lammps_input(lambdas, sigmas, T, debye=0.1):
    """
    Generate a new input file with the parameters arising from the new lambdas.
    """
    def pair_coefs(lambdas, sigmas):
        """
        Generate the pair coefficients
        """
        epsilon = 0.2
        lines = []
        sigmaij, lambdaij = 0., 0.
        charged, sorted_residues = [], []
        for res_key in definitions.residues:
            if definitions.residues[res_key]["q"] != 0.:
                charged.append(definitions.residues[res_key]["name"])
            sorted_residues.append(definitions.residues[res_key]["name"])

        for i in range(len(lambdas)):
            for j in range(i, len(lambdas)):
                sigmaij = (sigmas[i] + sigmas[j]) / 2
                lambdaij = (lambdas[i] + lambdas[j]) / 2
                if sorted_residues[i] in charged and sorted_residues[j] in charged:
                    cutoff = 35.00
                else:
                    cutoff = 0.0
                line = 'pair_coeff         {:2d}      {:2d}       {:.6f}   {:.3f}    {:.6f}  {:6.3f}  {:6.3f}\n'.format(
                    i + 1, j + 1, epsilon, sigmaij, lambdaij, 3 * sigmaij, cutoff)
                lines.append(line)
        return lines

    v_seed = int(debye*10000)
    lang_seed = int(debye*20000)
    d = {"pair_coeff": ''.join(pair_coefs(lambdas, sigmas)), "debye": round(debye, 3),
         "v_seed": v_seed, "langevin_seed": lang_seed, "temp": T}

    filein = open('../templates/input_template.lmp')
    topo_template = Template(filein.read())
    result = topo_template.safe_substitute(d)

    # filename = f'../default_output/hps{debye:.3f}.lmp'
    filename = f'../default_output/hps{round(debye, 3):.3f}.lmp'
    with open(filename, 'tw') as fileout:
        fileout.write(result)
    return filename


def generate_lammps_topo(seq, nchains=1):
    """
    Generate "topology" file from 1 letter aminoacidic sequence.
    Ex. for asyn : MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA
    """

    filein = open('../templates/topo_template.data')
    topo_template = Template(filein.read())

    masses = []
    for i, key in enumerate(definitions.residues):
        masses.append(f'           {i + 1:2d}  {definitions.residues[key]["mass"]} \n')

    atoms, bonds, coords = [], [], []

    k = 1
    for chain in range(1, nchains+1):
        coords = [-240., -240 + chain*20, -240]
        for aa in seq:
            # coords[random.choice([0, 1, 2])] += definitions.bond_length
            coords[0] += definitions.bond_length
            atoms.append(f'       {k :3d}          {chain}    '
                         f'      {definitions.residues[aa]["id"]:2d}   '
                         f'    {definitions.residues[aa]["q"]: .2f}'
                         f'    {coords[0]: .3f}'
                         f'    {coords[1]: .3f}'
                         f'    {coords[2]: .3f} \n')
            if k != chain*(len(seq)):
                bonds.append(f'       {k:3d}       1       {k:3d}       {k + 1:3d}\n')
            k += 1

    d = {"natoms": nchains*len(seq), "nbonds": nchains*(len(seq) - 1), "atom_types": len(definitions.residues),
         "masses": ''.join(masses), "atoms": ''.join(atoms), "bonds": ''.join(bonds)}

    result = topo_template.safe_substitute(d)
    filename = f'../default_output/hps.data'
    with open(filename, 'tw') as fileout:
        fileout.write(result)
    return filename


def generate_queue_sub(np, input, work_dir, job_name):
    filein = open('../templates/qsub_template.tmp')
    qsub_template = Template(filein.read())
    dict = {}
    dict["work_dir"] = work_dir
    dict["command"] = f"/home/ramon/local/openmpi/202_gcc630/bin/mpirun -np {np} /home/adria/local/lammps/bin/lmp -in {input}"
    dict["np"] = np
    dict["jobname"] = job_name
    result = qsub_template.safe_substitute(dict)
    filename = f'../default_output/{job_name}.qsub'
    with open(filename, 'tw') as fileout:
        fileout.write(result)
    return filename


def get_lammps_energies(source, from_file_name=False):
    """
    Get the energies of a re-run LAMMPS output.
    Input: source: a str with the output or a file name with the output.
    from_file_name: If true, source is a string corresponding to the name of a file
    """
    # Choose function to open depending on whether 'source' is a StringIO or a file name.
    if from_file_name:
        openfunc = open
    else:
        openfunc = io.StringIO

    with openfunc(source) as filein:
        reading = False
        energies = []
        for line in filein:
            if 'Loop time of' in line:  # end of energy output
                reading = False

            if reading:
                if line.startswith('WARNING'):  # this warning does not affect the final energy
                    continue
                else:
                    ene = line.split()[1]
                    energies.append(ene)

            if 'Step PotEng KinEng Temp' in line:  # start of energy output
                reading = True
    energies = np.array(energies, dtype=np.float)
    return energies


def run_lammps():
    """
    Run a LAMMPS job

    """
    filename = 'hps.data'
    command = '/home/adria/local/lammps/bin/lmp -in ' + filename
    out = run(command.split(), stdout=PIPE, stderr=PIPE, universal_newlines=True)
    return out



