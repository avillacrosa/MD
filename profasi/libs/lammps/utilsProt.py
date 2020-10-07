"""
Objects here are system dependent. They need the definitions in the data directory
"""

from data import definitions
from string import Template
import random
import Bio.Data.IUPACData as iupac


def generate_lammps_input(lambdas, sigmas):
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

    d = {"pair_coeff": ''.join(pair_coefs(lambdas, sigmas))}

    filein = open('data/input_template.data')
    topo_template = Template(filein.read())
    result = topo_template.safe_substitute(d)

    filename = 'hps.lmp'
    with open(filename, 'tw') as fileout:
        fileout.write(result)


# Initialize
names = []
for res in definitions.residues:
    names.append(definitions.residues[res]["name"])
sigmas = [definitions.sigmas[k] for k in names]
lambdas = [definitions.lambdas[k] for k in names]


def generate_lammps_topo(seq, nchains=1):
    """
    Generate "topology" file from 1 letter aminoacidic sequence.
    Ex. for asyn : MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA
    """

    filein = open('data/topo_template.data')
    topo_template = Template(filein.read())

    masses = []

    for i, key in enumerate(definitions.residues):
        masses.append(f'           {i + 1:2d}  {definitions.residues[key]["mass"]} \n')

    atoms, bonds, coords = [], [], []

    k = 1
    for chain in range(1, nchains+1):
        coords = [-240., -240  + chain*20, -240]
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
         "masses": ''.join(masses), "atoms": ''.join(atoms), "bonds": ''.join(bonds), }

    result = topo_template.safe_substitute(d)
    filename = 'hps.data'
    with open(filename, 'tw') as fileout:
        fileout.write(result)


cpeb4 = "MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE"
cpeb4d4 = "MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE"
generate_lammps_topo(cpeb4d4, nchains=2)
