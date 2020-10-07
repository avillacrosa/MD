import numpy as np
from data import definitions
from utilsProt import *
import math
import scipy.constants as cnt


# Initialize
names = []
for res in definitions.residues:
    names.append(definitions.residues[res]["name"])
sigmas = [definitions.sigmas[k] for k in names]
lambdas = [definitions.lambdas[k] for k in names]

ionic_strength = 100e-3
lb = cnt.e * cnt.e / (4 * math.pi * cnt.epsilon_0 * 80 * cnt.Boltzmann * 300)
kappa = np.sqrt(8 * math.pi * lb * ionic_strength)
print(kappa)

seq = 'MGDYGFGVLVQSNTGNKSAFPVRFHPHLQPPHHHQNATPSPAAFINNNTAANGSSAGSAWLFPAPATHNIQDEILGSEKAKSQQQEQQDPLEKQQLSPSPGQEAGILPETEKAKSEENQGDNSSENGNGKEKIRIESPVLTGFDYQEATGLGTSTQPLTSSASSLTGFSNWSAAIAPSSSTIINEDASFFHQGGVPAASANNGALLFQNFPHHVSPGFGGSFSPQIGPLSQHHPHHPHFQHHHSQHQQQRRSPASPHPPPFTHRNAAFNQLPHLANNLNKPPSPWSSYQSPSPTPSSSWSPGGGGYGGWGGSQGRDHRRGLNGGITPLNSISPLKKNFASNHIQLQKYARPSSAFAPKSWMEDSLNRADNIFPFPDRPRTFDMHSLESSLIDIMRAENDTIKARTYGRRRGQSSLFPMEDGFLDDGRGDQPLHSGLGSPHCFSHQNGE'
generate_lammps_input(lambdas, sigmas)
generate_lammps_topo(seq, nchains=1)
