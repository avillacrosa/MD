bond_length = 3.8

# ARG, CYS, TRP masses from http://www.bioinfor.com/amino-acid/
residues = {
    "A": {"id": 1, "name": 'ALA', "mass": 71.0800018, "q": 0.00},
    "R": {"id": 2, "name": 'ARG', "mass": 156.101110, "q": 1.00},
    "N": {"id": 3, "name": 'ASN', "mass": 114.099998, "q": 0.00},
    "D": {"id": 4, "name": 'ASP', "mass": 115.099998, "q": -1.00},
    "C": {"id": 5, "name": 'CYS', "mass": 103.009190, "q": 0.00},
    "E": {"id": 6, "name": 'GLU', "mass": 129.100006, "q": -1.00},
    "Q": {"id": 7, "name": 'GLN', "mass": 128.100006, "q": 0.00},
    "G": {"id": 8, "name": 'GLY', "mass": 57.0499992, "q": 0.00},
    "H": {"id": 9, "name": 'HIS', "mass": 137.100006, "q": 0.50},
    "I": {"id": 10, "name": 'ILE', "mass": 113.199997, "q": 0.00},
    "L": {"id": 11, "name": 'LEU', "mass": 113.199997, "q": 0.00},
    "K": {"id": 12, "name": 'LYS', "mass": 128.199997, "q": 1.00},
    "M": {"id": 13, "name": 'MET', "mass": 131.199997, "q": 0.00},
    "F": {"id": 14, "name": 'PHE', "mass": 147.199997, "q": 0.00},
    "P": {"id": 15, "name": 'PRO', "mass": 97.1200027, "q": 0.00},
    "S": {"id": 16, "name": 'SER', "mass": 87.0800018, "q": 0.00},
    "T": {"id": 17, "name": 'THR', "mass": 101.099998, "q": 0.00},
    "W": {"id": 18, "name": 'TRP', "mass": 186.079310, "q": 0.00},
    "Y": {"id": 19, "name": 'TYR', "mass": 163.199997, "q": 0.00},
    "V": {"id": 20, "name": 'VAL', "mass": 99.0699997, "q": 0.00},
}

sigmas = {}
with open('../data/hps/sigmas.dat') as filein:
    for line in filein:
        line = line.split()
        sigmas[line[0]] = float(line[1])

lambdas = {}
with open('../data/hps/lambdas.dat') as filein:
    for line in filein:
        line = line.split()
        lambdas[line[0]] = float(line[1])

# Contact Thresholds in Angstroms
vdw_th = 6.
hh_th = 3.
spi_th = 4.9
catpi_th = 7.
salt_th = 6.