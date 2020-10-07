import numpy as np
import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(8, 6))

alphasPath = [
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_0.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_1.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_2.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_3.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_4.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_5.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_6.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_7.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_8.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_9.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_10.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_11.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_12.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_13.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_14.txt",
    "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_15.txt"
]
# alphasPath = [
#     "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_0.txt",
#     "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_1.txt",
# ]

# temps = np.linspace(273, 353, 16)
temps = [353.000000,
         347.003537,
         341.108937,
         335.314469,
         329.618433,
         324.019156,
         318.514994,
         313.104333,
         307.785584,
         302.557185,
         297.417601,
         292.365325,
         287.398872,
         282.516785,
         277.717630,
         273.000000, ]
labels = ["exvol", "locexvol", "bias", "torsionterm", "hbmm", "hbms", "hydrophobicity", "chargedscinteraction"]

dAlphadRG = []

for alphaPath in alphasPath:
    alphas = np.genfromtxt(alphaPath)
    alphas[:, 0] = alphas[:, 0] - np.mean(alphas[:, 0])
    min_rg_idx = np.where(alphas[:, 0] == np.min(np.abs(alphas[:, 0])))
    if len(alphas[min_rg_idx]) == 0:
        min_rg_idx = np.where(alphas[:, 0] == -np.min(np.abs(alphas[:, 0])))
    min_rg_idx = min_rg_idx[0][0]
    dRg = alphas[min_rg_idx + 1][0] - alphas[min_rg_idx][0]
    dAlpha = alphas[min_rg_idx + 1, :] - alphas[min_rg_idx, :]
    # dAlphadRG.append(dAlpha / dRg)
    dAlphadRG.append(dRg / dAlpha)

#   TODO OK UP TO HERE

dAlphadRG = np.array(dAlphadRG)
dAlphadRG = dAlphadRG.transpose()[1:9, :]

for i, a in enumerate(dAlphadRG):
    plt.plot(temps, a, label=labels[i])
plt.xlabel("T")
plt.ylabel("\u0394Rg/\u0394\u03B1")
plt.title("Discrete derivatives for all \u03B1 Energy scalings against T")
plt.legend()


singleAlpha = "/home/adria/scripts/profasi/default_output/alphas_integrase-ogT_8.txt"
alphaT = np.genfromtxt(singleAlpha)
# alphaT = alphaT[0:-5, :]
da = np.roll(alphaT, -1, axis=0) - alphaT
# print("OG", alphaT, "\n ROLL", np.roll(alphaT, -1, axis=0))
# print("DA", da)
dAlpha_dRg = []
for a in da.transpose():
    dAlpha_dRg.append(np.divide(a, da[:, 0]))
# print(dAlpha_dRg)
dAlpha_dRg = np.array(dAlpha_dRg)
# dAlpha_dRg = np.divide(da[:, 1:9], da[:, 0])
plt.figure(figsize=(8, 6))

print(dAlpha_dRg.shape)
dAlpha_dRg = dAlpha_dRg[1:9, 5:-15]
for i,a in enumerate(dAlpha_dRg):
    print(i)
    plt.plot(alphaT[5:-15, 0], a, label=labels[i])
# for i, a in enumerate(dAlphadRG):
#     plt.plot(temps, a, label=labels[i])
# plt.plot(alphaT[5:-15, 0], dAlpha_dRg.transpose()[5:-15, 1:9])
plt.xlabel("Rg")
plt.ylabel("\u0394\u03B1/\u0394Rg")
plt.legend()
plt.title("Discrete derivatives for all \u03B1 Energy scalings against Rg")
plt.show()
