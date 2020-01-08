import numpy as np

file = open("./alphas.txt", "w+")
alphas = np.array([1.00442179, 1.00364647, 1.0013874,  0.99985978, 0.98083998, 0.99778672, 0.97621343, 0.99728118])
scales = [0.8/1.05, 0.85/1.05, 0.9/1.05, 0.95/1.05, 1.10/1.05, 1.15/1.05, 1.20/1.05]
for scale in scales:
    text = alphas * scale
    for t in text:
        file.write(str(t) + " ")
    file.write("\n")
file.close()
print(scales)
