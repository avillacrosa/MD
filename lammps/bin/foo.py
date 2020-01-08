import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (x+y)*np.exp(-5.0*(x**2+y**2))

x,y = np.mgrid[-1:1:100j, -1:1:100j]


z = f(x,y)
fig, ax = plt.subplots(1,1)

print(z)
img = ax.imshow(z,extent=[-1,1,-1,1])

x_label_list = ['A2', 'B2', 'C2', 'D2']

ax.set_xticks([-0.75,-0.25,0.25,0.75])

ax.set_xticklabels(x_label_list)

fig.colorbar(img)

plt.show()