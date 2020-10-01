import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

C_W='darkblue'
C_7='royalblue'

C_4 = 'red'
C_12 = 'orange'

L_W = 'WT'
L_7 = '7D'

L_4 = '$\Delta 4$'
L_12 = '12D'


viridis = plt.get_cmap('viridis', 256)
ds = viridis(np.linspace(0, 1, 256))

ds[:]=0
ds[:,3] = 1
ds[:64,0]=np.linspace(0.1,0.3,64)
ds[:64,1]=np.linspace(0.5,0.8,64)
ds[:64,2]=np.linspace(0.1,0.3,64)

ds[64:128,0]=np.linspace(0.3,1,64)
ds[64:128,1]=np.linspace(0.8,1,64)
ds[64:128,2]=np.linspace(0.3,1,64)

ds[128:192,0]=np.linspace(1,0.5,64)
ds[128:192,1]=np.linspace(1,0.3,64)
ds[128:192,2]=np.linspace(1,1,64)

ds[192:256,0]=np.linspace(0.5,0.5,64)
ds[192:256,1]=np.linspace(0.3,0.3,64)
ds[192:256,2]=np.linspace(1,0.8,64)

# ds[128:192,0]=1
# ds[192:256,2]=np.linspace(1,0.3,64)
# ds[128:192,0]=np.linspace(1,0,64)
# ds[128:192,1]=np.linspace(1,0,64)

ds = np.flip(ds,axis=0)
newcmp = ListedColormap(ds)