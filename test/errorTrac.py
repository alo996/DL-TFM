from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import shapely

file = loadmat('/home/alexrichard/DL-TFM/data/MLData250-15_dspl.mat')
brdx = file['brdx']
brdy = file['brdy']
zipped = np.array(list(zip(brdx[0], brdy[0])))
polygon = Polygon(zipped)
fig,ax = plt.subplots()

X, Y = np.meshgrid(np.arange(1, file['dspl'].shape[0]), np.arange(1, file['dspl'].shape[1]))





ax.add_patch(polygon)
ax.set_xlim([20,100])
ax.set_ylim([20,100])
plt.show()


#def errorT(trac, tracGT, brdx, brdy, cutoff=0):
 #   polygon = Polygon(/home/alexrichard/DL-TFM/data)
