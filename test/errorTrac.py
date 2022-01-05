import numpy as np
from scipy.io import loadmat
import shapely as sh
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd


def errorTrac(filepath, filepath_GT):
    file = loadmat(filepath)  # load prediction
    file_GT = loadmat(filepath_GT) # load ground truth
    brdx = file['brdx']  # x-values of predicted cell border
    brdy = file['brdy']  # y-values of predicted cell border
    trac = file['trac']
    tracGT = file_GT['trac']
    zipped = np.array(list(zip(brdx[0], brdy[0])))  # array with (x,y) pairs of cell border coordinates
    polygon = sh.geometry.Polygon(zipped)  # create polygon

    interior = np.zeros((file['dspl'].shape[0], file['dspl'].shape[1]), dtype=int)  # create all zero matrix of desired shape
    for i in range(len(interior)):  # set all elements in interior matrix to 1 that actually lie within the cell
        for j in range(len(interior[i])):
            point = Point(i, j)
            if polygon.contains(point):
                interior[i][j] = 1

    # update prediction and ground truth by discarding areas outside of cell borders
    trac[:, :, 1] = trac[:, :, 1] * interior
    trac[:, :, 2] = trac[:, :, 2] * interior
    tracGT[:, :, 1] = tracGT[:, :, 1] * interior
    tracGT[:, :, 2] = tracGT[:, :, 2] * interior

    # TODO: compute mse and rsme

    # plot polygons using geopandas
    p = gpd.GeoSeries(polygon)
    p.plot()
    plt.show()




'''
np.set_printoptions(threshold=sys.maxsize)
file = loadmat('/home/alexrichard/DL-TFM/data/MLData250-15_dspl.mat')
brdx = file['brdx']
brdy = file['brdy']
zipped = np.array(list(zip(brdx[0], brdy[0]))).round()

# PIL approach
# width = 100
# height = 100
# img = Image.new('L', (width, height), 0)
# ImageDraw.Draw(img).polygon(zipped, outline=1, fill=1)
# mask = numpy.array(img)
# print(mask)

X, Y = np.meshgrid(np.arange(0, file['dspl'].shape[0]), np.arange(0, file['dspl'].shape[1]))
x, y = X.flatten(), Y.flatten()
points = np.vstack((x, y)).T

print(points)

X = np.zeros((104, 104))
for i in range(len(X)):
    for j in range(len(X[i])):
        point = Point(i, j)
        if polygon_1.contains(point):
            X[i][j] = 1

print(X)

grid = np.reshape(polygon_1.contains_points(points), (104, 104)) * 1
file = loadmat('/home/alexrichard/DL-TFM/DL-TFM-main/main/interior.mat')
matlab_grid = np.array(file['interior'])

equality = np.equal(X, matlab_grid) * 1
unique, counts = np.unique(equality, return_counts=True)
print(dict(zip(unique, counts)))


# def errorT(trac, tracGT, brdx, brdy, cutoff=0):
#   polygon = Polygon(/home/alexrichard/DL-TFM/data)
'''
