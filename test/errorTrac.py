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

    # plot polygons using geopandas
    p = gpd.GeoSeries(polygon)
    p.plot()
    plt.show()

    # update prediction and ground truth by discarding areas outside of cell borders
    trac[:, :, 1] = trac[:, :, 1] * interior
    trac[:, :, 2] = trac[:, :, 2] * interior
    tracGT[:, :, 1] = tracGT[:, :, 1] * interior
    tracGT[:, :, 2] = tracGT[:, :, 2] * interior

    # compute rmse
    mse = np.sum(np.pow((trac[:, :, 1] - tracGT[:, :, 1], 2)), np.pow((trac[:, :, 2] - tracGT[:, :, 2], 2)))
    rmse = np.sqrt(mse)
    msm = np.sum(np.pow(tracGT[:, :, 1], 2) + np.pow(tracGT[:, :, 2], 2))
    rmsm = np.sqrt(msm)
    error = rmse / rmsm

    return error
