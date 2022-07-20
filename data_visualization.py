import geopandas as gpd
import matplotlib as plt
import numpy as np

from shapely.geometry import Point, Polygon


def plotCell(dspl_dict):
    width = dspl_dict['dspl'].shape[0]
    height = width
    zipped = np.array(list(zip(dspl_dict['brdx'][0], dspl_dict['brdy'][0])))  # array with (x,y) pairs of cell border coordinates
    polygon = Polygon(zipped)  # create polygon

    interior = np.zeros(shape=(width, height), dtype=int)  # create all zero matrix
    for i in range(len(interior)):  # set all elements in interior matrix to 1 that actually lie within the cell
        for j in range(len(interior[i])):
            point = Point(i, j)
            if polygon.contains(point):
                interior[i][j] = 1

    p = gpd.GeoSeries(polygon)
    p.plot()

