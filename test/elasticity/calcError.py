import numpy as np
from scipy.io import loadmat
import shapely as sh
from shapely.geometry import Point
import matplotlib.pyplot as plt

def calcError(elast_cell):
    ymoduli = [2500, 5000, 10000, 20000, 400000]
